import { useCallback, useEffect, useMemo, useRef } from "react";
import {
  FRAME_STEP_SETTLE_MS,
  acceptRepeatedStep,
  createFrameStepper,
  createHoldRepeater,
  frameStepKey,
  frameStepWidth,
} from "../frameStepping.mjs";
import { fetch } from "../shared/api.js";
import { recordingFrameUrl, recordingFramesUrl } from "../shared/mediaUrls.js";

// A frame that takes longer than this is given up and asked for again.
const FRAME_REQUEST_TIMEOUT_MS = 20_000;

async function fetchWithTimeout(url, signal) {
  // Withdrawn before it started: never sent.
  if (signal.aborted) throw Object.assign(new Error("Frame request withdrawn"), { name: "AbortError" });
  const controller = new AbortController();
  let timedOut = false;
  const timer = window.setTimeout(() => { timedOut = true; controller.abort(); }, FRAME_REQUEST_TIMEOUT_MS);
  const abort = () => controller.abort();
  signal.addEventListener("abort", abort);
  try {
    return await fetch(url, { signal: controller.signal });
  } catch (error) {
    // A timeout is a failure to retry, not a request the page withdrew.
    if (timedOut) throw new Error("Frame request timed out");
    throw error;
  } finally {
    window.clearTimeout(timer);
    signal.removeEventListener("abort", abort);
  }
}

// Frame stepping for the Timeline player. Returns one stable object:
//   step(direction)     one frame back (-1) or forward (+1)
//   press / release     a step control held down (repeats)
//   leave()             stop stepping and take the frame down
//   resume()            about to play from the stepped frame: its epoch, or null
//   epoch()             the stepped frame's time, or null
//   subscribe / getSnapshot   for useSyncExternalStore (FrameStepOverlay)
// The page does not render per step: only the overlay and the time readout
// subscribe. `currentEpoch()` says where the player is when stepping starts;
// `onSettle(epoch)` is called once stepping pauses, to seek the real player
// there (paused), so that play resumes from the stepped frame. `enabled`
// says whether stepping is allowed now (paused, one camera, media loaded);
// `measure()` the player's width in CSS pixels.
export function useFrameStepper({ cameraId, source, enabled, currentEpoch, onSettle, measure }) {
  const listenersRef = useRef(new Set());
  const snapshotRef = useRef({ active: false });
  const stepperRef = useRef(null);
  const settleTimerRef = useRef(null);
  const enabledRef = useRef(enabled);
  const currentEpochRef = useRef(currentEpoch);
  const onSettleRef = useRef(onSettle);
  const measureRef = useRef(measure);
  enabledRef.current = enabled;
  currentEpochRef.current = currentEpoch;
  onSettleRef.current = onSettle;
  measureRef.current = measure;

  const notify = useCallback((snapshot) => {
    snapshotRef.current = snapshot;
    for (const listener of listenersRef.current) listener();
  }, []);

  const clearSettle = useCallback(() => {
    if (settleTimerRef.current) window.clearTimeout(settleTimerRef.current);
    settleTimerRef.current = null;
  }, []);

  useEffect(() => {
    if (!cameraId) return undefined;
    const width = () => frameStepWidth({
      playerWidth: measureRef.current?.() || window.innerWidth,
      pixelRatio: window.devicePixelRatio,
    });
    const stepper = createFrameStepper({
      loadFrames: async (epoch, signal) => {
        const response = await fetchWithTimeout(recordingFramesUrl(cameraId, epoch, source), signal);
        if (response.status === 404) return null;
        if (!response.ok) throw new Error(`Frame list failed (${response.status})`);
        return response.json();
      },
      loadImage: async (epoch, signal) => {
        const response = await fetchWithTimeout(recordingFrameUrl(cameraId, epoch, source, width()), signal);
        if (!response.ok) throw new Error(`Frame failed (${response.status})`);
        const actual = Number(response.headers.get("X-SurvNG-Actual-Timestamp"));
        const url = URL.createObjectURL(await response.blob());
        // Decoded before it is shown, so a step never flashes an empty frame.
        const image = new Image();
        image.src = url;
        try {
          await image.decode();
        } catch (error) {
          URL.revokeObjectURL(url);
          throw error;
        }
        return { image: url, actual: response.headers.get("X-SurvNG-Actual-Timestamp") === null ? null : actual };
      },
      release: (entry) => URL.revokeObjectURL(entry.image),
      onChange: notify,
    });
    stepperRef.current = stepper;
    return () => {
      // A new camera or stream: the stepped frame belongs to the old one.
      clearSettle();
      stepper.close();
      if (stepperRef.current === stepper) stepperRef.current = null;
      notify({ active: false });
    };
  }, [cameraId, clearSettle, notify, source]);

  const step = useCallback((direction) => {
    const stepper = stepperRef.current;
    if (!stepper || !enabledRef.current) return false;
    stepper.step(direction, snapshotRef.current.active ? null : currentEpochRef.current?.());
    clearSettle();
    settleTimerRef.current = window.setTimeout(() => {
      settleTimerRef.current = null;
      const state = snapshotRef.current;
      if (state.active && state.known && Number.isFinite(state.epoch)) onSettleRef.current?.(state.epoch);
    }, FRAME_STEP_SETTLE_MS);
    return true;
  }, [clearSettle]);

  const holdRef = useRef(null);
  if (!holdRef.current) {
    holdRef.current = createHoldRepeater({
      onStep: (direction) => {
        if (!step(direction)) holdRef.current.release();
      },
      schedule: (callback, ms) => window.setTimeout(callback, ms),
      cancel: (timer) => window.clearTimeout(timer),
    });
  }

  // Play from the stepped frame: no seek-when-paused may follow it (that
  // would pause the playback just asked for). The frame stays up until
  // the page sees playback start and calls leave().
  const resume = useCallback(() => {
    holdRef.current.release();
    clearSettle();
    return snapshotRef.current.active && snapshotRef.current.known ? snapshotRef.current.epoch : null;
  }, [clearSettle]);

  const leave = useCallback(() => {
    holdRef.current.release();
    clearSettle();
    stepperRef.current?.leave();
  }, [clearSettle]);

  useEffect(() => () => holdRef.current.release(), []);

  // "," and "." step while the player is paused, as in most players.
  useEffect(() => {
    if (!cameraId) return undefined;
    let lastAt = null;
    function onKey(event) {
      const direction = frameStepKey(event);
      if (!direction || !enabledRef.current) return;
      event.preventDefault();
      const at = performance.now();
      if (event.repeat && !acceptRepeatedStep(lastAt, at)) return;
      lastAt = at;
      step(direction);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [cameraId, step]);

  return useMemo(() => ({
    step,
    leave,
    resume,
    press: (direction) => holdRef.current.press(direction),
    release: () => holdRef.current.release(),
    epoch: () => (snapshotRef.current.active && snapshotRef.current.known ? snapshotRef.current.epoch : null),
    subscribe: (listener) => {
      listenersRef.current.add(listener);
      return () => listenersRef.current.delete(listener);
    },
    getSnapshot: () => snapshotRef.current,
  }), [leave, resume, step]);
}
