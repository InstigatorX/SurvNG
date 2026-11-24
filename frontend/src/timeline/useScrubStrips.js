import { useCallback, useEffect, useRef } from "react";
import { createScrubStripStore, loadScrubStrip, scrubStripTile, scrubStripTileWidth } from "../scrubStrips.mjs";
import { recordingPreviewStripUrl } from "../shared/mediaUrls.js";

// Frames for a timeline being scrubbed, from preview strips. Returns one
// stable function: call it with the time under the playhead as it moves, and
// with null when the scrub ends ({ committed: true } if it ended in a seek).
// `onFrame` receives the frame to show, or null with the same options.
// After a committed scrub the released time keeps loading for a while and its
// frame is still delivered, marked `held`: a rail commits the moment its
// scroll settles, usually before that frame has arrived, and the player's own
// seek takes longer still.
const HELD_SECONDS = 4;

export function useScrubStrips({ cameraId, source, onFrame }) {
  const storeRef = useRef(null);
  const epochRef = useRef(null);
  const onFrameRef = useRef(onFrame);
  onFrameRef.current = onFrame;
  const heldRef = useRef(false);
  const heldTimerRef = useRef(null);

  const publish = useCallback(() => {
    const store = storeRef.current;
    const epoch = epochRef.current;
    if (!store || !Number.isFinite(epoch)) return;
    const found = store.frameAt(epoch);
    const tile = found.strip
      ? scrubStripTile(found.strip, epoch, found.strip.image.naturalWidth, found.strip.image.naturalHeight)
      : null;
    onFrameRef.current?.({
      epoch, strip: found.strip, tile, gap: found.gap, loading: found.loading, held: heldRef.current,
    });
    // The released frame has arrived: nothing more to load for it.
    if (heldRef.current && !found.loading) release();
    // release reads refs only.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function release() {
    if (heldTimerRef.current) window.clearTimeout(heldTimerRef.current);
    heldTimerRef.current = null;
    heldRef.current = false;
    epochRef.current = null;
    storeRef.current?.idle();
  }

  useEffect(() => {
    if (!cameraId) return undefined;
    const width = scrubStripTileWidth({
      coarsePointer: Boolean(window.matchMedia?.("(pointer: coarse)").matches),
      playerWidth: window.innerWidth,
      pixelRatio: window.devicePixelRatio,
    });
    const store = createScrubStripStore({
      load: (epoch, signal) => loadScrubStrip(recordingPreviewStripUrl(cameraId, epoch, source, width), signal),
      onChange: publish,
      release: (strip) => { if (strip.url) URL.revokeObjectURL(strip.url); },
    });
    storeRef.current = store;
    return () => {
      release();
      store.close();
      storeRef.current = null;
      // Whatever is up belongs to the camera or stream being left, also
      // when the scrub itself ended and only its last frame is still shown.
      onFrameRef.current?.(null, { committed: false });
    };
  }, [cameraId, publish, source]);

  return useCallback((epoch, { committed = false } = {}) => {
    const store = storeRef.current;
    if (Number.isFinite(epoch)) {
      if (!store) return;
      if (heldTimerRef.current) window.clearTimeout(heldTimerRef.current);
      heldTimerRef.current = null;
      heldRef.current = false;
      epochRef.current = epoch;
      store.want(epoch);
      publish();
      return;
    }
    if (!Number.isFinite(epochRef.current) || heldRef.current) return;
    const released = epochRef.current;
    if (!committed || !store || !store.frameAt(released).loading) {
      release();
      onFrameRef.current?.(null, { committed });
      return;
    }
    heldRef.current = true;
    heldTimerRef.current = window.setTimeout(release, HELD_SECONDS * 1000);
    onFrameRef.current?.(null, { committed });
    // release reads refs only.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [publish]);
}
