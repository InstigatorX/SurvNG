// Frame-by-frame stepping of the paused Timeline player.
//
// The frames are decoded on the server (frame.jpg), not by seeking the
// <video>: Safari's native HLS seeks to within about a third of a second,
// can drop a seek or land on a keyframe, so a one-frame seek is not a
// one-frame step on an iPhone. The server's frame is the exact one on every
// browser. "One frame" is the next frame the file holds: the server lists
// every frame time of a segment (frames.json), and stepping walks that list,
// into the neighbouring segment where the recording continues.
//
// No queue anywhere: a step moves the wanted frame at once; one image
// request runs at a time, and when it finishes the next is for the frame
// wanted then. Holding a step control outruns the decoder rather than
// replaying a backlog.

export const FRAME_STEP_HOLD_DELAY_MS = 400;
// About 11 frames a second while held: fast enough to cover a second of
// video in two, slow enough to see each frame go by.
export const FRAME_STEP_REPEAT_MS = 90;
// After the last step, the real player is seeked to the stepped frame.
export const FRAME_STEP_SETTLE_MS = 600;
export const FRAME_STEP_IMAGE_LIMIT = 24;
export const FRAME_STEP_SEGMENT_LIMIT = 8;
// A failed frame is asked for again after these waits, then given up.
export const FRAME_STEP_RETRY_MS = [500, 1000, 2000];
// The newest segment has no "next" until the one after it is indexed.
export const FRAME_STEP_EDGE_RECHECK_MS = 5000;
// Frame times come back printed to the microsecond.
const SAME_FRAME_SECONDS = 0.0005;

// Image widths asked for, so that frames are shared between similar screens.
const FRAME_WIDTHS = [640, 960, 1280, 1920];

/** The frame width for a player this many CSS pixels wide on this screen. */
export function frameStepWidth({ playerWidth, pixelRatio = 1 }) {
  const wanted = Math.max(1, Number(playerWidth) || 1) * Math.max(1, Number(pixelRatio) || 1);
  return FRAME_WIDTHS.find((width) => width >= wanted * 0.9) ?? FRAME_WIDTHS.at(-1);
}

/** The index of the frame on screen at `epoch`: the last at or before it, else the first. */
export function frameIndexAt(frames, epoch) {
  let low = 0;
  let high = frames.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (frames[middle] <= epoch + SAME_FRAME_SECONDS) low = middle + 1;
    else high = middle;
  }
  return Math.max(0, low - 1);
}

/** "2:05:07.350 PM": the time of a frame, to the millisecond. */
export function formatFrameTime(epoch, timeZone) {
  if (!Number.isFinite(epoch)) return "--";
  const date = new Date(Math.round(epoch * 1000));
  const parts = Object.fromEntries(new Intl.DateTimeFormat("en-US", {
    timeZone, hour: "numeric", minute: "2-digit", second: "2-digit", hourCycle: "h12",
  }).formatToParts(date).map((part) => [part.type, part.value]));
  const millis = String(date.getUTCMilliseconds()).padStart(3, "0");
  return `${parts.hour}:${parts.minute}:${parts.second}.${millis}${parts.dayPeriod ? ` ${parts.dayPeriod}` : ""}`;
}

/** A frames.json answer, checked; null when it cannot be stepped through. */
export function frameListing(payload) {
  const start = Number(payload?.start);
  const end = Number(payload?.end);
  const frames = Array.isArray(payload?.frames) ? payload.frames.map(Number) : [];
  if (!Number.isFinite(start) || !(end > start) || !frames.length || frames.some((time) => !Number.isFinite(time))) return null;
  const neighbour = (value) => (value === null || value === undefined || !Number.isFinite(Number(value)) ? null : Number(value));
  return { start, end, frames, previous: neighbour(payload.previous), next: neighbour(payload.next), source: payload.source || null };
}

// Steps through frames. `loadFrames(epoch, signal)` resolves to a frames.json
// payload, or null when nothing is recorded there; `loadImage(epoch, signal)`
// to `{ image, actual }` (`image` is whatever the page draws, `release`
// frees it). `onChange` hears of every change; `snapshot()` describes it.
export function createFrameStepper({
  loadFrames,
  loadImage,
  release = () => {},
  onChange = () => {},
  now = () => Date.now(),
  schedule = (callback, ms) => setTimeout(callback, ms),
  cancel = (timer) => clearTimeout(timer),
  imageLimit = FRAME_STEP_IMAGE_LIMIT,
  segmentLimit = FRAME_STEP_SEGMENT_LIMIT,
} = {}) {
  let segments = []; // listings, most recently used last
  let active = false;
  let anchor = null; // where stepping began, until its listing is known
  let position = null; // { start, index }
  let pending = 0; // a step waiting for a listing
  let notice = null; // "end" | "start" | "error" | null
  let lastDirection = 1;
  let listingRequest = null; // { epoch, controller }
  const images = new Map(); // key -> { key, epoch, image, actual }
  let shown = null;
  let imageRequest = null; // { key, controller, prefetch }
  let failures = new Map();
  let retryTimer = null;
  let closed = false;
  let snapshotValue = { active: false };

  const keyOf = (epoch) => epoch.toFixed(6);
  const segmentAt = (epoch) => segments.find((item) => item.start <= epoch && epoch < item.end) || null;
  const segmentByStart = (start) => segments.find((item) => item.start === start) || null;

  function wantedEpoch() {
    const segment = position && segmentByStart(position.start);
    return segment ? segment.frames[position.index] : null;
  }

  function publish() {
    const epoch = wantedEpoch();
    snapshotValue = active ? {
      active: true,
      epoch: epoch ?? anchor,
      known: epoch !== null,
      image: shown?.image ?? null,
      shownEpoch: shown?.actual ?? shown?.epoch ?? null,
      loading: epoch === null || !shown || shown.key !== keyOf(epoch),
      notice,
    } : { active: false };
    onChange(snapshotValue);
  }

  function touchSegment(segment) {
    segments = [...segments.filter((item) => item !== segment), segment];
  }

  function addSegment(listing) {
    segments = segments.filter((item) => item.start !== listing.start);
    segments.push({ ...listing, checkedAt: now() });
    // Never the one being shown.
    while (segments.length > segmentLimit) {
      const evict = segments.findIndex((item) => item.start !== position?.start);
      if (evict < 0) break;
      segments.splice(evict, 1);
    }
  }

  function requestListing(epoch, { prefetch = false } = {}) {
    if (listingRequest) {
      if (Math.abs(listingRequest.epoch - epoch) < SAME_FRAME_SECONDS) return;
      // A neighbour fetched ahead gives way to one that is needed now.
      if (!prefetch && listingRequest.prefetch) listingRequest.controller.abort();
      else if (prefetch) return;
      else listingRequest.controller.abort();
    }
    const controller = new AbortController();
    const request = { epoch, controller, prefetch };
    listingRequest = request;
    Promise.resolve()
      .then(() => loadFrames(epoch, controller.signal))
      .then((payload) => {
        if (listingRequest !== request || closed) return;
        listingRequest = null;
        const listing = payload ? frameListing(payload) : null;
        if (listing) addSegment(listing);
        if (!request.prefetch) arrived(epoch, listing);
        pump();
      }, (error) => {
        if (listingRequest !== request || closed) return;
        listingRequest = null;
        if (error?.name === "AbortError" || request.prefetch) return;
        // Not a gap: say so, and let the next press ask again.
        pending = 0;
        notice = "error";
        publish();
      });
  }

  // A listing that a step was waiting for.
  function arrived(epoch, listing) {
    if (!active) return;
    const direction = pending;
    pending = 0;
    if (!position) {
      const segment = listing && segmentAt(anchor);
      if (!segment) {
        // The time stepping began at is not recorded (any more).
        notice = direction < 0 ? "start" : "end";
        publish();
        return;
      }
      position = { start: segment.start, index: frameIndexAt(segment.frames, anchor) };
      if (direction) move(direction);
      else wantImage();
      return;
    }
    if (!direction) return;
    if (!listing) {
      // The neighbour named a moment ago is gone (retention, a race).
      notice = direction > 0 ? "end" : "start";
      publish();
      return;
    }
    move(direction);
  }

  function move(direction) {
    const segment = segmentByStart(position.start);
    if (!segment) return;
    touchSegment(segment);
    const current = segment.frames[position.index];
    const target = position.index + direction;
    if (target >= 0 && target < segment.frames.length) {
      position = { start: segment.start, index: target };
      notice = null;
      wantImage();
      return;
    }
    const neighbourEpoch = direction > 0 ? segment.next : segment.previous;
    if (neighbourEpoch === null) {
      if (direction > 0 && now() - segment.checkedAt > FRAME_STEP_EDGE_RECHECK_MS) {
        // The newest segment: the next may have been indexed since.
        pending = direction;
        requestListing(current);
        publish();
        return;
      }
      notice = direction > 0 ? "end" : "start";
      publish();
      return;
    }
    const neighbour = segmentAt(neighbourEpoch);
    if (!neighbour) {
      pending = direction;
      requestListing(neighbourEpoch);
      publish();
      return;
    }
    touchSegment(neighbour);
    // The first frame after this one, or the last before it: segments
    // can overlap by a few frames where the recorder cut them.
    let index;
    if (direction > 0) {
      index = neighbour.frames.findIndex((time) => time > current + SAME_FRAME_SECONDS);
      if (index < 0) index = neighbour.frames.length - 1;
    } else {
      index = -1;
      for (let item = neighbour.frames.length - 1; item >= 0; item -= 1) {
        if (neighbour.frames[item] < current - SAME_FRAME_SECONDS) { index = item; break; }
      }
      if (index < 0) index = 0;
    }
    position = { start: neighbour.start, index };
    notice = null;
    wantImage();
  }

  function remember(entry) {
    images.delete(entry.key);
    images.set(entry.key, entry);
    // The oldest go first; never the frame on screen.
    for (const [key, value] of images) {
      if (images.size <= imageLimit) break;
      if (value === shown) continue;
      images.delete(key);
      release(value);
    }
  }

  function wantImage() {
    const epoch = wantedEpoch();
    if (epoch === null) { publish(); return; }
    const cached = images.get(keyOf(epoch));
    if (cached) {
      shown = cached;
      remember(cached);
    }
    publish();
    pump();
  }

  // The next image to load: the wanted frame, else the one after it in the
  // direction of travel (and the listing of the segment ahead near its end).
  function pump() {
    if (!active || closed) return;
    const epoch = wantedEpoch();
    if (epoch === null) return;
    const key = keyOf(epoch);
    const segment = segmentByStart(position.start);
    // Near the end of a segment, list the one ahead (no decode: cheap).
    const edge = lastDirection > 0 ? segment.frames.length - 1 - position.index : position.index;
    const neighbourEpoch = lastDirection > 0 ? segment.next : segment.previous;
    if (edge < 20 && neighbourEpoch !== null && !segmentAt(neighbourEpoch) && !listingRequest) {
      requestListing(neighbourEpoch, { prefetch: true });
    }
    if (retryTimer !== null) return;
    if (imageRequest) {
      // A frame fetched ahead never holds up the one wanted now.
      if (imageRequest.prefetch && !images.has(key)) {
        if (imageRequest.key === key) imageRequest.prefetch = false;
        else {
          imageRequest.controller.abort();
          imageRequest = null;
        }
      }
      if (imageRequest) return;
    }
    if (!images.has(key)) {
      if (notice === "error" && (failures.get(key) || 0) >= FRAME_STEP_RETRY_MS.length) return;
      fetchImage(epoch, false);
      return;
    }
    const ahead = position.index + lastDirection;
    if (ahead >= 0 && ahead < segment.frames.length) {
      const next = segment.frames[ahead];
      if (!images.has(keyOf(next)) && !failures.get(keyOf(next))) fetchImage(next, true);
    }
  }

  function fetchImage(epoch, prefetch) {
    const key = keyOf(epoch);
    const controller = new AbortController();
    const request = { key, controller, prefetch };
    imageRequest = request;
    Promise.resolve()
      .then(() => loadImage(epoch, controller.signal))
      .then((loaded) => {
        if (imageRequest !== request || closed) {
          if (loaded) release({ key, epoch, ...loaded });
          return;
        }
        imageRequest = null;
        failures.delete(key);
        const entry = { key, epoch, image: loaded.image, actual: Number.isFinite(loaded.actual) ? loaded.actual : epoch };
        remember(entry);
        const wanted = wantedEpoch();
        if (wanted !== null && keyOf(wanted) === key) {
          shown = entry;
          if (notice === "error") notice = null;
        }
        if (active) publish();
        pump();
      }, (error) => {
        if (imageRequest !== request || closed) return;
        imageRequest = null;
        if (error?.name === "AbortError") { pump(); return; }
        const count = (failures.get(key) || 0) + 1;
        failures.set(key, count);
        const wanted = wantedEpoch();
        if (wanted === null || keyOf(wanted) !== key) { pump(); return; }
        if (count > FRAME_STEP_RETRY_MS.length) {
          notice = "error";
          publish();
          return;
        }
        retryTimer = schedule(() => { retryTimer = null; pump(); }, FRAME_STEP_RETRY_MS[count - 1]);
      });
  }

  function stopRequests() {
    listingRequest?.controller.abort();
    listingRequest = null;
    imageRequest?.controller.abort();
    imageRequest = null;
    if (retryTimer !== null) cancel(retryTimer);
    retryTimer = null;
  }

  return {
    /** One frame back (-1) or forward (+1); `fromEpoch` is where the player is, if not stepping yet. */
    step(direction, fromEpoch) {
      if (closed || !(direction === 1 || direction === -1)) return;
      if (!active) {
        if (!Number.isFinite(fromEpoch)) return;
        active = true;
        anchor = fromEpoch;
        position = null;
        notice = null;
        failures = new Map();
      }
      lastDirection = direction;
      if (pending) {
        // Still waiting for the segment to step into: this press is not
        // queued, but the latest direction wins.
        pending = direction;
        return;
      }
      if (!position) {
        const segment = segmentAt(anchor);
        if (!segment) {
          pending = direction;
          requestListing(anchor);
          publish();
          return;
        }
        position = { start: segment.start, index: frameIndexAt(segment.frames, anchor) };
      }
      // A frame that failed may be asked for again by stepping onto it.
      if (notice === "error") {
        notice = null;
        failures = new Map();
      }
      move(direction);
    },
    /** Stop stepping: the frame comes down, requests stop. Loaded frames are kept. */
    leave() {
      if (!active && !listingRequest && !imageRequest) return;
      const was = active;
      active = false;
      anchor = null;
      position = null;
      pending = 0;
      notice = null;
      shown = null;
      stopRequests();
      if (was) publish();
    },
    close() {
      this.leave();
      closed = true;
      for (const entry of images.values()) release(entry);
      images.clear();
      segments = [];
    },
    epoch() {
      return active ? (wantedEpoch() ?? anchor) : null;
    },
    snapshot() {
      return snapshotValue;
    },
    // For tests.
    stats() {
      return { segments: segments.length, images: images.size, listing: listingRequest?.epoch ?? null, image: imageRequest?.key ?? null, prefetch: Boolean(imageRequest?.prefetch) };
    },
  };
}

// Press-and-hold for a step control: one step at once, more after a delay,
// at a fixed rate, until released.
export function createHoldRepeater({
  onStep,
  delay = FRAME_STEP_HOLD_DELAY_MS,
  interval = FRAME_STEP_REPEAT_MS,
  schedule = (callback, ms) => setTimeout(callback, ms),
  cancel = (timer) => clearTimeout(timer),
} = {}) {
  let timer = null;
  let direction = 0;
  const tick = () => {
    timer = schedule(tick, interval);
    onStep(direction);
  };
  return {
    press(nextDirection) {
      if (timer !== null) cancel(timer);
      direction = nextDirection;
      onStep(direction);
      timer = schedule(tick, delay);
    },
    release() {
      if (timer !== null) cancel(timer);
      timer = null;
      direction = 0;
    },
    holding() {
      return timer !== null;
    },
  };
}

// Key repeat runs at the operating system's rate (often 30 a second); steps
// are taken at the hold rate instead.
export function acceptRepeatedStep(lastAt, at, interval = FRAME_STEP_REPEAT_MS) {
  return !Number.isFinite(lastAt) || at - lastAt >= interval - 5;
}

/** Whether a key press is a frame step: "," back, "." forward. */
export function frameStepKey(event) {
  if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey || event.isComposing) return 0;
  const target = event.target;
  if (target && typeof target.closest === "function"
    && target.closest("input, textarea, select, [contenteditable=''], [contenteditable=true], [role=dialog]")) return 0;
  if (event.key === ",") return -1;
  if (event.key === ".") return 1;
  return 0;
}
