// Scrub-preview strips: one image per recording segment holding a frame for
// each second of it. The timeline shows these in the player while the playhead
// is dragged, so moving within a loaded segment needs no request at all.

// Kept in memory at once. A strip of twelve 320 px tiles decodes to a few MB.
export const SCRUB_STRIP_LIMIT = 24;
// A time with no recording is not asked about again for this long.
export const SCRUB_STRIP_GAP_SECONDS = 30;
export const SCRUB_STRIP_GAP_BUCKET_SECONDS = 5;
// While the strip for the playhead's own time is loading, the nearest loaded
// one within this distance stands in for it. A rail scrolls through minutes a
// second; a frame from just behind the playhead says more than none.
export const SCRUB_STRIP_NEAR_SECONDS = 600;
// Requests running at once, and how many segments are fetched ahead of the
// playhead in the direction it is moving. The server builds two strips at a
// time, so the playhead's own segment and the next one load together.
export const SCRUB_STRIP_CONCURRENCY = 2;
export const SCRUB_STRIP_AHEAD = 2;
// Until a strip has shown how long this camera's segments are.
export const SCRUB_STRIP_SEGMENT_SECONDS = 10;
// After a failed request (busy generator, storage error) wait before another.
export const SCRUB_STRIP_RETRY_MS = 1000;

// Layout of a strip from the response headers of preview-strip.jpg.
export function scrubStripLayout(headers) {
  const number = (name) => Number(headers.get(name));
  const layout = {
    start: number("X-SurvNG-Strip-Start"),
    end: number("X-SurvNG-Strip-End"),
    step: number("X-SurvNG-Strip-Step"),
    count: number("X-SurvNG-Strip-Count"),
    columns: number("X-SurvNG-Strip-Columns"),
    rows: number("X-SurvNG-Strip-Rows"),
    source: headers.get("X-SurvNG-Strip-Source") || "",
  };
  const valid = Number.isFinite(layout.start) && layout.end > layout.start && layout.step > 0
    && Number.isInteger(layout.count) && layout.count > 0
    && Number.isInteger(layout.columns) && layout.columns > 0
    && Number.isInteger(layout.rows) && layout.rows > 0
    && layout.count <= layout.columns * layout.rows;
  return valid ? layout : null;
}

// The tile of `strip` to show for `epoch`, as a source rectangle in an image
// of the given size. Tile i is the frame i * step seconds into the segment.
export function scrubStripTile(strip, epoch, imageWidth, imageHeight) {
  if (!strip || !(imageWidth > 0) || !(imageHeight > 0)) return null;
  const index = Math.max(0, Math.min(strip.count - 1, Math.floor((epoch - strip.start) / strip.step)));
  const width = imageWidth / strip.columns;
  const height = imageHeight / strip.rows;
  return {
    index,
    epoch: strip.start + index * strip.step,
    sx: (index % strip.columns) * width,
    sy: Math.floor(index / strip.columns) * height,
    sw: width,
    sh: height,
  };
}

// Tile width to ask for. The time is in the decode, not the scaling, so a
// large player gets sharper tiles; a phone gets small ones it can hold many of.
export function scrubStripTileWidth({ coarsePointer = false, playerWidth = 0, pixelRatio = 1 } = {}) {
  if (coarsePointer) return 320;
  return playerWidth * Math.min(2, pixelRatio || 1) > 1000 ? 480 : 320;
}

// Fetch one strip and decode its image. Null when nothing is recorded there.
export async function loadScrubStrip(url, signal) {
  const response = await fetch(url, { signal });
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`Preview strip failed (${response.status})`);
  const layout = scrubStripLayout(response.headers);
  if (!layout) throw new Error("Preview strip has no layout");
  const objectUrl = URL.createObjectURL(await response.blob());
  const image = new Image();
  image.decoding = "async";
  image.src = objectUrl;
  try {
    await image.decode();
  } catch (error) {
    URL.revokeObjectURL(objectUrl);
    throw error;
  }
  return { ...layout, image, url: objectUrl };
}

// Holds loaded strips and decides what to load next. `load(epoch, signal)`
// resolves to a strip ({ start, end, step, count, columns, rows, image, ... }),
// or null when nothing is recorded at that time, and rejects on failure.
// There is no queue: when a request finishes, the next is chosen from where
// the playhead is then. A queue would only bring frames for places it has
// already left.
export function createScrubStripStore({
  load,
  onChange = () => {},
  release = () => {},
  limit = SCRUB_STRIP_LIMIT,
  now = () => Date.now(),
} = {}) {
  let strips = [];
  let gaps = new Map();
  let inFlight = [];
  let wanted = null;
  let direction = 0;
  let retryAt = 0;
  let closed = false;
  let segmentSeconds = SCRUB_STRIP_SEGMENT_SECONDS;

  const gapKey = (epoch) => Math.floor(epoch / SCRUB_STRIP_GAP_BUCKET_SECONDS);
  const find = (epoch) => strips.find((strip) => strip.start <= epoch && epoch < strip.end) || null;
  const isGap = (epoch) => {
    const until = gaps.get(gapKey(epoch));
    if (until === undefined) return false;
    if (until > now()) return true;
    gaps.delete(gapKey(epoch));
    return false;
  };
  // A request names a time, not a segment: one for a time this close is
  // taken to be for the same segment.
  const requested = (epoch) => inFlight.some((request) => Math.abs(request.epoch - epoch) < segmentSeconds);
  const needed = (epoch) => !find(epoch) && !isGap(epoch) && !requested(epoch);

  function keep(strip) {
    // A segment asked for twice (the main and sub streams cut at different
    // times) replaces what overlaps it.
    strips = strips.filter((item) => {
      const overlaps = item.start < strip.end && strip.start < item.end;
      if (overlaps) release(item);
      return !overlaps;
    });
    strips.push(strip);
    strip.usedAt = now();
    if (strip.end - strip.start > 0) segmentSeconds = strip.end - strip.start;
    while (strips.length > limit) {
      const oldest = strips.reduce((a, b) => (a.usedAt <= b.usedAt ? a : b));
      strips = strips.filter((item) => item !== oldest);
      release(oldest);
    }
  }

  // The most useful time not yet loaded or asked for: the playhead's own,
  // then the segments it is heading into, nearest first.
  function target() {
    if (needed(wanted)) return wanted;
    if (direction === 0) return null;
    let edge = wanted;
    for (let step = 0; step < SCRUB_STRIP_AHEAD; step += 1) {
      const strip = find(edge);
      // Past a segment that is not here yet its neighbour cannot be named.
      if (!strip) return null;
      edge = direction > 0 ? strip.end + 0.5 : strip.start - 0.5;
      if (needed(edge)) return edge;
    }
    return null;
  }

  function next() {
    while (!closed && wanted !== null && inFlight.length < SCRUB_STRIP_CONCURRENCY && now() >= retryAt) {
      const epoch = target();
      if (epoch === null) return;
      start(epoch);
    }
  }

  function start(epoch) {
    const controller = typeof AbortController === "undefined" ? null : new AbortController();
    const request = { epoch, controller };
    inFlight.push(request);
    const current = () => inFlight.includes(request);
    Promise.resolve()
      .then(() => load(epoch, controller?.signal))
      .then((strip) => {
        if (closed || !current()) {
          if (strip) release(strip);
          return;
        }
        if (strip) keep(strip);
        else gaps.set(gapKey(epoch), now() + SCRUB_STRIP_GAP_SECONDS * 1000);
      })
      .catch(() => {
        if (!closed && current()) retryAt = now() + SCRUB_STRIP_RETRY_MS;
      })
      .finally(() => {
        if (!current()) return;
        inFlight = inFlight.filter((item) => item !== request);
        if (closed) return;
        onChange();
        next();
      });
  }

  return {
    // What can be shown for `epoch` right now.
    frameAt(epoch) {
      const strip = find(epoch);
      if (strip) {
        strip.usedAt = now();
        return { strip, gap: false, loading: false };
      }
      if (isGap(epoch)) return { strip: null, gap: true, loading: false };
      let nearest = null;
      let nearestDistance = SCRUB_STRIP_NEAR_SECONDS;
      for (const item of strips) {
        const distance = epoch < item.start ? item.start - epoch : epoch - item.end;
        if (distance <= nearestDistance) {
          nearest = item;
          nearestDistance = distance;
        }
      }
      return { strip: nearest, gap: false, loading: true };
    },
    // The playhead is at `epoch`: load its segment unless it is already here.
    want(epoch) {
      if (closed || !Number.isFinite(epoch)) return;
      if (wanted !== null && epoch !== wanted) direction = epoch > wanted ? 1 : -1;
      wanted = epoch;
      next();
    },
    // A failed request is retried by the next want() once the wait has passed.
    idle() {
      wanted = null;
      direction = 0;
    },
    close() {
      closed = true;
      wanted = null;
      inFlight.forEach((request) => request.controller?.abort());
      inFlight = [];
      strips.forEach(release);
      strips = [];
      gaps = new Map();
    },
    get size() {
      return strips.length;
    },
  };
}
