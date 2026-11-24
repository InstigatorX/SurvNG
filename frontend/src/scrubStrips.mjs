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
// One request runs at a time: the server builds one strip at a time, so a
// queue would only show frames for places the playhead has already left.
export function createScrubStripStore({
  load,
  onChange = () => {},
  release = () => {},
  limit = SCRUB_STRIP_LIMIT,
  now = () => Date.now(),
} = {}) {
  let strips = [];
  let gaps = new Map();
  let inFlight = null;
  let wanted = null;
  let direction = 0;
  let retryAt = 0;
  let closed = false;

  const gapKey = (epoch) => Math.floor(epoch / SCRUB_STRIP_GAP_BUCKET_SECONDS);
  const find = (epoch) => strips.find((strip) => strip.start <= epoch && epoch < strip.end) || null;
  const isGap = (epoch) => {
    const until = gaps.get(gapKey(epoch));
    if (until === undefined) return false;
    if (until > now()) return true;
    gaps.delete(gapKey(epoch));
    return false;
  };

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
    while (strips.length > limit) {
      const oldest = strips.reduce((a, b) => (a.usedAt <= b.usedAt ? a : b));
      strips = strips.filter((item) => item !== oldest);
      release(oldest);
    }
  }

  function next() {
    if (closed || inFlight || wanted === null || now() < retryAt) return;
    let target = null;
    const current = find(wanted);
    if (!current && !isGap(wanted)) {
      target = wanted;
    } else if (current && direction !== 0) {
      // Idle at a loaded segment: fetch the one the drag is heading into.
      const ahead = direction > 0 ? current.end + 0.5 : current.start - 0.5;
      if (!find(ahead) && !isGap(ahead)) target = ahead;
    }
    if (target === null) return;
    const controller = typeof AbortController === "undefined" ? null : new AbortController();
    const request = { epoch: target, controller };
    inFlight = request;
    Promise.resolve()
      .then(() => load(target, controller?.signal))
      .then((strip) => {
        if (closed || inFlight !== request) {
          if (strip) release(strip);
          return;
        }
        if (strip) keep(strip);
        else gaps.set(gapKey(target), now() + SCRUB_STRIP_GAP_SECONDS * 1000);
      })
      .catch(() => {
        if (!closed && inFlight === request) retryAt = now() + SCRUB_STRIP_RETRY_MS;
      })
      .finally(() => {
        if (inFlight !== request) return;
        inFlight = null;
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
      inFlight?.controller?.abort();
      inFlight = null;
      strips.forEach(release);
      strips = [];
      gaps = new Map();
    },
    get size() {
      return strips.length;
    },
  };
}
