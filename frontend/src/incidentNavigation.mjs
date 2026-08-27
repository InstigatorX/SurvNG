function sameEventId(left, right) {
  if (left == null || right == null) return false;
  return String(left) === String(right);
}

export function incidentSelectionHref(currentHref, eventId) {
  const numericId = Number(eventId);
  if (!Number.isInteger(numericId) || numericId <= 0) return String(currentHref || "");
  const url = new URL(String(currentHref || "http://localhost/incidents"), "http://localhost");
  url.searchParams.set("event_ids", String(numericId));
  return url.href;
}

export function incidentDetectionFrameSize(event) {
  const detected = (Array.isArray(event?.objects) ? event.objects : []).find((object) => (
    Number(object?.detection_frame_width) > 0
    && Number(object?.detection_frame_height) > 0
  ));
  if (detected) return {
    width: Number(detected.detection_frame_width),
    height: Number(detected.detection_frame_height),
  };
  return incidentTrackingFrameSize(event, false);
}

export function incidentTrackingFrameSize(event, fallbackToDetection = true) {
  const trackedWidth = Number(event?.object_tracking?.frame_width);
  const trackedHeight = Number(event?.object_tracking?.frame_height);
  if (trackedWidth > 0 && trackedHeight > 0) {
    return { width: trackedWidth, height: trackedHeight };
  }
  return fallbackToDetection ? incidentDetectionFrameSize(event) : null;
}

function incidentRecencyEpoch(incident) {
  for (const value of [
    incident?.last_epoch,
    incident?.end_at,
    incident?.start_epoch,
    incident?.created_epoch,
    incident?.created_at,
  ]) {
    const numeric = Number(value);
    if (Number.isFinite(numeric)) return numeric;
    const parsed = new Date(value || 0).getTime() / 1000;
    if (Number.isFinite(parsed) && parsed > 0) return parsed;
  }
  return 0;
}

export function incidentsNewestFirst(incidents) {
  if (!Array.isArray(incidents)) return [];
  return incidents
    .map((incident, index) => ({ incident, index, epoch: incidentRecencyEpoch(incident) }))
    .sort((left, right) => right.epoch - left.epoch || left.index - right.index)
    .map(({ incident }) => incident);
}

export function retainFocusedIncident(incidents, incidentId, retained = null) {
  if (incidentId == null) return null;
  const selected = Array.isArray(incidents)
    ? incidents.find((incident) => sameEventId(incident?.id, incidentId))
    : null;
  if (selected) return selected;
  return sameEventId(retained?.id, incidentId) ? retained : null;
}

export function createIncidentPageCache(loader) {
  let entries = new Map();
  return {
    load(key) {
      const existing = entries.get(key);
      if (existing) return existing.pending;
      const entry = { pending: null, value: undefined };
      const pending = Promise.resolve()
        .then(() => loader(key))
        .then((value) => {
          entry.value = value;
          return value;
        });
      entry.pending = pending;
      entries.set(key, entry);
      pending.catch(() => {
        if (entries.get(key) === entry) entries.delete(key);
      });
      return pending;
    },
    peek(key) {
      return entries.get(key)?.value;
    },
    retain(keys) {
      const retained = new Set(keys.filter(Boolean));
      for (const key of entries.keys()) {
        if (!retained.has(key)) entries.delete(key);
      }
    },
    invalidate(key) {
      if (key) entries.delete(key);
    },
    clear() {
      entries = new Map();
    },
    size() {
      return entries.size;
    },
  };
}

export function incidentDetailQuery(incident) {
  const eventIds = (incident?.events || [])
    .map((event) => Number(event?.id))
    .filter((eventId) => Number.isInteger(eventId) && eventId > 0);
  if (!eventIds.length) return "";
  return new URLSearchParams({
    event_ids: [...new Set(eventIds)].join(","),
    gap_seconds: "45",
  }).toString();
}

export function linkedIncidentEventFilter(incident) {
  return Number(incident?.object_event_count || 0) > 0 || incident?.has_objects || incident?.kind === "object"
    ? "object"
    : "motion";
}

export function incidentMosaicEvents(incident) {
  if (!Array.isArray(incident?.events)) return [];
  return incident.events
    .map((event, index) => {
      const parsedEpoch = new Date(event?.created_at || 0).getTime();
      return { event, index, epoch: Number.isFinite(parsedEpoch) ? parsedEpoch : Number.POSITIVE_INFINITY };
    })
    .sort((left, right) => left.epoch - right.epoch || left.index - right.index)
    .map(({ event }) => event);
}

export function incidentMosaicPage(events, page, pageSize = 6) {
  const items = Array.isArray(events) ? events : [];
  const size = Math.max(1, Math.floor(Number(pageSize) || 6));
  const pageCount = Math.max(1, Math.ceil(items.length / size));
  const pageIndex = Math.max(0, Math.min(pageCount - 1, Math.floor(Number(page) || 0)));
  return {
    items: items.slice(pageIndex * size, (pageIndex + 1) * size),
    page: pageIndex,
    pageCount,
  };
}

function eventEpoch(event) {
  for (const value of [event?.created_epoch, event?.created_at]) {
    const numeric = Number(value);
    if (Number.isFinite(numeric) && numeric > 0) return numeric;
    const parsed = new Date(value || 0).getTime() / 1000;
    if (Number.isFinite(parsed) && parsed > 0) return parsed;
  }
  return 0;
}

function largestTrackFrame(tracking) {
  let best = null;
  for (const track of Array.isArray(tracking?.tracks) ? tracking.tracks : []) {
    for (const sample of Array.isArray(track?.box_history) ? track.box_history : []) {
      const [capturedAt, x1, y1, x2, y2] = sample.map(Number);
      const area = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
      if (Number.isFinite(capturedAt) && capturedAt > 0 && (!best || area > best.area)) {
        best = { epoch: capturedAt, area, label: track.label || "object" };
      }
    }
  }
  return best;
}

export function incidentEvidenceFrames(event) {
  const epoch = eventEpoch(event);
  if (!epoch) return [];
  const objects = Array.isArray(event?.objects) ? event.objects : [];
  const primary = objects
    .filter((object) => object?.label && object?.incident_eligible !== false)
    .sort((left, right) => Number(right.temporal_peak_confidence || right.confidence || 0) - Number(left.temporal_peak_confidence || left.confidence || 0))[0];
  const hasPeakOffset = Number.isFinite(Number(primary?.temporal_peak_confidence_offset_seconds));
  const detectionOffset = Number(primary?.temporal_peak_confidence_offset_seconds ?? primary?.temporal_sample_offset_seconds);
  const selectedOffset = Number(primary?.temporal_sample_offset_seconds);
  const selectedEpoch = Number.isFinite(selectedOffset) ? epoch + selectedOffset : epoch;
  const tracking = largestTrackFrame(event?.object_tracking);
  return [
    { key: "trigger", label: "Trigger", epoch, kind: "recording" },
    {
      key: "detection",
      label: primary?.label
        ? `${hasPeakOffset ? "Best" : "Detected"} ${primary.label}`
        : "Detection",
      epoch: Number.isFinite(detectionOffset) ? epoch + detectionOffset : selectedEpoch,
      kind: "recording",
      confidence: Number(primary?.temporal_peak_confidence || primary?.confidence || 0),
    },
    { key: "selected", label: "Selected", epoch: selectedEpoch, kind: "snapshot" },
    ...(tracking ? [{ key: "tracking", label: `Best ${tracking.label} track`, epoch: tracking.epoch, kind: "recording" }] : []),
  ];
}

export function incidentIndexForEvent(incidents, event) {
  if (!Array.isArray(incidents) || !event) return -1;
  return incidents.findIndex((incident) => (
    sameEventId(incident?.id, event.id)
    || (incident?.events || []).some((child) => sameEventId(child?.id, event.id))
  ));
}

export function adjacentIncident(incidents, event, direction) {
  if (!Array.isArray(incidents) || incidents.length < 2) return null;
  const currentIndex = incidentIndexForEvent(incidents, event);
  if (currentIndex < 0) return null;
  const step = direction < 0 ? -1 : 1;
  return incidents[(currentIndex + step + incidents.length) % incidents.length] || null;
}

export function incidentArrowNavigationAllowed(target) {
  if (!target || typeof target.closest !== "function") return true;
  return !target.closest('button, a, input, select, textarea, video, audio, [contenteditable="true"], [role="slider"], [role="spinbutton"], [role="tablist"]');
}

export function showIncidentCardAnnotations(expanded, thumbnailAnnotations) {
  return !expanded && Boolean(thumbnailAnnotations);
}

export const INCIDENT_THUMBNAIL_OBJECT_FOCUS_MODES = Object.freeze(["off", "auto", "button"]);

export function normalizeIncidentThumbnailObjectFocus(mode) {
  const value = String(mode || "off").trim().toLowerCase();
  return INCIDENT_THUMBNAIL_OBJECT_FOCUS_MODES.includes(value) ? value : "off";
}

export function incidentThumbnailObjectFocusEnabled(mode) {
  const normalized = normalizeIncidentThumbnailObjectFocus(mode);
  return normalized === "auto" || normalized === "button";
}

export function normalizeIncidentThumbnailObjectFocusZoom(zoom) {
  const value = Number(zoom);
  if (!Number.isFinite(value)) return 1;
  return Math.min(5.5, Math.max(0.25, value));
}

export function incidentObjectFocusAspect(aspect, serverCrop = false) {
  const width = Number(aspect?.width);
  const height = Number(aspect?.height);
  if (Number.isFinite(width) && Number.isFinite(height) && width > 0 && height > 0) {
    return { width, height };
  }
  // The thumbnail endpoint applies this same default when object_focus is
  // enabled without aspect parameters. Keep the client crop plane identical.
  return serverCrop ? { width: 16, height: 9 } : null;
}

export function incidentImageRenderRect(frameSize, imageSize, fit = "contain") {
  const frameWidth = Number(frameSize?.width);
  const frameHeight = Number(frameSize?.height);
  const imageWidth = Number(imageSize?.width);
  const imageHeight = Number(imageSize?.height);
  if (![frameWidth, frameHeight, imageWidth, imageHeight].every(Number.isFinite)
    || frameWidth <= 0 || frameHeight <= 0 || imageWidth <= 0 || imageHeight <= 0) {
    return null;
  }
  const scale = fit === "cover"
    ? Math.max(frameWidth / imageWidth, frameHeight / imageHeight)
    : Math.min(frameWidth / imageWidth, frameHeight / imageHeight);
  const width = imageWidth * scale;
  const height = imageHeight * scale;
  return {
    x: (frameWidth - width) / 2,
    y: (frameHeight - height) / 2,
    width,
    height,
    scale,
  };
}

export function incidentObjectFocusCropRect(sourceWidth, sourceHeight, boxes, zoom = 1, aspectWidth = 0, aspectHeight = 0) {
  const width = Math.max(0, Number(sourceWidth) || 0);
  const height = Math.max(0, Number(sourceHeight) || 0);
  if (!width || !height || !Array.isArray(boxes) || !boxes.length) return null;
  const zoomFactor = normalizeIncidentThumbnailObjectFocusZoom(zoom);
  const minX = Math.min(...boxes.map((box) => Number(box.x1)));
  const minY = Math.min(...boxes.map((box) => Number(box.y1)));
  const maxX = Math.max(...boxes.map((box) => Number(box.x2)));
  const maxY = Math.max(...boxes.map((box) => Number(box.y2)));
  if (![minX, minY, maxX, maxY].every(Number.isFinite) || maxX <= minX || maxY <= minY) return null;
  const boxWidth = Math.max(1, maxX - minX);
  const boxHeight = Math.max(1, maxY - minY);
  const padX = Math.max(width * 0.06, boxWidth * 0.45) / zoomFactor;
  const padY = Math.max(height * 0.08, boxHeight * 0.65) / zoomFactor;
  let x1 = Math.max(0, Math.floor(minX - padX));
  let y1 = Math.max(0, Math.floor(minY - padY));
  let x2 = Math.min(width, Math.ceil(maxX + padX));
  let y2 = Math.min(height, Math.ceil(maxY + padY));
  if (x2 <= x1 || y2 <= y1) return null;
  const aspectW = Number(aspectWidth);
  const aspectH = Number(aspectHeight);
  if (Number.isFinite(aspectW) && Number.isFinite(aspectH) && aspectW > 0 && aspectH > 0) {
    const targetAspect = aspectW / aspectH;
    let cropWidth = x2 - x1;
    let cropHeight = y2 - y1;
    const centerX = (x1 + x2) * 0.5;
    const centerY = (y1 + y2) * 0.5;
    if (cropWidth / cropHeight > targetAspect) {
      cropHeight = cropWidth / targetAspect;
    } else {
      cropWidth = cropHeight * targetAspect;
    }
    x1 = centerX - cropWidth * 0.5;
    x2 = centerX + cropWidth * 0.5;
    y1 = centerY - cropHeight * 0.5;
    y2 = centerY + cropHeight * 0.5;
    if (x2 - x1 > width) {
      x1 = 0;
      x2 = width;
    } else if (x1 < 0) {
      x2 -= x1;
      x1 = 0;
    } else if (x2 > width) {
      x1 -= x2 - width;
      x2 = width;
      x1 = Math.max(0, x1);
    }
    if (y2 - y1 > height) {
      y1 = 0;
      y2 = height;
    } else if (y1 < 0) {
      y2 -= y1;
      y1 = 0;
    } else if (y2 > height) {
      y1 -= y2 - height;
      y2 = height;
      y1 = Math.max(0, y1);
    }
    x1 = Math.floor(x1);
    y1 = Math.floor(y1);
    x2 = Math.ceil(x2);
    y2 = Math.ceil(y2);
  }
  if (x2 <= x1 || y2 <= y1) return null;
  return { x1, y1, x2, y2, width: x2 - x1, height: y2 - y1 };
}

export function incidentObjectFocusThumbnailWidth(frameWidth, devicePixelRatio = 1, zoomFactor = 1) {
  const width = Math.max(160, Number(frameWidth) || 160);
  const ratio = Math.max(1, Math.min(4, Number(devicePixelRatio) || 1));
  const zoom = normalizeIncidentThumbnailObjectFocusZoom(zoomFactor);
  // Object focus CSS-scales the raster; request enough pixels for a sharp crop.
  const required = Math.ceil(width * ratio * Math.min(5.5, Math.max(3.5, zoom * 2.5)));
  if (required <= 1280) return 1280;
  if (required <= 1920) return 1920;
  return 2560;
}

export function incidentObjectFocusMaxScale(sourceWidth, renderedImageWidth, devicePixelRatio = 1, maxStretch = 1.35) {
  const source = Math.max(0, Number(sourceWidth) || 0);
  const rendered = Math.max(0, Number(renderedImageWidth) || 0);
  const ratio = Math.max(1, Math.min(4, Number(devicePixelRatio) || 1));
  if (!source || !rendered) return 5.5;
  return Math.min(5.5, Math.max(1, (maxStretch * source) / (rendered * ratio)));
}

export function incidentObjectFocusStyle(frameSize, renderedBoxes, zoom = 1, maxScale = 5.5) {
  if (!frameSize?.width || !frameSize?.height || !Array.isArray(renderedBoxes) || !renderedBoxes.length) {
    return null;
  }
  const minX = Math.max(0, Math.min(...renderedBoxes.map((box) => Number(box.left) || 0)));
  const minY = Math.max(0, Math.min(...renderedBoxes.map((box) => Number(box.top) || 0)));
  const maxX = Math.min(frameSize.width, Math.max(...renderedBoxes.map((box) => (Number(box.left) || 0) + (Number(box.width) || 0))));
  const maxY = Math.min(frameSize.height, Math.max(...renderedBoxes.map((box) => (Number(box.top) || 0) + (Number(box.height) || 0))));
  const boxWidth = Math.max(1, maxX - minX);
  const boxHeight = Math.max(1, maxY - minY);
  const padX = Math.max(frameSize.width * 0.04, boxWidth * 0.35);
  const padY = Math.max(frameSize.height * 0.04, boxHeight * 0.35);
  const cropX1 = Math.max(0, minX - padX);
  const cropY1 = Math.max(0, minY - padY);
  const cropX2 = Math.min(frameSize.width, maxX + padX);
  const cropY2 = Math.min(frameSize.height, maxY + padY);
  const cropWidth = Math.max(1, cropX2 - cropX1);
  const cropHeight = Math.max(1, cropY2 - cropY1);
  const centerX = cropX1 + cropWidth / 2;
  const centerY = cropY1 + cropHeight / 2;
  const fitScale = Math.min((frameSize.width * 0.82) / cropWidth, (frameSize.height * 0.82) / cropHeight);
  const zoomFactor = normalizeIncidentThumbnailObjectFocusZoom(zoom);
  const scaleCap = Math.min(5.5, Math.max(1, Number(maxScale) || 5.5));
  const scale = Math.min(scaleCap, Math.max(1, fitScale * zoomFactor));
  return {
    transform: `translate3d(${frameSize.width / 2 - centerX * scale}px, ${frameSize.height / 2 - centerY * scale}px, 0) scale(${scale})`,
    transformOrigin: "0 0",
  };
}

export function incidentProgressiveImageWidth(renderedWidth, devicePixelRatio = 1) {
  const width = Math.max(0, Number(renderedWidth) || 0);
  const ratio = Math.max(1, Math.min(4, Number(devicePixelRatio) || 1));
  if (!width) return 1280;
  const requiredPixels = width * ratio;
  if (requiredPixels <= 1280) return 1280;
  if (requiredPixels <= 1920) return 1920;
  return 2560;
}

export function incidentZoomLayout(frameSize, zoom) {
  const width = Math.max(0, Number(frameSize?.width) || 0);
  const height = Math.max(0, Number(frameSize?.height) || 0);
  const scale = Math.max(1, Number(zoom?.scale) || 1);
  if (!width || !height || scale === 1) return null;
  const scaledWidth = width * scale;
  const scaledHeight = height * scale;
  return {
    left: (width - scaledWidth) / 2 + (Number(zoom?.x) || 0),
    top: (height - scaledHeight) / 2 + (Number(zoom?.y) || 0),
    width: scaledWidth,
    height: scaledHeight,
  };
}

export function incidentTriggerLabel(incident) {
  const source = String(incident?.trigger_source || "camera").toLowerCase();
  return ["ema", "adaptive", "visual_backup", "adaptive/visual_backup"].includes(source)
    ? "EMA"
    : "Camera";
}

export function incidentObjectIconName(label) {
  const normalized = String(label || "").trim().toLowerCase().replaceAll("-", "_").replaceAll(" ", "_");
  if (["person", "human"].includes(normalized)) return "person";
  if (normalized === "face") return "face";
  if (["car", "vehicle"].includes(normalized)) return "car";
  if (normalized === "truck") return "truck";
  if (normalized === "bus") return "bus";
  if (["motorcycle", "motorbike", "bicycle", "bike"].includes(normalized)) return "bike";
  if (normalized === "cat") return "cat";
  if (normalized === "dog") return "dog";
  if (["robot_lawnmower", "robot_mower", "lawnmower", "mower"].includes(normalized)) return "mower";
  return "object";
}

export function incidentThumbnailPageSize({ width, height, density, columns: requestedColumns, gap: requestedGap, horizontalPadding: requestedPadding, rowHeight: requestedRowHeight }) {
  const safeWidth = Math.max(0, Number(width) || 0);
  const safeHeight = Math.max(0, Number(height) || 0);
  if (!safeWidth || !safeHeight) return density === "comfortable" ? 8 : 12;
  const compact = density !== "comfortable";
  const gap = Math.max(0, Number.isFinite(Number(requestedGap)) ? Number(requestedGap) : compact ? 6 : 9);
  if (Number.isFinite(Number(requestedColumns))) {
    const columns = Math.max(1, Math.floor(Number(requestedColumns)));
    const horizontalPadding = Math.max(0, Number.isFinite(Number(requestedPadding)) ? Number(requestedPadding) : 16);
    const usableWidth = Math.max(1, safeWidth - horizontalPadding);
    const cardWidth = Math.max(1, (usableWidth - gap * (columns - 1)) / columns);
    const cardHeight = cardWidth * 10 / 16 + 2;
    const rows = Math.max(1, Math.floor((safeHeight + gap) / (cardHeight + gap)));
    return rows * columns;
  }
  const rowHeight = Math.max(44, Number.isFinite(Number(requestedRowHeight)) ? Number(requestedRowHeight) : compact ? 78 : 98);
  return Math.max(1, Math.floor((safeHeight + gap) / (rowHeight + gap)));
}
