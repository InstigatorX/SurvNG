import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  Activity,
  ArrowLeft,
  ArrowRight,
  Bike,
  Bot,
  BusFront,
  Camera,
  CarFront,
  Cat,
  CircleDot,
  Crop,
  Download,
  Dog,
  Gauge,
  Search,
  ListTree,
  Play,
  Radar,
  ScanFace,
  Siren,
  Truck,
  UserRound,
  Video,
  X,
} from "lucide-react";
import { containedFrameTransform, hlsPlaybackOffset, hlsProgramStartEpoch, incidentTrackingSource, playbackEpochAt, storedObjectTracks, trackFrameAt } from "../objectTrackReplay.mjs";
import { liveActivityEventId, liveActivityIncidentHref } from "../liveWorkspace.mjs";
import { adjacentIncident, incidentArrowNavigationAllowed, incidentDetectionFrameSize, incidentImageRenderRect, incidentObjectFocusAspect, incidentObjectFocusCropRect, incidentObjectFocusMaxScale, incidentObjectFocusStyle, incidentObjectIconName, incidentProgressiveImageWidth, incidentTrackingFrameSize, incidentZoomLayout, incidentTriggerLabel, normalizeIncidentThumbnailObjectFocus, normalizeIncidentThumbnailObjectFocusZoom } from "../incidentNavigation.mjs";
import { appUrl, fetch } from "./api.js";
import { formatDateTime } from "./format.js";
import { useStoredState, useModalFocus } from "./hooks.js";
import { eventSnapshotUrl, eventThumbnailUrl, eventClipUrl, eventStreamUrl } from "./mediaUrls.js";
import { prefersNativeMobilePlayback, ShakaVideo } from "./media.jsx";

export function eventObjects(event) {
  return event.objects || [];
}

export function eventEpoch(event) {
  const explicit = Number(event?.created_epoch);
  if (Number.isFinite(explicit)) return explicit;
  const parsed = new Date(event?.created_at || 0).getTime() / 1000;
  return Number.isFinite(parsed) ? parsed : null;
}

export function incidentClipWindow(event, before, after) {
  const anchor = eventEpoch(event);
  const children = event?.events || [];
  const childEpochs = children.map(eventEpoch).filter(Number.isFinite);
  const explicitStart = Number(event?.start_epoch);
  const explicitEnd = Number(event?.last_epoch);
  const start = Number.isFinite(explicitStart) ? explicitStart : childEpochs.length ? Math.min(...childEpochs) : anchor;
  const end = Number.isFinite(explicitEnd) ? explicitEnd : childEpochs.length ? Math.max(...childEpochs) : anchor;
  return {
    before: Math.max(0, before + (Number.isFinite(anchor) && Number.isFinite(start) ? anchor - start : 0)),
    after: Math.max(0, after + (Number.isFinite(anchor) && Number.isFinite(end) ? end - anchor : 0)),
  };
}

export function incidentLabels(incident) {
  const labels = Array.isArray(incident.labels)
    ? incident.labels
    : eventObjects(incident).filter((object) => object.incident_eligible !== false).map((object) => object.label).filter(Boolean);
  return Array.from(new Set(labels.filter(Boolean)));
}

export function IncidentObjectIcon({ label, size = 14 }) {
  const icons = {
    person: UserRound,
    face: ScanFace,
    car: CarFront,
    truck: Truck,
    bus: BusFront,
    bike: Bike,
    cat: Cat,
    dog: Dog,
    mower: Bot,
    object: CircleDot,
  };
  const Icon = icons[incidentObjectIconName(label)] || CircleDot;
  return <Icon size={size} strokeWidth={2.2} aria-hidden="true" />;
}

export function IncidentObjectBadges({ labels }) {
  if (!labels.length) {
    return <span className="pill quiet object-icon-pill" aria-label="Motion only" title="Motion only"><Radar size={14} strokeWidth={2.2} aria-hidden="true" /></span>;
  }
  return labels.slice(0, 3).map((label) => (
    <span className="pill object-icon-pill" key={label} aria-label={label} title={label}>
      <IncidentObjectIcon label={label} />
    </span>
  ));
}

export function IncidentSourceDot({ trigger, className = "", onClick = null, ariaLabel = "", title = "" }) {
  const source = String(trigger || "Camera").toUpperCase() === "EMA" ? "EMA" : "Camera";
  const classes = `incident-source-dot source-${source.toLowerCase()} ${className}`.trim();
  const label = ariaLabel || `${source} trigger`;
  if (onClick) {
    return <button type="button" className={classes} onClick={onClick} aria-label={label} title={title || label}><span className="sr-only">{source}</span></button>;
  }
  return <span className={classes} role="img" aria-label={label} title={title || label} />;
}

export function hasDetectedObjects(event) {
  if (typeof event.has_objects === "boolean") return event.has_objects;
  return eventObjects(event).some((object) => object.label && object.incident_eligible !== false) || incidentLabels(event).length > 0;
}

export function incidentZones(incident) {
  const zones = Array.isArray(incident.zones)
    ? incident.zones
    : eventObjects(incident).filter((object) => object.incident_eligible !== false).flatMap((object) => object.zones || []);
  return Array.from(new Set(zones.filter(Boolean)));
}


export function visualSearchObjects(event) {
  return eventObjects(event).filter((object) => (
    object?.label
    && String(object.label).trim()
    && object.snapshot_visible !== false
  ));
}

export function objectBoxes(event, incidentEligibleOnly = false) {
  return visualSearchObjects(event)
    .map((object, objectIndex) => ({ object, objectIndex, box: object?.box }))
    .filter(({ object, box }) => (!incidentEligibleOnly || object.incident_eligible !== false) && box && [box.x1, box.y1, box.x2, box.y2].every((value) => Number.isFinite(Number(value))))
    .map(({ object, objectIndex, box }) => ({
      objectIndex,
      trackId: Number.isInteger(Number(object.track_id)) ? Number(object.track_id) : null,
      label: object.label,
      confidence: object.confidence,
      depthMeters: Number(object?.depth_stats?.median_m),
      maskPolygon: Array.isArray(object.mask_polygon)
        ? object.mask_polygon.filter((point) => Array.isArray(point) && point.length >= 2).map((point) => [Number(point[0]), Number(point[1])])
        : [],
      x1: Number(box.x1),
      y1: Number(box.y1),
      x2: Number(box.x2),
      y2: Number(box.y2),
    }))
    .filter((box) => box.x2 > box.x1 && box.y2 > box.y1);
}

export function SnapshotImage({ event, alt, iconSize = 24, className = "", layerStyle = null, zoom = null, allowObjectFocus = true, objectFocusMode = null, objectFocusZoom = 1, objectFocusAspect = { width: 16, height: 9 }, objectFocusControls = true, showAnnotations = true, showTracking = false, incidentEligibleOnly = false, thumbnail = false, progressive = false, fullResolution = false, highQualityZoom = false, selectedObjectIndex = null, onSelectObject = null, onRequestFullResolution, onImageSize, children }) {
  const boxes = objectBoxes(event, incidentEligibleOnly);
  const tracks = storedObjectTracks(event);
  const boxCoordinateSize = incidentDetectionFrameSize(event);
  const trackCoordinateSize = incidentTrackingFrameSize(event);
  const coordinateSize = showTracking ? trackCoordinateSize : boxCoordinateSize;
  const progressiveImageKey = `${event?.representative_event_id || event?.id || "none"}:${event?.snapshot_path || "none"}:${event?.snapshot_url || "stored"}`;
  const focusMode = normalizeIncidentThumbnailObjectFocus(
    objectFocusMode ?? (allowObjectFocus ? "button" : "off"),
  );
  const focusZoom = normalizeIncidentThumbnailObjectFocusZoom(objectFocusZoom);
  const frameRef = useRef(null);
  const [imageSize, setImageSize] = useState(() => coordinateSize);
  const [loadedImageKey, setLoadedImageKey] = useState("");
  const [frameSize, setFrameSize] = useState(null);
  const [objectFocused, setObjectFocused] = useState(focusMode === "auto");
  const [progressiveState, setProgressiveState] = useState({ key: "", base: false, intermediate: false, full: false });
  const progressiveReady = progressiveState.key === progressiveImageKey ? progressiveState : { base: false, intermediate: false, full: false };
  const devicePixelRatio = Math.max(1, Math.min(4, Number(window.devicePixelRatio) || 1));
  const displayPixelWidth = (frameSize?.width || 0) * devicePixelRatio;
  const progressiveWidth = incidentProgressiveImageWidth(frameSize?.width, devicePixelRatio);
  const progressiveQuality = progressiveWidth > 1280 ? 90 : 86;
  const shouldLoadFullResolution = fullResolution || highQualityZoom || displayPixelWidth > 2560;
  const zoomLayout = useMemo(() => incidentZoomLayout(frameSize, zoom), [frameSize, zoom?.scale, zoom?.x, zoom?.y]);
  const renderingFrameSize = zoomLayout
    ? { width: zoomLayout.width, height: zoomLayout.height }
    : frameSize;
  const hasFocusableObjects = focusMode !== "off" && boxes.length > 0;
  const preferFocused = focusMode === "auto" || (focusMode === "button" && !objectFocusControls);
  const useServerObjectCrop = Boolean(thumbnail && hasFocusableObjects && (objectFocused || preferFocused));
  const renderedImage = useMemo(() => {
    // The server has already framed focused snapshots. Keep the whole crop
    // visible (especially in tall Mosaic cells) and map overlays to it.
    return incidentImageRenderRect(
      renderingFrameSize,
      imageSize,
      "contain",
    );
  }, [imageSize, renderingFrameSize?.height, renderingFrameSize?.width]);
  const canFocus = hasFocusableObjects && (useServerObjectCrop || renderedImage);
  const showFocusButton = canFocus && objectFocusControls;
  // Crop from the full snapshot on the server, then encode near display size.
  const focusThumbnailWidth = Math.max(320, Math.min(1280, Math.ceil((frameSize?.width || 160) * devicePixelRatio)));
  const focusThumbnailQuality = 90;
  const focusAspect = incidentObjectFocusAspect(objectFocusAspect, useServerObjectCrop);
  const thumbnailSrc = useServerObjectCrop
    ? eventThumbnailUrl(event, focusThumbnailWidth, focusThumbnailQuality, {
      objectFocus: true,
      incidentEligibleOnly,
      zoom: focusZoom,
      aspectWidth: focusAspect?.width,
      aspectHeight: focusAspect?.height,
    })
    : eventThumbnailUrl(event);
  const focusImageKey = `${progressiveImageKey}:object-focus:${focusThumbnailWidth}:${focusZoom}:${focusAspect?.width || 0}x${focusAspect?.height || 0}`;
  const imageReady = loadedImageKey === progressiveImageKey
    || (useServerObjectCrop && loadedImageKey === focusImageKey);
  const focusCrop = useMemo(() => {
    if (!useServerObjectCrop) return null;
    const sourceWidth = boxCoordinateSize?.width || imageSize?.width;
    const sourceHeight = boxCoordinateSize?.height || imageSize?.height;
    return incidentObjectFocusCropRect(
      sourceWidth,
      sourceHeight,
      boxes,
      focusZoom,
      focusAspect?.width,
      focusAspect?.height,
    );
  }, [boxCoordinateSize?.height, boxCoordinateSize?.width, boxes, focusAspect?.height, focusAspect?.width, focusZoom, imageSize?.height, imageSize?.width, useServerObjectCrop]);

  useLayoutEffect(() => {
    setObjectFocused(preferFocused && boxes.length > 0);
    setImageSize(coordinateSize);
  }, [progressiveImageKey, preferFocused, boxes.length]);

  useLayoutEffect(() => {
    const frame = frameRef.current;
    if (!frame) return undefined;
    function updateFrameSize() {
      const rect = frame.getBoundingClientRect();
      if (rect.width && rect.height) setFrameSize({ width: rect.width, height: rect.height });
    }
    updateFrameSize();
    const observer = new ResizeObserver(updateFrameSize);
    observer.observe(frame);
    return () => observer.disconnect();
  }, []);

  useLayoutEffect(() => {
    if (!coordinateSize || imageReady) return;
    setImageSize(coordinateSize);
  }, [coordinateSize?.height, coordinateSize?.width, imageReady]);

  function recordImageLoad(image, imageKey = progressiveImageKey) {
    if (image.naturalWidth && image.naturalHeight) {
      const size = { width: image.naturalWidth, height: image.naturalHeight };
      setImageSize(size);
      setLoadedImageKey(imageKey);
      onImageSize?.(size);
    }
  }

  function onImageLoad(loadEvent, imageKey = progressiveImageKey) {
    recordImageLoad(loadEvent.currentTarget, imageKey);
  }

  function markProgressiveReady(stage, loadEvent) {
    const image = loadEvent.currentTarget;
    if (!image.isConnected) return;
    recordImageLoad(image, progressiveImageKey);
    setProgressiveState((current) => ({
      ...(current.key === progressiveImageKey ? current : { key: progressiveImageKey, base: false, intermediate: false, full: false }),
      key: progressiveImageKey,
      [stage]: true,
    }));
  }

  const renderedBoxes = useMemo(() => {
    if (!renderedImage || !frameSize) return [];
    if (useServerObjectCrop && focusCrop) {
      const scaleX = renderedImage.width / focusCrop.width;
      const scaleY = renderedImage.height / focusCrop.height;
      return boxes.map((box) => ({
        ...box,
        left: renderedImage.x + (box.x1 - focusCrop.x1) * scaleX,
        top: renderedImage.y + (box.y1 - focusCrop.y1) * scaleY,
        width: (box.x2 - box.x1) * scaleX,
        height: (box.y2 - box.y1) * scaleY,
        maskPoints: box.maskPolygon
          .map(([x, y]) => `${renderedImage.x + (x - focusCrop.x1) * scaleX},${renderedImage.y + (y - focusCrop.y1) * scaleY}`)
          .join(" "),
      })).filter((box) => box.width > 0 && box.height > 0);
    }
    const sourceWidth = boxCoordinateSize?.width || imageSize?.width;
    const sourceHeight = boxCoordinateSize?.height || imageSize?.height;
    if (!sourceWidth || !sourceHeight) return [];
    const scaleX = renderedImage.width / sourceWidth;
    const scaleY = renderedImage.height / sourceHeight;
    return boxes.map((box) => ({
      ...box,
      left: renderedImage.x + box.x1 * scaleX,
      top: renderedImage.y + box.y1 * scaleY,
      width: (box.x2 - box.x1) * scaleX,
      height: (box.y2 - box.y1) * scaleY,
      maskPoints: box.maskPolygon.map(([x, y]) => `${renderedImage.x + x * scaleX},${renderedImage.y + y * scaleY}`).join(" "),
    })).filter((box) => box.width > 0 && box.height > 0);
  }, [boxes, boxCoordinateSize?.height, boxCoordinateSize?.width, focusCrop, frameSize, imageSize, renderedImage, useServerObjectCrop]);

  const renderedTracks = useMemo(() => {
    if (!renderedImage || !frameSize) return [];
    const sourceWidth = trackCoordinateSize?.width || imageSize?.width;
    const sourceHeight = trackCoordinateSize?.height || imageSize?.height;
    if (!sourceWidth || !sourceHeight) return [];
    const scaleX = renderedImage.width / sourceWidth;
    const scaleY = renderedImage.height / sourceHeight;
    return tracks.map((track) => ({
      ...track,
      left: renderedImage.x + track.x1 * scaleX,
      top: renderedImage.y + track.y1 * scaleY,
      width: (track.x2 - track.x1) * scaleX,
      height: (track.y2 - track.y1) * scaleY,
      pathPoints: track.trajectory.map(([, x, y]) => `${renderedImage.x + x * scaleX},${renderedImage.y + y * scaleY}`).join(" "),
    })).filter((track) => track.width > 0 && track.height > 0);
  }, [frameSize, imageSize, renderedImage, trackCoordinateSize?.height, trackCoordinateSize?.width, tracks]);

  const focusMaxScale = useMemo(
    () => incidentObjectFocusMaxScale(imageSize?.width, renderedImage?.width, devicePixelRatio),
    [devicePixelRatio, imageSize?.width, renderedImage?.width],
  );
  // Compact thumbs use server crop; expanded/manual viewers still CSS-zoom a full-frame raster.
  const focusStyle = useMemo(
    () => (!useServerObjectCrop && canFocus ? incidentObjectFocusStyle(frameSize, renderedBoxes, focusZoom, focusMaxScale) : null),
    [canFocus, focusMaxScale, focusZoom, frameSize, renderedBoxes, useServerObjectCrop],
  );

  const zoomLayerStyle = zoomLayout ? {
    inset: "auto",
    left: `${zoomLayout.left}px`,
    top: `${zoomLayout.top}px`,
    width: `${zoomLayout.width}px`,
    height: `${zoomLayout.height}px`,
  } : null;
  const activeLayerStyle = objectFocused && focusStyle ? focusStyle : zoomLayerStyle || layerStyle;
  const aspect = !useServerObjectCrop && imageSize ? `${imageSize.width} / ${imageSize.height}` : undefined;
  const prefersHighQualityRaster = highQualityZoom || (objectFocused && !useServerObjectCrop);
  const isTransforming = Boolean(activeLayerStyle?.transform);

  return (
    <div ref={frameRef} className={`snapshot-frame ${objectFocused ? "object-focused" : ""} ${useServerObjectCrop ? "object-focus-crop" : ""} ${prefersHighQualityRaster ? "high-quality-zoom" : ""} ${isTransforming ? "is-transforming" : ""} ${className}`} style={aspect ? { "--snapshot-aspect": aspect } : undefined}>
      <div className="snapshot-layer" style={activeLayerStyle || undefined}>
        {event?.snapshot_path && eventSnapshotUrl(event) ? (
          progressive && !useServerObjectCrop ? (
            <div className={`snapshot-progressive-stack ${progressiveReady.full ? "full-resolution-ready" : ""}`}>
              <img
                key={`${progressiveImageKey}-base`}
                className="snapshot-progressive-base"
                src={eventThumbnailUrl(event)}
                alt={alt}
                decoding="async"
                onLoad={(loadEvent) => markProgressiveReady("base", loadEvent)}
              />
              {progressiveReady.base ? (
                <img
                  key={`${progressiveImageKey}-intermediate`}
                  className={`snapshot-progressive-image snapshot-intermediate-image ${progressiveReady.intermediate ? "ready" : ""}`}
                  src={eventThumbnailUrl(event, progressiveWidth, progressiveQuality)}
                  alt=""
                  aria-hidden="true"
                  decoding="async"
                  onLoad={(loadEvent) => markProgressiveReady("intermediate", loadEvent)}
                />
              ) : null}
              {shouldLoadFullResolution ? (
                <img
                  key={`${progressiveImageKey}-full`}
                  className={`snapshot-progressive-image snapshot-full-resolution-image ${progressiveReady.full ? "ready" : ""}`}
                  src={eventSnapshotUrl(event)}
                  alt=""
                  aria-hidden="true"
                  decoding="sync"
                  fetchPriority="high"
                  onLoad={(loadEvent) => markProgressiveReady("full", loadEvent)}
                />
              ) : null}
            </div>
          ) : <img className={thumbnail ? "snapshot-thumbnail-image" : "snapshot-original-image"} src={thumbnail ? thumbnailSrc : eventSnapshotUrl(event)} alt={alt} loading={thumbnail ? "lazy" : undefined} decoding="async" onLoad={(loadEvent) => onImageLoad(loadEvent, thumbnail && useServerObjectCrop ? focusImageKey : progressiveImageKey)} />
        ) : <div className="empty-thumb"><Camera size={iconSize} /></div>}
        {imageReady && showAnnotations && (!showTracking || !renderedTracks.length) && renderedBoxes.length ? (
          <div className={`object-box-layer${onSelectObject ? " interactive" : ""}`} aria-hidden={onSelectObject ? undefined : "true"}>
            <svg className="object-mask-layer" viewBox={`0 0 ${frameSize.width} ${frameSize.height}`} preserveAspectRatio="none">
              {renderedBoxes.filter((box) => box.maskPoints).map((box, index) => (
                <polygon key={`mask-${box.label}-${index}`} points={box.maskPoints} />
              ))}
            </svg>
            {renderedBoxes.map((box, index) => {
              const selected = selectedObjectIndex != null && selectedObjectIndex !== ""
                && Number(selectedObjectIndex) === Number(box.objectIndex);
              const distanceLabel = Number.isFinite(box.depthMeters) ? ` ~${box.depthMeters.toFixed(1)}m` : "";
              const label = `${box.label}${box.confidence ? ` ${(box.confidence * 100).toFixed(0)}%` : ""}${distanceLabel}`;
              if (onSelectObject) {
                return (
                  <button
                    type="button"
                    className={`object-box selectable${selected ? " selected" : ""}`}
                    key={`${box.label}-${box.objectIndex}-${box.x1}-${box.y1}`}
                    style={{ left: `${box.left}px`, top: `${box.top}px`, width: `${box.width}px`, height: `${box.height}px` }}
                    onClick={(clickEvent) => {
                      clickEvent.stopPropagation();
                      onSelectObject({
                        objectIndex: box.objectIndex,
                        trackId: box.trackId,
                        label: box.label,
                      });
                    }}
                    aria-pressed={selected}
                    title={`Find similar: ${box.label}`}
                    aria-label={`Find similar incidents for ${box.label}`}
                  >
                    <strong>{label}</strong>
                  </button>
                );
              }
              return (
                <span
                  className="object-box"
                  key={`${box.label}-${index}-${box.x1}-${box.y1}`}
                  style={{ left: `${box.left}px`, top: `${box.top}px`, width: `${box.width}px`, height: `${box.height}px` }}
                >
                  <strong>{label}</strong>
                </span>
              );
            })}
          </div>
        ) : null}
        {imageReady && showTracking && renderedTracks.length ? (
          <div className="object-track-layer" aria-hidden="true">
            <svg viewBox={`0 0 ${frameSize.width} ${frameSize.height}`} preserveAspectRatio="none" aria-hidden="true">
              {renderedTracks.map((track) => (
                <g className={`object-track-color-${Math.abs(track.trackId) % 6}`} key={`trail-${track.trackId}`}>
                  {track.pathPoints ? <polyline className="object-track-trail" points={track.pathPoints} vectorEffect="non-scaling-stroke" /> : null}
                  <circle className="object-track-end" cx={track.left + track.width / 2} cy={track.top + track.height / 2} r="4" vectorEffect="non-scaling-stroke" />
                </g>
              ))}
            </svg>
            {renderedTracks.map((track) => (
              <span
                className={`object-track-box object-track-color-${Math.abs(track.trackId) % 6}`}
                key={`track-${track.trackId}`}
                style={{ left: `${track.left}px`, top: `${track.top}px`, width: `${track.width}px`, height: `${track.height}px` }}
              >
                <strong>#{track.trackId} {track.label}</strong>
              </span>
            ))}
          </div>
        ) : null}
      </div>
      {showFocusButton ? (
        <button
          type="button"
          className={`snapshot-focus-button ${objectFocused ? "active" : ""}`}
          onClick={(event) => {
            event.stopPropagation();
            if (!objectFocused) onRequestFullResolution?.();
            setObjectFocused((focused) => !focused);
          }}
          title={objectFocused ? "Show full snapshot" : "Focus detected objects"}
          aria-label={objectFocused ? "Show full snapshot" : "Focus detected objects"}
        >
          <Crop size={14} />
        </button>
      ) : null}
      {children}
    </div>
  );
}

export function StoredTrackVideoOverlay({ videoRef, tracks, coordinateSize, windowStartEpoch, mediaStartTime, mediaKey, sampleFps, lostTimeoutSeconds }) {
  const layerRef = useRef(null);
  const [playbackEpoch, setPlaybackEpoch] = useState(null);
  const [layerSize, setLayerSize] = useState(null);

  useEffect(() => {
    const layer = layerRef.current;
    if (!layer) return undefined;
    function updateLayerSize() {
      const rect = layer.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) setLayerSize({ width: rect.width, height: rect.height });
    }
    updateLayerSize();
    const observer = new ResizeObserver(updateLayerSize);
    observer.observe(layer);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !Number.isFinite(windowStartEpoch) || !Number.isFinite(mediaStartTime)) return undefined;
    let timer = null;
    let stopped = false;

    function update() {
      const epoch = playbackEpochAt(windowStartEpoch, video.currentTime, mediaStartTime);
      if (epoch !== null) setPlaybackEpoch(epoch);
    }

    function schedule() {
      if (stopped) return;
      update();
      if (!video.paused && !video.ended) timer = window.setTimeout(schedule, 100);
    }

    function onPlaying() {
      if (timer !== null) window.clearTimeout(timer);
      schedule();
    }

    function onSeeked() {
      update();
    }

    video.addEventListener("playing", onPlaying);
    video.addEventListener("seeked", onSeeked);
    video.addEventListener("timeupdate", update);
    if (!video.paused) onPlaying();
    return () => {
      stopped = true;
      if (timer !== null) window.clearTimeout(timer);
      video.removeEventListener("playing", onPlaying);
      video.removeEventListener("seeked", onSeeked);
      video.removeEventListener("timeupdate", update);
    };
  }, [videoRef, windowStartEpoch, mediaStartTime, mediaKey]);

  const visibleTracks = useMemo(() => {
    if (!Number.isFinite(playbackEpoch)) return [];
    const holdSeconds = Math.max(0.5, Number(lostTimeoutSeconds) || 3);
    return tracks.flatMap((track) => {
      const frame = trackFrameAt(track, playbackEpoch, { holdSeconds, sampleFps });
      return frame ? [{ ...track, ...frame }] : [];
    });
  }, [lostTimeoutSeconds, playbackEpoch, sampleFps, tracks]);

  const secondsUntilTracking = useMemo(() => {
    if (!Number.isFinite(playbackEpoch) || visibleTracks.length) return null;
    const nextEpoch = tracks
      .flatMap((track) => track.boxHistory.length ? [track.boxHistory[0][0]] : [])
      .filter((epoch) => epoch > playbackEpoch)
      .sort((left, right) => left - right)[0];
    return Number.isFinite(nextEpoch) ? Math.max(1, Math.ceil(nextEpoch - playbackEpoch)) : null;
  }, [playbackEpoch, tracks, visibleTracks.length]);

  const coordinateTransform = useMemo(
    () => containedFrameTransform(layerSize, coordinateSize),
    [coordinateSize?.height, coordinateSize?.width, layerSize],
  );
  const renderedTracks = useMemo(() => {
    if (!coordinateTransform) return [];
    const { x, y, scale } = coordinateTransform;
    return visibleTracks.map((track) => ({
      ...track,
      renderedBox: [
        x + track.box[0] * scale,
        y + track.box[1] * scale,
        x + track.box[2] * scale,
        y + track.box[3] * scale,
      ],
      renderedPath: track.path.map(([pathX, pathY]) => [x + pathX * scale, y + pathY * scale]),
    }));
  }, [coordinateTransform, visibleTracks]);

  if (!coordinateSize?.width || !coordinateSize?.height || !tracks.some((track) => track.boxHistory.length)) return null;
  return (
    <div ref={layerRef} className="object-track-video-layer" aria-hidden="true">
      {layerSize ? <svg viewBox={`0 0 ${layerSize.width} ${layerSize.height}`} preserveAspectRatio="none" aria-hidden="true">
        {renderedTracks.map((track) => (
          <g className={`object-track-color-${Math.abs(track.trackId) % 6} ${track.recovery ? "reid-recovered" : ""} ${track.estimated ? "track-estimated" : ""}`} key={`video-path-${track.trackId}`}>
            {track.renderedPath.length > 1 ? <polyline className="object-track-trail" points={track.renderedPath.map(([x, y]) => `${x},${y}`).join(" ")} vectorEffect="non-scaling-stroke" /> : null}
            <rect className="object-track-video-box" x={track.renderedBox[0]} y={track.renderedBox[1]} width={track.renderedBox[2] - track.renderedBox[0]} height={track.renderedBox[3] - track.renderedBox[1]} vectorEffect="non-scaling-stroke" />
          </g>
        ))}
      </svg> : null}
      {renderedTracks.map((track) => (
        <span
          className={`object-track-video-label object-track-color-${Math.abs(track.trackId) % 6} ${track.recovery ? "reid-recovered" : ""} ${track.estimated ? "track-estimated" : ""}`}
          key={`video-label-${track.trackId}`}
          style={{ left: `${track.renderedBox[0]}px`, top: `${track.renderedBox[1]}px` }}
        >
          #{track.trackId} {track.label}{track.estimated ? " · estimated" : ""}{track.recovery ? ` · ReID ${Math.round(track.recovery.similarity * 100)}%` : ""}
        </span>
      ))}
      {secondsUntilTracking ? <span className="object-track-video-waiting">Tracking begins in {secondsUntilTracking}s</span> : null}
    </div>
  );
}


export function IncidentListItem({ incident, cameraName, timeZone, selected, thumbnailAnnotations, thumbnailObjectFocus = "off", thumbnailObjectFocusZoom = 1, onSelect, onOpenOverlay }) {
  const labels = incidentLabels(incident);
  const trigger = incidentTriggerLabel(incident);
  const eventId = liveActivityEventId(incident);
  const time = incident.start_at || incident.created_at;
  const activityLabel = labels.length ? labels.join(", ") : "Motion only";
  return (
    <article className={`live-activity-item${selected ? " selected" : ""}`} aria-current={selected ? "true" : undefined}>
      <button type="button" className="live-activity-select" onClick={() => onSelect(incident)} aria-label={`Open ${cameraName} activity at ${formatDateTime(time, timeZone)}`}>
        <span className="live-activity-thumb"><SnapshotImage event={incident} alt="" className="live-activity-snapshot" thumbnail objectFocusMode={thumbnailObjectFocus} objectFocusZoom={thumbnailObjectFocusZoom} objectFocusControls={false} showAnnotations={thumbnailAnnotations} showTracking={false} /></span>
        <span className="live-activity-copy">
          <span className="live-activity-kind"><IncidentObjectBadges labels={labels} /><span className="sr-only">{activityLabel}</span></span>
          <b>{cameraName}</b>
          <time>{formatDateTime(time, timeZone)}</time>
        </span>
      </button>
      <IncidentSourceDot trigger={trigger} className="live-activity-trigger" onClick={() => onOpenOverlay(incident)} ariaLabel={`Preview exact ${trigger} event`} title={`${trigger} trigger`} />
      {!eventId ? <span className="sr-only">No exact event link is available.</span> : null}
    </article>
  );
}

export function detectionIou(left, right) {
  const x1 = Math.max(Number(left.x1), Number(right.x1));
  const y1 = Math.max(Number(left.y1), Number(right.y1));
  const x2 = Math.min(Number(left.x2), Number(right.x2));
  const y2 = Math.min(Number(left.y2), Number(right.y2));
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const leftArea = Math.max(0, left.x2 - left.x1) * Math.max(0, left.y2 - left.y1);
  const rightArea = Math.max(0, right.x2 - right.x1) * Math.max(0, right.y2 - right.y1);
  return intersection / Math.max(1, leftArea + rightArea - intersection);
}

export function formatDepthMeters(value) {
  const meters = Number(value);
  if (!Number.isFinite(meters) || meters <= 0) return "";
  return meters >= 10 ? `${Math.round(meters)}m` : `${meters.toFixed(1)}m`;
}

function depthOverlayColor(meters, minM = 0.5, maxM = 30) {
  const span = Math.max(0.1, maxM - minM);
  const ratio = Math.max(0, Math.min(1, (meters - minM) / span));
  const hue = 190 - ratio * 150;
  return `hsl(${hue}, 82%, 52%)`;
}

export function DebugDetectionOverlay({
  videoRef,
  active,
  confidence = 0.35,
  depth = false,
  depthLayer = "both",
  onStats,
}) {
  const canvasRef = useRef(null);
  const captureRef = useRef(document.createElement("canvas"));
  const tracksRef = useRef([]);
  const nextTrackIdRef = useRef(1);
  const heatmapImageRef = useRef(null);
  const pendingHeatmapRef = useRef("");

  useEffect(() => {
    if (!active) {
      tracksRef.current = [];
      heatmapImageRef.current = null;
      pendingHeatmapRef.current = "";
      const canvas = canvasRef.current;
      canvas?.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
      return undefined;
    }
    let disposed = false;
    let timer = null;
    let controller = null;

    function updateTracks(detections) {
      const now = performance.now();
      const available = [...tracksRef.current];
      const next = detections.map((object) => {
        let bestIndex = -1;
        let bestScore = 0.25;
        available.forEach((track, index) => {
          if (!track || track.label !== object.label) return;
          const score = detectionIou(track.box, object.box);
          if (score > bestScore) {
            bestScore = score;
            bestIndex = index;
          }
        });
        const previous = bestIndex >= 0 ? available.splice(bestIndex, 1)[0] : null;
        const depthMeters = Number(object?.depth_stats?.median_m);
        return {
          id: previous?.id || nextTrackIdRef.current++,
          label: object.label,
          confidence: Number(object.confidence) || 0,
          box: object.box,
          depthMeters: Number.isFinite(depthMeters) && depthMeters > 0 ? depthMeters : null,
          seenAt: now,
        };
      });
      tracksRef.current = [...next, ...available.filter((track) => now - track.seenAt < 1200)].slice(0, 40);
      return next;
    }

    function drawHeatmap(context, frameWidth, frameHeight, width, height, scale, offsetX, offsetY) {
      const image = heatmapImageRef.current;
      if (!image || !image.complete || !image.naturalWidth) return;
      context.save();
      context.globalAlpha = 0.32;
      context.drawImage(
        image,
        offsetX,
        offsetY,
        frameWidth * scale,
        frameHeight * scale,
      );
      context.restore();
    }

    function draw(tracks, frameWidth, frameHeight, heatmapRange) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas) return;
      const width = Math.max(1, video.clientWidth);
      const height = Math.max(1, video.clientHeight);
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      const context = canvas.getContext("2d");
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      context.clearRect(0, 0, width, height);
      const scale = Math.min(width / frameWidth, height / frameHeight);
      const offsetX = (width - frameWidth * scale) / 2;
      const offsetY = (height - frameHeight * scale) / 2;
      const showHeatmap = depth && ["both", "heatmap"].includes(depthLayer);
      const showBoxes = !depth || ["both", "boxes"].includes(depthLayer);
      if (showHeatmap) drawHeatmap(context, frameWidth, frameHeight, width, height, scale, offsetX, offsetY);
      if (!showBoxes) return;
      const minM = Number(heatmapRange?.min_m) || 0.5;
      const maxM = Number(heatmapRange?.max_m) || 30;
      context.font = "700 12px system-ui, sans-serif";
      context.lineWidth = 2;
      tracks.forEach((track) => {
        const x = offsetX + track.box.x1 * scale;
        const y = offsetY + track.box.y1 * scale;
        const boxWidth = (track.box.x2 - track.box.x1) * scale;
        const boxHeight = (track.box.y2 - track.box.y1) * scale;
        const distanceText = depth && track.depthMeters ? ` · ${formatDepthMeters(track.depthMeters)}` : "";
        const label = `#${track.id} ${track.label} ${Math.round(track.confidence * 100)}%${distanceText}`;
        const labelWidth = context.measureText(label).width + 10;
        const strokeColor = depth && track.depthMeters
          ? depthOverlayColor(track.depthMeters, minM, maxM)
          : "#2dd4bf";
        context.strokeStyle = strokeColor;
        context.fillStyle = depth && track.depthMeters
          ? "rgba(15, 23, 42, 0.84)"
          : "rgba(13, 148, 136, 0.88)";
        context.strokeRect(x, y, boxWidth, boxHeight);
        context.fillRect(x, Math.max(0, y - 20), labelWidth, 20);
        context.fillStyle = "#ffffff";
        context.fillText(label, x + 5, Math.max(14, y - 6));
      });
    }

    function queueHeatmap(base64Value, tracks, frameWidth, frameHeight, heatmapRange) {
      if (!base64Value) {
        heatmapImageRef.current = null;
        draw(tracks, frameWidth, frameHeight, heatmapRange);
        return;
      }
      if (pendingHeatmapRef.current === base64Value && heatmapImageRef.current?.complete) {
        draw(tracks, frameWidth, frameHeight, heatmapRange);
        return;
      }
      pendingHeatmapRef.current = base64Value;
      const image = new Image();
      image.onload = () => {
        if (disposed || pendingHeatmapRef.current !== base64Value) return;
        heatmapImageRef.current = image;
        draw(tracks, frameWidth, frameHeight, heatmapRange);
      };
      image.onerror = () => {
        if (disposed) return;
        heatmapImageRef.current = null;
        draw(tracks, frameWidth, frameHeight, heatmapRange);
      };
      image.src = `data:image/png;base64,${base64Value}`;
    }

    async function sample() {
      const video = videoRef.current;
      if (disposed) return;
      if (!video || video.readyState < 2 || video.paused || document.hidden) {
        timer = window.setTimeout(sample, depth ? 450 : 350);
        return;
      }
      const capture = captureRef.current;
      const width = Math.min(960, video.videoWidth || 960);
      const height = Math.max(1, Math.round(width * (video.videoHeight || 540) / (video.videoWidth || 960)));
      capture.width = width;
      capture.height = height;
      capture.getContext("2d").drawImage(video, 0, 0, width, height);
      try {
        const blob = await new Promise((resolve) => capture.toBlob(resolve, "image/jpeg", 0.78));
        if (!blob || disposed) return;
        controller = new AbortController();
        const params = new URLSearchParams({
          confidence: Number(confidence).toFixed(2),
        });
        if (depth) {
          params.set("depth", "1");
          if (["both", "heatmap"].includes(depthLayer)) params.set("heatmap", "1");
        }
        const response = await fetch(`/api/detector/frame?${params.toString()}`, {
          method: "POST",
          headers: { "Content-Type": "image/jpeg" },
          body: blob,
          signal: controller.signal,
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const payload = await response.json();
        if (disposed) return;
        const tracks = updateTracks(payload.objects || []);
        const frameWidth = payload.width || width;
        const frameHeight = payload.height || height;
        if (depth && payload.heatmap_png_b64 && ["both", "heatmap"].includes(depthLayer)) {
          queueHeatmap(payload.heatmap_png_b64, tracks, frameWidth, frameHeight, payload.heatmap_range_m);
        } else {
          draw(tracks, frameWidth, frameHeight, payload.heatmap_range_m);
        }
        onStats?.({
          inferenceMs: payload.elapsed_ms,
          detectMs: payload.detect_ms,
          depthMs: payload.depth_ms,
          objects: tracks.length,
          tracks: tracks.map((track) => track.id),
          depthError: payload.depth_error || "",
          heatmapRange: payload.heatmap_range_m || null,
        });
      } catch (error) {
        if (!disposed && error.name !== "AbortError") onStats?.({ error: error.message || "Detection failed" });
      }
      if (!disposed) timer = window.setTimeout(sample, depth ? 900 : 500);
    }

    sample();
    return () => {
      disposed = true;
      controller?.abort();
      window.clearTimeout(timer);
    };
  }, [active, confidence, depth, depthLayer, videoRef, onStats]);

  return <canvas ref={canvasRef} className="event-detection-canvas" aria-hidden="true" />;
}

export function EventOverlay({ event, events, timeZone, onClose, onSelect, onRefresh }) {
  const modalRef = useModalFocus(onClose);
  const clipVideoRef = useRef(null);
  const mediaRef = useRef(null);
  const comparisonPanelRef = useRef(null);
  const gestureRef = useRef({ mode: null, pointerId: null, startX: 0, startY: 0, panX: 0, panY: 0, moved: false, pinchDistance: 0, scale: 1 });
  const [clipInfo, setClipInfo] = useState(null);
  const [clipLoading, setClipLoading] = useState(false);
  const [clipError, setClipError] = useState("");
  const [playback, setPlayback] = useState(null);
  const [playbackOriginTime, setPlaybackOriginTime] = useState(null);
  const [videoActive, setVideoActive] = useState(false);
  const [detectionDebug, setDetectionDebug] = useState(false);
  const [detectionDebugStats, setDetectionDebugStats] = useState(null);
  const [trackingVisible, setTrackingVisible] = useState(false);
  const [trackingComparison, setTrackingComparison] = useState(null);
  const [trackingComparisonEngine, setTrackingComparisonEngine] = useState(null);
  const [trackingComparisonLoading, setTrackingComparisonLoading] = useState(false);
  const [trackingComparisonError, setTrackingComparisonError] = useState("");
  const [trackingComparisonHistory, setTrackingComparisonHistory] = useState({ items: [], summary: null });
  const [trackingVerdictLoading, setTrackingVerdictLoading] = useState(false);
  const [analysisToolsOpen, setAnalysisToolsOpen] = useState(false);
  const [zoom, setZoom] = useState({ scale: 1, x: 0, y: 0 });
  const [mediaSize, setMediaSize] = useState(() => incidentTrackingFrameSize(event));
  const [fullSnapshotRequested, setFullSnapshotRequested] = useState(false);
  const zoomRef = useRef(zoom);
  const [manualConfidence, setManualConfidence] = useStoredState("survng.manualDetectionConfidence.v1", "0.35");
  const [manualDetection, setManualDetection] = useState(null);
  const [manualLoading, setManualLoading] = useState(false);
  const [manualError, setManualError] = useState("");
  const trackingSource = incidentTrackingSource(event);
  const incidentTrackingEvent = trackingSource && trackingSource !== event ? trackingSource : null;
  const viewerEvent = incidentTrackingEvent ? {
    ...incidentTrackingEvent,
    start_epoch: event.start_epoch,
    last_epoch: event.last_epoch,
    start_at: event.start_at,
    end_at: event.end_at,
    event_count: event.event_count,
    events: event.events || [],
  } : event;
  const displayedEvent = manualDetection ? { ...viewerEvent, objects: manualDetection.objects || [] } : viewerEvent;
  const comparisonTracking = trackingComparisonEngine ? trackingComparison?.engines?.[trackingComparisonEngine] : null;
  const trackingEvent = comparisonTracking
    ? {
      ...displayedEvent, object_tracking: {
        ...comparisonTracking,
        sample_fps: trackingComparison.sample_fps,
        lost_timeout_seconds: trackingComparison.lost_timeout_seconds,
        frame_width: trackingComparison.frame_width,
        frame_height: trackingComparison.frame_height,
      }
    }
    : displayedEvent;
  const storedTracks = storedObjectTracks(trackingEvent);
  const reidDiagnostics = trackingEvent.object_tracking?.reid_diagnostics || {};
  const reidAttemptReasons = reidDiagnostics.inference_attempts_by_reason || {};
  const replayTrackCount = storedTracks.filter((track) => track.boxHistory.length).length;
  const manualConfidenceNumber = Number(manualConfidence);
  const safeManualConfidence = Number.isFinite(manualConfidenceNumber) ? Math.max(0.01, Math.min(0.99, manualConfidenceNumber)) : 0.35;
  const manualEventId = Number(viewerEvent.representative_event_id || viewerEvent.id);
  const downloadName = `survng-${String(viewerEvent.camera_id || "camera")}-${String(viewerEvent.created_at || viewerEvent.id || "event").replace(/[^0-9A-Za-z_-]+/g, "-")}.mp4`;

  useEffect(() => {
    setTrackingVisible(false);
  }, [event.id]);

  useEffect(() => {
    let cancelled = false;
    async function loadClipSettings() {
      const eventId = Number(viewerEvent.representative_event_id || viewerEvent.id);
      if (!Number.isFinite(eventId)) {
        setClipInfo(null);
        setClipLoading(false);
        setClipError("No event video available");
        return;
      }
      setClipInfo(null);
      setPlayback(null);
      setPlaybackOriginTime(null);
      setClipLoading(true);
      setClipError("");
      setVideoActive(false);
      const info = await loadIncidentClipInfo(viewerEvent, () => cancelled, prefersNativeMobilePlayback());
      if (!info) return;
      setClipInfo(info);
      setPlayback(prefersNativeMobilePlayback()
        ? { url: info.downloadUrl, mimeType: "video/mp4" }
        : { url: info.streamUrl, mimeType: "application/vnd.apple.mpegurl" });
    }
    loadClipSettings();
    return () => { cancelled = true; };
  }, [viewerEvent.id, viewerEvent.representative_event_id, viewerEvent.start_epoch, viewerEvent.last_epoch]);

  useEffect(() => {
    let cancelled = false;
    const cameraId = String(event.camera_id || "");
    setTrackingComparisonHistory({ items: [], summary: null });
    if (!cameraId) return undefined;
    fetch(`/api/tracking-comparisons?camera_id=${encodeURIComponent(cameraId)}&limit=10`)
      .then(async (response) => {
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.detail || "Comparison history unavailable");
        if (!cancelled) setTrackingComparisonHistory(payload);
      })
      .catch(() => { });
    return () => { cancelled = true; };
  }, [event.camera_id]);

  function playEventClip() {
    if (!clipInfo || clipError) return;
    setVideoActive(true);
    const video = clipVideoRef.current;
    if (!video) return;
    if (video.ended || video.currentTime >= Math.max(0, video.duration - 0.1)) video.currentTime = 0;
    video.play().catch(() => { });
  }


  function clampZoom(nextZoom) {
    const scale = Math.max(1, Math.min(6, nextZoom.scale));
    if (scale === 1) return { scale: 1, x: 0, y: 0 };
    const box = mediaRef.current?.getBoundingClientRect();
    const limitX = box ? box.width * (scale - 1) / 2 : 0;
    const limitY = box ? box.height * (scale - 1) / 2 : 0;
    return {
      scale,
      x: Math.max(-limitX, Math.min(limitX, nextZoom.x || 0)),
      y: Math.max(-limitY, Math.min(limitY, nextZoom.y || 0)),
    };
  }

  function updateZoom(updater) {
    setZoom((current) => {
      const candidate = typeof updater === "function" ? updater(current) : updater;
      const next = clampZoom(candidate);
      zoomRef.current = next;
      return next;
    });
  }

  function zoomAround(clientX, clientY, factor) {
    const box = mediaRef.current?.getBoundingClientRect();
    if (!box) return;
    setVideoActive(false);
    setFullSnapshotRequested(true);
    updateZoom((current) => {
      const nextScale = Math.max(1, Math.min(6, current.scale * factor));
      if (nextScale === 1) return { scale: 1, x: 0, y: 0 };
      const anchorX = clientX - box.left - box.width / 2;
      const anchorY = clientY - box.top - box.height / 2;
      const scaleRatio = nextScale / current.scale;
      return {
        scale: nextScale,
        x: anchorX - (anchorX - current.x) * scaleRatio,
        y: anchorY - (anchorY - current.y) * scaleRatio,
      };
    });
  }

  function onMediaWheel(wheelEvent) {
    if (videoActive) return;
    wheelEvent.preventDefault();
    const delta = Math.max(-120, Math.min(120, wheelEvent.deltaY));
    zoomAround(wheelEvent.clientX, wheelEvent.clientY, Math.exp(-delta * 0.0017));
  }

  function touchDistance(touches) {
    if (touches.length < 2) return 0;
    const dx = touches[0].clientX - touches[1].clientX;
    const dy = touches[0].clientY - touches[1].clientY;
    return Math.hypot(dx, dy);
  }

  function touchCenter(touches) {
    return {
      x: (touches[0].clientX + touches[1].clientX) / 2,
      y: (touches[0].clientY + touches[1].clientY) / 2,
    };
  }

  function onMediaTouchStart(touchEvent) {
    if (videoActive) return;
    if (touchEvent.touches.length === 2) {
      touchEvent.preventDefault();
      setFullSnapshotRequested(true);
      const center = touchCenter(touchEvent.touches);
      const current = zoomRef.current;
      gestureRef.current = { mode: "pinch", pointerId: null, pinchDistance: touchDistance(touchEvent.touches), scale: current.scale, startX: center.x, startY: center.y, panX: current.x, panY: current.y, moved: true };
    } else if (touchEvent.touches.length === 1 && zoomRef.current.scale > 1) {
      touchEvent.preventDefault();
      const touch = touchEvent.touches[0];
      const current = zoomRef.current;
      gestureRef.current = { mode: "touch-pan", pointerId: null, pinchDistance: 0, scale: current.scale, startX: touch.clientX, startY: touch.clientY, panX: current.x, panY: current.y, moved: false };
    }
  }

  function onMediaTouchMove(touchEvent) {
    if (videoActive) return;
    const gesture = gestureRef.current;
    if (touchEvent.touches.length === 2 && gesture.mode === "pinch" && gesture.pinchDistance) {
      touchEvent.preventDefault();
      const box = mediaRef.current?.getBoundingClientRect();
      if (!box) return;
      const distance = touchDistance(touchEvent.touches);
      const center = touchCenter(touchEvent.touches);
      const nextScale = Math.max(1, Math.min(6, gesture.scale * (distance / gesture.pinchDistance)));
      const scaleRatio = nextScale / gesture.scale;
      const anchorX = gesture.startX - box.left - box.width / 2;
      const anchorY = gesture.startY - box.top - box.height / 2;
      updateZoom({
        scale: nextScale,
        x: anchorX - (anchorX - gesture.panX) * scaleRatio + center.x - gesture.startX,
        y: anchorY - (anchorY - gesture.panY) * scaleRatio + center.y - gesture.startY,
      });
      return;
    }
    if (touchEvent.touches.length === 1 && gesture.mode === "touch-pan" && zoomRef.current.scale > 1) {
      touchEvent.preventDefault();
      const touch = touchEvent.touches[0];
      const dx = touch.clientX - gesture.startX;
      const dy = touch.clientY - gesture.startY;
      if (Math.abs(dx) + Math.abs(dy) > 4) gesture.moved = true;
      updateZoom({ scale: zoomRef.current.scale, x: gesture.panX + dx, y: gesture.panY + dy });
    }
  }

  function onMediaTouchEnd(touchEvent) {
    if (touchEvent.touches.length === 1 && zoomRef.current.scale > 1) {
      const touch = touchEvent.touches[0];
      const current = zoomRef.current;
      gestureRef.current = { mode: "touch-pan", pointerId: null, pinchDistance: 0, scale: current.scale, startX: touch.clientX, startY: touch.clientY, panX: current.x, panY: current.y, moved: true };
      return;
    }
    gestureRef.current.mode = null;
    gestureRef.current.pinchDistance = 0;
  }

  function onMediaPointerDown(pointerEvent) {
    if (videoActive || pointerEvent.pointerType === "touch" || zoomRef.current.scale <= 1) return;
    pointerEvent.preventDefault();
    pointerEvent.currentTarget.setPointerCapture(pointerEvent.pointerId);
    const current = zoomRef.current;
    gestureRef.current = { mode: "pointer-pan", pointerId: pointerEvent.pointerId, startX: pointerEvent.clientX, startY: pointerEvent.clientY, panX: current.x, panY: current.y, moved: false, pinchDistance: 0, scale: current.scale };
  }

  function onMediaPointerMove(pointerEvent) {
    const gesture = gestureRef.current;
    if (gesture.pointerId !== pointerEvent.pointerId || zoomRef.current.scale <= 1) return;
    pointerEvent.preventDefault();
    const dx = pointerEvent.clientX - gesture.startX;
    const dy = pointerEvent.clientY - gesture.startY;
    if (Math.abs(dx) + Math.abs(dy) > 4) gesture.moved = true;
    updateZoom({ scale: zoomRef.current.scale, x: gesture.panX + dx, y: gesture.panY + dy });
  }

  function onMediaPointerUp(pointerEvent) {
    if (gestureRef.current.pointerId === pointerEvent.pointerId) {
      gestureRef.current.pointerId = null;
      gestureRef.current.mode = null;
    }
  }

  function onMediaClick() {
    if (gestureRef.current.moved) {
      gestureRef.current.moved = false;
      return;
    }
    if (zoom.scale > 1 || videoActive) return;
    if (clipInfo && !clipError) playEventClip();
  }

  function resetZoom() {
    const reset = { scale: 1, x: 0, y: 0 };
    zoomRef.current = reset;
    setZoom(reset);
  }

  useLayoutEffect(() => {
    resetZoom();
    setMediaSize(incidentTrackingFrameSize(trackingEvent));
    setFullSnapshotRequested(false);
    setDetectionDebug(false);
    setDetectionDebugStats(null);
    setTrackingVisible(false);
    setAnalysisToolsOpen(false);
    setTrackingComparison(null);
    setTrackingComparisonEngine(null);
    setTrackingComparisonLoading(false);
    setTrackingComparisonError("");
    setManualDetection(null);
    setManualError("");
    setManualLoading(false);
  }, [event.id]);

  useEffect(() => {
    if (!trackingComparison || !comparisonPanelRef.current) return undefined;
    const frame = window.requestAnimationFrame(() => {
      comparisonPanelRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [trackingComparison]);

  const mediaStyle = useMemo(() => {
    if (!mediaSize?.width || !mediaSize?.height) return undefined;
    const ratio = mediaSize.width / mediaSize.height;
    return {
      "--snapshot-aspect": `${mediaSize.width} / ${mediaSize.height}`,
      "--event-media-fit-width": `${ratio * 72}vh`,
      "--event-media-mobile-fit-width": `${ratio * 70}dvh`,
      "--event-panel-fit-width": `calc(${ratio * 72}vh + 24px)`,
    };
  }, [mediaSize]);

  async function runManualDetection() {
    if (!Number.isFinite(manualEventId)) {
      setManualError("No event id available");
      return;
    }
    setManualLoading(true);
    setManualError("");
    try {
      const response = await fetch(`/api/events/${manualEventId}/detect?confidence=${safeManualConfidence.toFixed(2)}`, { method: "POST" });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "Manual detection failed");
      setManualDetection(payload);
      if (onRefresh) onRefresh();
      setVideoActive(false);
      resetZoom();
    } catch (error) {
      setManualError(error.message || "Manual detection failed");
    } finally {
      setManualLoading(false);
    }
  }

  async function runTrackingComparison() {
    if (!Number.isFinite(manualEventId) || trackingComparisonLoading) return;
    setAnalysisToolsOpen(true);
    setTrackingComparisonLoading(true);
    setTrackingComparisonError("");
    setTrackingComparisonEngine(null);
    try {
      const response = await fetch(`/api/events/${manualEventId}/tracking-comparison?duration_seconds=30`, { method: "POST" });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "Tracking comparison failed");
      setTrackingComparison(payload);
      setTrackingComparisonHistory((current) => ({
        items: payload.comparison
          ? [payload.comparison, ...current.items.filter((item) => Number(item.id) !== Number(payload.comparison.id))].slice(0, 10)
          : current.items,
        summary: payload.evidence_summary || current.summary,
      }));
    } catch (error) {
      setTrackingComparisonError(error.message || "Tracking comparison failed");
    } finally {
      setTrackingComparisonLoading(false);
    }
  }

  async function saveTrackingVerdict(verdict) {
    const comparisonId = Number(trackingComparison?.comparison_id);
    if (!Number.isFinite(comparisonId) || trackingVerdictLoading) return;
    setTrackingVerdictLoading(true);
    setTrackingComparisonError("");
    try {
      const response = await fetch(`/api/tracking-comparisons/${comparisonId}/verdict`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ verdict }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "Could not save comparison verdict");
      setTrackingComparison((current) => ({ ...current, verdict }));
      setTrackingComparisonHistory((current) => {
        const comparison = payload.comparison;
        const items = comparison
          ? [comparison, ...current.items.filter((item) => Number(item.id) !== Number(comparison.id))].slice(0, 10)
          : current.items;
        return { items, summary: payload.summary || current.summary };
      });
    } catch (error) {
      setTrackingComparisonError(error.message || "Could not save comparison verdict");
    } finally {
      setTrackingVerdictLoading(false);
    }
  }

  async function replayTrackingComparison(implementation) {
    const engine = trackingComparison?.engines?.[implementation];
    if (!engine || !clipInfo) return;
    const after = Math.max(0.1, Number(trackingComparison.requested_duration_seconds || trackingComparison.duration_seconds || 0));
    const anchorEpoch = eventEpoch(viewerEvent);
    const requestedWindowStartEpoch = Number.isFinite(anchorEpoch) ? anchorEpoch : null;
    const streamUrl = eventStreamUrl(manualEventId, 0, after);
    const timelineStartEpoch = Number.isFinite(requestedWindowStartEpoch)
      ? await eventStreamTimelineStart(streamUrl, requestedWindowStartEpoch)
      : null;
    const nextClip = {
      ...clipInfo,
      before: 0,
      after,
      duration: after,
      streamUrl,
      downloadUrl: eventClipUrl(manualEventId, 0, after),
      windowStartEpoch: timelineStartEpoch,
      requestedWindowStartEpoch,
      initialPlaybackOffset: 0,
      playbackStartOffset: hlsPlaybackOffset(requestedWindowStartEpoch, timelineStartEpoch, 0),
    };
    setTrackingComparisonEngine(implementation);
    setTrackingVisible(true);
    setDetectionDebug(false);
    setClipError("");
    setClipLoading(true);
    setPlaybackOriginTime(null);
    setClipInfo(nextClip);
    setPlayback({
      url: nextClip.streamUrl,
      mimeType: "application/vnd.apple.mpegurl",
      key: `comparison-${implementation}-${Date.now()}`,
    });
    setVideoActive(true);
  }

  useEffect(() => {
    function onKey(keyEvent) {
      if (keyEvent.key !== "ArrowLeft" && keyEvent.key !== "ArrowRight") return;
      if (!incidentArrowNavigationAllowed(keyEvent.target)) return;
      const direction = keyEvent.key === "ArrowRight" ? 1 : -1;
      const nextIncident = adjacentIncident(events, event, direction);
      if (!nextIncident) return;
      keyEvent.preventDefault();
      onSelect(nextIncident);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [event.id, events, onClose, onSelect]);

  return createPortal((
    <div ref={modalRef} className="event-overlay" role="dialog" aria-modal="true" aria-label="Event image">
      <button className="live-overlay-backdrop" type="button" onClick={onClose} aria-label="Close event image" />
      <section className="event-overlay-panel" style={mediaStyle}>
        <div className="event-detail-head">
          <div>
            <h2>{viewerEvent.camera_id}</h2>
            <time>{formatDateTime(viewerEvent.created_at, timeZone)}</time>
          </div>
          <div className="overlay-actions">
            <a
              className="nav-button tile-control-button"
              href={appUrl(liveActivityIncidentHref(viewerEvent))}
              aria-label="Open this event in Incidents"
              title="Open this event in Incidents"
            >
              <Siren size={16} /> Incident
            </a>
            <button
              type="button"
              className="tile-control-button"
              onClick={runTrackingComparison}
              disabled={trackingComparisonLoading || !Number.isFinite(manualEventId)}
              title="Run Hybrid and FastTrack on the same 30-second recording window"
            >
              <Gauge size={16} /> {trackingComparisonLoading ? "Comparing" : "Compare"}
            </button>
            <button
              type="button"
              className={`tile-control-button tracking-trail-toggle ${trackingVisible ? "active" : ""}`}
              onClick={() => setTrackingVisible((visible) => !visible)}
              disabled={!storedTracks.length}
              title={storedTracks.length ? `${trackingVisible ? "Hide" : "Show"} stored object tracking${replayTrackCount ? " on the snapshot and video" : " on the event snapshot"}` : "No stored tracks for this event; run Compare to generate an offline replay"}
              aria-label={storedTracks.length ? `${trackingVisible ? "Hide" : "Show"} stored object tracks` : "Stored object tracks unavailable"}
              aria-pressed={trackingVisible}
            >
              <ListTree size={16} /> Tracks
            </button>
            <button
              type="button"
              className={`tile-control-button debug-detection-toggle ${detectionDebug ? "active" : ""}`}
              onClick={() => {
                if (!videoActive) playEventClip();
                setDetectionDebug((enabled) => {
                  if (!enabled) setTrackingVisible(false);
                  return !enabled;
                });
              }}
              disabled={!clipInfo || Boolean(clipError)}
              title="Toggle real-time OpenVINO detection and tracking"
              aria-pressed={detectionDebug}
            >
              <Activity size={16} /> AI
            </button>
            {clipInfo && !clipError ? (
              <a className="tile-control-button icon-only" href={clipInfo.downloadUrl} download={downloadName} title="Download event video" aria-label="Download event video">
                <Download size={18} />
              </a>
            ) : (
              <span className="tile-control-button icon-only disabled" title={clipError || "Event video not ready"} aria-label="Event video not ready">
                <Download size={18} />
              </span>
            )}
            <button type="button" className="tile-control-button icon-only" data-modal-initial onClick={onClose} aria-label="Close event image">
              <X size={18} />
            </button>
          </div>
        </div>
        <div
          ref={mediaRef}
          className={`event-detail-media event-detail-play-target ${zoom.scale > 1 ? "zoomed" : ""}`}
          role="button"
          tabIndex={0}
          onClick={onMediaClick}
          onDoubleClick={resetZoom}
          onWheel={onMediaWheel}
          onTouchStart={onMediaTouchStart}
          onTouchMove={onMediaTouchMove}
          onTouchEnd={onMediaTouchEnd}
          onTouchCancel={onMediaTouchEnd}
          onPointerDown={onMediaPointerDown}
          onPointerMove={onMediaPointerMove}
          onPointerUp={onMediaPointerUp}
          onPointerCancel={onMediaPointerUp}
          onKeyDown={(keyEvent) => {
            if (keyEvent.key === "0") resetZoom();
            if ((keyEvent.key === "Enter" || keyEvent.key === " ") && clipInfo && !clipError && zoom.scale === 1) {
              keyEvent.preventDefault();
              playEventClip();
            }
          }}
          title={zoom.scale > 1 ? "Drag to pan. Double-click to reset zoom." : clipError || (clipLoading ? "Preparing event clip" : "Scroll or pinch to zoom. Click to play event video.")}
        >
          <SnapshotImage
            event={trackingEvent}
            alt="selected event snapshot"
            iconSize={42}
            className="event-snapshot-frame"
            zoom={zoom}
            allowObjectFocus={zoom.scale === 1 && !videoActive}
            progressive
            fullResolution={fullSnapshotRequested}
            highQualityZoom={zoom.scale > 1}
            onRequestFullResolution={() => setFullSnapshotRequested(true)}
            showAnnotations
            showTracking={trackingVisible && !manualDetection}
            incidentEligibleOnly
            onImageSize={setMediaSize}
          />
          {videoActive && clipInfo && playback && !clipError ? (
            <>
              {playback.mimeType === "video/mp4" ? <video
                key={playback.key || playback.url}
                className="event-video-layer"
                ref={clipVideoRef}
                src={playback.url}
                autoPlay
                controls
                playsInline
                preload="metadata"
                onLoadedMetadata={(event) => {
                  const video = event.currentTarget;
                  setPlaybackOriginTime(0);
                  const playbackStartOffset = Math.max(0, Number(clipInfo.playbackStartOffset) || 0);
                  if (playbackStartOffset > 0) {
                    video.currentTime = Number.isFinite(video.duration)
                      ? Math.min(playbackStartOffset, Math.max(0, video.duration - 0.25))
                      : playbackStartOffset;
                  }
                  setClipLoading(false);
                  setClipError("");
                }}
                onError={() => {
                  setClipLoading(false);
                  setVideoActive(false);
                  setClipError("No recording window found");
                }}
                onClick={(event) => event.stopPropagation()}
              /> : <ShakaVideo
                key={playback.key || playback.url}
                className="event-video-layer"
                ref={clipVideoRef}
                src={playback.url}
                mimeType={playback.mimeType}
                autoPlay
                controls
                playsInline
                preload="metadata"
                onReady={(_player, video) => {
                  setPlaybackOriginTime(0);
                  const playbackStartOffset = Math.max(0, Number(clipInfo.playbackStartOffset) || 0);
                  if (playbackStartOffset > 0 && video) {
                    const seekToSelectedEvent = () => {
                      const targetTime = playbackStartOffset;
                      video.currentTime = Number.isFinite(video.duration)
                        ? Math.min(targetTime, Math.max(0, video.duration - 0.25))
                        : targetTime;
                    };
                    if (video.paused) video.addEventListener("playing", seekToSelectedEvent, { once: true });
                    else seekToSelectedEvent();
                  }
                  setClipLoading(false);
                  setClipError("");
                }}
                onError={() => {
                  if (playback.url !== clipInfo.downloadUrl) {
                    setClipLoading(true);
                    setPlaybackOriginTime(null);
                    setClipInfo((current) => current ? {
                      ...current,
                      windowStartEpoch: current.requestedWindowStartEpoch,
                      playbackStartOffset: current.initialPlaybackOffset,
                    } : current);
                    setPlayback({ url: clipInfo.downloadUrl, mimeType: "video/mp4" });
                  } else {
                    setClipLoading(false);
                    setVideoActive(false);
                    setClipError("No recording window found");
                  }
                }}
                onClick={(event) => event.stopPropagation()}
              />}
              <DebugDetectionOverlay
                videoRef={clipVideoRef}
                active={detectionDebug}
                confidence={safeManualConfidence}
                onStats={setDetectionDebugStats}
              />
              {trackingVisible ? (
                <StoredTrackVideoOverlay
                  videoRef={clipVideoRef}
                  tracks={storedTracks}
                  coordinateSize={{
                    width: Number(trackingEvent.object_tracking?.frame_width) || mediaSize?.width,
                    height: Number(trackingEvent.object_tracking?.frame_height) || mediaSize?.height,
                  }}
                  windowStartEpoch={clipInfo.windowStartEpoch}
                  mediaStartTime={playbackOriginTime}
                  mediaKey={playback.key || playback.url}
                  sampleFps={trackingEvent.object_tracking?.sample_fps}
                  lostTimeoutSeconds={trackingEvent.object_tracking?.lost_timeout_seconds}
                />
              ) : null}
              {clipLoading ? <div className="event-video-preparing">Preparing incident video...</div> : null}
            </>
          ) : null}
          {videoActive && detectionDebug && detectionDebugStats ? (
            <div className={`event-detection-stats ${detectionDebugStats.error ? "error" : ""}`}>
              {detectionDebugStats.error
                ? detectionDebugStats.error
                : `${detectionDebugStats.inferenceMs ?? "--"} ms · ${detectionDebugStats.objects ?? 0} objects`}
            </div>
          ) : null}
        </div>
        <div className="event-detail-body">
          <details
            className="event-analysis-details"
            open={analysisToolsOpen}
            onToggle={(toggleEvent) => setAnalysisToolsOpen(toggleEvent.currentTarget.open)}
          >
            <summary>Analysis tools &amp; diagnostics</summary>
            <div className="event-analysis-details-body">
              {storedTracks.length ? (
                <div className="event-track-summary">
                  <span className="muted">{trackingComparisonEngine ? "Comparison replay" : "Stored tracking"}</span>
                  <strong>{storedTracks.length} track{storedTracks.length === 1 ? "" : "s"} · {String(trackingEvent.object_tracking?.implementation || "tracker").replaceAll("_", " ")} · {Number(trackingEvent.object_tracking?.sample_fps || 0) || "?"} FPS</strong>
                  <small>{replayTrackCount ? `${replayTrackCount} track${replayTrackCount === 1 ? "" : "s"} can replay over video. Dashed, faded boxes are estimated positions during the configured lost-object grace period. Snapshot boxes mark each track's last stored position.` : "Paths show sampled object centers over time. The box marks each track's last stored position."}</small>
                  {Number(reidDiagnostics.inference_attempts || 0) || Number(reidDiagnostics.reid_avoided_geometry_matches || 0) ? (
                    <div className="tracking-comparison-shared">
                      <span>Appearance checks <strong>{Number(reidDiagnostics.inference_attempts || 0)}</strong></span>
                      <span>Checks avoided <strong>{Number(reidDiagnostics.reid_avoided_geometry_matches || 0)}</strong></span>
                      {Object.entries(reidAttemptReasons).map(([reason, count]) => <span key={reason}>{String(reason).replaceAll("_", " ")} <strong>{count}</strong></span>)}
                    </div>
                  ) : null}
                </div>
              ) : null}
              {trackingComparisonError ? <div className="tracking-comparison-error">{trackingComparisonError}</div> : null}
              {trackingComparison ? (
                <div className="tracking-comparison-panel" ref={comparisonPanelRef}>
                  <div className="tracking-comparison-head">
                    <div><span className="muted">Same-frame comparison</span><strong>{trackingComparison.frames_processed} frames · {Number(trackingComparison.duration_seconds || 0).toFixed(1)}s · {trackingComparison.sample_fps} FPS · {(Number(trackingComparison.elapsed_ms || 0) / 1000).toFixed(1)}s analysis</strong></div>
                    <small>Detection and appearance extraction are shared by both engines. Extra track IDs are a comparison signal, not ground-truth identity accuracy.</small>
                  </div>
                  <div className="tracking-comparison-shared"><span>Recording decode <strong>{trackingComparison.average_frame_decode_ms} ms/frame</strong></span><span>OpenVINO detection <strong>{trackingComparison.average_detection_ms_per_frame} ms/frame</strong></span>{Number(trackingComparison.appearance_ms || 0) > 0 ? <span>Appearance extraction <strong>{trackingComparison.average_appearance_ms_per_frame} ms/frame</strong></span> : null}{trackingComparison.appearance_failures ? <span>Appearance failures <strong>{trackingComparison.appearance_failures}</strong></span> : null}<span>Clip preparation <strong>{(Number(trackingComparison.clip_preparation_ms || 0) / 1000).toFixed(1)}s</strong></span></div>
                  <div className="tracking-comparison-grid">
                    {["survng_hybrid", "ultralytics_fasttrack"].map((implementation) => {
                      const engine = trackingComparison.engines?.[implementation];
                      if (!engine) return null;
                      const comparisonEvent = {
                        ...viewerEvent, object_tracking: {
                          ...engine,
                          sample_fps: trackingComparison.sample_fps,
                          frame_width: trackingComparison.frame_width,
                          frame_height: trackingComparison.frame_height,
                        }
                      };
                      return (
                        <article className={`tracking-comparison-card ${trackingComparisonEngine === implementation ? "active" : ""}`} key={implementation}>
                          <header><strong>{implementation === "survng_hybrid" ? "SurvNG Hybrid" : "FastTrack"}</strong><span>{engine.average_ms_per_frame} ms/frame · {engine.initialization_ms} ms init</span></header>
                          <SnapshotImage event={comparisonEvent} alt={`${implementation} tracking result`} allowObjectFocus={false} showAnnotations={false} showTracking />
                          <dl>
                            <div><dt>Tracks</dt><dd>{engine.track_count}</dd></div>
                            <div><dt>Extra track IDs</dt><dd>{engine.fragmentation_proxy}</dd></div>
                            <div><dt>Observations</dt><dd>{engine.observations}</dd></div>
                            <div><dt>ReID recoveries</dt><dd>{engine.reid_recoveries}</dd></div>
                            <div><dt>Geometry matches</dt><dd>{engine.reid_diagnostics?.association_counts?.geometry || 0}</dd></div>
                          </dl>
                          <button type="button" className="tile-control-button" onClick={() => replayTrackingComparison(implementation)}><Play size={15} /> Replay this result</button>
                        </article>
                      );
                    })}
                  </div>
                  <div className="tracking-comparison-verdict">
                    <div><span className="muted">Your visual review</span><strong>Which replay kept identities most accurately?</strong></div>
                    <div className="tracking-comparison-verdict-actions">
                      {[
                        ["survng_hybrid", "Hybrid looked better"],
                        ["ultralytics_fasttrack", "FastTrack looked better"],
                        ["inconclusive", "No clear winner"],
                      ].map(([verdict, label]) => (
                        <button type="button" className={`tile-control-button ${trackingComparison.verdict === verdict ? "active" : ""}`} disabled={trackingVerdictLoading} onClick={() => saveTrackingVerdict(verdict)} key={verdict}>{label}</button>
                      ))}
                    </div>
                  </div>
                </div>
              ) : null}
              {trackingComparisonHistory.summary?.total ? (
                <div className="tracking-comparison-history">
                  <div className="tracking-comparison-head">
                    <div><span className="muted">{event.camera_id} comparison evidence</span><strong>{trackingComparisonHistory.summary.reviewed} reviewed · {trackingComparisonHistory.summary.verdicts?.unreviewed || 0} awaiting review</strong></div>
                    <small>SurvNG records your judgment but never changes the configured tracker automatically.</small>
                  </div>
                  <div className="tracking-comparison-shared">
                    <span>Hybrid better <strong>{trackingComparisonHistory.summary.verdicts?.survng_hybrid || 0}</strong></span>
                    <span>FastTrack better <strong>{trackingComparisonHistory.summary.verdicts?.ultralytics_fasttrack || 0}</strong></span>
                    {trackingComparisonHistory.summary.verdicts?.ultralytics_botsort ? <span>BoT-SORT better (historic) <strong>{trackingComparisonHistory.summary.verdicts.ultralytics_botsort}</strong></span> : null}
                    <span>No clear winner <strong>{trackingComparisonHistory.summary.verdicts?.inconclusive || 0}</strong></span>
                  </div>
                  {trackingComparisonHistory.items.some((item) => item.verdict) ? (
                    <div className="tracking-comparison-history-list">
                      {trackingComparisonHistory.items.filter((item) => item.verdict).slice(0, 5).map((item) => (
                        <div key={item.id}>
                          <time>{formatDateTime(item.event_created_at || item.created_at, timeZone)}</time>
                          <strong>{item.verdict === "survng_hybrid" ? "Hybrid" : item.verdict === "ultralytics_fasttrack" ? "FastTrack" : item.verdict === "ultralytics_deepocsort" ? "Deep OC-SORT (historic)" : item.verdict === "ultralytics_botsort" ? "BoT-SORT (historic)" : "No clear winner"}</strong>
                          <span>{item.result?.frames_processed || 0} frames</span>
                        </div>
                      ))}
                    </div>
                  ) : null}
                </div>
              ) : null}
              <div className="manual-detect-panel" onClick={(event) => event.stopPropagation()}>
                <div className="manual-detect-head">
                  <div>
                    <span className="muted">Manual OpenVINO</span>
                    <strong>{Math.round(safeManualConfidence * 100)}% confidence</strong>
                  </div>
                  <button type="button" className="tile-control-button" onClick={runManualDetection} disabled={manualLoading || !Number.isFinite(manualEventId)}>
                    <Search size={15} /> {manualLoading ? "Running" : "Run"}
                  </button>
                </div>
                <input
                  type="range"
                  min="0.05"
                  max="0.95"
                  step="0.01"
                  value={safeManualConfidence}
                  onChange={(event) => setManualConfidence(event.target.value)}
                  aria-label="Manual detection confidence"
                />
                {manualError ? <span className="manual-debug error">{manualError}</span> : null}
                {manualDetection ? (
                  <div className="manual-debug">
                    <span>{manualDetection.object_count} objects</span>
                    <span>{manualDetection.elapsed_ms} ms</span>
                    <span>{manualDetection.detector?.loaded_backend || "detector"}</span>
                    <span>{manualDetection.detector?.loaded_device || manualDetection.detector?.configured_device || "device"}</span>
                    <span>{manualDetection.snapshot_width}x{manualDetection.snapshot_height}</span>
                    {manualDetection.labels?.length ? <code>{manualDetection.labels.join(", ")}</code> : <code>no labels</code>}
                  </div>
                ) : null}
              </div>
            </div>
          </details>
        </div>
      </section>
    </div>
  ), document.body);
}

export async function eventStreamTimelineStart(streamUrl, requestedWindowStartEpoch) {
  try {
    const response = await fetch(streamUrl);
    if (!response.ok) return requestedWindowStartEpoch;
    return hlsProgramStartEpoch(await response.text()) ?? requestedWindowStartEpoch;
  } catch {
    return requestedWindowStartEpoch;
  }
}

export async function loadIncidentClipInfo(event, isCancelled = () => false, preferNativeMp4 = false) {
  const eventId = Number(event?.representative_event_id || event?.id);
  if (!Number.isFinite(eventId)) return null;
  let before = 5;
  let after = 5;
  try {
    const response = await fetch("/api/event-clip/settings");
    if (response.ok) {
      const settings = await response.json();
      before = Number(settings.before_seconds ?? before);
      after = Number(settings.after_seconds ?? after);
    }
  } catch {
    // Defaults keep incident playback useful if settings are temporarily unavailable.
  }
  if (isCancelled()) return null;
  const safeBefore = Number.isFinite(before) ? before : 5;
  const safeAfter = Number.isFinite(after) ? after : 5;
  const window = incidentClipWindow(event, safeBefore, safeAfter);
  const anchorEpoch = eventEpoch(event);
  const requestedWindowStartEpoch = Number.isFinite(anchorEpoch) ? anchorEpoch - window.before : null;
  const streamUrl = eventStreamUrl(eventId, window.before, window.after);
  const timelineStartEpoch = !preferNativeMp4 && Number.isFinite(requestedWindowStartEpoch)
    ? await eventStreamTimelineStart(streamUrl, requestedWindowStartEpoch)
    : requestedWindowStartEpoch;
  if (isCancelled()) return null;
  const initialPlaybackOffset = Math.max(0, window.before - safeBefore);
  return {
    streamUrl,
    downloadUrl: eventClipUrl(eventId, window.before, window.after),
    before: window.before,
    after: window.after,
    duration: window.before + window.after,
    windowStartEpoch: timelineStartEpoch,
    requestedWindowStartEpoch,
    initialPlaybackOffset,
    playbackStartOffset: hlsPlaybackOffset(requestedWindowStartEpoch, timelineStartEpoch, initialPlaybackOffset),
  };
}
