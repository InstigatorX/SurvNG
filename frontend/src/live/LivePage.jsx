import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  Activity,
  ArrowLeft,
  ArrowRight,
  Camera,
  ChevronLeft,
  ChevronRight,
  Grid2X2,
  GripVertical,
  Maximize2,
  Power,
  Radar,
  Radio,
  RefreshCcw,
  RotateCcw,
  SlidersHorizontal,
  Video,
  X,
} from "lucide-react";
import { aspectFromDimensions, cameraSourceAspect, initialCameraAspect, liveAspectStorageKey, normalizedLiveSource, validLiveAspect } from "../liveAspect.mjs";
import { resetLiveDefaultsForServer } from "../liveDefaults.mjs";
import { browserStorage, writeStoredValue } from "../storage.mjs";
import { liveMediaShouldRun, liveSnapshotRefreshMs } from "../pollingPolicy.mjs";
import { useVisiblePolling } from "../visibilityPolling.mjs";
import { liveCustomDropTarget, liveCustomGridMetrics, liveCustomTilePlacement, moveLiveCamera, readLiveCustomLayout, resizeLiveCamera, resizeLiveCameraToAspect } from "../liveCustomLayout.mjs";
import { focusedLiveCameraId, LIVE_DENSITY_OPTIONS, liveActivityQuickFilter, liveActivityQuickSelection, liveDensityPage, normalizedLiveDensity, orderedLiveCamerasForFocus, uniformLiveGridLayout } from "../liveWorkspace.mjs";
import { liveFramingStyle } from "../liveFraming.mjs";
import { createIncidentPageCache, incidentDetailQuery, incidentThumbnailPageSize, incidentsNewestFirst, retainFocusedIncident } from "../incidentNavigation.mjs";
import { appUrl, incidentRecordingContext, fetch } from "../shared/api.js";
import { INCIDENT_REFRESH_FALLBACK_MS, STREAM_MODES, STREAM_LABELS, MOTION_WEBRTC_HOLD_MS } from "../shared/constants.js";
import { formatTimeOnly } from "../shared/format.js";
import { useStoredState, useViewportQuery, useModalFocus } from "../shared/hooks.js";
import { clearLegacyIncidentFilterStorage, preferredStreamSource, sourceLabel, liveTransportLabel } from "../shared/cameras.js";
import { useAppEvents } from "../shared/events.js";
import { WebRtcLive } from "../shared/media.jsx";
import { MobileCameraSelect } from "../shared/MobileCameraSelect.jsx";
import { usePollingData, useIncidentDetails } from "../shared/polling.js";
import { IncidentListItem, EventOverlay } from "../shared/evidence.jsx";
import {
  LIVE_CAMERA_HOLD_PREVIEW_MS,
  LIVE_CAMERA_OVERLAY_MOTION_MS,
  liveCameraHoldExceededMove,
  shouldArmLiveCameraHoldPreview,
  shouldSuppressLiveCameraOpenClick,
} from "../liveCameraHoldPreview.mjs";

export function mediaAspect(element) {
  const width = element?.videoWidth || element?.naturalWidth || 0;
  const height = element?.videoHeight || element?.naturalHeight || 0;
  return aspectFromDimensions(width, height) || "16 / 9";
}

export function mediaAspectRatio(aspect) {
  const [width, height] = String(aspect || "").split("/").map((value) => Number(value.trim()));
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return 16 / 9;
  }
  return width / height;
}

export function CameraTile({ camera, timeZone, refresh, onOpen, onPreviewOpen, onPreviewClose, onAspectChange, layout, customLayout = false, customStyle, resizeHandleProps = {}, startDelayMs = 0, dragHandleProps = {}, resizing = false, aspectSnapped = false, mobileView = false, mobilePrimary = false }) {
  const tileRef = useRef(null);
  const controlMenuButtonRef = useRef(null);
  const hoverTimerRef = useRef(null);
  const holdPreviewRef = useRef(null);
  const suppressOpenClickRef = useRef(false);
  const openTargetRef = useRef(null);
  const tileWasVisibleRef = useRef(true);
  const [tileVisible, setTileVisible] = useState(true);
  const [documentVisible, setDocumentVisible] = useState(() => !document.hidden);
  const [mediaActive, setMediaActive] = useState(true);
  const [streamMode, setStreamMode] = useStoredState(`survng.streamMode.v3.${camera.id}`, "motion");
  const [sourceMode, setSourceMode] = useStoredState(`survng.sourceMode.${camera.id}`, "live");
  const [motionWindowNow, setMotionWindowNow] = useState(() => Date.now());
  const normalizedStreamMode = STREAM_MODES.includes(streamMode) ? streamMode : "motion";
  const lastMotionMs = new Date(camera.last_motion_at || 0).getTime();
  const motionActive = camera.running && Number.isFinite(lastMotionMs) && motionWindowNow - lastMotionMs <= MOTION_WEBRTC_HOLD_MS;
  const activeTransport = normalizedStreamMode === "motion" ? (motionActive ? "webrtc" : "snapshot") : normalizedStreamMode;
  const [deliveredSource, setDeliveredSource] = useState(sourceMode === "main" ? "main" : "live");
  const [aspect, setAspect] = useState(() => initialCameraAspect(camera, sourceMode, browserStorage(window)));
  const [mjpegToken, setMjpegToken] = useState(() => String(Date.now()));
  const [snapshotToken, setSnapshotToken] = useState(() => String(Date.now()));
  const [streamReady, setStreamReady] = useState(false);
  const [recordingBusy, setRecordingBusy] = useState(false);
  const [recordingError, setRecordingError] = useState("");
  const [detectionBusy, setDetectionBusy] = useState(false);
  const [detectionError, setDetectionError] = useState("");
  const [cameraActionBusy, setCameraActionBusy] = useState(false);
  const [cameraActionError, setCameraActionError] = useState("");
  const [controlMenuOpen, setControlMenuOpen] = useState(false);
  const [hoverPreview, setHoverPreview] = useState(false);
  const [holdPreviewState, setHoldPreviewState] = useState("");
  const [tileLiveReady, setTileLiveReady] = useState(false);
  const displayedTransport = hoverPreview ? "webrtc" : activeTransport;
  const shouldUseLiveMedia = liveMediaShouldRun({ running: camera.running, streamReady: hoverPreview || streamReady, mediaActive, transport: displayedTransport });
  const shouldUseWebRtc = shouldUseLiveMedia && displayedTransport === "webrtc";
  const shouldUseMjpegStream = shouldUseLiveMedia && activeTransport === "mjpeg";
  const cameraConnected = camera.connected ?? camera.running;

  useEffect(() => {
    onAspectChange?.(camera.id, mediaAspectRatio(aspect));
  }, [aspect, camera.id, onAspectChange]);

  useEffect(() => () => {
    window.clearTimeout(hoverTimerRef.current);
    window.clearTimeout(holdPreviewRef.current?.timer);
  }, []);

  useEffect(() => {
    if (!shouldUseWebRtc) setTileLiveReady(false);
  }, [shouldUseWebRtc]);

  function clearHoldPreviewArm() {
    const hold = holdPreviewRef.current;
    if (!hold) return null;
    window.clearTimeout(hold.timer);
    if (hold.selectionTimer) window.clearInterval(hold.selectionTimer);
    holdPreviewRef.current = null;
    setHoldPreviewState("");
    return hold;
  }

  function suppressBrowserHoldUi() {
    const selection = window.getSelection?.();
    selection?.removeAllRanges?.();
    try {
      document.activeElement?.blur?.();
    } catch {
      // Ignore hosts that reject blur during gesture handling.
    }
  }

  function armHoldPreview({ pointerId, startX, startY, currentTarget = null, capturePointer = false } = {}) {
    clearHoldPreviewArm();
    suppressBrowserHoldUi();
    const selectionTimer = window.setInterval(suppressBrowserHoldUi, 50);
    const timer = window.setTimeout(() => {
      const current = holdPreviewRef.current;
      if (!current || current.pointerId !== pointerId) return;
      current.opened = true;
      suppressBrowserHoldUi();
      setHoldPreviewState("active");
      onPreviewOpen?.(camera);
    }, LIVE_CAMERA_HOLD_PREVIEW_MS);
    holdPreviewRef.current = {
      pointerId,
      startX,
      startY,
      opened: false,
      timer,
      selectionTimer,
      fromTouch: !capturePointer,
    };
    setHoldPreviewState("pending");
    if (!capturePointer || !currentTarget) return;
    try {
      currentTarget.setPointerCapture(pointerId);
    } catch {
      // Some browsers reject capture during scrolling; window listeners still close preview.
    }
  }

  function beginHoldPreview(event) {
    if (!shouldArmLiveCameraHoldPreview({ mobileView, pointerType: event.pointerType })) return;
    if (typeof event.button === "number" && event.button !== 0) return;
    // Touch gestures are owned by non-passive touch listeners so iOS callout can be canceled.
    if (event.pointerType === "touch") return;
    armHoldPreview({
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      currentTarget: event.currentTarget,
      capturePointer: true,
    });
  }

  function blockBrowserHoldMenu(event) {
    event.preventDefault();
    suppressBrowserHoldUi();
  }

  function onOpenTargetTouchStart(event) {
    if (!mobileView || event.touches.length !== 1) return;
    // Required to stop iOS hard-press cut/paste / callout on the tile surface.
    event.preventDefault();
    const touch = event.touches[0];
    armHoldPreview({
      pointerId: touch.identifier,
      startX: touch.clientX,
      startY: touch.clientY,
    });
  }

  function onOpenTargetTouchMove(event) {
    const hold = holdPreviewRef.current;
    if (!hold || hold.opened || !event.touches.length) return;
    const touch = event.touches[0];
    if (hold.pointerId !== touch.identifier) return;
    if (!liveCameraHoldExceededMove(hold.startX, hold.startY, touch.clientX, touch.clientY)) return;
    clearHoldPreviewArm();
  }

  function onOpenTargetTouchEnd(event) {
    const hold = holdPreviewRef.current;
    if (!hold) return;
    event.preventDefault();
    const opened = hold.opened;
    clearHoldPreviewArm();
    if (opened) {
      suppressOpenClickRef.current = true;
      onPreviewClose?.();
      return;
    }
    // touchstart preventDefault suppresses the synthetic click; open sticky on tap here.
    onOpen(camera);
  }

  function bindOpenTarget(node) {
    if (openTargetRef.current && openTargetRef.current !== node) {
      const previous = openTargetRef.current;
      previous.removeEventListener("selectstart", blockBrowserHoldMenu);
      previous.removeEventListener("dragstart", blockBrowserHoldMenu);
      previous.removeEventListener("touchstart", onOpenTargetTouchStart);
      previous.removeEventListener("touchmove", onOpenTargetTouchMove);
      previous.removeEventListener("touchend", onOpenTargetTouchEnd);
      previous.removeEventListener("touchcancel", onOpenTargetTouchEnd);
    }
    openTargetRef.current = node;
    if (!node) return;
    node.addEventListener("selectstart", blockBrowserHoldMenu);
    node.addEventListener("dragstart", blockBrowserHoldMenu);
    node.addEventListener("touchstart", onOpenTargetTouchStart, { passive: false });
    node.addEventListener("touchmove", onOpenTargetTouchMove, { passive: true });
    node.addEventListener("touchend", onOpenTargetTouchEnd, { passive: false });
    node.addEventListener("touchcancel", onOpenTargetTouchEnd, { passive: false });
  }

  useEffect(() => () => {
    const node = openTargetRef.current;
    if (!node) return;
    node.removeEventListener("selectstart", blockBrowserHoldMenu);
    node.removeEventListener("dragstart", blockBrowserHoldMenu);
    node.removeEventListener("touchstart", onOpenTargetTouchStart);
    node.removeEventListener("touchmove", onOpenTargetTouchMove);
    node.removeEventListener("touchend", onOpenTargetTouchEnd);
    node.removeEventListener("touchcancel", onOpenTargetTouchEnd);
  }, []);

  useEffect(() => {
    if (!holdPreviewState) return undefined;
    function blockDuringHold(event) {
      event.preventDefault();
      suppressBrowserHoldUi();
    }
    document.addEventListener("selectstart", blockDuringHold, true);
    document.addEventListener("selectionchange", suppressBrowserHoldUi, true);
    document.addEventListener("contextmenu", blockDuringHold, true);
    document.addEventListener("gesturestart", blockDuringHold, true);
    return () => {
      document.removeEventListener("selectstart", blockDuringHold, true);
      document.removeEventListener("selectionchange", suppressBrowserHoldUi, true);
      document.removeEventListener("contextmenu", blockDuringHold, true);
      document.removeEventListener("gesturestart", blockDuringHold, true);
    };
  }, [holdPreviewState]);

  function moveHoldPreview(event) {
    const hold = holdPreviewRef.current;
    if (!hold || hold.pointerId !== event.pointerId || hold.opened || hold.fromTouch) return;
    if (!liveCameraHoldExceededMove(hold.startX, hold.startY, event.clientX, event.clientY)) return;
    clearHoldPreviewArm();
    try {
      event.currentTarget.releasePointerCapture(event.pointerId);
    } catch {
      // Capture may already be released when the gesture turned into a scroll.
    }
  }

  function endHoldPreview(event) {
    const hold = holdPreviewRef.current;
    if (!hold || hold.pointerId !== event.pointerId || hold.fromTouch) return;
    const opened = hold.opened;
    clearHoldPreviewArm();
    if (!opened) return;
    suppressOpenClickRef.current = true;
    onPreviewClose?.();
  }

  function openCameraFromTarget(event) {
    if (shouldSuppressLiveCameraOpenClick({
      holdOpened: Boolean(holdPreviewRef.current?.opened),
      suppressClick: suppressOpenClickRef.current,
    })) {
      suppressOpenClickRef.current = false;
      event.preventDefault();
      return;
    }
    onOpen(camera);
  }

  useEffect(() => {
    if (!mobileView) return undefined;
    function onGlobalPointerEnd(event) {
      const hold = holdPreviewRef.current;
      if (!hold || hold.fromTouch || hold.pointerId !== event.pointerId) return;
      const opened = hold.opened;
      clearHoldPreviewArm();
      if (!opened) return;
      suppressOpenClickRef.current = true;
      onPreviewClose?.();
    }
    window.addEventListener("pointerup", onGlobalPointerEnd);
    window.addEventListener("pointercancel", onGlobalPointerEnd);
    return () => {
      window.removeEventListener("pointerup", onGlobalPointerEnd);
      window.removeEventListener("pointercancel", onGlobalPointerEnd);
    };
  }, [camera, mobileView, onPreviewClose]);

  useEffect(() => {
    if (!controlMenuOpen) return undefined;
    function closeMenu(event) {
      if (event.type === "keydown" && event.key !== "Escape") return;
      if (event.type === "pointerdown" && tileRef.current?.contains(event.target)) return;
      setControlMenuOpen(false);
      if (event.type === "keydown") window.requestAnimationFrame(() => controlMenuButtonRef.current?.focus());
    }
    window.addEventListener("keydown", closeMenu);
    window.addEventListener("pointerdown", closeMenu);
    return () => {
      window.removeEventListener("keydown", closeMenu);
      window.removeEventListener("pointerdown", closeMenu);
    };
  }, [controlMenuOpen]);

  function beginHoverPreview(event) {
    if (event.pointerType && event.pointerType !== "mouse") return;
    window.clearTimeout(hoverTimerRef.current);
    hoverTimerRef.current = window.setTimeout(() => setHoverPreview(true), 180);
  }

  function endHoverPreview() {
    window.clearTimeout(hoverTimerRef.current);
    setHoverPreview(false);
  }

  useEffect(() => {
    if (!STREAM_MODES.includes(streamMode)) setStreamMode("motion");
  }, [streamMode, setStreamMode]);

  useEffect(() => {
    const now = Date.now();
    setMotionWindowNow(now);
    if (!Number.isFinite(lastMotionMs)) return undefined;
    const remaining = MOTION_WEBRTC_HOLD_MS - (now - lastMotionMs);
    if (remaining <= 0) return undefined;
    const timer = window.setTimeout(() => setMotionWindowNow(Date.now()), remaining + 50);
    return () => window.clearTimeout(timer);
  }, [lastMotionMs]);

  useEffect(() => {
    const tile = tileRef.current;
    if (!tile || typeof IntersectionObserver !== "function") return undefined;
    const observer = new IntersectionObserver(([entry]) => setTileVisible(entry.isIntersecting), { rootMargin: "80px" });
    observer.observe(tile);
    return () => observer.disconnect();
  }, [camera.id]);

  useEffect(() => {
    if (tileVisible && !tileWasVisibleRef.current) setSnapshotToken(String(Date.now()));
    tileWasVisibleRef.current = tileVisible;
  }, [tileVisible]);

  useEffect(() => {
    const onVisibility = () => {
      const visible = !document.hidden;
      setDocumentVisible(visible);
      if (visible && tileVisible) setSnapshotToken(String(Date.now()));
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => document.removeEventListener("visibilitychange", onVisibility);
  }, [tileVisible]);

  useEffect(() => {
    if (tileVisible && documentVisible) {
      setMediaActive(true);
      return undefined;
    }
    const timer = window.setTimeout(() => setMediaActive(false), 1500);
    return () => window.clearTimeout(timer);
  }, [documentVisible, tileVisible]);

  useEffect(() => {
    setMjpegToken(String(Date.now()));
    setSnapshotToken(String(Date.now()));
    setStreamReady(false);
  }, [camera.id, sourceMode, activeTransport]);

  const authoritativeAspect = cameraSourceAspect(camera, deliveredSource);

  useEffect(() => {
    const nextSource = activeTransport === "webrtc" ? deliveredSource : normalizedLiveSource(sourceMode);
    const nextAspect = cameraSourceAspect(camera, nextSource)
      || initialCameraAspect(camera, nextSource, browserStorage(window));
    setAspect(nextAspect);
  }, [camera.id, sourceMode, activeTransport, deliveredSource, authoritativeAspect]);

  function rememberAspect(media, source, activate = true) {
    const normalizedSource = normalizedLiveSource(source);
    const measuredAspect = validLiveAspect(mediaAspect(media));
    if (!measuredAspect) return;
    writeStoredValue(
      browserStorage(window),
      liveAspectStorageKey(camera.id, normalizedSource),
      measuredAspect,
    );
    if (activate && !cameraSourceAspect(camera, normalizedSource)) setAspect(measuredAspect);
  }

  const handleTileStageChange = React.useCallback((_stage, nextSource) => {
    setTileLiveReady(false);
    setDeliveredSource(normalizedLiveSource(nextSource));
  }, []);

  useEffect(() => {
    setStreamReady(false);
    setSnapshotToken(String(Date.now()));
    if (!camera.running) return undefined;
    const timer = window.setTimeout(() => setStreamReady(true), startDelayMs);
    return () => window.clearTimeout(timer);
  }, [camera.id, camera.running, sourceMode, activeTransport, startDelayMs]);

  useEffect(() => {
    const refreshMs = liveSnapshotRefreshMs({
      running: camera.running,
      visible: tileVisible,
      documentVisible,
      streamReady,
      transport: activeTransport,
      mobile: mobileView,
      primary: mobilePrimary,
    });
    if (!refreshMs) return undefined;
    const timer = window.setInterval(
      () => setSnapshotToken(String(Date.now())),
      refreshMs,
    );
    return () => window.clearInterval(timer);
  }, [activeTransport, camera.running, documentVisible, mobilePrimary, mobileView, streamReady, tileVisible]);

  async function post(action) {
    if (cameraActionBusy) return;
    setCameraActionBusy(true);
    setCameraActionError("");
    try {
      const response = await fetch(`/api/cameras/${camera.id}/${action}`, { method: "POST" });
      if (!response.ok) throw new Error(`Camera control failed (${response.status})`);
      await refresh();
    } catch (error) {
      setCameraActionError(error instanceof Error ? error.message : "Camera control failed");
    } finally {
      setCameraActionBusy(false);
    }
  }

  async function toggleRecording() {
    if (recordingBusy || !camera.recording_configured) return;
    setRecordingBusy(true);
    setRecordingError("");
    try {
      const response = await fetch(`/api/cameras/${camera.id}/recording`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: !camera.recording_enabled }),
      });
      if (!response.ok) throw new Error(`Recording control failed (${response.status})`);
      await refresh();
    } catch (error) {
      setRecordingError(error instanceof Error ? error.message : "Recording control failed");
    } finally {
      setRecordingBusy(false);
    }
  }

  async function toggleDetection() {
    if (detectionBusy) return;
    setDetectionBusy(true);
    setDetectionError("");
    try {
      const response = await fetch(`/api/cameras/${camera.id}/detection`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: !camera.detection_enabled }),
      });
      if (!response.ok) throw new Error(`Detection control failed (${response.status})`);
      await refresh();
    } catch (error) {
      setDetectionError(error instanceof Error ? error.message : "Detection control failed");
    } finally {
      setDetectionBusy(false);
    }
  }

  function cycleStreamMode() {
    const index = STREAM_MODES.indexOf(normalizedStreamMode);
    setStreamMode(STREAM_MODES[(index + 1) % STREAM_MODES.length]);
  }

  function toggleSourceMode() {
    const nextSource = sourceMode === "main" ? "live" : "main";
    setDeliveredSource(nextSource);
    setAspect(initialCameraAspect(camera, nextSource, browserStorage(window)));
    setSourceMode(nextSource);
  }

  const posterSource = activeTransport === "webrtc" && sourceMode === "main" ? "live" : sourceMode;
  const imageUrl = appUrl(shouldUseMjpegStream
    ? `/api/cameras/${camera.id}/stream.mjpg?source=${sourceMode}&t=${mjpegToken}`
    : `/api/cameras/${camera.id}/snapshot.jpg?source=${posterSource}&t=${snapshotToken}`);

  return (
    <article
      ref={tileRef}
      className={`bento-card camera-tile ${layout ? "viewport-layout" : ""} ${customLayout ? "custom-layout-tile" : ""} ${motionActive ? "motion-active" : ""} ${resizing ? "resizing" : ""} ${aspectSnapped ? "aspect-snapped" : ""} ${mobilePrimary ? "mobile-primary" : ""}`}
      data-motion-active={motionActive ? "true" : "false"}
      data-camera-id={camera.id}
      data-hover-preview={hoverPreview ? "true" : "false"}
      data-hold-preview={holdPreviewState || undefined}
      onPointerEnter={beginHoverPreview}
      onPointerLeave={endHoverPreview}
      style={customLayout ? customStyle : layout ? {
        left: `${layout.x}px`,
        top: `${layout.y}px`,
        width: `${layout.width}px`,
        height: `${layout.height}px`,
      } : undefined}
    >
      <div
        className="video-frame"
        style={{ "--media-aspect": aspect }}
      >
        <div className="camera-tile-chrome">
          <span className="camera-tile-name"><i className={cameraConnected ? "online" : "offline"} aria-hidden="true" /><strong>{camera.name}</strong></span>
          <time className="camera-tile-time">{formatTimeOnly(Date.now() / 1000, timeZone)}</time>
          <span className="camera-tile-actions">
            <span className="camera-tile-live-state">{cameraConnected ? "LIVE" : camera.running ? "WAIT" : "OFF"}</span>
            <button
              type="button"
              ref={controlMenuButtonRef}
              className="camera-tile-menu"
              aria-label={`Open ${camera.name} controls`}
              aria-expanded={controlMenuOpen}
              aria-haspopup="true"
              onClick={() => setControlMenuOpen((open) => !open)}
            >⋮</button>
          </span>
        </div>
        {!camera.running ? (
          <div className="camera-offline-state" role="img" aria-label={`${camera.name} is powered off`}>
            <Power size={24} />
          </div>
        ) : (
          <>
            <img
              className="camera-tile-poster"
              style={liveFramingStyle(camera, posterSource)}
              src={imageUrl}
              alt={`${camera.name} ${sourceMode === "main" ? "main" : "sub"} live stream`}
              draggable={false}
              onLoad={(event) => rememberAspect(
                event.currentTarget,
                posterSource,
                activeTransport !== "webrtc" || normalizedLiveSource(posterSource) === normalizedLiveSource(deliveredSource),
              )}
            />
            {shouldUseWebRtc ? (
              <div className={`camera-live-layer${tileLiveReady ? " ready" : ""}`} style={liveFramingStyle(camera, deliveredSource)}>
                <WebRtcLive
                  cameraId={camera.id}
                  source={sourceMode}
                  timeZone={timeZone}
                  muted
                  showPoster={false}
                  onStageChange={handleTileStageChange}
                  onReady={(media, _stage, readySource) => {
                    rememberAspect(media, readySource || deliveredSource);
                    setTileLiveReady(true);
                  }}
                />
              </div>
            ) : null}
          </>
        )}
        <button
          type="button"
          ref={bindOpenTarget}
          className="camera-open-target media-surface-action"
          onClick={openCameraFromTarget}
          onPointerDown={beginHoldPreview}
          onPointerMove={moveHoldPreview}
          onPointerUp={endHoldPreview}
          onPointerCancel={endHoldPreview}
          onContextMenu={blockBrowserHoldMenu}
          aria-label={mobileView ? `Open ${camera.name} live view. Press and hold to preview.` : `Open ${camera.name} live view`}
        />
        <span className="sr-only" aria-live="polite">{motionActive ? `${camera.name} motion active` : ""}</span>
        {controlMenuOpen ? <div className="camera-tile-control-menu" role="group" aria-label={`${camera.name} controls`}>
          <div className="tile-controls">
            {dragHandleProps.onPointerDown ? <button
              type="button"
              className="tile-control-button camera-drag-handle"
              title="Drag to move camera"
              aria-label={`Move ${camera.name}`}
              {...dragHandleProps}
            >
              <GripVertical size={16} />
              <span>Move camera</span>
            </button> : null}
            <button type="button" className="tile-control-button" onClick={toggleSourceMode} title="Switch main/sub stream">
              <Radio size={15} />
              <span>Stream: {sourceMode === "main" ? "Main" : "Sub"}</span>
            </button>
            <button
              type="button"
              className="tile-control-button"
              onClick={cycleStreamMode}
              title={normalizedStreamMode === "motion" ? `Automatic motion switching: ${activeTransport === "webrtc" ? "WebRTC active" : "snapshot idle"}` : "Cycle transport: Auto, MJPEG, WebRTC"}
            >
              <span>
                {normalizedStreamMode === "motion" ? `Auto ${activeTransport === "webrtc" ? "RTC" : "Snap"}` : STREAM_LABELS[normalizedStreamMode]}
              </span>
            </button>
            <button
              type="button"
              className={`status-pill hud-toggle hud-icon ${camera.recording_enabled ? "ok" : ""} ${recordingError ? "bad" : ""}`}
              onClick={toggleRecording}
              disabled={recordingBusy || !camera.recording_configured}
              title={recordingError || (!camera.recording_configured ? "Recording is not configured for this camera" : camera.recording_enabled ? "Stop recording" : "Start recording")}
              aria-label={`${camera.recording_enabled ? "Stop" : "Start"} recording ${camera.name}`}
            >
              <Video size={13} /> <span>{recordingBusy ? "Updating recording…" : camera.recording_enabled ? "Stop recording" : "Start recording"}</span>
            </button>
            <button
              type="button"
              className={`status-pill hud-toggle hud-icon ${camera.detection_enabled ? "ok" : ""} ${detectionError ? "bad" : ""}`}
              onClick={toggleDetection}
              disabled={detectionBusy}
              title={detectionError || (camera.detection_enabled ? "Stop motion and object detection" : "Start motion and object detection")}
              aria-label={`${camera.detection_enabled ? "Stop" : "Start"} motion and object detection for ${camera.name}`}
            >
              <Radar size={13} /> <span>{detectionBusy ? "Updating detection…" : camera.detection_enabled ? "Stop detection" : "Start detection"}</span>
            </button>
            <button
              type="button"
              className={`tile-control-button ${camera.running ? "danger" : ""} ${cameraActionError ? "bad" : ""}`}
              onClick={() => post(camera.running ? "camera/stop" : "camera/start")}
              disabled={cameraActionBusy}
              title={cameraActionError || (camera.running ? "Stop camera" : "Start camera")}
              aria-label={`${camera.running ? "Stop" : "Start"} ${camera.name}`}
            >
              {cameraActionBusy ? <RefreshCcw className="spin" size={16} /> : <Power size={16} />}
              <span>{cameraActionBusy ? "Updating camera…" : camera.running ? "Stop camera" : "Start camera"}</span>
            </button>
          </div>
        </div> : null}
      </div>
      {customLayout ? <button
        type="button"
        className="camera-resize-handle"
        title="Drag to resize camera"
        aria-label={`Resize ${camera.name}`}
        {...resizeHandleProps}
      /> : null}
      {resizing ? <span className="camera-resize-hint">{aspectSnapped ? "Fit video" : "Free size"}</span> : null}
    </article>
  );
}

export function LiveCameraOverlay({ camera, timeZone, onClose, onClosed, mode = "sticky", phase = "open" }) {
  const preview = mode === "preview";
  const modalRef = useModalFocus(onClose, { trapFocus: !preview });
  const [source, setSource] = useStoredState(
    `survng.liveOverlaySource.${camera.id}`,
    preferredStreamSource(),
  );
  const [mediaReady, setMediaReady] = useState(false);
  const [transport, setTransport] = useState("webrtc");
  const activeSource = source === "main" ? "main" : "live";
  const [deliveredSource, setDeliveredSource] = useState(activeSource);
  const [aspect, setAspect] = useState(() => initialCameraAspect(camera, activeSource, browserStorage(window)));
  const closeFinishedRef = useRef(false);

  useEffect(() => {
    setMediaReady(false);
    setTransport("webrtc");
    setDeliveredSource(activeSource);
    setAspect(initialCameraAspect(camera, activeSource, browserStorage(window)));
  }, [camera.id, activeSource]);

  const authoritativeOverlayAspect = cameraSourceAspect(camera, deliveredSource);

  useEffect(() => {
    if (authoritativeOverlayAspect) setAspect(authoritativeOverlayAspect);
  }, [authoritativeOverlayAspect]);

  useEffect(() => {
    closeFinishedRef.current = false;
  }, [camera.id, mode]);

  useEffect(() => {
    if (phase !== "closing") return undefined;
    const timer = window.setTimeout(() => {
      if (closeFinishedRef.current) return;
      closeFinishedRef.current = true;
      onClosed?.();
    }, LIVE_CAMERA_OVERLAY_MOTION_MS);
    return () => window.clearTimeout(timer);
  }, [onClosed, phase]);

  function rememberAspect(media, sourceName = activeSource) {
    const nextAspect = mediaAspect(media);
    if (!cameraSourceAspect(camera, sourceName)) setAspect(nextAspect);
    writeStoredValue(
      browserStorage(window),
      liveAspectStorageKey(camera.id, sourceName),
      nextAspect,
    );
  }

  function handleMotionEnd(event) {
    if (phase !== "closing") return;
    if (!String(event.animationName || "").includes("live-overlay-panel-out")) return;
    if (closeFinishedRef.current) return;
    closeFinishedRef.current = true;
    onClosed?.();
  }

  return createPortal((
    <div
      ref={modalRef}
      className="live-overlay"
      data-mode={mode}
      data-phase={phase}
      role="dialog"
      aria-modal={preview ? undefined : "true"}
      aria-label={preview ? `${camera.name} live preview` : `${camera.name} full live view`}
      onAnimationEnd={handleMotionEnd}
    >
      <button className="live-overlay-backdrop" type="button" onClick={onClose} aria-label="Close live view" tabIndex={preview ? -1 : undefined} />
      <section
        className="live-overlay-panel"
        style={{
          "--media-aspect": aspect,
          "--media-ratio": mediaAspectRatio(aspect),
        }}
      >
        <div className="live-overlay-head">
          <div>
            <h2>{camera.name}</h2>
            <div className="live-overlay-subtitle">
              <span>{sourceLabel(activeSource)} stream</span>
              {deliveredSource !== activeSource ? <span>using {sourceLabel(deliveredSource)} fallback</span> : null}
              <span className="live-transport-badge" aria-label={`Stream transport ${liveTransportLabel(transport)}`}>
                {liveTransportLabel(transport)}
              </span>
              {preview ? <span className="live-preview-hint">Release to close</span> : null}
            </div>
          </div>
          {preview ? null : (
            <button type="button" className="tile-control-button" onClick={() => setSource(activeSource === "main" ? "live" : "main")} aria-label="Switch live stream">
              <Radio size={15} /> {sourceLabel(activeSource)}
            </button>
          )}
          {preview ? null : (
            <button type="button" className="tile-control-button icon-only" data-modal-initial onClick={onClose} aria-label="Close live view">
              <X size={18} />
            </button>
          )}
        </div>
        <div className="live-overlay-media" style={liveFramingStyle(camera, deliveredSource)}>
          {!mediaReady ? (
            <div className="live-media-status" role="status" aria-live="polite">
              <RefreshCcw className="spin" size={20} />
              <span>Connecting live stream...</span>
            </div>
          ) : null}
          <WebRtcLive
            cameraId={camera.id}
            source={activeSource}
            timeZone={timeZone}
            muted
            controls={!preview}
            onStageChange={(nextTransport, nextSource) => {
              setTransport(nextTransport);
              setDeliveredSource(nextSource);
              if (nextSource !== deliveredSource) setMediaReady(false);
            }}
            onReady={(media, readyTransport, readySource) => {
              rememberAspect(media, readySource || deliveredSource);
              setMediaReady(true);
              if (readyTransport) setTransport(readyTransport);
            }}
          />
        </div>
      </section>
    </div>
  ), document.body);
}

export function LiveCommandBar({ cameras = [], focusedCameraId = "", onFocusedCameraChange = () => {}, cameraCount, totalCameraCount, density, densityPage, densityPageCount, layoutMode, customAvailable, onDensityChange, onDensityPageChange, onLayoutModeChange, onResetLayout, onFullscreen }) {
  return (
    <header className="live-command-bar">
      <div className="live-command-context">
        <span className="live-command-scope"><Grid2X2 size={15} /><strong>All cameras</strong><small>{cameraCount} of {totalCameraCount}</small></span>
        <strong className="live-command-mobile-title">Command Center</strong>
      </div>
      <MobileCameraSelect
        className="live-mobile-camera-select"
        cameras={cameras}
        value={focusedCameraId}
        onChange={onFocusedCameraChange}
        ariaLabel="Live camera"
      />
      <div className="live-density-control" role="group" aria-label="Visible camera density">
        {LIVE_DENSITY_OPTIONS.map((option) => <button type="button" key={option} className={density === option ? "active" : ""} aria-pressed={density === option} onClick={() => onDensityChange(option)}>{option === "fit" ? <Grid2X2 size={15} /> : option === "4" ? <><Grid2X2 size={13} /> 4</> : option}</button>)}
        {densityPageCount > 1 ? <span className="live-density-pages"><button type="button" onClick={() => onDensityPageChange(densityPage - 1)} disabled={densityPage === 0} aria-label="Previous camera page"><ChevronLeft size={15} /></button><small>{densityPage + 1}/{densityPageCount}</small><button type="button" onClick={() => onDensityPageChange(densityPage + 1)} disabled={densityPage >= densityPageCount - 1} aria-label="Next camera page"><ChevronRight size={15} /></button></span> : null}
      </div>
      <div className="live-layout-control" role="group" aria-label="Live camera layout">
        <button type="button" className={layoutMode === "auto" ? "active" : ""} aria-pressed={layoutMode === "auto"} onClick={() => onLayoutModeChange("auto")}><Grid2X2 size={15} /> Automatic</button>
        <button type="button" className={layoutMode === "custom" ? "active" : ""} aria-pressed={layoutMode === "custom"} onClick={() => onLayoutModeChange("custom")} disabled={!customAvailable} title={customAvailable ? "Arrange and resize cameras" : "Custom layout is available on desktop"}><GripVertical size={15} /> Custom</button>
        {layoutMode === "custom" && customAvailable ? <button type="button" className="secondary live-layout-reset" onClick={onResetLayout}><RotateCcw size={14} /> Reset</button> : null}
        <button type="button" className="live-fullscreen" onClick={onFullscreen} aria-label="View Live fullscreen" title="Fullscreen"><Maximize2 size={16} /></button>
      </div>
    </header>
  );
}

export function LivePage({ timeZone, onRecordingContextChange, onAssistantContextChange }) {
  const { cameras, appConfig, refresh: refreshBase } = usePollingData();
  const thumbnailAnnotations = appConfig?.incident_thumbnail_annotations ?? false;
  const [eventFilter, setEventFilter] = useState("object");
  const [incidentCameraFilter, setIncidentCameraFilter] = useState("all");
  const [incidentObjectFilter, setIncidentObjectFilter] = useState("all");
  const [incidentZoneFilter, setIncidentZoneFilter] = useState("all");
  const [cameraOrder] = useStoredState("survng.liveCameraOrder.v1", "[]");
  const [liveLayoutMode, setLiveLayoutMode] = useStoredState("survng.liveLayoutMode.v1", "auto");
  const [liveDensityValue, setLiveDensityValue] = useStoredState("survng.liveDensity.v1", "fit");
  const [liveDensityPageValue, setLiveDensityPageValue] = useState(0);
  const [customLayoutValue, setCustomLayoutValue] = useStoredState("survng.liveCustomLayout.v1", "{}");
  const [storedMobileFocus, setStoredMobileFocus] = useStoredState("survng.liveFocusedCamera.v1", "");
  const customLayoutAvailable = useViewportQuery("(min-width: 1051px)");
  const mobileLiveView = useViewportQuery("(max-width: 760px)");
  const [customSizePreview, setCustomSizePreview] = useState({});
  const [keyboardLayoutPreview, setKeyboardLayoutPreview] = useState(null);
  const [layoutAnnouncement, setLayoutAnnouncement] = useState("");
  const [resizingCameraId, setResizingCameraId] = useState("");
  const customMoveCleanupRef = useRef(null);
  const liveWorkspaceRef = useRef(null);
  const liveCameraGridRef = useRef(null);
  const [liveCameraGridSize, setLiveCameraGridSize] = useState({ width: 0, height: 0 });
  const [liveCameraAspects, setLiveCameraAspects] = useState({});
  const {
    incidentDetailCacheRef,
    incidentDetails,
    setIncidentDetails,
    incidentSelectionRequestRef,
    selectedEvent,
    setSelectedEvent,
    openIncidentOverlay,
    closeIncidentOverlay,
  } = useIncidentDetails();
  const [expandedIncidentId, setExpandedIncidentId] = useState(null);
  const [expandedCamera, setExpandedCamera] = useState(null);
  const [overlayMode, setOverlayMode] = useState("sticky");
  const [overlayPhase, setOverlayPhase] = useState("closed");
  const overlayModeRef = useRef("sticky");
  const ignoreStickyOpenUntilRef = useRef(0);
  const [liveDefaultsReady, setLiveDefaultsReady] = useState(false);
  const [liveDefaultsInstance, setLiveDefaultsInstance] = useState("");
  const linkedCameraIdRef = useRef(new URLSearchParams(window.location.search).get("camera") || "");
  const [incidentPage, setIncidentPage] = useState(0);
  const [incidents, setIncidents] = useState([]);
  const [incidentFacets, setIncidentFacets] = useState({ camera_ids: [], labels: [], zones: [] });
  const [incidentHasMore, setIncidentHasMore] = useState(false);
  const [incidentLoading, setIncidentLoading] = useState(true);
  const [incidentLoadError, setIncidentLoadError] = useState("");
  const [incidentRefreshToken, setIncidentRefreshToken] = useState(0);
  const incidentLoadedQueryRef = useRef("");
  const incidentEventRefreshTimer = useRef(null);
  const incidentFeedCacheRef = useRef(null);
  if (!incidentFeedCacheRef.current) {
    incidentFeedCacheRef.current = createIncidentPageCache(async (query) => {
      const response = await fetch(`/api/incidents/feed?${query}`);
      if (!response.ok) throw new Error("Unable to load recent incidents");
      return response.json();
    });
  }
  const liveIncidentGalleryRef = useRef(null);
  const liveIncidentZoneRef = useRef(null);
  const [liveIncidentGallerySize, setLiveIncidentGallerySize] = useState({ width: 0, height: 0 });
  const liveIncidentGalleryReady = mobileLiveView || (liveIncidentGallerySize.width > 0 && liveIncidentGallerySize.height > 0);
  const incidentsPerPage = mobileLiveView
    ? 5
    : liveIncidentGalleryReady
      ? incidentThumbnailPageSize({ ...liveIncidentGallerySize, density: "compact", columns: 2, gap: 10, horizontalPadding: 24 })
      : 12;
  const orderedCameras = useMemo(() => {
    let order = [];
    try {
      order = JSON.parse(cameraOrder);
    } catch {
      order = [];
    }
    const cameraById = new Map(cameras.map((camera) => [camera.id, camera]));
    const sorted = order.map((id) => cameraById.get(id)).filter(Boolean);
    const seen = new Set(sorted.map((camera) => camera.id));
    return [...sorted, ...cameras.filter((camera) => !seen.has(camera.id))];
  }, [cameras, cameraOrder]);
  const normalizedLayoutMode = liveLayoutMode === "custom" ? "custom" : "auto";
  const effectiveLayoutMode = customLayoutAvailable ? normalizedLayoutMode : "auto";
  const liveDensity = normalizedLiveDensity(liveDensityValue);
  const effectiveLiveDensity = effectiveLayoutMode === "custom" || mobileLiveView ? "fit" : liveDensity;
  const densitySelection = useMemo(
    () => liveDensityPage(orderedCameras, effectiveLiveDensity, liveDensityPageValue),
    [effectiveLiveDensity, liveDensityPageValue, orderedCameras],
  );
  const visibleLiveCameras = densitySelection.cameras;
  const mobileFocusedCameraId = focusedLiveCameraId(orderedCameras, storedMobileFocus);
  const renderedCameras = useMemo(
    () => orderedLiveCamerasForFocus(visibleLiveCameras, mobileFocusedCameraId, mobileLiveView),
    [mobileFocusedCameraId, mobileLiveView, visibleLiveCameras],
  );
  const liveCameraStartIndex = useMemo(() => new Map(orderedCameras.map((camera, index) => [camera.id, index])), [orderedCameras]);
  const customLayout = useMemo(
    () => readLiveCustomLayout(customLayoutValue, cameras, liveCameraAspects),
    [cameras, customLayoutValue, liveCameraAspects],
  );
  const displayedCustomLayout = keyboardLayoutPreview || customLayout;
  const liveCameraLayout = useMemo(
    () => uniformLiveGridLayout(
      visibleLiveCameras,
      liveCameraGridSize.width,
      liveCameraGridSize.height,
      4,
    ),
    [liveCameraGridSize.height, liveCameraGridSize.width, visibleLiveCameras],
  );
  const liveCameraLayoutById = useMemo(
    () => new Map(liveCameraLayout.map((item) => [item.camera.id, item])),
    [liveCameraLayout],
  );
  const liveCameraLayoutReady = liveCameraLayout.length === visibleLiveCameras.length && visibleLiveCameras.length > 0;
  const customGridMetrics = useMemo(
    () => liveCustomGridMetrics(liveCameraGridSize.width, liveCameraGridSize.height),
    [liveCameraGridSize.height, liveCameraGridSize.width],
  );
  const updateLiveCameraAspect = React.useCallback((cameraId, aspect) => {
    const normalized = Number(aspect);
    if (!(normalized > 0)) return;
    setLiveCameraAspects((current) => (
      Math.abs(Number(current[cameraId]) - normalized) < 0.0001
        ? current
        : { ...current, [cameraId]: normalized }
    ));
  }, []);

  const clearExpandedCamera = React.useCallback(() => {
    setExpandedCamera(null);
    overlayModeRef.current = "sticky";
    setOverlayMode("sticky");
    setOverlayPhase("closed");
  }, []);

  const openExpandedCamera = React.useCallback((camera, mode = "sticky") => {
    if (!camera) return;
    const nextMode = mode === "preview" ? "preview" : "sticky";
    overlayModeRef.current = nextMode;
    setExpandedCamera(camera);
    setOverlayMode(nextMode);
    setOverlayPhase("open");
  }, []);

  const closeExpandedCamera = React.useCallback(() => {
    if (overlayModeRef.current === "preview") {
      ignoreStickyOpenUntilRef.current = Date.now() + 450;
    }
    setOverlayPhase((current) => {
      if (current === "closing" || current === "closed") return current;
      return "closing";
    });
  }, []);

  const openStickyLiveCamera = React.useCallback((camera) => {
    if (Date.now() < ignoreStickyOpenUntilRef.current) return;
    openExpandedCamera(camera, "sticky");
  }, [openExpandedCamera]);

  useEffect(() => {
    if (overlayMode !== "preview" || overlayPhase !== "open") return undefined;
    function endPreview() {
      closeExpandedCamera();
    }
    window.addEventListener("pointerup", endPreview);
    window.addEventListener("pointercancel", endPreview);
    window.addEventListener("blur", endPreview);
    return () => {
      window.removeEventListener("pointerup", endPreview);
      window.removeEventListener("pointercancel", endPreview);
      window.removeEventListener("blur", endPreview);
    };
  }, [closeExpandedCamera, overlayMode, overlayPhase]);

  useLayoutEffect(() => {
    const grid = liveCameraGridRef.current;
    if (!grid) return undefined;
    const update = () => {
      const next = {
        width: Math.max(0, grid.clientWidth),
        height: Math.max(0, grid.clientHeight),
      };
      setLiveCameraGridSize((current) => (
        current.width === next.width && current.height === next.height ? current : next
      ));
    };
    update();
    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", update);
      return () => window.removeEventListener("resize", update);
    }
    const observer = new ResizeObserver(update);
    observer.observe(grid);
    return () => observer.disconnect();
  }, []);
  useVisiblePolling(async (signal) => {
    try {
      const response = await fetch("/api/system/status", { cache: "no-store", signal });
      if (!response.ok) return;
      const payload = await response.json();
      const instanceId = String(payload.instance_id || "");
      if (!instanceId) return;
      const reset = resetLiveDefaultsForServer(browserStorage(window), instanceId);
      if (reset) clearExpandedCamera();
      setLiveDefaultsInstance(instanceId);
    } catch {
      // A reconnecting server is expected to be temporarily unavailable.
    } finally {
      setLiveDefaultsReady(true);
    }
  }, 15_000);
  useEffect(() => {
    const linkedCameraId = linkedCameraIdRef.current;
    if (!linkedCameraId || !cameras.length) return;
    const linkedCamera = cameras.find((camera) => camera.id === linkedCameraId);
    if (linkedCamera) openExpandedCamera(linkedCamera, "sticky");
    linkedCameraIdRef.current = "";
  }, [cameras, openExpandedCamera]);
  const cameraNameById = useMemo(() => new Map(cameras.map((camera) => [camera.id, camera.name || camera.id])), [cameras]);
  const incidentCameraOptions = incidentFacets.camera_ids || [];
  const incidentObjectOptions = incidentFacets.labels || [];
  const incidentZoneOptions = incidentFacets.zones || [];
  const visibleIncidents = incidents;
  const [retainedFocusedIncident, setRetainedFocusedIncident] = useState(null);
  const listedFocusedIncident = retainFocusedIncident(visibleIncidents, expandedIncidentId);
  const focusedSummary = listedFocusedIncident
    || retainFocusedIncident([], expandedIncidentId, retainedFocusedIncident);
  const focusedDetailQuery = incidentDetailQuery(focusedSummary);
  const focusedIncident = focusedSummary ? incidentDetails[focusedDetailQuery] || focusedSummary : null;
  const galleryIncidents = visibleIncidents;
  const incidentPageCount = incidentPage + (incidentHasMore ? 2 : 1);
  const clampedIncidentPage = incidentPage;
  const pagedIncidents = galleryIncidents;
  const activityQuickFilter = liveActivityQuickFilter(eventFilter, incidentObjectFilter);
  const advancedActivityFilterCount = [incidentCameraFilter, incidentZoneFilter].filter((value) => value !== "all").length
    + (activityQuickFilter === "custom" && incidentObjectFilter !== "all" ? 1 : 0);

  useEffect(() => {
    const contextualIncident = selectedEvent || focusedIncident;
    onAssistantContextChange?.({
      page: "live",
      camera_id: expandedCamera?.id || contextualIncident?.camera_id || (incidentCameraFilter === "all" ? "" : incidentCameraFilter),
      incident_event_id: Number(contextualIncident?.representative_event_id || contextualIncident?.id) || null,
      filters: {
        event_type: eventFilter,
        camera: incidentCameraFilter,
        object: incidentObjectFilter,
        zone: incidentZoneFilter,
      },
    });
  }, [eventFilter, expandedCamera?.id, focusedIncident?.camera_id, focusedIncident?.id, focusedIncident?.representative_event_id, incidentCameraFilter, incidentObjectFilter, incidentZoneFilter, onAssistantContextChange, selectedEvent?.camera_id, selectedEvent?.id, selectedEvent?.representative_event_id]);

  useEffect(() => {
    clearLegacyIncidentFilterStorage();
  }, []);

  useEffect(() => {
    const zone = liveIncidentZoneRef.current;
    const gallery = liveIncidentGalleryRef.current;
    if (!zone || !gallery) return undefined;
    function updateGallerySize() {
      const zoneRect = zone.getBoundingClientRect();
      const rect = gallery.getBoundingClientRect();
      const fixedHeight = [...zone.children]
        .filter((child) => child !== gallery && !child.classList.contains("incident-focus"))
        .reduce((total, child) => total + child.getBoundingClientRect().height, 0);
      const availableHeight = Math.max(0, zoneRect.height - fixedHeight);
      if (rect.width <= 0 || availableHeight <= 0) return;
      const next = {
        width: Math.round(rect.width),
        height: Math.round(availableHeight),
      };
      setLiveIncidentGallerySize((current) => (
        current.width === next.width && current.height === next.height ? current : next
      ));
    }
    updateGallerySize();
    const observer = new ResizeObserver(updateGallerySize);
    observer.observe(zone);
    observer.observe(gallery);
    return () => observer.disconnect();
  }, []);

  function saveCustomLayout(order, sizes) {
    setCustomLayoutValue(JSON.stringify({ version: 1, order, sizes }));
  }

  function resetCustomLayout() {
    if (!window.confirm("Reset the saved custom camera positions and sizes?")) return;
    setCustomLayoutValue("{}");
    setCustomSizePreview({});
    setKeyboardLayoutPreview(null);
  }

  function changeLiveDensity(nextDensity) {
    setLiveDensityValue(normalizedLiveDensity(nextDensity));
    setLiveDensityPageValue(0);
  }

  async function openLiveFullscreen() {
    const workspace = liveWorkspaceRef.current;
    if (!workspace?.requestFullscreen) return;
    try {
      await workspace.requestFullscreen();
    } catch {
      setLayoutAnnouncement("Fullscreen could not be opened by this browser.");
    }
  }

  function handleCustomLayoutKey(event, cameraId, actionType) {
    const activation = event.key === "Enter" || event.key === " ";
    const active = keyboardLayoutPreview?.cameraId === cameraId && keyboardLayoutPreview?.type === actionType;
    if (!active && !activation) return;
    event.preventDefault();
    event.stopPropagation();
    const cameraName = cameras.find((camera) => camera.id === cameraId)?.name || cameraId;
    if (!active) {
      setKeyboardLayoutPreview({ type: actionType, cameraId, order: [...customLayout.order], sizes: structuredClone(customLayout.sizes) });
      setLayoutAnnouncement(`${actionType === "move" ? "Move" : "Resize"} mode for ${cameraName}. Use arrow keys, Enter to save, or Escape to cancel.`);
      return;
    }
    if (event.key === "Escape") {
      setKeyboardLayoutPreview(null);
      setLayoutAnnouncement(`${actionType === "move" ? "Move" : "Resize"} cancelled for ${cameraName}.`);
      return;
    }
    if (event.key === "Enter") {
      saveCustomLayout(keyboardLayoutPreview.order, keyboardLayoutPreview.sizes);
      setKeyboardLayoutPreview(null);
      setLayoutAnnouncement(`${cameraName} layout saved.`);
      return;
    }
    const arrows = ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"];
    if (actionType === "move" && arrows.includes(event.key)) {
      const order = keyboardLayoutPreview.order;
      const index = order.indexOf(String(cameraId));
      const delta = ["ArrowLeft", "ArrowUp"].includes(event.key) ? -1 : 1;
      const targetIndex = Math.max(0, Math.min(order.length - 1, index + delta));
      if (targetIndex === index) return;
      const nextOrder = moveLiveCamera(order, String(cameraId), order[targetIndex], "swap");
      setKeyboardLayoutPreview((current) => ({ ...current, order: nextOrder }));
      setLayoutAnnouncement(`${cameraName} moved to position ${targetIndex + 1} of ${order.length}.`);
      return;
    }
    if (actionType === "resize" && (arrows.includes(event.key) || event.key.toLowerCase() === "s")) {
      const currentSize = keyboardLayoutPreview.sizes[cameraId];
      let nextSize;
      if (event.key.toLowerCase() === "s") {
        nextSize = currentSize.aspectLocked
          ? { ...currentSize, aspectLocked: false }
          : resizeLiveCameraToAspect(currentSize, 0, 0, customGridMetrics, liveCameraAspects[cameraId]);
      } else {
        const step = event.shiftKey ? 2 : 1;
        nextSize = resizeLiveCamera(
          currentSize,
          event.key === "ArrowRight" ? step : event.key === "ArrowLeft" ? -step : 0,
          event.key === "ArrowDown" ? step : event.key === "ArrowUp" ? -step : 0,
        );
      }
      setKeyboardLayoutPreview((current) => ({ ...current, sizes: { ...current.sizes, [cameraId]: nextSize } }));
      setLayoutAnnouncement(`${cameraName} size ${nextSize.columns} columns by ${nextSize.rows} rows${nextSize.aspectLocked ? ", fitted to video" : ""}.`);
    }
  }

  function beginCustomMove(event, cameraId) {
    if (event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    customMoveCleanupRef.current?.(false);
    const source = String(cameraId);
    const originalOrder = [...customLayout.order];
    let latestOrder = originalOrder;
    let changed = false;
    let activeTarget = null;
    let animationFrame = 0;
    let pendingPoint = null;
    const sourceTile = event.currentTarget.closest(".camera-tile");
    const sourceRect = sourceTile?.getBoundingClientRect();
    const grid = liveCameraGridRef.current;
    const startScrollLeft = grid?.scrollLeft || 0;
    const startScrollTop = grid?.scrollTop || 0;
    const tiles = [...(grid?.querySelectorAll(".camera-tile[data-camera-id]") || [])];
    const tilesById = new Map(tiles.map((tile) => [String(tile.dataset.cameraId), tile]));
    const dragSlots = tiles.map((tile) => {
      const rect = tile.getBoundingClientRect();
      return { id: String(tile.dataset.cameraId), left: rect.left, top: rect.top, width: rect.width, height: rect.height };
    });
    const offsetX = sourceRect ? event.clientX - sourceRect.left : 24;
    const offsetY = sourceRect ? event.clientY - sourceRect.top : 20;
    let ghost = null;
    if (sourceTile && sourceRect) {
      ghost = document.createElement("div");
      ghost.className = "camera-drag-ghost";
      ghost.setAttribute("aria-hidden", "true");
      const poster = sourceTile.querySelector(".camera-tile-poster")?.cloneNode(true);
      if (poster) ghost.appendChild(poster);
      const label = document.createElement("strong");
      label.textContent = sourceTile.querySelector(".camera-tile-name strong")?.textContent || source;
      ghost.appendChild(label);
      ghost.style.width = `${sourceRect.width}px`;
      ghost.style.height = `${sourceRect.height}px`;
      ghost.style.transform = `translate3d(${event.clientX - offsetX}px, ${event.clientY - offsetY}px, 0)`;
      document.body.appendChild(ghost);
    }
    const applyOrder = (order) => order.forEach((id, index) => {
      const tile = tilesById.get(String(id));
      if (tile) tile.style.order = String(index);
    });
    const markTarget = (targetId) => {
      activeTarget?.classList.remove("drag-over");
      activeTarget = targetId && targetId !== source ? tilesById.get(String(targetId)) : null;
      activeTarget?.classList.add("drag-over");
    };
    sourceTile?.classList.add("dragging");
    document.documentElement.classList.add("camera-layout-dragging");
    event.currentTarget.setPointerCapture?.(event.pointerId);

    const renderPoint = () => {
      animationFrame = 0;
      const point = pendingPoint;
      if (!point) return;
      pendingPoint = null;
      if (ghost) ghost.style.transform = `translate3d(${point.x - offsetX}px, ${point.y - offsetY}px, 0)`;
      const gridRect = grid?.getBoundingClientRect();
      if (!gridRect || point.x < gridRect.left || point.x > gridRect.right || point.y < gridRect.top || point.y > gridRect.bottom) {
        markTarget("");
        return;
      }
      const scrollOffsetX = startScrollLeft - (grid?.scrollLeft || 0);
      const scrollOffsetY = startScrollTop - (grid?.scrollTop || 0);
      const target = liveCustomDropTarget(
        dragSlots.map((slot) => ({ ...slot, left: slot.left + scrollOffsetX, top: slot.top + scrollOffsetY })),
        point.x,
        point.y,
        source,
      );
      if (!target) {
        markTarget("");
        return;
      }
      const next = target.position === "original"
        ? originalOrder
        : moveLiveCamera(originalOrder, source, target.targetId, target.position);
      if (next.some((id, index) => id !== latestOrder[index])) {
        latestOrder = next;
        changed = latestOrder.some((id, index) => id !== originalOrder[index]);
        applyOrder(latestOrder);
      }
      markTarget(target.targetId);
    };

    const onMove = (moveEvent) => {
      moveEvent.preventDefault();
      pendingPoint = { x: moveEvent.clientX, y: moveEvent.clientY };
      if (!animationFrame) animationFrame = window.requestAnimationFrame(renderPoint);
    };

    let finished = false;
    const finish = (commit) => {
      if (finished) return;
      finished = true;
      if (animationFrame) window.cancelAnimationFrame(animationFrame);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onPointerUp);
      window.removeEventListener("pointercancel", onPointerCancel);
      window.removeEventListener("blur", onPointerCancel);
      ghost?.remove();
      activeTarget?.classList.remove("drag-over");
      sourceTile?.classList.remove("dragging");
      document.documentElement.classList.remove("camera-layout-dragging");
      if (commit && changed) saveCustomLayout(latestOrder, customLayout.sizes);
      else applyOrder(originalOrder);
      if (customMoveCleanupRef.current === finish) customMoveCleanupRef.current = null;
    };
    const onPointerUp = (pointerEvent) => {
      pendingPoint = { x: pointerEvent.clientX, y: pointerEvent.clientY };
      if (animationFrame) window.cancelAnimationFrame(animationFrame);
      renderPoint();
      finish(true);
    };
    const onPointerCancel = () => finish(false);
    window.addEventListener("pointermove", onMove, { passive: false });
    window.addEventListener("pointerup", onPointerUp, { once: true });
    window.addEventListener("pointercancel", onPointerCancel, { once: true });
    window.addEventListener("blur", onPointerCancel, { once: true });
    customMoveCleanupRef.current = finish;
  }

  function beginCustomResize(event, cameraId) {
    event.preventDefault();
    event.stopPropagation();
    const start = customSizePreview[cameraId] || customLayout.sizes[cameraId];
    const startX = event.clientX;
    const startY = event.clientY;
    let latest = start;
    setResizingCameraId(cameraId);
    document.documentElement.classList.add("camera-layout-resizing");
    event.currentTarget.setPointerCapture?.(event.pointerId);
    const onMove = (moveEvent) => {
      latest = resizeLiveCameraToAspect(
        start,
        moveEvent.clientX - startX,
        moveEvent.clientY - startY,
        customGridMetrics,
        liveCameraAspects[cameraId],
      );
      setCustomSizePreview((current) => ({ ...current, [cameraId]: latest }));
    };
    const onEnd = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onEnd);
      window.removeEventListener("pointercancel", onEnd);
      window.removeEventListener("blur", onEnd);
      const sizes = { ...customLayout.sizes, ...customSizePreview, [cameraId]: latest };
      saveCustomLayout(customLayout.order, sizes);
      setCustomSizePreview({});
      setResizingCameraId("");
      document.documentElement.classList.remove("camera-layout-resizing");
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onEnd, { once: true });
    window.addEventListener("pointercancel", onEnd, { once: true });
    window.addEventListener("blur", onEnd, { once: true });
  }

  useEffect(() => () => {
    customMoveCleanupRef.current?.(false);
    document.documentElement.classList.remove("camera-layout-resizing");
    document.documentElement.classList.remove("camera-layout-dragging");
    document.querySelectorAll(".camera-drag-ghost").forEach((ghost) => ghost.remove());
  }, []);

  useEffect(() => {
    incidentFeedCacheRef.current.clear();
    incidentLoadedQueryRef.current = "";
    setIncidents([]);
    setIncidentHasMore(false);
    setIncidentLoading(true);
    setIncidentLoadError("");
    setIncidentPage(0);
    setExpandedIncidentId(null);
    setRetainedFocusedIncident(null);
  }, [eventFilter, incidentCameraFilter, incidentObjectFilter, incidentZoneFilter]);

  useEffect(() => {
    incidentFeedCacheRef.current.clear();
    setIncidentPage(0);
  }, [incidentsPerPage]);

  function refreshIncidents() {
    incidentSelectionRequestRef.current += 1;
    incidentFeedCacheRef.current.clear();
    incidentDetailCacheRef.current.clear();
    setIncidentDetails({});
    setIncidentRefreshToken((value) => value + 1);
  }

  useAppEvents(({ type }) => {
    if (type !== "incident" || incidentPage !== 0 || document.hidden) return;
    window.clearTimeout(incidentEventRefreshTimer.current);
    incidentEventRefreshTimer.current = window.setTimeout(() => {
      incidentEventRefreshTimer.current = null;
      incidentFeedCacheRef.current.clear();
      if (focusedDetailQuery) {
        incidentDetailCacheRef.current.invalidate(focusedDetailQuery);
        incidentDetailCacheRef.current.load(focusedDetailQuery).then((detail) => {
          setIncidentDetails((current) => ({ ...current, [focusedDetailQuery]: detail }));
        }).catch(() => {
          // Keep the existing detail visible; the next event or fallback poll retries.
        });
      }
      setIncidentRefreshToken((value) => value + 1);
    }, 1000);
  });

  useVisiblePolling(() => {
    incidentFeedCacheRef.current.clear();
    setIncidentRefreshToken((value) => value + 1);
  }, INCIDENT_REFRESH_FALLBACK_MS, incidentPage === 0, { immediate: false });

  useEffect(() => () => {
    incidentSelectionRequestRef.current += 1;
    window.clearTimeout(incidentEventRefreshTimer.current);
  }, []);

  useEffect(() => {
    if (!liveIncidentGalleryReady) return undefined;
    let cancelled = false;
    function feedQuery(page) {
      const query = new URLSearchParams({
        event_type: eventFilter,
        limit: String(incidentsPerPage),
        offset: String(page * incidentsPerPage),
        gap_seconds: "45",
      });
      if (incidentCameraFilter !== "all") query.set("camera_id", incidentCameraFilter);
      if (incidentObjectFilter !== "all") query.set("object_label", incidentObjectFilter);
      if (incidentZoneFilter !== "all") query.set("zone", incidentZoneFilter);
      return query.toString();
    }
    async function loadIncidentFeed() {
      const query = feedQuery(incidentPage);
      const cachedPayload = incidentFeedCacheRef.current.peek(query);
      const foregroundLoad = incidentLoadedQueryRef.current !== query;
      if (foregroundLoad && !cachedPayload) setIncidentLoading(true);
      setIncidentLoadError("");
      try {
        const payload = cachedPayload || await incidentFeedCacheRef.current.load(query);
        if (cancelled) return;
        incidentLoadedQueryRef.current = query;
        setIncidents(incidentsNewestFirst(payload.items || []));
        setIncidentFacets(payload.facets || { camera_ids: [], labels: [], zones: [] });
        setIncidentHasMore(Boolean(payload.has_more));
        const previousQuery = incidentPage > 0 ? feedQuery(incidentPage - 1) : "";
        const nextQuery = payload.has_more ? feedQuery(incidentPage + 1) : "";
        for (const adjacentQuery of [previousQuery, nextQuery].filter(Boolean)) {
          incidentFeedCacheRef.current.load(adjacentQuery).catch(() => {
            // A foreground navigation will retry a failed speculative request.
          });
        }
        incidentFeedCacheRef.current.retain([previousQuery, query, nextQuery]);
      } catch (error) {
        if (cancelled) return;
        setIncidentLoadError(error.message || "Unable to load recent incidents");
      } finally {
        if (!cancelled && foregroundLoad) setIncidentLoading(false);
      }
    }
    loadIncidentFeed();
    return () => {
      cancelled = true;
    };
  }, [eventFilter, incidentCameraFilter, incidentObjectFilter, incidentZoneFilter, incidentPage, incidentsPerPage, incidentRefreshToken, liveIncidentGalleryReady]);

  useEffect(() => {
    if (!focusedSummary || !focusedDetailQuery || incidentDetails[focusedDetailQuery]) return;
    let cancelled = false;
    incidentDetailCacheRef.current.load(focusedDetailQuery).then((detail) => {
      if (!cancelled) setIncidentDetails((current) => ({ ...current, [focusedDetailQuery]: detail }));
    }).catch(() => {
      // The compact incident remains usable if investigation details fail.
    });
    return () => { cancelled = true; };
  }, [focusedSummary?.id, focusedDetailQuery, incidentDetails]);

  useEffect(() => {
    const context = incidentRecordingContext(selectedEvent || focusedIncident);
    if (context) onRecordingContextChange(context);
  }, [selectedEvent?.id, focusedIncident?.id, focusedIncident?.created_at, focusedIncident?.camera_id, onRecordingContextChange]);

  useEffect(() => {
    if (listedFocusedIncident) setRetainedFocusedIncident(listedFocusedIncident);
  }, [listedFocusedIncident]);

  useEffect(() => {
    function onKey(keyEvent) {
      if (keyEvent.key === "Escape" && expandedIncidentId && !selectedEvent) {
        keyEvent.preventDefault();
        setExpandedIncidentId(null);
        setRetainedFocusedIncident(null);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expandedIncidentId, selectedEvent]);

  function toggleIncident(incidentId) {
    const closing = String(expandedIncidentId) === String(incidentId);
    if (closing) {
      setExpandedIncidentId(null);
      setRetainedFocusedIncident(null);
      return;
    }
    const incident = visibleIncidents.find((candidate) => String(candidate.id) === String(incidentId));
    setRetainedFocusedIncident(incident || null);
    setExpandedIncidentId(incidentId);
  }

  function changeIncidentPage(nextPage) {
    setExpandedIncidentId(null);
    setRetainedFocusedIncident(null);
    setIncidentPage(Math.max(0, nextPage));
  }

  function selectActivityQuickFilter(mode) {
    const selection = liveActivityQuickSelection(mode);
    setEventFilter(selection.eventType);
    setIncidentObjectFilter(selection.objectFilter);
    setIncidentPage(0);
  }

  return (
    <main ref={liveWorkspaceRef} className="bento-grid live-grid">
      <LiveCommandBar cameras={orderedCameras} focusedCameraId={mobileFocusedCameraId} onFocusedCameraChange={setStoredMobileFocus} cameraCount={visibleLiveCameras.length} totalCameraCount={orderedCameras.length} density={effectiveLiveDensity} densityPage={densitySelection.page} densityPageCount={densitySelection.pageCount} layoutMode={effectiveLayoutMode} customAvailable={customLayoutAvailable} onDensityChange={changeLiveDensity} onDensityPageChange={setLiveDensityPageValue} onLayoutModeChange={setLiveLayoutMode} onResetLayout={resetCustomLayout} onFullscreen={openLiveFullscreen} />
      <div className="sr-only" role="status" aria-live="polite">{layoutAnnouncement}</div>
      <section className="bento-card camera-zone live-camera-zone">
        <div className="mobile-camera-picker" role="group" aria-label="Primary live camera">
          {orderedCameras.map((camera) => {
            const online = Boolean(camera.connected ?? camera.running);
            const active = camera.id === mobileFocusedCameraId;
            return (
              <button
                type="button"
                key={camera.id}
                className={active ? "active" : ""}
                aria-pressed={active}
                onClick={() => setStoredMobileFocus(camera.id)}
              >
                <Camera size={16} aria-hidden="true" />
                <span>{camera.name || camera.id}</span>
                <i className={online ? "online" : ""} aria-hidden="true" />
              </button>
            );
          })}
        </div>
        <div
          ref={liveCameraGridRef}
          className={`camera-grid live-camera-grid${effectiveLayoutMode === "custom" ? " custom-layout" : liveCameraLayoutReady ? " viewport-layout" : ""}`}
          style={effectiveLayoutMode === "custom" ? { "--custom-pack-row-height": `${customGridMetrics.packRowHeight}px` } : undefined}
        >
          {liveDefaultsReady ? renderedCameras.map((camera) => (
            <CameraTile
              key={`${camera.id}:${liveDefaultsInstance}`}
              camera={camera}
              timeZone={timeZone}
              refresh={refreshBase}
              onOpen={openStickyLiveCamera}
              onPreviewOpen={(camera) => openExpandedCamera(camera, "preview")}
              onPreviewClose={closeExpandedCamera}
              onAspectChange={updateLiveCameraAspect}
              layout={effectiveLayoutMode === "auto" ? liveCameraLayoutById.get(camera.id) : null}
              customLayout={effectiveLayoutMode === "custom"}
              customStyle={effectiveLayoutMode === "custom" ? (() => {
                const size = customSizePreview[camera.id] || displayedCustomLayout.sizes[camera.id];
                const measuredAspect = Number(liveCameraAspects[camera.id]);
                const placement = liveCustomTilePlacement(size, customGridMetrics, measuredAspect);
                return {
                  order: displayedCustomLayout.order.indexOf(String(camera.id)),
                  gridColumn: `span ${placement.columns}`,
                  gridRow: `span ${placement.packedRows}`,
                  height: `${placement.height}px`,
                };
              })() : undefined}
              startDelayMs={(liveCameraStartIndex.get(camera.id) || 0) * 450}
              resizing={resizingCameraId === camera.id}
              aspectSnapped={Boolean((customSizePreview[camera.id] || displayedCustomLayout.sizes[camera.id]).aspectLocked)}
              dragHandleProps={effectiveLayoutMode === "custom" ? {
                onPointerDown: (event) => beginCustomMove(event, camera.id),
                onKeyDown: (event) => handleCustomLayoutKey(event, camera.id, "move"),
                "aria-pressed": keyboardLayoutPreview?.cameraId === camera.id && keyboardLayoutPreview?.type === "move",
                title: "Drag to move, or press Enter and use arrow keys",
                "aria-label": `Move ${camera.name}. Press Enter, use arrow keys, then Enter to save or Escape to cancel`,
              } : {}}
              resizeHandleProps={effectiveLayoutMode === "custom" ? {
                onPointerDown: (event) => beginCustomResize(event, camera.id),
                onKeyDown: (event) => handleCustomLayoutKey(event, camera.id, "resize"),
                "aria-pressed": keyboardLayoutPreview?.cameraId === camera.id && keyboardLayoutPreview?.type === "resize",
                title: "Drag to resize, or press Enter and use arrow keys; S fits the video",
                "aria-label": `Resize ${camera.name}. Press Enter, use arrow keys, S to fit the video, then Enter to save or Escape to cancel`,
              } : {}}
              mobilePrimary={camera.id === mobileFocusedCameraId}
              mobileView={mobileLiveView}
            />
          )) : null}
        </div>
      </section>
      <section className="bento-card events-zone" ref={liveIncidentZoneRef}>
        <div className="section-head compact incident-head">
          <div><h2>Recent Activity</h2></div>
          <div className="incident-head-actions">
            <div className="incident-filter-toggle compact live-quick-filters" role="group" aria-label="Recent activity filter">
              <button className={activityQuickFilter === "object" ? "active" : ""} aria-pressed={activityQuickFilter === "object"} onClick={() => selectActivityQuickFilter("object")}>Object</button>
              <button className={activityQuickFilter === "motion" ? "active" : ""} aria-pressed={activityQuickFilter === "motion"} onClick={() => selectActivityQuickFilter("motion")}>Motion</button>
            </div>
          </div>
        </div>
        <details className="live-activity-filters">
          <summary><SlidersHorizontal size={14} /> Advanced filters{advancedActivityFilterCount ? ` (${advancedActivityFilterCount})` : ""}</summary>
          <div className="event-filter incident-filter-panel" aria-label="Incident filters">
            <div className="incident-filter-selects">
              <label>
                <span>Camera</span>
                <select value={incidentCameraFilter} onChange={(event) => setIncidentCameraFilter(event.target.value)}>
                  <option value="all">All cameras</option>
                  {incidentCameraOptions.map((id) => <option value={id} key={id}>{cameraNameById.get(id) || id}</option>)}
                </select>
              </label>
              <label>
                <span>Object</span>
                <select value={incidentObjectFilter} onChange={(event) => setIncidentObjectFilter(event.target.value)}>
                  <option value="all">All objects</option>
                  {incidentObjectOptions.map((label) => <option value={label} key={label}>{label}</option>)}
                </select>
              </label>
              <label>
                <span>Zone</span>
                <select value={incidentZoneFilter} onChange={(event) => setIncidentZoneFilter(event.target.value)}>
                  <option value="all">All zones</option>
                  {incidentZoneOptions.map((zone) => <option value={zone} key={zone}>{zone}</option>)}
                </select>
              </label>
            </div>
          </div>
        </details>
        <div className="live-activity-list" ref={liveIncidentGalleryRef}>
          {incidentLoading && !visibleIncidents.length ? <div className="empty-state">Loading {eventFilter} incidents...</div> : null}
          {!visibleIncidents.length && incidentLoadError ? <div className="empty-state live-activity-error"><span>{incidentLoadError}</span><button type="button" onClick={refreshIncidents}>Retry</button></div> : null}
          {visibleIncidents.length
            ? pagedIncidents.map((incident) => (
              <IncidentListItem
                key={incident.id}
                incident={incident.id === focusedIncident?.id ? focusedIncident : incident}
                cameraName={cameraNameById.get(incident.camera_id) || incident.camera_id}
                timeZone={timeZone}
                selected={incident.id === focusedIncident?.id}
                thumbnailAnnotations={thumbnailAnnotations}
                onSelect={openIncidentOverlay}
                onOpenOverlay={openIncidentOverlay}
              />
            ))
            : null}
          {!incidentLoading && !incidentLoadError && !visibleIncidents.length ? <div className="empty-state">No incidents match the current filters.</div> : null}
        </div>
        <div className="live-activity-footer">
          <div className={`incident-pager ${incidentPage > 0 || incidentHasMore ? "" : "placeholder"}`} aria-label="Incident pages" aria-hidden={incidentPage === 0 && !incidentHasMore}>
            <button type="button" onClick={() => changeIncidentPage(incidentPage - 1)} disabled={clampedIncidentPage === 0}>Prev</button>
            <span>{clampedIncidentPage + 1} / {incidentPageCount}</span>
            <button type="button" onClick={() => changeIncidentPage(incidentPage + 1)} disabled={!incidentHasMore}>Next</button>
          </div>
          <a href={appUrl("/incidents")}>View all incidents <ChevronRight size={14} /></a>
        </div>
      </section>
      {selectedEvent ? <EventOverlay event={selectedEvent} events={visibleIncidents} timeZone={timeZone} onClose={closeIncidentOverlay} onSelect={openIncidentOverlay} onRefresh={refreshIncidents} /> : null}
      {expandedCamera ? (
        <LiveCameraOverlay
          camera={expandedCamera}
          timeZone={timeZone}
          mode={overlayMode}
          phase={overlayPhase}
          onClose={closeExpandedCamera}
          onClosed={clearExpandedCamera}
        />
      ) : null}
    </main>
  );
}
