import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  ChevronLeft,
  ChevronRight,
  Crop,
  Clock3,
  Download,
  Grid2X2,
  Images,
  Layers,
  Pentagon,
  ListTree,
  Link2,
  Play,
  Route,
  ScanEye,
  Sparkles,
  X,
} from "lucide-react";
import { crossCameraMatchCameraLabel, crossCameraMatchLabel, crossCameraTracePath } from "../crossCameraTrace.mjs";
import { incidentReplayTracking, trackingCoverageLabel } from "../objectTrackReplay.mjs";
import { incidentSceneObjects, observedObjectSummaries } from "../incidentScene.mjs";
import { incidentEvidenceTimeline, incidentMosaicEvents, incidentMosaicPage, incidentTriggerLabel, showIncidentCardAnnotations } from "../incidentNavigation.mjs";
import { relatedEvidenceReasons, relatedIncidentThumbnailPath, relatedIncidentsPath, visibleRelatedAppearances } from "../relatedIncidents.mjs";
import {
  appearanceCapableLabel,
  appearanceMatchesPath,
  hybridFindSimilarSubtitle,
  hybridMatchLabel,
  isValidObjectIndex,
  mergeHybridFindSimilarResults,
  resolveObjectTrackId,
  visualSearchObjects,
  visualSearchRequest,
} from "../visualSearch.mjs";
import { writeVisualSearchTrail } from "../visualSearchTrail.mjs";
import { appUrl, fetch, incidentRecordingContext, recordingsHref } from "../shared/api.js";
import { formatDateTime, formatTimeOnly, formatDuration } from "../shared/format.js";
import { eventSnapshotDownloadUrl, eventClipUrl } from "../shared/mediaUrls.js";
import {
  IncidentObjectBadges,
  IncidentSourceDot,
  SnapshotImage,
  formatDepthMeters,
  hasDetectedObjects,
  incidentClipWindow,
  incidentLabels,
} from "../shared/evidence.jsx";
import { IncidentRecordingPlayer, incidentRecordingBounds } from "./IncidentRecordingPlayer.jsx";

function evidenceSpanLabel(seconds) {
  if (!Number.isFinite(seconds) || seconds < 1) return "";
  if (seconds < 60) return `+${Math.round(seconds)}s`;
  return `+${formatDuration(seconds)}`;
}

function EvidenceTimeline({ incident, hero, timeline, timeZone, stripRef, onSelect, onKeyDown, onPlay, playLabel }) {
  const frames = timeline.frames;
  const motionCount = timeline.motionMarks.length;
  const summary = motionCount
    ? `${frames.length} event ${frames.length === 1 ? "image" : "images"}, ${motionCount} motion ${motionCount === 1 ? "update" : "updates"}`
    : `${frames.length} event ${frames.length === 1 ? "image" : "images"}`;
  const eventCountLabel = `${frames.length} ${frames.length === 1 ? "event" : "events"}`;
  const spanLabel = evidenceSpanLabel(timeline.durationSeconds);
  const heroEvent = hero?.event || incident;
  const heroLabels = incidentLabels(heroEvent);
  const heroTime = formatTimeOnly(heroEvent.created_at || hero?.epoch || incident.created_at, timeZone);
  return (
    <div className="incident-evidence" role="group" aria-label={`Incident evidence timeline, ${summary}`}>
      <div className="incident-evidence-hero">
        <SnapshotImage event={heroEvent} alt="Selected evidence frame" objectFocusMode="off" objectFocusControls={false} showAnnotations showTracking={false}>
          <div className="incident-snapshot-hud">
            <div className="incident-snapshot-main">
              <strong>{heroEvent.camera_id || incident.camera_id}</strong>
              <time>{heroTime}</time>
            </div>
            <div className="pill-row compact incident-labels">
              <IncidentObjectBadges labels={heroLabels} />
            </div>
          </div>
        </SnapshotImage>
        <button type="button" className="incident-preview-media-action media-surface-action" onClick={onPlay} aria-label={playLabel} />
      </div>
      <div className="incident-evidence-rail">
        <div className="incident-evidence-strip" ref={stripRef} onKeyDown={onKeyDown}>
          {frames.map((frame) => {
            const selected = frame.key === hero?.key;
            const time = formatTimeOnly(frame.event.created_at || frame.epoch || incident.created_at, timeZone);
            const labels = incidentLabels(frame.event);
            const labelText = labels.length ? labels.join(", ") : "motion";
            return (
              <button
                type="button"
                key={frame.key}
                data-evidence-key={frame.key}
                className={`incident-evidence-thumb${selected ? " selected" : ""}`}
                aria-current={selected ? "true" : undefined}
                aria-pressed={selected}
                aria-label={`Show ${labelText} at ${time}`}
                onClick={(clickEvent) => { clickEvent.stopPropagation(); onSelect(frame); }}
              >
                <SnapshotImage event={frame.event} alt="" thumbnail objectFocusMode="off" objectFocusControls={false} showAnnotations={false} showTracking={false} />
                <time dateTime={frame.event.created_at || undefined}>{time}</time>
                <span>{labelText}</span>
              </button>
            );
          })}
        </div>
        <div className="incident-evidence-ruler">
          <time>{timeline.startEpoch ? formatTimeOnly(timeline.startEpoch, timeZone) : ""}</time>
          <div className="incident-evidence-ruler-track" aria-hidden="true">
            {frames.map((frame) => (
              <i key={frame.key} className={`incident-evidence-tick${frame.key === hero?.key ? " selected" : ""}`} style={{ left: `${frame.position * 100}%` }} />
            ))}
            {timeline.motionMarks.map((mark) => (
              <i key={mark.key} className="incident-evidence-tick motion" style={{ left: `${mark.position * 100}%` }} title={mark.label} />
            ))}
          </div>
          <span>{spanLabel ? `${eventCountLabel} · ${spanLabel}` : eventCountLabel}</span>
        </div>
      </div>
    </div>
  );
}

export function IncidentCard({ incident, scenePlayback, timeZone, expanded, selected = false, thumbnailAnnotations = true, thumbnailObjectFocus = "off", thumbnailObjectFocusZoom = 1, desktopWorkspace = false, zones = null, analysisMode = "clean", depthLayer = "both", replayRequest = 0, observationPreviewRequest = null, selectedObjectIndex = null, onSelectObject = null, onReturnToSelected = null, onAnalysisStats, onToggle, onSelect, onPreviewChange, onImageSize }) {
  const rawEvents = incident.events || [];
  const motionObservations = incident.motion_observations || [];
  const [selectedPreview, setSelectedPreview] = useState(null);
  const [workspaceView, setWorkspaceView] = useState(() => {
    try {
      const stored = window.sessionStorage.getItem("survng.incidentWorkspaceView.v1");
      return ["mosaic", "evidence"].includes(stored) ? stored : "focus";
    } catch { return "focus"; }
  });
  const [mosaicPageIndex, setMosaicPageIndex] = useState(0);
  const [inlineVideoActive, setInlineVideoActive] = useState(false);
  const [snapshotZoom, setSnapshotZoom] = useState({ scale: 1, x: 0, y: 0 });
  const [zonesVisible, setZonesVisible] = useState(false);
  const previewRef = useRef(null);
  const evidenceStripRef = useRef(null);
  const snapshotZoomRef = useRef(snapshotZoom);
  const panGestureRef = useRef({ pointerId: null, startX: 0, startY: 0, panX: 0, panY: 0, moved: false });
  const replayRequestRef = useRef(replayRequest);
  const evidenceTimeline = useMemo(() => incidentEvidenceTimeline(incident), [incident]);
  const evidenceHero = useMemo(() => {
    const frames = evidenceTimeline.frames;
    const explicit = selectedPreview?.id == null
      ? null
      : frames.find((frame) => String(frame.event?.id) === String(selectedPreview.id));
    const representative = incident.representative_event_id == null
      ? null
      : frames.find((frame) => String(frame.event?.id) === String(incident.representative_event_id));
    return explicit || representative || frames[0] || null;
  }, [evidenceTimeline, incident.representative_event_id, selectedPreview]);
  const preview = scenePlayback?.clip || selectedPreview || incident;
  const videoActive = Boolean(scenePlayback?.clip) || inlineVideoActive;
  const playbackBounds = videoActive ? incidentRecordingBounds(scenePlayback?.clip || preview) : null;
  const mosaicEvents = useMemo(() => incidentMosaicEvents(incident), [incident]);
  const mosaic = useMemo(() => incidentMosaicPage(mosaicEvents, mosaicPageIndex), [mosaicEvents, mosaicPageIndex]);
  const canShowMosaic = desktopWorkspace && expanded && mosaicEvents.length > 1;
  const canShowEvidence = desktopWorkspace && expanded && evidenceTimeline.frames.length > 0;
  const zoneShapes = (Array.isArray(zones) ? zones : []).filter((zone) => zone?.enabled !== false && (zone.points || []).length >= 3);
  const activeWorkspaceView = workspaceView === "mosaic" && !canShowMosaic
    ? "focus"
    : workspaceView === "evidence" && !canShowEvidence ? "focus" : workspaceView;
  const showZoneToggle = desktopWorkspace && expanded && activeWorkspaceView === "focus" && zoneShapes.length > 0;
  const trackingPreview = incidentReplayTracking(scenePlayback?.clip || preview, incident) || preview;
  const labels = incidentLabels(incident);
  const eventCount = incident.event_count || rawEvents.length || 1;
  const observationCount = Number(incident.motion_observation_count || motionObservations.length || 0);
  const countText = `${eventCount} ${eventCount === 1 ? "event" : "events"}${observationCount ? ` · ${observationCount} additional motion update${observationCount === 1 ? "" : "s"}` : ""}`;
  const triggerLabel = incidentTriggerLabel(incident);
  const triggerTitle = triggerLabel === "EMA" ? "EMA visual backup trigger" : "Camera motion trigger";
  const timeText = incident.start_at && incident.end_at && incident.start_at !== incident.end_at
    ? `${formatDateTime(incident.start_at, timeZone)} - ${formatDuration(incident.duration_seconds)}`
    : formatDateTime(incident.created_at, timeZone);
  const previewTimeText = preview.created_at ? formatDateTime(preview.created_at, timeZone) : timeText;

  useEffect(() => {
    if (!expanded) {
      setSelectedPreview(null);
      setInlineVideoActive(false);
    }
  }, [expanded]);

  useEffect(() => {
    setSelectedPreview(null);
    setMosaicPageIndex(0);
    setInlineVideoActive(false);
    replayRequestRef.current = replayRequest;
  }, [incident.id]);

  useEffect(() => {
    if (!observationPreviewRequest) return;
    const source = rawEvents.find((event) => Number(event.id) === Number(observationPreviewRequest.eventId));
    if (source) { setSelectedPreview(source); setInlineVideoActive(false); setWorkspaceView("focus"); }
  }, [observationPreviewRequest]);

  useEffect(() => {
    if (scenePlayback?.selection) setWorkspaceView("focus");
  }, [scenePlayback?.selection]);

  useEffect(() => {
    try { window.sessionStorage.setItem("survng.incidentWorkspaceView.v1", workspaceView); } catch { /* Session preference is optional. */ }
  }, [workspaceView]);

  useEffect(() => {
    if (activeWorkspaceView !== "evidence") return;
    const strip = evidenceStripRef.current;
    const selected = strip?.querySelector("[aria-current='true']");
    if (!strip || !selected) return;
    const stripBox = strip.getBoundingClientRect();
    const itemBox = selected.getBoundingClientRect();
    if (itemBox.left < stripBox.left || itemBox.right > stripBox.right) {
      selected.scrollIntoView({ inline: "center", block: "nearest" });
    }
  }, [activeWorkspaceView, evidenceHero?.key]);

  useEffect(() => {
    setInlineVideoActive(false);
    resetSnapshotZoom();
  }, [preview.id, preview.created_at]);

  useEffect(() => {
    const previousRequest = replayRequestRef.current;
    replayRequestRef.current = replayRequest;
    if (expanded && replayRequest > previousRequest) { if (scenePlayback) scenePlayback.playAll(); else setInlineVideoActive(true); }
  }, [expanded, replayRequest]);

  useEffect(() => {
    if (activeWorkspaceView !== "focus") setInlineVideoActive(false);
  }, [activeWorkspaceView]);

  useEffect(() => {
    if (!expanded || !onPreviewChange) return;
    const representative = rawEvents.find((event) => Number(event.id) === Number(incident.representative_event_id));
    onPreviewChange(Number((selectedPreview || representative || incident).id));
  }, [expanded, incident, onPreviewChange, rawEvents, selectedPreview]);

  function toggle() {
    onToggle(incident.id);
  }

  function openPreview(pointerEvent) {
    pointerEvent.stopPropagation();
    if (panGestureRef.current.moved) {
      panGestureRef.current.moved = false;
      return;
    }
    if (desktopWorkspace && expanded && snapshotZoomRef.current.scale > 1) return;
    if (expanded) { if (scenePlayback) scenePlayback.playAll(); else setInlineVideoActive(true); }
    else toggle();
  }

  function clampSnapshotZoom(nextZoom) {
    const scale = Math.max(1, Math.min(6, nextZoom.scale));
    if (scale === 1) return { scale: 1, x: 0, y: 0 };
    const box = previewRef.current?.getBoundingClientRect();
    const limitX = box ? box.width * (scale - 1) / 2 : 0;
    const limitY = box ? box.height * (scale - 1) / 2 : 0;
    return {
      scale,
      x: Math.max(-limitX, Math.min(limitX, nextZoom.x || 0)),
      y: Math.max(-limitY, Math.min(limitY, nextZoom.y || 0)),
    };
  }

  function updateSnapshotZoom(updater) {
    setSnapshotZoom((current) => {
      const candidate = typeof updater === "function" ? updater(current) : updater;
      const next = clampSnapshotZoom(candidate);
      snapshotZoomRef.current = next;
      return next;
    });
  }

  function resetSnapshotZoom() {
    const reset = { scale: 1, x: 0, y: 0 };
    snapshotZoomRef.current = reset;
    setSnapshotZoom(reset);
  }

  function onPreviewWheel(wheelEvent) {
    if (!desktopWorkspace || !expanded) return;
    wheelEvent.preventDefault();
    const box = previewRef.current?.getBoundingClientRect();
    if (!box) return;
    const delta = Math.max(-120, Math.min(120, wheelEvent.deltaY));
    const factor = Math.exp(-delta * 0.0017);
    setInlineVideoActive(false);
    updateSnapshotZoom((current) => {
      const nextScale = Math.max(1, Math.min(6, current.scale * factor));
      if (nextScale === 1) return { scale: 1, x: 0, y: 0 };
      const anchorX = wheelEvent.clientX - box.left - box.width / 2;
      const anchorY = wheelEvent.clientY - box.top - box.height / 2;
      const scaleRatio = nextScale / current.scale;
      return {
        scale: nextScale,
        x: anchorX - (anchorX - current.x) * scaleRatio,
        y: anchorY - (anchorY - current.y) * scaleRatio,
      };
    });
  }

  function onPreviewPointerDown(pointerEvent) {
    if (!desktopWorkspace || !expanded || pointerEvent.pointerType === "touch" || snapshotZoomRef.current.scale <= 1) return;
    pointerEvent.preventDefault();
    pointerEvent.currentTarget.setPointerCapture(pointerEvent.pointerId);
    const current = snapshotZoomRef.current;
    panGestureRef.current = { pointerId: pointerEvent.pointerId, startX: pointerEvent.clientX, startY: pointerEvent.clientY, panX: current.x, panY: current.y, moved: false };
  }

  function onPreviewPointerMove(pointerEvent) {
    const gesture = panGestureRef.current;
    if (gesture.pointerId !== pointerEvent.pointerId || snapshotZoomRef.current.scale <= 1) return;
    pointerEvent.preventDefault();
    const dx = pointerEvent.clientX - gesture.startX;
    const dy = pointerEvent.clientY - gesture.startY;
    if (Math.abs(dx) + Math.abs(dy) > 4) gesture.moved = true;
    updateSnapshotZoom({ scale: snapshotZoomRef.current.scale, x: gesture.panX + dx, y: gesture.panY + dy });
  }

  function onPreviewPointerUp(pointerEvent) {
    if (panGestureRef.current.pointerId !== pointerEvent.pointerId) return;
    panGestureRef.current.pointerId = null;
  }

  function openOverlay(pointerEvent) {
    pointerEvent.stopPropagation();
    if (!onSelect) return;
    onSelect({
      ...preview,
      start_epoch: incident.start_epoch,
      last_epoch: incident.last_epoch,
      start_at: incident.start_at,
      end_at: incident.end_at,
      event_count: eventCount,
      events: rawEvents,
    });
  }

  function selectWorkspaceView(view) {
    setWorkspaceView(view);
    if (view !== "focus") {
      setInlineVideoActive(false);
      resetSnapshotZoom();
    }
  }

  function selectMosaicEvent(event) {
    setSelectedPreview(event);
    setInlineVideoActive(false);
    setWorkspaceView("focus");
  }

  function selectEvidenceFrame(frame) {
    setSelectedPreview(frame.event);
    setInlineVideoActive(false);
  }

  function onEvidenceKeyDown(keyEvent) {
    if (keyEvent.key !== "ArrowLeft" && keyEvent.key !== "ArrowRight") return;
    const frames = evidenceTimeline.frames;
    const index = frames.findIndex((frame) => frame.key === evidenceHero?.key);
    const next = frames[index + (keyEvent.key === "ArrowRight" ? 1 : -1)];
    if (!next) return;
    keyEvent.preventDefault();
    keyEvent.stopPropagation();
    selectEvidenceFrame(next);
    evidenceStripRef.current?.querySelector(`[data-evidence-key="${CSS.escape(next.key)}"]`)?.focus();
  }

  return (
    <article
      className={`incident-card ${hasDetectedObjects(incident) ? "has-objects" : ""} ${expanded ? "expanded" : ""} ${selected ? "selected" : ""}`}
      aria-current={selected ? "true" : undefined}
      title={`${incident.camera_id} ${timeText}`}
    >
      <div
        ref={previewRef}
        className={`incident-preview ${activeWorkspaceView !== "focus" ? "mosaic-view" : ""} ${desktopWorkspace && expanded && activeWorkspaceView === "focus" ? "zoomable" : ""} ${snapshotZoom.scale > 1 ? "zoomed" : ""}`}
        onDoubleClick={(pointerEvent) => {
          if (!desktopWorkspace || !expanded) return;
          pointerEvent.preventDefault();
          pointerEvent.stopPropagation();
          resetSnapshotZoom();
        }}
        onWheel={activeWorkspaceView === "focus" ? onPreviewWheel : undefined}
        onPointerDown={activeWorkspaceView === "focus" ? onPreviewPointerDown : undefined}
        onPointerMove={activeWorkspaceView === "focus" ? onPreviewPointerMove : undefined}
        onPointerUp={activeWorkspaceView === "focus" ? onPreviewPointerUp : undefined}
        onPointerCancel={activeWorkspaceView === "focus" ? onPreviewPointerUp : undefined}
        title={desktopWorkspace && expanded && activeWorkspaceView === "focus" ? (snapshotZoom.scale > 1 ? "Drag to pan. Double-click to reset zoom." : "Scroll to zoom. Click to play event video.") : undefined}
      >
        {activeWorkspaceView === "mosaic" ? (
          <div className={`incident-mosaic incident-mosaic-${mosaic.items.length}`} role="group" aria-label={`Events ${mosaic.page * 6 + 1} through ${mosaic.page * 6 + mosaic.items.length} of ${mosaicEvents.length}`}>
            {mosaic.items.map((event, index) => {
              const eventLabels = incidentLabels(event);
              const eventTrigger = incidentTriggerLabel(event);
              return (
                <button
                  type="button"
                  className="incident-mosaic-tile"
                  key={`${event.id || "event"}-${index}`}
                  onClick={(clickEvent) => { clickEvent.stopPropagation(); selectMosaicEvent(event); }}
                  aria-label={`Focus event at ${formatTimeOnly(event.created_at || incident.created_at, timeZone)}`}
                >
                  <SnapshotImage event={event} alt="incident event snapshot" className="incident-mosaic-snapshot" progressive thumbnail objectFocusMode={thumbnailObjectFocus} objectFocusZoom={thumbnailObjectFocusZoom} objectFocusAspect={null} objectFocusControls={false} showAnnotations showTracking={false}>
                    <IncidentSourceDot trigger={eventTrigger} className="incident-mosaic-source" />
                    <div className="incident-mosaic-hud">
                      <time>{formatTimeOnly(event.created_at || incident.created_at, timeZone)}</time>
                      <div className="pill-row compact"><IncidentObjectBadges labels={eventLabels} /></div>
                    </div>
                  </SnapshotImage>
                </button>
              );
            })}
            {mosaic.pageCount > 1 ? (
              <div className="incident-mosaic-pager" onClick={(event) => event.stopPropagation()}>
                <button type="button" onClick={() => setMosaicPageIndex((page) => Math.max(0, page - 1))} disabled={mosaic.page === 0} aria-label="Previous mosaic events"><ChevronLeft size={15} /></button>
                <span>{mosaic.page + 1} / {mosaic.pageCount}</span>
                <button type="button" onClick={() => setMosaicPageIndex((page) => Math.min(mosaic.pageCount - 1, page + 1))} disabled={mosaic.page >= mosaic.pageCount - 1} aria-label="Next mosaic events"><ChevronRight size={15} /></button>
              </div>
            ) : null}
          </div>
        ) : activeWorkspaceView === "evidence" ? (
          <EvidenceTimeline
            incident={incident}
            hero={evidenceHero}
            timeline={evidenceTimeline}
            timeZone={timeZone}
            stripRef={evidenceStripRef}
            onSelect={selectEvidenceFrame}
            onKeyDown={onEvidenceKeyDown}
            onPlay={openPreview}
            playLabel={scenePlayback ? "Play whole incident video" : "Play selected event video"}
          />
        ) : (
          <SnapshotImage
            event={preview}
            alt="incident snapshot"
            zoom={desktopWorkspace && expanded ? snapshotZoom : null}
            highQualityZoom={desktopWorkspace && expanded && snapshotZoom.scale > 1}
            objectFocusMode={(!desktopWorkspace || !expanded) ? thumbnailObjectFocus : "button"}
            objectFocusZoom={thumbnailObjectFocusZoom}
            objectFocusAspect={(!desktopWorkspace || !expanded) ? { width: 16, height: 10 } : null}
            showAnnotations={desktopWorkspace && expanded ? true : showIncidentCardAnnotations(expanded, thumbnailAnnotations)}
            showTracking={false}
            zones={desktopWorkspace && expanded ? zones : null}
            zonesVisible={zonesVisible}
            thumbnail={!desktopWorkspace || !expanded}
            selectedObjectIndex={desktopWorkspace && expanded ? selectedObjectIndex : null}
            onSelectObject={desktopWorkspace && expanded && onSelectObject ? onSelectObject : null}
            onImageSize={expanded && onImageSize ? (size) => onImageSize({
              ...size,
              eventId: Number(preview.representative_event_id || preview.id),
            }) : undefined}
          >
            {!desktopWorkspace || !expanded ? (
              <div className="incident-snapshot-hud">
                <div className="incident-snapshot-main">
                  <strong>{incident.camera_id}</strong>
                  <time>{expanded ? previewTimeText : timeText}</time>
                </div>
                <div className="pill-row compact incident-labels">
                  <IncidentObjectBadges labels={labels} />
                </div>
              </div>
            ) : null}
            {expanded && playbackBounds ? (
              <IncidentRecordingPlayer
                key={scenePlayback?.selection?.key || preview.id || "preview"}
                {...playbackBounds}
                trackingEvent={trackingPreview}
                analysisMode={analysisMode}
                depthLayer={depthLayer}
                timeZone={timeZone}
                onAnalysisStats={onAnalysisStats}
                onEnded={() => { if (scenePlayback?.clip) scenePlayback.ended(); else setInlineVideoActive(false); }}
                onClose={() => { scenePlayback?.stop(); setInlineVideoActive(false); }}
              />
            ) : null}
            {desktopWorkspace
              ? (!expanded ? <IncidentSourceDot trigger={triggerLabel} className="event-count" ariaLabel={`${triggerTitle}. ${countText}`} title={`${triggerTitle} · ${countText}`} /> : null)
              : <IncidentSourceDot trigger={triggerLabel} className="event-count" onClick={openOverlay} ariaLabel={`Open ${triggerTitle.toLowerCase()} incident`} title={`${triggerTitle} · Open incident`} />}
          </SnapshotImage>
        )}
        {!expanded ? <button type="button" className="incident-card-open media-surface-action" onClick={toggle} aria-label={`Open ${incident.camera_id} incident at ${timeText}`} /> : null}
        {expanded && activeWorkspaceView === "focus" && !videoActive && snapshotZoom.scale <= 1 ? (
          <button type="button" className="incident-preview-media-action media-surface-action" onClick={openPreview} aria-label={scenePlayback ? "Play whole incident video" : "Play selected event video"} />
        ) : null}
        {canShowMosaic || canShowEvidence || onReturnToSelected || showZoneToggle ? (
          <div className="incident-workspace-chrome" onClick={(event) => event.stopPropagation()}>
            {canShowMosaic || canShowEvidence || showZoneToggle ? (
              <div className="incident-workspace-view-toggle" role="group" aria-label="Incident image layout">
                {canShowMosaic || canShowEvidence ? <button type="button" className={activeWorkspaceView === "focus" ? "active" : ""} onClick={() => selectWorkspaceView("focus")} aria-pressed={activeWorkspaceView === "focus"} title="Focus selected event"><Crop size={14} /><span>Focus</span></button> : null}
                {canShowMosaic ? <button type="button" className={activeWorkspaceView === "mosaic" ? "active" : ""} onClick={() => selectWorkspaceView("mosaic")} aria-pressed={activeWorkspaceView === "mosaic"} title="Show all incident events"><Grid2X2 size={14} /><span>Mosaic</span></button> : null}
                {canShowEvidence ? <button type="button" className={activeWorkspaceView === "evidence" ? "active" : ""} onClick={() => selectWorkspaceView("evidence")} aria-pressed={activeWorkspaceView === "evidence"} title="Show every event image on a timeline"><Images size={14} /><span>Evidence</span></button> : null}
                {showZoneToggle ? <button type="button" className={zonesVisible ? "active" : ""} aria-pressed={zonesVisible} title={zonesVisible ? "Hide zones" : "Show zones"} aria-label={zonesVisible ? "Hide zones" : "Show zones"} onClick={() => setZonesVisible((current) => !current)}><Pentagon size={14} /><span>Zones</span></button> : null}
              </div>
            ) : null}
            {onReturnToSelected ? (
              <button type="button" className="incident-return-selected" onClick={onReturnToSelected}>
                Return to selected incident
              </button>
            ) : null}
          </div>
        ) : null}
      </div>
    </article>
  );
}

export function VisualSimilarIncidents({
  active = true,
  anchorEventId,
  visualAnchorEventId = anchorEventId,
  appearanceAnchorEventId = anchorEventId,
  objectIndex,
  objectLabel,
  trackId = null,
  event = null,
  cameraNameById,
  timeZone,
  onSelect,
  loadingEventId,
  selectedEventId,
  onClear,
}) {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [usedAppearance, setUsedAppearance] = useState(false);
  const [usedVisual, setUsedVisual] = useState(false);
  const resolvedTrackId = Number.isInteger(Number(trackId)) && Number(trackId) > 0
    ? Number(trackId)
    : resolveObjectTrackId({ label: objectLabel, track_id: trackId }, event);
  const preferAppearance = appearanceCapableLabel(objectLabel);
  const searchActive = Boolean(active)
    && Number.isInteger(Number(anchorEventId)) && Number(anchorEventId) > 0
    && isValidObjectIndex(objectIndex);

  useEffect(() => {
    if (!searchActive) {
      setResults([]);
      setError("");
      setUsedAppearance(false);
      setUsedVisual(false);
      setLoading(false);
      return undefined;
    }
    const controller = new AbortController();
    let cancelled = false;
    setLoading(true);
    setError("");
    setUsedAppearance(false);
    setUsedVisual(false);

    async function load() {
      const appearancePromise = preferAppearance
        ? fetch(appUrl(appearanceMatchesPath(appearanceAnchorEventId, {
          hours: 24,
          limit: 12,
          trackId: resolvedTrackId,
          crossCameraOnly: false,
        })), { signal: controller.signal })
          .then(async (response) => {
            const payload = await response.json().catch(() => ({}));
            if (!response.ok) {
              throw new Error(payload.detail || "Appearance search unavailable");
            }
            return Array.isArray(payload.matches) ? payload.matches : [];
          })
          .catch((requestError) => {
            if (requestError?.name === "AbortError") throw requestError;
            return { error: requestError.message || "Appearance search unavailable", matches: [] };
          })
        : Promise.resolve({ skipped: true, matches: [] });

      const visualPromise = fetch(appUrl("/api/semantic-search/visual"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify(visualSearchRequest({
          eventId: visualAnchorEventId,
          objectIndex,
          limit: 16,
        })),
      })
        .then(async (response) => {
          const payload = await response.json().catch(() => ({}));
          if (!response.ok) {
            throw new Error(payload.detail || "Visual search unavailable");
          }
          return Array.isArray(payload.results) ? payload.results : [];
        })
        .catch((requestError) => {
          if (requestError?.name === "AbortError") throw requestError;
          return { error: requestError.message || "Visual search unavailable", results: [] };
        });

      const [appearanceOutcome, visualOutcome] = await Promise.all([appearancePromise, visualPromise]);
      if (cancelled) return;

      const appearanceMatches = Array.isArray(appearanceOutcome)
        ? appearanceOutcome
        : (appearanceOutcome.matches || []);
      const visualResults = Array.isArray(visualOutcome)
        ? visualOutcome
        : (visualOutcome.results || []);
      const appearanceError = appearanceOutcome?.error || "";
      const visualError = visualOutcome?.error || "";
      const appearanceOk = preferAppearance && !appearanceOutcome?.skipped && !appearanceError;
      const visualOk = !visualError;
      const merged = mergeHybridFindSimilarResults({
        appearanceMatches: appearanceOk ? appearanceMatches : [],
        visualResults: visualOk ? visualResults : [],
        limit: 16,
      });
      setUsedAppearance(appearanceOk && appearanceMatches.some((match) => match?.visually_similar));
      setUsedVisual(visualOk && visualResults.length > 0);
      setResults(merged);
      if (!merged.length) {
        const messages = [appearanceError, visualError].filter(Boolean);
        setError(messages[0] || "");
      } else {
        setError("");
      }
    }

    load().catch((requestError) => {
      if (!cancelled && requestError?.name !== "AbortError") {
        setResults([]);
        setError(requestError.message || "Find similar unavailable");
      }
    }).finally(() => {
      if (!cancelled) setLoading(false);
    });

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [searchActive, visualAnchorEventId, appearanceAnchorEventId, objectIndex, objectLabel, preferAppearance, resolvedTrackId]);

  if (!searchActive) return null;

  return (
    <section className="incident-related incident-visual-similar">
      <div className="incident-related-head">
        <div>
          <h3>Find similar</h3>
          <small>{hybridFindSimilarSubtitle({
            objectLabel,
            usedAppearance,
            usedVisual,
            trackId: resolvedTrackId,
          })}</small>
        </div>
        {onClear ? <button type="button" onClick={onClear}>Clear</button> : null}
      </div>
      {loading ? <p>Searching appearance and visual indexes…</p> : null}
      {error ? <p className="incident-visual-similar-error">{error}</p> : null}
      {!loading && !error && !results.length ? <p>No similar incidents in the indexes yet.</p> : null}
      {results.length ? (
        <div className="incident-related-grid">
          {results.map((result) => {
            const eventRow = result.event || {};
            const eventId = Number(eventRow.id);
            const selected = eventId === Number(selectedEventId);
            const pending = eventId === Number(loadingEventId);
            const context = incidentRecordingContext(eventRow);
            const matchLabel = hybridMatchLabel(result);
            const modeLabel = result.query_mode === "appearance" ? "Appearance" : "Visual";
            return (
              <div className={`incident-visual-similar-card${selected ? " selected" : ""}`} key={`${result.query_mode}-${eventId}`}>
                <button
                  type="button"
                  className={selected ? "selected" : ""}
                  onClick={() => onSelect?.(eventRow)}
                  disabled={pending}
                  aria-pressed={selected}
                  title={`${modeLabel}: ${matchLabel} · ${cameraNameById.get(eventRow.camera_id) || eventRow.camera_id}`}
                >
                  <img
                    src={appUrl(relatedIncidentThumbnailPath(eventId))}
                    alt={`${cameraNameById.get(eventRow.camera_id) || eventRow.camera_id} similar incident`}
                    loading="lazy"
                  />
                  <strong>{cameraNameById.get(eventRow.camera_id) || eventRow.camera_id}</strong>
                  <small>
                    <span className={`incident-visual-mode ${result.query_mode || "visual"}`}>{modeLabel}</span>
                    {pending ? "Loading…" : `${matchLabel} · ${formatDateTime(eventRow.created_at, timeZone)}`}
                  </small>
                </button>
                {context ? (
                  <a
                    className="incident-visual-similar-timeline"
                    href={recordingsHref(context, {
                      trailEventIds: results.map((item) => Number(item?.event?.id)).filter((id) => Number.isInteger(id) && id > 0),
                    })}
                    onClick={() => {
                      writeVisualSearchTrail(window.sessionStorage, {
                        eventIds: results.map((item) => Number(item?.event?.id)),
                        hits: results,
                      });
                    }}
                  >
                    <Play size={13} /> Timeline
                  </a>
                ) : null}
              </div>
            );
          })}
        </div>
      ) : null}
    </section>
  );
}

export function RelatedAppearanceIncidents({
  anchorEventId,
  selectedEventId,
  loadingEventId,
  cameraNameById,
  timeZone,
  onSelect,
  onReturn,
}) {
  const [matches, setMatches] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!Number.isInteger(Number(anchorEventId)) || Number(anchorEventId) <= 0) {
      setMatches([]);
      setLoading(false);
      setError(false);
      return undefined;
    }
    const controller = new AbortController();
    let cancelled = false;
    setLoading(true);
    setError(false);
    setMatches([]);
    fetch(appUrl(relatedIncidentsPath(anchorEventId)), {
      signal: controller.signal,
    })
      .then((response) => response.ok ? response.json() : Promise.reject(new Error("Related incidents unavailable")))
      .then((payload) => {
        if (!cancelled) setMatches(visibleRelatedAppearances(payload, anchorEventId, 8));
      })
      .catch((requestError) => {
        if (!cancelled && requestError?.name !== "AbortError") {
          setMatches([]);
          setError(true);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [anchorEventId]);

  return (
    <section className="incident-related">
      <div className="incident-related-head">
        <h3>Related incidents</h3>
        {selectedEventId ? <button type="button" onClick={onReturn}>Selected incident</button> : null}
      </div>
      {loading ? <p>Finding related incidents…</p> : null}
      {!loading && error ? <p>Related incidents are unavailable.</p> : null}
      {!loading && !error && !matches.length ? <p>No related incidents were found in this time window.</p> : null}
      {matches.length ? <div className="incident-related-grid">
        {matches.map((match) => {
          const eventId = Number(match.event_id);
          const selected = eventId === Number(selectedEventId);
          const pending = eventId === Number(loadingEventId);
          const reasons = relatedEvidenceReasons(match);
          const relatedTitle = `${cameraNameById.get(match.camera_id) || match.camera_id}: ${reasons.map((reason) => reason.label).join("; ")}`;
          return (
            <button type="button" className={selected ? "selected" : ""} key={eventId} onClick={() => onSelect(match)} disabled={pending} aria-pressed={selected} title={relatedTitle} aria-label={relatedTitle}>
              <img src={appUrl(relatedIncidentThumbnailPath(eventId))} alt={`${cameraNameById.get(match.camera_id) || match.camera_id} related incident`} loading="lazy" />
              <span className="incident-related-reasons">
                {reasons.map(({ kind, label }) => {
                  const Icon = { appearance: ScanEye, route: Route, time: Clock3, related: Link2 }[kind];
                  return <span key={kind} title={label} role="img" aria-label={label}><Icon size={12} aria-hidden="true" /></span>;
                })}
              </span>
              <strong>{cameraNameById.get(match.camera_id) || match.camera_id}</strong>
              <small>{pending ? "Loading…" : formatDateTime(match.created_at, timeZone)}</small>
            </button>
          );
        })}
      </div> : null}
    </section>
  );
}

export function CrossCameraTracePanel({
  anchorEventId,
  cameraNameById,
  timeZone,
  onSelect,
  loadingEventId,
}) {
  const [trace, setTrace] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!open) {
      setTrace(null);
      setError("");
      setLoading(false);
      return undefined;
    }
    if (!Number.isInteger(Number(anchorEventId)) || Number(anchorEventId) <= 0) {
      setTrace(null);
      return undefined;
    }
    const controller = new AbortController();
    let cancelled = false;
    setLoading(true);
    setError("");
    fetch(appUrl(crossCameraTracePath(anchorEventId, { time_zone: timeZone })), {
      signal: controller.signal,
    })
      .then((response) => response.ok ? response.json() : response.json().then((payload) => Promise.reject(new Error(payload.detail || "Cross-camera trace unavailable"))))
      .then((payload) => {
        if (!cancelled) setTrace(payload);
      })
      .catch((requestError) => {
        if (!cancelled && requestError?.name !== "AbortError") {
          setTrace(null);
          setError(requestError.message || "Cross-camera trace unavailable");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [anchorEventId, open, timeZone]);

  if (!Number.isInteger(Number(anchorEventId)) || Number(anchorEventId) <= 0) return null;

  return (
    <section className="incident-cross-camera-trace">
      <div className="incident-related-head">
        <div>
          <h3>Cross-camera trace</h3>
          <small>Chronological matches across cameras without using the assistant.</small>
        </div>
        <button type="button" onClick={() => setOpen((current) => !current)}>
          {open ? "Hide trace" : "Trace across cameras"}
        </button>
      </div>
      {open && loading ? <p>Finding cross-camera matches…</p> : null}
      {open && error ? <p className="incident-cross-camera-trace-error">{error}</p> : null}
      {open && trace ? <>
        <p className="incident-cross-camera-trace-summary">{trace.summary}</p>
        <div className="assistant-timeline incident-cross-camera-timeline">
          {trace.matches?.length ? trace.matches.map((match) => {
            const eventId = Number(match.event_id);
            const pending = eventId === Number(loadingEventId);
            return (
              <button
                type="button"
                className="assistant-timeline-link"
                key={eventId}
                disabled={pending}
                onClick={() => onSelect?.(match)}
                title={`Open incident from ${crossCameraMatchCameraLabel(match, cameraNameById)}`}
              >
                <span>{formatDateTime(match.start_at, timeZone)}</span>
                <strong>{crossCameraMatchCameraLabel(match, cameraNameById)}</strong>
                <small>{pending ? "Loading…" : crossCameraMatchLabel(match)}</small>
              </button>
            );
          }) : <small>No related incidents were found in this time window.</small>}
        </div>
        {trace.limitations?.[3] ? <p className="incident-cross-camera-trace-limitations">{trace.limitations[3]}</p> : null}
      </> : null}
    </section>
  );
}

export function IncidentInspector({ open = false, incident, faceEvent, searchEvent = null, anchorEventId, visualAnchorEventId = anchorEventId, appearanceAnchorEventId = anchorEventId, selectedRelatedEventId, relatedLoadingEventId, relatedError = "", cameraNameById, appConfig, timeZone, imageSize, analysisMode = "clean", depthLayer = "both", analysisStats, analysisPanel = null, selectedObjectIndex = null, findSimilarObjectIndex = null, onSelectObject = null, onFindSimilar = null, onAnalysisModeChange, onDepthLayerChange, onFaceOpen, onRelatedSelect, onRelatedReturn, onClose, onAskAssistant = null }) {
  const inspectorRef = useRef(null);
  useEffect(() => {
    if (!open) return undefined;
    const inspector = inspectorRef.current;
    const focusable = () => [...(inspector?.querySelectorAll('button:not([disabled]), a[href], summary, [tabindex]:not([tabindex="-1"])') || [])]
      .filter((element) => element.offsetParent !== null);
    window.requestAnimationFrame(() => focusable()[0]?.focus());
    function containFocus(event) {
      if (event.key !== "Tab") return;
      const items = focusable();
      if (!items.length) return;
      const first = items[0];
      const last = items[items.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
    inspector?.addEventListener("keydown", containFocus);
    return () => inspector?.removeEventListener("keydown", containFocus);
  }, [open]);
  if (!incident) return <aside id="incident-inspector" className={`incident-inspector${open ? " open" : ""}`}><div className="empty-state">Select an incident.</div></aside>;
  const inspectedEvent = faceEvent || incident;
  const findSimilarSourceEvent = searchEvent || inspectedEvent;
  const searchableObjects = visualSearchObjects(findSimilarSourceEvent);
  const objects = searchableObjects;
  const findSimilarActive = isValidObjectIndex(findSimilarObjectIndex);
  const selectedSearchObject = findSimilarActive
    ? searchableObjects[Number(findSimilarObjectIndex)]
    : (isValidObjectIndex(selectedObjectIndex) ? searchableObjects[Number(selectedObjectIndex)] : null);
  const selectedTrackId = resolveObjectTrackId(selectedSearchObject, findSimilarSourceEvent);
  const incidentTracking = incidentReplayTracking(inspectedEvent, incident)?.object_tracking;
  const objectTracks = incidentTracking?.tracks || [];
  const trackingWindowSeconds = Number(incidentTracking?.window_end_epoch) - Number(incidentTracking?.window_start_epoch);
  const analyzedSeconds = incidentTracking?.analyzed_through
    ? Date.parse(incidentTracking.analyzed_through) / 1000 - Number(incidentTracking.window_start_epoch) : 0;

  const faces = faceEvent?.faces || [];
  const eventId = Number(inspectedEvent.representative_event_id || inspectedEvent.id);
  const findSimilarAnchorId = Number(
    anchorEventId
    || findSimilarSourceEvent?.representative_event_id
    || findSimilarSourceEvent?.id,
  );
  const before = Number(appConfig?.event_clip_before_seconds ?? 5);
  const after = Number(appConfig?.event_clip_after_seconds ?? 5);
  const depthConfigured = Boolean(
    appConfig?.detector?.depth?.enabled
    && String(appConfig?.detector?.depth?.model_path || "").trim(),
  );
  // Do not name this `window` — it shadows the DOM global and crashes the
  // open-focus effect (requestAnimationFrame) when Find similar / Details opens.
  const clipWindow = incidentClipWindow(incident, before, after);
  const clipUrl = Number.isFinite(eventId) ? eventClipUrl(eventId, clipWindow.before, clipWindow.after) : "";
  const observedObjects = incidentSceneObjects(incident);

  return (
    <aside ref={inspectorRef} id="incident-inspector" className={`incident-inspector${open ? " open" : ""}`} role={open ? "dialog" : undefined} aria-modal={open ? "true" : undefined} aria-labelledby={open ? "incident-inspector-title" : undefined}>
      <div className="incident-inspector-head">
        <div><strong id="incident-inspector-title">{cameraNameById.get(incident.camera_id) || incident.camera_id}</strong><time>{formatDateTime(inspectedEvent.created_at || incident.created_at, timeZone)}</time></div>
        {onClose ? <button type="button" className="incident-inspector-close" onClick={onClose} aria-label="Close incident details"><X size={17} /></button> : null}
      </div>
      <section className="incident-observed-objects" aria-label="Observed objects">
        <h3>Observed objects</h3>
        {observedObjects.length ? <ul>{observedObjectSummaries(observedObjects).map((detection) => (
          <li key={detection.label}><span>{detection.label}</span><span>{Math.round(detection.confidence * 100)}%</span></li>
        ))}</ul> : <p>No object observations are available.</p>}
        {onFindSimilar && searchableObjects.length ? <div className="incident-find-similar-actions" role="group" aria-label="Find similar objects">
          {searchableObjects.map((object, index) => <button
            type="button"
            key={`${object.label}-${index}`}
            aria-pressed={findSimilarActive && Number(findSimilarObjectIndex) === index}
            onClick={() => onFindSimilar({
              eventId: Number(findSimilarSourceEvent?.representative_event_id || findSimilarSourceEvent?.id),
              objectIndex: index,
              label: object.label,
              trackId: resolveObjectTrackId(object, findSimilarSourceEvent),
            })}
          >Find similar: {object.label}{searchableObjects.length > 1 ? ` (object ${index + 1})` : ""}</button>)}
        </div> : null}
      </section>
      <section className="incident-replay-analysis">
        <h3>Replay analysis</h3>
        <div className="incident-analysis-modes" role="group" aria-label="Replay analysis mode">
          <button type="button" className={analysisMode === "clean" ? "active" : ""} aria-pressed={analysisMode === "clean"} onClick={() => onAnalysisModeChange("clean")} title="Replay without an analysis overlay"><Play size={14} /> Clean</button>
          <button type="button" className={analysisMode === "tracks" ? "active" : ""} aria-pressed={analysisMode === "tracks"} onClick={() => onAnalysisModeChange("tracks")} disabled={!incidentTracking?.state && !objectTracks.length} title={incidentTracking?.state || objectTracks.length ? "Replay stored object tracks" : "No stored tracks for this incident"}><ListTree size={14} /> Tracks</button>
          <button type="button" className={analysisMode === "ai" ? "active" : ""} aria-pressed={analysisMode === "ai"} onClick={() => onAnalysisModeChange("ai")} title="Run OpenVINO detection while replaying"><Activity size={14} /> AI</button>
          <button type="button" className={analysisMode === "depth" ? "active" : ""} aria-pressed={analysisMode === "depth"} onClick={() => onAnalysisModeChange("depth")} disabled={!depthConfigured} title={depthConfigured ? "Run detection with monocular depth while replaying" : "Enable depth estimation in Intelligence settings"}><Layers size={14} /> Depth</button>
        </div>
        {analysisMode === "depth" ? (
          <div className="incident-depth-layers" role="group" aria-label="Depth overlay layers">
            <button type="button" className={depthLayer === "both" ? "active" : ""} aria-pressed={depthLayer === "both"} onClick={() => onDepthLayerChange?.("both")} title="Show heatmap and distance boxes">Both</button>
            <button type="button" className={depthLayer === "boxes" ? "active" : ""} aria-pressed={depthLayer === "boxes"} onClick={() => onDepthLayerChange?.("boxes")} title="Show distance boxes only">Boxes</button>
            <button type="button" className={depthLayer === "heatmap" ? "active" : ""} aria-pressed={depthLayer === "heatmap"} onClick={() => onDepthLayerChange?.("heatmap")} title="Show depth heatmap only">Heatmap</button>
          </div>
        ) : null}
        {analysisMode === "tracks" && trackingWindowSeconds > 0 ? <progress
          className="incident-tracking-progress" aria-label="Tracking analysis coverage"
          max={trackingWindowSeconds} value={Math.max(0, Math.min(trackingWindowSeconds, analyzedSeconds))}
        /> : null}
        {analysisMode === "tracks" ? <small>{trackingCoverageLabel(incidentTracking)}{incidentTracking?.analyzed_through ? ` · analyzed through ${formatTimeOnly(incidentTracking.analyzed_through, timeZone)}` : ""} · {objectTracks.length} stored track{objectTracks.length === 1 ? "" : "s"} · {Number(incidentTracking?.sample_fps || 0) || "?"} FPS</small> : null}
        {analysisMode === "ai" && analysisStats ? <small className={analysisStats.error ? "analysis-error" : ""}>{analysisStats.error || `${analysisStats.inferenceMs ?? "--"} ms · ${analysisStats.objects ?? 0} current objects`}</small> : null}
        {analysisMode === "depth" && analysisStats ? (
          <small className={analysisStats.error || analysisStats.depthError ? "analysis-error" : ""}>
            {analysisStats.error
              || analysisStats.depthError
              || `${analysisStats.inferenceMs ?? "--"} ms (${analysisStats.detectMs ?? "--"} detect · ${analysisStats.depthMs ?? "--"} depth) · ${analysisStats.objects ?? 0} objects`}
            {analysisStats.heatmapRange ? ` · ${formatDepthMeters(analysisStats.heatmapRange.min_m)}–${formatDepthMeters(analysisStats.heatmapRange.max_m)}` : ""}
          </small>
        ) : null}
      </section>
      <section>
        <h3>Faces</h3>
        {faces.length ? faces.map((face, index) => (
          <button type="button" className={`inspector-face ${face.status || "unknown"}`} key={`${face.status}-${face.name}-${index}`} onClick={() => onFaceOpen(face)}>
            <strong>{face.name || "Unknown"}</strong>
            <span>{face.status === "automatic" ? "Automatic · " : ""}{Math.round(Number(face.confidence || 0) * 100)}%{Number(face.candidate_count || 0) > 1 ? ` · ${face.candidate_count} frames` : ""}</span>
          </button>
        )) : <p>No recognized faces.</p>}
      </section>
      {analysisPanel ? <section className="incident-inspector-extra-analysis">
        {analysisPanel}
      </section> : null}
      {relatedError ? <p role="alert">{relatedError}</p> : null}
      <RelatedAppearanceIncidents anchorEventId={anchorEventId} selectedEventId={selectedRelatedEventId} loadingEventId={relatedLoadingEventId} cameraNameById={cameraNameById} timeZone={timeZone} onSelect={onRelatedSelect} onReturn={onRelatedReturn} />
      <VisualSimilarIncidents
        active={findSimilarActive}
        anchorEventId={Number.isInteger(findSimilarAnchorId) && findSimilarAnchorId > 0 ? findSimilarAnchorId : null}
        visualAnchorEventId={visualAnchorEventId}
        appearanceAnchorEventId={appearanceAnchorEventId}
        objectIndex={findSimilarObjectIndex}
        objectLabel={selectedSearchObject?.label || ""}
        trackId={selectedTrackId}
        event={findSimilarSourceEvent}
        cameraNameById={cameraNameById}
        timeZone={timeZone}
        onSelect={onRelatedSelect}
        loadingEventId={relatedLoadingEventId}
        selectedEventId={selectedRelatedEventId}
        onClear={onFindSimilar ? () => onFindSimilar(null) : (onSelectObject ? () => onSelectObject(null) : null)}
      />
      <CrossCameraTracePanel anchorEventId={open ? anchorEventId : null} cameraNameById={cameraNameById} timeZone={timeZone} onSelect={onRelatedSelect} loadingEventId={relatedLoadingEventId} />
      <details className="incident-technical-details">
        <summary>Technical details</summary>
        <div className="incident-technical-body">
          {objects.length ? <div className="incident-technical-objects">{objects.map((object, index) => {
            const box = object.box || {};
            return <code key={`${object.label}-${index}`}>{object.label}: {Math.round(Number(box.x1 || 0))}, {Math.round(Number(box.y1 || 0))} → {Math.round(Number(box.x2 || 0))}, {Math.round(Number(box.y2 || 0))}</code>;
          })}</div> : null}
          <dl>
            <div><dt>Events</dt><dd>{incident.event_count || incident.events?.length || 1}</dd></div>
            <div><dt>Selected trigger</dt><dd>{incidentTriggerLabel(inspectedEvent)}</dd></div>
            <div><dt>Additional motion</dt><dd>{incident.motion_observation_count || incident.motion_observations?.length || 0}</dd></div>
            <div><dt>Duration</dt><dd>{formatDuration(incident.duration_seconds || 0)}</dd></div>
            <div><dt>Start</dt><dd>{formatTimeOnly(incident.start_at || incident.created_at, timeZone)}</dd></div>
            <div><dt>End</dt><dd>{formatTimeOnly(incident.end_at || incident.created_at, timeZone)}</dd></div>
            <div><dt>Loaded image</dt><dd>{imageSize?.width && imageSize?.height ? `${imageSize.width} × ${imageSize.height} px` : "—"}</dd></div>
          </dl>
        </div>
      </details>
      <div className="incident-inspector-actions">
        {clipUrl ? <a href={clipUrl} download={`survng-${incident.camera_id}-${eventId}.mp4`}><Download size={15} /> Video</a> : null}
        {inspectedEvent.snapshot_path && eventSnapshotDownloadUrl(inspectedEvent) ? <a href={eventSnapshotDownloadUrl(inspectedEvent)}><Download size={15} /> Snapshot</a> : null}
        {onAskAssistant ? <button type="button" className="incident-ask-assistant" onClick={() => onAskAssistant("Analyze this incident")}><Sparkles size={15} /> Ask assistant</button> : null}
      </div>
    </aside>
  );
}
