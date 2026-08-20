import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  ChevronLeft,
  ChevronRight,
  Crop,
  Download,
  Grid2X2,
  Images,
  ListTree,
  Play,
  X,
} from "lucide-react";
import { incidentTrackingSource, storedObjectTracks } from "../objectTrackReplay.mjs";
import { incidentEvidenceFrames, incidentMosaicEvents, incidentMosaicPage, incidentTriggerLabel, showIncidentCardAnnotations } from "../incidentNavigation.mjs";
import { relatedEvidenceLabel, relatedIncidentThumbnailPath, relatedIncidentsPath, visibleRelatedAppearances } from "../relatedIncidents.mjs";
import { appUrl, fetch } from "../shared/api.js";
import { formatDateTime, formatTimeOnly, formatDuration } from "../shared/format.js";
import { eventSnapshotDownloadUrl, eventClipUrl } from "../shared/mediaUrls.js";
import { ShakaVideo } from "../shared/media.jsx";
import {
  DebugDetectionOverlay,
  IncidentObjectBadges,
  IncidentSourceDot,
  SnapshotImage,
  StoredTrackVideoOverlay,
  eventObjects,
  hasDetectedObjects,
  incidentClipWindow,
  incidentLabels,
  incidentZones,
  loadIncidentClipInfo,
} from "../shared/evidence.jsx";

export function IncidentClipLayer({ event, trackingEvent, active, analysisMode = "clean", onAnalysisStats, onEnded }) {
  const videoRef = useRef(null);
  const [clipInfo, setClipInfo] = useState(null);
  const [clipLoading, setClipLoading] = useState(false);
  const [clipError, setClipError] = useState("");
  const [playback, setPlayback] = useState(null);
  const [playbackOriginTime, setPlaybackOriginTime] = useState(null);
  const storedTracks = storedObjectTracks(trackingEvent || event);

  useEffect(() => {
    let cancelled = false;
    async function loadClipSettings() {
      const eventId = Number(event?.representative_event_id || event?.id);
      if (!active || !Number.isFinite(eventId)) {
        setClipInfo(null);
        setPlayback(null);
        setPlaybackOriginTime(null);
        setClipLoading(false);
        setClipError(active ? "No event video available" : "");
        return;
      }
      setClipInfo(null);
      setPlayback(null);
      setPlaybackOriginTime(null);
      setClipLoading(true);
      setClipError("");
      const info = await loadIncidentClipInfo(event, () => cancelled);
      if (!info) return;
      setClipInfo(info);
      setPlayback({ url: info.streamUrl, mimeType: "application/vnd.apple.mpegurl" });
    }
    loadClipSettings();
    return () => { cancelled = true; };
  }, [active, event?.id, event?.representative_event_id, event?.start_epoch, event?.last_epoch]);

  if (!active) return null;
  return (
    <div className="incident-video-layer" onClick={(event) => event.stopPropagation()}>
      {clipInfo && playback && !clipError ? (
        <>
          <ShakaVideo
            ref={videoRef}
            src={playback.url}
            mimeType={playback.mimeType}
            autoPlay
            controls
            playsInline
            preload="metadata"
            onReady={(_player, video) => {
              setPlaybackOriginTime(0);
              if (clipInfo.playbackStartOffset > 0 && video) {
                const seekToIncident = () => {
                  const targetTime = clipInfo.playbackStartOffset;
                  video.currentTime = Number.isFinite(video.duration)
                    ? Math.min(targetTime, Math.max(0, video.duration - 0.25))
                    : targetTime;
                };
                if (video.paused) video.addEventListener("playing", seekToIncident, { once: true });
                else seekToIncident();
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
                setClipError("No recording window found");
              }
            }}
            onEnded={onEnded}
          />
          {analysisMode === "tracks" && storedTracks.length ? (
            <StoredTrackVideoOverlay
              videoRef={videoRef}
              tracks={storedTracks}
              coordinateSize={{
                width: Number(trackingEvent?.object_tracking?.frame_width),
                height: Number(trackingEvent?.object_tracking?.frame_height),
              }}
              windowStartEpoch={clipInfo.windowStartEpoch}
              mediaStartTime={playbackOriginTime}
              mediaKey={playback.url}
              sampleFps={trackingEvent?.object_tracking?.sample_fps}
              lostTimeoutSeconds={trackingEvent?.object_tracking?.lost_timeout_seconds}
            />
          ) : null}
          <DebugDetectionOverlay videoRef={videoRef} active={analysisMode === "ai"} confidence={0.35} onStats={onAnalysisStats} />
          {clipLoading ? <div className="incident-video-status preparing">Preparing incident video...</div> : null}
        </>
      ) : (
        <div className="incident-video-status">{clipLoading ? "Preparing video..." : clipError || "No event video available"}</div>
      )}
    </div>
  );
}

export function IncidentCard({ incident, timeZone, expanded, selected = false, thumbnailAnnotations = true, desktopWorkspace = false, analysisMode = "clean", replayRequest = 0, onAnalysisStats, onToggle, onSelect, onPreviewChange, onImageSize }) {
  const rawEvents = incident.events || [];
  const motionObservations = incident.motion_observations || [];
  const showSubEvents = rawEvents.length > 1 || motionObservations.length > 0;
  const [selectedPreview, setSelectedPreview] = useState(null);
  const [workspaceView, setWorkspaceView] = useState(() => {
    try {
      const stored = window.sessionStorage.getItem("survng.incidentWorkspaceView.v1");
      return ["mosaic", "evidence"].includes(stored) ? stored : "focus";
    } catch { return "focus"; }
  });
  const [selectedEvidence, setSelectedEvidence] = useState(null);
  const [mosaicPageIndex, setMosaicPageIndex] = useState(0);
  const [subEventsOpen, setSubEventsOpen] = useState(false);
  const [inlineVideoActive, setInlineVideoActive] = useState(false);
  const [snapshotZoom, setSnapshotZoom] = useState({ scale: 1, x: 0, y: 0 });
  const previewRef = useRef(null);
  const snapshotZoomRef = useRef(snapshotZoom);
  const panGestureRef = useRef({ pointerId: null, startX: 0, startY: 0, panX: 0, panY: 0, moved: false });
  const replayRequestRef = useRef(replayRequest);
  const evidenceSource = selectedPreview || incident;
  const evidenceFrames = useMemo(() => incidentEvidenceFrames(evidenceSource), [evidenceSource]);
  const evidenceItems = useMemo(() => evidenceFrames.map((frame) => {
    if (frame.kind === "snapshot") return {
      ...frame,
      event: {
        ...evidenceSource,
        created_at: new Date(frame.epoch * 1000).toISOString(),
      },
    };
    const snapshotUrl = `/api/cameras/${encodeURIComponent(incident.camera_id)}/recordings/preview.jpg?epoch=${encodeURIComponent(frame.epoch)}&source=main&width=1280&exact=true`;
    return {
      ...frame,
      event: {
        ...evidenceSource,
        snapshot_path: "recording-evidence",
        snapshot_url: snapshotUrl,
        created_at: new Date(frame.epoch * 1000).toISOString(),
        objects: [],
        object_tracking: null,
      },
    };
  }), [evidenceFrames, evidenceSource, incident.camera_id]);
  const preview = selectedEvidence?.event || selectedPreview || incident;
  const mosaicEvents = useMemo(() => incidentMosaicEvents(incident), [incident]);
  const mosaic = useMemo(() => incidentMosaicPage(mosaicEvents, mosaicPageIndex), [mosaicEvents, mosaicPageIndex]);
  const canShowMosaic = desktopWorkspace && expanded && mosaicEvents.length > 1;
  const canShowEvidence = desktopWorkspace && expanded && evidenceItems.length > 0;
  const activeWorkspaceView = workspaceView === "mosaic" && !canShowMosaic
    ? "focus"
    : workspaceView === "evidence" && !canShowEvidence ? "focus" : workspaceView;
  const trackingPreview = incidentTrackingSource(preview, incident) || preview;
  const labels = incidentLabels(incident);
  const eventCount = incident.event_count || rawEvents.length || 1;
  const observationCount = Number(incident.motion_observation_count || motionObservations.length || 0);
  const countText = `${eventCount} ${eventCount === 1 ? "event" : "events"}${observationCount ? ` · ${observationCount} additional motion update${observationCount === 1 ? "" : "s"}` : ""}`;
  const triggerLabel = incidentTriggerLabel(incident);
  const triggerTitle = triggerLabel === "EMA" ? "EMA visual backup trigger" : "Camera motion trigger";
  const incidentTimeline = [
    ...rawEvents.map((event) => ({ kind: "event", item: event })),
    ...motionObservations.map((observation) => ({ kind: "activity", item: observation })),
  ].sort((left, right) => Date.parse(right.item.created_at || 0) - Date.parse(left.item.created_at || 0));
  const timeText = incident.start_at && incident.end_at && incident.start_at !== incident.end_at
    ? `${formatDateTime(incident.start_at, timeZone)} - ${formatDuration(incident.duration_seconds)}`
    : formatDateTime(incident.created_at, timeZone);
  const previewTimeText = preview.created_at ? formatDateTime(preview.created_at, timeZone) : timeText;

  useEffect(() => {
    if (!expanded) {
      setSelectedPreview(null);
      setSubEventsOpen(false);
      setInlineVideoActive(false);
    }
  }, [expanded]);

  useEffect(() => {
    setSelectedPreview(null);
    setSelectedEvidence(null);
    setMosaicPageIndex(0);
    setSubEventsOpen(false);
    setInlineVideoActive(false);
    replayRequestRef.current = replayRequest;
  }, [incident.id]);

  useEffect(() => {
    try { window.sessionStorage.setItem("survng.incidentWorkspaceView.v1", workspaceView); } catch { /* Session preference is optional. */ }
  }, [workspaceView]);

  useEffect(() => {
    setInlineVideoActive(false);
    resetSnapshotZoom();
  }, [preview.id, preview.created_at]);

  useEffect(() => {
    const previousRequest = replayRequestRef.current;
    replayRequestRef.current = replayRequest;
    if (expanded && replayRequest > previousRequest) setInlineVideoActive(true);
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
    if (expanded) setInlineVideoActive(true);
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
    setSelectedEvidence(null);
    setInlineVideoActive(false);
    setWorkspaceView("focus");
  }

  function selectEvidenceItem(item) {
    setSelectedEvidence(item);
    setInlineVideoActive(false);
    setWorkspaceView("focus");
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
                  <SnapshotImage event={event} alt="incident event snapshot" className="incident-mosaic-snapshot" progressive thumbnail allowObjectFocus={false} showAnnotations showTracking={false}>
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
          <div className={`incident-evidence incident-evidence-${evidenceItems.length}`} role="group" aria-label="Incident evidence frames">
            {evidenceItems.map((item) => (
              <button type="button" className="incident-evidence-tile" key={item.key} onClick={(event) => { event.stopPropagation(); selectEvidenceItem(item); }} aria-label={`Focus ${item.label.toLowerCase()} frame`}>
                <SnapshotImage event={item.event} alt={`${item.label} evidence frame`} className="incident-evidence-snapshot" thumbnail allowObjectFocus={false} showAnnotations={item.kind === "snapshot"} showTracking={false}>
                  <div className="incident-evidence-hud">
                    <strong>{item.label}</strong>
                    <time>{formatTimeOnly(item.event.created_at, timeZone)}</time>
                    {item.confidence > 0 ? <span>{Math.round(item.confidence * 100)}%</span> : null}
                  </div>
                </SnapshotImage>
              </button>
            ))}
          </div>
        ) : (
          <SnapshotImage
            event={preview}
            alt="incident snapshot"
            zoom={desktopWorkspace && expanded ? snapshotZoom : null}
            highQualityZoom={desktopWorkspace && expanded && snapshotZoom.scale > 1}
            showAnnotations={desktopWorkspace && expanded ? true : showIncidentCardAnnotations(expanded, thumbnailAnnotations)}
            showTracking={false}
            incidentEligibleOnly
            thumbnail={!desktopWorkspace || !expanded}
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
            <IncidentClipLayer
              event={incident}
              trackingEvent={trackingPreview}
              active={expanded && inlineVideoActive}
              analysisMode={analysisMode}
              onAnalysisStats={onAnalysisStats}
              onEnded={() => setInlineVideoActive(false)}
            />
            {desktopWorkspace
              ? (!expanded ? <IncidentSourceDot trigger={triggerLabel} className="event-count" ariaLabel={`${triggerTitle}. ${countText}`} title={`${triggerTitle} · ${countText}`} /> : null)
              : <IncidentSourceDot trigger={triggerLabel} className="event-count" onClick={openOverlay} ariaLabel={`Open ${triggerTitle.toLowerCase()} incident`} title={`${triggerTitle} · Open incident`} />}
          </SnapshotImage>
        )}
        {!expanded ? <button type="button" className="incident-card-open media-surface-action" onClick={toggle} aria-label={`Open ${incident.camera_id} incident at ${timeText}`} /> : null}
        {expanded && activeWorkspaceView === "focus" && !inlineVideoActive && snapshotZoom.scale <= 1 ? (
          <button type="button" className="incident-preview-media-action media-surface-action" onClick={openPreview} aria-label="Play selected event video" />
        ) : null}
        {canShowMosaic || canShowEvidence ? (
          <div className="incident-workspace-view-toggle" role="group" aria-label="Incident image layout" onClick={(event) => event.stopPropagation()}>
            <button type="button" className={activeWorkspaceView === "focus" ? "active" : ""} onClick={() => selectWorkspaceView("focus")} aria-pressed={activeWorkspaceView === "focus"} title="Focus selected event"><Crop size={14} /><span>Focus</span></button>
            {canShowMosaic ? <button type="button" className={activeWorkspaceView === "mosaic" ? "active" : ""} onClick={() => selectWorkspaceView("mosaic")} aria-pressed={activeWorkspaceView === "mosaic"} title="Show all incident events"><Grid2X2 size={14} /><span>Mosaic</span></button> : null}
            {canShowEvidence ? <button type="button" className={activeWorkspaceView === "evidence" ? "active" : ""} onClick={() => selectWorkspaceView("evidence")} aria-pressed={activeWorkspaceView === "evidence"} title="Compare trigger, detection, selected, and tracking frames"><Images size={14} /><span>Evidence</span></button> : null}
          </div>
        ) : null}
      </div>
      {expanded && showSubEvents ? (
        <div className="incident-meta">
          <div className="incident-detail" onClick={(event) => event.stopPropagation()}>
            <button
              className="incident-events-toggle"
              type="button"
              onClick={() => setSubEventsOpen((open) => !open)}
              aria-expanded={subEventsOpen}
            >
              <span>{countText}</span>
              <strong>{subEventsOpen ? "Hide" : "Show"}</strong>
            </button>
            {subEventsOpen ? (
              <div className="incident-events">
                {incidentTimeline.map(({ kind, item }, index) => {
                  if (kind === "activity") {
                    const activityLabel = item.reason === "event_state_cooldown" ? "motion during cooldown" : "continued motion";
                    return (
                      <div className="incident-activity-row" key={`activity-${item.id || index}`}>
                        <span>{formatTimeOnly(item.created_at || incident.created_at, timeZone)}</span>
                        <strong>{activityLabel}</strong>
                      </div>
                    );
                  }
                  const event = item;
                  const eventLabels = incidentLabels(event);
                  const eventLabelText = eventLabels.length ? eventLabels.join(", ") : "motion";
                  const isActive = (preview.id || incident.id) === event.id && (preview.created_at || incident.created_at) === event.created_at;
                  return (
                    <button type="button" key={`${event.id || "event"}-${index}`} className={isActive ? "active" : ""} onClick={() => { setSelectedPreview(event); setSelectedEvidence(null); setInlineVideoActive(false); }}>
                      <span>{formatTimeOnly(event.created_at || incident.created_at, timeZone)}</span>
                      <strong>{eventLabelText}</strong>
                    </button>
                  );
                })}
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
    </article>
  );
}

export function RelatedAppearanceIncidents({ anchorEventId, selectedEventId, loadingEventId, cameraNameById, timeZone, onSelect, onReturn }) {
  const [matches, setMatches] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!Number.isInteger(Number(anchorEventId)) || Number(anchorEventId) <= 0) {
      setMatches([]);
      return undefined;
    }
    const controller = new AbortController();
    let cancelled = false;
    setLoading(true);
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

  if (!loading && !matches.length) return null;

  return (
    <section className="incident-related">
      <div className="incident-related-head">
        <h3>Related incidents</h3>
        {selectedEventId ? <button type="button" onClick={onReturn}>Selected incident</button> : null}
      </div>
      {loading ? <p>Finding related incidents…</p> : null}
      {matches.length ? <div className="incident-related-grid">
        {matches.map((match) => {
          const eventId = Number(match.event_id);
          const selected = eventId === Number(selectedEventId);
          const pending = eventId === Number(loadingEventId);
          return (
            <button type="button" className={selected ? "selected" : ""} key={eventId} onClick={() => onSelect(match)} disabled={pending} aria-pressed={selected} title={match.route_name ? `${match.route_name}: ${relatedEvidenceLabel(match)}` : `Preview related incident from ${cameraNameById.get(match.camera_id) || match.camera_id}`}>
              <img src={appUrl(relatedIncidentThumbnailPath(eventId))} alt={`${cameraNameById.get(match.camera_id) || match.camera_id} related incident`} loading="lazy" />
              <strong>{cameraNameById.get(match.camera_id) || match.camera_id}</strong>
              <small>{pending ? "Loading…" : formatDateTime(match.created_at, timeZone)}</small>
            </button>
          );
        })}
      </div> : null}
    </section>
  );
}

export function IncidentInspector({ open = false, incident, faceEvent, anchorEventId, selectedRelatedEventId, relatedLoadingEventId, cameraNameById, appConfig, timeZone, imageSize, analysisMode = "clean", analysisStats, onAnalysisModeChange, onFaceOpen, onRelatedSelect, onRelatedReturn, onClose }) {
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
  const objects = eventObjects(inspectedEvent).filter((object) => object.label && object.incident_eligible !== false);
  const incidentTracking = incidentTrackingSource(inspectedEvent, incident)?.object_tracking;
  const objectTracks = incidentTracking?.tracks || [];
  const faces = faceEvent?.faces || [];
  const zones = incidentZones(inspectedEvent);
  const eventId = Number(inspectedEvent.representative_event_id || inspectedEvent.id);
  const before = Number(appConfig?.event_clip_before_seconds ?? 5);
  const after = Number(appConfig?.event_clip_after_seconds ?? 5);
  const window = incidentClipWindow(incident, before, after);
  const clipUrl = Number.isFinite(eventId) ? eventClipUrl(eventId, window.before, window.after) : "";

  return (
    <aside ref={inspectorRef} id="incident-inspector" className={`incident-inspector${open ? " open" : ""}`} role={open ? "dialog" : undefined} aria-modal={open ? "true" : undefined} aria-labelledby={open ? "incident-inspector-title" : undefined}>
      <div className="incident-inspector-head">
        <div><strong id="incident-inspector-title">{cameraNameById.get(incident.camera_id) || incident.camera_id}</strong><time>{formatDateTime(inspectedEvent.created_at || incident.created_at, timeZone)}</time></div>
        {onClose ? <button type="button" className="incident-inspector-close" onClick={onClose} aria-label="Close incident details"><X size={17} /></button> : null}
      </div>
      <section className="incident-current-summary">
        <h3>Current incident</h3>
        <div className="incident-summary-objects">
          {objects.length ? objects.map((object, index) => <div className="inspector-detection summary" key={`${object.label}-${index}`}><div><strong>{object.label}</strong><span>{Math.round(Number(object.confidence || 0) * 100)}%</span></div></div>) : <p>No eligible object detections.</p>}
        </div>
        <dl>
          <div><dt>Trigger</dt><dd>{incidentTriggerLabel(inspectedEvent)}</dd></div>
          <div><dt>Duration</dt><dd>{formatDuration(incident.duration_seconds || 0)}</dd></div>
          <div><dt>Zones</dt><dd>{zones.length ? zones.join(", ") : "None"}</dd></div>
        </dl>
      </section>
      <section className="incident-replay-analysis">
        <h3>Replay analysis</h3>
        <div className="incident-analysis-modes" role="group" aria-label="Replay analysis mode">
          <button type="button" className={analysisMode === "clean" ? "active" : ""} aria-pressed={analysisMode === "clean"} onClick={() => onAnalysisModeChange("clean")} title="Replay without an analysis overlay"><Play size={14} /> Clean</button>
          <button type="button" className={analysisMode === "tracks" ? "active" : ""} aria-pressed={analysisMode === "tracks"} onClick={() => onAnalysisModeChange("tracks")} disabled={!objectTracks.length} title={objectTracks.length ? "Replay stored object tracks" : "No stored tracks for this incident"}><ListTree size={14} /> Tracks</button>
          <button type="button" className={analysisMode === "ai" ? "active" : ""} aria-pressed={analysisMode === "ai"} onClick={() => onAnalysisModeChange("ai")} title="Run OpenVINO detection while replaying"><Activity size={14} /> AI</button>
        </div>
        {analysisMode === "tracks" ? <small>{objectTracks.length} stored track{objectTracks.length === 1 ? "" : "s"} · {Number(incidentTracking?.sample_fps || 0) || "?"} FPS</small> : null}
        {analysisMode === "ai" && analysisStats ? <small className={analysisStats.error ? "analysis-error" : ""}>{analysisStats.error || `${analysisStats.inferenceMs ?? "--"} ms · ${analysisStats.objects ?? 0} current objects`}</small> : null}
      </section>
      <section>
        <h3>Faces</h3>
        {faces.length ? faces.map((face, index) => (
          <button type="button" className={`inspector-face ${face.status || "unknown"}`} key={`${face.status}-${face.name}-${index}`} onClick={() => onFaceOpen(face)}>
            <strong>{face.name || "Unknown"}</strong>
            <span>{Math.round(Number(face.confidence || 0) * 100)}%{Number(face.candidate_count || 0) > 1 ? ` · ${face.candidate_count} frames` : ""}</span>
          </button>
        )) : <p>No recognized faces.</p>}
      </section>
      <RelatedAppearanceIncidents anchorEventId={anchorEventId} selectedEventId={selectedRelatedEventId} loadingEventId={relatedLoadingEventId} cameraNameById={cameraNameById} timeZone={timeZone} onSelect={onRelatedSelect} onReturn={onRelatedReturn} />
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
      </div>
    </aside>
  );
}
