import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Camera,
  Crop,
  Check,
  CircleAlert,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  Clock3,
  Download,
  Film,
  Search,
  Pause,
  Play,
  Plus,
  Radar,
  RefreshCcw,
  Save,
  ShieldCheck,
  Sparkles,
  SkipBack,
  SkipForward,
  Trash2,
  Volume2,
  VolumeX,
  X,
  Mic,
  Rows3,
  Columns3,
  FastForward,
  RotateCcw,
} from "lucide-react";
import { liveFramingStyle } from "../liveFraming.mjs";
import { browserStorage } from "../storage.mjs";
import { canSetClipBoundaryAtPlayhead, clipPreviewReachedEnd, clipRangeIsValid, setClipBoundaryAtPlayhead } from "../recordingClipSelection.mjs";
import { useVisiblePolling } from "../visibilityPolling.mjs";
import { ACTIVE_EXPORT_STATUSES, cacheExportJobs, exportIsActive, fetchExportJob, removeCachedExportJobs } from "../exportPolling.mjs";
import { nativePlaylistStart, nativePlaylistStartStep, recordingCoverageUnion, recordingPlaybackTimeline, recordingPlaybackTransport, recordingTransportOverride, supportsNativeRecordingHls, recordingSegmentAt, recordingPlayableEpoch, recordingSegmentLocalTime, recordingEpochAfterSegment, recordingEdgeSeekEpoch, recordingLiveEdgeFollow, extendedPlaybackRows, describePlaybackError, ignorePauseAfterSeekMs, isRecordingCompatibilityError, mergeRecordingAvailability, playbackRowsCoverEpoch, prefersJpegScrubPreview, recordingSeekToleranceSeconds, seekVideoToTime, seekWatchdogDelayMs, shouldResumePlaybackAfterSeek, videoReachedSeekTarget } from "../recordingPlayback.mjs";
import { useScrubStrips } from "./useScrubStrips.js";
import { ScrubFrameOverlay } from "./ScrubFrameOverlay.jsx";
import { useFrameStepper } from "./useFrameStepper.js";
import { FrameStepButtons, FrameStepOverlay, FrameStepTime } from "./FrameStepOverlay.jsx";
import { ALL_CAMERAS_LIVE_DELAY_SECONDS, allCamerasCustomIds, allCamerasGridSelection, allCamerasWindowStart, recordingGridBestEpoch } from "../recordingGrid.mjs";
import { CameraDensityControl, CameraLayoutModeControl } from "../shared/CameraLayoutControls.jsx";
import { useStoredState, useViewportQuery } from "../shared/hooks.js";
import { AllCamerasGrid } from "./AllCamerasGrid.jsx";
import { expectedTimelineCameras, filteredTimelineCameras, invalidateTimelineIdentityCache, mergeTimelineIncidentIdentity, normalizedTimelinePlaybackRate, parseTimelineView, resolveTimelineHeroCameraId, timelineEventMatchesFilter, timelineEvidenceWindow, timelineIdentityDetailEventId, timelineIncidentIncludesEvent, timelineNearbyRadiusSeconds, timelinePlayheadInComfortZone, timelineViewport, TIMELINE_PLAYBACK_RATES } from "../timelineWorkspace.mjs";
import { SpeechSearchWorkspace } from "../search/SpeechSearchResults.jsx";
import { speechSearchMode } from "../speechSettings.mjs";
import { addSemanticSearchHistory, clearSemanticSearchSession, readSemanticSearchHistory, readSemanticSearchSession, semanticSearchResultsForCamera, writeSemanticSearchHistory, writeSemanticSearchSession } from "../semanticSearchState.mjs";
import {
  clearVisualSearchTrail,
  readVisualSearchTrail,
  serializeTrailEventIds,
  trailHitForEvent,
  writeVisualSearchTrail,
} from "../visualSearchTrail.mjs";
import {
  drawContainedCropPreview,
  semanticResultThumbnailUrl,
  semanticSearchResultImageUrl,
  visualFrameCropFromPoints,
  visualFrameSearchRequest,
} from "../visualSearch.mjs";
import { appPathname, appUrl, mediaUrl, incidentRecordingContext, recordingsHref, fetch } from "../shared/api.js";
import { incidentOpenedFromHref, normalizeIncidentThumbnailObjectFocus, normalizeIncidentThumbnailObjectFocusZoom } from "../incidentNavigation.mjs";
import { ALL_RECORDING_CAMERAS_ID } from "../shared/constants.js";
import { formatDateTime, formatTimeOnly, formatBytes, formatDuration } from "../shared/format.js";
import { dateKeyForTimeZone, addDaysToDateKey, zonedDateSecondToEpoch } from "../shared/datetime.js";
import { preferredStreamSource } from "../shared/cameras.js";
import { IdentityChip } from "../shared/identity.jsx";
import { eventThumbnailUrl, recordingDayUrl, recordingWindowUrl, recordingUpdatesUrl, recordingDayHlsUrl, recordingGridDayUrl, recordingGridUpdatesUrl, recordingPreviewUrl, recordingSegmentUrl } from "../shared/mediaUrls.js";
import { CameraLanes, CatchUpStory, EventRail } from "./TimelineRail.jsx";
import { condenseStory, nextStoryEvent, storyEventAt } from "../timelineRail.mjs";
import { RecordingHlsVideo } from "../shared/RecordingHlsVideo.jsx";
import { PlaybackDiagnostics, playbackDiagnosticsEnabled } from "./PlaybackDiagnostics.jsx";
import { NativeRecordingVideo } from "../shared/NativeRecordingVideo.jsx";
import { DebugDetectionOverlay } from "../shared/evidence.jsx";
import { MobileCameraSelect } from "../shared/MobileCameraSelect.jsx";
import { usePollingData } from "../shared/polling.js";
import { useAppEvents } from "../shared/events.js";


export function mergeRecordingEvents(current, updates) {
  const byId = new Map(current.map((event) => [event.id, event]));
  updates.forEach((event) => byId.set(event.id, event));
  return [...byId.values()]
    .sort((a, b) => new Date(a.created_at) - new Date(b.created_at))
    .slice(-5000);
}

export function recordingIncidentEpoch(incident) {
  const direct = Number(incident?.start_epoch);
  if (Number.isFinite(direct)) return direct;
  const parsed = new Date(incident?.start_at || incident?.created_at || "").getTime() / 1000;
  return Number.isFinite(parsed) ? parsed : null;
}

function containedImageBounds(surface, image) {
  if (!surface || !image?.naturalWidth || !image?.naturalHeight) return null;
  const surfaceRect = surface.getBoundingClientRect();
  const imageAspect = image.naturalWidth / image.naturalHeight;
  const surfaceAspect = surfaceRect.width / Math.max(1, surfaceRect.height);
  const width = surfaceAspect > imageAspect
    ? surfaceRect.height * imageAspect
    : surfaceRect.width;
  const height = surfaceAspect > imageAspect
    ? surfaceRect.height
    : surfaceRect.width / imageAspect;
  return {
    left: (surfaceRect.width - width) / 2,
    top: (surfaceRect.height - height) / 2,
    width,
    height,
    pageLeft: surfaceRect.left + (surfaceRect.width - width) / 2,
    pageTop: surfaceRect.top + (surfaceRect.height - height) / 2,
  };
}

export function RecordingCompanionStrip({ cameras, routes, activeCameraId, source, epoch, onSelect }) {
  const previewEpoch = Math.floor(Math.max(0, Number(epoch) || 0) / 15) * 15;
  const companions = expectedTimelineCameras(cameras, routes, activeCameraId, 6);
  if (!companions.length) return null;
  return <div
    className="recording-selected-companions"
    data-camera-count={companions.length}
    aria-label="Linked camera previews"
  >
    {companions.map((camera) => <button key={camera.id} type="button" style={liveFramingStyle(camera, source)} onClick={() => onSelect(camera.id)} aria-label={`Show ${camera.name} recording at the current time`}>
      <img src={recordingPreviewUrl(camera.id, previewEpoch, source)} alt="" loading="lazy" decoding="async" />
      <span><i className={(source === "main" ? camera.recording : camera.sub_recording) ? "online" : ""} />{camera.name}</span>
    </button>)}
  </div>;
}

// A camera picker menu: focuses its search when opened, and closes on Escape
// or a press outside it.
function useCameraPickerMenu() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const rootRef = useRef(null);
  const searchRef = useRef(null);
  useEffect(() => {
    if (!open) return undefined;
    searchRef.current?.focus();
    function onPointerDown(event) {
      if (!rootRef.current?.contains(event.target)) setOpen(false);
    }
    function onKey(event) {
      if (event.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);
  return { open, setOpen, query, setQuery, rootRef, searchRef };
}

export function TimelineCameraPicker({ cameras, value, onChange, allOption = null, ariaLabel = "Select timeline camera" }) {
  const { open, setOpen, query, setQuery, rootRef, searchRef } = useCameraPickerMenu();
  const allSelected = Boolean(allOption) && value === allOption.value;
  const selectedCamera = cameras.find((camera) => camera.id === value);
  const selected = allSelected
    ? { id: allOption.value, name: allOption.label, recording: cameras.some((camera) => camera.recording || camera.sub_recording) }
    : selectedCamera || cameras[0] || (allOption ? { id: allOption.value, name: allOption.label, recording: false } : null);
  const matches = filteredTimelineCameras(cameras, query);
  if (!selected) return null;
  return (
    <div ref={rootRef} className={`timeline-camera-picker${open ? " open" : ""}`}>
      <button
        type="button"
        className="timeline-camera-picker-toggle"
        aria-expanded={open}
        aria-haspopup="listbox"
        onClick={() => { setQuery(""); setOpen((current) => !current); }}
      >
        <Camera size={15} />
        <strong>{selected.name}</strong>
        <i className={(selected.recording || selected.sub_recording) ? "online" : ""} />
      </button>
      {open ? (
        <div className="timeline-camera-picker-menu" role="listbox" aria-label={ariaLabel}>
          <label className="timeline-camera-picker-search">
            <Search size={14} aria-hidden="true" />
            <span className="sr-only">Search cameras</span>
            <input ref={searchRef} value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search cameras" />
          </label>
          <div className="timeline-camera-picker-list">
            {allOption ? (
              <button
                type="button"
                role="option"
                aria-selected={allSelected}
                className={allSelected ? "active" : ""}
                onClick={() => { onChange(allOption.value); setOpen(false); setQuery(""); }}
              >
                <Camera size={15} />
                <span>{allOption.label}</span>
                <i className={cameras.some((camera) => camera.recording || camera.sub_recording) ? "online" : ""} />
              </button>
            ) : null}
            {matches.map((camera) => (
              <button
                key={camera.id}
                type="button"
                role="option"
                aria-selected={camera.id === selected.id}
                className={camera.id === selected.id ? "active" : ""}
                onClick={() => { onChange(camera.id); setOpen(false); setQuery(""); }}
              >
                <Camera size={15} />
                <span>{camera.name}</span>
                <i className={(camera.recording || camera.sub_recording) ? "online" : ""} />
              </button>
            ))}
            {!matches.length ? <p>No matching cameras</p> : null}
          </div>
        </div>
      ) : null}
    </div>
  );
}

/** Choose a set of cameras (All cameras' Custom layout); one always stays. */
export function TimelineCameraChecklist({ cameras, selectedIds, onChange }) {
  const { open, setOpen, query, setQuery, rootRef, searchRef } = useCameraPickerMenu();
  const matches = filteredTimelineCameras(cameras, query);
  const toggle = (cameraId) => {
    const selected = selectedIds.includes(cameraId);
    if (selected && selectedIds.length === 1) return;
    const next = new Set(selected ? selectedIds.filter((id) => id !== cameraId) : [...selectedIds, cameraId]);
    onChange(cameras.filter((camera) => next.has(camera.id)).map((camera) => camera.id));
  };
  return (
    <div ref={rootRef} className={`timeline-camera-picker timeline-camera-checklist${open ? " open" : ""}`}>
      <button
        type="button"
        className="timeline-camera-picker-toggle"
        aria-expanded={open}
        aria-haspopup="true"
        onClick={() => { setQuery(""); setOpen((current) => !current); }}
      >
        <Camera size={15} />
        <strong>Choose cameras</strong>
        <small>{selectedIds.length}/{cameras.length}</small>
      </button>
      {open ? (
        <div className="timeline-camera-picker-menu" role="group" aria-label="Cameras to show">
          <label className="timeline-camera-picker-search">
            <Search size={14} aria-hidden="true" />
            <span className="sr-only">Search cameras</span>
            <input ref={searchRef} value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search cameras" />
          </label>
          <div className="timeline-camera-picker-list">
            {matches.map((camera) => {
              const checked = selectedIds.includes(camera.id);
              return (
                <button
                  key={camera.id}
                  type="button"
                  role="checkbox"
                  aria-checked={checked}
                  className={checked ? "active" : ""}
                  disabled={checked && selectedIds.length === 1}
                  title={checked && selectedIds.length === 1 ? "At least one camera stays selected" : undefined}
                  onClick={() => toggle(camera.id)}
                >
                  {checked ? <Check size={15} /> : <Camera size={15} />}
                  <span>{camera.name}</span>
                  <i className={(camera.recording || camera.sub_recording) ? "online" : ""} />
                </button>
              );
            })}
            {!matches.length ? <p>No matching cameras</p> : null}
          </div>
        </div>
      ) : null}
    </div>
  );
}

function parseDateKey(dateKey) {
  const [year, month, day] = String(dateKey || "").split("-").map(Number);
  return { year, month, day };
}

function formatTimelineDateLabel(dateKey) {
  const { year, month, day } = parseDateKey(dateKey);
  if (!year || !month || !day) return dateKey || "";
  return new Date(year, month - 1, day).toLocaleDateString(undefined, { month: "2-digit", day: "2-digit", year: "numeric" });
}

function shiftCalendarMonth(year, month, delta) {
  const next = new Date(year, month - 1 + delta, 1);
  return { year: next.getFullYear(), month: next.getMonth() + 1 };
}

export function TimelineDatePicker({ value, max, onChange, label = null, title = undefined }) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef(null);
  const menuRef = useRef(null);
  const selected = parseDateKey(value);
  const latest = parseDateKey(max);
  const [viewYear, setViewYear] = useState(selected.year);
  const [viewMonth, setViewMonth] = useState(selected.month);

  useEffect(() => {
    if (!open) return undefined;
    setViewYear(selected.year);
    setViewMonth(selected.month);
    function onPointerDown(event) {
      if (!rootRef.current?.contains(event.target) && !menuRef.current?.contains(event.target)) setOpen(false);
    }
    function onKey(event) {
      if (event.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open, selected.month, selected.year]);

  useEffect(() => {
    if (!open) return undefined;
    function place() {
      const button = rootRef.current;
      const menu = menuRef.current;
      if (!button || !menu) return;
      const rect = button.getBoundingClientRect();
      const menuRect = menu.getBoundingClientRect();
      const width = menuRect.width || 228;
      const height = menuRect.height || 236;
      menu.style.left = `${Math.max(8, Math.min(window.innerWidth - width - 8, rect.left + (rect.width / 2) - (width / 2)))}px`;
      menu.style.top = `${Math.max(8, rect.top - height - 6)}px`;
    }
    const frame = window.requestAnimationFrame(() => {
      place();
      window.requestAnimationFrame(place);
    });
    window.addEventListener("resize", place);
    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener("resize", place);
    };
  }, [open, viewMonth, viewYear]);

  const firstWeekday = new Date(viewYear, viewMonth - 1, 1).getDay();
  const daysInMonth = new Date(viewYear, viewMonth, 0).getDate();
  const cells = [
    ...Array.from({ length: firstWeekday }, () => null),
    ...Array.from({ length: daysInMonth }, (_, index) => index + 1),
  ];
  const monthLabel = new Date(viewYear, viewMonth - 1, 1).toLocaleDateString(undefined, { month: "short", year: "numeric" });
  const previousMonth = shiftCalendarMonth(viewYear, viewMonth, -1);
  const nextMonth = shiftCalendarMonth(viewYear, viewMonth, 1);
  const nextMonthDisabled = latest.year && (nextMonth.year > latest.year || (nextMonth.year === latest.year && nextMonth.month > latest.month));

  return (
    <div className={`recordings-v2-date-picker${open ? " open" : ""}`}>
      <input
        className="recordings-mobile-date"
        type="date"
        aria-label="Recording day"
        value={value}
        max={max}
        onChange={(event) => {
          const next = event.target.value;
          if (next && (!max || next <= max)) onChange(next);
        }}
      />
      <button
        ref={rootRef}
        type="button"
        className="recordings-v2-date-toggle"
        aria-label="Recording day"
        aria-expanded={open}
        aria-haspopup="dialog"
        title={title}
        onClick={() => setOpen((current) => !current)}
      >
        {label || formatTimelineDateLabel(value)}
      </button>
      {open ? (
        <div ref={menuRef} className="recordings-v2-date-calendar" role="dialog" aria-label="Choose recording day">
          <header>
            <button type="button" aria-label="Previous month" onClick={() => { setViewYear(previousMonth.year); setViewMonth(previousMonth.month); }}><ChevronLeft size={14} /></button>
            <strong>{monthLabel}</strong>
            <button type="button" aria-label="Next month" disabled={nextMonthDisabled} onClick={() => { setViewYear(nextMonth.year); setViewMonth(nextMonth.month); }}><ChevronRight size={14} /></button>
          </header>
          <div className="recordings-v2-date-weekdays" aria-hidden="true">
            {["S", "M", "T", "W", "T", "F", "S"].map((label, index) => <span key={`${label}-${index}`}>{label}</span>)}
          </div>
          <div className="recordings-v2-date-days">
            {cells.map((day, index) => {
              if (!day) return <span key={`empty-${index}`} />;
              const dateKey = `${viewYear}-${String(viewMonth).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
              const disabled = Boolean(max) && dateKey > max;
              const active = dateKey === value;
              return (
                <button
                  key={dateKey}
                  type="button"
                  className={active ? "active" : ""}
                  disabled={disabled}
                  onClick={() => { onChange(dateKey); setOpen(false); }}
                >
                  {day}
                </button>
              );
            })}
          </div>
        </div>
      ) : null}
    </div>
  );
}

function SearchResultImage({ result, alt, objectFocus, objectFocusZoom }) {
  const mode = normalizeIncidentThumbnailObjectFocus(objectFocus);
  const zoom = normalizeIncidentThumbnailObjectFocusZoom(objectFocusZoom);
  const observationId = String(result?.evidence?.observation_id || "").trim();
  const bboxKey = Array.isArray(result?.evidence?.bbox) ? result.evidence.bbox.join(",") : "";
  const [focused, setFocused] = useState(mode === "auto");
  useEffect(() => {
    setFocused(mode === "auto");
  }, [mode, result?.event?.id, result?.snapshot_url, observationId, bboxKey]);
  const imageOptions = {
    objectFocus: mode,
    zoom,
    aspectWidth: 16,
    aspectHeight: 9,
  };
  const src = mediaUrl(semanticSearchResultImageUrl(result, { ...imageOptions, focused }));
  const canFocus = semanticSearchResultImageUrl(result, { ...imageOptions, objectFocus: "button", focused: true })
    !== semanticSearchResultImageUrl(result, { ...imageOptions, objectFocus: "button", focused: false });
  const showButton = mode === "button" && canFocus;
  return <>
    {src ? <img src={src} alt={alt} loading="lazy" /> : null}
    {showButton ? <button
      type="button"
      className={`snapshot-focus-button ${focused ? "active" : ""}`}
      onClick={(event) => {
        event.preventDefault();
        event.stopPropagation();
        setFocused((value) => !value);
      }}
      title={focused ? "Show full snapshot" : "Focus detected objects"}
      aria-label={focused ? "Show full snapshot" : "Focus detected objects"}
    ><Crop size={14} /></button> : null}
  </>;
}

export function SemanticSearchPage({ timeZone, onAssistantContextChange }) {
  const initialQuery = useMemo(() => new URLSearchParams(window.location.search), []);
  const localStorage = useMemo(() => browserStorage(window), []);
  const sessionStorage = useMemo(() => {
    try { return window.sessionStorage; } catch { return null; }
  }, []);
  const initialQueryText = initialQuery.get("q") || "";
  const initialCameraId = initialQuery.get("camera") || "";
  const initialMode = speechSearchMode(initialQuery.get("mode"));
  const restoredSearch = useMemo(
    () => (initialMode === "visual" ? readSemanticSearchSession(sessionStorage, initialQueryText, initialCameraId) : null),
    [initialCameraId, initialMode, initialQueryText, sessionStorage],
  );
  const [mode, setMode] = useState(initialMode);
  const [speechStatus, setSpeechStatus] = useState(null);
  const [cameras, setCameras] = useState([]);
  const [cameraId, setCameraId] = useState(initialCameraId);
  const [query, setQuery] = useState(initialQueryText);
  const [submittedQuery, setSubmittedQuery] = useState(initialQueryText);
  const [resultsCameraId, setResultsCameraId] = useState(initialCameraId);
  const [results, setResults] = useState(() => restoredSearch?.results || []);
  const [searchHistory, setSearchHistory] = useState(() => readSemanticSearchHistory(localStorage));
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [setupError, setSetupError] = useState("");
  const searchRequestRef = useRef(null);
  const initialSearchRunRef = useRef(false);
  const { appConfig } = usePollingData();
  const thumbnailObjectFocus = appConfig?.incident_thumbnail_object_focus ?? "off";
  const thumbnailObjectFocusZoom = appConfig?.incident_thumbnail_object_focus_zoom ?? 1;
  const visibleResults = useMemo(
    () => semanticSearchResultsForCamera(results, resultsCameraId),
    [results, resultsCameraId],
  );

  async function loadSearchSetup() {
    setSetupError("");
    try {
      const [cameraResponse, statusResponse] = await Promise.all([fetch("/api/cameras"), fetch("/api/semantic-search/status")]);
      if (!cameraResponse.ok || !statusResponse.ok) throw new Error("Search status is unavailable.");
      const [cameraRows, semanticStatus] = await Promise.all([cameraResponse.json(), statusResponse.json()]);
      setCameras(cameraRows || []);
      setStatus(semanticStatus);
    } catch (reason) {
      setSetupError(reason.message || "Could not load Smart Search.");
    }
    try {
      const speechResponse = await fetch("/api/speech/status");
      setSpeechStatus(speechResponse.ok ? await speechResponse.json() : null);
    } catch {
      setSpeechStatus(null);
    }
  }
  useEffect(() => { void loadSearchSetup(); }, []);
  useEffect(() => {
    if (initialMode !== "visual" || initialSearchRunRef.current || restoredSearch || !initialQueryText.trim() || !status) return;
    initialSearchRunRef.current = true;
    void runSearch(null, initialQueryText, initialCameraId);
  }, [status]);
  useEffect(() => {
    onAssistantContextChange?.({ page: "search", camera_id: cameraId, filters: { semantic_query: submittedQuery } });
  }, [cameraId, onAssistantContextChange, submittedQuery]);
  useEffect(() => () => {
    const activeRequest = searchRequestRef.current;
    searchRequestRef.current = null;
    activeRequest?.abort();
  }, []);

  async function runSearch(event, requestedQuery = query, requestedCameraId = cameraId) {
    event?.preventDefault();
    const searchQuery = String(requestedQuery || "").trim();
    const searchCameraId = String(requestedCameraId || "");
    if (!searchQuery) return;
    searchRequestRef.current?.abort();
    const controller = new AbortController();
    searchRequestRef.current = controller;
    setQuery(searchQuery);
    setCameraId(searchCameraId);
    setLoading(true); setError("");
    try {
      const response = await fetch("/api/semantic-search", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: searchQuery, camera_ids: searchCameraId ? [searchCameraId] : [], limit: 100 }),
        signal: controller.signal,
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Smart Search failed.");
      const nextResults = payload.results || [];
      setResults(nextResults);
      setSubmittedQuery(searchQuery);
      setResultsCameraId(searchCameraId);
      writeSemanticSearchSession(sessionStorage, { query: searchQuery, cameraId: searchCameraId, results: nextResults });
      setSearchHistory((current) => {
        const next = addSemanticSearchHistory(current, {
          query: searchQuery,
          cameraId: searchCameraId,
          searchedAt: new Date().toISOString(),
        });
        writeSemanticSearchHistory(localStorage, next);
        return next;
      });
      const params = new URLSearchParams({ q: searchQuery });
      if (searchCameraId) params.set("camera", searchCameraId);
      window.history.replaceState(null, "", appUrl(`/search?${params.toString()}`));
    } catch (reason) {
      if (reason?.name !== "AbortError") {
        setCameraId(resultsCameraId);
        setError(reason.message || "Smart Search failed.");
      }
    } finally {
      if (searchRequestRef.current === controller) {
        searchRequestRef.current = null;
        setLoading(false);
      }
    }
  }

  function resetSearch() {
    searchRequestRef.current?.abort();
    searchRequestRef.current = null;
    setLoading(false);
    setQuery("");
    setSubmittedQuery("");
    setCameraId("");
    setResultsCameraId("");
    setResults([]);
    setError("");
    clearSemanticSearchSession(sessionStorage);
    window.history.replaceState(null, "", appUrl("/search"));
  }

  function selectMode(nextMode) {
    if (loading || nextMode === mode) return;
    searchRequestRef.current?.abort();
    setMode(nextMode);
    setError("");
    if (nextMode === "visual") {
      const params = new URLSearchParams();
      if (submittedQuery) params.set("q", submittedQuery);
      if (cameraId) params.set("camera", cameraId);
      window.history.replaceState(null, "", appUrl(`/search${params.toString() ? `?${params}` : ""}`));
    }
  }

  function selectCamera(nextCameraId) {
    if (loading) return;
    const selectedCameraId = String(nextCameraId || "");
    setCameraId(selectedCameraId);
    // Speech results follow the camera selection on their own.
    if (mode === "speech") return;
    const committedQuery = String(submittedQuery || query || "").trim();
    if (committedQuery) void runSearch(null, committedQuery, selectedCameraId);
  }

  return <main className="search-page">
    <header className="search-commandbar">
      <div><Search size={18} /><span><strong>Smart Search</strong><small>{mode === "speech" ? "Find what was said near your cameras" : "Find incidents by describing what you remember"}</small></span></div>
      {speechStatus?.enabled || mode === "speech" ? <div className="search-mode-switch" role="group" aria-label="Search type">
        <button type="button" aria-pressed={mode === "visual"} className={mode === "visual" ? "active" : ""} onClick={() => selectMode("visual")} disabled={loading}><Search size={14} />Visual</button>
        <button type="button" aria-pressed={mode === "speech"} className={mode === "speech" ? "active" : ""} onClick={() => selectMode("speech")} disabled={loading}><Mic size={14} />Speech</button>
      </div> : null}
      <MobileCameraSelect
        className="search-mobile-camera-select"
        cameras={cameras}
        value={cameraId}
        onChange={selectCamera}
        ariaLabel="Smart Search camera"
        allOption={{ value: "", label: "All cameras" }}
        disabled={loading}
      />
      {mode === "speech" ? <span className={`semantic-status ${speechStatus?.enabled ? "ready" : "disabled"}`}>{speechStatus?.enabled ? `Speech ${speechStatus.state || "on"}` : "Speech off"}</span> : setupError ? <button type="button" className="search-setup-retry" onClick={() => void loadSearchSetup()}><RefreshCcw size={15} />Retry status</button> : <span className={`semantic-status ${status?.state || ""}`}>{status?.state === "ready" ? `${Number(status.event_count || 0).toLocaleString()} incidents indexed` : status?.state || "Loading"}</span>}
    </header>
    <aside className="search-scope-panel">
      <div className="search-scope-heading"><strong>Camera</strong><small>Refine current results</small></div>
      <div className="search-camera-list">
        <button type="button" className={!cameraId ? "active" : ""} onClick={() => selectCamera("")} disabled={loading} aria-pressed={!cameraId}><Search size={16} /><span>All cameras</span><i /></button>
        {cameras.map((camera) => <button type="button" key={camera.id} className={cameraId === camera.id ? "active" : ""} onClick={() => selectCamera(camera.id)} disabled={loading} aria-pressed={cameraId === camera.id}><Camera size={16} /><span>{camera.name}</span><i className={camera.running ? "online" : ""} /></button>)}
      </div>
      {mode === "visual" && searchHistory.length ? <section className="search-history-panel" aria-labelledby="search-history-title">
        <div id="search-history-title"><Clock3 size={15} /><strong>Recent searches</strong></div>
        {searchHistory.map((item) => {
          const cameraName = item.cameraId ? cameras.find((camera) => camera.id === item.cameraId)?.name || item.cameraId : "All cameras";
          return <button type="button" key={`${item.query.toLocaleLowerCase()}-${item.cameraId}`} onClick={() => void runSearch(null, item.query, item.cameraId)} disabled={loading} title={`Search ${cameraName}`}><strong>{item.query}</strong><small>{cameraName}</small></button>;
        })}
      </section> : null}
    </aside>
    {mode === "speech" ? <SpeechSearchWorkspace
      cameras={cameras}
      cameraId={cameraId}
      timeZone={timeZone}
      initialQuery={initialMode === "speech" ? initialQueryText : ""}
      disabledReason={speechStatus && !speechStatus.enabled ? "Speech detection is turned off. Past transcripts remain searchable until retention removes them." : ""}
    /> : <section className="semantic-search-workspace">
      <form onSubmit={runSearch} className="semantic-search-form"><Search size={20} /><input value={query} onChange={(event) => setQuery(event.target.value)} disabled={loading} placeholder='Try “person in a red jacket” or “white delivery truck”' aria-label="Describe what to find" autoFocus /><div className="semantic-search-actions"><button type="button" className="secondary" onClick={resetSearch} disabled={!query && !cameraId && !results.length && !error}>Reset</button><button type="submit" disabled={loading || !query.trim()}>{loading ? "Searching…" : "Search"}</button></div></form>
      <div className="search-results-heading" aria-live="polite"><strong>{loading ? "Searching…" : results.length ? `${visibleResults.length} visual matches${submittedQuery ? ` for “${submittedQuery}”` : ""}` : "Visual matches"}</strong>{resultsCameraId ? <small>{cameras.find((camera) => camera.id === resultsCameraId)?.name || resultsCameraId}</small> : submittedQuery ? <small>All cameras</small> : null}</div>
      {error ? <div className="semantic-search-error"><CircleAlert size={17} />{error}</div> : null}
      <div className="semantic-search-results">
        {visibleResults.map((result) => {
          const item = result.event || {};
          const context = incidentRecordingContext(item);
          const matchLabel = ({ strong_match: "Strong match", possible_match: "Possible match" })[result.match_strength] || "Visually similar";
          const cameraName = cameras.find((camera) => camera.id === item.camera_id)?.name || item.camera_id;
          const observedAt = formatDateTime(new Date(item.created_at).getTime() / 1000, timeZone);
          return <article key={item.id} aria-label={`${matchLabel} at ${cameraName}, ${observedAt}`}><div className="semantic-result-image"><SearchResultImage result={result} alt={`${cameraName} search result`} objectFocus={thumbnailObjectFocus} objectFocusZoom={thumbnailObjectFocusZoom} /><span title={`Raw visual similarity ${Number(result.score || 0).toFixed(3)}`}>{matchLabel}</span></div><footer><div><strong>{cameraName}</strong><small>{observedAt}</small><IdentityChip item={item} className="semantic-result-identity" /></div><nav aria-label={`Actions for ${cameraName} result`}><a href={appUrl(incidentOpenedFromHref(item.id, `${appPathname()}${window.location.search}${window.location.hash}`))}>Open incident</a><a href={recordingsHref(context)}><Play size={14} />Timeline</a></nav></footer></article>;
        })}
        {!loading && !error && !visibleResults.length ? <div className="semantic-search-empty"><Search size={28} /><strong>{results.length && cameraId ? "No matching results from this camera" : "Search indexed incidents by appearance"}</strong><span>{results.length && cameraId ? "Choose All cameras or another camera to widen the current results." : "Results link to the exact incident and recording time."}</span></div> : null}
      </div>
    </section>}
  </main>;
}

export function RecordingsPage({ timeZone, onAssistantContextChange }) {
  const { cameras: sharedCameras, appConfig } = usePollingData();
  const initialQuery = useMemo(() => new URLSearchParams(window.location.search), []);
  const today = dateKeyForTimeZone(Date.now(), timeZone);
  const initialView = useMemo(() => parseTimelineView(initialQuery, today), [initialQuery, today]);
  const initialEpoch = initialView.at;
  const initialDate = !initialQuery.get("date") && initialEpoch ? dateKeyForTimeZone(initialEpoch * 1000, timeZone) : initialView.date;
  const videoRef = useRef(null);
  const desiredEpochRef = useRef(initialEpoch);
  const autoplayRef = useRef(false);
  const codecFallbackRef = useRef(false);
  const originalFallbackRef = useRef(null);
  const [originalScope, setOriginalScope] = useState(null);
  const transcodeFallbackRef = useRef(null);
  const failedSubScopeRef = useRef(null);
  const [transcodeScope, setTranscodeScope] = useState(null);
  const [nativeHls] = useState(supportsNativeRecordingHls);
  const [transportOverride] = useState(recordingTransportOverride);
  const [liveEdgeFollow] = useState(recordingLiveEdgeFollow);
  const manifestStartRef = useRef(null);
  const playbackDetailRef = useRef(null);
  const growingRowsRef = useRef({ at: 0, inFlight: false });
  const [showPlaybackDiagnostics] = useState(() => playbackDiagnosticsEnabled(transportOverride));
  const heroMediaRef = useRef(null);
  const [playbackTransport, setPlaybackTransport] = useState(null);
  const playbackTransportRef = useRef(null);
  const playbackRequestRef = useRef(0);
  const playbackWindowCacheRef = useRef(new Map());
  const latestAvailabilityRef = useRef(null);
  const pendingSeekEpochRef = useRef(null);
  const pendingSeekModeRef = useRef(null);
  const nativePlaylistStartRef = useRef(null);
  const [visibleNativePlaylist, setVisibleNativePlaylist] = useState(null);
  const seekWatchdogRef = useRef(null);
  const ignorePauseUntilRef = useRef(0);
  const playbackRetryRef = useRef({ attempts: 0, timer: null });
  const gridRefreshCursorRef = useRef(null);
  const recordingUpdatesInFlightRef = useRef(false);
  const gridUpdatesInFlightRef = useRef(false);
  const selectedIncidentIdentityCacheRef = useRef(new Map());
  const frameSearchSurfaceRef = useRef(null);
  const frameSearchImageRef = useRef(null);
  const frameSearchDragRef = useRef(null);
  const frameSearchCropPreviewRef = useRef(null);
  const frameSearchRequestRef = useRef(null);
  const [cameras, setCameras] = useState([]);
  const [cameraTransitionRoutes, setCameraTransitionRoutes] = useState([]);
  const [cameraId, setCameraId] = useState(initialView.cameraId);
  const [source, setSource] = useState(initialView.source || (initialView.cameraId === ALL_RECORDING_CAMERAS_ID ? "live" : preferredStreamSource()));
  const [date, setDate] = useState(initialDate);
  const [recordings, setRecordings] = useState([]);
  const [playbackDetail, setPlaybackDetail] = useState(null);
  const [events, setEvents] = useState([]);
  const [eventFilter, setEventFilter] = useState(initialView.eventFilter);
  const [incidentRangeHours, setIncidentRangeHours] = useState(initialView.windowHours);
  const [availableSources, setAvailableSources] = useState([]);
  const [playhead, setPlayhead] = useState(null);
  const [loading, setLoading] = useState(true);
  const [playbackError, setPlaybackError] = useState("");
  const [playbackErrorStage, setPlaybackErrorStage] = useState("");
  const [playbackNotice, setPlaybackNotice] = useState("");
  const [playbackBlocked, setPlaybackBlocked] = useState(false);
  const [heroPlaying, setHeroPlaying] = useState(false);
  const [heroSeeking, setHeroSeeking] = useState(false);
  const scrubOverlayRef = useRef(null);
  const [playbackWindow, setPlaybackWindow] = useState(null);
  const [playbackWindowRevision, setPlaybackWindowRevision] = useState(0);
  const [manifestRetryToken, setManifestRetryToken] = useState(0);
  const [nativeSegment, setNativeSegment] = useState(null);
  const [nativeSegmentRetryToken, setNativeSegmentRetryToken] = useState(0);
  const [nativePrefetchDetail, setNativePrefetchDetail] = useState(null);
  const [recordingIndexRevision, setRecordingIndexRevision] = useState(0);
  const [followTarget, setFollowTarget] = useState(null);
  const [exportRange, setExportRange] = useState(null);
  const [exportKind, setExportKind] = useState("recording");
  const [exportOptions, setExportOptions] = useState({ interval: 30, fps: 30, clipHeight: 0, timelapseHeight: 720 });
  const [exportLabel, setExportLabel] = useState("");
  const [exportJob, setExportJob] = useState(null);
  const [exportError, setExportError] = useState("");
  const [exportSubmitting, setExportSubmitting] = useState(false);
  const [clipPreviewing, setClipPreviewing] = useState(false);
  const clipPreviewEndRef = useRef(null);
  const [gridPlaying, setGridPlaying] = useState(false);
  const [selectedEventId, setSelectedEventId] = useState(initialView.eventId);
  const [trailEventIds, setTrailEventIds] = useState(() => (
    initialView.trailEventIds?.length
      ? initialView.trailEventIds
      : (readVisualSearchTrail(window.sessionStorage)?.eventIds || [])
  ));
  const [trailMeta, setTrailMeta] = useState(() => readVisualSearchTrail(window.sessionStorage));
  const [trailNotice, setTrailNotice] = useState("");
  const [selectedIncidentIdentity, setSelectedIncidentIdentity] = useState(null);
  const [selectedIdentityRevision, setSelectedIdentityRevision] = useState(0);
  const [investigationOpen, setInvestigationOpen] = useState(
    () => Boolean(initialView.eventId || initialView.trailEventIds?.length),
  );
  const [heroMuted, setHeroMuted] = useState(false);
  const [aiOverlayEnabled, setAiOverlayEnabled] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(initialView.speed);
  const [timelineLayout, setTimelineLayout] = useState(initialView.view);
  const [railMinutes, setRailMinutes] = useState(initialView.railMinutes);
  const [quietMinutes, setQuietMinutes] = useState(5);
  const [laneData, setLaneData] = useState({ scope: "", recordings: [], events: [] });
  const [timelineViewportAnchor, setTimelineViewportAnchor] = useState(initialView.at);
  const [semanticStatus, setSemanticStatus] = useState(null);
  const [frameSearchEpoch, setFrameSearchEpoch] = useState(null);
  const [frameSearchCrop, setFrameSearchCrop] = useState(null);
  const [frameSearchImageBounds, setFrameSearchImageBounds] = useState(null);
  const [frameSearchImageReady, setFrameSearchImageReady] = useState(false);
  const [frameSearchResults, setFrameSearchResults] = useState([]);
  const [frameSearchQueryImageUrl, setFrameSearchQueryImageUrl] = useState("");
  const [frameSearchQueryEpoch, setFrameSearchQueryEpoch] = useState(null);
  const [frameSearchLoading, setFrameSearchLoading] = useState(false);
  const [frameSearchError, setFrameSearchError] = useState("");

  useEffect(() => () => {
    frameSearchRequestRef.current?.abort();
    frameSearchRequestRef.current = null;
  }, []);

  // All cameras is a desktop and tablet view; a phone opens the first camera.
  const phoneTimeline = useViewportQuery("(max-width: 760px)");
  const isAllCameras = cameraId === ALL_RECORDING_CAMERAS_ID && !phoneTimeline;
  // Its layout follows Live: Automatic (density and pages), Focus, and Custom
  // (here, a chosen set of cameras); Focus and Custom need desktop width.
  const gridLayoutsAvailable = useViewportQuery("(min-width: 1051px)");
  const [gridLayoutMode, setGridLayoutMode] = useStoredState("survng.timelineGridLayoutMode.v1", "auto");
  const [gridDensity, setGridDensity] = useStoredState("survng.timelineGridDensity.v1", "fit");
  const [gridFocusCamera, setGridFocusCamera] = useStoredState("survng.timelineGridFocusCamera.v1", "");
  const [gridCustomCameras, setGridCustomCameras] = useStoredState("survng.timelineGridCameras.v1", "");
  // Focus entered from Custom keeps the chosen cameras around the primary.
  const [gridFocusScope, setGridFocusScope] = useStoredState("survng.timelineGridFocusScope.v1", "all");
  const [gridPage, setGridPage] = useState(0);
  const gridCameras = useMemo(() => cameras.filter((camera) => camera.enabled !== false), [cameras]);
  const gridCustomIds = useMemo(() => allCamerasCustomIds(gridCustomCameras, gridCameras), [gridCameras, gridCustomCameras]);
  const activeCameraId = resolveTimelineHeroCameraId(cameras, cameraId);
  const dayStart = useMemo(() => zonedDateSecondToEpoch(date, 0, timeZone), [date, timeZone]);
  const nextDate = addDaysToDateKey(date, 1);
  const dayEnd = useMemo(() => zonedDateSecondToEpoch(nextDate, 0, timeZone), [nextDate, timeZone]);

  useEffect(() => {
    onAssistantContextChange?.({
      page: "recordings",
      camera_id: isAllCameras ? "" : activeCameraId,
      recording_epoch: Number(playhead) || initialEpoch || dayStart,
      filters: { date, source, event_type: eventFilter, camera_view: isAllCameras ? "all" : "single" },
    });
  }, [activeCameraId, date, dayStart, eventFilter, initialEpoch, isAllCameras, onAssistantContextChange, playhead, source]);

  const timeline = useMemo(() => {
    return recordingPlaybackTimeline(recordings);
  }, [recordings]);
  latestAvailabilityRef.current = timeline[timeline.length - 1]?.end_epoch ?? null;
  // All cameras gets every camera's ranges; the rail shows where any has recording.
  // Only cameras with recording in the playhead's window get a tile.
  const gridSelection = useMemo(
    () => allCamerasGridSelection({
      cameras: isAllCameras ? gridCameras : [],
      ranges: timeline,
      epoch: playhead,
      mode: gridLayoutMode,
      density: gridDensity,
      page: gridPage,
      customIds: gridCustomIds,
      focusCameraId: gridFocusCamera,
      focusScope: gridFocusScope,
      layoutsAvailable: gridLayoutsAvailable,
    }),
    // playhead matters only through its 15-minute window.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [gridCameras, gridCustomIds, gridDensity, gridFocusCamera, gridFocusScope, gridLayoutMode, gridLayoutsAvailable, gridPage, isAllCameras, timeline, Number.isFinite(playhead) ? allCamerasWindowStart(playhead) : null],
  );
  const recordedGridCameras = gridSelection.recorded;
  // Keep the stored page on the page shown, so a shrinking set cannot bring
  // an old page back later.
  useEffect(() => {
    if (isAllCameras && gridPage !== gridSelection.page) setGridPage(gridSelection.page);
  }, [gridPage, gridSelection.page, isAllCameras]);
  // A camera=all link on a phone opens the first camera (no grid on phones).
  useEffect(() => {
    if (phoneTimeline && cameraId === ALL_RECORDING_CAMERAS_ID && activeCameraId) setCameraId(activeCameraId);
  }, [activeCameraId, cameraId, phoneTimeline]);
  const unrecordedGridCameras = gridSelection.unrecorded;
  // All cameras has no Nearby panel.
  const investigationShown = investigationOpen && !isAllCameras;
  const railRecordings = useMemo(() => (isAllCameras ? recordingCoverageUnion(timeline) : timeline), [isAllCameras, timeline]);
  const nativeScope = `${activeCameraId}:${source}:${dayStart}:${dayEnd}`;
  const playbackTimeline = useMemo(() => {
    if (playbackDetail?.scope !== nativeScope) return [];
    return recordingPlaybackTimeline(playbackDetail.rows);
  }, [playbackDetail, nativeScope]);
  const loadedPlaybackWindow = playbackDetail?.scope === nativeScope ? playbackDetail : null;
  playbackDetailRef.current = playbackDetail;
  const requestedTransport = recordingPlaybackTransport({
    nativeHls, rate: playbackRate, incompatible: transcodeScope === nativeScope, preferOriginal: originalScope === nativeScope,
    override: transportOverride,
  });
  const transport = playbackTransport?.scope === nativeScope ? playbackTransport.mode : requestedTransport;
  const useTranscodedPlayback = transport === "transcode";
  const useSegmentPlayback = transport !== "hls";
  playbackTransportRef.current = { scope: nativeScope, mode: transport };
  // Snapshot the start offset only when loading a playback resource. The day
  // overview changes as new recordings arrive; including it here changes src
  // during playback and makes native HLS discard its buffer every poll.
  const manifestStartTime = useMemo(() => {
    if (!playbackTimeline.length) return null;
    // A growing playlist gains rows while it plays. Its start offset belongs
    // to the load, not to the rows: recomputing it would change src too.
    const loadKey = playbackDetail.growing
      ? `${nativeScope}:${playbackDetail.revision}:${manifestRetryToken}:${transport}` : null;
    if (loadKey && manifestStartRef.current?.key === loadKey) return manifestStartRef.current.value;
    const retainedEpoch = desiredEpochRef.current;
    const initialEpoch = Number.isFinite(retainedEpoch) && retainedEpoch >= dayStart && retainedEpoch < dayEnd
      ? retainedEpoch
      : playbackTimeline[0].start_epoch;
    const value = epochToPlaybackMediaTime(initialEpoch);
    if (loadKey) manifestStartRef.current = { key: loadKey, value };
    return value;
  }, [playbackTimeline, dayStart, dayEnd, manifestRetryToken, transport]);
  const manifestUrl = !useSegmentPlayback && !isAllCameras && activeCameraId && playbackDetail && playbackTimeline.length
    ? `${recordingDayHlsUrl(activeCameraId, playbackDetail.start, playbackDetail.end, source, manifestStartTime, Boolean(playbackDetail.growing))}&reload=${playbackDetail.revision || 0}-${manifestRetryToken}`
    : "";
  const nativeSegmentUrl = useSegmentPlayback && !isAllCameras && activeCameraId && loadedPlaybackWindow && nativeSegment
    ? `${recordingSegmentUrl(activeCameraId, nativeSegment.start_epoch, source, useTranscodedPlayback)}&reload=${nativeSegmentRetryToken}`
    : "";
  const nativeNextEpoch = nativeSegment ? recordingEpochAfterSegment(nativeSegment, timeline) : null;
  const prefetchedNativeWindow = nativePrefetchDetail?.scope === nativeScope ? nativePrefetchDetail : null;
  const nativeNextSegment = nativeNextEpoch === null ? null
    : recordingSegmentAt(playbackTimeline, nativeNextEpoch)
      || recordingSegmentAt(prefetchedNativeWindow?.rows, nativeNextEpoch);
  const nativeNextUrl = nativeSegmentUrl && heroPlaying && nativeNextSegment
    ? `${recordingSegmentUrl(activeCameraId, nativeNextSegment.start_epoch, source, useTranscodedPlayback)}&reload=${nativeSegmentRetryToken}`
    : "";

  useEffect(() => {
    // Warm only the next detail window when playback reaches its boundary.
    // The standby video can then preload across a 15-minute index boundary too.
    if (!nativeSegmentUrl || !heroPlaying || nativeNextEpoch === null || nativeNextSegment) return undefined;
    const controller = new AbortController();
    const nextWindow = windowAround(nativeNextEpoch);
    fetch(recordingWindowUrl(activeCameraId, nextWindow.start, nextWindow.end, source), { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) return null;
        return response.json();
      })
      .then((payload) => {
        if (!payload || controller.signal.aborted) return;
        setNativePrefetchDetail({
          scope: nativeScope, start: Number(payload.start_epoch), end: Number(payload.end_epoch),
          rows: payload.recordings || [],
        });
      })
      .catch(() => { /* Foreground loading retains its normal retry/error handling. */ });
    return () => controller.abort();
  }, [nativeScope, nativeSegmentUrl, heroPlaying, nativeNextEpoch, Boolean(nativeNextSegment)]);
  const hasPlaybackMedia = Boolean(manifestUrl || nativeSegmentUrl);
  const coveringNativeStartup = nativeHls && Boolean(manifestUrl) && visibleNativePlaylist !== manifestUrl;
  const playbackMuted = heroMuted || coveringNativeStartup;
  useEffect(() => {
    if (!manifestUrl) setVisibleNativePlaylist(null);
  }, [manifestUrl]);
  // Keep AI analysis strictly local to an actively playing Timeline video. The
  // overlay's key resets its temporary tracks when the recording scope changes.
  const aiOverlayActive = aiOverlayEnabled
    && !isAllCameras
    && heroPlaying
    && !heroSeeking
    && hasPlaybackMedia
    && !Number.isFinite(frameSearchEpoch)
    && !Number.isFinite(pendingSeekEpochRef.current);
  const aiOverlayKey = `${activeCameraId}:${source}:${playbackDetail?.revision || 0}:${nativeSegmentUrl || manifestUrl}`;

  useEffect(() => {
    if (videoRef.current) videoRef.current.muted = playbackMuted;
  }, [playbackMuted, manifestUrl, nativeSegmentUrl]);

  const semanticReady = Boolean(
    semanticStatus?.enabled && semanticStatus?.state === "ready"
  );
  const semanticIndexedCount = Math.max(0, Number(semanticStatus?.event_count) || 0);
  const semanticStatusLabel = semanticStatus?.backfill_active
    ? `${semanticIndexedCount} indexed · Indexing…`
    : semanticReady
      ? `${semanticIndexedCount} indexed`
      : semanticStatus == null
        ? "Checking search…"
        : semanticStatus.enabled
          ? "Search unavailable"
          : "Smart Search off";
  const frameSearchPreviewUrl = activeCameraId && Number.isFinite(frameSearchEpoch)
    ? recordingPreviewUrl(activeCameraId, frameSearchEpoch, source, {
      width: 1280,
      exact: true,
    })
    : "";

  function switchRecordingTransport() {
    if (playbackTransport?.scope !== nativeScope) {
      setPlaybackTransport({ scope: nativeScope, mode: transport });
      return;
    }
    if (transport === requestedTransport) return;
    const target = Number.isFinite(pendingSeekEpochRef.current)
      ? pendingSeekEpochRef.current : desiredEpochRef.current;
    clearSeekWatchdog();
    if (playbackRetryRef.current.timer) window.clearTimeout(playbackRetryRef.current.timer);
    playbackRetryRef.current = { attempts: 0, timer: null };
    pendingSeekEpochRef.current = target;
    const segment = recordingSegmentAt(playbackTimeline, target);
    pendingSeekModeRef.current = !segment ? "window" : requestedTransport === "hls" ? "window-ready" : "native-ready";
    setHeroSeeking(Number.isFinite(target));
    setNativeSegment(segment);
    setPlaybackTransport({ scope: nativeScope, mode: requestedTransport });
    if (!segment && Number.isFinite(target)) requestPlaybackWindow(windowAround(target));
  }

  useEffect(switchRecordingTransport, [nativeScope, transport, requestedTransport]);

  useEffect(() => {
    // Do not apply 4× to the outgoing native HLS element during a transport switch.
    if (transport === requestedTransport && videoRef.current) {
      videoRef.current.playbackRate = normalizedTimelinePlaybackRate(playbackRate);
    }
  }, [playbackRate, transport, requestedTransport]);

  const filteredEvents = useMemo(() => events
    .filter((event) => timelineEventMatchesFilter(event, eventFilter))
    .map((event) => ({ ...event, incident_epoch: recordingIncidentEpoch(event) }))
    .filter((event) => Number.isFinite(event.incident_epoch))
    .sort((left, right) => left.incident_epoch - right.incident_epoch), [eventFilter, events]);

  const viewportAnchor = Number.isFinite(timelineViewportAnchor)
    ? timelineViewportAnchor
    : Number.isFinite(playhead) ? playhead : desiredEpochRef.current;
  const timelineView = useMemo(
    () => timelineViewport(dayStart, dayEnd, viewportAnchor, incidentRangeHours),
    [dayEnd, dayStart, incidentRangeHours, viewportAnchor],
  );

  const timelineEvents = useMemo(() => events
    .map((event) => ({ ...event, incident_epoch: recordingIncidentEpoch(event) }))
    .filter((event) => Number.isFinite(event.incident_epoch))
    .sort((left, right) => left.incident_epoch - right.incident_epoch), [events]);

  const viewportEvents = useMemo(() => filteredEvents.filter((event) => (
    event.incident_epoch >= timelineView.startEpoch
    && event.incident_epoch <= timelineView.endEpoch
  )), [filteredEvents, timelineView.endEpoch, timelineView.startEpoch]);
  const nearbyEvents = useMemo(() => {
    const anchor = Number.isFinite(playhead) ? playhead : viewportAnchor;
    const radius = timelineNearbyRadiusSeconds(incidentRangeHours);
    const limit = incidentRangeHours >= 24 ? 24 : 16;
    return timelineEvidenceWindow(filteredEvents, anchor, limit, radius);
  }, [filteredEvents, incidentRangeHours, playhead, viewportAnchor]);
  const selectedEventSummary = viewportEvents.find((event) => Number(event.id) === Number(selectedEventId))
    || timelineEvents.find((event) => Number(event.id) === Number(selectedEventId))
    || null;
  const trailHit = trailHitForEvent(trailMeta, selectedEventId);
  const trailSyntheticEvent = !selectedEventSummary && trailHit?.event
    ? {
      id: trailHit.event.id,
      camera_id: trailHit.event.camera_id,
      created_at: trailHit.event.created_at,
      incident_epoch: trailHit.event.incident_epoch
        || (trailHit.event.created_at ? new Date(trailHit.event.created_at).getTime() / 1000 : null),
      labels: trailHit.event.labels || (trailHit.query_mode === "appearance" ? ["Appearance match"] : ["Visual match"]),
      has_objects: true,
      // Truthy so the selected-incident panel can request /api/events/{id}/thumbnail.jpg
      // before the day event list (or nearby rail) has loaded this hit.
      snapshot_path: trailHit.event.snapshot_path || "available",
    }
    : null;
  const selectedEventBase = selectedEventSummary || trailSyntheticEvent;
  const selectedIdentityEventId = timelineIdentityDetailEventId(selectedEventBase);
  const selectedEvent = selectedEventBase && selectedIncidentIdentity?.eventId === selectedIdentityEventId
    ? mergeTimelineIncidentIdentity(selectedEventBase, selectedIncidentIdentity.detail)
    : selectedEventBase;
  const forensicNav = useMemo(() => {
    const count = frameSearchResults.length;
    if (!count) {
      return { label: "1 / 1", previous: null, next: null };
    }
    const index = Number.isInteger(Number(selectedEventId))
      ? frameSearchResults.findIndex((result) => Number(result?.event?.id) === Number(selectedEventId))
      : -1;
    if (index < 0) {
      return {
        label: "1 / 1",
        previous: null,
        next: frameSearchResults[0] || null,
      };
    }
    return {
      label: `${index + 1} / ${count}`,
      previous: index > 0 ? frameSearchResults[index - 1] : null,
      next: index < count - 1 ? frameSearchResults[index + 1] : null,
      deselectOnPrevious: index === 0,
    };
  }, [frameSearchResults, selectedEventId]);
  const forensicContext = useMemo(() => {
    if (!frameSearchResults.length || !frameSearchQueryImageUrl) return null;
    return {
      imageUrl: frameSearchQueryImageUrl,
      timeLabel: Number.isFinite(frameSearchQueryEpoch)
        ? formatDateTime(frameSearchQueryEpoch, timeZone)
        : "--",
      kind: "object",
    };
  }, [frameSearchResults.length, frameSearchQueryImageUrl, frameSearchQueryEpoch, timeZone]);
  const showForensicPanel = frameSearchResults.length > 0;

  useAppEvents(({ type, data }) => {
    if (type !== "identity_update") return;
    const eventId = Number(data?.event_id);
    if (!Number.isInteger(eventId) || eventId <= 0) return;
    const invalidatedCacheIds = invalidateTimelineIdentityCache(selectedIncidentIdentityCacheRef.current, eventId);
    const selectedCacheInvalidated = invalidatedCacheIds.includes(selectedIdentityEventId);
    if (selectedCacheInvalidated || timelineIncidentIncludesEvent(selectedEventBase, eventId)) {
      setSelectedIncidentIdentity(null);
      setSelectedIdentityRevision((revision) => revision + 1);
    }
  });

  useEffect(() => {
    if (!investigationOpen || !selectedIdentityEventId) {
      setSelectedIncidentIdentity(null);
      return undefined;
    }
    const cached = selectedIncidentIdentityCacheRef.current.get(selectedIdentityEventId);
    if (cached) {
      setSelectedIncidentIdentity({ eventId: selectedIdentityEventId, detail: cached });
      return undefined;
    }
    const controller = new AbortController();
    setSelectedIncidentIdentity(null);
    fetch(`/api/incidents/by-event/${encodeURIComponent(selectedIdentityEventId)}`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Incident identity failed (${response.status})`);
        return response.json();
      })
      .then((detail) => {
        if (controller.signal.aborted) return;
        selectedIncidentIdentityCacheRef.current.set(selectedIdentityEventId, detail);
        setSelectedIncidentIdentity({ eventId: selectedIdentityEventId, detail });
      })
      .catch(() => {
        // The compact timeline incident remains usable when identity detail is unavailable.
      });
    return () => controller.abort();
  }, [investigationOpen, selectedIdentityEventId, selectedIdentityRevision]);

  useEffect(() => {
    setTimelineViewportAnchor(null);
  }, [activeCameraId, date, source]);

  useEffect(() => {
    if (!Number.isFinite(timelineViewportAnchor) && Number.isFinite(playhead)) {
      setTimelineViewportAnchor(playhead);
    }
  }, [playhead, timelineViewportAnchor]);

  useEffect(() => {
    if (!Number.isFinite(playhead)) return;
    const playing = isAllCameras ? gridPlaying : autoplayRef.current;
    if (!playing) return;
    if (timelinePlayheadInComfortZone(timelineView, playhead)) return;
    setTimelineViewportAnchor(playhead);
  }, [gridPlaying, isAllCameras, playhead, timelineView]);

  // Event Rail, camera lanes and catch-up share these. The rail top is "now"
  // on today (rounded to five minutes so it does not shift every render).
  const railNow = date === today ? Math.ceil(Math.max(Date.now() / 1000, latestAvailabilityRef.current || dayStart) / 300) * 300 : dayEnd;
  const railTopEpoch = Math.min(dayEnd, railNow);
  const railPlayhead = Number.isFinite(playhead) ? playhead : (latestAvailabilityRef.current ?? dayStart);
  const previousRailEvent = useMemo(() => [...filteredEvents].reverse().find((event) => event.incident_epoch < railPlayhead - 2) ?? null, [filteredEvents, railPlayhead]);
  const nextRailEvent = useMemo(() => filteredEvents.find((event) => event.incident_epoch > railPlayhead + 2) ?? null, [filteredEvents, railPlayhead]);
  const storyItems = useMemo(() => condenseStory(filteredEvents, { quietSeconds: quietMinutes * 60, startEpoch: dayStart, endEpoch: railTopEpoch }), [dayStart, filteredEvents, quietMinutes, railTopEpoch]);
  const storyCurrent = timelineLayout === "catchup" && Number.isFinite(playhead) ? storyEventAt(storyItems, playhead, 3, selectedEventId) : null;
  const storyFocus = storyCurrent || selectedEvent || nextStoryEvent(storyItems, railPlayhead);
  const laneScope = `${dayStart}:${dayEnd}:${source}`;
  const laneEvents = useMemo(() => (laneData.scope === laneScope ? laneData.events : [])
    .filter((event) => timelineEventMatchesFilter(event, eventFilter))
    .map((event) => ({ ...event, incident_epoch: recordingIncidentEpoch(event) }))
    .filter((event) => Number.isFinite(event.incident_epoch)), [eventFilter, laneData, laneScope]);

  useVisiblePolling(async (signal) => {
    const response = await fetch(recordingGridDayUrl(dayStart, dayEnd, source, true), { signal });
    if (!response.ok) throw new Error(`Camera lanes failed (${response.status})`);
    const payload = await response.json();
    setLaneData({ scope: `${dayStart}:${dayEnd}:${source}`, recordings: payload.availability || [], events: payload.incidents || payload.events || [] });
  }, 30_000, timelineLayout === "cameras", { restartKey: laneScope });

  // Catch-up skips quiet stretches. Playback that runs off the end of an event,
  // or starts inside a quiet stretch, continues at the next event; a seek the
  // user made into a quiet stretch (or a clip preview) plays as asked.
  const storyTickRef = useRef(null);
  const pagePlaying = isAllCameras ? gridPlaying : heroPlaying;
  useEffect(() => {
    const previous = storyTickRef.current;
    storyTickRef.current = { epoch: playhead, inEvent: Boolean(storyCurrent), playing: pagePlaying };
    if (timelineLayout !== "catchup" || !Number.isFinite(playhead) || !pagePlaying) return;
    if (storyCurrent) {
      if (String(storyCurrent.id) !== String(selectedEventId)) setSelectedEventId(storyCurrent.id);
      return;
    }
    if (clipPreviewing) return;
    const ranOffEvent = previous?.inEvent && previous.playing && playhead > previous.epoch && playhead - previous.epoch < 3;
    const startedInQuiet = previous && !previous.playing;
    if (!ranOffEvent && !startedInQuiet) return;
    const next = nextStoryEvent(storyItems, playhead);
    if (!next) return;
    const target = Math.max(dayStart, next.incident_epoch - 1);
    // An event without recording would snap back here; leave playback alone.
    if ((snapToRecording(target) ?? playhead) <= playhead + 1) return;
    setSelectedEventId(next.id);
    playAt(target, true);
    // playAt and snapToRecording are recreated every render; the playhead drives this effect.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clipPreviewing, pagePlaying, playhead, storyCurrent?.id, storyItems, timelineLayout]);

  useVisiblePolling(async (signal) => {
    const response = await fetch("/api/semantic-search/status", { signal });
    if (!response.ok) throw new Error(`Semantic search status failed (${response.status})`);
    setSemanticStatus(await response.json());
  }, 15_000);

  useEffect(() => {
    const image = frameSearchImageRef.current;
    const canvas = frameSearchCropPreviewRef.current;
    if (!image || !canvas || !frameSearchCrop || !image.naturalWidth || !image.naturalHeight) return;
    drawContainedCropPreview(canvas, image, frameSearchCrop);
  }, [frameSearchCrop, frameSearchImageReady]);

  useEffect(() => {
    if (!Number.isFinite(frameSearchEpoch)) return undefined;
    const surface = frameSearchSurfaceRef.current;
    const image = frameSearchImageRef.current;
    if (!surface) return undefined;
    const updateBounds = () => setFrameSearchImageBounds(
      containedImageBounds(frameSearchSurfaceRef.current, frameSearchImageRef.current)
    );
    updateBounds();
    window.addEventListener("resize", updateBounds);
    const resizeObserver = typeof ResizeObserver === "function"
      ? new ResizeObserver(updateBounds)
      : null;
    resizeObserver?.observe(surface);
    if (image) resizeObserver?.observe(image);
    return () => {
      window.removeEventListener("resize", updateBounds);
      resizeObserver?.disconnect();
    };
  }, [frameSearchEpoch, frameSearchImageReady]);

  useVisiblePolling(async (signal) => {
    try {
      const next = await fetchExportJob(exportJob?.id, fetch, { signal, maxAgeMs: 750 });
      if (next) {
        setExportJob(next);
        setExportError("");
      }
    } catch (error) {
      if (error?.name !== "AbortError") setExportError(error.message || "Unable to refresh export status");
    }
  }, 1_000, exportIsActive(exportJob), { restartKey: exportJob?.id || "" });

  useEffect(() => {
    cancelClipPreview();
    setExportRange(null);
    setExportJob(null);
    setExportError("");
    setExportLabel("");
    selectedIncidentIdentityCacheRef.current.clear();
    setSelectedIncidentIdentity(null);
    setFrameSearchEpoch(null);
    setFrameSearchCrop(null);
    setFrameSearchError("");
  }, [activeCameraId, date, isAllCameras, source]);

  useEffect(() => {
    if (!activeCameraId) return undefined;
    const controller = new AbortController();
    fetch(appUrl("/api/exports?limit=100"), { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Export history failed (${response.status})`);
        return response.json();
      })
      .then((payload) => {
        cacheExportJobs(payload.exports || []);
        const active = (payload.exports || []).find((job) => (
          job.camera_id === activeCameraId
          && job.source === source
          && ACTIVE_EXPORT_STATUSES.includes(job.status)
        ));
        if (!active || controller.signal.aborted) return;
        setExportKind(active.kind === "timelapse" ? "timelapse" : "recording");
        setExportRange({ start: Number(active.start_epoch), end: Number(active.end_epoch) });
        setExportOptions({
          interval: Number(active.options?.sample_interval_seconds) || 30,
          fps: Number(active.options?.output_fps) || 30,
          clipHeight: active.kind === "recording" ? Number(active.options?.height) || 0 : 0,
          timelapseHeight: active.kind === "timelapse" ? Number(active.options?.height) || 720 : 720,
        });
        setExportLabel(active.label || "");
        setExportJob(active);
      })
      .catch((error) => {
        if (error.name !== "AbortError") console.warn("Unable to restore active export", error);
      });
    return () => controller.abort();
  }, [activeCameraId, date, source]);

  // Every timeline layout reports its scrub position here. The frames go
  // straight to the overlay: a scrub reports on every scroll event, and this
  // page must not render for each one.
  const scrubPlayer = useScrubStrips({
    cameraId: isAllCameras ? "" : activeCameraId,
    source,
    onFrame: (frame, options) => scrubOverlayRef.current?.show(frame, options),
  });

  // Frame stepping of the paused player (one camera only). The stepped frame
  // is drawn over the player from the server; once stepping pauses the
  // player is seeked there too, so play resumes from it. Anything else that
  // moves playback ends stepping (playAt, a scrub, another camera or day).
  const frameStepSeekRef = useRef(false);
  const frameStep = useFrameStepper({
    cameraId: isAllCameras ? "" : activeCameraId,
    source,
    enabled: !isAllCameras && !heroPlaying && hasPlaybackMedia && !playbackError
      && Number.isFinite(playhead) && !Number.isFinite(frameSearchEpoch),
    currentEpoch: currentPlaybackEpoch,
    onSettle: (epoch) => seekToSteppedFrame(epoch, false),
    measure: () => heroMediaRef.current?.clientWidth,
  });
  const timelineScrub = useCallback((epoch, options) => {
    if (Number.isFinite(epoch)) frameStep.leave();
    scrubPlayer(epoch, options);
  }, [frameStep, scrubPlayer]);
  function seekToSteppedFrame(epoch, autoplay) {
    frameStepSeekRef.current = true;
    try {
      playAt(epoch, autoplay);
    } finally {
      frameStepSeekRef.current = false;
    }
  }
  useEffect(() => {
    if (heroPlaying && !heroSeeking) frameStep.leave();
  }, [frameStep, heroPlaying, heroSeeking]);
  useEffect(() => { frameStep.leave(); }, [date, frameStep]);

  function snapToRecording(epoch, continuing = false) {
    if (!timeline.length) return null;
    const bounded = Math.max(dayStart, Math.min(dayEnd - 0.01, epoch));
    const latest = timeline[timeline.length - 1];
    const edge = recordingEdgeSeekEpoch(
      latest, Date.now() / 1000, continuing, liveEdgeFollow && nativeHls && !useSegmentPlayback,
    );
    if (bounded >= edge) return edge;
    const containing = timeline.find((item) => item.start_epoch <= bounded && bounded < item.end_epoch);
    if (containing) return bounded;
    let nearest = timeline[0].start_epoch;
    let distance = Math.abs(nearest - bounded);
    timeline.forEach((item) => {
      [item.start_epoch, item.end_epoch - 0.01].forEach((candidate) => {
        const nextDistance = Math.abs(candidate - bounded);
        if (nextDistance < distance) {
          nearest = candidate;
          distance = nextDistance;
        }
      });
    });
    return nearest;
  }

  function mediaTimeToEpoch(mediaTime) {
    const clip = playbackTimeline.find((item) => item.media_start <= mediaTime && mediaTime < item.media_end)
      || playbackTimeline[playbackTimeline.length - 1];
    if (!clip) return null;
    return clip.start_epoch + Math.max(0, Math.min(clip.end_epoch - clip.start_epoch, mediaTime - clip.media_start));
  }

  function epochToPlaybackMediaTime(epoch) {
    if (!playbackTimeline.length) return null;
    const clip = playbackTimeline.find((item) => item.start_epoch <= epoch && epoch < item.end_epoch)
      || playbackTimeline.reduce((nearest, item) => {
        if (!nearest) return item;
        return Math.abs(item.start_epoch - epoch) < Math.abs(nearest.start_epoch - epoch) ? item : nearest;
      }, null);
    return clip.media_start + Math.max(0, Math.min(clip.media_end - clip.media_start - 0.01, epoch - clip.start_epoch));
  }

  function windowAround(epoch) {
    const windowSeconds = 15 * 60;
    const bucket = Math.floor(Math.max(0, epoch - dayStart) / windowSeconds);
    const start = dayStart + bucket * windowSeconds;
    return {
      start,
      end: Math.min(dayEnd, start + windowSeconds),
    };
  }

  function requestPlaybackWindow(nextWindow) {
    setPlaybackWindow(nextWindow);
    setPlaybackWindowRevision((revision) => revision + 1);
  }

  function clearSeekWatchdog() {
    if (seekWatchdogRef.current) {
      window.clearTimeout(seekWatchdogRef.current);
      seekWatchdogRef.current = null;
    }
  }

  function scheduleSeekWatchdog(video, mediaTime, options = {}) {
    clearSeekWatchdog();
    if (!video || !Number.isFinite(mediaTime)) return;
    const tolerance = recordingSeekToleranceSeconds();
    const requestedSource = video.getAttribute("src");
    const local = Boolean(options.local);
    const isCurrent = () => video === videoRef.current && video.getAttribute("src") === requestedSource;
    let waits = 0;
    let wrote = false;
    const retryDelay = local ? 400 : (prefersJpegScrubPreview() ? 400 : 150);
    const tick = () => {
      seekWatchdogRef.current = null;
      if (!isCurrent() || !Number.isFinite(pendingSeekEpochRef.current)) return;
      const pendingMode = pendingSeekModeRef.current;
      if (pendingMode !== "local" && pendingMode !== "window-ready") return;
      if (video.seeking && waits < (local ? 8 : 4)) {
        waits += 1;
        seekWatchdogRef.current = window.setTimeout(tick, retryDelay);
        return;
      }
      if (!videoReachedSeekTarget(video, mediaTime, tolerance)) {
        if (!wrote && !video.seeking) {
          wrote = true;
          video.currentTime = mediaTime;
          seekWatchdogRef.current = window.setTimeout(tick, retryDelay);
          return;
        }
      }
      completePendingRecordingSeek(video, {
        fromSeeked: Boolean(video.seeking) && videoReachedSeekTarget(video, mediaTime, tolerance),
      });
    };
    seekWatchdogRef.current = window.setTimeout(
      tick,
      local ? seekWatchdogDelayMs({ local: true }) : seekWatchdogDelayMs(),
    );
  }

  function completePendingRecordingSeek(video, options = {}) {
    if (video !== videoRef.current) return;
    // seeked can fire while iPhone still reports seeking. Trust that event
    // once currentTime matches; other callers must wait so play() does not
    // abort the seek that is still in flight.
    if (video.seeking && !options.fromSeeked) return;
    const pendingMode = pendingSeekModeRef.current;
    if (pendingMode !== "local" && pendingMode !== "window-ready") return;
    // A delayed seeked event can belong to the previous scrub on this same video.
    const target = epochToPlaybackMediaTime(pendingSeekEpochRef.current);
    if (!videoReachedSeekTarget(video, target, recordingSeekToleranceSeconds())) return;
    clearSeekWatchdog();
    const epoch = mediaTimeToEpoch(video.currentTime);
    if (Number.isFinite(epoch)) {
      desiredEpochRef.current = epoch;
      setPlayhead(epoch);
    }
    pendingSeekEpochRef.current = null;
    pendingSeekModeRef.current = null;
    setHeroSeeking(false);
    setPlaybackNotice("");
    ignorePauseUntilRef.current = performance.now() + ignorePauseAfterSeekMs();
    if (shouldResumePlaybackAfterSeek({ pendingSeekMode: pendingMode, autoplay: autoplayRef.current }) && video.paused) {
      requestRecordingPlay(video);
    }
  }

  function completePendingNativeSeek(video, options = {}) {
    if (video !== videoRef.current) return;
    if (video.seeking && !options.fromSeeked) return;
    const pendingMode = pendingSeekModeRef.current;
    if (pendingMode !== "native-local" && pendingMode !== "native-ready") return;
    const target = recordingSegmentLocalTime(nativeSegment, pendingSeekEpochRef.current, video);
    if (!videoReachedSeekTarget(video, target, recordingSeekToleranceSeconds({ preferNativeHls: true }))) return;
    clearSeekWatchdog();
    const segment = nativeSegment;
    if (segment) {
      const epoch = segment.start_epoch + Math.max(0, Number(video.currentTime) || 0);
      desiredEpochRef.current = epoch;
      setPlayhead(epoch);
    }
    pendingSeekEpochRef.current = null;
    pendingSeekModeRef.current = null;
    setHeroSeeking(false);
    setPlaybackNotice("");
    ignorePauseUntilRef.current = performance.now() + ignorePauseAfterSeekMs();
    if (autoplayRef.current && video.paused) requestRecordingPlay(video);
  }

  function scheduleNativeSeekWatchdog(video, localTime, options = {}) {
    clearSeekWatchdog();
    if (!video || !Number.isFinite(localTime)) return;
    const tolerance = recordingSeekToleranceSeconds({ preferNativeHls: true });
    const requestedSource = video.getAttribute("src");
    const local = Boolean(options.local);
    const isCurrent = () => video === videoRef.current && video.getAttribute("src") === requestedSource;
    let waits = 0;
    let wrote = false;
    const tick = () => {
      seekWatchdogRef.current = null;
      if (!isCurrent() || !Number.isFinite(pendingSeekEpochRef.current)) return;
      // currentTime written while seeking is ignored on iPhone. Wait for the
      // in-flight seek to finish, then write the queued target once.
      if (video.seeking && waits < (local ? 8 : 4)) {
        waits += 1;
        seekWatchdogRef.current = window.setTimeout(tick, 400);
        return;
      }
      if (!videoReachedSeekTarget(video, localTime, tolerance)) {
        if (!wrote && !video.seeking) {
          wrote = true;
          video.currentTime = localTime;
          seekWatchdogRef.current = window.setTimeout(tick, 400);
          return;
        }
      }
      completePendingNativeSeek(video, {
        fromSeeked: Boolean(video.seeking) && videoReachedSeekTarget(video, localTime, tolerance),
      });
    };
    seekWatchdogRef.current = window.setTimeout(
      tick,
      seekWatchdogDelayMs({ preferNativeHls: true, local }),
    );
  }

  function handleNativeSegmentMetadata(event) {
    const video = event.currentTarget;
    if (!nativeSegment) return;
    clearSeekWatchdog();
    video.playbackRate = normalizedTimelinePlaybackRate(playbackRate);
    if (playbackRetryRef.current.timer) window.clearTimeout(playbackRetryRef.current.timer);
    playbackRetryRef.current = { attempts: 0, timer: null };
    const retained = Number.isFinite(pendingSeekEpochRef.current)
      ? pendingSeekEpochRef.current
      : desiredEpochRef.current;
    const target = Number.isFinite(retained)
      ? Math.max(nativeSegment.start_epoch, Math.min(nativeSegment.end_epoch - 0.01, retained))
      : nativeSegment.start_epoch;
    const localTime = recordingSegmentLocalTime(nativeSegment, target, video);
    const seekRequired = Number.isFinite(localTime) && Math.abs(video.currentTime - localTime) > 0.05;
    desiredEpochRef.current = target;
    setPlayhead(target);
    setPlaybackError("");
    setPlaybackErrorStage("");
    if (seekRequired) {
      pendingSeekEpochRef.current = target;
      pendingSeekModeRef.current = "native-ready";
      setPlaybackNotice("Seeking...");
      seekVideoToTime(video, localTime, { allowFastSeek: false });
      scheduleNativeSeekWatchdog(video, localTime);
      return;
    }
    pendingSeekEpochRef.current = null;
    pendingSeekModeRef.current = null;
    setHeroSeeking(false);
    setPlaybackNotice("");
    if (autoplayRef.current) requestRecordingPlay(video);
  }

  function handleNativeSegmentTimeUpdate(event) {
    if (!nativeSegment) return;
    if (Number.isFinite(pendingSeekEpochRef.current)) {
      const localTime = recordingSegmentLocalTime(nativeSegment, pendingSeekEpochRef.current, event.currentTarget);
      if (videoReachedSeekTarget(event.currentTarget, localTime, recordingSeekToleranceSeconds({ preferNativeHls: true }))) {
        completePendingNativeSeek(event.currentTarget);
      }
      return;
    }
    const epoch = nativeSegment.start_epoch + Math.max(0, Number(event.currentTarget.currentTime) || 0);
    desiredEpochRef.current = epoch;
    setPlayhead(epoch);
    finishClipPreviewAtEnd(event.currentTarget, epoch);
  }

  function requestRecordingPlay(video, showBlocked = true) {
    if (!video) return;
    video.play().then(() => {
      setPlaybackBlocked(false);
      setHeroPlaying(true);
    }).catch((error) => {
      if (showBlocked && error?.name === "NotAllowedError") {
        setHeroPlaying(false);
        setPlaybackBlocked(true);
        setPlaybackNotice("Tap to play recording");
      }
    });
  }

  function playAt(epoch, autoplay = true, preserveTimelineViewport = false, fastSeek = false, continuing = false) {
    if (!frameStepSeekRef.current) frameStep.leave();
    const target = snapToRecording(epoch, continuing);
    if (target === null || !activeCameraId) return;
    if (
      !preserveTimelineViewport
      && (
      target < timelineView.startEpoch
      || target > timelineView.endEpoch
      )
    ) {
      setTimelineViewportAnchor(target);
    }
    if (isAllCameras) {
      autoplayRef.current = autoplay;
      setGridPlaying(autoplay);
      setPlaybackError("");
      setPlaybackErrorStage("");
      setPlaybackNotice("");
      setPlayhead(target);
      desiredEpochRef.current = target;
      return;
    }
    clearSeekWatchdog();
    setHeroSeeking(true);
    autoplayRef.current = autoplay;
    setFollowTarget(null);
    setPlaybackError("");
    setPlaybackErrorStage("");
    setPlayhead(target);
    desiredEpochRef.current = target;
    const nextWindow = windowAround(target);
    const inCurrentWindow = loadedPlaybackWindow
      && target >= loadedPlaybackWindow.start
      && target < loadedPlaybackWindow.end;
    const coveredByCurrentManifest = playbackRowsCoverEpoch(playbackTimeline, target);
    const video = videoRef.current;
    const previousSeekMode = pendingSeekModeRef.current;
    // Mark the seek before any pause so an ended event cannot start the next clip.
    // Do not pause a playing element for an in-window seek: pause()+play()+currentTime
    // is the iPhone sequence that drops the seek or leaves the video paused.
    pendingSeekEpochRef.current = target;
    pendingSeekModeRef.current = "window";
    if (useSegmentPlayback) {
      clearSeekWatchdog();
      if (!autoplay) setHeroPlaying(false);
      const warmWindow = prefetchedNativeWindow && target >= prefetchedNativeWindow.start && target < prefetchedNativeWindow.end
        ? prefetchedNativeWindow : null;
      const recording = (inCurrentWindow ? recordingSegmentAt(playbackTimeline, target) : null)
        || recordingSegmentAt(warmWindow?.rows, target);
      if (recording) {
        if (!inCurrentWindow && warmWindow) setPlaybackDetail({ ...warmWindow, revision: ++playbackRequestRef.current });
        const segment = { start_epoch: recording.start_epoch, end_epoch: recording.end_epoch };
        const sameSegment = nativeSegment
          && nativeSegment.start_epoch === segment.start_epoch
          && nativeSegment.end_epoch === segment.end_epoch;
        if (sameSegment && video && video.readyState >= 1) {
          const localTime = recordingSegmentLocalTime(segment, target, video);
          pendingSeekEpochRef.current = target;
          pendingSeekModeRef.current = "native-local";
          playbackRequestRef.current += 1;
          setPlaybackWindow(null);
          setPlaybackNotice(autoplay ? "Seeking..." : "");
          // iOS applies the seek only after pause, and only honors play() in this gesture.
          video.pause();
          if (!video.seeking) seekVideoToTime(video, localTime, { allowFastSeek: false });
          if (autoplay) requestRecordingPlay(video, false);
          scheduleNativeSeekWatchdog(video, localTime, { local: true });
          if (!autoplay) setPlaybackNotice("");
        } else {
          video?.pause();
          pendingSeekEpochRef.current = target;
          pendingSeekModeRef.current = "native-ready";
          setPlaybackWindow(null);
          setPlaybackNotice("Loading recording...");
          setNativeSegment({
            start_epoch: segment.start_epoch,
            end_epoch: segment.end_epoch,
          });
        }
      } else {
        video?.pause();
        pendingSeekEpochRef.current = target;
        pendingSeekModeRef.current = "window";
        setPlaybackNotice("Loading recording...");
        requestPlaybackWindow(nextWindow);
      }
      return;
    }
    const mediaTime = epochToPlaybackMediaTime(target);
    const canSeekWithinManifest = !nativeHls || (video?.readyState >= 2
      && previousSeekMode !== "playlist-start" && previousSeekMode !== "window" && previousSeekMode !== "window-ready");
    if (inCurrentWindow && coveredByCurrentManifest && canSeekWithinManifest && video && Number.isFinite(mediaTime)) {
      pendingSeekEpochRef.current = target;
      pendingSeekModeRef.current = "local";
      playbackRequestRef.current += 1;
      setPlaybackWindow(null);
      setPlaybackNotice(autoplay ? "Seeking..." : "");
      // iOS applies the seek only after pause, and only honors play() in this gesture.
      video.pause();
      if (!video.seeking) seekVideoToTime(video, mediaTime, { allowFastSeek: !nativeHls });
      if (autoplay) requestRecordingPlay(video, false);
      scheduleSeekWatchdog(video, mediaTime, { local: true });
      if (!autoplay) setPlaybackNotice("");
    } else {
      video?.pause();
      pendingSeekEpochRef.current = target;
      pendingSeekModeRef.current = "window";
      setPlaybackNotice("Loading recording...");
      requestPlaybackWindow(nextWindow);
    }
  }

  function selectPlaybackCamera(camera) {
    checkpointTimelineView();
    autoplayRef.current = true;
    ignorePauseUntilRef.current = performance.now() + 2000;
    if (videoRef.current) requestRecordingPlay(videoRef.current, false);
    // The grid plays every camera at once; it starts on the lighter Sub stream.
    if (camera === ALL_RECORDING_CAMERAS_ID && camera !== cameraId) setSource("live");
    setCameraId(camera);
  }


  useEffect(() => {
    if (!isAllCameras || !gridPlaying || !timeline.length) return undefined;
    let previous = performance.now();
    const timer = window.setInterval(() => {
      const now = performance.now();
      const elapsed = Math.max(0, Math.min(2, (now - previous) / 1000));
      previous = now;
      setPlayhead((current) => {
        const base = Number.isFinite(current) ? current : timeline[0].start_epoch;
        const requested = base + (elapsed * playbackRate);
        const containing = timeline.find((item) => item.start_epoch <= requested && requested < item.end_epoch);
        let next = containing ? requested : null;
        if (!Number.isFinite(next)) {
          const following = timeline.find((item) => item.end_epoch > requested);
          next = following ? Math.max(requested, following.start_epoch) : null;
        }
        if (!Number.isFinite(next) || next >= dayEnd) {
          autoplayRef.current = false;
          setGridPlaying(false);
          return base;
        }
        desiredEpochRef.current = next;
        return next;
      });
    }, 500);
    return () => window.clearInterval(timer);
  }, [dayEnd, gridPlaying, isAllCameras, playbackRate, timeline]);

  useEffect(() => {
    if (!isAllCameras || !gridPlaying) return undefined;
    const pauseWhenHidden = () => {
      if (!document.hidden) return;
      autoplayRef.current = false;
      setGridPlaying(false);
    };
    document.addEventListener("visibilitychange", pauseWhenHidden);
    return () => document.removeEventListener("visibilitychange", pauseWhenHidden);
  }, [gridPlaying, isAllCameras]);

  useEffect(() => {
    setCameras(sharedCameras);
    setCameraTransitionRoutes(appConfig?.detector?.tracking?.camera_transition_routes || []);
  }, [appConfig, sharedCameras]);

  useEffect(() => {
    if (!activeCameraId) return undefined;
    const controller = new AbortController();
    setLoading(true);
    setPlaybackError("");
    setPlaybackErrorStage("");
    setPlaybackBlocked(false);
    ignorePauseUntilRef.current = performance.now() + 2000;
    setHeroPlaying(false);
    setHeroSeeking(false);
    setRecordings([]);
    playbackRequestRef.current += 1;
    setPlaybackDetail(null);
    setEvents([]);
    setAvailableSources([]);
    setPlaybackWindow(null);
    setGridPlaying(false);
    gridRefreshCursorRef.current = null;
    setManifestRetryToken(0);
    setNativeSegment(null);
    setNativePrefetchDetail(null);
    setNativeSegmentRetryToken(0);
    setFollowTarget(null);
    clearSeekWatchdog();
    if (Number.isFinite(playhead)) desiredEpochRef.current = playhead;
    // Ignore outgoing media events while the new camera/day index is loading.
    pendingSeekEpochRef.current = Number.isFinite(desiredEpochRef.current) ? desiredEpochRef.current : dayStart;
    pendingSeekModeRef.current = "window";
    setPlayhead(null);
    const video = videoRef.current;
    if (video) {
      video.pause();
    }
    codecFallbackRef.current = false;
    originalFallbackRef.current = null;
    setOriginalScope(null);
    transcodeFallbackRef.current = null;
    setTranscodeScope(null);
    if (playbackRetryRef.current.timer) window.clearTimeout(playbackRetryRef.current.timer);
    playbackRetryRef.current = { attempts: 0, timer: null };
    const indexUrl = isAllCameras
      ? recordingGridDayUrl(dayStart, dayEnd, source, false)
      : recordingDayUrl(activeCameraId, dayStart, dayEnd, source, false);
    fetch(indexUrl, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Recording index failed (${response.status})`);
        return response.json();
      })
      .then((payload) => {
        const nextAvailableSources = payload.available_sources || [];
        const nextAvailability = payload.availability || payload.recordings || [];
        setAvailableSources(nextAvailableSources);
        if (!nextAvailability.length && nextAvailableSources.includes(source === "main" ? "live" : "main")
          && !(source === "main" && failedSubScopeRef.current === `${activeCameraId}:${dayStart}:${dayEnd}`)) {
          codecFallbackRef.current = true;
          setPlaybackNotice(`No ${source === "main" ? "Main" : "Sub"} recordings exist for this day; using ${source === "main" ? "Sub" : "Main"}.`);
          setSource(source === "main" ? "live" : "main");
          return;
        }
        setRecordings(nextAvailability);
        setEvents(payload.incidents || payload.events || []);
        if (isAllCameras) gridRefreshCursorRef.current = Date.now() / 1000;
      })
      .catch((error) => {
        if (error.name !== "AbortError") {
          setPlaybackErrorStage("index");
          setPlaybackError(error.message || "Unable to load recordings");
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => {
      controller.abort();
      if (playbackRetryRef.current.timer) window.clearTimeout(playbackRetryRef.current.timer);
      playbackRetryRef.current = { attempts: 0, timer: null };
    };
  }, [activeCameraId, isAllCameras, source, dayStart, dayEnd, recordingIndexRevision]);

  useVisiblePolling(async (signal) => {
    if (recordingUpdatesInFlightRef.current) return;
    recordingUpdatesInFlightRef.current = true;
    try {
      const afterEpoch = Number.isFinite(latestAvailabilityRef.current)
        ? latestAvailabilityRef.current
        : dayStart;
      const response = await fetch(recordingUpdatesUrl(activeCameraId, dayStart, dayEnd, afterEpoch, source, false), { signal });
      if (!response.ok) throw new Error(`Recording update failed (${response.status})`);
      const payload = await response.json();
      const additions = payload.availability || [];
      if (additions.length) setRecordings((current) => mergeRecordingAvailability(current, additions));
      const eventUpdates = payload.incidents || payload.events || [];
      if (eventUpdates.length) setEvents((current) => mergeRecordingEvents(current, eventUpdates));
    } catch {
      // The next poll retries without disrupting active playback.
    } finally {
      recordingUpdatesInFlightRef.current = false;
    }
  }, 10_000, Boolean(activeCameraId) && !isAllCameras && date === today, {
    immediate: false,
    restartKey: `${activeCameraId}\u0000${source}\u0000${dayStart}\u0000${dayEnd}`,
  });

  useVisiblePolling(async (signal) => {
    if (gridUpdatesInFlightRef.current) return;
    gridUpdatesInFlightRef.current = true;
    try {
      const requestStartedAt = Date.now() / 1000;
      const afterEpoch = Number.isFinite(gridRefreshCursorRef.current)
        ? gridRefreshCursorRef.current
        : Math.max(dayStart, requestStartedAt - 120);
      const response = await fetch(recordingGridUpdatesUrl(dayStart, dayEnd, afterEpoch, source, false), { signal });
      if (!response.ok) return;
      const payload = await response.json();
      const additions = payload.availability || payload.recordings || [];
      if (additions.length) setRecordings((current) => mergeRecordingAvailability(current, additions));
      const eventUpdates = payload.incidents || payload.events || [];
      if (eventUpdates.length) setEvents((current) => mergeRecordingEvents(current, eventUpdates));
      gridRefreshCursorRef.current = requestStartedAt;
    } catch {
      // The next refresh retries without interrupting synchronized playback.
    } finally {
      gridUpdatesInFlightRef.current = false;
    }
  }, 10_000, isAllCameras && date === today, {
    immediate: false,
    restartKey: `${source}\u0000${dayStart}\u0000${dayEnd}`,
  });

  useEffect(() => {
    if (!activeCameraId || isAllCameras || !playbackWindow) return undefined;
    const controller = new AbortController();
    const requestId = ++playbackRequestRef.current;
    const requestedWindow = { ...playbackWindow };
    const cacheKey = `${nativeScope}:${requestedWindow.start}:${requestedWindow.end}`;
    async function loadWindow() {
      const cache = playbackWindowCacheRef.current;
      const cached = cache.get(cacheKey);
      const nearLive = requestedWindow.end >= Date.now() / 1000 - 60;
      if (cached && performance.now() - cached.at < (nearLive ? 2000 : 30000)) return cached.payload;
      const response = await fetch(
        recordingWindowUrl(activeCameraId, requestedWindow.start, requestedWindow.end, source),
        { signal: controller.signal },
      );
      if (!response.ok) throw new Error(`Recording window failed (${response.status})`);
      const payload = await response.json();
      if (!controller.signal.aborted) {
        cache.delete(cacheKey);
        cache.set(cacheKey, { at: performance.now(), payload });
        if (cache.size > 8) cache.delete(cache.keys().next().value);
      }
      return payload;
    }
    loadWindow()
      .then((payload) => {
        if (controller.signal.aborted || requestId !== playbackRequestRef.current) return;
        const rows = payload.recordings || [];
        if (!rows.length) throw new Error("No recording segments exist in this window");
        const requestedEpoch = Number.isFinite(pendingSeekEpochRef.current)
          ? pendingSeekEpochRef.current : desiredEpochRef.current;
        const playableEpoch = recordingPlayableEpoch(rows, requestedEpoch);
        if (Number.isFinite(playableEpoch)) {
          pendingSeekEpochRef.current = playableEpoch;
          pendingSeekModeRef.current = "window";
          desiredEpochRef.current = playableEpoch;
          setPlayhead(playableEpoch);
        }
        setPlaybackDetail({
          scope: nativeScope,
          start: Number(payload.start_epoch),
          end: Number(payload.end_epoch),
          rows,
          revision: requestId,
          // Opt-in: a window that is still being recorded plays as a growing
          // playlist, so native HLS follows it without loading a new one.
          growing: liveEdgeFollow && nativeHls && Number(payload.end_epoch) > Date.now() / 1000,
        });
        if (playbackTransportRef.current?.scope === nativeScope && playbackTransportRef.current.mode !== "hls") {
          const target = Number.isFinite(pendingSeekEpochRef.current)
            ? pendingSeekEpochRef.current
            : desiredEpochRef.current;
          const segment = recordingSegmentAt(rows, target);
          if (!segment) throw new Error("No recording segment exists at this time");
          pendingSeekModeRef.current = "native-ready";
          setNativeSegment({ start_epoch: Number(segment.start_epoch), end_epoch: Number(segment.end_epoch) });
        }
      })
      .catch((error) => {
        if (error.name !== "AbortError" && requestId === playbackRequestRef.current) {
          pendingSeekEpochRef.current = null;
          pendingSeekModeRef.current = null;
          setPlaybackNotice("");
          setHeroSeeking(false);
          setPlaybackErrorStage("window");
          setPlaybackError(error.message || "Unable to load recording window");
        }
      });
    return () => controller.abort();
  }, [activeCameraId, isAllCameras, source, playbackWindow?.start, playbackWindow?.end, playbackWindowRevision]);

  useEffect(() => {
    if (!timeline.length || Number.isFinite(playhead)) return;
    const retainedEpoch = desiredEpochRef.current;
    let initialEpoch = Number.isFinite(retainedEpoch) && retainedEpoch >= dayStart && retainedEpoch < dayEnd
      ? retainedEpoch
      : date === today ? Date.now() / 1000 : timeline[0].start_epoch;
    if (isAllCameras && !Number.isFinite(retainedEpoch) && date === today) {
      // Start behind the newest indexed recording: at that edge every tile
      // would reload as each new recording file lands.
      const newest = timeline.reduce((latest, item) => Math.max(latest, item.end_epoch), -Infinity);
      const behindLive = Math.max(dayStart, Math.min(initialEpoch, newest) - ALL_CAMERAS_LIVE_DELAY_SECONDS);
      initialEpoch = recordingGridBestEpoch(timeline, behindLive) ?? behindLive;
    }
    playAt(initialEpoch, autoplayRef.current);
  }, [date, dayEnd, dayStart, isAllCameras, playhead, timeline, today]);

  useEffect(() => {
    if (!Number.isFinite(followTarget) || !timeline.length) return;
    const nextEpoch = recordingEpochAfterSegment({ end_epoch: followTarget - 0.01 }, timeline);
    if (nextEpoch === null) return;
    playAt(nextEpoch, true, false, false, true);
  }, [followTarget, timeline]);

  useEffect(() => {
    if (!activeCameraId) return;
    const params = new URLSearchParams({ camera: isAllCameras ? ALL_RECORDING_CAMERAS_ID : activeCameraId, date, source });
    const playbackDebug = initialQuery.get("playbackdebug");
    if (playbackDebug === "0" || playbackDebug === "1") params.set("playbackdebug", playbackDebug);
    const retainedEpoch = desiredEpochRef.current;
    if (Number.isFinite(retainedEpoch) && retainedEpoch >= dayStart && retainedEpoch < dayEnd) {
      params.set("at", String(Math.round(retainedEpoch * 1000) / 1000));
    }
    if (eventFilter !== "all") params.set("filter", eventFilter);
    if (selectedEventId) params.set("event", String(selectedEventId));
    const trailValue = serializeTrailEventIds(trailEventIds);
    if (trailValue) params.set("trail", trailValue);
    if (incidentRangeHours !== 1) params.set("window", String(incidentRangeHours));
    if (playbackRate !== 1) params.set("speed", String(playbackRate));
    if (timelineLayout !== "rail") params.set("view", timelineLayout);
    if (railMinutes !== 60) params.set("rail", String(railMinutes));
    window.history.replaceState(null, "", appUrl(`/timeline?${params.toString()}`));
  }, [activeCameraId, date, dayEnd, dayStart, eventFilter, isAllCameras, incidentRangeHours, initialQuery, playbackRate, railMinutes, selectedEventId, source, timelineLayout, trailEventIds]);

  useEffect(() => {
    const restoreView = () => {
      const view = parseTimelineView(window.location.search, today);
      const params = new URLSearchParams(window.location.search);
      const restoredDate = !params.get("date") && view.at ? dateKeyForTimeZone(view.at * 1000, timeZone) : view.date;
      const restoredSource = view.source || (view.cameraId === ALL_RECORDING_CAMERAS_ID ? "live" : preferredStreamSource());
      const samePlaybackScope = view.cameraId === cameraId && restoredDate === date && restoredSource === source;
      setCameraId(view.cameraId);
      setDate(restoredDate);
      setSource(restoredSource);
      if (Number.isFinite(view.at)) {
        if (samePlaybackScope) playAt(view.at, false);
        else {
          desiredEpochRef.current = view.at;
          setPlayhead(view.at);
        }
      }
      setEventFilter(view.eventFilter);
      setSelectedEventId(view.eventId);
      setTrailEventIds(view.trailEventIds || []);
      if (view.eventId || view.trailEventIds?.length) setInvestigationOpen(true);
      setIncidentRangeHours(view.windowHours);
      setPlaybackRate(view.speed);
      setTimelineLayout(view.view);
      setRailMinutes(view.railMinutes);
    };
    window.addEventListener("popstate", restoreView);
    return () => window.removeEventListener("popstate", restoreView);
  }, [cameraId, date, source, timeZone, today]);

  function checkpointTimelineView() {
    window.history.pushState(null, "", window.location.href);
  }

  async function openFindSimilarResult(result) {
    const item = result?.event || {};
    const eventId = Number(item.id);
    if (!Number.isInteger(eventId) || eventId <= 0) return;
    let camera = String(item.camera_id || "");
    let epoch = Number(item.incident_epoch);
    if (!Number.isFinite(epoch) && item.created_at) {
      epoch = new Date(item.created_at).getTime() / 1000;
    }
    if (!camera || !Number.isFinite(epoch)) {
      try {
        const response = await fetch(`/api/incidents/by-event/${encodeURIComponent(eventId)}`);
        if (!response.ok) throw new Error("Result unavailable");
        const detail = await response.json();
        camera = String(detail.camera_id || camera);
        epoch = Number(detail.created_epoch);
        if (!Number.isFinite(epoch) && detail.created_at) {
          epoch = new Date(detail.created_at).getTime() / 1000;
        }
      } catch {
        return;
      }
    }
    if (!camera || !Number.isFinite(epoch)) return;
    const nextDate = dateKeyForTimeZone(epoch * 1000, timeZone);
    setTrailNotice("");
    setInvestigationOpen(true);
    setSelectedEventId(eventId);
    desiredEpochRef.current = epoch;
    setPlayhead(epoch);
    const sameScope = camera === cameraId && nextDate === date;
    if (camera !== cameraId) setCameraId(camera);
    if (nextDate !== date) setDate(nextDate);
    if (sameScope) playAt(epoch, true);
  }

  function stepForensicNav(direction) {
    if (direction < 0) {
      if (forensicNav.deselectOnPrevious) {
        setSelectedEventId(null);
        return;
      }
      if (forensicNav.previous) openFindSimilarResult(forensicNav.previous);
      return;
    }
    if (forensicNav.next) openFindSimilarResult(forensicNav.next);
  }

  function handleRecordingReady(_player, video) {
    if (video !== videoRef.current || useSegmentPlayback || transport !== requestedTransport || originalFallbackRef.current === nativeScope || transcodeFallbackRef.current === nativeScope) return;
    video.playbackRate = normalizedTimelinePlaybackRate(playbackRate);
    if (playbackRetryRef.current.timer) window.clearTimeout(playbackRetryRef.current.timer);
    playbackRetryRef.current = { attempts: 0, timer: null };
    const retained = Number.isFinite(pendingSeekEpochRef.current)
      ? pendingSeekEpochRef.current
      : desiredEpochRef.current;
    // A retained target was snapped when it was requested; do not move it again.
    const target = Number.isFinite(retained) ? snapToRecording(retained, true) : snapToRecording(Date.now() / 1000);
    // A speed change can mount HLS while a seek's new index window is pending.
    // Reject that old window, but allow gaps within the requested window: the
    // day overview merges short gaps and epochToPlaybackMediaTime snaps them
    // to recorded footage. Requiring exact row coverage leaves seeks pending.
    if (!loadedPlaybackWindow || !Number.isFinite(target)
      || target < loadedPlaybackWindow.start || target >= loadedPlaybackWindow.end) return;
    const mediaTime = epochToPlaybackMediaTime(target);
    if (nativeHls && Number.isFinite(mediaTime)) {
      if (pendingSeekModeRef.current === "local") return;
      // EXT-X-START owns native startup. A currentTime write during metadata
      // loading can interrupt the seek that the native player is already doing.
      pendingSeekEpochRef.current = target;
      pendingSeekModeRef.current = "playlist-start";
      nativePlaylistStartRef.current = {
        video, src: video.getAttribute("src"), ...nativePlaylistStart(mediaTime),
      };
      desiredEpochRef.current = target;
      setPlayhead(target);
      setPlaybackError("");
      setPlaybackErrorStage("");
      checkNativePlaylistStart({ currentTarget: video });
      if (autoplayRef.current && video.paused) requestRecordingPlay(video);
      return;
    }
    const atMediaTime = Number.isFinite(mediaTime) && Math.abs(video.currentTime - mediaTime) <= 0.05;
    const seekRequired = Number.isFinite(mediaTime) && (!atMediaTime || Boolean(video.seeking));
    if (seekRequired) {
      pendingSeekEpochRef.current = target;
      pendingSeekModeRef.current = "window-ready";
      setPlaybackNotice("Seeking...");
      if (!atMediaTime && !video.seeking) seekVideoToTime(video, mediaTime, { allowFastSeek: !nativeHls });
      scheduleSeekWatchdog(video, mediaTime);
    }
    if (Number.isFinite(target)) {
      desiredEpochRef.current = target;
      setPlayhead(target);
    }
    if (!seekRequired) {
      pendingSeekEpochRef.current = null;
      pendingSeekModeRef.current = null;
      setHeroSeeking(false);
      setPlaybackNotice("");
    }
    setPlaybackError("");
    setPlaybackErrorStage("");
    if (autoplayRef.current && !seekRequired) requestRecordingPlay(video);
  }

  function checkNativePlaylistStart(event) {
    if (pendingSeekModeRef.current !== "playlist-start") return;
    const video = event.currentTarget;
    const start = nativePlaylistStartRef.current;
    if (!start || video !== start.video || video !== videoRef.current
      || video.getAttribute("src") !== start.src || start.src !== manifestUrl) return;
    let step = nativePlaylistStartStep(start, video, event.type);
    if (step.action === "needs-play") {
      setPlaybackNotice("Play to finish positioning the recording.");
      return;
    }
    if (step.action === "correct") {
      try {
        seekVideoToTime(video, start.mediaTime, { allowFastSeek: false });
        return;
      } catch (error) {
        console.warn("Native HLS startup correction failed", error);
        step = { action: "settle", atTarget: false };
      }
    }
    if (step.action !== "settle") return;
    const { atTarget } = step;
    const epoch = mediaTimeToEpoch(video.currentTime);
    if (Number.isFinite(epoch)) {
      desiredEpochRef.current = epoch;
      setPlayhead(epoch);
    }
    nativePlaylistStartRef.current = null;
    pendingSeekEpochRef.current = null;
    pendingSeekModeRef.current = null;
    setHeroSeeking(false);
    setVisibleNativePlaylist(start.src);
    setPlaybackNotice(atTarget ? "" : "Playback started at a different recording time.");
    const wasPlaying = !video.paused;
    video.muted = heroMuted;
    // Safari can pause when muted autoplay becomes audible. Use the normal
    // play request so a blocked resume offers the existing play button.
    if (autoplayRef.current && video.paused && (wasPlaying || (atTarget && start.corrected))) {
      requestRecordingPlay(video);
    }
  }

  function handleRecordingTimeUpdate(event) {
    if (event.currentTarget !== videoRef.current || useSegmentPlayback || transport !== requestedTransport || originalFallbackRef.current === nativeScope || transcodeFallbackRef.current === nativeScope) return;
    if (Number.isFinite(pendingSeekEpochRef.current)) {
      const pendingMode = pendingSeekModeRef.current;
      if (pendingMode === "playlist-start") {
        checkNativePlaylistStart(event);
        return;
      }
      if (pendingMode === "local" || pendingMode === "window-ready") {
        const targetEpoch = pendingSeekEpochRef.current;
        const targetMediaTime = epochToPlaybackMediaTime(targetEpoch);
        if (videoReachedSeekTarget(event.currentTarget, targetMediaTime, recordingSeekToleranceSeconds())) {
          completePendingRecordingSeek(event.currentTarget);
        }
      }
      return;
    }
    extendGrowingPlaybackRows(event.currentTarget.currentTime);
    const epoch = mediaTimeToEpoch(event.currentTarget.currentTime);
    if (!Number.isFinite(epoch)) return;
    desiredEpochRef.current = epoch;
    setPlayhead(epoch);
    finishClipPreviewAtEnd(event.currentTarget, epoch);
  }

  // The player appends recordings to a growing playlist by itself. The page's
  // clock is the loaded rows, so re-read them as playback nears the last one.
  function extendGrowingPlaybackRows(mediaTime) {
    const detail = loadedPlaybackWindow;
    const lastRow = playbackTimeline[playbackTimeline.length - 1];
    if (!detail?.growing || !lastRow || mediaTime < lastRow.media_end - 3) return;
    const state = growingRowsRef.current;
    if (state.inFlight || performance.now() - state.at < 3000) return;
    state.inFlight = true;
    state.at = performance.now();
    fetch(recordingWindowUrl(activeCameraId, detail.start, detail.end, source))
      .then((response) => (response.ok ? response.json() : null))
      .then((payload) => {
        const rows = payload?.recordings || [];
        if (rows.length <= detail.rows.length) return;
        const extended = extendedPlaybackRows(detail.rows, rows);
        const stillLoaded = (current) => current?.scope === detail.scope && current.revision === detail.revision;
        if (extended) {
          setPlaybackDetail((current) => (
            stillLoaded(current) && extended.length > current.rows.length ? { ...current, rows: extended } : current
          ));
        } else if (stillLoaded(playbackDetailRef.current)) {
          // An earlier row changed, so the playing media times no longer
          // match any row list: load the window again from the playhead.
          requestPlaybackWindow({ start: detail.start, end: detail.end });
        }
      })
      .catch(() => { /* The next time update retries. */ })
      .finally(() => { state.inFlight = false; });
  }

  function finishClipPreviewAtEnd(video, epoch) {
    if (!clipPreviewReachedEnd(epoch, clipPreviewEndRef.current)) return false;
    cancelClipPreview({ pause: true, notice: "Clip preview complete", video });
    return true;
  }

  function cancelClipPreview({ pause = false, notice = "", video = videoRef.current } = {}) {
    const active = Number.isFinite(clipPreviewEndRef.current);
    clipPreviewEndRef.current = null;
    if (active) setClipPreviewing(false);
    if (pause) {
      autoplayRef.current = false;
      setHeroPlaying(false);
      video?.pause();
    }
    if (notice || active) setPlaybackNotice(notice);
  }

  function handleRecordingEnded(event) {
    const video = event?.currentTarget;
    if (video !== videoRef.current || Number.isFinite(pendingSeekEpochRef.current)) return;
    const epoch = useSegmentPlayback && nativeSegment
      ? nativeSegment.start_epoch + Math.max(0, Number(video?.currentTime) || 0)
      : mediaTimeToEpoch(Number(video?.currentTime));
    if (finishClipPreviewAtEnd(video, epoch)) return;
    continueRecordingPlayback();
  }

  function handleRecordingSeeked(event) {
    const video = event.currentTarget;
    if (video !== videoRef.current) return;
    const pendingMode = pendingSeekModeRef.current;
    if (pendingMode === "playlist-start") {
      checkNativePlaylistStart(event);
      return;
    }
    if (pendingMode === "local" || pendingMode === "window-ready") {
      const target = epochToPlaybackMediaTime(pendingSeekEpochRef.current);
      if (Number.isFinite(target) && !videoReachedSeekTarget(video, target, recordingSeekToleranceSeconds()) && !video.seeking) {
        seekVideoToTime(video, target, { allowFastSeek: !nativeHls });
        scheduleSeekWatchdog(video, target, { local: pendingMode === "local" });
        return;
      }
      completePendingRecordingSeek(video, { fromSeeked: true });
      return;
    }
    completePendingRecordingSeek(video);
  }

  function handleNativeSegmentSeeked(event) {
    const video = event.currentTarget;
    if (video !== videoRef.current) return;
    const pendingMode = pendingSeekModeRef.current;
    if (pendingMode === "native-local" || pendingMode === "native-ready") {
      const target = recordingSegmentLocalTime(nativeSegment, pendingSeekEpochRef.current, video);
      if (Number.isFinite(target) && !videoReachedSeekTarget(video, target, recordingSeekToleranceSeconds({ preferNativeHls: true })) && !video.seeking) {
        seekVideoToTime(video, target, { allowFastSeek: false });
        scheduleNativeSeekWatchdog(video, target, { local: pendingMode === "native-local" });
        return;
      }
      completePendingNativeSeek(video, { fromSeeked: true });
      return;
    }
    completePendingNativeSeek(video);
  }

  function toggleGridPlayback() {
    // Index reloads (Main/Sub, date) resume from autoplayRef, so it tracks the choice.
    autoplayRef.current = !gridPlaying;
    setGridPlaying(!gridPlaying);
  }

  function toggleHeroPlayback() {
    const video = videoRef.current;
    if (!video) return;
    const stepped = frameStep.epoch();
    if (Number.isFinite(stepped) && video.paused) {
      // Resume from the stepped frame; its picture stays up until playing.
      seekToSteppedFrame(stepped, true);
      return;
    }
    if (!video.paused || (heroSeeking && autoplayRef.current)) {
      if (typeof cancelClipPreview === "function") cancelClipPreview();
      autoplayRef.current = false;
      setHeroPlaying(false);
      video.pause();
      return;
    }
    autoplayRef.current = true;
    requestRecordingPlay(video);
  }

  function beginFrameSearch() {
    if (!semanticReady || !activeCameraId || !Number.isFinite(playhead)) return;
    frameSearchRequestRef.current?.abort();
    frameSearchRequestRef.current = null;
    setFrameSearchResults([]);
    setTrailEventIds([]);
    setTrailMeta(null);
    clearVisualSearchTrail(window.sessionStorage);
    setFrameSearchQueryImageUrl("");
    setFrameSearchQueryEpoch(null);
    autoplayRef.current = false;
    setHeroPlaying(false);
    videoRef.current?.pause();
    // The exact frame being stepped through, if any.
    setFrameSearchEpoch(frameStep.epoch() ?? playhead);
    setFrameSearchCrop(null);
    setFrameSearchImageBounds(null);
    setFrameSearchImageReady(false);
    setFrameSearchError("");
  }

  function closeFrameSearch() {
    frameSearchRequestRef.current?.abort();
    frameSearchRequestRef.current = null;
    frameSearchDragRef.current = null;
    setFrameSearchEpoch(null);
    setFrameSearchCrop(null);
    setFrameSearchImageBounds(null);
    setFrameSearchImageReady(false);
    setFrameSearchResults([]);
    setFrameSearchQueryImageUrl("");
    setFrameSearchQueryEpoch(null);
    setTrailEventIds([]);
    setTrailMeta(null);
    clearVisualSearchTrail(window.sessionStorage);
    setFrameSearchLoading(false);
    setFrameSearchError("");
  }

  function exitForensicMode() {
    closeFrameSearch();
    setTrailNotice("");
    setSelectedEventId(null);
    setInvestigationOpen(true);
  }

  function frameSearchPoint(event, bounds) {
    return {
      x: Math.max(0, Math.min(1, (event.clientX - bounds.pageLeft) / bounds.width)),
      y: Math.max(0, Math.min(1, (event.clientY - bounds.pageTop) / bounds.height)),
    };
  }

  function startFrameSearchCrop(event) {
    if (!frameSearchImageReady) return;
    const bounds = containedImageBounds(
      frameSearchSurfaceRef.current,
      frameSearchImageRef.current,
    );
    if (!bounds) return;
    const point = frameSearchPoint(event, bounds);
    frameSearchDragRef.current = point;
    setFrameSearchImageBounds(bounds);
    setFrameSearchCrop(null);
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function moveFrameSearchCrop(event) {
    if (!frameSearchDragRef.current) return;
    const bounds = containedImageBounds(
      frameSearchSurfaceRef.current,
      frameSearchImageRef.current,
    );
    if (!bounds) return;
    setFrameSearchImageBounds(bounds);
    setFrameSearchCrop(
      visualFrameCropFromPoints(
        frameSearchDragRef.current,
        frameSearchPoint(event, bounds),
      ),
    );
  }

  function finishFrameSearchCrop(event) {
    if (!frameSearchDragRef.current) return;
    moveFrameSearchCrop(event);
    frameSearchDragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  async function submitFrameSearch() {
    if (!frameSearchCrop || frameSearchLoading || !semanticReady) return;
    frameSearchRequestRef.current?.abort();
    const controller = new AbortController();
    frameSearchRequestRef.current = controller;
    const request = controller.signal;
    setFrameSearchLoading(true);
    setFrameSearchError("");
    try {
      const response = await fetch("/api/semantic-search/visual-frame", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: request,
        body: JSON.stringify(visualFrameSearchRequest({
          cameraId: activeCameraId,
          epoch: frameSearchEpoch,
          source,
          crop: frameSearchCrop,
          limit: 24,
        })),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.detail || `Frame search failed (${response.status})`);
      }
      const results = Array.isArray(payload.results) ? payload.results : [];
      const eventIds = results
        .map((result) => Number(result?.event?.id))
        .filter((eventId) => Number.isInteger(eventId) && eventId > 0);
      const nextTrail = writeVisualSearchTrail(window.sessionStorage, {
        eventIds,
        hits: results,
        queryMode: "visual",
      });
      if (frameSearchRequestRef.current !== controller || request.aborted) return;
      const canvas = frameSearchCropPreviewRef.current;
      if (canvas) {
        try {
          setFrameSearchQueryImageUrl(canvas.toDataURL("image/jpeg", 0.92));
        } catch {
          setFrameSearchQueryImageUrl("");
        }
      }
      setFrameSearchQueryEpoch(frameSearchEpoch);
      setFrameSearchResults(results);
      setTrailEventIds(nextTrail.eventIds);
      setTrailMeta(nextTrail);
      setInvestigationOpen(true);
      setSelectedEventId(null);
      setFrameSearchEpoch(null);
      setFrameSearchCrop(null);
      setFrameSearchImageBounds(null);
      setFrameSearchImageReady(false);
      if (!results.length) {
        setTrailNotice("No similar indexed incidents found.");
      } else {
        setTrailNotice("");
      }
    } catch (error) {
      if (error?.name !== "AbortError" && frameSearchRequestRef.current === controller) {
        setFrameSearchError(error.message || "Unable to search this frame.");
      }
    } finally {
      if (frameSearchRequestRef.current === controller) {
        frameSearchRequestRef.current = null;
        setFrameSearchLoading(false);
      }
    }
  }

  // Where playback is: the player's own time where it has one.
  function currentPlaybackEpoch() {
    const video = videoRef.current;
    let currentEpoch = Number.isFinite(playhead) ? playhead : desiredEpochRef.current;
    if (!isAllCameras && pendingSeekModeRef.current !== "playlist-start" && video && Number.isFinite(video.currentTime)) {
      const mediaEpoch = useSegmentPlayback
        ? (nativeSegment ? nativeSegment.start_epoch + Math.max(0, video.currentTime) : null)
        : mediaTimeToEpoch(video.currentTime);
      if (Number.isFinite(mediaEpoch)) currentEpoch = mediaEpoch;
    }
    return currentEpoch;
  }

  function skipPlayback(seconds, playing = autoplayRef.current || heroPlaying) {
    cancelClipPreview();
    // Ten seconds from a stepped frame, not from where the player was.
    const currentEpoch = frameStep.epoch() ?? currentPlaybackEpoch();
    if (!Number.isFinite(currentEpoch)) return;
    // The transport buttons should feel immediate. Timeline clicks remain
    // exact; a fast seek may land on the nearest decodable keyframe first.
    playAt(currentEpoch + seconds, playing, false, true);
  }

  async function handleNativeRecordingError(event) {
    const video = event.currentTarget;
    const requestedSource = video.getAttribute("src");
    const error = { code: video.error?.code, message: video.error?.message || "Recording playback failed" };
    // Native media also uses format errors for missing/unauthorized MP4 URLs.
    // Verify availability with a one-byte request before escalating to encoding.
    if (isRecordingCompatibilityError(error)) {
      const controller = new AbortController();
      const timer = window.setTimeout(() => controller.abort(), 10_000);
      try {
        const response = await fetch(requestedSource, { signal: controller.signal, headers: { Range: "bytes=0-0" } });
        await response.body?.cancel();
        if (!response.ok) {
          error.category = 1;
          error.message = `Recording file unavailable (${response.status})`;
        }
      } catch {
        error.category = 1;
        error.message = "Recording file request failed";
      } finally {
        window.clearTimeout(timer);
      }
    }
    if (video !== videoRef.current || video.getAttribute("src") !== requestedSource
      || playbackTransportRef.current?.scope !== nativeScope || playbackTransportRef.current.mode !== transport) return;
    handleRecordingError(error);
  }

  function handleRecordingError(error) {
    // Ignore duplicate errors from the outgoing player during handoff.
    if (!useSegmentPlayback && originalFallbackRef.current === nativeScope) return;
    if (!useTranscodedPlayback && transcodeFallbackRef.current === nativeScope) return;
    const detail = describePlaybackError(error);
    setHeroSeeking(true);
    if (source === "live" && availableSources.includes("main") && isRecordingCompatibilityError(error)) {
      failedSubScopeRef.current = `${activeCameraId}:${dayStart}:${dayEnd}`;
      setPlaybackNotice("Sub is not supported by this browser; trying Main.");
      setSource("main");
      return;
    }
    if (!useTranscodedPlayback && isRecordingCompatibilityError(error)) {
      const target = Number.isFinite(pendingSeekEpochRef.current)
        ? pendingSeekEpochRef.current : desiredEpochRef.current;
      clearSeekWatchdog();
      if (playbackRetryRef.current.timer) window.clearTimeout(playbackRetryRef.current.timer);
      playbackRetryRef.current = { attempts: 0, timer: null };
      if (useSegmentPlayback) {
        transcodeFallbackRef.current = nativeScope;
        setTranscodeScope(nativeScope);
      } else {
        originalFallbackRef.current = nativeScope;
        setOriginalScope(nativeScope);
      }
      pendingSeekEpochRef.current = target;
      const segment = recordingSegmentAt(playbackTimeline, target);
      pendingSeekModeRef.current = segment ? "native-ready" : "window";
      setNativeSegment(segment);
      setPlaybackError("");
      setPlaybackErrorStage("");
      setPlaybackNotice(useSegmentPlayback ? "Preparing compatible playback..." : "Loading original recording...");
      if (!segment) requestPlaybackWindow(windowAround(target));
      return;
    }
    console.warn("Recording playback error", { camera: activeCameraId, source, detail, error });
    if (useTranscodedPlayback && source === "main" && failedSubScopeRef.current !== `${activeCameraId}:${dayStart}:${dayEnd}` && availableSources.includes("live") && !codecFallbackRef.current && isRecordingCompatibilityError(error)) {
      codecFallbackRef.current = true;
      setPlaybackNotice(`Main stream is not supported by this browser; using Sub. (${detail})`);
      setSource("live");
      return;
    }
    if (playbackRetryRef.current.timer) return;
    if (playbackRetryRef.current.attempts < 4 && hasPlaybackMedia) {
      const retryTarget = Number.isFinite(pendingSeekEpochRef.current)
        ? pendingSeekEpochRef.current : desiredEpochRef.current;
      pendingSeekEpochRef.current = retryTarget;
      pendingSeekModeRef.current = useSegmentPlayback
        ? "native-ready" : "window-ready";
      // Source replacement pauses the existing element. That pause is an
      // internal retry transition, not a change to the user's play intent.
      ignorePauseUntilRef.current = performance.now() + 2000;
      playbackRetryRef.current.attempts += 1;
      const attempt = playbackRetryRef.current.attempts;
      const delay = Math.min(5_000, 750 * (2 ** (attempt - 1)));
      setPlaybackError("");
      setPlaybackErrorStage("");
      setPlaybackNotice(`Playback interrupted (${detail}). Retrying ${attempt}/4...`);
      playbackRetryRef.current.timer = window.setTimeout(() => {
        playbackRetryRef.current.timer = null;
        if (useSegmentPlayback) setNativeSegmentRetryToken((token) => token + 1);
        else setManifestRetryToken((token) => token + 1);
      }, delay);
      return;
    }
    setPlaybackNotice("");
    setPlaybackErrorStage("media");
    setPlaybackError(`Playback failed: ${detail}`);
  }

  function retryRecordingPlayback() {
    playbackWindowCacheRef.current.clear();
    if (playbackRetryRef.current.timer) window.clearTimeout(playbackRetryRef.current.timer);
    playbackRetryRef.current = { attempts: 0, timer: null };
    setPlaybackError("");
    setPlaybackErrorStage("");
    setPlaybackNotice("Retrying recording...");
    if (playbackErrorStage === "window" && playbackWindow) {
      requestPlaybackWindow(playbackWindow);
      return;
    }
    if (playbackErrorStage === "index") {
      if (!activeCameraId) {
        window.location.reload();
        return;
      }
      setRecordingIndexRevision((revision) => revision + 1);
      return;
    }
    if (manifestUrl) {
      setManifestRetryToken((token) => token + 1);
      return;
    }
    if (nativeSegmentUrl) {
      setNativeSegmentRetryToken((token) => token + 1);
      return;
    }
    const target = Number.isFinite(desiredEpochRef.current) ? desiredEpochRef.current : Date.now() / 1000;
    requestPlaybackWindow(windowAround(target));
  }

  function continueRecordingPlayback() {
    const endedSegment = useSegmentPlayback ? nativeSegment : playbackTimeline[playbackTimeline.length - 1];
    if (!endedSegment) return;
    const nextEpoch = endedSegment.end_epoch + 0.01;
    const nextRecordedEpoch = recordingEpochAfterSegment(endedSegment, timeline);
    if (nextRecordedEpoch !== null) {
      playAt(nextRecordedEpoch, true, false, false, true);
      return;
    }
    if (date === today) {
      desiredEpochRef.current = nextEpoch;
      autoplayRef.current = true;
      setFollowTarget(nextEpoch);
      setPlaybackNotice("Waiting for the next recording...");
    }
  }

  function changeDate(next) {
    checkpointTimelineView();
    setDate(next > today ? today : next);
  }

  function toggleExport() {
    if (exportJob && ["queued", "running", "cancelling"].includes(exportJob.status)) return;
    if (exportRange) {
      cancelClipPreview();
      setExportRange(null);
      setExportJob(null);
      setExportError("");
      return;
    }
    const center = Number.isFinite(playhead) ? playhead : timeline[0]?.start_epoch ?? dayStart;
    const start = Math.max(dayStart, center - 5 * 60);
    const end = Math.min(dayEnd, Math.max(start + 30, center + 5 * 60));
    setExportRange({ start, end });
    setExportJob(null);
    setExportError("");
    setExportLabel("");
  }

  const exportActive = Boolean(exportJob && ["queued", "running", "cancelling"].includes(exportJob.status));

  function setClipBoundary(kind) {
    if (!exportRange || exportJob) return;
    const next = setClipBoundaryAtPlayhead({
      range: exportRange,
      kind,
      playhead,
      startEpoch: timelineView.startEpoch,
      endEpoch: timelineView.endEpoch,
    });
    if (next !== exportRange) {
      cancelClipPreview();
      setExportRange(next);
    }
  }

  function jumpToClipBoundary(epoch) {
    cancelClipPreview();
    playAt(epoch, false);
  }

  function previewClip() {
    if (!exportRange || !clipRangeIsValid(exportRange, timelineView.startEpoch, timelineView.endEpoch)) return;
    clipPreviewEndRef.current = exportRange.end;
    setClipPreviewing(true);
    setPlaybackNotice("Previewing selected clip");
    playAt(exportRange.start, true);
  }

  async function startExport() {
    if (!activeCameraId || !exportRange || exportSubmitting) return;
    cancelClipPreview();
    setExportSubmitting(true);
    setExportError("");
    try {
      const response = await fetch(appUrl("/api/exports"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          kind: exportKind,
          camera_id: activeCameraId,
          source,
          start_epoch: exportRange.start,
          end_epoch: exportRange.end,
          sample_interval_seconds: exportOptions.interval,
          output_fps: exportOptions.fps,
          height: exportKind === "timelapse" ? exportOptions.timelapseHeight : exportOptions.clipHeight,
          label: exportLabel.trim(),
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `Export failed (${response.status})`);
      cacheExportJobs([payload]);
      setExportJob(payload);
    } catch (error) {
      setExportError(error.message || "Unable to start export");
    } finally {
      setExportSubmitting(false);
    }
  }

  async function cancelExport() {
    if (!exportJob?.id) return;
    try {
      const response = await fetch(appUrl(`/api/exports/${exportJob.id}`), { method: "DELETE" });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `Cancel failed (${response.status})`);
      if (!payload.deleted) cacheExportJobs([payload]);
      setExportJob(payload.deleted ? null : payload);
    } catch (error) {
      setExportError(error.message || "Unable to cancel export");
    }
  }

  function timelineDayControls() {
    return (
      <>
        <div className="recordings-v2-date">
          <button type="button" onClick={() => changeDate(addDaysToDateKey(date, -1))} aria-label="Previous day"><SkipBack size={15} /></button>
          <TimelineDatePicker value={date} max={today} onChange={changeDate} />
          <button type="button" onClick={() => changeDate(addDaysToDateKey(date, 1))} disabled={date >= today} aria-label="Next day"><SkipForward size={15} /></button>
          <button type="button" onClick={() => changeDate(today)} disabled={date === today}>Today</button>
        </div>
        <div className="recordings-v2-player-source" role="group" aria-label="Recording stream">
          <button type="button" className={source === "main" ? "active" : ""} aria-pressed={source === "main"} title="High" onClick={() => { checkpointTimelineView(); failedSubScopeRef.current = null; setSource("main"); }} disabled={availableSources.length > 0 && !availableSources.includes("main")}>Main</button>
          <button type="button" className={source === "live" ? "active" : ""} aria-pressed={source === "live"} title="Medium" onClick={() => { checkpointTimelineView(); failedSubScopeRef.current = null; setSource("live"); }} disabled={availableSources.length > 0 && !availableSources.includes("live")}>Sub</button>
        </div>
        <label className="recordings-playback-rate">
          <span className="sr-only">Playback speed</span>
          <select value={playbackRate} onChange={(event) => { checkpointTimelineView(); setPlaybackRate(normalizedTimelinePlaybackRate(event.target.value)); }}>
            {TIMELINE_PLAYBACK_RATES.map((rate) => <option key={rate} value={rate}>{rate}x</option>)}
          </select>
        </label>
      </>
    );
  }

  // Player HUD: previous/next day on either side of the date/time picker.
  function hudDayControls() {
    return (
      <div className="timeline-hud-day" role="group" aria-label="Day">
        <button type="button" onClick={() => changeDate(addDaysToDateKey(date, -1))} aria-label="Previous day" title="Previous day"><ChevronLeft size={16} /></button>
        <TimelineDatePicker
          value={date}
          max={today}
          onChange={changeDate}
          label={Number.isFinite(playhead) ? (
            <FrameStepTime stepper={frameStep} timeZone={timeZone} prefix={`${formatTimelineDateLabel(date)} `}>
              {`${formatTimelineDateLabel(date)} ${formatTimeOnly(playhead, timeZone)}`}
            </FrameStepTime>
          ) : null}
          title={Number.isFinite(playhead) ? `${formatDateTime(playhead, timeZone)} · choose a day` : "Choose a day"}
        />
        <button type="button" onClick={() => changeDate(addDaysToDateKey(date, 1))} disabled={date >= today} aria-label="Next day" title="Next day"><ChevronRight size={16} /></button>
      </div>
    );
  }

  // Today and playback speed join the transport on the HUD's left.
  function hudPlayExtras() {
    return (
      <>
        <button type="button" className="timeline-hud-today" onClick={() => changeDate(today)} disabled={date === today}>Today</button>
        <label className="recordings-playback-rate timeline-hud-rate">
          <span className="sr-only">Playback speed</span>
          <select value={playbackRate} onChange={(event) => { checkpointTimelineView(); setPlaybackRate(normalizedTimelinePlaybackRate(event.target.value)); }}>
            {TIMELINE_PLAYBACK_RATES.map((rate) => <option key={rate} value={rate}>{rate}x</option>)}
          </select>
        </label>
      </>
    );
  }

  const exportPanel = exportRange ? (
            <section className="recordings-v2-export-panel" aria-label="Export recording">
              <div className="recordings-v2-export-kind" role="group" aria-label="Export type">
                <button type="button" className={exportKind === "recording" ? "active" : ""} onClick={() => setExportKind("recording")} disabled={Boolean(exportJob)}>Video clip</button>
                <button type="button" className={exportKind === "timelapse" ? "active" : ""} onClick={() => setExportKind("timelapse")} disabled={Boolean(exportJob)}>Timelapse</button>
              </div>
              <div className="recordings-v2-export-range-label">
                <span><b>Start</b>{formatDateTime(exportRange.start, timeZone)}</span>
                <span><b>End</b>{formatDateTime(exportRange.end, timeZone)}</span>
                <span><b>Length</b>{formatDuration(exportRange.end - exportRange.start)}</span>
              </div>
              <div className="recordings-v2-clip-composer" aria-label="Clip selection controls">
                <div className="recordings-v2-clip-composer-heading">
                  <span><Camera size={14} />{cameras.find((camera) => camera.id === activeCameraId)?.name || activeCameraId}</span>
                  <small>Choose within {formatTimeOnly(timelineView.startEpoch, timeZone)}–{formatTimeOnly(timelineView.endEpoch, timeZone)}. Minimum 1 sec.</small>
                </div>
                <div className="recordings-v2-clip-boundaries">
                  <button type="button" onClick={() => jumpToClipBoundary(exportRange.start)} disabled={Boolean(exportJob)} title="Jump to selected start"><SkipBack size={14} />Start</button>
                  <button type="button" onClick={() => setClipBoundary("start")} disabled={Boolean(exportJob) || !canSetClipBoundaryAtPlayhead({ range: exportRange, kind: "start", playhead, startEpoch: timelineView.startEpoch, endEpoch: timelineView.endEpoch })}>Set start here</button>
                  <button type="button" onClick={() => setClipBoundary("end")} disabled={Boolean(exportJob) || !canSetClipBoundaryAtPlayhead({ range: exportRange, kind: "end", playhead, startEpoch: timelineView.startEpoch, endEpoch: timelineView.endEpoch })}>Set end here</button>
                  <button type="button" onClick={() => jumpToClipBoundary(exportRange.end)} disabled={Boolean(exportJob)} title="Jump to selected end">End<SkipForward size={14} /></button>
                  {clipPreviewing ? <button type="button" className="active" onClick={() => cancelClipPreview({ pause: true, notice: "Clip preview stopped" })}><Pause size={14} />Stop preview</button> : <button type="button" onClick={previewClip} disabled={Boolean(exportJob) || !clipRangeIsValid(exportRange, timelineView.startEpoch, timelineView.endEpoch)}><Play size={14} fill="currentColor" />Preview clip</button>}
                </div>
              </div>
              <div className="recordings-v2-export-options">
                <label className="export-name-field"><span>Name</span><input value={exportLabel} onChange={(event) => setExportLabel(event.target.value)} placeholder="Optional export name" maxLength="120" disabled={Boolean(exportJob)} /></label>
                {exportKind === "timelapse" ? <>
                  <label><span>Capture every</span><select value={exportOptions.interval} onChange={(event) => setExportOptions((current) => ({ ...current, interval: Number(event.target.value) }))} disabled={Boolean(exportJob)}><option value="5">5 sec</option><option value="10">10 sec</option><option value="30">30 sec</option><option value="60">1 min</option><option value="300">5 min</option></select></label>
                  <label><span>Playback</span><select value={exportOptions.fps} onChange={(event) => setExportOptions((current) => ({ ...current, fps: Number(event.target.value) }))} disabled={Boolean(exportJob)}><option value="24">24 FPS</option><option value="30">30 FPS</option><option value="60">60 FPS</option></select></label>
                </> : null}
                <label><span>Resolution</span><select value={exportKind === "timelapse" ? exportOptions.timelapseHeight : exportOptions.clipHeight} onChange={(event) => setExportOptions((current) => ({ ...current, [exportKind === "timelapse" ? "timelapseHeight" : "clipHeight"]: Number(event.target.value) }))} disabled={Boolean(exportJob)}><option value="0">Original</option><option value="2160">2160p</option><option value="1440">1440p</option><option value="1080">1080p</option><option value="720">720p</option><option value="480">480p</option></select></label>
              </div>
              {exportKind === "recording" ? <p>Creates one broadly compatible H.264 MP4. Width follows the camera aspect ratio; gaps are skipped and source-format changes are joined automatically.</p> : <p>Resolution is the output height; width follows the camera aspect ratio.</p>}
              {exportJob ? (
                <div className={`recordings-v2-export-status ${exportJob.status}`}>
                  <span><b>{exportJob.phase || exportJob.status}</b><small>{Math.round(Number(exportJob.progress) || 0)}%</small></span>
                  <i><b style={{ width: `${Math.max(0, Math.min(100, Number(exportJob.progress) || 0))}%` }} /></i>
                  {exportJob.error ? <em>{exportJob.error}</em> : null}
                </div>
              ) : null}
              {exportError ? <div className="recordings-v2-export-error"><CircleAlert size={15} />{exportError}</div> : null}
              <div className="recordings-v2-export-actions">
                {exportJob?.status === "completed" && exportJob.download_url ? <a className="nav-button" href={mediaUrl(exportJob.download_url)}><Download size={15} />Download</a> : null}
                {exportJob?.status === "completed" ? <a className="nav-button" href={appUrl(`/exports?camera=${encodeURIComponent(activeCameraId)}`)}><Film size={15} />Open export</a> : null}
                {exportJob && ["queued", "running", "cancelling"].includes(exportJob.status) ? <button type="button" onClick={cancelExport} disabled={exportJob.status === "cancelling"}><X size={15} />{exportJob.status === "cancelling" ? "Cancelling" : "Cancel"}</button> : null}
                {!exportJob ? <button type="button" className="primary" onClick={startExport} disabled={exportSubmitting}><Download size={15} />{exportSubmitting ? "Starting..." : `Start ${exportKind === "timelapse" ? "timelapse" : "export"}`}</button> : null}
                {exportJob && ["completed", "failed", "cancelled"].includes(exportJob.status) ? <button type="button" onClick={() => { setExportJob(null); setExportError(""); setExportLabel(""); }}>New export</button> : null}
              </div>
            </section>
          ) : null;

  const cameraNames = Object.fromEntries(cameras.map((camera) => [camera.id, camera.name || camera.id]));
  function changeGridLayoutMode(nextMode) {
    const current = gridSelection.layoutMode;
    if (nextMode === "focus" && current !== "focus") setGridFocusScope(current === "custom" ? "custom" : "all");
    setGridLayoutMode(nextMode);
    setGridPage(0);
  }

  function changeGridCustomCameras(value) {
    setGridCustomCameras(value);
    setGridPage(0);
  }

  // Live's makePrimaryCamera: the camera becomes Focus's primary.
  function focusGridCamera(nextCameraId) {
    if (!nextCameraId) return;
    setGridFocusCamera(String(nextCameraId));
    if (gridLayoutsAvailable) changeGridLayoutMode("focus");
  }

  function changeGridDensity(nextDensity) {
    setGridDensity(nextDensity);
    setGridPage(0);
  }

  function selectRailEvent(event) {
    if (!event) return;
    checkpointTimelineView();
    cancelClipPreview();
    setSelectedEventId(event.id);
    // All cameras keeps the grid and focuses the event's camera instead.
    if (isAllCameras) focusGridCamera(event.camera_id);
    else if (event.camera_id && event.camera_id !== activeCameraId) selectPlaybackCamera(event.camera_id);
    playAt(event.incident_epoch, true);
  }
  function changeTimelineLayout(nextLayout) {
    if (nextLayout === timelineLayout) return;
    checkpointTimelineView();
    setTimelineLayout(nextLayout);
  }
  const hudEventActions = (
    <div className="timeline-hud-actions" role="group" aria-label="Events">
      <button type="button" onClick={() => selectRailEvent(previousRailEvent)} disabled={!previousRailEvent} aria-label="Previous event" title="Previous event"><SkipBack size={15} /><span className="timeline-hud-label">Previous</span></button>
      <button type="button" onClick={() => selectRailEvent(nextRailEvent)} disabled={!nextRailEvent} aria-label="Next event" title="Next event"><span className="timeline-hud-label">Next</span><SkipForward size={15} /></button>
      {!isAllCameras ? <button
        type="button"
        aria-pressed={investigationOpen}
        aria-expanded={investigationOpen}
        aria-controls="timeline-investigation"
        onClick={() => setInvestigationOpen((current) => !current)}
      >
        {investigationOpen ? <ChevronDown size={15} /> : <ChevronUp size={15} />}Nearby
      </button> : null}
      <button type="button" className={exportRange ? "active" : ""} aria-pressed={Boolean(exportRange)} onClick={toggleExport} disabled={isAllCameras || !timeline.length || exportActive}
        title={isAllCameras ? "Select one camera to export" : undefined}>
        <Download size={15} />{isAllCameras ? "Select camera" : exportActive ? "Export running" : exportRange ? "Close export" : "Export"}
      </button>
    </div>
  );
  const railControls = (
    <div className="rail-controls">
      {exportPanel}
      {timelineLayout === "cameras" ? (
        <CameraLanes
          cameras={cameras}
          recordings={laneData.scope === laneScope ? laneData.recordings : []}
          events={laneEvents}
          source={source}
          startEpoch={timelineView.startEpoch}
          endEpoch={timelineView.endEpoch}
          topEpoch={railTopEpoch}
          dayStart={dayStart}
          playhead={railPlayhead}
          onSeek={(epoch) => { cancelClipPreview(); playAt(epoch, true); }}
          onScrub={isAllCameras ? null : timelineScrub}
          onSelectCamera={(nextCameraId) => {
            if (isAllCameras) focusGridCamera(nextCameraId);
            else if (nextCameraId !== activeCameraId) selectPlaybackCamera(nextCameraId);
          }}
          timeZone={timeZone}
          spanMinutes={railMinutes}
          onSpanChange={(minutes) => { checkpointTimelineView(); setRailMinutes(minutes); }}
          windowHours={incidentRangeHours}
          onWindowHoursChange={(hours) => { checkpointTimelineView(); setIncidentRangeHours(hours); }}
        />
      ) : null}
    </div>
  );
  const railAside = timelineLayout === "rail" ? (
    <EventRail
      cameraId={isAllCameras ? (gridSelection.layoutMode === "focus" ? gridSelection.focusCameraId : activeCameraId) : activeCameraId}
      cameraNames={isAllCameras ? cameraNames : null}
      source={source}
      dayStart={dayStart}
      dayEnd={dayEnd}
      topEpoch={railTopEpoch}
      playhead={railPlayhead}
      recordings={railRecordings}
      events={filteredEvents}
      densityEvents={timelineEvents}
      spanMinutes={railMinutes}
      onSpanChange={(minutes) => { checkpointTimelineView(); setRailMinutes(minutes); }}
      filter={eventFilter}
      onFilterChange={(value) => { checkpointTimelineView(); setEventFilter(value); }}
      selectedEventId={selectedEvent?.id}
      onEventSelect={selectRailEvent}
      onSeek={(epoch) => { cancelClipPreview(); playAt(epoch, true); }}
      onScrub={isAllCameras ? null : timelineScrub}
      timeZone={timeZone}
      onPreviousDay={() => changeDate(addDaysToDateKey(date, -1))}
      onNextDay={date < today ? () => changeDate(addDaysToDateKey(date, 1)) : null}
    />
  ) : timelineLayout === "catchup" ? (
    <CatchUpStory
      events={filteredEvents}
      dayStart={dayStart}
      endEpoch={railTopEpoch}
      playhead={railPlayhead}
      selectedEventId={storyFocus?.id}
      onPlayEvent={selectRailEvent}
      onShowRealTime={(epoch) => { changeTimelineLayout("rail"); playAt(epoch, true); }}
      cameraNames={cameraNames}
      timeZone={timeZone}
      quietMinutes={quietMinutes}
      onQuietMinutesChange={setQuietMinutes}
      filter={eventFilter}
      onFilterChange={(value) => { checkpointTimelineView(); setEventFilter(value); }}
    />
  ) : null;

  return (
    <main className={`recordings-v2-page timeline-layout-${timelineLayout}${investigationShown ? " has-investigation" : " investigation-hidden"}`}>
      <nav className="recordings-tabs recordings-commandbar" aria-label="Timeline controls">
        <TimelineCameraPicker
          cameras={cameras}
          value={isAllCameras ? ALL_RECORDING_CAMERAS_ID : activeCameraId}
          allOption={{ value: ALL_RECORDING_CAMERAS_ID, label: "All cameras" }}
          onChange={(nextCameraId) => { selectPlaybackCamera(nextCameraId); }}
        />
        <MobileCameraSelect
          className="timeline-mobile-camera-select"
          cameras={cameras}
          value={activeCameraId}
          onChange={(nextCameraId) => { selectPlaybackCamera(nextCameraId); }}
          ariaLabel="Timeline camera"
        />
        <div className="timeline-view-switch" role="group" aria-label="Timeline view">
          {[["rail", "Timeline", Rows3], ["catchup", "Catch up", FastForward], ["cameras", "Cameras", Columns3]].map(([value, label, Icon]) => (
            <button key={value} type="button" aria-pressed={timelineLayout === value} aria-label={label} title={label} onClick={() => changeTimelineLayout(value)}>
              <Icon size={15} aria-hidden="true" /><span className="timeline-view-label">{label}</span>
            </button>
          ))}
        </div>
        {isAllCameras ? (
          <div className="timeline-grid-layout">
            {/* Density only applies to Automatic (Focus and Custom show their whole set). */}
            {gridSelection.layoutMode === "auto" ? <CameraDensityControl density={gridSelection.density} page={gridSelection.page} pageCount={gridSelection.pageCount} onDensityChange={changeGridDensity} onPageChange={setGridPage} /> : null}
            <CameraLayoutModeControl mode={gridSelection.layoutMode} customAvailable={gridLayoutsAvailable} onModeChange={changeGridLayoutMode} ariaLabel="All cameras layout" customTitle="Choose which cameras to show">
              {gridSelection.layoutMode === "custom" ? <button type="button" className="secondary live-layout-reset" onClick={() => changeGridCustomCameras("")} title="Show every camera"><RotateCcw size={14} /> Reset</button> : null}
            </CameraLayoutModeControl>
            {gridSelection.layoutMode === "custom" ? <TimelineCameraChecklist cameras={gridCameras} selectedIds={gridCustomIds} onChange={(ids) => changeGridCustomCameras(JSON.stringify(ids))} /> : null}
          </div>
        ) : null}
        <span className="recordings-commandbar-live"><i />{date === today ? "Live archive" : "Archive"}</span>
        <div className="recordings-commandbar-day-controls">{timelineDayControls()}</div>
      </nav>
      <section className="recordings-v2-workspace">
        <div className={`recordings-v2-player${isAllCameras ? " all-cameras-player" : " selected-camera-stage"}`}>
          {isAllCameras ? <div className="recording-hero-media all-cameras-media">
            {Number.isFinite(playhead) ? <AllCamerasGrid
              cameras={gridSelection.shown}
              source={source}
              epoch={playhead}
              playing={gridPlaying}
              rate={playbackRate}
              layoutMode={gridSelection.layoutMode}
              focusCameraId={gridSelection.focusCameraId}
              canFocus={gridLayoutsAvailable}
              onFocusCamera={focusGridCamera}
            /> : null}
          </div> : null}
          {!isAllCameras ? <div className="recording-hero-media" ref={heroMediaRef}>
          {showPlaybackDiagnostics ? <PlaybackDiagnostics containerRef={heroMediaRef} transport={transport} override={transportOverride} playlistStart={nativeHls && !useSegmentPlayback && manifestStartTime > 0} /> : null}
          {nativeSegmentUrl ? (
            <NativeRecordingVideo
              key={nativeScope}
              ref={videoRef}
              src={nativeSegmentUrl}
              nextSrc={nativeNextUrl}
              muted={heroMuted}
              autoPlay={autoplayRef.current}
              playbackRate={normalizedTimelinePlaybackRate(playbackRate)}
              onLoadedMetadata={handleNativeSegmentMetadata}
              onError={handleNativeRecordingError}
              onTimeUpdate={handleNativeSegmentTimeUpdate}
              onSeeked={handleNativeSegmentSeeked}
              onEnded={handleRecordingEnded}
              onPlay={() => {
                autoplayRef.current = true;
                setHeroPlaying(true);
                setPlaybackBlocked(false);
                if (!Number.isFinite(pendingSeekEpochRef.current)) setPlaybackNotice("");
              }}
              onPause={(event) => {
                if (performance.now() < ignorePauseUntilRef.current) return;
                if (!event.currentTarget.error && !event.currentTarget.ended
                  && !Number.isFinite(pendingSeekEpochRef.current)) {
                  cancelClipPreview();
                  autoplayRef.current = false;
                  setHeroPlaying(false);
                }
              }}
            />
          ) : null}
          {!useSegmentPlayback && !isAllCameras && activeCameraId ? (
            <RecordingHlsVideo
              ref={videoRef}
              src={manifestUrl}
              mimeType="application/vnd.apple.mpegurl"
              startTime={nativeHls ? null : manifestStartTime}
              bufferingGoal={8}
              controls={false}
              muted={playbackMuted}
              playsInline
              preload="auto"
              onReady={handleRecordingReady}
              onError={handleRecordingError}
              onLoadedData={checkNativePlaylistStart}
              onCanPlay={checkNativePlaylistStart}
              onPlaying={checkNativePlaylistStart}
              onTimeUpdate={handleRecordingTimeUpdate}
              onSeeking={checkNativePlaylistStart}
              onSeeked={handleRecordingSeeked}
              onEnded={handleRecordingEnded}
              onPlay={() => {
                autoplayRef.current = true;
                setHeroPlaying(true);
                setPlaybackBlocked(false);
                if (!Number.isFinite(pendingSeekEpochRef.current)) setPlaybackNotice("");
              }}
              onPause={(event) => {
                if (event.currentTarget !== videoRef.current || !event.currentTarget.paused) return;
                if (performance.now() < ignorePauseUntilRef.current) return;
                if (!event.currentTarget.error && !event.currentTarget.ended
                  && !Number.isFinite(pendingSeekEpochRef.current)) {
                  cancelClipPreview();
                  autoplayRef.current = false;
                  setHeroPlaying(false);
                }
              }}
            />
          ) : null}
          {coveringNativeStartup ? <div className="recording-hero-startup-cover" aria-hidden="true" /> : null}
          <FrameStepOverlay stepper={frameStep} />
          <ScrubFrameOverlay ref={scrubOverlayRef} seeking={heroSeeking} />
          <DebugDetectionOverlay
            key={aiOverlayKey}
            videoRef={videoRef}
            active={aiOverlayActive}
            confidence={0.35}
          />
          {Number.isFinite(frameSearchEpoch) ? (
            <div className="recording-frame-search-overlay">
              <div
                ref={frameSearchSurfaceRef}
                className={`recording-frame-search-surface${frameSearchImageReady ? " ready" : ""}`}
                onPointerDown={startFrameSearchCrop}
                onPointerMove={moveFrameSearchCrop}
                onPointerUp={finishFrameSearchCrop}
                onPointerCancel={finishFrameSearchCrop}
              >
                <img
                  ref={frameSearchImageRef}
                  src={frameSearchPreviewUrl}
                  alt={`Frozen recording frame at ${formatDateTime(frameSearchEpoch, timeZone)}`}
                  draggable="false"
                  onLoad={() => {
                    setFrameSearchImageReady(true);
                    window.requestAnimationFrame(() => setFrameSearchImageBounds(
                      containedImageBounds(frameSearchSurfaceRef.current, frameSearchImageRef.current)
                    ));
                  }}
                  onError={() => setFrameSearchError("Could not load the exact recording frame.")}
                />
                {frameSearchCrop && frameSearchImageBounds ? (
                  <i
                    className="recording-frame-search-selection"
                    style={{
                      left: frameSearchImageBounds.left + frameSearchCrop.x * frameSearchImageBounds.width,
                      top: frameSearchImageBounds.top + frameSearchCrop.y * frameSearchImageBounds.height,
                      width: frameSearchCrop.width * frameSearchImageBounds.width,
                      height: frameSearchCrop.height * frameSearchImageBounds.height,
                    }}
                  />
                ) : null}
                {!frameSearchImageReady && !frameSearchError ? <span>Loading exact frame…</span> : null}
              </div>
              <div className="recording-frame-search-panel">
                <header><div><Search size={15} /><strong>Select area</strong></div><button type="button" onClick={closeFrameSearch} aria-label="Close frame search"><X size={15} /></button></header>
                <small>Drag around the person, vehicle, or object to match.</small>
                {frameSearchCrop ? <canvas ref={frameSearchCropPreviewRef} width="176" height="121" aria-label="Selected crop preview" /> : null}
                {frameSearchError ? <p>{frameSearchError}</p> : null}
                <footer>
                  <span>{semanticStatusLabel}</span>
                  <button type="button" className="primary" disabled={!frameSearchCrop || frameSearchLoading || !semanticReady} onClick={submitFrameSearch}>{frameSearchLoading ? "Searching…" : "Find similar"}</button>
                </footer>
              </div>
            </div>
          ) : null}
          </div> : null}
          {!isAllCameras && Number.isFinite(playhead) ? <RecordingCompanionStrip cameras={cameras} routes={cameraTransitionRoutes} activeCameraId={activeCameraId} source={source} epoch={playhead} onSelect={(camera) => { selectPlaybackCamera(camera); }} /> : null}
          {loading ? <div className="recordings-v2-message"><Film size={28} />Loading recordings</div> : null}
          {!loading && !timeline.length ? <div className="recordings-v2-message"><Film size={28} />No recordings on this day</div> : null}
          {playbackError ? <div className="recordings-v2-error"><span>{playbackError}</span><button type="button" onClick={retryRecordingPlayback}><RefreshCcw size={14} />Retry</button></div> : null}
          {playbackNotice && !playbackError ? <div className="recordings-v2-notice">{playbackNotice}</div> : null}
          {!isAllCameras && playbackBlocked && !playbackError ? (
            <button type="button" className="recordings-v2-play" onClick={() => requestRecordingPlay(videoRef.current)}>
              <Play size={22} fill="currentColor" />
              Play recording
            </button>
          ) : null}
          {!isAllCameras && !Number.isFinite(frameSearchEpoch) ? (
            <div className="recording-grid-controls recording-hero-controls">
              {Number.isFinite(playhead) && hasPlaybackMedia && !playbackError ? <>
              <button type="button" onClick={() => skipPlayback(-10)} aria-label="Back 10 seconds"><SkipBack size={16} /></button>
              <button type="button" className="primary" onClick={toggleHeroPlayback}>
                {heroPlaying ? <Pause size={17} /> : <Play size={17} fill="currentColor" />}
                {heroPlaying ? "Pause" : "Play"}
              </button>
              <button type="button" onClick={() => skipPlayback(10)} aria-label="Forward 10 seconds"><SkipForward size={16} /></button>
              <FrameStepButtons stepper={frameStep} disabled={heroPlaying} />
              <button
                type="button"
                onClick={() => setHeroMuted((current) => !current)}
                aria-label={heroMuted ? "Unmute recording" : "Mute recording"}
                aria-pressed={heroMuted}
              >
                {heroMuted ? <VolumeX size={16} /> : <Volume2 size={16} />}
              </button>
              <button
                type="button"
                className={aiOverlayEnabled ? "active" : ""}
                aria-pressed={aiOverlayEnabled}
                onClick={() => setAiOverlayEnabled((current) => !current)}
                title="Overlay live AI detections while this recording plays"
              >
                <Sparkles size={15} />AI
              </button>
              </> : null}
              {hudPlayExtras()}
              <>{hudDayControls()}<span className="timeline-hud-break" aria-hidden="true" /></>
              {Number.isFinite(playhead) && hasPlaybackMedia && !playbackError ? <>
              <button
                type="button"
                className="timeline-hud-find"
                disabled={!semanticReady}
                onClick={beginFrameSearch}
                title={semanticReady ? "Freeze this frame and select an area" : "Smart Search is not ready"}
              >
                <Search size={15} />Find similar
              </button>
              <small className="recording-frame-search-status">{semanticStatusLabel}</small>
              </> : null}
              {hudEventActions}
              <div className="incident-recording-source" role="group" aria-label="Playback stream">
                {["main", "live"].map(value => <button key={value} type="button" aria-pressed={source === value}
                  disabled={availableSources.length > 0 && !availableSources.includes(value)}
                  onClick={() => { checkpointTimelineView(); failedSubScopeRef.current = null; setSource(value); }}>
                  {value === "main" ? "Main" : "Sub"}
                </button>)}
              </div>
            </div>
          ) : null}
          {isAllCameras ? <div className="recording-grid-controls recording-hero-controls all-cameras-controls">
            {Number.isFinite(playhead) ? <>
            <button type="button" onClick={() => skipPlayback(-10, gridPlaying)} aria-label="Back 10 seconds"><SkipBack size={16} /></button>
            <button type="button" className="primary" onClick={toggleGridPlayback}>{gridPlaying ? <Pause size={17} /> : <Play size={17} fill="currentColor" />}{gridPlaying ? "Pause" : "Play"}</button>
            <button type="button" onClick={() => skipPlayback(10, gridPlaying)} aria-label="Forward 10 seconds"><SkipForward size={16} /></button>
            </> : null}
            {hudPlayExtras()}
            <>{hudDayControls()}<span className="timeline-hud-break" aria-hidden="true" /></>
            {Number.isFinite(playhead) ? <span
              className="all-cameras-summary"
              role="status"
              title={unrecordedGridCameras.length ? `No recording in this 15-minute window: ${unrecordedGridCameras.map((camera) => camera.name || camera.id).join(", ")}` : undefined}
            >
              {recordedGridCameras.length} of {gridSelection.chosen.length} cameras
              {unrecordedGridCameras.length ? (
                <em> · No recording: {unrecordedGridCameras.map((camera) => camera.name || camera.id).join(", ")}</em>
              ) : null}
            </span> : null}
            {hudEventActions}
            <div className="incident-recording-source" role="group" aria-label="Playback stream">
              {["main", "live"].map(value => <button key={value} type="button" aria-pressed={source === value}
                disabled={availableSources.length > 0 && !availableSources.includes(value)}
                onClick={() => { checkpointTimelineView(); setSource(value); }}>
                {value === "main" ? "Main" : "Sub"}
              </button>)}
            </div>
          </div> : null}
        </div>

        {railControls}
        {railAside}
      </section>

      <div id="timeline-investigation" className="recordings-v2-incidents" hidden={!investigationShown}>
        <div className={`recordings-v2-investigation${showForensicPanel ? " forensic-active" : " nearby-only"}`}>
          {showForensicPanel && forensicContext ? (
            <aside className="recordings-v2-forensic-context" aria-label="Forensic context">
              <header>
                <div className="recordings-v2-forensic-title">
                  <span>Forensic</span>
                  <button
                    type="button"
                    className="recordings-v2-forensic-exit"
                    onClick={exitForensicMode}
                    aria-label="Exit forensic mode"
                  >
                    Exit
                  </button>
                </div>
                <div className="recordings-v2-trail-nav" role="group" aria-label="Find similar context">
                  <small>{forensicNav.label}</small>
                  <button
                    type="button"
                    disabled={!(forensicNav.previous || forensicNav.deselectOnPrevious)}
                    onClick={() => stepForensicNav(-1)}
                    aria-label="Previous Find similar context"
                  ><ChevronLeft size={15} /></button>
                  <button
                    type="button"
                    disabled={!forensicNav.next}
                    onClick={() => stepForensicNav(1)}
                    aria-label="Next Find similar context"
                  ><ChevronRight size={15} /></button>
                </div>
              </header>
              {trailNotice ? <p className="recordings-v2-trail-notice">{trailNotice}</p> : null}
              <div className={`recordings-v2-forensic-image ${forensicContext.kind}`}>
                <img src={forensicContext.imageUrl} alt="" loading="lazy" decoding="async" />
                <span className="recordings-v2-forensic-hud">
                  <time>{forensicContext.timeLabel}</time>
                </span>
              </div>
            </aside>
          ) : null}
          {frameSearchResults.length ? (
            <section className="recordings-related-events recording-frame-search-results" aria-label="Find similar results">
              <header><strong>Find similar</strong><span>{frameSearchResults.length} matches</span></header>
              <div className="recordings-v2-events recording-frame-search-results-grid">
                {frameSearchResults.map((result) => {
                  const item = result.event || {};
                  const cameraName = cameras.find((camera) => camera.id === item.camera_id)?.name || item.camera_id;
                  const epoch = Number.isFinite(Number(item.incident_epoch))
                    ? Number(item.incident_epoch)
                    : (item.created_at ? new Date(item.created_at).getTime() / 1000 : null);
                  const observedAt = Number.isFinite(epoch) ? formatDateTime(epoch, timeZone) : "";
                  const objectLabel = result.evidence?.object_label || item.labels?.[0] || "Match";
                  const selected = Number(item.id) === Number(selectedEventId);
                  return (
                    <button
                      key={item.id}
                      type="button"
                      className={`${item.has_objects !== false ? "object" : "motion"}${selected ? " selected" : ""}`}
                      onClick={() => openFindSimilarResult(result)}
                      aria-pressed={selected}
                      aria-label={`${objectLabel} at ${cameraName}, ${observedAt}`}
                      title={`${observedAt} · ${objectLabel}`}
                    >
                      <span className="recordings-v2-event-image">
                        <Radar size={20} />
                        <img src={mediaUrl(semanticResultThumbnailUrl(result, 240, 72))} alt="" loading="lazy" decoding="async" />
                      </span>
                      <span className="recordings-v2-event-caption">
                        <time>{observedAt}</time>
                        <b>{objectLabel}</b>
                      </span>
                    </button>
                  );
                })}
              </div>
            </section>
          ) : (
            <section className="recordings-related-events recordings-related-events-full" aria-label="Nearby evidence">
              <header><strong>Nearby evidence</strong><span>{nearbyEvents.length} nearby</span></header>
              <div className="recordings-v2-events">
                {nearbyEvents.length ? nearbyEvents.map((event) => (
                  <button
                    key={event.id}
                    type="button"
                    className={`${event.has_objects ? "object" : "motion"}${Number(selectedEvent?.id) === Number(event.id) ? " selected" : ""}`}
                    onClick={() => { setInvestigationOpen(true); setSelectedEventId(event.id); playAt(event.incident_epoch, true); }}
                    aria-pressed={Number(selectedEvent?.id) === Number(event.id)}
                    aria-label={`${event.labels?.length ? event.labels.join(", ") : "Motion only"} at ${formatDateTime(event.incident_epoch, timeZone)}`}
                    title={`${formatDateTime(event.incident_epoch, timeZone)} · ${event.labels?.length ? event.labels.join(", ") : "Motion only"}`}
                  >
                    <span className="recordings-v2-event-image">
                      <Radar size={20} />
                      {event.snapshot_path ? <img src={eventThumbnailUrl(event, 240, 72, { objectFocus: true, aspectWidth: 16, aspectHeight: 11 })} alt="" loading="lazy" decoding="async" onError={(loadEvent) => { loadEvent.currentTarget.hidden = true; }} /> : null}
                    </span>
                    <span className="recordings-v2-event-caption">
                      <time>{formatDateTime(event.incident_epoch, timeZone)}</time>
                      <b>{isAllCameras ? `${cameras.find((camera) => camera.id === event.camera_id)?.name || event.camera_id} · ` : ""}{event.labels?.length ? event.labels.join(", ") : "Motion only"}</b>
                    </span>
                  </button>
                )) : <div className="recordings-v2-no-events"><Radar size={17} />No {eventFilter === "all" ? "events" : `${eventFilter} incidents`} {incidentRangeHours >= 24 ? "on this day" : `within ${incidentRangeHours === 1 ? "30 minutes" : `${incidentRangeHours / 2} hours`} of this time`}</div>}
              </div>
            </section>
          )}
        </div>
      </div>
    </main>
  );
}

export function exportStatusLabel(status) {
  return ({
    queued: "Queued",
    running: "Creating",
    cancelling: "Cancelling",
    completed: "Ready",
    failed: "Failed",
    cancelled: "Cancelled",
  })[status] || String(status || "Unknown");
}

export function ExportCenterPage({ timeZone, onAssistantContextChange }) {
  const initialQuery = useMemo(() => new URLSearchParams(window.location.search), []);
  const [cameras, setCameras] = useState([]);
  const [cameraId, setCameraId] = useState(initialQuery.get("camera") || "");
  const [kind, setKind] = useState("all");
  const [status, setStatus] = useState("all");
  const [protectedOnly, setProtectedOnly] = useState(false);
  const [exportsList, setExportsList] = useState([]);
  const [total, setTotal] = useState(0);
  const [summary, setSummary] = useState({});
  const [limit, setLimit] = useState(50);
  const [selectedId, setSelectedId] = useState("");
  const [selectionMode, setSelectionMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState([]);
  const [editLabel, setEditLabel] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [actionBusy, setActionBusy] = useState("");
  const [revision, setRevision] = useState(0);

  const filteredExports = exportsList;
  const selected = filteredExports.find((item) => item.id === selectedId) || filteredExports[0] || null;
  const activeCount = Math.max(
    Number(summary.active) || 0,
    exportsList.filter((item) => ["queued", "running", "cancelling"].includes(item.status)).length,
  );
  const readyBytes = exportsList
    .filter((item) => item.status === "completed")
    .reduce((sum, item) => sum + (Number(item.size_bytes) || 0), 0);

  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/cameras", { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Camera status failed (${response.status})`);
        return response.json();
      })
      .then(setCameras)
      .catch(() => {});
    return () => controller.abort();
  }, []);

  async function loadExportCenter(signal) {
    const params = new URLSearchParams({ limit: String(limit) });
    if (cameraId) params.set("camera_id", cameraId);
    if (kind !== "all") params.set("kind", kind);
    if (status !== "all") params.set("status", status);
    if (protectedOnly) params.set("protected", "true");
    setLoading(true);
    setError("");
    const listRequest = fetch(`/api/exports?${params.toString()}`, { signal })
      .then(async (response) => {
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.detail || `Export list failed (${response.status})`);
        return payload;
      });
    const summaryRequest = fetch("/api/exports/summary", { signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Export summary failed (${response.status})`);
        return response.json();
      });
    const [listResult, summaryResult] = await Promise.allSettled([listRequest, summaryRequest]);
    if (signal.aborted) return;
    if (listResult.status === "fulfilled") {
      const jobs = listResult.value.exports || [];
      cacheExportJobs(jobs);
      setExportsList(jobs);
      setTotal(Number(listResult.value.total) || 0);
    }
    if (summaryResult.status === "fulfilled") setSummary(summaryResult.value);
    const failure = listResult.status === "rejected" ? listResult.reason : null;
    if (failure && failure.name !== "AbortError") setError(failure.message || "Unable to load exports");
    setLoading(false);
  }

  useEffect(() => {
    setLimit(50);
    setSelectedIds([]);
    setSelectionMode(false);
  }, [cameraId, kind, protectedOnly, status]);

  useVisiblePolling(loadExportCenter, activeCount ? 2_000 : 15_000, true, {
    restartKey: `${cameraId}|${kind}|${status}|${protectedOnly}|${limit}|${revision}`,
  });

  useEffect(() => {
    if (!selectedId || !filteredExports.some((item) => item.id === selectedId)) {
      setSelectedId(filteredExports[0]?.id || "");
    }
  }, [filteredExports, selectedId]);

  useEffect(() => {
    setEditLabel(selected?.label || "");
  }, [selected?.id, selected?.label]);

  useEffect(() => {
    const params = new URLSearchParams();
    if (cameraId) params.set("camera", cameraId);
    window.history.replaceState(null, "", appUrl(`/exports${params.size ? `?${params.toString()}` : ""}`));
  }, [cameraId]);

  useEffect(() => {
    onAssistantContextChange?.({
      page: "exports",
      camera_id: cameraId || selected?.camera_id || "",
      export_id: selected?.id || "",
      filters: { kind, status, protected: protectedOnly },
    });
  }, [cameraId, kind, onAssistantContextChange, protectedOnly, selected?.camera_id, selected?.id, status]);

  async function setExportProtection(item, nextProtected) {
    if (!item?.id || actionBusy) return;
    setActionBusy(item.id);
    setError("");
    try {
      const response = await fetch(`/api/exports/${encodeURIComponent(item.id)}/protection`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ protected: nextProtected }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `Protection update failed (${response.status})`);
      cacheExportJobs([payload]);
      setExportsList((current) => current.map((candidate) => candidate.id === payload.id ? payload : candidate));
      setRevision((current) => current + 1);
    } catch (actionError) {
      setError(actionError.message || "Unable to update export protection");
    } finally {
      setActionBusy("");
    }
  }

  async function saveExportLabel(item) {
    if (!item?.id || actionBusy) return;
    setActionBusy(item.id);
    setError("");
    try {
      const response = await fetch(`/api/exports/${encodeURIComponent(item.id)}/metadata`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label: editLabel.trim() }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `Rename failed (${response.status})`);
      cacheExportJobs([payload]);
      setExportsList((current) => current.map((candidate) => candidate.id === payload.id ? payload : candidate));
    } catch (actionError) {
      setError(actionError.message || "Unable to rename export");
    } finally {
      setActionBusy("");
    }
  }

  function toggleExportSelection(itemId) {
    setSelectedIds((current) => current.includes(itemId)
      ? current.filter((id) => id !== itemId)
      : [...current, itemId]);
  }

  async function runBatchAction(action) {
    if (!selectedIds.length || actionBusy) return;
    if (action === "delete" && !window.confirm(`Permanently delete ${selectedIds.length} selected export${selectedIds.length === 1 ? "" : "s"}? Protected selections will also be deleted.`)) return;
    setActionBusy("batch");
    setError("");
    try {
      const response = await fetch("/api/exports/batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ids: selectedIds, action }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `Batch action failed (${response.status})`);
      if (payload.errors?.length) {
        setError(`${payload.errors.length} export${payload.errors.length === 1 ? "" : "s"} could not be updated`);
      }
      cacheExportJobs(payload.results || []);
      if (action === "delete") removeCachedExportJobs(selectedIds);
      setSelectedIds([]);
      setSelectionMode(false);
      setSelectedId("");
      setRevision((current) => current + 1);
    } catch (actionError) {
      setError(actionError.message || "Unable to update selected exports");
    } finally {
      setActionBusy("");
    }
  }

  async function removeExport(item) {
    if (!item?.id || actionBusy) return;
    const active = ["queued", "running", "cancelling"].includes(item.status);
    const prompt = active
      ? `Cancel this ${item.kind === "timelapse" ? "timelapse" : "video export"}?`
      : `Permanently delete ${item.output_name || "this export"}?${item.protected ? " It is currently protected." : ""}`;
    if (!window.confirm(prompt)) return;
    setActionBusy(item.id);
    setError("");
    try {
      const force = item.protected && !active ? "?force=true" : "";
      const response = await fetch(`/api/exports/${encodeURIComponent(item.id)}${force}`, { method: "DELETE" });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `Delete failed (${response.status})`);
      if (payload.deleted) {
        removeCachedExportJobs([item.id]);
        setExportsList((current) => current.filter((candidate) => candidate.id !== item.id));
        setSelectedId("");
      } else {
        cacheExportJobs([payload]);
        setExportsList((current) => current.map((candidate) => candidate.id === payload.id ? payload : candidate));
      }
      setRevision((current) => current + 1);
    } catch (actionError) {
      setError(actionError.message || "Unable to remove export");
    } finally {
      setActionBusy("");
    }
  }

  return (
    <main className="recordings-v2-page export-center-page">
      <section className="export-center-workspace">
        <header className="export-center-toolbar">
          <div className="export-center-filters">
            <TimelineCameraPicker
              cameras={cameras}
              value={cameraId}
              onChange={setCameraId}
              allOption={{ value: "", label: "All cameras" }}
              ariaLabel="Select export camera"
            />
            <label>Type<select value={kind} onChange={(event) => setKind(event.target.value)}><option value="all">All exports</option><option value="recording">Video clips</option><option value="timelapse">Timelapses</option></select></label>
            <label>Status<select value={status} onChange={(event) => setStatus(event.target.value)}><option value="all">Any status</option><option value="completed">Ready</option><option value="active">In progress</option><option value="failed">Failed</option><option value="cancelled">Cancelled</option></select></label>
            <button type="button" className={protectedOnly ? "active" : ""} onClick={() => setProtectedOnly((current) => !current)}><ShieldCheck size={15} />Protected</button>
            <button type="button" className={selectionMode ? "active" : ""} onClick={() => { setSelectionMode((current) => !current); setSelectedIds([]); }}><Check size={15} />Select</button>
            <button type="button" onClick={() => setRevision((current) => current + 1)} disabled={loading}><RefreshCcw className={loading ? "spin" : ""} size={15} />Refresh</button>
          </div>
          <span>{filteredExports.length.toLocaleString()} shown · {total.toLocaleString()} matching · {Number(summary.total || 0).toLocaleString()} overall · {formatBytes(Number(summary.bytes) || readyBytes)} / {formatBytes(Number(summary.max_storage_bytes))}</span>
        </header>

        <section className="export-center-review">
          <div className="export-center-player">
            {selected?.status === "completed" && selected.media_url ? <video key={selected.id} src={mediaUrl(selected.media_url)} controls playsInline preload="metadata" /> : null}
            {!selected && !loading ? <div className="export-center-empty"><Film size={30} /><strong>No exports match these filters</strong><span>Create a clip or timelapse from recording history or ask the SurvNG Assistant.</span></div> : null}
            {selected && selected.status !== "completed" ? <div className={`export-center-job-state ${selected.status}`}><RefreshCcw className={["queued", "running", "cancelling"].includes(selected.status) ? "spin" : ""} size={30} /><strong>{selected.phase || exportStatusLabel(selected.status)}</strong><span>{selected.error || `${Math.round(Number(selected.progress) || 0)}% complete`}</span></div> : null}
          </div>
          <aside className="export-center-details">
            {selected ? <>
              <header><div><strong>{selected.label || (selected.kind === "timelapse" ? "Timelapse" : "Video clip")}</strong><span className={`export-status ${selected.status}`}>{exportStatusLabel(selected.status)}</span></div><small>{selected.output_name || `${selected.camera_id} export`}</small></header>
              <div className="export-center-rename"><input value={editLabel} onChange={(event) => setEditLabel(event.target.value)} placeholder="Add a useful name" maxLength="120" /><button type="button" onClick={() => saveExportLabel(selected)} disabled={actionBusy === selected.id || editLabel.trim() === (selected.label || "")}><Save size={14} />Save</button></div>
              <dl>
                <div><dt>Camera</dt><dd>{cameras.find((camera) => camera.id === selected.camera_id)?.name || selected.camera_id}</dd></div>
                <div><dt>Stream</dt><dd>{selected.source === "live" ? "Sub" : "Main"}</dd></div>
                <div><dt>From</dt><dd>{formatDateTime(Number(selected.start_epoch), timeZone)}</dd></div>
                <div><dt>To</dt><dd>{formatDateTime(Number(selected.end_epoch), timeZone)}</dd></div>
                <div><dt>Duration</dt><dd>{formatDuration(Number(selected.end_epoch) - Number(selected.start_epoch))}</dd></div>
                <div><dt>Resolution</dt><dd>{Number(selected.options?.height) > 0 ? `${Number(selected.options.height)}p` : Number(selected.options?.width) > 0 ? `${Number(selected.options.width)}px wide (legacy)` : "Original"}</dd></div>
                {selected.kind === "timelapse" ? <div><dt>Timelapse</dt><dd>Every {formatDuration(Number(selected.options?.sample_interval_seconds) || 30)} · {Number(selected.options?.output_fps) || 30} FPS</dd></div> : null}
                <div><dt>File size</dt><dd>{formatBytes(Number(selected.size_bytes))}</dd></div>
                <div><dt>Created</dt><dd>{formatDateTime(selected.created_at, timeZone)}</dd></div>
                <div><dt>Created by</dt><dd>{selected.origin === "assistant" ? "SurvNG Assistant" : "Recording viewer"}</dd></div>
                <div><dt>Retention</dt><dd>{selected.protected ? "Protected" : selected.expires_at ? `Until ${formatDateTime(selected.expires_at, timeZone)}` : "While active"}</dd></div>
              </dl>
              <div className="export-center-actions">
                {selected.status === "completed" && selected.download_url ? <a className="primary" href={mediaUrl(selected.download_url)}><Download size={15} />Download</a> : null}
                {selected.status === "completed" ? <button type="button" onClick={() => setExportProtection(selected, !selected.protected)} disabled={actionBusy === selected.id}><ShieldCheck size={15} />{selected.protected ? "Unprotect" : "Protect"}</button> : null}
                <button type="button" className="danger" onClick={() => removeExport(selected)} disabled={actionBusy === selected.id || selected.status === "cancelling"}><Trash2 size={15} />{["queued", "running"].includes(selected.status) ? "Cancel" : "Delete"}</button>
              </div>
            </> : null}
          </aside>
        </section>

        {error ? <div className="export-center-error"><CircleAlert size={15} />{error}</div> : null}
        {selectionMode ? <section className="export-center-batch" aria-label="Selected export actions"><strong>{selectedIds.length} selected</strong><button type="button" onClick={() => runBatchAction("protect")} disabled={!selectedIds.length || Boolean(actionBusy)}><ShieldCheck size={14} />Protect</button><button type="button" onClick={() => runBatchAction("unprotect")} disabled={!selectedIds.length || Boolean(actionBusy)}><ShieldCheck size={14} />Unprotect</button><button type="button" className="danger" onClick={() => runBatchAction("delete")} disabled={!selectedIds.length || Boolean(actionBusy)}><Trash2 size={14} />Delete</button><button type="button" onClick={() => { setSelectedIds([]); setSelectionMode(false); }}>Done</button></section> : null}
        <section className="export-center-library" aria-label="Saved exports">
          {filteredExports.map((item) => (
            <button key={item.id} type="button" className={`${item.id === selected?.id && !selectionMode ? "active" : ""} ${selectedIds.includes(item.id) ? "selected" : ""} ${item.status}`} onClick={() => selectionMode ? toggleExportSelection(item.id) : setSelectedId(item.id)} aria-pressed={selectionMode ? selectedIds.includes(item.id) : undefined}>
              <span className="export-center-card-icon">{item.kind === "timelapse" ? <Clock3 size={23} /> : <Film size={23} />}</span>
              <span className="export-center-card-copy"><strong>{item.label || cameras.find((camera) => camera.id === item.camera_id)?.name || item.camera_id}</strong><small>{item.label ? `${cameras.find((camera) => camera.id === item.camera_id)?.name || item.camera_id} · ` : ""}{item.kind === "timelapse" ? "Timelapse" : "Video clip"} · {formatDateTime(Number(item.start_epoch), timeZone)}</small><small>{formatDuration(Number(item.end_epoch) - Number(item.start_epoch))} · {formatBytes(Number(item.size_bytes))}</small></span>
              <span className={`export-center-card-status ${item.status}`}>{item.protected ? <ShieldCheck size={13} /> : null}{exportStatusLabel(item.status)}</span>
            </button>
          ))}
          {loading && !filteredExports.length ? <div className="export-center-loading"><RefreshCcw className="spin" size={20} />Loading exports</div> : null}
          {!loading && exportsList.length < total && limit < 1000 ? <button type="button" className="export-center-load-more" onClick={() => setLimit((current) => Math.min(1000, current + 50))}><Plus size={18} /><span>Load older</span><small>{(total - exportsList.length).toLocaleString()} remaining</small></button> : null}
        </section>
      </section>
    </main>
  );
}

