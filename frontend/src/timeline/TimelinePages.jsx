import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  Camera,
  CarFront,
  Check,
  CircleAlert,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  Clock3,
  Download,
  Film,
  Grid2X2,
  Images,
  Search,
  Pause,
  Play,
  Plus,
  Radar,
  RefreshCcw,
  Save,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  SkipBack,
  SkipForward,
  Trash2,
  UserRound,
  Video,
  X,
} from "lucide-react";
import { browserStorage } from "../storage.mjs";
import { useVisiblePolling } from "../visibilityPolling.mjs";
import { ACTIVE_EXPORT_STATUSES, cacheExportJobs, exportIsActive, fetchExportJob, removeCachedExportJobs } from "../exportPolling.mjs";
import { adjustRecordingExportRange, describePlaybackError, gridPlaybackNeedsSeek, isUnsupportedPlaybackError, mergeRecordingAvailability, playbackMediaTimeForEpoch, playbackRowsCoverEpoch } from "../recordingPlayback.mjs";
import { recordingCameraAspect, recordingGridBestEpoch } from "../recordingGrid.mjs";
import { expectedTimelineCameras, filteredTimelineCameras, invalidateTimelineIdentityCache, mergeTimelineIncidentIdentity, normalizedTimelinePlaybackRate, parseTimelineView, resolveTimelineHeroCameraId, timelineEventMatchesFilter, timelineIdentityDetailEventId, timelineIncidentIncludesEvent, timelinePanViewport, timelinePlayheadInComfortZone, timelineStageCameras, timelineStagePage, timelineTickIntervalSeconds, timelineViewport, TIMELINE_PLAYBACK_RATES } from "../timelineWorkspace.mjs";
import { addSemanticSearchHistory, clearSemanticSearchSession, readSemanticSearchHistory, readSemanticSearchSession, semanticSearchResultsForCamera, writeSemanticSearchHistory, writeSemanticSearchSession } from "../semanticSearchState.mjs";
import { appUrl, mediaUrl, incidentRecordingContext, recordingsHref, fetch } from "../shared/api.js";
import { ALL_RECORDING_CAMERAS_ID } from "../shared/constants.js";
import { formatDateTime, formatTimeOnly, formatExportHandleTime, formatBytes, formatDuration } from "../shared/format.js";
import { dateKeyForTimeZone, addDaysToDateKey, zonedDateSecondToEpoch } from "../shared/datetime.js";
import { preferredStreamSource } from "../shared/cameras.js";
import { IdentityChip } from "../shared/identity.jsx";
import { eventThumbnailUrl, recordingDayUrl, recordingWindowUrl, recordingUpdatesUrl, recordingDayHlsUrl, recordingGridDayUrl, recordingGridUpdatesUrl, recordingPreviewUrl } from "../shared/mediaUrls.js";
import { ShakaVideo } from "../shared/media.jsx";
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

export function recordingIncidentEndEpoch(incident) {
  const direct = Number(incident?.last_epoch);
  if (Number.isFinite(direct)) return direct;
  const parsed = new Date(incident?.end_at || incident?.created_at || "").getTime() / 1000;
  return Number.isFinite(parsed) ? parsed : recordingIncidentEpoch(incident);
}

export function recordingEvidenceTypeLabel(type) {
  if (type === "motion") return "motion-only";
  if (type === "object") return "object";
  return "total";
}

export function recordingPlaybackTimeline(rows) {
  let mediaOffset = 0;
  return (rows || [])
    .map((item) => ({
      ...item,
      start_epoch: Number(item.start_epoch),
      end_epoch: Number(item.end_epoch),
    }))
    .filter((item) => Number.isFinite(item.start_epoch) && Number.isFinite(item.end_epoch))
    .sort((left, right) => left.start_epoch - right.start_epoch)
    .map((item) => {
      const duration = Math.max(0.01, Number(item.duration_seconds) || item.end_epoch - item.start_epoch);
      const result = { ...item, media_start: mediaOffset, media_end: mediaOffset + duration };
      mediaOffset += duration;
      return result;
    });
}

export function RecordingGridTile({ camera, source, epoch, playing, primary, synchronized = false, onFocus, onSelect }) {
  const videoRef = useRef(null);
  const previousEpochRef = useRef(null);
  const [playback, setPlayback] = useState(null);
  const [error, setError] = useState("");
  const bucket = Math.floor(Math.max(0, Number(epoch) || 0) / (15 * 60)) * 15 * 60;
  const previewEpoch = Math.floor(Math.max(0, Number(epoch) || 0) / 2) * 2;
  const retryScope = `${camera?.id || ""}:${source}:${bucket}`;
  const [retryState, setRetryState] = useState({ scope: retryScope, count: 0 });
  const retryCount = retryState.scope === retryScope ? retryState.count : 0;
  const timeline = useMemo(
    () => recordingPlaybackTimeline(playback?.rows || []),
    [playback?.rows],
  );
  const exactMediaTime = playbackMediaTimeForEpoch(timeline, epoch);
  const mediaTime = Number.isFinite(exactMediaTime)
    ? exactMediaTime
    : playbackMediaTimeForEpoch(timeline, epoch, 1.25);
  const briefGap = !Number.isFinite(exactMediaTime) && Number.isFinite(mediaTime);
  const hasCoverage = Number.isFinite(mediaTime);

  useEffect(() => {
    if ((!primary && !synchronized) || !camera?.id || !bucket) return undefined;
    const controller = new AbortController();
    let cancelled = false;
    setPlayback(null);
    setError("");
    const candidates = source === "main" ? ["main", "live"] : ["live", "main"];
    async function load() {
      for (const candidate of candidates) {
        const response = await fetch(
          recordingWindowUrl(camera.id, bucket, bucket + 15 * 60, candidate),
          { signal: controller.signal },
        );
        if (!response.ok) continue;
        const payload = await response.json();
        if (!(payload.recordings || []).length) continue;
        if (!cancelled) {
          setPlayback({
            source: candidate,
            start: Number(payload.start_epoch),
            end: Number(payload.end_epoch),
            rows: payload.recordings,
            targetEpoch: epoch,
          });
        }
        return;
      }
      if (!cancelled) setError("No recording at this time");
    }
    load().catch((loadError) => {
      if (!cancelled && loadError.name !== "AbortError") setError("Recording unavailable");
    });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [bucket, camera?.id, primary, retryCount, source, synchronized]);

  useEffect(() => {
    const nearLive = Math.abs(Date.now() / 1000 - epoch) <= 5 * 60;
    if (retryCount >= 4 || !nearLive || (!error && (!playback || hasCoverage))) return undefined;
    const timer = window.setTimeout(() => setRetryState((current) => ({
      scope: retryScope,
      count: current.scope === retryScope ? current.count + 1 : 1,
    })), 3000);
    return () => window.clearTimeout(timer);
  }, [epoch, error, hasCoverage, playback, retryCount, retryScope]);

  useEffect(() => {
    const video = videoRef.current;
    const previousEpoch = previousEpochRef.current;
    previousEpochRef.current = epoch;
    if (!video || !Number.isFinite(mediaTime)) {
      video?.pause();
      return;
    }
    if (briefGap) {
      video.pause();
      return;
    }
    if (gridPlaybackNeedsSeek({
      currentTime: video.currentTime,
      targetTime: mediaTime,
      playing,
      epochDelta: Number.isFinite(previousEpoch) ? epoch - previousEpoch : null,
    })) video.currentTime = mediaTime;
    if (playing) video.play().catch(() => { });
    else video.pause();
  }, [briefGap, mediaTime, playing]);

  const manifestUrl = (primary || synchronized) && playback
    ? recordingDayHlsUrl(camera.id, playback.start, playback.end, playback.source)
    : "";
  const initialMediaTime = playbackMediaTimeForEpoch(timeline, playback?.targetEpoch, 1.25);
  const displayedSource = playback?.source || source;
  const aspect = recordingCameraAspect(camera, displayedSource);
  return (
    <article
      className={`recording-grid-tile${primary ? " primary" : " companion"}${primary && hasCoverage ? "" : primary ? " gap" : ""}`}
      style={{ "--recording-aspect": aspect }}
    >
      {manifestUrl ? <ShakaVideo
        ref={videoRef}
        src={manifestUrl}
        mimeType="application/vnd.apple.mpegurl"
        startTime={Number.isFinite(initialMediaTime) ? initialMediaTime : 0}
        bufferingGoal={20}
        muted
        playsInline
        preload="auto"
        onReady={(_player, video) => {
          setError("");
          if (Number.isFinite(mediaTime)) video.currentTime = mediaTime;
          if (playing && Number.isFinite(mediaTime)) video.play().catch(() => { });
        }}
        onError={() => setError("Playback unavailable")}
      /> : <img className="recording-grid-preview" src={recordingPreviewUrl(camera.id, previewEpoch, source)} alt="" />}
      <button
        type="button"
        className="recording-grid-focus-hit media-surface-action"
        onClick={() => onFocus(camera.id)}
        aria-label={primary ? `${camera.name} is the primary recording` : `Show ${camera.name} as primary recording`}
        title={primary ? "Primary camera" : "Show as primary"}
        disabled={primary}
      />
      <button type="button" className="recording-grid-camera" onClick={() => onSelect(camera.id)} title={`Open ${camera.name} recording`}>
        <Camera size={14} /><span>{camera.name}</span>
        {playback?.source && playback.source !== source ? <em>{playback.source === "live" ? "Sub" : "Main"}</em> : null}
      </button>
      {(primary || synchronized) && !playback && !error ? <div className="recording-grid-status"><RefreshCcw className="spin" size={17} />Loading</div> : null}
      {(primary || synchronized) && (!hasCoverage || error) && playback ? <div className="recording-grid-status"><Film size={17} />{error || "No recording at this time"}</div> : null}
      {(primary || synchronized) && error && !playback ? <div className="recording-grid-status"><Film size={17} />{error}</div> : null}
    </article>
  );
}

export function RecordingCameraGrid({ cameras, source, epoch, playing, onSelect }) {
  const [cameraPage, setCameraPage] = useState(0);
  const page = useMemo(() => timelineStagePage(cameras, cameraPage), [cameraPage, cameras]);
  const [primaryCameraId, setPrimaryCameraId] = useState(page.cameras[0]?.id || "");
  const gridRef = useRef(null);
  const displayedCameras = useMemo(
    () => timelineStageCameras(page.cameras, primaryCameraId),
    [page.cameras, primaryCameraId],
  );

  useEffect(() => {
    if (!displayedCameras.some((camera) => camera.id === primaryCameraId)) {
      setPrimaryCameraId(displayedCameras[0]?.id || "");
    }
  }, [displayedCameras, primaryCameraId]);

  const primaryCamera = displayedCameras.find((camera) => camera.id === primaryCameraId) || displayedCameras[0];
  const orderedCameras = primaryCamera
    ? [primaryCamera, ...displayedCameras.filter((camera) => camera.id !== primaryCamera.id)]
    : [];
  return <div ref={gridRef} className="recording-camera-grid stage-layout">
    {primaryCamera ? <RecordingGridTile key={primaryCamera.id} camera={primaryCamera} source={source} epoch={epoch} playing={playing} primary onFocus={setPrimaryCameraId} onSelect={onSelect} /> : null}
    <div className="recording-grid-companions">
      {orderedCameras.slice(1).map((camera) => <RecordingGridTile
        key={camera.id}
        camera={camera}
        source={source}
        epoch={epoch}
        playing={playing}
        primary={false}
        synchronized
        onFocus={setPrimaryCameraId}
        onSelect={onSelect}
      />)}
    </div>
    {page.pages > 1 ? <div className="recording-stage-pagination">
      <button type="button" onClick={() => setCameraPage((current) => Math.max(0, current - 1))} disabled={page.page === 0} aria-label="Previous camera page"><ChevronLeft size={15} /></button>
      <span>{page.page + 1} / {page.pages}</span>
      <button type="button" onClick={() => setCameraPage((current) => Math.min(page.pages - 1, current + 1))} disabled={page.page >= page.pages - 1} aria-label="Next camera page"><ChevronRight size={15} /></button>
    </div> : null}
  </div>;
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
    {companions.map((camera) => <button key={camera.id} type="button" onClick={() => onSelect(camera.id)} aria-label={`Show ${camera.name} recording at the current time`}>
      <img src={recordingPreviewUrl(camera.id, previewEpoch, source)} alt="" loading="lazy" decoding="async" />
      <span><i className={(source === "main" ? camera.recording : camera.sub_recording) ? "online" : ""} />{camera.name}</span>
    </button>)}
  </div>;
}

export function TimelineCameraPicker({ cameras, value, onChange, allOption = null, ariaLabel = "Select timeline camera" }) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const rootRef = useRef(null);
  const searchRef = useRef(null);
  const allSelected = Boolean(allOption) && value === allOption.value;
  const selectedCamera = cameras.find((camera) => camera.id === value);
  const selected = allSelected
    ? { id: allOption.value, name: allOption.label, recording: cameras.some((camera) => camera.recording || camera.sub_recording) }
    : selectedCamera || cameras[0] || (allOption ? { id: allOption.value, name: allOption.label, recording: false } : null);
  const matches = filteredTimelineCameras(cameras, query);
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

export function TimelineDatePicker({ value, max, onChange }) {
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
      <button
        ref={rootRef}
        type="button"
        className="recordings-v2-date-toggle"
        aria-label="Recording day"
        aria-expanded={open}
        aria-haspopup="dialog"
        onClick={() => setOpen((current) => !current)}
      >
        {formatTimelineDateLabel(value)}
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

export function SemanticSearchPage({ timeZone, onAssistantContextChange }) {
  const initialQuery = useMemo(() => new URLSearchParams(window.location.search), []);
  const localStorage = useMemo(() => browserStorage(window), []);
  const sessionStorage = useMemo(() => {
    try { return window.sessionStorage; } catch { return null; }
  }, []);
  const initialQueryText = initialQuery.get("q") || "";
  const initialCameraId = initialQuery.get("camera") || "";
  const restoredSearch = useMemo(
    () => readSemanticSearchSession(sessionStorage, initialQueryText, initialCameraId),
    [initialCameraId, initialQueryText, sessionStorage],
  );
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
  }
  useEffect(() => { void loadSearchSetup(); }, []);
  useEffect(() => {
    if (initialSearchRunRef.current || restoredSearch || !initialQueryText.trim() || !status) return;
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

  function selectCamera(nextCameraId) {
    if (loading) return;
    const selectedCameraId = String(nextCameraId || "");
    setCameraId(selectedCameraId);
    const committedQuery = String(submittedQuery || query || "").trim();
    if (committedQuery) void runSearch(null, committedQuery, selectedCameraId);
  }

  return <main className="search-page">
    <header className="search-commandbar">
      <div><Search size={18} /><span><strong>Smart Search</strong><small>Find incidents by describing what you remember</small></span></div>
      {setupError ? <button type="button" className="search-setup-retry" onClick={() => void loadSearchSetup()}><RefreshCcw size={15} />Retry status</button> : <span className={`semantic-status ${status?.state || ""}`}>{status?.state === "ready" ? `${Number(status.event_count || 0).toLocaleString()} incidents indexed` : status?.state || "Loading"}</span>}
    </header>
    <aside className="search-scope-panel">
      <div className="search-scope-heading"><strong>Camera</strong><small>Refine current results</small></div>
      <div className="search-camera-list">
        <button type="button" className={!cameraId ? "active" : ""} onClick={() => selectCamera("")} disabled={loading} aria-pressed={!cameraId}><Search size={16} /><span>All cameras</span><i /></button>
        {cameras.map((camera) => <button type="button" key={camera.id} className={cameraId === camera.id ? "active" : ""} onClick={() => selectCamera(camera.id)} disabled={loading} aria-pressed={cameraId === camera.id}><Camera size={16} /><span>{camera.name}</span><i className={camera.running ? "online" : ""} /></button>)}
      </div>
      {searchHistory.length ? <section className="search-history-panel" aria-labelledby="search-history-title">
        <div id="search-history-title"><Clock3 size={15} /><strong>Recent searches</strong></div>
        {searchHistory.map((item) => {
          const cameraName = item.cameraId ? cameras.find((camera) => camera.id === item.cameraId)?.name || item.cameraId : "All cameras";
          return <button type="button" key={`${item.query.toLocaleLowerCase()}-${item.cameraId}`} onClick={() => void runSearch(null, item.query, item.cameraId)} disabled={loading} title={`Search ${cameraName}`}><strong>{item.query}</strong><small>{cameraName}</small></button>;
        })}
      </section> : null}
    </aside>
    <section className="semantic-search-workspace">
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
          return <article key={item.id} aria-label={`${matchLabel} at ${cameraName}, ${observedAt}`}><div className="semantic-result-image"><img src={mediaUrl(result.snapshot_url)} alt={`${cameraName} search result`} loading="lazy" /><span title={`Raw visual similarity ${Number(result.score || 0).toFixed(3)}`}>{matchLabel}</span></div><footer><div><strong>{cameraName}</strong><small>{observedAt}</small><IdentityChip item={item} className="semantic-result-identity" /></div><nav aria-label={`Actions for ${cameraName} result`}><a href={appUrl(`/incidents?event_ids=${item.id}`)}>Open incident</a><a href={recordingsHref(context)}><Play size={14} />Timeline</a></nav></footer></article>;
        })}
        {!loading && !error && !visibleResults.length ? <div className="semantic-search-empty"><Search size={28} /><strong>{results.length && cameraId ? "No matching results from this camera" : "Search indexed incidents by appearance"}</strong><span>{results.length && cameraId ? "Choose All cameras or another camera to widen the current results." : "Results link to the exact incident and recording time."}</span></div> : null}
      </div>
    </section>
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
  const timelineInspectorTriggerRef = useRef(null);
  const desiredEpochRef = useRef(initialEpoch);
  const autoplayRef = useRef(false);
  const codecFallbackRef = useRef(false);
  const playbackRequestRef = useRef(0);
  const latestAvailabilityRef = useRef(null);
  const pendingSeekEpochRef = useRef(null);
  const pendingSeekModeRef = useRef(null);
  const playbackRetryRef = useRef({ attempts: 0, timer: null });
  const gridRefreshCursorRef = useRef(null);
  const recordingUpdatesInFlightRef = useRef(false);
  const gridUpdatesInFlightRef = useRef(false);
  const selectedIncidentIdentityCacheRef = useRef(new Map());
  const [cameras, setCameras] = useState([]);
  const [cameraTransitionRoutes, setCameraTransitionRoutes] = useState([]);
  const [cameraId, setCameraId] = useState(initialView.cameraId);
  const [source, setSource] = useState(initialView.source || (initialView.cameraId === ALL_RECORDING_CAMERAS_ID ? "live" : preferredStreamSource()));
  const [date, setDate] = useState(initialDate);
  const [recordings, setRecordings] = useState([]);
  const [playbackDetail, setPlaybackDetail] = useState(null);
  const [events, setEvents] = useState([]);
  const [eventFilter, setEventFilter] = useState(initialView.eventFilter);
  const [timelineLanes, setTimelineLanes] = useState(initialView.lanes);
  const [incidentRangeHours, setIncidentRangeHours] = useState(initialView.windowHours);
  const [availableSources, setAvailableSources] = useState([]);
  const [playhead, setPlayhead] = useState(null);
  const [loading, setLoading] = useState(true);
  const [playbackError, setPlaybackError] = useState("");
  const [playbackErrorStage, setPlaybackErrorStage] = useState("");
  const [playbackNotice, setPlaybackNotice] = useState("");
  const [playbackBlocked, setPlaybackBlocked] = useState(false);
  const [playbackWindow, setPlaybackWindow] = useState(null);
  const [playbackWindowRevision, setPlaybackWindowRevision] = useState(0);
  const [manifestRetryToken, setManifestRetryToken] = useState(0);
  const [recordingIndexRevision, setRecordingIndexRevision] = useState(0);
  const [followTarget, setFollowTarget] = useState(null);
  const [exportRange, setExportRange] = useState(null);
  const [exportKind, setExportKind] = useState("recording");
  const [exportOptions, setExportOptions] = useState({ interval: 30, fps: 30, clipHeight: 0, timelapseHeight: 720 });
  const [exportLabel, setExportLabel] = useState("");
  const [exportJob, setExportJob] = useState(null);
  const [exportError, setExportError] = useState("");
  const [exportSubmitting, setExportSubmitting] = useState(false);
  const [gridPlaying, setGridPlaying] = useState(false);
  const [selectedEventId, setSelectedEventId] = useState(initialView.eventId);
  const [selectedIncidentIdentity, setSelectedIncidentIdentity] = useState(null);
  const [selectedIdentityRevision, setSelectedIdentityRevision] = useState(0);
  const [timelineInspectorTab, setTimelineInspectorTab] = useState(initialView.inspector);
  const [timelineInspectorOpen, setTimelineInspectorOpen] = useState(false);
  const [investigationOpen, setInvestigationOpen] = useState(false);
  const [followPlayhead, setFollowPlayhead] = useState(true);
  const [playbackRate, setPlaybackRate] = useState(initialView.speed);
  const [timelineViewportAnchor, setTimelineViewportAnchor] = useState(initialView.at);

  useEffect(() => {
    if (!timelineInspectorOpen) return undefined;
    const closeOnEscape = (event) => {
      if (event.key !== "Escape") return;
      setTimelineInspectorOpen(false);
      window.requestAnimationFrame(() => timelineInspectorTriggerRef.current?.focus());
    };
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [timelineInspectorOpen]);

  const isAllCameras = false;
  const activeCameraId = resolveTimelineHeroCameraId(cameras, cameraId);
  const dayStart = useMemo(() => zonedDateSecondToEpoch(date, 0, timeZone), [date, timeZone]);
  const nextDate = addDaysToDateKey(date, 1);
  const dayEnd = useMemo(() => zonedDateSecondToEpoch(nextDate, 0, timeZone), [nextDate, timeZone]);
  const daySeconds = Math.max(1, dayEnd - dayStart);

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
  const playbackTimeline = useMemo(() => {
    if (!playbackDetail) return [];
    return recordingPlaybackTimeline(playbackDetail.rows);
  }, [playbackDetail]);
  const loadedPlaybackWindow = playbackDetail;
  const manifestUrl = !isAllCameras && activeCameraId && playbackDetail && playbackTimeline.length
    ? `${recordingDayHlsUrl(activeCameraId, playbackDetail.start, playbackDetail.end, source)}&reload=${playbackDetail.revision || 0}-${manifestRetryToken}`
    : "";
  const manifestStartTime = useMemo(() => {
    if (!playbackTimeline.length) return null;
    const retainedEpoch = desiredEpochRef.current;
    const initialEpoch = Number.isFinite(retainedEpoch) && retainedEpoch >= dayStart && retainedEpoch < dayEnd
      ? retainedEpoch
      : date === today ? Date.now() / 1000 : timeline[0].start_epoch;
    return epochToPlaybackMediaTime(initialEpoch);
  }, [manifestUrl, playbackTimeline]);

  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = normalizedTimelinePlaybackRate(playbackRate);
  }, [playbackRate]);

  const filteredEvents = useMemo(() => events
    .filter((event) => timelineEventMatchesFilter(event, eventFilter))
    .filter((event) => (event.has_objects ? timelineLanes.object : timelineLanes.motion))
    .map((event) => ({ ...event, incident_epoch: recordingIncidentEpoch(event) }))
    .filter((event) => Number.isFinite(event.incident_epoch))
    .sort((left, right) => left.incident_epoch - right.incident_epoch), [eventFilter, events, timelineLanes.motion, timelineLanes.object]);

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

  const nearbyEvents = useMemo(() => filteredEvents.filter((event) => (
    event.incident_epoch >= timelineView.startEpoch
    && event.incident_epoch <= timelineView.endEpoch
  )), [filteredEvents, timelineView.endEpoch, timelineView.startEpoch]);
  const selectedEventSummary = nearbyEvents.find((event) => event.id === selectedEventId)
    || timelineEvents.find((event) => event.id === selectedEventId)
    || null;
  const selectedIdentityEventId = timelineIdentityDetailEventId(selectedEventSummary);
  const selectedEvent = selectedEventSummary && selectedIncidentIdentity?.eventId === selectedIdentityEventId
    ? mergeTimelineIncidentIdentity(selectedEventSummary, selectedIncidentIdentity.detail)
    : selectedEventSummary;

  useAppEvents(({ type, data }) => {
    if (type !== "identity_update") return;
    const eventId = Number(data?.event_id);
    if (!Number.isInteger(eventId) || eventId <= 0) return;
    const invalidatedCacheIds = invalidateTimelineIdentityCache(selectedIncidentIdentityCacheRef.current, eventId);
    const selectedCacheInvalidated = invalidatedCacheIds.includes(selectedIdentityEventId);
    if (selectedCacheInvalidated || timelineIncidentIncludesEvent(selectedEventSummary, eventId)) {
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
    if (!followPlayhead || !Number.isFinite(playhead)) return;
    const playing = isAllCameras ? gridPlaying : autoplayRef.current;
    if (!playing) return;
    if (timelinePlayheadInComfortZone(timelineView, playhead)) return;
    setTimelineViewportAnchor(playhead);
  }, [followPlayhead, gridPlaying, isAllCameras, playhead, timelineView]);
  const selectedEventEnd = selectedEvent ? recordingIncidentEndEpoch(selectedEvent) : null;
  const selectedEventDuration = selectedEvent && Number.isFinite(selectedEventEnd)
    ? Math.max(0, selectedEventEnd - selectedEvent.incident_epoch)
    : 0;
  const selectedEventConfidence = selectedEvent
    ? Math.max(0, ...(selectedEvent.objects || []).map((object) => Number(object.confidence) || 0), Number(selectedEvent.confidence) || 0)
    : 0;
  const displayedTimelineEvents = useMemo(() => {
    if (!selectedEvent || nearbyEvents.some((event) => event.id === selectedEvent.id)) return nearbyEvents;
    if (
      selectedEvent.incident_epoch < timelineView.startEpoch
      || selectedEvent.incident_epoch > timelineView.endEpoch
    ) return nearbyEvents;
    return [...nearbyEvents, selectedEvent].sort((left, right) => left.incident_epoch - right.incident_epoch);
  }, [nearbyEvents, selectedEvent, timelineView.endEpoch, timelineView.startEpoch]);

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
    setExportRange(null);
    setExportJob(null);
    setExportError("");
    setExportLabel("");
    selectedIncidentIdentityCacheRef.current.clear();
    setSelectedIncidentIdentity(null);
  }, [activeCameraId, date, source]);

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

  function snapToRecording(epoch) {
    if (!timeline.length) return null;
    const bounded = Math.max(dayStart, Math.min(dayEnd - 0.01, epoch));
    const latest = timeline[timeline.length - 1];
    if (bounded >= latest.end_epoch) return latest.end_epoch - 0.01;
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

  function requestRecordingPlay(video, showBlocked = true) {
    if (!video) return;
    video.play().then(() => {
      setPlaybackBlocked(false);
    }).catch((error) => {
      if (showBlocked && error?.name === "NotAllowedError") {
        setPlaybackBlocked(true);
        setPlaybackNotice("Tap to play recording");
      }
    });
  }

  function playAt(epoch, autoplay = true, preserveTimelineViewport = false) {
    const target = snapToRecording(epoch);
    if (target === null || !activeCameraId) return;
    if (
      !preserveTimelineViewport
      && (
      target < timelineView.startEpoch
      || target > timelineView.endEpoch
      )
    ) {
      setFollowPlayhead(true);
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
    if (autoplay && video) requestRecordingPlay(video, false);
    const mediaTime = epochToPlaybackMediaTime(target);
    if (inCurrentWindow && coveredByCurrentManifest && video && Number.isFinite(mediaTime)) {
      pendingSeekEpochRef.current = target;
      pendingSeekModeRef.current = "local";
      playbackRequestRef.current += 1;
      setPlaybackWindow(null);
      setPlaybackNotice("Seeking...");
      video.currentTime = mediaTime;
      if (autoplay) requestRecordingPlay(video);
      else setPlaybackNotice("");
    } else {
      pendingSeekEpochRef.current = target;
      pendingSeekModeRef.current = "window";
      setPlaybackNotice("Loading recording...");
      requestPlaybackWindow(nextWindow);
    }
  }

  function panTimelineViewport(deltaSeconds) {
    const next = timelinePanViewport(dayStart, dayEnd, timelineView, deltaSeconds);
    if (next.startEpoch === timelineView.startEpoch) return;
    setFollowPlayhead(false);
    setTimelineViewportAnchor((next.startEpoch + next.endEpoch) / 2);
  }

  function returnToPlayhead() {
    if (!Number.isFinite(playhead)) return;
    setFollowPlayhead(true);
    setTimelineViewportAnchor(playhead);
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
      if (document.hidden) setGridPlaying(false);
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
    setRecordings([]);
    playbackRequestRef.current += 1;
    setPlaybackDetail(null);
    setEvents([]);
    setAvailableSources([]);
    setPlaybackWindow(null);
    setGridPlaying(false);
    gridRefreshCursorRef.current = null;
    setManifestRetryToken(0);
    setFollowTarget(null);
    pendingSeekEpochRef.current = null;
    pendingSeekModeRef.current = null;
    if (Number.isFinite(playhead)) desiredEpochRef.current = playhead;
    setPlayhead(null);
    const video = videoRef.current;
    if (video) {
      video.pause();
    }
    codecFallbackRef.current = false;
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
        if (!nextAvailability.length && source === "main" && nextAvailableSources.includes("live")) {
          codecFallbackRef.current = true;
          setPlaybackNotice(isAllCameras
            ? "No Main recordings exist for this day; using Sub."
            : "No Main recording exists for this day; using Sub.");
          setSource("live");
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
    fetch(
      recordingWindowUrl(activeCameraId, requestedWindow.start, requestedWindow.end, source),
      { signal: controller.signal },
    )
      .then(async (response) => {
        if (!response.ok) throw new Error(`Recording window failed (${response.status})`);
        return response.json();
      })
      .then((payload) => {
        if (controller.signal.aborted || requestId !== playbackRequestRef.current) return;
        const rows = payload.recordings || [];
        if (!rows.length) throw new Error("No recording segments exist in this window");
        setPlaybackDetail({
          start: Number(payload.start_epoch),
          end: Number(payload.end_epoch),
          rows,
          revision: requestId,
        });
      })
      .catch((error) => {
        if (error.name !== "AbortError" && requestId === playbackRequestRef.current) {
          pendingSeekEpochRef.current = null;
          pendingSeekModeRef.current = null;
          setPlaybackNotice("");
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
      initialEpoch = recordingGridBestEpoch(timeline, initialEpoch) ?? initialEpoch;
    }
    playAt(initialEpoch, false);
  }, [date, dayEnd, dayStart, isAllCameras, playhead, timeline, today]);

  useEffect(() => {
    if (!Number.isFinite(followTarget) || !timeline.length) return;
    const nextRange = timeline.find((item) => item.end_epoch > followTarget);
    if (!nextRange) return;
    const nextEpoch = Math.max(followTarget, nextRange.start_epoch);
    playAt(nextEpoch, true);
  }, [followTarget, timeline]);

  useEffect(() => {
    if (!activeCameraId) return;
    const params = new URLSearchParams({ camera: activeCameraId, date, source });
    const retainedEpoch = desiredEpochRef.current;
    if (Number.isFinite(retainedEpoch) && retainedEpoch >= dayStart && retainedEpoch < dayEnd) {
      params.set("at", String(Math.round(retainedEpoch * 1000) / 1000));
    }
    if (eventFilter !== "all") params.set("filter", eventFilter);
    if (selectedEventId) params.set("event", String(selectedEventId));
    if (timelineInspectorTab !== "details") params.set("inspector", timelineInspectorTab);
    if (incidentRangeHours !== 1) params.set("window", String(incidentRangeHours));
    if (!timelineLanes.object) params.set("objects", "0");
    if (!timelineLanes.motion) params.set("motion", "0");
    if (playbackRate !== 1) params.set("speed", String(playbackRate));
    window.history.replaceState(null, "", appUrl(`/timeline?${params.toString()}`));
  }, [activeCameraId, date, dayEnd, dayStart, eventFilter, incidentRangeHours, playbackRate, selectedEventId, source, timelineInspectorTab, timelineLanes.motion, timelineLanes.object]);

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
      setTimelineInspectorTab(view.inspector);
      setIncidentRangeHours(view.windowHours);
      setTimelineLanes(view.lanes);
      setPlaybackRate(view.speed);
    };
    window.addEventListener("popstate", restoreView);
    return () => window.removeEventListener("popstate", restoreView);
  }, [cameraId, date, source, timeZone, today]);

  function checkpointTimelineView() {
    window.history.pushState(null, "", window.location.href);
  }

  function handleRecordingReady(_player, video) {
    video.playbackRate = normalizedTimelinePlaybackRate(playbackRate);
    if (playbackRetryRef.current.timer) window.clearTimeout(playbackRetryRef.current.timer);
    playbackRetryRef.current = { attempts: 0, timer: null };
    const retained = Number.isFinite(pendingSeekEpochRef.current)
      ? pendingSeekEpochRef.current
      : desiredEpochRef.current;
    const target = Number.isFinite(retained) ? snapToRecording(retained) : snapToRecording(Date.now() / 1000);
    const mediaTime = epochToPlaybackMediaTime(target);
    const seekRequired = Number.isFinite(mediaTime) && Math.abs(video.currentTime - mediaTime) > 0.05;
    if (seekRequired) {
      pendingSeekEpochRef.current = target;
      pendingSeekModeRef.current = "window-ready";
      setPlaybackNotice("Seeking...");
      video.currentTime = mediaTime;
    }
    if (Number.isFinite(target)) {
      desiredEpochRef.current = target;
      setPlayhead(target);
    }
    if (!seekRequired) {
      pendingSeekEpochRef.current = null;
      pendingSeekModeRef.current = null;
      setPlaybackNotice("");
    }
    setPlaybackError("");
    setPlaybackErrorStage("");
    if (autoplayRef.current) requestRecordingPlay(video);
  }

  function handleRecordingTimeUpdate(event) {
    if (Number.isFinite(pendingSeekEpochRef.current)) return;
    const epoch = mediaTimeToEpoch(event.currentTarget.currentTime);
    if (!Number.isFinite(epoch)) return;
    desiredEpochRef.current = epoch;
    setPlayhead(epoch);
  }

  function handleRecordingSeeked(event) {
    if (pendingSeekModeRef.current === "local" || pendingSeekModeRef.current === "window-ready") {
      const epoch = mediaTimeToEpoch(event.currentTarget.currentTime);
      if (Number.isFinite(epoch)) {
        desiredEpochRef.current = epoch;
        setPlayhead(epoch);
      }
      pendingSeekEpochRef.current = null;
      pendingSeekModeRef.current = null;
      setPlaybackNotice("");
    }
  }

  function handleRecordingError(error) {
    const detail = describePlaybackError(error);
    console.warn("Recording playback error", { camera: activeCameraId, source, detail, error });
    if (source === "main" && availableSources.includes("live") && !codecFallbackRef.current && isUnsupportedPlaybackError(error)) {
      codecFallbackRef.current = true;
      setPlaybackNotice(`Main stream is not supported by this browser; using Sub. (${detail})`);
      setSource("live");
      return;
    }
    if (playbackRetryRef.current.timer) return;
    if (playbackRetryRef.current.attempts < 4 && manifestUrl) {
      playbackRetryRef.current.attempts += 1;
      const attempt = playbackRetryRef.current.attempts;
      const delay = Math.min(5_000, 750 * (2 ** (attempt - 1)));
      setPlaybackError("");
      setPlaybackErrorStage("");
      setPlaybackNotice(`Playback interrupted (${detail}). Retrying ${attempt}/4...`);
      playbackRetryRef.current.timer = window.setTimeout(() => {
        playbackRetryRef.current.timer = null;
        setManifestRetryToken((token) => token + 1);
      }, delay);
      return;
    }
    setPlaybackNotice("");
    setPlaybackErrorStage("media");
    setPlaybackError(`Playback failed: ${detail}`);
  }

  function retryRecordingPlayback() {
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
    const target = Number.isFinite(desiredEpochRef.current) ? desiredEpochRef.current : Date.now() / 1000;
    requestPlaybackWindow(windowAround(target));
  }

  function continueRecordingPlayback() {
    const lastSegment = playbackTimeline[playbackTimeline.length - 1];
    if (!lastSegment) return;
    const nextEpoch = lastSegment.end_epoch + 0.01;
    const nextRange = timeline.find((item) => item.end_epoch > nextEpoch);
    if (nextRange) {
      playAt(Math.max(nextEpoch, nextRange.start_epoch), true);
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

  async function startExport() {
    if (!activeCameraId || !exportRange || exportSubmitting) return;
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
          <button type="button" className={source === "main" ? "active" : ""} aria-pressed={source === "main"} title="High" onClick={() => { checkpointTimelineView(); setSource("main"); }} disabled={availableSources.length > 0 && !availableSources.includes("main")}>Main</button>
          <button type="button" className={source === "live" ? "active" : ""} aria-pressed={source === "live"} title="Medium" onClick={() => { checkpointTimelineView(); setSource("live"); }} disabled={availableSources.length > 0 && !availableSources.includes("live")}>Sub</button>
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

  return (
    <main className={`recordings-v2-page${investigationOpen ? " has-investigation" : " investigation-hidden"}`}>
      <nav className="recordings-tabs recordings-commandbar" aria-label="Timeline controls">
        <TimelineCameraPicker
          cameras={cameras}
          value={activeCameraId}
          onChange={(nextCameraId) => { checkpointTimelineView(); setCameraId(nextCameraId); }}
        />
        <span className="recordings-commandbar-live"><i />{date === today ? "Live archive" : "Archive"}</span>
        <div className="recordings-commandbar-day-controls">{timelineDayControls()}</div>
      </nav>
      <section className="recordings-v2-workspace">
        <div className={`recordings-v2-player${isAllCameras ? " all-camera-grid" : " selected-camera-stage"}`}>
          {isAllCameras && Number.isFinite(playhead) ? <RecordingCameraGrid
            cameras={cameras}
            source={source}
            epoch={playhead}
            playing={gridPlaying}
            onSelect={(selectedCameraId) => { checkpointTimelineView(); setCameraId(selectedCameraId); }}
          /> : null}
          {!isAllCameras && manifestUrl ? (
            <ShakaVideo
              ref={videoRef}
              src={manifestUrl}
              mimeType="application/vnd.apple.mpegurl"
              startTime={manifestStartTime}
              bufferingGoal={8}
              controls
              playsInline
              preload="auto"
              onReady={handleRecordingReady}
              onError={handleRecordingError}
              onTimeUpdate={handleRecordingTimeUpdate}
              onSeeked={handleRecordingSeeked}
              onEnded={continueRecordingPlayback}
              onPlay={() => {
                autoplayRef.current = true;
                setPlaybackBlocked(false);
                if (!Number.isFinite(pendingSeekEpochRef.current)) setPlaybackNotice("");
              }}
              onPause={(event) => {
                if (!event.currentTarget.ended && !Number.isFinite(pendingSeekEpochRef.current)) {
                  autoplayRef.current = false;
                }
              }}
            />
          ) : null}
          {!isAllCameras && Number.isFinite(playhead) ? <RecordingCompanionStrip cameras={cameras} routes={cameraTransitionRoutes} activeCameraId={activeCameraId} source={source} epoch={playhead} onSelect={(camera) => { checkpointTimelineView(); setCameraId(camera); }} /> : null}
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
          {isAllCameras && Number.isFinite(playhead) ? <div className="recording-grid-controls">
            <button type="button" onClick={() => playAt(playhead - 10, gridPlaying)} aria-label="Back 10 seconds"><SkipBack size={16} /></button>
            <button type="button" className="primary" onClick={() => setGridPlaying((current) => !current)}>{gridPlaying ? <Pause size={17} /> : <Play size={17} fill="currentColor" />}{gridPlaying ? "Pause" : "Play"}</button>
            <button type="button" onClick={() => playAt(playhead + 10, gridPlaying)} aria-label="Forward 10 seconds"><SkipForward size={16} /></button>
            <time>{formatDateTime(playhead, timeZone)}</time>
          </div> : null}
        </div>

        <div className="recordings-v2-controls">
          <div className="recordings-v2-timeline-toolbar">
            <div className="recordings-v2-incidents-tools">
              <span className="recordings-v2-filter-label" title="Evidence" aria-label="Evidence"><SlidersHorizontal size={14} /></span>
              <div className="recordings-v2-event-filter" role="group" aria-label="Recording incident type">
                <button type="button" className={eventFilter === "all" ? "active" : ""} aria-pressed={eventFilter === "all"} onClick={() => { checkpointTimelineView(); setEventFilter("all"); }}><Images size={14} />All events</button>
                <button type="button" className={eventFilter === "people" ? "active" : ""} aria-pressed={eventFilter === "people"} onClick={() => { checkpointTimelineView(); setEventFilter("people"); }}><UserRound size={14} />People</button>
                <button type="button" className={eventFilter === "vehicles" ? "active" : ""} aria-pressed={eventFilter === "vehicles"} onClick={() => { checkpointTimelineView(); setEventFilter("vehicles"); }}><CarFront size={14} />Vehicles</button>
              </div>
              <div className="recordings-v2-scale" role="group" aria-label="Visible time scale">
                {[[1, "1h"], [2, "2h"], [4, "4h"], [8, "8h"], [12, "12h"], [24, "Day"]].map(([hours, label]) => (
                  <button
                    key={hours}
                    type="button"
                    className={incidentRangeHours === hours ? "active" : ""}
                    aria-pressed={incidentRangeHours === hours}
                    onClick={() => { checkpointTimelineView(); setIncidentRangeHours(hours); }}
                  >{label}</button>
                ))}
              </div>
              <div className="recordings-toolbar-day-controls">{timelineDayControls()}</div>
            </div>
            <div className="recordings-timeline-display-controls">
              <button type="button" className={timelineLanes.object ? "active object" : "object"} aria-pressed={timelineLanes.object} onClick={() => { checkpointTimelineView(); setTimelineLanes((current) => ({ ...current, object: !current.object })); }}>Objects</button>
              <button type="button" className={timelineLanes.motion ? "active motion" : "motion"} aria-pressed={timelineLanes.motion} onClick={() => { checkpointTimelineView(); setTimelineLanes((current) => ({ ...current, motion: !current.motion })); }}>Motion</button>
            </div>
            <button
              type="button"
              className={`recordings-investigation-toggle${investigationOpen ? " active" : ""}`}
              aria-pressed={investigationOpen}
              aria-expanded={investigationOpen}
              aria-controls="timeline-investigation"
              onClick={() => setInvestigationOpen((current) => !current)}
            >
              {investigationOpen ? <ChevronDown size={15} /> : <ChevronUp size={15} />}
              Nearby
            </button>
            <button type="button" className={`recordings-v2-export-toggle${exportRange ? " active" : ""}`} onClick={toggleExport} disabled={isAllCameras || !timeline.length || exportActive}>
              <Download size={15} />{isAllCameras ? "Select camera" : exportActive ? "Export running" : exportRange ? "Close export" : "Export"}
            </button>
            {!followPlayhead && Number.isFinite(playhead) ? <button type="button" className="recordings-return-playhead" onClick={returnToPlayhead}>Return to playhead</button> : null}
          </div>
          <RecordingTimeline
            cameraId={isAllCameras ? "" : activeCameraId}
            source={source}
            previewManifestUrl={manifestUrl}
            previewStartTime={manifestStartTime}
            previewTimeline={playbackTimeline}
            startEpoch={timelineView.startEpoch}
            endEpoch={timelineView.endEpoch}
            recordings={timeline}
            events={displayedTimelineEvents}
            evidenceFilter={eventFilter}
            laneVisibility={timelineLanes}
            selectedEventId={selectedEvent?.id}
            onEventSelect={(event) => { checkpointTimelineView(); setInvestigationOpen(true); setSelectedEventId(event.id); playAt(event.incident_epoch, true); }}
            playhead={playhead ?? dayStart}
            timeZone={timeZone}
            windowHours={incidentRangeHours}
            onSeek={(epoch) => playAt(epoch, true)}
            onPanViewport={panTimelineViewport}
            exportRange={isAllCameras ? null : exportRange}
            onExportRangeChange={isAllCameras || exportJob ? null : setExportRange}
          />
          {exportRange ? (
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
          ) : null}
        </div>

      </section>

      <div id="timeline-investigation" className="recordings-v2-incidents" hidden={!investigationOpen}>
        <div className="recordings-v2-investigation">
          {selectedEvent ? <aside className="recordings-v2-selected-event" aria-label="Selected incident">
            <header>Selected incident</header>
            <a className="recordings-v2-selected-event-image" href={appUrl(`/incidents?event_ids=${encodeURIComponent(selectedEvent.representative_event_id || selectedEvent.id)}`)} aria-label={`Open selected incident at ${formatTimeOnly(selectedEvent.incident_epoch, timeZone)}`}>
              <Radar size={22} />
              {selectedEvent.snapshot_path ? <img src={eventThumbnailUrl(selectedEvent, 720, 95)} alt="" loading="lazy" decoding="async" onError={(loadEvent) => { loadEvent.currentTarget.hidden = true; }} /> : null}
              {selectedEventDuration > 0 ? <time>{formatDuration(selectedEventDuration)}</time> : null}
            </a>
            <div>
              <strong>{formatTimeOnly(selectedEvent.incident_epoch, timeZone)}</strong>
              <small>{cameras.find((camera) => camera.id === selectedEvent.camera_id)?.name || selectedEvent.camera_id}</small>
              <IdentityChip item={selectedEvent} />
              <em>{selectedEvent.labels?.length ? selectedEvent.labels.join(", ") : "Motion only"}</em>
              {selectedEventConfidence > 0 ? <span>Confidence {Math.round(selectedEventConfidence * 100)}%</span> : null}
            </div>
          </aside> : <aside className="recordings-v2-selected-event empty"><span>Select an event on the timeline to investigate</span></aside>}
          <section className="recordings-related-events" aria-label="Nearby evidence">
            <header><strong>Nearby evidence</strong><span>{nearbyEvents.length} in view</span><button ref={timelineInspectorTriggerRef} type="button" className="recordings-inspector-toggle" onClick={() => setTimelineInspectorOpen(true)}>Details</button></header>
            <div className="recordings-v2-events">
              {nearbyEvents.length ? nearbyEvents.map((event) => (
                <button
                  key={event.id}
                  type="button"
                  className={`${event.has_objects ? "object" : "motion"}${selectedEvent?.id === event.id ? " selected" : ""}`}
                  onClick={() => { checkpointTimelineView(); setInvestigationOpen(true); setSelectedEventId(event.id); playAt(event.incident_epoch, true); }}
                  aria-pressed={selectedEvent?.id === event.id}
                  aria-label={`${event.labels?.length ? event.labels.join(", ") : "Motion only"} at ${formatTimeOnly(event.incident_epoch, timeZone)}`}
                  title={`${formatDateTime(event.incident_epoch, timeZone)} · ${event.labels?.length ? event.labels.join(", ") : "Motion only"}`}
                >
                  <span className="recordings-v2-event-image">
                    <Radar size={20} />
                    {event.snapshot_path ? <img src={eventThumbnailUrl(event, 240, 72)} alt="" loading="lazy" decoding="async" onError={(loadEvent) => { loadEvent.currentTarget.hidden = true; }} /> : null}
                  </span>
                  <span className="recordings-v2-event-caption">
                    <time>{formatTimeOnly(event.incident_epoch, timeZone).replace(/:\d{2}(?=\s)/, "")}</time>
                    <b>{isAllCameras ? `${cameras.find((camera) => camera.id === event.camera_id)?.name || event.camera_id} · ` : ""}{event.labels?.length ? event.labels.join(", ") : "Motion only"}</b>
                  </span>
                </button>
              )) : <div className="recordings-v2-no-events"><Radar size={17} />No {eventFilter === "all" ? "events" : `${eventFilter} incidents`} {incidentRangeHours >= 24 ? "on this day" : `within ${incidentRangeHours === 1 ? "30 minutes" : `${incidentRangeHours / 2} hours`} of this time`}</div>}
            </div>
          </section>
          <aside className={`recordings-event-inspector${timelineInspectorOpen ? " open" : ""}`} aria-label="Event inspector">
            <div className="recordings-event-inspector-tabs" role="tablist" aria-label="Incident information">
              {[["details", "Details"], ["ai", "AI"], ["related", "Nearby"]].map(([id, label], index, tabs) => <button
                key={id}
                id={`timeline-inspector-tab-${id}`}
                type="button"
                role="tab"
                aria-controls="timeline-inspector-panel"
                aria-selected={timelineInspectorTab === id}
                tabIndex={timelineInspectorTab === id ? 0 : -1}
                className={timelineInspectorTab === id ? "active" : ""}
                onClick={() => { checkpointTimelineView(); setTimelineInspectorTab(id); }}
                onKeyDown={(event) => {
                  const direction = event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
                  const targetIndex = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : direction ? (index + direction + tabs.length) % tabs.length : -1;
                  if (targetIndex < 0) return;
                  event.preventDefault();
                  checkpointTimelineView();
                  setTimelineInspectorTab(tabs[targetIndex][0]);
                  event.currentTarget.parentElement?.querySelectorAll('[role="tab"]')[targetIndex]?.focus();
                }}
              >{label}</button>)}
            </div>
            <button type="button" className="recordings-inspector-close" onClick={() => { setTimelineInspectorOpen(false); window.requestAnimationFrame(() => timelineInspectorTriggerRef.current?.focus()); }} aria-label="Close event details"><X size={15} /></button>
            <div id="timeline-inspector-panel" role="tabpanel" aria-labelledby={`timeline-inspector-tab-${timelineInspectorTab}`}>
              {selectedEvent && timelineInspectorTab === "details" ? <dl>
                <div><dt>Type</dt><dd>{selectedEvent.labels?.join(", ") || "Motion only"}</dd></div>
                <div><dt>Start</dt><dd>{formatTimeOnly(selectedEvent.incident_epoch, timeZone)}</dd></div>
                <div><dt>End</dt><dd>{Number.isFinite(selectedEventEnd) ? formatTimeOnly(selectedEventEnd, timeZone) : "—"}</dd></div>
                <div><dt>Duration</dt><dd>{selectedEventDuration > 0 ? formatDuration(selectedEventDuration) : "—"}</dd></div>
                <div><dt>Camera</dt><dd>{cameras.find((camera) => camera.id === selectedEvent.camera_id)?.name || selectedEvent.camera_id}</dd></div>
                <div><dt>Event ID</dt><dd>{selectedEvent.id}</dd></div>
              </dl> : null}
              {selectedEvent && timelineInspectorTab === "ai" ? <div className="recordings-inspector-message"><Sparkles size={18} /><strong>No AI summary generated</strong><span>Ask the SurvNG Assistant to analyze this incident using its exact event context.</span></div> : null}
              {selectedEvent && timelineInspectorTab === "related" ? <div className="recordings-inspector-message"><Images size={18} /><strong>{Math.max(0, nearbyEvents.length - 1)} nearby events</strong><span>These events are close in time; they are not asserted to be the same activity.</span></div> : null}
              {!selectedEvent ? <div className="recordings-inspector-message"><Radar size={18} /><strong>No event selected</strong><span>Choose an event from the timeline or related rail.</span></div> : null}
            </div>
          </aside>
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
      .catch((loadError) => {
        if (loadError.name !== "AbortError") setError(loadError.message || "Unable to load cameras");
      });
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
    const failure = listResult.status === "rejected"
      ? listResult.reason
      : summaryResult.status === "rejected"
        ? summaryResult.reason
        : null;
    if (failure?.name !== "AbortError") setError(failure?.message || "Unable to load exports");
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

export function RecordingTimeline({ cameraId, source, previewManifestUrl, previewStartTime, previewTimeline, startEpoch, endEpoch, recordings, events, evidenceFilter, laneVisibility, selectedEventId, onEventSelect, playhead, timeZone, onSeek, onPanViewport, windowHours = 1, exportRange, onExportRangeChange }) {
  const duration = Math.max(1, endEpoch - startEpoch);
  const offset = Math.max(0, Math.min(duration, playhead - startEpoch));
  const trackRef = useRef(null);
  const [draft, setDraft] = useState(offset);
  const [scrubbing, setScrubbing] = useState(false);
  const [preview, setPreview] = useState({ epoch: null, url: "", loading: false, gap: false, unavailable: false, mode: "jpeg" });
  const [localPreviewEnabled, setLocalPreviewEnabled] = useState(false);
  const [localPreviewReady, setLocalPreviewReady] = useState(false);
  const [localPreviewFrameReady, setLocalPreviewFrameReady] = useState(false);
  const draftRef = useRef(offset);
  const dragRef = useRef(null);
  const previewTimerRef = useRef(null);
  const previewAbortRef = useRef(null);
  const previewLastRequestRef = useRef({ epoch: null, at: 0 });
  const previewUrlRef = useRef("");
  const previewInFlightRef = useRef(false);
  const previewPendingRef = useRef(null);
  const previewHideTimerRef = useRef(null);
  const previewGenerationRef = useRef(0);
  const localPreviewRef = useRef(null);
  const localPreviewCanvasRef = useRef(null);
  const localPreviewTargetRef = useRef(null);
  const exportDragRef = useRef(null);
  const ticks = useMemo(() => {
    const tickInterval = timelineTickIntervalSeconds(windowHours);
    const formatter = new Intl.DateTimeFormat("en-US", {
      timeZone,
      hour: "numeric",
      minute: "2-digit",
    });
    const partFormatter = new Intl.DateTimeFormat("en-US", {
      timeZone,
      hourCycle: "h23",
      hour: "2-digit",
      minute: "2-digit",
    });
    const items = [];
    const first = Math.ceil(startEpoch / tickInterval) * tickInterval;
    for (let epoch = first; epoch < endEpoch; epoch += tickInterval) {
      const parts = partFormatter.formatToParts(new Date(epoch * 1000));
      const minute = Number(parts.find((item) => item.type === "minute")?.value || 0);
      const kind = minute === 0 ? "hour" : minute === 30 ? "half" : "quarter";
      items.push({
        epoch,
        kind,
        label: kind === "hour" ? formatter.format(new Date(epoch * 1000)).replace(":00", "") : "",
      });
    }
    return items;
  }, [endEpoch, startEpoch, timeZone, windowHours]);
  useEffect(() => {
    const track = trackRef.current;
    if (!track || !onPanViewport) return undefined;
    function onWheel(event) {
      const horizontal = Math.abs(event.deltaX) >= Math.abs(event.deltaY);
      event.preventDefault();
      if (!horizontal && !event.shiftKey) return;
      const deltaPx = horizontal ? event.deltaX : event.deltaY;
      const width = track.clientWidth || 1;
      onPanViewport((deltaPx / width) * duration);
    }
    track.addEventListener("wheel", onWheel, { passive: false });
    return () => track.removeEventListener("wheel", onWheel);
  }, [duration, onPanViewport]);
  const eventMarkers = useMemo(() => (events || []).map((event) => {
    const start = recordingIncidentEpoch(event);
    const end = recordingIncidentEndEpoch(event);
    if (!Number.isFinite(start) || start >= endEpoch || (Number.isFinite(end) && end < startEpoch)) return null;
    const boundedStart = Math.max(startEpoch, start);
    const boundedEnd = Math.min(endEpoch, Math.max(boundedStart + 1, Number.isFinite(end) ? end : start + 1));
    return {
      id: event.id,
      hasObjects: Boolean(event.has_objects),
      left: ((boundedStart - startEpoch) / duration) * 100,
      width: ((boundedEnd - boundedStart) / duration) * 100,
      event: { ...event, incident_epoch: start },
    };
  }).filter(Boolean), [duration, endEpoch, events, startEpoch]);
  useEffect(() => {
    if (dragRef.current) return;
    draftRef.current = offset;
    setDraft(offset);
  }, [offset]);
  useEffect(() => () => {
    previewGenerationRef.current += 1;
    if (previewTimerRef.current) window.clearTimeout(previewTimerRef.current);
    if (previewHideTimerRef.current) window.clearTimeout(previewHideTimerRef.current);
    previewAbortRef.current?.abort();
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
  }, []);
  useEffect(() => {
    previewGenerationRef.current += 1;
    previewLastRequestRef.current = { epoch: null, at: 0 };
    previewPendingRef.current = null;
    previewInFlightRef.current = false;
    if (previewTimerRef.current) window.clearTimeout(previewTimerRef.current);
    if (previewHideTimerRef.current) window.clearTimeout(previewHideTimerRef.current);
    previewAbortRef.current?.abort();
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = "";
    setScrubbing(false);
    setLocalPreviewEnabled(false);
    setLocalPreviewReady(false);
    setLocalPreviewFrameReady(false);
    localPreviewTargetRef.current = null;
    setPreview({ epoch: null, url: "", loading: false, gap: false, unavailable: false, mode: "jpeg" });
  }, [cameraId, source, startEpoch]);
  useEffect(() => {
    setLocalPreviewReady(false);
    setLocalPreviewFrameReady(false);
    localPreviewTargetRef.current = null;
  }, [previewManifestUrl]);
  const percent = ((scrubbing ? draft : offset) / duration) * 100;

  function hidePreviewAfterDelay() {
    if (previewHideTimerRef.current) window.clearTimeout(previewHideTimerRef.current);
    previewHideTimerRef.current = window.setTimeout(() => {
      previewHideTimerRef.current = null;
      if (!dragRef.current) setScrubbing(false);
    }, 1400);
  }

  async function requestPreview(request) {
    if (previewInFlightRef.current) {
      previewPendingRef.current = request;
      return;
    }
    const generation = previewGenerationRef.current;
    previewInFlightRef.current = true;
    previewLastRequestRef.current = { epoch: request.bucket, at: performance.now() };
    const controller = new AbortController();
    previewAbortRef.current = controller;
    setPreview((current) => ({
      ...current,
      epoch: request.epoch,
      loading: true,
      gap: false,
      unavailable: false,
      mode: "jpeg",
    }));
    try {
      const response = await fetch(recordingPreviewUrl(cameraId, request.epoch, source), {
        signal: controller.signal,
      });
      if (controller.signal.aborted || generation !== previewGenerationRef.current) return;
      if (response.status === 404) {
        setPreview((current) => ({
          ...current,
          epoch: request.epoch,
          loading: false,
          gap: true,
          unavailable: false,
          mode: "jpeg",
        }));
      } else {
        if (!response.ok) throw new Error(`Preview failed (${response.status})`);
        const objectUrl = URL.createObjectURL(await response.blob());
        if (controller.signal.aborted || generation !== previewGenerationRef.current) {
          URL.revokeObjectURL(objectUrl);
          return;
        }
        if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
        previewUrlRef.current = objectUrl;
        setPreview({
          epoch: request.epoch,
          url: objectUrl,
          loading: false,
          gap: false,
          unavailable: false,
          mode: "jpeg",
        });
      }
    } catch (error) {
      if (error.name !== "AbortError" && generation === previewGenerationRef.current) {
        setPreview((current) => ({
          ...current,
          epoch: request.epoch,
          loading: false,
          gap: false,
          unavailable: true,
          mode: "jpeg",
        }));
      }
    } finally {
      if (generation !== previewGenerationRef.current) return;
      previewInFlightRef.current = false;
      const pending = previewPendingRef.current;
      previewPendingRef.current = null;
      if (pending && pending.bucket !== request.bucket) {
        if (!requestLocalPreview(pending)) requestPreview(pending);
      } else if (!dragRef.current) {
        setScrubbing(true);
        hidePreviewAfterDelay();
      }
    }
  }

  function requestLocalPreview(request) {
    const video = localPreviewRef.current;
    const mediaTime = playbackMediaTimeForEpoch(previewTimeline, request.epoch);
    if (!localPreviewReady || previewInFlightRef.current || !video || !Number.isFinite(mediaTime)) {
      return false;
    }
    previewLastRequestRef.current = { epoch: request.bucket, at: performance.now() };
    previewPendingRef.current = null;
    const target = { epoch: request.epoch, mediaTime };
    localPreviewTargetRef.current = target;
    setPreview((current) => ({
      ...current,
      loading: true,
      gap: false,
      unavailable: false,
      mode: localPreviewFrameReady ? "local" : current.mode,
    }));
    video.pause();
    if (Math.abs(video.currentTime - mediaTime) <= 0.04 && video.readyState >= 2) {
      publishLocalPreviewFrame(video, target);
    } else {
      video.currentTime = mediaTime;
    }
    return true;
  }

  function handleLocalPreviewReady(_player, video) {
    video.pause();
    setLocalPreviewReady(true);
    const target = localPreviewTargetRef.current;
    if (target && Number.isFinite(target.mediaTime)) video.currentTime = target.mediaTime;
  }

  function handleLocalPreviewSeeked(event) {
    const target = localPreviewTargetRef.current;
    if (!target || Math.abs(event.currentTarget.currentTime - target.mediaTime) > 0.08) return;
    publishLocalPreviewFrame(event.currentTarget, target);
  }

  function publishLocalPreviewFrame(video, target) {
    const publish = () => {
      if (localPreviewTargetRef.current !== target || video.readyState < 2) return;
      const canvas = localPreviewCanvasRef.current;
      const sourceWidth = Number(video.videoWidth) || 0;
      const sourceHeight = Number(video.videoHeight) || 0;
      if (!canvas || !sourceWidth || !sourceHeight) return;
      const width = Math.min(480, sourceWidth);
      const height = Math.max(1, Math.round(sourceHeight * (width / sourceWidth)));
      if (canvas.width !== width) canvas.width = width;
      if (canvas.height !== height) canvas.height = height;
      canvas.getContext("2d", { alpha: false })?.drawImage(video, 0, 0, width, height);
      setLocalPreviewFrameReady(true);
      setPreview((current) => ({
        ...current,
        epoch: target.epoch,
        loading: false,
        gap: false,
        unavailable: false,
        mode: "local",
      }));
      if (!dragRef.current) {
        setScrubbing(true);
        hidePreviewAfterDelay();
      }
    };
    if (typeof video.requestVideoFrameCallback === "function") {
      let published = false;
      const fallback = window.setTimeout(() => {
        if (!published) publish();
      }, 80);
      video.requestVideoFrameCallback(() => {
        published = true;
        window.clearTimeout(fallback);
        publish();
      });
    } else {
      window.requestAnimationFrame(publish);
    }
  }

  function handleLocalPreviewError() {
    setLocalPreviewReady(false);
    setLocalPreviewFrameReady(false);
    localPreviewTargetRef.current = null;
    previewLastRequestRef.current = { epoch: null, at: 0 };
    if (dragRef.current) schedulePreview(draftRef.current, true);
  }

  function schedulePreview(value, immediate = false) {
    if (!cameraId) return;
    const requestedEpoch = startEpoch + Math.max(0, Math.min(duration, Number(value) || 0));
    const localMediaTime = localPreviewReady && !previewInFlightRef.current
      ? playbackMediaTimeForEpoch(previewTimeline, requestedEpoch)
      : null;
    const previewBucket = Number.isFinite(localMediaTime)
      ? `local:${Math.floor(requestedEpoch * 2)}`
      : `jpeg:${Math.floor((requestedEpoch - startEpoch) / 5)}`;
    if (previewLastRequestRef.current.epoch === previewBucket) {
      previewPendingRef.current = null;
      return;
    }
    if (previewTimerRef.current) window.clearTimeout(previewTimerRef.current);
    const elapsed = performance.now() - previewLastRequestRef.current.at;
    const delay = immediate ? 0 : Math.max(0, 250 - elapsed);
    previewTimerRef.current = window.setTimeout(() => {
      previewTimerRef.current = null;
      const request = { epoch: requestedEpoch, bucket: previewBucket };
      if (!requestLocalPreview(request)) requestPreview(request);
    }, delay);
  }

  function updateDraft(value, requestPreview = false) {
    const next = Math.max(0, Math.min(duration, Number(value) || 0));
    draftRef.current = next;
    setDraft(next);
    if (requestPreview) schedulePreview(next);
    return next;
  }

  function commit(value) {
    const next = updateDraft(value);
    onSeek(startEpoch + next);
  }

  function pointerValue(event, drag) {
    const pointerX = Math.max(0, Math.min(drag.width, event.clientX - drag.left));
    if (drag.precise) {
      return drag.startValue + (pointerX - drag.startX);
    }
    return (pointerX / drag.width) * duration;
  }

  function startDrag(event) {
    const rect = event.currentTarget.getBoundingClientRect();
    if (!rect.width) return;
    const pointerX = Math.max(0, Math.min(rect.width, event.clientX - rect.left));
    const playheadX = (draftRef.current / duration) * rect.width;
    const grabRadius = event.pointerType === "touch" ? 24 : 14;
    const drag = {
      pointerId: event.pointerId,
      left: rect.left,
      width: rect.width,
      startX: pointerX,
      startValue: draftRef.current,
      lastX: pointerX,
      precise: Math.abs(pointerX - playheadX) <= grabRadius,
      panning: false,
    };
    dragRef.current = drag;
    if (previewHideTimerRef.current) window.clearTimeout(previewHideTimerRef.current);
    if (previewManifestUrl) setLocalPreviewEnabled(true);
    if (drag.precise) setScrubbing(true);
    event.currentTarget.setPointerCapture(event.pointerId);
    event.preventDefault();
    if (drag.precise) schedulePreview(draftRef.current, true);
  }

  function moveDrag(event) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    event.preventDefault();
    const dx = event.clientX - drag.lastX;
    if (!drag.precise) {
      if (!drag.panning && Math.abs(event.clientX - drag.startX) > 10) {
        drag.panning = true;
        setScrubbing(false);
      }
      if (drag.panning && onPanViewport) {
        onPanViewport(-(dx / drag.width) * duration);
        drag.lastX = event.clientX;
      }
      return;
    }
    updateDraft(pointerValue(event, drag), true);
  }

  function finishDrag(event, cancelled = false) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    event.preventDefault();
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    dragRef.current = null;
    if (previewTimerRef.current) window.clearTimeout(previewTimerRef.current);
    previewTimerRef.current = null;
    if (drag.panning) {
      hidePreviewAfterDelay();
      return;
    }
    if (cancelled) {
      updateDraft(offset);
      hidePreviewAfterDelay();
      return;
    }
    if (!drag.precise) {
      commit(pointerValue(event, drag));
      hidePreviewAfterDelay();
      return;
    }
    schedulePreview(pointerValue(event, drag), true);
    hidePreviewAfterDelay();
    commit(pointerValue(event, drag));
  }

  function exportEpochAtPointer(event, drag) {
    const pointerX = Math.max(0, Math.min(drag.width, event.clientX - drag.left));
    return startEpoch + (pointerX / drag.width) * duration;
  }

  function updateExportHandle(kind, epoch) {
    if (!exportRange) return;
    const minimumGap = 1;
    const next = kind === "start"
      ? { ...exportRange, start: Math.max(startEpoch, Math.min(exportRange.end - minimumGap, epoch)) }
      : { ...exportRange, end: Math.min(endEpoch, Math.max(exportRange.start + minimumGap, epoch)) };
    onExportRangeChange?.(next);
  }

  function handleExportKey(kind, event) {
    if (!exportRange || !onExportRangeChange) return;
    const next = adjustRecordingExportRange({ range: exportRange, kind, key: event.key, shiftKey: event.shiftKey, startEpoch, endEpoch });
    if (!next) return;
    event.preventDefault();
    onExportRangeChange(next);
  }

  function startExportDrag(kind, event) {
    const track = event.currentTarget.parentElement;
    const rect = track?.getBoundingClientRect();
    if (!rect?.width) return;
    exportDragRef.current = { kind, pointerId: event.pointerId, left: rect.left, width: rect.width };
    event.currentTarget.setPointerCapture(event.pointerId);
    event.preventDefault();
  }

  function moveExportDrag(event) {
    const drag = exportDragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    updateExportHandle(drag.kind, exportEpochAtPointer(event, drag));
    event.preventDefault();
  }

  function finishExportDrag(event) {
    const drag = exportDragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    updateExportHandle(drag.kind, exportEpochAtPointer(event, drag));
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    exportDragRef.current = null;
    event.preventDefault();
  }

  const exportStartPercent = exportRange ? ((exportRange.start - startEpoch) / duration) * 100 : 0;
  const exportEndPercent = exportRange ? ((exportRange.end - startEpoch) / duration) * 100 : 0;

  return (
    <div className="recordings-v2-timeline">
      <div ref={trackRef} className="recordings-v2-track">
        <div className="recordings-v2-ticks" aria-hidden="true">
          {ticks.map((tick) => (
            <span className={tick.kind} key={tick.epoch} style={{ left: `${((tick.epoch - startEpoch) / duration) * 100}%` }}>
              {tick.label ? <small>{tick.label}</small> : null}
            </span>
          ))}
        </div>
        {recordings.map((item) => (
          <span
            key={`${item.start_epoch}:${item.end_epoch}`}
            style={{
              left: `${((Math.max(startEpoch, item.start_epoch) - startEpoch) / duration) * 100}%`,
              width: `${((Math.min(endEpoch, item.end_epoch) - Math.max(startEpoch, item.start_epoch)) / duration) * 100}%`,
            }}
          />
        ))}
        <div className={`recordings-v2-event-lane object${!laneVisibility?.object ? " hidden" : evidenceFilter === "motion" ? " muted" : ""}`} aria-hidden="true">
          {eventMarkers.filter((event) => event.hasObjects).map((event) => <b key={event.id} style={{ left: `${event.left}%`, width: `${event.width}%` }} />)}
        </div>
        <div className={`recordings-v2-event-lane motion${!laneVisibility?.motion ? " hidden" : evidenceFilter === "object" ? " muted" : ""}`} aria-hidden="true">
          {eventMarkers.filter((event) => !event.hasObjects).map((event) => <b key={event.id} style={{ left: `${event.left}%`, width: `${event.width}%` }} />)}
        </div>
        {exportRange ? (
          <>
            <div
              className="recordings-v2-export-selection"
              style={{ left: `${exportStartPercent}%`, width: `${Math.max(0, exportEndPercent - exportStartPercent)}%` }}
              aria-hidden="true"
            />
            <button
              type="button"
              className="recordings-v2-export-handle start"
              style={{ left: `${exportStartPercent}%` }}
              onPointerDown={(event) => startExportDrag("start", event)}
              onPointerMove={moveExportDrag}
              onPointerUp={finishExportDrag}
              onPointerCancel={finishExportDrag}
              onKeyDown={(event) => handleExportKey("start", event)}
              disabled={!onExportRangeChange}
              aria-label={`Export start ${formatTimeOnly(exportRange.start, timeZone)}`}
            ><span>{formatExportHandleTime(exportRange.start, timeZone)}</span></button>
            <button
              type="button"
              className="recordings-v2-export-handle end"
              style={{ left: `${exportEndPercent}%` }}
              onPointerDown={(event) => startExportDrag("end", event)}
              onPointerMove={moveExportDrag}
              onPointerUp={finishExportDrag}
              onPointerCancel={finishExportDrag}
              onKeyDown={(event) => handleExportKey("end", event)}
              disabled={!onExportRangeChange}
              aria-label={`Export end ${formatTimeOnly(exportRange.end, timeZone)}`}
            ><span>{formatExportHandleTime(exportRange.end, timeZone)}</span></button>
          </>
        ) : null}
        {localPreviewEnabled && previewManifestUrl ? (
          <ShakaVideo
            ref={localPreviewRef}
            src={previewManifestUrl}
            mimeType="application/vnd.apple.mpegurl"
            startTime={previewStartTime}
            bufferingGoal={2}
            muted
            playsInline
            preload="metadata"
            className="recordings-v2-local-preview-decoder"
            tabIndex={-1}
            aria-label="Local recording scrub preview"
            onReady={handleLocalPreviewReady}
            onError={handleLocalPreviewError}
            onSeeked={handleLocalPreviewSeeked}
          />
        ) : null}
        <div
          className={`recordings-v2-scrub-preview${scrubbing ? " active" : ""}${preview.loading ? " loading" : ""}${preview.mode === "local" ? " local" : ""}`}
          style={{ left: `${Math.max(8, Math.min(92, percent))}%` }}
          role="status"
          aria-live="polite"
          aria-hidden={!scrubbing}
        >
          <div>
            <canvas ref={localPreviewCanvasRef} aria-label="Decoded recording preview frame" />
            {preview.mode === "jpeg" && preview.url && !preview.gap && !preview.unavailable ? <img src={mediaUrl(preview.url)} alt="Recording preview" /> : null}
            {preview.mode === "jpeg" && preview.gap ? <span><Film size={20} />No recording</span> : null}
            {preview.mode === "jpeg" && preview.unavailable ? <span><RefreshCcw size={18} />Preview unavailable</span> : null}
            {preview.mode === "jpeg" && !preview.url && !preview.gap && !preview.unavailable ? <span><RefreshCcw size={18} />Loading preview</span> : null}
          </div>
          <time>{formatTimeOnly(Number.isFinite(preview.epoch) ? preview.epoch : startEpoch + draft, timeZone)}</time>
        </div>
        <i style={{ left: `${percent}%` }} />
        <output style={{ left: `${Math.max(4, Math.min(96, percent))}%` }}>{formatTimeOnly(startEpoch + (scrubbing ? draft : offset), timeZone)}</output>
        <input
          type="range"
          min="0"
          max={duration}
          step="0.1"
          value={scrubbing ? draft : offset}
          onChange={(event) => {
            if (!dragRef.current) {
              updateDraft(event.target.value);
              commit(event.target.value);
            }
          }}
          onPointerDown={startDrag}
          onPointerMove={moveDrag}
          onPointerUp={(event) => finishDrag(event)}
          onPointerCancel={(event) => finishDrag(event, true)}
          onKeyUp={(event) => commit(event.currentTarget.value)}
          aria-label="Recording timeline"
        />
      </div>
    </div>
  );
}
