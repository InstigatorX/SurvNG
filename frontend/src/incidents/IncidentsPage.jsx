import React, { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ChevronLeft,
  ChevronRight,
  Grid2X2,
  LayoutGrid,
  PanelLeft,
  Radar,
  ScanSearch,
  Search,
  SlidersHorizontal,
  Rows3,
  X,
} from "lucide-react";
import { loadRelatedIncident } from "../relatedIncidents.mjs";
import { useVisiblePolling } from "../visibilityPolling.mjs";
import { incidentReplayTracking, incidentTrackingSource, storedObjectTracks } from "../objectTrackReplay.mjs";
import { incidentDetailQuery, incidentFocusStep, incidentReturnPath, incidentSelectionHref, incidentThumbnailPageSize, incidentGalleryColumns, incidentGalleryWanted, INCIDENT_GALLERY_CHUNK, incidentFilterValues, keepIncidentIdentity, mergeIncidentHead, toggleIncidentFilterValue, linkedIncidentEventFilter } from "../incidentNavigation.mjs";
import { mapWithConcurrency, rankSemanticIncidentDetails, semanticIncidentRequest } from "../incidentSemanticSearch.mjs";
import { APP_BASE_PATH, appUrl, incidentRecordingContext, fetch } from "../shared/api.js";
import { eventThumbnailUrl, thumbnailWidthFor } from "../shared/mediaUrls.js";
import { INCIDENT_REFRESH_FALLBACK_MS } from "../shared/constants.js";
import { dateKeyForTimeZone, addDaysToDateKey, zonedDateSecondToEpoch } from "../shared/datetime.js";
import { useStoredState, isMobileViewport } from "../shared/hooks.js";
import { clearLegacyIncidentFilterStorage } from "../shared/cameras.js";
import { useAppEvents } from "../shared/events.js";
import { usePollingData, useIncidentDetails } from "../shared/polling.js";
import { IncidentListItem, IncidentObjectIcon, EventOverlay } from "../shared/evidence.jsx";
import { IncidentCard, IncidentInspector } from "./IncidentCard.jsx";
import { IncidentAnalysisStatus } from "./IncidentAnalysisStatus.jsx";
import { useIncidentPlayback } from "./useIncidentPlayback.js";
import "./mobile-incidents.css";

const REVIEWED_VERDICTS = new Set(["reviewed", "nothing_here"]);
// A long gallery re-renders often, and most tiles have not changed.
const IncidentTile = memo(IncidentListItem);
const FILTER_PANE_WIDTH = 264;

// One multi-select filter: any checked value matches, and nothing checked means all.
function IncidentFilterChecks({ title, options, filter, onChange, columns = false, empty }) {
  const chosen = incidentFilterValues(filter);
  // A choice made on another day stays listed so it can be cleared.
  const known = new Set(options.map((option) => String(option.value)));
  const listed = [...options, ...chosen.filter((value) => !known.has(value)).map((value) => ({ value, label: value }))];
  return (
    <details className="incident-filter-section incident-filter-group" open>
      <summary><h3>{title}</h3><span className="incident-filter-count">{chosen.length ? `${chosen.length} selected` : "All"}</span></summary>
      {listed.length ? (
        <div className={`incident-filter-checks${columns ? " columns" : ""}`} role="group" aria-label={title}>
          {listed.map((option) => (
            <label className="incident-filter-check" key={option.value}>
              <input type="checkbox" value={option.value} checked={chosen.includes(String(option.value))} onChange={() => onChange(toggleIncidentFilterValue(filter, option.value))} />
              {option.icon || null}
              <span>{option.label}</span>
            </label>
          ))}
        </div>
      ) : <small>{empty}</small>}
    </details>
  );
}

export function IncidentsPage({ timeZone, canCorrectIncident = false, onRecordingContextChange, onAssistantContextChange, onAskAssistant = null }) {
  const { cameras, appConfig, refresh: refreshBase } = usePollingData();
  const thumbnailAnnotations = appConfig?.incident_thumbnail_annotations ?? false;
  const thumbnailObjectFocus = appConfig?.incident_thumbnail_object_focus ?? "off";
  const thumbnailObjectFocusZoom = appConfig?.incident_thumbnail_object_focus_zoom ?? 1;
  const [mobileFiltersOpen, setMobileFiltersOpen] = useState(false);
  const [eventFilter, setEventFilter] = useState("object");
  const [incidentCameraFilter, setIncidentCameraFilter] = useState("all");
  const [incidentObjectFilter, setIncidentObjectFilter] = useState("all");
  const [incidentZoneFilter, setIncidentZoneFilter] = useState("all");
  const [incidentPersonFilter, setIncidentPersonFilter] = useState("all");
  // Percent. The slider moves the draft; the search follows once it settles.
  const [incidentMinConfidence, setIncidentMinConfidence] = useState(0);
  const [confidenceDraft, setConfidenceDraft] = useState(0);
  const [incidentPeople, setIncidentPeople] = useState([]);
  const [incidentDensity, setIncidentDensity] = useStoredState("survng.incidentDensity.v1", "compact");
  const today = dateKeyForTimeZone(Date.now(), timeZone);
  const [incidentDay, setIncidentDay] = useState(today);
  const [incidents, setIncidents] = useState([]);
  const [incidentTotal, setIncidentTotal] = useState(0);
  const [incidentFacets, setIncidentFacets] = useState({ camera_ids: [], labels: [], zones: [] });
  const [incidentLoading, setIncidentLoading] = useState(true);
  const [incidentLoadError, setIncidentLoadError] = useState("");
  const [incidentRefreshToken, setIncidentRefreshToken] = useState(0);
  const [semanticIncidentQuery, setSemanticIncidentQuery] = useState("");
  const [semanticIncidentActiveQuery, setSemanticIncidentActiveQuery] = useState("");
  const [semanticIncidentResults, setSemanticIncidentResults] = useState([]);
  const [semanticIncidentLoading, setSemanticIncidentLoading] = useState(false);
  const [semanticIncidentError, setSemanticIncidentError] = useState("");
  const semanticIncidentRequestRef = useRef(null);
  const incidentLoadedQueryRef = useRef("");
  const incidentEventRefreshTimer = useRef(null);
  const incidentEvidenceRefreshSequence = useRef(0);
  const {
    incidentDetailCacheRef,
    incidentDetails,
    setIncidentDetails,
    selectedEvent,
    setSelectedEvent,
    incidentPreviewLoading,
    openIncidentOverlay,
    closeIncidentOverlay,
  } = useIncidentDetails();
  const [focusedFaceEventId, setFocusedFaceEventId] = useState(null);
  const [linkedIncidentDetail, setLinkedIncidentDetail] = useState(null);
  const [linkedIncidentEventId, setLinkedIncidentEventId] = useState(null);
  const [linkedIncidentError, setLinkedIncidentError] = useState("");
  const [linkedIncidentLoading, setLinkedIncidentLoading] = useState(() => (
    Boolean(typeof window !== "undefined" && (new URLSearchParams(window.location.search).get("event_ids") || new URLSearchParams(window.location.search).get("incident_id")))
  ));
  const linkedIncidentBootRef = useRef(
    typeof window !== "undefined" ? (new URLSearchParams(window.location.search).get("event_ids") || new URLSearchParams(window.location.search).get("incident_id")) : null,
  );
  const [expandedIncidentId, setExpandedIncidentId] = useState(null);
  const [incidentPage, setIncidentPage] = useState(0);
  const incidentRailListRef = useRef(null);
  const incidentPageEdgeRef = useRef(null);
  const incidentGalleryToggleRef = useRef(null);
  // The gallery is the page's home. A link to one incident opens its viewer.
  const [galleryExpanded, setGalleryExpanded] = useState(() => !linkedIncidentBootRef.current);
  const [storedGalleryColumns, setStoredGalleryColumns] = useStoredState("survng.incidentGalleryColumns.v1", 4);
  const galleryColumns = incidentGalleryColumns(storedGalleryColumns);
  // How many incidents the gallery has asked for. Scrolling near the end asks for more.
  const [galleryWanted, setGalleryWanted] = useState(INCIDENT_GALLERY_CHUNK);
  const [galleryMoreError, setGalleryMoreError] = useState("");
  const [filtersPinned, setFiltersPinned] = useState(false);
  const galleryLoadedRef = useRef({ key: "", token: -1, items: [] });
  const galleryGrowthRef = useRef({ count: 0, total: 0, wanted: INCIDENT_GALLERY_CHUNK });
  // 0: nothing to reveal. Infinity: waiting for the gallery to hold the open
  // incident. Otherwise the time until which a layout change repeats the scroll.
  const galleryRevealRef = useRef(0);
  // The next chunk, fetched before it is needed: { key, token, offset, limit, promise }.
  const galleryAheadRef = useRef(null);
  const warmedImagesRef = useRef([]);
  // Whether the next refresh must re-read everything shown, or only the newest incidents.
  const galleryFullRefreshRef = useRef(true);
  const refreshPollCountRef = useRef(0);
  const [viewportWidth, setViewportWidth] = useState(() => (typeof window !== "undefined" ? window.innerWidth : 1440));
  const [incidentPageSize, setIncidentPageSize] = useState(12);
  const retainedGallerySelectionRef = useRef(null);
  const galleryPageAnchorRef = useRef(null);
  const incidentPagingRef = useRef(null);
  const [desktopAnalysisMode, setDesktopAnalysisMode] = useStoredState("survng.incidentDesktopAnalysis.v1", "clean");
  const [desktopDepthLayer, setDesktopDepthLayer] = useStoredState("survng.incidentDesktopDepthLayer.v1", "both");
  const [desktopAnalysisStats, setDesktopAnalysisStats] = useState(null);
  const [desktopReplayRequest, setDesktopReplayRequest] = useState(0);
  const [focusedImageSize, setFocusedImageSize] = useState(null);
  const [relatedPreviewError, setRelatedPreviewError] = useState("");
  const [relatedPreviewIncident, setRelatedPreviewIncident] = useState(null);
  const [relatedPreviewEventId, setRelatedPreviewEventId] = useState(null);
  const [relatedPreviewLoadingEventId, setRelatedPreviewLoadingEventId] = useState(null);
  const [selectedVisualObject, setSelectedVisualObject] = useState(null);
  const [observationPreviewRequest, setObservationPreviewRequest] = useState(null);
  const [findSimilarObject, setFindSimilarObject] = useState(null);
  const [tabletInspectorOpen, setTabletInspectorOpen] = useState(false);
  const [focusedDetailError, setFocusedDetailError] = useState("");
  const [focusedDetailLoading, setFocusedDetailLoading] = useState(false);
  const [focusedDetailRetry, setFocusedDetailRetry] = useState(0);
  const tabletInspectorToggleRef = useRef(null);
  const relatedPreviewRequestRef = useRef(0);
  const mobileView = isMobileViewport();
  // The phone layout is always the gallery; a tile there opens the incident's page.
  const galleryMode = mobileView || galleryExpanded;
  const galleryModeRef = useRef(galleryMode);
  galleryModeRef.current = galleryMode;
  const incidentsPerPage = galleryMode ? galleryWanted : incidentPageSize;
  const cameraNameById = useMemo(() => new Map(cameras.map((camera) => [camera.id, camera.name || camera.id])), [cameras]);
  const incidentCameraOptions = incidentFacets.camera_ids || [];
  const incidentObjectOptions = incidentFacets.labels || [];
  const incidentZoneOptions = incidentFacets.zones || [];
  const semanticIncidentActive = Boolean(semanticIncidentActiveQuery);
  const confidenceApplies = eventFilter !== "motion";
  const activeIncidentFilterCount = [incidentCameraFilter, incidentObjectFilter, incidentZoneFilter, incidentPersonFilter].filter((value) => value !== "all").length
    + (confidenceApplies && incidentMinConfidence > 0 ? 1 : 0);
  const incidentResultSource = semanticIncidentActive ? semanticIncidentResults : incidents;
  const displayedIncidentTotal = semanticIncidentActive ? semanticIncidentResults.length : incidentTotal;
  const displayedIncidentLoading = semanticIncidentActive ? semanticIncidentLoading : incidentLoading;
  const displayedIncidentError = semanticIncidentActive ? semanticIncidentError : incidentLoadError;

  function closeTabletInspector({ restoreFocus = true } = {}) {
    setTabletInspectorOpen(false);
    if (restoreFocus) window.requestAnimationFrame(() => tabletInspectorToggleRef.current?.focus());
  }

  function closeOpenedIncident() {
    const returnTo = incidentReturnPath(window.location.search);
    closeIncidentOverlay();
    if (returnTo) window.location.assign(appUrl(returnTo));
  }

  useEffect(() => {
    if (!tabletInspectorOpen) return undefined;
    function closeOnEscape(event) {
      if (event.key !== "Escape") return;
      event.preventDefault();
      closeTabletInspector();
    }
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [tabletInspectorOpen]);
  const visibleIncidents = semanticIncidentActive
    ? incidentResultSource.slice(incidentPage * incidentsPerPage, (incidentPage + 1) * incidentsPerPage)
    : incidentResultSource;
  const sameIncidentId = (left, right) => left != null && right != null && String(left) === String(right);
  const explicitlyFocusedSummary = visibleIncidents.find((incident) => sameIncidentId(incident.id, expandedIncidentId))
    || (sameIncidentId(retainedGallerySelectionRef.current?.id, expandedIncidentId) ? retainedGallerySelectionRef.current : null)
    || (sameIncidentId(linkedIncidentDetail?.id, expandedIncidentId) ? linkedIncidentDetail : null);
  // While a deep-linked event_ids bootstrap is in flight, avoid auto-focusing the first
  // gallery row (that clobbers the deep link URL and can leave a blank investigation).
  const focusedSummary = mobileView
    ? explicitlyFocusedSummary
    : (explicitlyFocusedSummary || (linkedIncidentBootRef.current || galleryMode ? null : visibleIncidents[0]) || null);
  const focusedDetailQuery = incidentDetailQuery(focusedSummary);
  const focusedIncident = focusedSummary ? incidentDetails[focusedDetailQuery] || focusedSummary : null;
  const focusedEvent = (focusedIncident?.events || []).find((event) => Number(event.id) === Number(focusedFaceEventId))
    || (focusedIncident?.events || []).find((event) => Number(event.id) === Number(focusedIncident.representative_event_id))
    || focusedIncident;
  const relatedAnchorEvent = incidentTrackingSource(focusedEvent, focusedIncident) || focusedEvent;
  const relatedAnchorEventId = Number(relatedAnchorEvent?.representative_event_id || relatedAnchorEvent?.id) || null;
  const displayedIncident = relatedPreviewIncident || focusedIncident;
  const scenePlayback = useIncidentPlayback(displayedIncident);
  const displayedEvent = relatedPreviewIncident
    ? (
      (relatedPreviewIncident.events || []).find((event) => Number(event.id) === Number(relatedPreviewEventId))
      || (relatedPreviewIncident.events || []).find((event) => Number(event.id) === Number(relatedPreviewIncident.representative_event_id))
      || relatedPreviewIncident
    )
    : (
      (focusedIncident?.events || []).find((event) => Number(event.id) === Number(focusedFaceEventId))
      || (focusedIncident?.events || []).find((event) => Number(event.id) === Number(focusedIncident.representative_event_id))
      || focusedIncident
    );
  const findSimilarEvent = (focusedIncident?.events || []).find((event) => Number(event.id) === Number(findSimilarObject?.eventId)) || focusedEvent;
  const findSimilarEventId = findSimilarObject?.eventId
    ? Number(incidentTrackingSource(findSimilarEvent, focusedIncident)?.id || findSimilarEvent.id)
    : relatedAnchorEventId;
  const visualFindSimilarEventId = Number(findSimilarEvent?.representative_event_id || findSimilarEvent?.id) || null;
  const focusedSnapshotEvent = displayedEvent;
  const focusedSnapshotEventId = Number(focusedSnapshotEvent?.representative_event_id || focusedSnapshotEvent?.id);
  const focusedLoadedImageSize = Number(focusedImageSize?.eventId) === focusedSnapshotEventId ? focusedImageSize : null;
  const galleryIncidents = visibleIncidents;
  const incidentPageCount = Math.max(1, Math.ceil(displayedIncidentTotal / incidentsPerPage));
  const clampedIncidentPage = Math.min(incidentPage, incidentPageCount - 1);
  const pagedIncidents = galleryIncidents;
  incidentPagingRef.current = {
    // The old page stays visible while the next request is in flight. Anchor
    // its actual offset if the user toggles the gallery before that load ends.
    offset: semanticIncidentActive ? incidentPage * incidentsPerPage : Number(new URLSearchParams(incidentLoadedQueryRef.current).get("offset")) || 0,
    pageSize: incidentsPerPage, items: visibleIncidents, selectedId: focusedIncident?.id,
  };
  // Tiles are memoized, so they get handlers that never change identity.
  const tileHandlersRef = useRef({});
  const selectTile = useCallback((incident) => tileHandlersRef.current.select(incident), []);
  const previewTile = useCallback((incident) => tileHandlersRef.current.preview(incident), []);
  galleryGrowthRef.current = { count: visibleIncidents.length, total: displayedIncidentTotal, wanted: galleryWanted };
  // Ask for pictures near the size they are shown at, not one large size for every layout.
  const pixelRatio = typeof window !== "undefined" ? window.devicePixelRatio : 1;
  const tileCssWidth = mobileView ? viewportWidth - 40
    : galleryMode ? (viewportWidth - FILTER_PANE_WIDTH - 16 - 8 * (galleryColumns - 1)) / galleryColumns
    : 160;
  // A phone shows one picture per row; past 1.5x the extra detail is not worth the bytes on a mobile link.
  const tileThumbnailWidth = thumbnailWidthFor(tileCssWidth, mobileView ? Math.min(1.5, pixelRatio) : pixelRatio);
  const tileImageRef = useRef({});
  tileImageRef.current = { width: tileThumbnailWidth, columns: mobileView ? 2 : galleryColumns, plain: thumbnailObjectFocus === "off" };

  // Review marks: the checkbox answers at once, then the server confirms.
  const [reviewOverrides, setReviewOverrides] = useState({});
  const [reviewError, setReviewError] = useState("");
  const isReviewed = (incident) => reviewOverrides[String(incident.id)] ?? REVIEWED_VERDICTS.has(incident.review_verdict);
  async function changeReview(incident, checked) {
    const key = String(incident.id);
    setReviewError("");
    setReviewOverrides((current) => ({ ...current, [key]: checked }));
    try {
      const response = await fetch(`/api/incidents/${encodeURIComponent(incident.incident_id || incident.id)}/review`, {
        method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ verdict: checked ? "reviewed" : null }),
      });
      if (!response.ok) throw new Error(response.status === 403 ? "Only an administrator can mark incidents as reviewed." : "Could not save the review mark. Try again.");
    } catch (failure) {
      setReviewOverrides((current) => { const next = { ...current }; delete next[key]; return next; });
      setReviewError(failure.message || "Could not save the review mark. Try again.");
    }
  }
  const reviewTile = useCallback((incident, checked) => tileHandlersRef.current.review(incident, checked), []);

  useEffect(() => {
    if (!displayedIncident && tabletInspectorOpen) setTabletInspectorOpen(false);
  }, [displayedIncident, tabletInspectorOpen]);

  useEffect(() => {
    onAssistantContextChange?.({
      page: "incidents",
      camera_id: focusedIncident?.camera_id || (incidentFilterValues(incidentCameraFilter).length === 1 ? incidentCameraFilter : ""),
      incident_event_id: Number(focusedEvent?.representative_event_id || focusedEvent?.id) || null,
      filters: {
        day: incidentDay,
        event_type: eventFilter,
        camera: incidentCameraFilter,
        object: incidentObjectFilter,
        zone: incidentZoneFilter,
      },
    });
  }, [eventFilter, focusedEvent?.id, focusedEvent?.representative_event_id, focusedIncident?.camera_id, incidentCameraFilter, incidentDay, incidentObjectFilter, incidentZoneFilter, onAssistantContextChange]);

  useEffect(() => {
    if (mobileView || !focusedEvent || relatedPreviewIncident || linkedIncidentBootRef.current) return;
    const eventId = Number(focusedEvent.representative_event_id || focusedEvent.id);
    if (!Number.isInteger(eventId) || eventId <= 0) return;
    const nextHref = focusedIncident?.incident_id && focusedIncident?.revision != null
      ? (() => { const url = new URL(window.location.href); url.searchParams.set("incident_id", focusedIncident.incident_id); url.searchParams.delete("event_ids"); return `${url.pathname}${url.search}${url.hash}`; })()
      : incidentSelectionHref(
      `${window.location.pathname}${window.location.search}${window.location.hash}`,
      eventId,
      APP_BASE_PATH,
    );
    const currentHref = `${window.location.pathname}${window.location.search}${window.location.hash}`;
    if (nextHref && nextHref !== currentHref) {
      window.history.replaceState(window.history.state, "", nextHref);
    }
  }, [focusedIncident?.incident_id, focusedEvent?.id, focusedEvent?.representative_event_id, mobileView, relatedPreviewIncident]);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const canonicalId = params.get("incident_id");
    const eventIds = canonicalId ? null : (linkedIncidentBootRef.current || params.get("event_ids"));
    if (!eventIds && !canonicalId) {
      setLinkedIncidentLoading(false);
      return undefined;
    }
    let cancelled = false;
    setLinkedIncidentLoading(true);
    setLinkedIncidentError("");
    const query = new URLSearchParams(canonicalId ? { incident_id: canonicalId } : { event_ids: eventIds, gap_seconds: "45" });
    fetch(`/api/incidents/detail?${query}`)
      .then((response) => response.ok ? response.json() : Promise.reject(new Error("Linked incident unavailable")))
      .then((detail) => {
        if (cancelled) return;
        const requestedEventId = eventIds ? Number(String(eventIds).split(",")[0]) : Number(detail.representative_event_id);
        if (mobileView) {
          setSelectedEvent(detail);
          linkedIncidentBootRef.current = null;
          setLinkedIncidentLoading(false);
          return;
        }
        setLinkedIncidentDetail(detail);
        setLinkedIncidentEventId(Number.isInteger(requestedEventId) ? requestedEventId : Number(detail.representative_event_id) || null);
        setExpandedIncidentId(detail.id);
        setFocusedFaceEventId(Number.isInteger(requestedEventId) ? requestedEventId : Number(detail.representative_event_id) || null);
        const linkedEpoch = new Date(detail.start_at || detail.created_at || 0).getTime();
        if (Number.isFinite(linkedEpoch) && linkedEpoch > 0) setIncidentDay(dateKeyForTimeZone(linkedEpoch, timeZone));
        setEventFilter(linkedIncidentEventFilter(detail));
        linkedIncidentBootRef.current = null;
        setLinkedIncidentLoading(false);
        setLinkedIncidentError("");
      })
      .catch(() => {
        if (cancelled) return;
        linkedIncidentBootRef.current = null;
        setLinkedIncidentLoading(false);
        setLinkedIncidentError("Linked incident unavailable.");
      });
    return () => { cancelled = true; };
  }, [mobileView, timeZone]);

  useEffect(() => {
    clearLegacyIncidentFilterStorage();
  }, []);

  useEffect(() => {
    if (mobileView) return undefined;
    const rail = incidentRailListRef.current;
    if (!rail) return undefined;
    let resizeTimer;
    function updateRailSize() {
      const rect = rail.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return;
      const width = rect.width;
      const height = rect.height;
      setViewportWidth(window.innerWidth);
      if (galleryExpanded) {
        // The gallery has no pages; a taller or wider list may need more rows.
        galleryPageAnchorRef.current = null;
        // The list animates to full width, which moves every row.
        revealChosenInGallery();
        loadMoreGalleryIfNearEnd();
        return;
      }
      const pageSize = incidentThumbnailPageSize({ width, height, density: incidentDensity,
          ...(incidentDensity === "comfortable" ? { columns: 2, gap: 6, horizontalPadding: 16 } : {}) });
      const paging = incidentPagingRef.current;
      if (paging.pageSize !== pageSize) {
        const selectedIndex = paging.items.findIndex((item) => sameIncidentId(item.id, paging.selectedId));
        const anchor = galleryPageAnchorRef.current ?? (paging.offset + Math.max(0, selectedIndex));
        // Change limit and offset together, after layout settles, so animation
        // frames cannot each start another incident search.
        setIncidentPage(Math.floor(anchor / pageSize));
        setIncidentPageSize(pageSize);
      }
      galleryPageAnchorRef.current = null;
    }
    function scheduleResize() {
      window.clearTimeout(resizeTimer);
      resizeTimer = window.setTimeout(updateRailSize, 100);
    }
    scheduleResize();
    const observer = new ResizeObserver(scheduleResize);
    observer.observe(rail);
    return () => { observer.disconnect(); window.clearTimeout(resizeTimer); };
  }, [mobileView, galleryExpanded, incidentDensity]);

  function revealChosenInGallery() {
    const armed = galleryRevealRef.current;
    if (!armed || !galleryModeRef.current) return;
    if (armed !== Infinity && performance.now() > armed) { galleryRevealRef.current = 0; return; }
    // Wait for the gallery's own list; until then the viewer's page is still shown.
    if (!semanticIncidentActive && !incidentLoadedQueryRef.current.startsWith("gallery:")) return;
    const chosen = incidentRailListRef.current?.querySelector(".live-activity-item.selected");
    if (!chosen) return;
    if (armed === Infinity) galleryRevealRef.current = performance.now() + 1000;
    chosen.scrollIntoView({ block: "center" });
  }

  // Arrow keys move between pictures in the gallery; Enter or Space opens one as usual.
  function onGalleryKeyDown(event) {
    if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
    const current = event.target instanceof HTMLElement ? event.target.closest(".live-activity-select") : null;
    if (!current || !galleryModeRef.current) return;
    const columns = mobileView ? 1 : galleryColumns;
    const tiles = [...event.currentTarget.querySelectorAll(".live-activity-select")];
    const index = tiles.indexOf(current);
    const step = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -columns, ArrowDown: columns }[event.key];
    const target = event.key === "Home" ? 0 : event.key === "End" ? tiles.length - 1 : step == null ? null : index + step;
    if (target == null || index < 0) return;
    event.preventDefault();
    tiles[Math.max(0, Math.min(tiles.length - 1, target))]?.focus();
  }

  function loadMoreGalleryIfNearEnd() {
    // The desktop gallery scrolls inside its own list. On a phone the list
    // scrolls too when the layout bounds it, and otherwise the page does.
    const list = incidentRailListRef.current || document.getElementById("incident-results");
    const scroller = list && list.scrollHeight > list.clientHeight + 1 && /auto|scroll/.test(getComputedStyle(list).overflowY)
      ? list : document.scrollingElement;
    const growth = galleryGrowthRef.current;
    if (!scroller || !galleryModeRef.current) return;
    // Fewer shown than asked for means a load is in flight or has failed.
    if (growth.count >= growth.total || growth.count < growth.wanted) return;
    // Start early: the next incidents are already fetched, so appending them is immediate.
    if (scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight > Math.max(600, scroller.clientHeight * 1.5)) return;
    setGalleryWanted((wanted) => (wanted === growth.wanted ? wanted + INCIDENT_GALLERY_CHUNK : wanted));
  }

  useEffect(() => {
    if (!galleryMode) return undefined;
    const frame = window.requestAnimationFrame(() => {
      revealChosenInGallery();
      loadMoreGalleryIfNearEnd();
    });
    return () => window.cancelAnimationFrame(frame);
  }, [galleryMode, incidentResultSource, displayedIncidentTotal, galleryColumns, galleryWanted, semanticIncidentActive]);

  useEffect(() => {
    if (!mobileView) return undefined;
    const onScroll = () => loadMoreGalleryIfNearEnd();
    const onResize = () => setViewportWidth(window.innerWidth);
    // Capture, so a scroll inside the list is heard as well as a page scroll.
    window.addEventListener("scroll", onScroll, { passive: true, capture: true });
    window.addEventListener("resize", onResize);
    return () => { window.removeEventListener("scroll", onScroll, { capture: true }); window.removeEventListener("resize", onResize); };
  }, [mobileView]);

  useEffect(() => {
    if (confidenceDraft === incidentMinConfidence) return undefined;
    const timer = window.setTimeout(() => setIncidentMinConfidence(confidenceDraft), 300);
    return () => window.clearTimeout(timer);
  }, [confidenceDraft, incidentMinConfidence]);

  function refresh() {
    refreshBase();
    refreshIncidentEvidence();
  }

  function refreshIncidentEvidence(full = true) {
    if (full !== false) galleryFullRefreshRef.current = true;
    const sequence = ++incidentEvidenceRefreshSequence.current;
    incidentDetailCacheRef.current.clear();
    setIncidentRefreshToken((value) => value + 1);
    // Keep visible evidence until its replacement arrives, including on failure.
    // Drop only inactive details so future selections fetch fresh evidence.
    const selections = [
      [focusedSummary, null],
      [selectedEvent, setSelectedEvent],
      [linkedIncidentDetail, setLinkedIncidentDetail],
      [relatedPreviewIncident, setRelatedPreviewIncident],
    ];
    const retainedQueries = new Set(selections.map(([selection]) => incidentDetailQuery(selection)).filter(Boolean));
    setIncidentDetails((current) => Object.fromEntries(
      Object.entries(current).filter(([query]) => retainedQueries.has(query)),
    ));
    for (const [selection, setSelection] of selections) {
      const query = incidentDetailQuery(selection);
      if (!query) continue;
      incidentDetailCacheRef.current.load(query).then((detail) => {
        if (sequence !== incidentEvidenceRefreshSequence.current) return;
        setIncidentDetails((current) => ({ ...current, [query]: detail }));
        setSelection?.((current) => incidentDetailQuery(current) === query ? detail : current);
      }).catch(() => {
        // SSE reconnect or the fallback poll retries without hiding the selection.
      });
    }
  }

  async function runSemanticIncidentSearch(event, requestedQuery = semanticIncidentQuery) {
    event?.preventDefault();
    const queryText = String(requestedQuery || "").trim();
    if (!queryText) return;
    retainedGallerySelectionRef.current = null;
    galleryPageAnchorRef.current = null;
    semanticIncidentRequestRef.current?.abort();
    const controller = new AbortController();
    semanticIncidentRequestRef.current = controller;
    setSemanticIncidentQuery(queryText);
    setSemanticIncidentActiveQuery(queryText);
    setSemanticIncidentResults([]);
    setSemanticIncidentLoading(true);
    setSemanticIncidentError("");
    setIncidentPage(0);
    setGalleryWanted(INCIDENT_GALLERY_CHUNK);
    setEventFilter("object");
    try {
      const nextDay = addDaysToDateKey(incidentDay || today, 1);
      const startEpoch = zonedDateSecondToEpoch(incidentDay || today, 0, timeZone);
      const endEpoch = zonedDateSecondToEpoch(nextDay, 0, timeZone);
      const response = await fetch("/api/semantic-search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify(semanticIncidentRequest({
          query: queryText,
          cameraFilter: incidentCameraFilter,
          objectFilter: incidentObjectFilter,
          startAt: new Date(startEpoch * 1000).toISOString(),
          endAt: new Date(endEpoch * 1000).toISOString(),
        })),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Semantic incident search failed");
      const hits = Array.isArray(payload.results) ? payload.results : [];
      const hydrated = await mapWithConcurrency(hits, 6, async (hit) => {
        const eventId = Number(hit?.event?.id);
        if (!Number.isInteger(eventId) || eventId <= 0) return null;
        const detailResponse = await fetch(`/api/incidents/by-event/${eventId}`, { signal: controller.signal });
        if (!detailResponse.ok) return null;
        const detail = await detailResponse.json();
        return {
          ...detail,
          semantic_search: {
            query: queryText,
            score: Number(hit.score),
            rank_score: Number(hit.rank_score ?? hit.score),
            match_strength: hit.match_strength || "visual_similarity",
            component_scores: hit.component_scores || null,
            evidence: hit.evidence || null,
            event_id: eventId,
          },
        };
      });
      if (semanticIncidentRequestRef.current !== controller) return;
      setSemanticIncidentResults(rankSemanticIncidentDetails(hydrated, incidentZoneFilter, {
        minConfidence: incidentMinConfidence / 100, labels: incidentFilterValues(incidentObjectFilter),
      }));
    } catch (error) {
      if (error?.name !== "AbortError") setSemanticIncidentError(error.message || "Semantic incident search failed");
    } finally {
      if (semanticIncidentRequestRef.current === controller) {
        semanticIncidentRequestRef.current = null;
        setSemanticIncidentLoading(false);
      }
    }
  }

  function resetSemanticIncidentSearch() {
    retainedGallerySelectionRef.current = null;
    galleryPageAnchorRef.current = null;
    semanticIncidentRequestRef.current?.abort();
    semanticIncidentRequestRef.current = null;
    setSemanticIncidentQuery("");
    setSemanticIncidentActiveQuery("");
    setSemanticIncidentResults([]);
    setSemanticIncidentLoading(false);
    setSemanticIncidentError("");
    setIncidentPage(0);
  }

  function clearIncidentFilters() {
    setIncidentCameraFilter("all");
    setIncidentObjectFilter("all");
    setIncidentZoneFilter("all");
    setIncidentPersonFilter("all");
    setConfidenceDraft(0);
    setIncidentMinConfidence(0);
  }

  useEffect(() => () => semanticIncidentRequestRef.current?.abort(), []);

  useEffect(() => {
    if (!semanticIncidentActiveQuery) return;
    runSemanticIncidentSearch(null, semanticIncidentActiveQuery);
  }, [incidentCameraFilter, incidentDay, incidentObjectFilter, incidentZoneFilter, incidentMinConfidence]);

  useAppEvents(({ type, data }) => {
    if (document.hidden || (type !== "incident" && type !== "resync")) return;
    const affectedId = Number(data?.event_id);
    const affectedIncidentId = data?.incident_id;
    const forceReconcile = data?.reason === "operator_correction";
    const visible = [focusedIncident, selectedEvent, linkedIncidentDetail, relatedPreviewIncident, ...incidents];
    const affectsVisibleEvidence = visible.some((incident) => (
      sameIncidentId(incident?.incident_id || incident?.id, affectedIncidentId)
      || Number(incident?.representative_event_id) === affectedId
      || Number(incident?.id) === affectedId
      || incident?.events?.some((event) => Number(event.id) === affectedId)
      || incident?.event_ids?.some((eventId) => Number(eventId) === affectedId)
    ));
    if (type !== "resync" && !forceReconcile && !affectsVisibleEvidence && (incidentDay !== today || incidentPage !== 0)) return;
    if (incidentEventRefreshTimer.current) return;
    incidentEventRefreshTimer.current = window.setTimeout(
      () => {
        incidentEventRefreshTimer.current = null;
        // A new incident elsewhere only changes the top of the list.
        refreshIncidentEvidence(type === "resync" || forceReconcile || affectsVisibleEvidence);
      },
      1000,
    );
  }, true, ["incident"]);

  useEffect(() => () => {
    window.clearTimeout(incidentEventRefreshTimer.current);
    incidentEvidenceRefreshSequence.current += 1;
  }, []);

  useVisiblePolling(
    // The fallback poll re-reads the newest incidents; every fourth one re-reads
    // everything shown, in case an event about an older incident was missed.
    () => refreshIncidentEvidence((refreshPollCountRef.current += 1) % 4 === 0),
    INCIDENT_REFRESH_FALLBACK_MS,
    true,
    { immediate: false },
  );

  useEffect(() => {
    if (incidentDay > today) setIncidentDay(today);
  }, [incidentDay, setIncidentDay, today]);

  useEffect(() => {
    const controller = new AbortController();
    // People named on the People page (face capture), id and name only.
    fetch("/api/face-names", { signal: controller.signal })
      .then((response) => response.ok ? response.json() : {})
      .then((payload) => {
        if (!controller.signal.aborted) setIncidentPeople(Array.isArray(payload?.people) ? payload.people : []);
      })
      .catch(() => {
        if (!controller.signal.aborted) setIncidentPeople([]);
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function searchPage(filters, limit, offset) {
      const query = new URLSearchParams(filters);
      query.set("limit", String(limit));
      query.set("offset", String(offset));
      const response = await fetch(`/api/incidents/search?${query}`);
      if (!response.ok) throw new Error("Unable to load incidents");
      return response.json();
    }
    // Keep one chunk in hand so reaching the end shows more at once, and start
    // the pictures of its first rows. A failure here is silent: the same
    // request is made again, and reported, when the incidents are needed.
    function fetchAhead(filters, baseKey, offset) {
      const current = galleryAheadRef.current;
      if (current && current.key === baseKey && current.token === incidentRefreshToken && current.offset === offset) return;
      const promise = searchPage(filters, INCIDENT_GALLERY_CHUNK, offset).then((payload) => {
        const tile = tileImageRef.current;
        if (tile.plain && galleryAheadRef.current?.promise === promise) {
          warmedImagesRef.current = (payload.items || []).slice(0, tile.columns * 2).map((incident) => {
            const image = new Image();
            image.decoding = "async";
            image.src = eventThumbnailUrl(incident, tile.width);
            return image;
          });
        }
        return payload;
      }).catch(() => null);
      galleryAheadRef.current = { key: baseKey, token: incidentRefreshToken, offset, limit: INCIDENT_GALLERY_CHUNK, promise };
    }
    async function loadGallery(filters) {
      const baseKey = `gallery:${filters}`;
      const loaded = galleryLoadedRef.current;
      const sameList = loaded.key === baseKey && loaded.token === incidentRefreshToken;
      if (sameList && loaded.items.length >= galleryWanted) return;
      const previous = loaded.key === baseKey ? loaded.items : [];
      const foregroundLoad = loaded.key !== baseKey;
      if (foregroundLoad) {
        setIncidentLoading(true);
        setIncidentLoadError("");
      }
      setGalleryMoreError("");
      try {
        // Scrolling adds to the end. A filter change reads again from the top. A
        // routine refresh re-reads the newest incidents and keeps the rest.
        let items = sameList ? loaded.items : [];
        let offset = items.length;
        let payload = null;
        if (!sameList && previous.length && !galleryFullRefreshRef.current) {
          const head = await searchPage(filters, INCIDENT_GALLERY_CHUNK, 0);
          if (cancelled) return;
          const merged = mergeIncidentHead(previous, head.items || []);
          if (merged) {
            payload = head;
            items = merged;
            offset = items.length;
          }
        }
        while (items.length < galleryWanted) {
          if (payload && offset >= Number(payload.total || 0)) break;
          const limit = Math.min(100, galleryWanted - items.length);
          const ahead = galleryAheadRef.current;
          const prepared = ahead && ahead.key === baseKey && ahead.token === incidentRefreshToken && ahead.offset === offset && ahead.limit === limit
            ? await ahead.promise : null;
          payload = prepared || await searchPage(filters, limit, offset);
          if (cancelled) return;
          const page = payload.items || [];
          const seen = new Set(items.map((item) => String(item.id)));
          items = [...items, ...page.filter((item) => !seen.has(String(item.id)))];
          offset += page.length;
          if (page.length < limit || offset >= Number(payload.total || 0)) break;
        }
        if (!payload) return;
        if (!sameList) {
          items = keepIncidentIdentity(previous, items);
          galleryFullRefreshRef.current = false;
        }
        galleryLoadedRef.current = { key: baseKey, token: incidentRefreshToken, items };
        incidentLoadedQueryRef.current = `${baseKey}&offset=0`;
        setIncidents(items);
        setIncidentTotal(Math.max(Number(payload.total || 0), items.length));
        setIncidentFacets(payload.facets || { camera_ids: [], labels: [], zones: [] });
        setIncidentLoadError("");
        if (offset < Number(payload.total || 0)) fetchAhead(filters, baseKey, offset);
      } catch (error) {
        if (cancelled) return;
        if (foregroundLoad) {
          galleryLoadedRef.current = { key: "", token: -1, items: [] };
          setIncidents([]);
          setIncidentTotal(0);
          setIncidentLoadError(error.message || "Unable to load incidents");
        } else if (sameList) {
          // Keep what is shown; say that the rest did not arrive.
          setGalleryMoreError("Could not load more incidents.");
        }
      } finally {
        if (!cancelled && foregroundLoad) setIncidentLoading(false);
      }
    }
    async function loadIncidentPage() {
      const filters = new URLSearchParams({
        day: incidentDay || today,
        time_zone: timeZone,
        event_type: eventFilter,
        gap_seconds: "45",
      });
      if (incidentCameraFilter !== "all") filters.set("camera_id", incidentCameraFilter);
      if (incidentObjectFilter !== "all") filters.set("object_label", incidentObjectFilter);
      if (incidentZoneFilter !== "all") filters.set("zone", incidentZoneFilter);
      if (incidentPersonFilter !== "all") filters.set("person_id", incidentPersonFilter);
      if (confidenceApplies && incidentMinConfidence > 0) filters.set("min_confidence", String(incidentMinConfidence / 100));
      if (galleryMode) {
        // Gallery tiles do not draw the per-event list, which is most of a busy incident's card.
        filters.set("compact", "true");
        return loadGallery(filters);
      }
      galleryLoadedRef.current = { key: "", token: -1, items: [] };
      const query = new URLSearchParams(filters);
      query.set("limit", String(incidentsPerPage));
      query.set("offset", String(incidentPage * incidentsPerPage));
      const queryKey = query.toString();
      const foregroundLoad = incidentLoadedQueryRef.current !== queryKey;
      if (foregroundLoad) {
        setIncidentLoading(true);
        setIncidentLoadError("");
      }
      try {
        const response = await fetch(`/api/incidents/search?${query}`);
        if (!response.ok) throw new Error("Unable to load incidents");
        const payload = await response.json();
        if (cancelled) return;
        incidentLoadedQueryRef.current = queryKey;
        const items = payload.items || [];
        if (items.some((item) => sameIncidentId(item.id, retainedGallerySelectionRef.current?.id))) {
          retainedGallerySelectionRef.current = null;
        }
        setIncidents(items);
        setIncidentTotal(Number(payload.total || 0));
        setIncidentFacets(payload.facets || { camera_ids: [], labels: [], zones: [] });
        setIncidentLoadError("");
        if (!mobileView && items.length && !incidentPageEdgeRef.current) {
          setExpandedIncidentId((current) => current || items[0].id);
        }
      } catch (error) {
        if (!cancelled && foregroundLoad) {
          setIncidents([]);
          setIncidentTotal(0);
          setIncidentLoadError(error.message || "Unable to load incidents");
        }
      } finally {
        if (!cancelled && foregroundLoad) setIncidentLoading(false);
      }
    }
    loadIncidentPage();
    return () => {
      cancelled = true;
    };
  }, [incidentDay, today, timeZone, eventFilter, incidentCameraFilter, incidentObjectFilter, incidentZoneFilter, incidentPersonFilter, incidentMinConfidence, confidenceApplies, incidentPage, incidentsPerPage, incidentRefreshToken, galleryMode]);

  useEffect(() => {
    retainedGallerySelectionRef.current = null;
    galleryPageAnchorRef.current = null;
    setIncidentPage(0);
    setGalleryWanted(INCIDENT_GALLERY_CHUNK);
    if (galleryModeRef.current && incidentRailListRef.current) incidentRailListRef.current.scrollTop = 0;
  }, [eventFilter, incidentCameraFilter, incidentObjectFilter, incidentZoneFilter, incidentPersonFilter, incidentMinConfidence, incidentDay, incidentDensity]);

  useEffect(() => {
    if (incidentPage >= incidentPageCount) setIncidentPage(Math.max(0, incidentPageCount - 1));
  }, [incidentPage, incidentPageCount]);

  useEffect(() => {
    setFocusedFaceEventId(
      sameIncidentId(linkedIncidentDetail?.id, focusedIncident?.id) ? linkedIncidentEventId : null,
    );
    setDesktopAnalysisStats(null);
    relatedPreviewRequestRef.current += 1;
    setRelatedPreviewIncident(null);
    setRelatedPreviewEventId(null);
    setRelatedPreviewLoadingEventId(null);
    setRelatedPreviewError("");
    setSelectedVisualObject(null);
    setObservationPreviewRequest(null);
    setFindSimilarObject(null);
    setTabletInspectorOpen(false);
  }, [focusedIncident?.id, linkedIncidentDetail?.id, linkedIncidentEventId]);

  useEffect(() => {
    setFocusedDetailError("");
    if (!focusedSummary || !focusedDetailQuery || incidentDetails[focusedDetailQuery]) {
      setFocusedDetailLoading(false);
      return;
    }
    let cancelled = false;
    const sequence = incidentEvidenceRefreshSequence.current;
    setFocusedDetailLoading(true);
    incidentDetailCacheRef.current.load(focusedDetailQuery).then((detail) => {
      if (!cancelled && sequence === incidentEvidenceRefreshSequence.current) {
        setIncidentDetails((current) => ({ ...current, [focusedDetailQuery]: detail }));
        setFocusedDetailError("");
        setFocusedDetailLoading(false);
      }
    }).catch(() => {
      if (!cancelled && sequence === incidentEvidenceRefreshSequence.current) {
        setFocusedDetailError("Could not load the full incident evidence. The compact summary is still shown.");
        setFocusedDetailLoading(false);
      }
    });
    return () => { cancelled = true; };
  }, [focusedSummary?.id, focusedDetailQuery, focusedDetailRetry, incidentDetails]);

  useEffect(() => {
    if (desktopAnalysisMode !== "tracks") return;
    const trackingEvent = incidentReplayTracking(displayedEvent, displayedIncident);
    if (!storedObjectTracks(trackingEvent).length) setDesktopAnalysisMode("clean");
  }, [desktopAnalysisMode, displayedEvent, displayedIncident, setDesktopAnalysisMode]);

  useEffect(() => {
    if (desktopAnalysisMode !== "depth") return;
    const depthConfigured = Boolean(
      appConfig?.detector?.depth?.enabled
      && String(appConfig?.detector?.depth?.model_path || "").trim(),
    );
    if (!depthConfigured) setDesktopAnalysisMode("clean");
  }, [appConfig?.detector?.depth?.enabled, appConfig?.detector?.depth?.model_path, desktopAnalysisMode, setDesktopAnalysisMode]);

  useEffect(() => {
    const context = incidentRecordingContext(selectedEvent || focusedEvent);
    if (context) onRecordingContextChange(context);
  }, [selectedEvent?.id, focusedEvent?.id, focusedEvent?.created_at, focusedEvent?.camera_id, onRecordingContextChange]);

  useEffect(() => {
    if (expandedIncidentId
      && !sameIncidentId(retainedGallerySelectionRef.current?.id, expandedIncidentId)
      && !sameIncidentId(linkedIncidentDetail?.id, expandedIncidentId)
      && !visibleIncidents.some((incident) => sameIncidentId(incident.id, expandedIncidentId))) {
      setExpandedIncidentId(null);
    }
  }, [expandedIncidentId, linkedIncidentDetail?.id, visibleIncidents]);

  useEffect(() => {
    function onKey(keyEvent) {
      if (keyEvent.defaultPrevented || tabletInspectorOpen) return;
      // Escape leaves the viewer for the gallery, the page's home.
      if (keyEvent.key === "Escape" && !mobileView && !galleryExpanded && !selectedEvent) {
        keyEvent.preventDefault();
        changeGallery(true);
        return;
      }
      if (keyEvent.key === "Escape" && mobileView && expandedIncidentId && !selectedEvent) {
        keyEvent.preventDefault();
        setExpandedIncidentId(null);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expandedIncidentId, selectedEvent, galleryExpanded, tabletInspectorOpen, focusedIncident]);

  function changeGallery(expanded, selectedIncident = focusedIncident) {
    retainedGallerySelectionRef.current = selectedIncident;
    const paging = incidentPagingRef.current;
    const index = paging.items.findIndex((item) => sameIncidentId(item.id, selectedIncident?.id));
    const anchor = paging.offset + Math.max(0, index);
    galleryPageAnchorRef.current = anchor;
    setTabletInspectorOpen(false);
    if (expanded) {
      // Load far enough to include the open incident, then bring it into view.
      galleryRevealRef.current = index >= 0 ? Infinity : 0;
      setGalleryWanted((wanted) => incidentGalleryWanted(anchor, wanted));
      setIncidentPage(0);
    } else {
      setIncidentPage(Math.floor(anchor / Math.max(1, incidentPageSize)));
      if (incidentRailListRef.current) incidentRailListRef.current.scrollTop = 0;
    }
    setGalleryExpanded(expanded);
    incidentGalleryToggleRef.current?.focus({ preventScroll: true });
  }

  function selectGalleryIncident(incident) {
    toggleIncident(incident.id);
  }
  tileHandlersRef.current = { select: !mobileView && galleryExpanded ? openIncidentOverlay : selectGalleryIncident, preview: openIncidentOverlay, review: changeReview };

  function changeIncidentPage(page, { scroll = true } = {}) {
    retainedGallerySelectionRef.current = null;
    galleryPageAnchorRef.current = null;
    setIncidentPage(page);
    if (scroll && incidentRailListRef.current) incidentRailListRef.current.scrollTop = 0;
  }

  function alignIncidentList(edge) {
    const rail = incidentRailListRef.current;
    if (!rail) return;
    rail.scrollTop = edge === "end" ? Math.max(0, rail.scrollHeight - rail.clientHeight) : 0;
  }

  function toggleIncident(incidentId) {
    if (mobileView) {
      const incident = visibleIncidents.find((candidate) => sameIncidentId(candidate.id, incidentId));
      if (incident) window.location.assign(appUrl(`/incidents/${encodeURIComponent(incident.id)}`));
      return;
    }
    if (!sameIncidentId(incidentId, linkedIncidentDetail?.id)) {
      setLinkedIncidentDetail(null);
      setLinkedIncidentEventId(null);
      setLinkedIncidentError("");
    }
    setExpandedIncidentId(incidentId);
  }

  const focusedIndex = focusedIncident
    ? visibleIncidents.findIndex((incident) => sameIncidentId(incident.id, focusedIncident.id))
    : -1;

  function moveFocus(direction) {
    const step = incidentFocusStep({
      index: focusedIndex,
      count: visibleIncidents.length,
      page: clampedIncidentPage,
      pageCount: incidentPageCount,
      direction,
    });
    if (!step) return;
    if (step.index != null) {
      setExpandedIncidentId(visibleIncidents[step.index].id);
      return;
    }
    if (incidentPageEdgeRef.current || (!semanticIncidentActive && incidentLoading)) return;
    incidentPageEdgeRef.current = step.edge;
    changeIncidentPage(step.page, { scroll: false });
  }

  useEffect(() => {
    const edge = incidentPageEdgeRef.current;
    if (!edge) return undefined;
    if (!semanticIncidentActive && incidentLoading) return undefined;
    if (!semanticIncidentActive) {
      const loadedOffset = Number(new URLSearchParams(incidentLoadedQueryRef.current).get("offset")) || 0;
      if (loadedOffset !== incidentPage * incidentsPerPage) return undefined;
    }
    if (!visibleIncidents.length) {
      incidentPageEdgeRef.current = null;
      return undefined;
    }
    incidentPageEdgeRef.current = null;
    const target = edge === "end" ? visibleIncidents[visibleIncidents.length - 1] : visibleIncidents[0];
    setExpandedIncidentId(target.id);
    const frame = window.requestAnimationFrame(() => alignIncidentList(edge));
    return () => window.cancelAnimationFrame(frame);
  }, [incidentLoading, incidentPage, incidentsPerPage, semanticIncidentActive, visibleIncidents]);

  function selectDesktopAnalysisMode(mode) {
    setDesktopAnalysisMode(mode);
    setDesktopAnalysisStats(null);
    setDesktopReplayRequest((request) => request + 1);
  }

  async function selectRelatedIncident(match) {
    const eventId = Number(match?.event_id || match?.id);
    if (!Number.isInteger(eventId) || eventId <= 0) return;
    const request = ++relatedPreviewRequestRef.current;
    setRelatedPreviewLoadingEventId(eventId);
    setRelatedPreviewError("");
    try {
      const detail = await loadRelatedIncident(eventId, fetch);
      if (request !== relatedPreviewRequestRef.current) return;
      setRelatedPreviewIncident(detail);
      setRelatedPreviewEventId(eventId);
      // Keep focusedFaceEventId on the anchor incident so URL sync / Find similar
      // do not jump to the related hit or remount into a blank investigation.
      setDesktopAnalysisMode("clean");
      setDesktopAnalysisStats(null);
    } catch (error) {
      if (request === relatedPreviewRequestRef.current) {
        setRelatedPreviewError(error.message || "Could not load related incident. Please try again.");
      }
    } finally {
      if (request === relatedPreviewRequestRef.current) setRelatedPreviewLoadingEventId(null);
    }
  }

  function returnToSelectedIncident() {
    relatedPreviewRequestRef.current += 1;
    setRelatedPreviewIncident(null);
    setRelatedPreviewEventId(null);
    setRelatedPreviewLoadingEventId(null);
    setRelatedPreviewError("");
    setDesktopAnalysisMode("clean");
    setDesktopAnalysisStats(null);
  }

  useEffect(() => {
    if (mobileView || selectedEvent || galleryExpanded) return undefined;
    function onIncidentArrow(event) {
      const target = event.target;
      if (target instanceof HTMLElement && (target.isContentEditable || ["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName))) return;
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      const direction = event.key === "ArrowLeft" ? -1 : 1;
      if (!incidentFocusStep({ index: focusedIndex, count: visibleIncidents.length, page: clampedIncidentPage, pageCount: incidentPageCount, direction })) return;
      event.preventDefault();
      moveFocus(direction);
    }
    window.addEventListener("keydown", onIncidentArrow);
    return () => window.removeEventListener("keydown", onIncidentArrow);
  }, [mobileView, selectedEvent, galleryExpanded, focusedIndex, visibleIncidents, clampedIncidentPage, incidentPageCount, incidentLoading, semanticIncidentActive]);

  function shiftIncidentDay(days) {
    const next = addDaysToDateKey(incidentDay || today, days);
    if (next <= today) setIncidentDay(next);
  }

  function incidentDayControl({ caption = false } = {}) {
    const day = incidentDay || today;
    return (
      <div className="incident-day-field">
        {caption ? <span>Day</span> : null}
        <div className="incident-day-stepper">
          <button type="button" className="incident-day-step" onClick={() => shiftIncidentDay(-1)} aria-label="Previous day" title="Previous day"><ChevronLeft size={16} /></button>
          <input type="date" value={day} max={today} onChange={(event) => setIncidentDay(event.target.value || today)} aria-label="Incident day" />
          <button type="button" className="incident-day-step" onClick={() => shiftIncidentDay(1)} disabled={day >= today} aria-label="Next day" title="Next day"><ChevronRight size={16} /></button>
        </div>
      </div>
    );
  }

  const semanticIncidentControl = (
    <div className={`incident-semantic-search ${semanticIncidentActive ? "active" : ""} ${semanticIncidentError ? "error" : ""}`}>
      <form onSubmit={runSemanticIncidentSearch} role="search" aria-label="Semantic incident search">
        <Search size={15} aria-hidden="true" />
        <input value={semanticIncidentQuery} onChange={(event) => setSemanticIncidentQuery(event.target.value)} placeholder="Search incidents…" aria-label="Describe incidents to find" disabled={semanticIncidentLoading} />
        {semanticIncidentActive ? <button type="button" className="secondary" onClick={resetSemanticIncidentSearch} aria-label="Reset semantic incident search">Reset</button> : null}
        <button type="submit" disabled={semanticIncidentLoading || !semanticIncidentQuery.trim()}>{semanticIncidentLoading ? "Searching…" : "Search"}</button>
      </form>
      {semanticIncidentError ? <small>{semanticIncidentError}</small> : semanticIncidentActive ? <small>Showing visual matches for “{semanticIncidentActiveQuery}” using the selected day and filters.</small> : null}
    </div>
  );

  const tileHud = galleryMode || incidentDensity === "comfortable";
  function renderTile(incident) {
    return <IncidentTile key={incident.id} incident={incident} cameraName={cameraNameById.get(incident.camera_id) || incident.camera_id} timeZone={timeZone}
      selected={sameIncidentId(incident.id, focusedIncident?.id)} thumbnailAnnotations={thumbnailAnnotations} thumbnailObjectFocus={thumbnailObjectFocus}
      thumbnailObjectFocusZoom={thumbnailObjectFocusZoom} thumbnailWidth={tileThumbnailWidth} hud={tileHud}
      reviewed={tileHud ? isReviewed(incident) : null} reviewDisabled={!canCorrectIncident} onReviewChange={reviewTile}
      onSelect={selectTile} onOpenOverlay={previewTile} />;
  }
  // The list covers one day. Its end offers the day before, one step away.
  const earlierDay = addDaysToDateKey(incidentDay || today, -1);
  const earlierDayLabel = new Intl.DateTimeFormat("en-US", { timeZone: "UTC", weekday: "long", month: "short", day: "numeric" }).format(new Date(`${earlierDay}T12:00:00Z`));
  const dayComplete = !semanticIncidentActive && !displayedIncidentLoading && !displayedIncidentError && visibleIncidents.length >= displayedIncidentTotal;
  function openEarlierDay() {
    setIncidentDay(earlierDay);
    if (incidentRailListRef.current) incidentRailListRef.current.scrollTop = 0;
    else window.scrollTo(0, 0);
  }
  const galleryStatus = (
    <div className="incident-gallery-status" role="status">
      {reviewError ? <span className="incident-gallery-error">{reviewError}</span> : null}
      {galleryMoreError ? <><span>{galleryMoreError}</span><button type="button" onClick={() => setIncidentRefreshToken((value) => value + 1)}>Retry</button></>
        : visibleIncidents.length && visibleIncidents.length < displayedIncidentTotal ? <span>Showing {visibleIncidents.length} of {displayedIncidentTotal}. Scroll for more.</span>
        : visibleIncidents.length && semanticIncidentActive ? <span>All {displayedIncidentTotal} {displayedIncidentTotal === 1 ? "match" : "matches"}.</span>
        : visibleIncidents.length ? <span>All {displayedIncidentTotal} {displayedIncidentTotal === 1 ? "incident" : "incidents"} for this day.</span> : null}
      {dayComplete ? <button type="button" className="incident-earlier-day" onClick={openEarlierDay}>Earlier: {earlierDayLabel}</button> : null}
    </div>
  );
  const confidenceControl = (
    <section className="incident-filter-section incident-filter-confidence">
      <h3 id="incident-confidence-heading">Confidence threshold</h3>
      <input type="range" min={0} max={95} step={5} value={confidenceApplies ? confidenceDraft : 0} disabled={!confidenceApplies} onChange={(event) => setConfidenceDraft(Number(event.target.value))} aria-labelledby="incident-confidence-heading" aria-valuetext={confidenceDraft > 0 ? `${confidenceDraft}% or higher` : "Any confidence"} style={{ "--fill": `${(confidenceApplies ? confidenceDraft : 0) / 95 * 100}%` }} />
      <div className="incident-filter-confidence-readout">
        <span>{displayedIncidentTotal} {displayedIncidentTotal === 1 ? "result" : "results"}</span>
        <output>{confidenceApplies && confidenceDraft > 0 ? `${confidenceDraft}%` : "Any"}</output>
      </div>
      {!confidenceApplies ? <small>Motion-only incidents have no detection confidence.</small> : null}
    </section>
  );
  const filterLists = (
    <>
      <IncidentFilterChecks title="Cameras" filter={incidentCameraFilter} onChange={setIncidentCameraFilter}
        options={incidentCameraOptions.map((id) => ({ value: id, label: cameraNameById.get(id) || id }))} empty="No cameras in this period." />
      <IncidentFilterChecks title="Objects" filter={incidentObjectFilter} onChange={setIncidentObjectFilter} columns
        options={incidentObjectOptions.map((label) => ({ value: label, label: label.replaceAll("_", " "), icon: <IncidentObjectIcon label={label} size={15} /> }))} empty="No objects in this period." />
      {confidenceControl}
      <IncidentFilterChecks title="Zones" filter={incidentZoneFilter} onChange={setIncidentZoneFilter}
        options={incidentZoneOptions.map((zone) => ({ value: zone, label: zone }))} empty="No zones in this period." />
      <IncidentFilterChecks title="People" filter={incidentPersonFilter} onChange={(next) => { resetSemanticIncidentSearch(); setIncidentPersonFilter(next); }}
        options={incidentPeople.map((person) => ({ value: String(person.id), label: person.name }))} empty="No named people yet." />
    </>
  );

  if (!mobileView) {
    return (
      <main className={`incidents-desktop-page with-inspector${galleryExpanded ? " gallery-expanded" : ""}`}>
        <section className="bento-card incidents-desktop-shell">
          {galleryExpanded || filtersPinned ? (
            <aside className="incident-filter-pane" aria-label="Incident filters">
              <div className="incident-filter-pane-top">
                <div className="incident-filter-toggle compact incident-filter-segments" role="group" aria-label="Incident type filter">
                  <button className={eventFilter === "object" ? "active" : ""} aria-pressed={eventFilter === "object"} onClick={() => setEventFilter("object")}><ScanSearch size={15} aria-hidden="true" />Objects</button>
                  <button className={eventFilter === "motion" ? "active" : ""} aria-pressed={eventFilter === "motion"} onClick={() => { resetSemanticIncidentSearch(); setEventFilter("motion"); }}><Radar size={15} aria-hidden="true" />Motion</button>
                </div>
                {semanticIncidentControl}
                {incidentDayControl()}
              </div>
              <div className="incident-filter-pane-scroll">
                {filterLists}
              </div>
              <div className="incident-filter-pane-foot incident-toolbar-summary" aria-label={`${activeIncidentFilterCount} active filters`}>
                <button type="button" onClick={clearIncidentFilters} disabled={!activeIncidentFilterCount}>Clear filters{activeIncidentFilterCount ? ` (${activeIncidentFilterCount})` : ""}</button>
              </div>
            </aside>
          ) : null}

          <div className="incidents-desktop-workspace">
            <aside className={`incident-rail ${incidentDensity}`}>
              <div className="incident-rail-head">
                <button ref={incidentGalleryToggleRef} type="button" className="incident-gallery-toggle" aria-expanded={galleryExpanded} aria-controls="incident-results" title={galleryExpanded ? "Open the incident viewer" : "Back to the gallery (Esc)"} onClick={() => changeGallery(!galleryExpanded)}>
                  {galleryExpanded ? <PanelLeft size={15} /> : <LayoutGrid size={15} />}
                  {galleryExpanded ? "Open viewer" : "Gallery"}
                </button>
                {galleryExpanded ? (
                  <>
                    <span className="shown-bubble">{displayedIncidentTotal} {semanticIncidentActive ? "matches" : "shown"}</span>
                    <label className="incident-gallery-columns">
                      <span>Per row</span>
                      <input type="range" min={2} max={8} step={1} value={galleryColumns} onChange={(event) => setStoredGalleryColumns(incidentGalleryColumns(event.target.value))} aria-label="Incidents per row" style={{ "--fill": `${(galleryColumns - 2) / 6 * 100}%` }} />
                      <output>{galleryColumns}</output>
                    </label>
                  </>
                ) : (
                  <button type="button" className="incident-filter-pin" aria-pressed={filtersPinned} onClick={() => setFiltersPinned((pinned) => !pinned)}>
                    <SlidersHorizontal size={15} />Filters{activeIncidentFilterCount ? ` (${activeIncidentFilterCount})` : ""}
                  </button>
                )}
                <div className="density-control" aria-label="Thumbnail density" hidden={galleryExpanded}>
                  <button type="button" className={incidentDensity === "compact" ? "active" : ""} aria-pressed={incidentDensity === "compact"} onClick={() => setIncidentDensity("compact")} title="List view" aria-label="List view"><Rows3 size={15} /></button>
                  <button type="button" className={incidentDensity === "comfortable" ? "active" : ""} aria-pressed={incidentDensity === "comfortable"} onClick={() => setIncidentDensity("comfortable")} title="Grid view" aria-label="Grid view"><Grid2X2 size={15} /></button>
                </div>
              </div>
              <div id="incident-results" className="incident-rail-list" ref={incidentRailListRef} aria-label="Incidents" onKeyDown={galleryExpanded ? onGalleryKeyDown : undefined} onScroll={galleryExpanded ? loadMoreGalleryIfNearEnd : undefined} onWheel={() => { galleryRevealRef.current = 0; }} onPointerDown={() => { galleryRevealRef.current = 0; }} style={galleryExpanded ? { "--incident-gallery-columns": galleryColumns } : undefined}>
                {displayedIncidentLoading && !galleryIncidents.length ? <div className="empty-state">{semanticIncidentActive ? "Searching indexed incidents..." : "Loading incidents..."}</div> : null}
                {!galleryIncidents.length && displayedIncidentError ? <div className="empty-state">{displayedIncidentError}</div> : null}
                {galleryIncidents.length ? pagedIncidents.map(renderTile) : null}
                {!displayedIncidentLoading && !displayedIncidentError && !galleryIncidents.length ? (
                  <div className="empty-state">
                    {linkedIncidentLoading
                      ? "Opening linked incident…"
                      : linkedIncidentError
                        ? linkedIncidentError
                        : (semanticIncidentActive ? "No semantic matches for the selected filters." : "No other incidents.")}
                  </div>
                ) : null}
              </div>
              {galleryExpanded ? (
                galleryStatus
              ) : (
                <div className={`incident-pager ${displayedIncidentTotal > incidentsPerPage ? "" : "placeholder"}`} aria-label="Incident pages" aria-hidden={displayedIncidentTotal <= incidentsPerPage}>
                  <button type="button" onClick={() => changeIncidentPage(Math.max(0, incidentPage - 1))} disabled={clampedIncidentPage === 0}>Prev</button>
                  <span>{clampedIncidentPage + 1} / {incidentPageCount}</span>
                  <button type="button" onClick={() => changeIncidentPage(Math.min(incidentPageCount - 1, incidentPage + 1))} disabled={clampedIncidentPage >= incidentPageCount - 1}>Next</button>
                </div>
              )}
            </aside>

            <section className="incident-investigation" inert={galleryExpanded} aria-hidden={galleryExpanded}>
              <div className="incident-desktop-focus">
                <div className="incident-focus-actions">
                  <button ref={tabletInspectorToggleRef} type="button" className="incident-inspector-toggle" onClick={() => setTabletInspectorOpen((open) => !open)} aria-expanded={tabletInspectorOpen} aria-controls="incident-inspector" disabled={!displayedIncident}>Details</button>
                  {incidentReturnPath(window.location.search) ? <button type="button" className="incident-return-close" onClick={closeOpenedIncident} aria-label="Close incident"><X size={18} /></button> : null}
                </div>
                {focusedIncident ? (
                  <>
                    <button type="button" className="incident-focus-arrow previous" onClick={() => moveFocus(-1)} disabled={!incidentFocusStep({ index: focusedIndex, count: visibleIncidents.length, page: clampedIncidentPage, pageCount: incidentPageCount, direction: -1 })} title="Previous incident" aria-label="Previous incident"><ChevronLeft size={26} /></button>
                    <button type="button" className="incident-focus-arrow next" onClick={() => moveFocus(1)} disabled={!incidentFocusStep({ index: focusedIndex, count: visibleIncidents.length, page: clampedIncidentPage, pageCount: incidentPageCount, direction: 1 })} title="Next incident" aria-label="Next incident"><ChevronRight size={26} /></button>
                  </>
                ) : null}
                {displayedIncident ? (
                  <>{focusedDetailLoading && !relatedPreviewIncident ? <div className="incident-focus-load-status" role="status">Loading full incident evidence…</div> : null}{focusedDetailError && !relatedPreviewIncident ? <div className="incident-focus-load-error" role="alert"><span>{focusedDetailError}</span><button type="button" onClick={() => setFocusedDetailRetry((value) => value + 1)}>Retry</button></div> : null}<IncidentCard
                    key={`${focusedIncident?.id || "none"}:${displayedIncident.id || displayedIncident.representative_event_id}`}
                    incident={displayedIncident}
                    scenePlayback={scenePlayback}
                    timeZone={timeZone}
                    expanded
                    thumbnailAnnotations={thumbnailAnnotations}
                    thumbnailObjectFocus={thumbnailObjectFocus}
                    thumbnailObjectFocusZoom={thumbnailObjectFocusZoom}
                    desktopWorkspace
                    zones={(appConfig?.cameras || []).find((camera) => camera.id === displayedIncident.camera_id)?.zones}
                    analysisMode={desktopAnalysisMode}
                    depthLayer={desktopDepthLayer}
                    replayRequest={desktopReplayRequest}
                    observationPreviewRequest={relatedPreviewIncident ? null : observationPreviewRequest}
                    selectedObjectIndex={relatedPreviewIncident ? null : (selectedVisualObject?.objectIndex ?? findSimilarObject?.objectIndex ?? null)}
                    onSelectObject={relatedPreviewIncident ? null : ((selection) => {
                      setSelectedVisualObject(selection);
                      if (!selection) {
                        setFindSimilarObject(null);
                        return;
                      }
                      // Main-viewer object click is an explicit Find similar action.
                      setFindSimilarObject(selection);
                      setTabletInspectorOpen(true);
                    })}
                    onReturnToSelected={relatedPreviewIncident ? returnToSelectedIncident : null}
                    onAnalysisStats={setDesktopAnalysisStats}
                    onToggle={toggleIncident}
                    onPreviewChange={relatedPreviewIncident ? undefined : setFocusedFaceEventId}
                    onImageSize={setFocusedImageSize}
                  /></>
                ) : (
                  <div className="empty-state">
                    {linkedIncidentLoading
                      ? "Opening linked incident…"
                      : (linkedIncidentError || "No incidents match the current filters.")}
                  </div>
                )}
              </div>
            </section>

            {tabletInspectorOpen ? <button type="button" className="incident-inspector-backdrop" onClick={() => closeTabletInspector()} aria-label="Close incident details" /> : null}
            <IncidentInspector
              open={tabletInspectorOpen}
              incident={displayedIncident}
              scenePlayback={scenePlayback}
              faceEvent={displayedEvent}
              searchEvent={findSimilarEvent}
              anchorEventId={findSimilarEventId}
              visualAnchorEventId={visualFindSimilarEventId}
              appearanceAnchorEventId={findSimilarEventId}
              selectedRelatedEventId={relatedPreviewEventId}
              relatedLoadingEventId={relatedPreviewLoadingEventId}
              relatedError={relatedPreviewError}
              cameraNameById={cameraNameById}
              appConfig={appConfig}
              timeZone={timeZone}
              imageSize={focusedLoadedImageSize}
              analysisMode={desktopAnalysisMode}
              depthLayer={desktopDepthLayer}
              analysisStats={desktopAnalysisStats}
              analysisPanel={!galleryExpanded && displayedIncident?.incident_id ? <IncidentAnalysisStatus key={displayedIncident.incident_id} incidentId={displayedIncident.incident_id} autoStart={false} embedded onDetail={(detail) => {
                const query = incidentDetailQuery(detail);
                setIncidentDetails((current) => Number(current[query]?.revision) > Number(detail.revision) ? current : { ...current, [query]: detail });
                setRelatedPreviewIncident((current) => current?.incident_id === detail.incident_id && Number(current.revision) <= Number(detail.revision) ? detail : current);
              }} /> : null}
              selectedObjectIndex={selectedVisualObject?.objectIndex ?? null}
              findSimilarObjectIndex={findSimilarObject?.objectIndex ?? null}
              onSelectObject={setSelectedVisualObject}
              onSelectObservation={(observation) => { if (observation.event_id) setObservationPreviewRequest({ eventId: observation.event_id }); }}
              onSceneChanged={(detail) => {
                incidentDetailCacheRef.current.clear();
                setIncidentDetails((current) => ({ ...current, [incidentDetailQuery(detail)]: detail }));
                setLinkedIncidentDetail(detail); setExpandedIncidentId(detail.id);
                setIncidentRefreshToken((value) => value + 1);
              }}
              onFindSimilar={(selection) => {
                if (selection?.eventId) setObservationPreviewRequest({ eventId: selection.eventId });
                // Find similar runs only from an explicit Find similar control.
                setFindSimilarObject(selection);
                if (selection) {
                  setSelectedVisualObject(selection);
                  setTabletInspectorOpen(true);
                }
              }}
              onAnalysisModeChange={selectDesktopAnalysisMode}
              onDepthLayerChange={setDesktopDepthLayer}
              onRelatedSelect={selectRelatedIncident}
              onRelatedReturn={returnToSelectedIncident}
              onClose={() => closeTabletInspector()}
              onAskAssistant={onAskAssistant}
            />
          </div>
        </section>
        {/* The same incident overlay the Live page opens. */}
        {selectedEvent ? <EventOverlay livePreview loading={incidentPreviewLoading} cameraNameById={cameraNameById} event={selectedEvent} events={visibleIncidents}
          timeZone={timeZone} canCorrectIncident={canCorrectIncident} onClose={closeOpenedIncident} onSelect={openIncidentOverlay} onRefresh={refresh} /> : null}
      </main>
    );
  }

  return (
    <main className="bento-grid incidents-grid">
      <section className="bento-card events-zone incidents-page-zone mobile-incidents">
        <div className="section-head compact incident-head">
          <div><h2>Incidents</h2></div>
          <div className="incident-head-actions">
            <button type="button" className="mobile-filter-button" aria-expanded={mobileFiltersOpen} aria-controls="mobile-incident-filters" onClick={() => setMobileFiltersOpen((open) => !open)}>
              <SlidersHorizontal size={17} /> Filters{activeIncidentFilterCount || incidentDay !== today || eventFilter !== "object" || semanticIncidentActive ? <span className="mobile-filter-active" aria-label="Filters active" /> : null}
            </button>
          </div>
        </div>
        {mobileFiltersOpen ? <div id="mobile-incident-filters" className="event-filter incident-filter-panel" aria-label="Incident filters">
          <div className="incident-filter-selects">
            <label><span>Activity</span><select value={eventFilter} aria-label="Incident type" onChange={(event) => { resetSemanticIncidentSearch(); setEventFilter(event.target.value); }}><option value="object">Objects</option><option value="motion">Motion</option></select></label>
            {incidentDayControl({ caption: true })}
          </div>
          {semanticIncidentControl}
          <div className="mobile-filter-actions">
            <button type="button" onClick={() => { clearIncidentFilters(); setIncidentDay(today); setEventFilter("object"); resetSemanticIncidentSearch(); }}>Reset filters</button>
            <button type="button" onClick={() => setMobileFiltersOpen(false)}>Done</button>
          </div>
          <div className="mobile-filter-lists">{filterLists}</div>
        </div> : null}
        <div id="incident-results" className="incident-gallery incident-phone-gallery" aria-label="Incidents" onKeyDown={onGalleryKeyDown}>
          {displayedIncidentLoading && !visibleIncidents.length ? <div className="empty-state">{semanticIncidentActive ? "Searching indexed incidents..." : "Loading incidents..."}</div> : null}
          {!visibleIncidents.length && displayedIncidentError ? <div className="empty-state">{displayedIncidentError}</div> : null}
          {visibleIncidents.length ? pagedIncidents.map(renderTile) : null}
          {!displayedIncidentLoading && !displayedIncidentError && !visibleIncidents.length ? <div className="empty-state">{semanticIncidentActive ? "No semantic matches for the selected filters." : "No incidents match the current filters."}</div> : null}
        </div>
        {galleryStatus}
      </section>
      {selectedEvent ? <EventOverlay event={selectedEvent} events={visibleIncidents} timeZone={timeZone} canCorrectIncident={canCorrectIncident} onClose={closeOpenedIncident} onSelect={openIncidentOverlay} onRefresh={refresh} /> : null}
    </main>
  );
}
