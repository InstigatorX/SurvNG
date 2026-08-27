import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  Camera,
  ChevronLeft,
  ChevronRight,
  Grid2X2,
  Search,
  Rows3,
} from "lucide-react";
import { useVisiblePolling } from "../visibilityPolling.mjs";
import { incidentTrackingSource, storedObjectTracks } from "../objectTrackReplay.mjs";
import { incidentDetailQuery, incidentSelectionHref, incidentThumbnailPageSize, linkedIncidentEventFilter } from "../incidentNavigation.mjs";
import { mapWithConcurrency, rankSemanticIncidentDetails, semanticIncidentRequest } from "../incidentSemanticSearch.mjs";
import { incidentRecordingContext, fetch } from "../shared/api.js";
import { INCIDENT_REFRESH_FALLBACK_MS } from "../shared/constants.js";
import { formatDateTime } from "../shared/format.js";
import { dateKeyForTimeZone, addDaysToDateKey, zonedDateSecondToEpoch } from "../shared/datetime.js";
import { useStoredState, isMobileViewport } from "../shared/hooks.js";
import { clearLegacyIncidentFilterStorage } from "../shared/cameras.js";
import { useAppEvents } from "../shared/events.js";
import { usePollingData, useIncidentDetails } from "../shared/polling.js";
import { incidentLabels, IncidentObjectBadges, IncidentListItem, EventOverlay } from "../shared/evidence.jsx";
import { IncidentCard, IncidentInspector } from "./IncidentCard.jsx";
import { FaceReviewDialog } from "../people/FacesPage.jsx";

export function IncidentsPage({ timeZone, onRecordingContextChange, onAssistantContextChange, onAskAssistant = null }) {
  const { cameras, appConfig, refresh: refreshBase } = usePollingData();
  const thumbnailAnnotations = appConfig?.incident_thumbnail_annotations ?? false;
  const thumbnailObjectFocus = appConfig?.incident_thumbnail_object_focus ?? "off";
  const thumbnailObjectFocusZoom = appConfig?.incident_thumbnail_object_focus_zoom ?? 1;
  const [eventFilter, setEventFilter] = useState("object");
  const [incidentCameraFilter, setIncidentCameraFilter] = useState("all");
  const [incidentObjectFilter, setIncidentObjectFilter] = useState("all");
  const [incidentZoneFilter, setIncidentZoneFilter] = useState("all");
  const [incidentPersonFilter, setIncidentPersonFilter] = useState("all");
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
  const [focusedFaceEventId, setFocusedFaceEventId] = useState(null);
  const [linkedIncidentDetail, setLinkedIncidentDetail] = useState(null);
  const [linkedIncidentEventId, setLinkedIncidentEventId] = useState(null);
  const [selectedFace, setSelectedFace] = useState(null);
  const [facePeople, setFacePeople] = useState([]);
  const [expandedIncidentId, setExpandedIncidentId] = useState(null);
  const [incidentPage, setIncidentPage] = useState(0);
  const incidentRailListRef = useRef(null);
  const [incidentRailSize, setIncidentRailSize] = useState({ width: 0, height: 0 });
  const [desktopAnalysisMode, setDesktopAnalysisMode] = useStoredState("survng.incidentDesktopAnalysis.v1", "clean");
  const [desktopAnalysisStats, setDesktopAnalysisStats] = useState(null);
  const [desktopReplayRequest, setDesktopReplayRequest] = useState(0);
  const [focusedImageSize, setFocusedImageSize] = useState(null);
  const [relatedPreviewIncident, setRelatedPreviewIncident] = useState(null);
  const [relatedPreviewEventId, setRelatedPreviewEventId] = useState(null);
  const [relatedPreviewLoadingEventId, setRelatedPreviewLoadingEventId] = useState(null);
  const [selectedVisualObject, setSelectedVisualObject] = useState(null);
  const [tabletInspectorOpen, setTabletInspectorOpen] = useState(false);
  const tabletInspectorToggleRef = useRef(null);
  const relatedPreviewRequestRef = useRef(0);
  const mobileView = isMobileViewport();
  const incidentRailReady = mobileView || (incidentRailSize.width > 0 && incidentRailSize.height > 0);
  const incidentsPerPage = mobileView
    ? 12
    : incidentThumbnailPageSize({
      ...incidentRailSize,
      density: incidentDensity,
      ...(incidentDensity === "comfortable" ? { columns: 2, gap: 6, horizontalPadding: 16 } : {}),
    });
  const previousIncidentsPerPageRef = useRef(incidentsPerPage);
  const cameraNameById = useMemo(() => new Map(cameras.map((camera) => [camera.id, camera.name || camera.id])), [cameras]);
  const incidentCameraOptions = incidentFacets.camera_ids || [];
  const incidentObjectOptions = incidentFacets.labels || [];
  const incidentZoneOptions = incidentFacets.zones || [];
  const semanticIncidentActive = Boolean(semanticIncidentActiveQuery);
  const activeIncidentFilterCount = [incidentCameraFilter, incidentObjectFilter, incidentZoneFilter, incidentPersonFilter].filter((value) => value !== "all").length;
  const incidentResultSource = semanticIncidentActive ? semanticIncidentResults : incidents;
  const displayedIncidentTotal = semanticIncidentActive ? semanticIncidentResults.length : incidentTotal;
  const displayedIncidentLoading = semanticIncidentActive ? semanticIncidentLoading : incidentLoading;
  const displayedIncidentError = semanticIncidentActive ? semanticIncidentError : incidentLoadError;

  function closeTabletInspector({ restoreFocus = true } = {}) {
    setTabletInspectorOpen(false);
    if (restoreFocus) window.requestAnimationFrame(() => tabletInspectorToggleRef.current?.focus());
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
  const explicitlyFocusedSummary = visibleIncidents.find((incident) => incident.id === expandedIncidentId)
    || (linkedIncidentDetail?.id === expandedIncidentId ? linkedIncidentDetail : null);
  const focusedSummary = mobileView ? explicitlyFocusedSummary : explicitlyFocusedSummary || visibleIncidents[0] || null;
  const focusedDetailQuery = incidentDetailQuery(focusedSummary);
  const focusedIncident = focusedSummary ? incidentDetails[focusedDetailQuery] || focusedSummary : null;
  const focusedEvent = (focusedIncident?.events || []).find((event) => Number(event.id) === Number(focusedFaceEventId))
    || (focusedIncident?.events || []).find((event) => Number(event.id) === Number(focusedIncident.representative_event_id))
    || focusedIncident;
  const relatedAnchorEvent = incidentTrackingSource(focusedEvent, focusedIncident) || focusedEvent;
  const relatedAnchorEventId = Number(relatedAnchorEvent?.representative_event_id || relatedAnchorEvent?.id) || null;
  const displayedIncident = relatedPreviewIncident || focusedIncident;
  const displayedEvent = (displayedIncident?.events || []).find((event) => Number(event.id) === Number(focusedFaceEventId))
    || (displayedIncident?.events || []).find((event) => Number(event.id) === Number(displayedIncident?.representative_event_id))
    || displayedIncident;
  const focusedSnapshotEvent = displayedEvent;
  const focusedSnapshotEventId = Number(focusedSnapshotEvent?.representative_event_id || focusedSnapshotEvent?.id);
  const focusedLoadedImageSize = Number(focusedImageSize?.eventId) === focusedSnapshotEventId ? focusedImageSize : null;
  const galleryIncidents = visibleIncidents;
  const incidentPageCount = Math.max(1, Math.ceil(displayedIncidentTotal / incidentsPerPage));
  const clampedIncidentPage = Math.min(incidentPage, incidentPageCount - 1);
  const pagedIncidents = galleryIncidents;
  useEffect(() => {
    if (!displayedIncident && tabletInspectorOpen) setTabletInspectorOpen(false);
  }, [displayedIncident, tabletInspectorOpen]);

  useEffect(() => {
    onAssistantContextChange?.({
      page: "incidents",
      camera_id: focusedIncident?.camera_id || (incidentCameraFilter === "all" ? "" : incidentCameraFilter),
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
    if (mobileView || !focusedEvent) return;
    const eventId = Number(focusedEvent.representative_event_id || focusedEvent.id);
    const nextHref = incidentSelectionHref(window.location.href, eventId);
    if (nextHref && nextHref !== window.location.href) window.history.replaceState(window.history.state, "", nextHref);
  }, [focusedEvent?.id, focusedEvent?.representative_event_id, mobileView]);

  useEffect(() => {
    const eventIds = new URLSearchParams(window.location.search).get("event_ids");
    if (!eventIds) return;
    const query = new URLSearchParams({ event_ids: eventIds, gap_seconds: "45" });
    fetch(`/api/incidents/detail?${query}`)
      .then((response) => response.ok ? response.json() : Promise.reject(new Error("Linked incident unavailable")))
      .then((detail) => {
        const requestedEventId = Number(String(eventIds).split(",")[0]);
        if (mobileView) {
          setSelectedEvent(detail);
          return;
        }
        setLinkedIncidentDetail(detail);
        setLinkedIncidentEventId(Number.isInteger(requestedEventId) ? requestedEventId : Number(detail.representative_event_id) || null);
        setExpandedIncidentId(detail.id);
        setFocusedFaceEventId(Number.isInteger(requestedEventId) ? requestedEventId : Number(detail.representative_event_id) || null);
        const linkedEpoch = new Date(detail.start_at || detail.created_at || 0).getTime();
        if (Number.isFinite(linkedEpoch) && linkedEpoch > 0) setIncidentDay(dateKeyForTimeZone(linkedEpoch, timeZone));
        setEventFilter(linkedIncidentEventFilter(detail));
      })
      .catch(() => { });
  }, [mobileView, timeZone]);

  useEffect(() => {
    clearLegacyIncidentFilterStorage();
  }, []);

  useEffect(() => {
    if (mobileView) return undefined;
    const rail = incidentRailListRef.current;
    if (!rail) return undefined;
    function updateRailSize() {
      const rect = rail.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) setIncidentRailSize({ width: rect.width, height: rect.height });
    }
    updateRailSize();
    const observer = new ResizeObserver(updateRailSize);
    observer.observe(rail);
    return () => observer.disconnect();
  }, [mobileView]);

  function refresh() {
    refreshBase();
    incidentDetailCacheRef.current.clear();
    setIncidentDetails({});
    setIncidentRefreshToken((value) => value + 1);
  }

  async function runSemanticIncidentSearch(event, requestedQuery = semanticIncidentQuery) {
    event?.preventDefault();
    const queryText = String(requestedQuery || "").trim();
    if (!queryText) return;
    semanticIncidentRequestRef.current?.abort();
    const controller = new AbortController();
    semanticIncidentRequestRef.current = controller;
    setSemanticIncidentQuery(queryText);
    setSemanticIncidentActiveQuery(queryText);
    setSemanticIncidentResults([]);
    setSemanticIncidentLoading(true);
    setSemanticIncidentError("");
    setIncidentPage(0);
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
      setSemanticIncidentResults(rankSemanticIncidentDetails(hydrated, incidentZoneFilter));
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
  }

  useEffect(() => () => semanticIncidentRequestRef.current?.abort(), []);

  useEffect(() => {
    if (!semanticIncidentActiveQuery) return;
    runSemanticIncidentSearch(null, semanticIncidentActiveQuery);
  }, [incidentCameraFilter, incidentDay, incidentObjectFilter, incidentZoneFilter]);

  useAppEvents(({ type }) => {
    if (type !== "incident" || incidentDay !== today || incidentPage !== 0 || document.hidden) return;
    if (incidentEventRefreshTimer.current) return;
    incidentEventRefreshTimer.current = window.setTimeout(
      () => {
        incidentEventRefreshTimer.current = null;
        if (focusedDetailQuery) {
          incidentDetailCacheRef.current.invalidate(focusedDetailQuery);
          incidentDetailCacheRef.current.load(focusedDetailQuery).then((detail) => {
            setIncidentDetails((current) => ({ ...current, [focusedDetailQuery]: detail }));
          }).catch(() => {
            // Keep the existing detail visible; the next event or fallback poll retries.
          });
        }
        setIncidentRefreshToken((value) => value + 1);
      },
      1000,
    );
  });

  useEffect(() => () => window.clearTimeout(incidentEventRefreshTimer.current), []);

  useVisiblePolling(
    () => setIncidentRefreshToken((value) => value + 1),
    INCIDENT_REFRESH_FALLBACK_MS,
    incidentDay === today && incidentPage === 0,
    { immediate: false },
  );

  async function openFaceReview(face) {
    const observationId = Number(face?.observation_id);
    if (!Number.isFinite(observationId)) return;
    try {
      const [observationResponse, peopleResponse] = await Promise.all([
        fetch(`/api/faces/observations/${observationId}`),
        fetch("/api/faces/people"),
      ]);
      if (!observationResponse.ok || !peopleResponse.ok) return;
      setFacePeople(await peopleResponse.json());
      setSelectedFace(await observationResponse.json());
    } catch {
      // Leave the current incident visible when face details are unavailable.
    }
  }

  useEffect(() => {
    if (incidentDay > today) setIncidentDay(today);
  }, [incidentDay, setIncidentDay, today]);

  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/faces/people", { signal: controller.signal })
      .then((response) => response.ok ? response.json() : [])
      .then((people) => {
        if (!controller.signal.aborted) setIncidentPeople(Array.isArray(people) ? people : []);
      })
      .catch(() => {
        if (!controller.signal.aborted) setIncidentPeople([]);
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!incidentRailReady) return undefined;
    let cancelled = false;
    async function loadIncidentPage() {
      const query = new URLSearchParams({
        day: incidentDay || today,
        time_zone: timeZone,
        event_type: eventFilter,
        limit: String(incidentsPerPage),
        offset: String(incidentPage * incidentsPerPage),
        gap_seconds: "45",
      });
      if (incidentCameraFilter !== "all") query.set("camera_id", incidentCameraFilter);
      if (incidentObjectFilter !== "all") query.set("object_label", incidentObjectFilter);
      if (incidentZoneFilter !== "all") query.set("zone", incidentZoneFilter);
      if (incidentPersonFilter !== "all") query.set("person_id", incidentPersonFilter);
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
        setIncidents(items);
        setIncidentTotal(Number(payload.total || 0));
        setIncidentFacets(payload.facets || { camera_ids: [], labels: [], zones: [] });
        setIncidentLoadError("");
        if (!mobileView && items.length) {
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
  }, [incidentDay, today, timeZone, eventFilter, incidentCameraFilter, incidentObjectFilter, incidentZoneFilter, incidentPersonFilter, incidentPage, incidentsPerPage, incidentRefreshToken, incidentRailReady]);

  useEffect(() => {
    setIncidentPage(0);
  }, [eventFilter, incidentCameraFilter, incidentObjectFilter, incidentZoneFilter, incidentPersonFilter, incidentDay, incidentDensity]);

  useEffect(() => {
    const previousPageSize = previousIncidentsPerPageRef.current;
    if (previousPageSize !== incidentsPerPage) {
      setIncidentPage((page) => Math.floor(page * previousPageSize / incidentsPerPage));
      previousIncidentsPerPageRef.current = incidentsPerPage;
    }
  }, [incidentsPerPage]);

  useEffect(() => {
    if (incidentPage >= incidentPageCount) setIncidentPage(Math.max(0, incidentPageCount - 1));
  }, [incidentPage, incidentPageCount]);

  useEffect(() => {
    setFocusedFaceEventId(
      linkedIncidentDetail?.id === focusedIncident?.id ? linkedIncidentEventId : null,
    );
    setDesktopAnalysisStats(null);
    relatedPreviewRequestRef.current += 1;
    setRelatedPreviewIncident(null);
    setRelatedPreviewEventId(null);
    setRelatedPreviewLoadingEventId(null);
    setSelectedVisualObject(null);
    setTabletInspectorOpen(false);
  }, [focusedIncident?.id, linkedIncidentDetail?.id, linkedIncidentEventId]);

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
    if (desktopAnalysisMode !== "tracks") return;
    const trackingEvent = incidentTrackingSource(displayedEvent, displayedIncident);
    if (!storedObjectTracks(trackingEvent).length) setDesktopAnalysisMode("clean");
  }, [desktopAnalysisMode, displayedEvent, displayedIncident, setDesktopAnalysisMode]);

  useEffect(() => {
    const context = incidentRecordingContext(selectedEvent || focusedEvent);
    if (context) onRecordingContextChange(context);
  }, [selectedEvent?.id, focusedEvent?.id, focusedEvent?.created_at, focusedEvent?.camera_id, onRecordingContextChange]);

  useEffect(() => {
    if (expandedIncidentId
      && linkedIncidentDetail?.id !== expandedIncidentId
      && !visibleIncidents.some((incident) => incident.id === expandedIncidentId)) {
      setExpandedIncidentId(null);
    }
  }, [expandedIncidentId, linkedIncidentDetail?.id, visibleIncidents]);

  useEffect(() => {
    function onKey(keyEvent) {
      if (keyEvent.key === "Escape" && expandedIncidentId && !selectedEvent) {
        keyEvent.preventDefault();
        setExpandedIncidentId(null);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expandedIncidentId, selectedEvent]);

  function toggleIncident(incidentId) {
    if (mobileView) {
      const incident = visibleIncidents.find((candidate) => candidate.id === incidentId);
      if (incident) openIncidentOverlay(incident);
      return;
    }
    if (incidentId !== linkedIncidentDetail?.id) {
      setLinkedIncidentDetail(null);
      setLinkedIncidentEventId(null);
    }
    setExpandedIncidentId(incidentId);
  }

  const focusedIndex = focusedIncident ? visibleIncidents.findIndex((incident) => incident.id === focusedIncident.id) : -1;

  function moveFocus(direction) {
    if (!visibleIncidents.length) return;
    const nextIndex = Math.max(0, Math.min(visibleIncidents.length - 1, focusedIndex + direction));
    setExpandedIncidentId(visibleIncidents[nextIndex].id);
  }

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
    try {
      const response = await fetch(`/api/incidents/by-event/${eventId}`);
      if (!response.ok) throw new Error("Related incident unavailable");
      const detail = await response.json();
      if (request !== relatedPreviewRequestRef.current) return;
      setRelatedPreviewIncident(detail);
      setRelatedPreviewEventId(eventId);
      setFocusedFaceEventId(eventId);
      setDesktopAnalysisMode("clean");
      setDesktopAnalysisStats(null);
    } catch {
      // Keep the currently displayed incident if a stale related event was removed.
    } finally {
      if (request === relatedPreviewRequestRef.current) setRelatedPreviewLoadingEventId(null);
    }
  }

  function returnToSelectedIncident() {
    relatedPreviewRequestRef.current += 1;
    setRelatedPreviewIncident(null);
    setRelatedPreviewEventId(null);
    setRelatedPreviewLoadingEventId(null);
    setFocusedFaceEventId(Number(focusedIncident?.representative_event_id || focusedIncident?.id) || null);
    setDesktopAnalysisMode("clean");
    setDesktopAnalysisStats(null);
  }

  useEffect(() => {
    if (mobileView || selectedEvent) return undefined;
    function onIncidentArrow(event) {
      const target = event.target;
      if (target instanceof HTMLElement && (target.isContentEditable || ["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName))) return;
      if (event.key === "ArrowLeft" && focusedIndex > 0) {
        event.preventDefault();
        moveFocus(-1);
      }
      if (event.key === "ArrowRight" && focusedIndex >= 0 && focusedIndex < visibleIncidents.length - 1) {
        event.preventDefault();
        moveFocus(1);
      }
    }
    window.addEventListener("keydown", onIncidentArrow);
    return () => window.removeEventListener("keydown", onIncidentArrow);
  }, [mobileView, selectedEvent, focusedIndex, visibleIncidents]);

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

  if (!mobileView) {
    return (
      <main className="incidents-desktop-page with-inspector">
        <section className="bento-card incidents-desktop-shell">
          <div className="incidents-desktop-toolbar">
            <div className="incidents-command-primary">
              <div className="incident-filter-toggle compact" role="group" aria-label="Incident type filter">
                <button className={eventFilter === "object" ? "active" : ""} aria-pressed={eventFilter === "object"} onClick={() => setEventFilter("object")}>Objects</button>
                <button className={eventFilter === "motion" ? "active" : ""} aria-pressed={eventFilter === "motion"} onClick={() => { resetSemanticIncidentSearch(); setEventFilter("motion"); }}>Motion</button>
              </div>
              <label className="incident-day-field"><input type="date" value={incidentDay} max={today} onChange={(event) => setIncidentDay(event.target.value || today)} aria-label="Incident day" /></label>
              <div className="incident-filter-selects desktop">
                <label><select value={incidentCameraFilter} onChange={(event) => setIncidentCameraFilter(event.target.value)} aria-label="Incident camera"><option value="all">All cameras</option>{incidentCameraOptions.map((id) => <option value={id} key={id}>{cameraNameById.get(id) || id}</option>)}</select></label>
                <label><select value={incidentObjectFilter} onChange={(event) => setIncidentObjectFilter(event.target.value)} aria-label="Incident object"><option value="all">All objects</option>{incidentObjectOptions.map((label) => <option value={label} key={label}>{label}</option>)}</select></label>
                <label><select value={incidentZoneFilter} onChange={(event) => setIncidentZoneFilter(event.target.value)} aria-label="Incident zone"><option value="all">All zones</option>{incidentZoneOptions.map((zone) => <option value={zone} key={zone}>{zone}</option>)}</select></label>
                <label><select value={incidentPersonFilter} onChange={(event) => { resetSemanticIncidentSearch(); setIncidentPersonFilter(event.target.value); }} aria-label="Known person"><option value="all">All people</option>{incidentPeople.map((person) => <option value={person.id} key={person.id}>{person.name}</option>)}</select></label>
              </div>
              {semanticIncidentControl}
              <div className="incident-toolbar-summary" aria-label={`${activeIncidentFilterCount} active filters`}>
                {activeIncidentFilterCount ? <button type="button" onClick={clearIncidentFilters}>Clear</button> : null}
                <span className="shown-bubble">{displayedIncidentTotal} {semanticIncidentActive ? "matches" : "shown"}</span>
              </div>
            </div>
          </div>

          <div className="incidents-desktop-workspace">
            <aside className={`incident-rail ${incidentDensity}`}>
              <div className="incident-rail-head">
                <strong>Incidents</strong>
                <div className="density-control" aria-label="Thumbnail density">
                  <button type="button" className={incidentDensity === "compact" ? "active" : ""} aria-pressed={incidentDensity === "compact"} onClick={() => setIncidentDensity("compact")} title="List view" aria-label="List view"><Rows3 size={15} /></button>
                  <button type="button" className={incidentDensity === "comfortable" ? "active" : ""} aria-pressed={incidentDensity === "comfortable"} onClick={() => setIncidentDensity("comfortable")} title="Grid view" aria-label="Grid view"><Grid2X2 size={15} /></button>
                </div>
              </div>
              <div className="incident-rail-list" ref={incidentRailListRef}>
                {displayedIncidentLoading && !galleryIncidents.length ? <div className="empty-state">{semanticIncidentActive ? "Searching indexed incidents..." : "Loading incidents..."}</div> : null}
                {!galleryIncidents.length && displayedIncidentError ? <div className="empty-state">{displayedIncidentError}</div> : null}
                {galleryIncidents.length ? pagedIncidents.map((incident) => (
                  <IncidentListItem key={incident.id} incident={incident} cameraName={cameraNameById.get(incident.camera_id) || incident.camera_id} timeZone={timeZone} selected={incident.id === focusedIncident?.id} thumbnailAnnotations={thumbnailAnnotations} thumbnailObjectFocus={thumbnailObjectFocus} thumbnailObjectFocusZoom={thumbnailObjectFocusZoom} onSelect={(selectedIncident) => toggleIncident(selectedIncident.id)} onOpenOverlay={openIncidentOverlay} />
                )) : null}
                {!displayedIncidentLoading && !displayedIncidentError && !galleryIncidents.length ? <div className="empty-state">{semanticIncidentActive ? "No semantic matches for the selected filters." : "No other incidents."}</div> : null}
              </div>
              <div className={`incident-pager ${displayedIncidentTotal > incidentsPerPage ? "" : "placeholder"}`} aria-label="Incident pages" aria-hidden={displayedIncidentTotal <= incidentsPerPage}>
                <button type="button" onClick={() => setIncidentPage((page) => Math.max(0, page - 1))} disabled={clampedIncidentPage === 0}>Prev</button>
                <span>{clampedIncidentPage + 1} / {incidentPageCount}</span>
                <button type="button" onClick={() => setIncidentPage((page) => Math.min(incidentPageCount - 1, page + 1))} disabled={clampedIncidentPage >= incidentPageCount - 1}>Next</button>
              </div>
            </aside>

            <section className="incident-investigation">
              <div className="incident-desktop-focus">
                <div className="incident-focus-actions">
                  {relatedPreviewIncident ? <button type="button" onClick={returnToSelectedIncident}>Return to selected incident</button> : null}
                  <button ref={tabletInspectorToggleRef} type="button" className="incident-inspector-toggle" onClick={() => setTabletInspectorOpen((open) => !open)} aria-expanded={tabletInspectorOpen} aria-controls="incident-inspector" disabled={!displayedIncident}>Details</button>
                </div>
                {focusedIncident ? (
                  <>
                    <button type="button" className="incident-focus-arrow previous" onClick={() => moveFocus(-1)} disabled={focusedIndex <= 0} title="Previous incident" aria-label="Previous incident"><ChevronLeft size={26} /></button>
                    <button type="button" className="incident-focus-arrow next" onClick={() => moveFocus(1)} disabled={focusedIndex < 0 || focusedIndex >= visibleIncidents.length - 1} title="Next incident" aria-label="Next incident"><ChevronRight size={26} /></button>
                  </>
                ) : null}
                {displayedIncident ? <IncidentCard key={`${focusedIncident?.id || "none"}:${displayedIncident.id || displayedIncident.representative_event_id}`} incident={displayedIncident} timeZone={timeZone} expanded thumbnailAnnotations={thumbnailAnnotations} thumbnailObjectFocus={thumbnailObjectFocus} thumbnailObjectFocusZoom={thumbnailObjectFocusZoom} desktopWorkspace analysisMode={desktopAnalysisMode} replayRequest={desktopReplayRequest} selectedObjectIndex={selectedVisualObject?.objectIndex ?? null} onSelectObject={(selection) => { setSelectedVisualObject(selection); if (selection) setTabletInspectorOpen(true); }} onAnalysisStats={setDesktopAnalysisStats} onToggle={toggleIncident} onPreviewChange={setFocusedFaceEventId} onImageSize={setFocusedImageSize} /> : <div className="empty-state">No incidents match the current filters.</div>}
              </div>
            </section>

            {tabletInspectorOpen ? <button type="button" className="incident-inspector-backdrop" onClick={() => closeTabletInspector()} aria-label="Close incident details" /> : null}
            <IncidentInspector open={tabletInspectorOpen} incident={displayedIncident} faceEvent={displayedEvent} anchorEventId={relatedAnchorEventId} selectedRelatedEventId={relatedPreviewEventId} relatedLoadingEventId={relatedPreviewLoadingEventId} cameraNameById={cameraNameById} appConfig={appConfig} timeZone={timeZone} imageSize={focusedLoadedImageSize} analysisMode={desktopAnalysisMode} analysisStats={desktopAnalysisStats} selectedObjectIndex={selectedVisualObject?.objectIndex ?? null} onSelectObject={setSelectedVisualObject} onAnalysisModeChange={selectDesktopAnalysisMode} onFaceOpen={openFaceReview} onRelatedSelect={selectRelatedIncident} onRelatedReturn={returnToSelectedIncident} onClose={() => closeTabletInspector()} onAskAssistant={onAskAssistant} />
          </div>
        </section>
        {selectedFace ? <FaceReviewDialog observation={selectedFace} people={facePeople} timeZone={timeZone} onClose={() => setSelectedFace(null)} onUpdated={() => { setSelectedFace(null); refresh(); }} /> : null}
      </main>
    );
  }

  return (
    <main className="bento-grid incidents-grid">
      <section className="bento-card events-zone incidents-page-zone">
        <div className="section-head compact incident-head">
          <div><h2>Incidents</h2></div>
          <div className="incident-head-actions">
            <div className="incident-filter-toggle compact" aria-label="Incident type filter">
              <button className={eventFilter === "object" ? "active" : ""} aria-pressed={eventFilter === "object"} onClick={() => setEventFilter("object")}>Object</button>
              <button className={eventFilter === "motion" ? "active" : ""} aria-pressed={eventFilter === "motion"} onClick={() => { resetSemanticIncidentSearch(); setEventFilter("motion"); }}>Motion</button>
            </div>
            <span className="shown-bubble">{displayedIncidentTotal} {semanticIncidentActive ? "matches" : "shown"}</span>
          </div>
        </div>
        <div className="event-filter incident-filter-panel" aria-label="Incident filters">
          <div className="incident-filter-selects">
            <label>
              <span>Day</span>
              <input type="date" value={incidentDay} max={today} onChange={(event) => setIncidentDay(event.target.value || today)} aria-label="Incident day" />
            </label>
            <label>
              <select value={incidentCameraFilter} onChange={(event) => setIncidentCameraFilter(event.target.value)} aria-label="Incident camera">
                <option value="all">All cameras</option>
                {incidentCameraOptions.map((id) => <option value={id} key={id}>{cameraNameById.get(id) || id}</option>)}
              </select>
            </label>
            <label>
              <select value={incidentObjectFilter} onChange={(event) => setIncidentObjectFilter(event.target.value)} aria-label="Incident object">
                <option value="all">All objects</option>
                {incidentObjectOptions.map((label) => <option value={label} key={label}>{label}</option>)}
              </select>
            </label>
            <label>
              <select value={incidentZoneFilter} onChange={(event) => setIncidentZoneFilter(event.target.value)} aria-label="Incident zone">
                <option value="all">All zones</option>
                {incidentZoneOptions.map((zone) => <option value={zone} key={zone}>{zone}</option>)}
              </select>
            </label>
            <label>
              <select value={incidentPersonFilter} onChange={(event) => { resetSemanticIncidentSearch(); setIncidentPersonFilter(event.target.value); }} aria-label="Known person">
                <option value="all">All people</option>
                {incidentPeople.map((person) => <option value={person.id} key={person.id}>{person.name}</option>)}
              </select>
            </label>
          </div>
          {semanticIncidentControl}
        </div>
        <div className="incident-gallery">
          {displayedIncidentLoading ? <div className="empty-state">{semanticIncidentActive ? "Searching indexed incidents..." : "Loading incidents..."}</div> : null}
          {!displayedIncidentLoading && displayedIncidentError ? <div className="empty-state">{displayedIncidentError}</div> : null}
          {!displayedIncidentLoading && !displayedIncidentError && visibleIncidents.length
            ? pagedIncidents.map((incident) => (
              <IncidentCard
                key={incident.id}
                incident={incident}
                timeZone={timeZone}
                expanded={false}
                thumbnailAnnotations={thumbnailAnnotations}
                thumbnailObjectFocus={thumbnailObjectFocus}
                thumbnailObjectFocusZoom={thumbnailObjectFocusZoom}
                onToggle={toggleIncident}
                onSelect={openIncidentOverlay}
              />
            ))
            : null}
          {!displayedIncidentLoading && !displayedIncidentError && !visibleIncidents.length ? <div className="empty-state">{semanticIncidentActive ? "No semantic matches for the selected filters." : "No incidents match the current filters."}</div> : null}
        </div>
        {displayedIncidentTotal > incidentsPerPage ? (
          <div className="incident-pager" aria-label="Incident pages">
            <button type="button" onClick={() => setIncidentPage((page) => Math.max(0, page - 1))} disabled={clampedIncidentPage === 0}>Prev</button>
            <span>{clampedIncidentPage + 1} / {incidentPageCount}</span>
            <button type="button" onClick={() => setIncidentPage((page) => Math.min(incidentPageCount - 1, page + 1))} disabled={clampedIncidentPage >= incidentPageCount - 1}>Next</button>
          </div>
        ) : null}
      </section>
      {selectedEvent ? <EventOverlay event={selectedEvent} events={visibleIncidents} timeZone={timeZone} onClose={closeIncidentOverlay} onSelect={openIncidentOverlay} onRefresh={refresh} /> : null}
    </main>
  );
}
