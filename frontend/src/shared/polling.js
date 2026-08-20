import { useEffect, useMemo, useRef, useState } from "react";
import { useVisiblePolling } from "../visibilityPolling.mjs";
import { camerasWithLiveFraming } from "../liveFraming.mjs";
import { createIncidentPageCache, incidentDetailQuery } from "../incidentNavigation.mjs";
import { fetch } from "./api.js";
import { useAppEvents } from "./events.js";

export function usePollingData() {
  const [cameras, setCameras] = useState([]);
  const [appConfig, setAppConfig] = useState(null);
  const [loading, setLoading] = useState(true);
  const loadSequence = useRef(0);

  async function load() {
    const sequence = ++loadSequence.current;
    try {
      const [cameraResponse, configResponse] = await Promise.all([
        fetch("/api/cameras"),
        fetch("/api/config"),
      ]);
      if (!cameraResponse.ok) throw new Error(`Camera status failed (${cameraResponse.status})`);
      const cameraPayload = await cameraResponse.json();
      const configPayload = configResponse.ok ? await configResponse.json() : null;
      if (sequence !== loadSequence.current) return;
      if (Array.isArray(cameraPayload)) setCameras(cameraPayload);
      if (configPayload) setAppConfig(configPayload);
    } catch {
      // SSE updates may still populate the page; periodic polling retries.
    } finally {
      if (sequence === loadSequence.current) setLoading(false);
    }
  }

  useAppEvents(({ type, data }) => {
    if (type === "cameras_state" && Array.isArray(data)) {
      setCameras(data);
      setLoading(false);
    } else if (type === "camera_state" && data?.id) {
      setCameras((current) => {
        const index = current.findIndex((camera) => camera.id === data.id);
        if (index < 0) return [...current, data];
        const next = [...current];
        next[index] = data;
        return next;
      });
    }
  });

  useVisiblePolling(load, 60_000);
  useEffect(() => () => { loadSequence.current += 1; }, []);

  const camerasWithPresentation = useMemo(
    () => camerasWithLiveFraming(cameras, appConfig?.cameras),
    [appConfig, cameras],
  );

  return { cameras: camerasWithPresentation, appConfig, loading, refresh: load };
}

export function useIncidentDetails() {
  const incidentDetailCacheRef = useRef(null);
  if (!incidentDetailCacheRef.current) {
    incidentDetailCacheRef.current = createIncidentPageCache(async (query) => {
      const response = await fetch(`/api/incidents/detail?${query}`);
      if (!response.ok) throw new Error("Unable to load incident details");
      return response.json();
    });
  }
  const [incidentDetails, setIncidentDetails] = useState({});
  const incidentSelectionRequestRef = useRef(0);
  const [selectedEvent, setSelectedEvent] = useState(null);

  async function loadIncidentDetail(incident) {
    const query = incidentDetailQuery(incident);
    if (!query) return incident;
    const cached = incidentDetails[query];
    if (cached) return cached;
    try {
      const detail = await incidentDetailCacheRef.current.load(query);
      setIncidentDetails((current) => ({ ...current, [query]: detail }));
      return detail;
    } catch {
      return incident;
    }
  }

  async function openIncidentOverlay(incident) {
    const request = ++incidentSelectionRequestRef.current;
    setSelectedEvent(incident);
    const detail = await loadIncidentDetail(incident);
    if (request === incidentSelectionRequestRef.current) setSelectedEvent(detail);
  }

  function closeIncidentOverlay() {
    incidentSelectionRequestRef.current += 1;
    setSelectedEvent(null);
  }

  return {
    incidentDetailCacheRef,
    incidentDetails,
    setIncidentDetails,
    incidentSelectionRequestRef,
    selectedEvent,
    setSelectedEvent,
    openIncidentOverlay,
    closeIncidentOverlay,
  };
}
