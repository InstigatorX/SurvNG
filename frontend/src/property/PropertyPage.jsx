import React, { useEffect, useMemo, useRef, useState } from "react";
import { Sparkles, Undo2 } from "lucide-react";
import { appUrl, fetch } from "../shared/api.js";
import { useAppEvents } from "../shared/events.js";
import { useRuntimeState } from "../shared/runtimeState.jsx";
import { PropertyCanvas } from "./PropertyCanvas.jsx";
import {
  applyDetection,
  configuredZoneRows,
  detectionIcon,
  highlightOpacity,
  overlayPresentation,
  propertyLocationFields,
  pruneHighlights,
  recordZonePointInsertion,
  undoZonePointInsertion,
  zoneActivityKey,
} from "./propertyView.mjs";

function numberOrBlank(value) {
  return value === null || value === undefined || value === "" ? "" : String(value);
}

export function PropertyPage({ canEdit = false }) {
  const runtime = useRuntimeState();
  const appConfig = runtime?.appConfig;
  const refresh = runtime?.refresh;
  const image = appConfig?.property?.image || null;
  const configuredZones = useMemo(() => configuredZoneRows(appConfig), [appConfig]);
  const cameras = appConfig?.cameras || [];
  const [layout, setLayout] = useState({ cameras: [], zones: [] });
  const [address, setAddress] = useState("");
  const [latitude, setLatitude] = useState("");
  const [longitude, setLongitude] = useState("");
  const [widthM, setWidthM] = useState("120");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [warning, setWarning] = useState("");
  const [candidates, setCandidates] = useState([]);
  const [highlights, setHighlights] = useState({});
  const [now, setNow] = useState(() => Date.now());
  const [overrides, setOverrides] = useState({});
  const [selected, setSelected] = useState(null);
  const [showCameras, setShowCameras] = useState(true);
  const [showZones, setShowZones] = useState(true);
  const dirtyRef = useRef(false);
  const layoutRef = useRef(layout);
  const prefilledRef = useRef(false);
  const locationSyncRef = useRef("");
  const [pointHistory, setPointHistory] = useState({});
  const [proposal, setProposal] = useState(null);
  const [conformingKey, setConformingKey] = useState("");
  const proposalRef = useRef(null);
  const proposalDragRef = useRef(false);
  const serverLayout = JSON.stringify(appConfig?.property ? {
    cameras: appConfig.property.cameras || [],
    zones: appConfig.property.zones || [],
  } : null);

  useEffect(() => {
    if (dirtyRef.current) return;
    const parsed = serverLayout ? JSON.parse(serverLayout) : { cameras: [], zones: [] };
    layoutRef.current = parsed || { cameras: [], zones: [] };
    setLayout(layoutRef.current);
  }, [serverLayout]);

  const locationFields = propertyLocationFields(appConfig);
  const locationSignature = `${locationFields.source}|${locationFields.latitude}|${locationFields.longitude}|${locationFields.address}|${locationFields.widthM}`;
  useEffect(() => {
    if (!locationFields.source) return undefined;
    if (locationFields.source === "weather") {
      if (prefilledRef.current) return undefined;
      prefilledRef.current = true;
    } else if (locationSyncRef.current === locationSignature) {
      return undefined;
    }
    const active = document.activeElement;
    if (active && active.closest("[data-property-location]")) return undefined;
    locationSyncRef.current = locationSignature;
    setLatitude(numberOrBlank(locationFields.latitude));
    setLongitude(numberOrBlank(locationFields.longitude));
    if (locationFields.source !== "weather") {
      setAddress(locationFields.address || "");
      setWidthM(numberOrBlank(locationFields.widthM) || "120");
    }
    return undefined;
  }, [locationFields, locationSignature]);

  useEffect(() => {
    let cancelled = false;
    async function loadActivity() {
      try {
        const response = await fetch("/api/property/activity");
        if (!response.ok) return;
        const payload = await response.json();
        if (cancelled) return;
        const stamped = Date.now();
        setHighlights((current) => (payload.events || []).reduce(
          (next, event) => applyDetection(next, event, stamped),
          current,
        ));
        setNow(stamped);
      } catch {
        // Live events still update the view if the seed request fails.
      }
    }
    void loadActivity();
    return () => {
      cancelled = true;
    };
  }, [image?.fetched_at]);

  useAppEvents(({ type, data }) => {
    if (type === "resync") {
      fetch("/api/property/activity")
        .then((response) => (response.ok ? response.json() : null))
        .then((payload) => {
          if (!payload) return;
          const stamped = Date.now();
          setHighlights((payload.events || []).reduce(
            (next, event) => applyDetection(next, event, stamped),
            {},
          ));
          setNow(stamped);
        })
        .catch(() => {});
      return;
    }
    if (type !== "object") return;
    const stamped = Date.now();
    setHighlights((current) => applyDetection(current, data, stamped));
    setNow(stamped);
  });

  const highlightCount = Object.keys(highlights).length;
  useEffect(() => {
    if (!highlightCount) return undefined;
    const timer = window.setInterval(() => {
      const stamped = Date.now();
      setNow(stamped);
      setHighlights((current) => {
        const next = pruneHighlights(current, stamped);
        const same = Object.keys(current).length === Object.keys(next).length
          && Object.entries(next).every(([key, value]) => current[key]?.detectedAt === value.detectedAt);
        return same ? current : next;
      });
    }, 200);
    return () => window.clearInterval(timer);
  }, [highlightCount]);

  async function saveLayout(nextLayout) {
    if (!canEdit) return;
    dirtyRef.current = true;
    const response = await fetch("/api/property/layout", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cameras: nextLayout.cameras, zones: nextLayout.zones }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || "Could not save the property layout");
    }
    dirtyRef.current = false;
    if (typeof refresh === "function") await refresh();
  }

  function rememberLayout(next) {
    layoutRef.current = next;
    dirtyRef.current = true;
    setLayout(next);
    return next;
  }

  function updateLayout(recipe) {
    setLayout((current) => {
      const next = recipe(current);
      layoutRef.current = next;
      dirtyRef.current = true;
      return next;
    });
  }

  async function commitLayout() {
    if (proposalDragRef.current) {
      proposalDragRef.current = false;
      return;
    }
    try {
      await saveLayout(layoutRef.current);
      setError("");
    } catch (saveError) {
      setError(saveError.message);
    }
  }

  function changeCamera(cameraId, patch) {
    updateLayout((current) => ({
      ...current,
      cameras: current.cameras.map((camera) => (
        camera.camera_id === cameraId ? { ...camera, ...patch } : camera
      )),
    }));
  }

  function changeZone(key, patch) {
    if (proposalRef.current?.key === key && Object.hasOwn(patch, "points")) {
      proposalDragRef.current = true;
      const next = { key, points: patch.points };
      proposalRef.current = next;
      setProposal(next);
      return;
    }
    updateLayout((current) => ({
      ...current,
      zones: current.zones.map((zone) => (
        zoneActivityKey(zone.camera_id, zone.zone_name) === key ? { ...zone, ...patch } : zone
      )),
    }));
  }

  function clearProposal() {
    proposalRef.current = null;
    setProposal(null);
  }

  function selectItem(next) {
    setSelected((current) => {
      if (current?.kind !== next?.kind || current?.id !== next?.id) clearProposal();
      return next;
    });
  }

  async function downloadImage(event) {
    event.preventDefault();
    if (!canEdit) return;
    setBusy(true);
    setError("");
    try {
      const lat = latitude.trim() === "" ? null : Number(latitude);
      const lon = longitude.trim() === "" ? null : Number(longitude);
      const body = {
        address: address.trim(),
        width_m: Number(widthM) || 120,
        height_m: Number(widthM) || 120,
      };
      if (lat !== null || lon !== null) {
        body.latitude = lat;
        body.longitude = lon;
      }
      const response = await fetch("/api/property/imagery", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "Could not download an aerial image");
      setCandidates(payload.candidates || []);
      setWarning(payload.placement_warning
        ? "The photo moved or changed size. Camera and zone placements were kept and may need to be moved."
        : "");
      if (typeof refresh === "function") await refresh();
    } catch (downloadError) {
      setError(downloadError.message);
    } finally {
      setBusy(false);
    }
  }

  async function lookupFov(cameraId, make, model) {
    if (!model.trim()) return;
    const response = await fetch(`/api/property/fov?make=${encodeURIComponent(make)}&model=${encodeURIComponent(model)}`);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || !payload.matched) {
      setError(payload.reason || "No field of view is listed for that make and model. Enter the angle yourself.");
      return;
    }
    setError("");
    const next = rememberLayout({
      ...layoutRef.current,
      cameras: layoutRef.current.cameras.map((camera) => (
        camera.camera_id === cameraId
          ? { ...camera, make: payload.make, model: payload.model, hfov_deg: payload.hfov_deg }
          : camera
      )),
    });
    try {
      await saveLayout(next);
    } catch (saveError) {
      setError(saveError.message);
    }
  }

  async function toggleCameraNotifications(cameraId, enabled) {
    const key = `camera:${cameraId}`;
    setOverrides((current) => ({ ...current, [key]: enabled }));
    const response = await fetch(`/api/cameras/${encodeURIComponent(cameraId)}/incident-notifications`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled }),
    });
    if (!response.ok) {
      setOverrides((current) => {
        const next = { ...current };
        delete next[key];
        return next;
      });
      setError("Could not update camera notifications");
      return;
    }
    if (typeof refresh === "function") await refresh();
  }

  async function toggleZoneNotifications(cameraId, zoneName, enabled) {
    const key = `zone:${cameraId}:${zoneName}`;
    setOverrides((current) => ({ ...current, [key]: enabled }));
    const response = await fetch(`/api/cameras/${encodeURIComponent(cameraId)}/zone-notifications`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ zone: zoneName, enabled }),
    });
    if (!response.ok) {
      setOverrides((current) => {
        const next = { ...current };
        delete next[key];
        return next;
      });
      setError("Could not update zone notifications");
      return;
    }
    if (typeof refresh === "function") await refresh();
  }

  function placeCamera(camera) {
    if (layoutRef.current.cameras.some((item) => item.camera_id === camera.id)) return;
    const next = rememberLayout({
      ...layoutRef.current,
      cameras: [...layoutRef.current.cameras, {
        camera_id: camera.id,
        make: "",
        model: "",
        hfov_deg: 80,
        x: 0.5,
        y: 0.5,
        heading_deg: 0,
        range_m: 20,
      }],
    });
    setSelected({ kind: "camera", id: camera.id });
    void saveLayout(next).catch((saveError) => setError(saveError.message));
  }

  function placeZone(zone) {
    if (layoutRef.current.zones.some((item) => item.camera_id === zone.cameraId && item.zone_name === zone.zoneName)) return;
    const next = rememberLayout({
      ...layoutRef.current,
      zones: [...layoutRef.current.zones, {
        camera_id: zone.cameraId,
        zone_name: zone.zoneName,
        points: [],
      }],
    });
    setSelected({ kind: "zone", id: zoneActivityKey(zone.cameraId, zone.zoneName) });
    void saveLayout(next).catch((saveError) => setError(saveError.message));
  }

  function replaceZonePoints(key, points) {
    const next = {
      ...layoutRef.current,
      zones: layoutRef.current.zones.map((zone) => (
        zoneActivityKey(zone.camera_id, zone.zone_name) === key ? { ...zone, points } : zone
      )),
    };
    rememberLayout(next);
    return next;
  }

  function addZonePoint(key, points, insertionIndex) {
    setPointHistory((current) => ({
      ...current,
      [key]: recordZonePointInsertion(current[key], insertionIndex),
    }));
    const next = replaceZonePoints(key, points);
    void saveLayout(next).catch((saveError) => setError(saveError.message));
  }

  function undoZonePoint(key) {
    const zone = layoutRef.current.zones.find((item) => zoneActivityKey(item.camera_id, item.zone_name) === key);
    const undone = undoZonePointInsertion(zone?.points || [], pointHistory[key]);
    setPointHistory((current) => ({ ...current, [key]: undone.history }));
    if (!zone || undone.points === zone.points) return;
    const next = replaceZonePoints(key, undone.points);
    void saveLayout(next).catch((saveError) => setError(saveError.message));
  }

  async function conformZone(cameraId, zoneName) {
    const key = zoneActivityKey(cameraId, zoneName);
    const zone = layoutRef.current.zones.find((item) => zoneActivityKey(item.camera_id, item.zone_name) === key);
    const points = zone?.points || [];
    if (points.length < 3) {
      setError("Add at least three points before conforming this zone.");
      return;
    }
    setConformingKey(key);
    setError("");
    try {
      const response = await fetch("/api/property/zones/conform", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ camera_id: cameraId, zone_name: zoneName, points }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(typeof payload.detail === "string" ? payload.detail : "Could not conform this zone");
      }
      const next = { key, points: payload.points || [] };
      proposalRef.current = next;
      setProposal(next);
    } catch (conformError) {
      setError(conformError.message);
    } finally {
      setConformingKey("");
    }
  }

  async function acceptProposal() {
    const pending = proposalRef.current;
    if (!pending) return;
    const previous = layoutRef.current;
    const next = {
      ...previous,
      zones: previous.zones.map((zone) => (
        zoneActivityKey(zone.camera_id, zone.zone_name) === pending.key
          ? { ...zone, points: pending.points.map((point) => ({ x: point.x, y: point.y })) }
          : zone
      )),
    };
    rememberLayout(next);
    try {
      await saveLayout(next);
    } catch (saveError) {
      rememberLayout(previous);
      setError(saveError.message);
      return;
    }
    setPointHistory((current) => ({ ...current, [pending.key]: [] }));
    clearProposal();
    setError("");
  }

  async function saveLocation() {
    if (!canEdit) return;
    const lat = latitude.trim() === "" ? null : Number(latitude);
    const lon = longitude.trim() === "" ? null : Number(longitude);
    if (lat === null || lon === null || !Number.isFinite(lat) || !Number.isFinite(lon)) return;
    const saved = propertyLocationFields(appConfig);
    const width = Number(widthM) || 120;
    if (
      saved.source !== "weather"
      && saved.source !== ""
      && Number(saved.latitude) === lat
      && Number(saved.longitude) === lon
      && (saved.address || "") === address.trim()
      && Number(saved.widthM) === width
    ) return;
    const response = await fetch("/api/property/location", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        address: address.trim(),
        latitude: lat,
        longitude: lon,
        width_m: width,
        height_m: width,
      }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      setError(payload.detail || "Could not save latitude and longitude");
      return;
    }
    setError("");
    if (typeof refresh === "function") await refresh();
  }

  function chooseSelection(next) {
    clearProposal();
    setSelected((current) => (
      current?.kind === next.kind && current.id === next.id ? null : next
    ));
  }

  function removeCamera(cameraId) {
    setSelected((current) => (current?.kind === "camera" && current.id === cameraId ? null : current));
    const next = rememberLayout({
      cameras: layoutRef.current.cameras.filter((camera) => camera.camera_id !== cameraId),
      zones: layoutRef.current.zones.filter((zone) => zone.camera_id !== cameraId),
    });
    void saveLayout(next).catch((saveError) => setError(saveError.message));
  }

  function removeZone(cameraId, zoneName) {
    const key = zoneActivityKey(cameraId, zoneName);
    if (proposalRef.current?.key === key) clearProposal();
    setSelected((current) => (current?.kind === "zone" && current.id === key ? null : current));
    const next = rememberLayout({
      ...layoutRef.current,
      zones: layoutRef.current.zones.filter((zone) => !(zone.camera_id === cameraId && zone.zone_name === zoneName)),
    });
    void saveLayout(next).catch((saveError) => setError(saveError.message));
  }

  const canvasZones = layout.zones.map((zone) => {
    const key = zoneActivityKey(zone.camera_id, zone.zone_name);
    const presentation = overlayPresentation(zone, configuredZones);
    const activity = highlights[key];
    const opacity = activity ? highlightOpacity(activity.detectedAt, now) : 0;
    const icons = [...new Set((activity?.classes || []).map(detectionIcon))];
    const preview = proposal?.key === key;
    return {
      ...zone,
      key,
      points: preview ? proposal.points : zone.points,
      preview,
      ...presentation,
      opacity,
      icons,
    };
  });
  const imageUrl = image ? appUrl(`/api/property/imagery?v=${encodeURIComponent(image.fetched_at || "")}`) : "";
  const groundWidth = Number(image?.width_m) || 120;
  const groundHeight = Number(image?.height_m) || groundWidth;

  return (
    <main className="property-page">
      <div className="property-stage-wrap">
        {warning ? <p className="property-banner" role="status">{warning}</p> : null}
        {error ? <p className="property-error" role="alert">{error}</p> : null}
        {image ? (
          <>
            <PropertyCanvas
              imageUrl={imageUrl}
              widthM={groundWidth}
              heightM={groundHeight}
              cameras={layout.cameras.map((camera) => ({
                ...camera,
                name: cameras.find((item) => item.id === camera.camera_id)?.name || camera.camera_id,
              }))}
              zones={canvasZones}
              canEdit={canEdit}
              showCameras={showCameras}
              showZones={showZones}
              selected={selected}
              reviewing={Boolean(proposal)}
              onSelect={selectItem}
              onShowCameras={setShowCameras}
              onShowZones={setShowZones}
              onChangeCamera={changeCamera}
              onChangeZone={changeZone}
              onAddPoint={addZonePoint}
              onGestureEnd={() => void commitLayout()}
            />
            <p className="property-meta">
              {image.source === "naip" ? "USGS NAIP" : "Esri World Imagery"}
              {image.captured_at ? ` · ${image.captured_at.slice(0, 10)}` : ""}
              {image.resolution_m ? ` · ${Number(image.resolution_m).toFixed(2)} m/pixel` : ""}
              {image.attribution ? ` · ${image.attribution}` : ""}
            </p>
          </>
        ) : (
          <div className="property-empty">
            <h2>No aerial photo yet</h2>
            <p>Download one from an address or latitude and longitude. SurvNG keeps the sharpest recent image.</p>
          </div>
        )}
      </div>
      <aside className="property-panel">
        <section>
          <h2>Aerial photo</h2>
          <form onSubmit={downloadImage} data-property-location="">
            <label>Address
              <input value={address} onChange={(event) => setAddress(event.target.value)} onBlur={() => void saveLocation()} disabled={!canEdit || busy} placeholder="123 Main St, Springfield" />
            </label>
            <label>Latitude
              <input value={latitude} onChange={(event) => setLatitude(event.target.value)} onBlur={() => void saveLocation()} disabled={!canEdit || busy} inputMode="decimal" />
            </label>
            <label>Longitude
              <input value={longitude} onChange={(event) => setLongitude(event.target.value)} onBlur={() => void saveLocation()} disabled={!canEdit || busy} inputMode="decimal" />
            </label>
            <label>Width and height (meters)
              <input value={widthM} onChange={(event) => setWidthM(event.target.value)} onBlur={() => void saveLocation()} disabled={!canEdit || busy} inputMode="decimal" />
            </label>
            <button type="submit" disabled={!canEdit || busy}>{busy ? "Downloading…" : "Download image"}</button>
          </form>
          {candidates.length ? (
            <ul className="property-candidates">
              {candidates.map((candidate) => (
                <li key={candidate.source} className={candidate.selected ? "selected" : ""}>
                  {candidate.selected ? "Kept" : "Passed"} · {candidate.source === "naip" ? "USGS NAIP" : "Esri"}
                  {candidate.captured_at ? ` · ${candidate.captured_at.slice(0, 10)}` : ""}
                  {` · ${Number(candidate.resolution_m).toFixed(2)} m/pixel`}
                </li>
              ))}
            </ul>
          ) : null}
        </section>
        <section>
          <h2>Cameras</h2>
          {cameras.length ? cameras.map((camera) => {
            const placement = layout.cameras.find((item) => item.camera_id === camera.id);
            const notifications = Object.hasOwn(overrides, `camera:${camera.id}`)
              ? overrides[`camera:${camera.id}`]
              : camera.incident_notifications_enabled !== false;
            const cameraSelected = selected?.kind === "camera" && selected.id === camera.id;
            return (
              <article className={`property-card${cameraSelected ? " is-selected" : ""}`} key={camera.id}>
                <header>
                  <button type="button" className="property-card-name" aria-expanded={Boolean(placement && cameraSelected)} onClick={() => chooseSelection({ kind: "camera", id: camera.id })}>
                    {camera.name || camera.id}
                  </button>
                  {placement ? <button type="button" disabled={!canEdit} onClick={() => removeCamera(camera.id)}>Remove</button> : <button type="button" disabled={!canEdit || !image} onClick={() => placeCamera(camera)}>Place</button>}
                </header>
                {placement && cameraSelected ? (
                  <>
                    <label>Make
                      <input value={placement.make} disabled={!canEdit} onChange={(event) => changeCamera(camera.id, { make: event.target.value })} onBlur={() => void lookupFov(camera.id, placement.make, placement.model)} />
                    </label>
                    <label>Model
                      <input value={placement.model} disabled={!canEdit} onChange={(event) => changeCamera(camera.id, { model: event.target.value })} onBlur={() => void lookupFov(camera.id, placement.make, placement.model)} />
                    </label>
                    <label>Horizontal field of view (degrees)
                      <input type="number" min="1" max="180" value={placement.hfov_deg} disabled={!canEdit} onChange={(event) => changeCamera(camera.id, { hfov_deg: Number(event.target.value) })} onBlur={() => void commitLayout()} />
                    </label>
                    <label className="property-check">
                      <input type="checkbox" checked={notifications} disabled={!canEdit} onChange={(event) => void toggleCameraNotifications(camera.id, event.target.checked)} />
                      Notifications
                    </label>
                  </>
                ) : null}
              </article>
            );
          }) : <p>No cameras are configured.</p>}
        </section>
        <section>
          <h2>Zones</h2>
          {configuredZones.length ? configuredZones.map((zone) => {
            const placed = layout.zones.some((item) => item.camera_id === zone.cameraId && item.zone_name === zone.zoneName);
            const key = `zone:${zone.cameraId}:${zone.zoneName}`;
            const zoneKey = zoneActivityKey(zone.cameraId, zone.zoneName);
            const zoneSelected = selected?.kind === "zone" && selected.id === zoneKey;
            const notifications = Object.hasOwn(overrides, key) ? overrides[key] : zone.notificationsEnabled;
            return (
              <article className={`property-card${zoneSelected ? " is-selected" : ""}`} key={`${zone.cameraId}:${zone.zoneName}`}>
                <header>
                  <button type="button" className="property-card-name" aria-expanded={Boolean(placed && zoneSelected)} onClick={() => chooseSelection({ kind: "zone", id: zoneKey })}>
                    <span className="property-swatch" style={{ background: zone.color }} />{zone.cameraName} · {zone.zoneName}
                  </button>
                  {placed
                    ? <button type="button" disabled={!canEdit} onClick={() => removeZone(zone.cameraId, zone.zoneName)}>Remove</button>
                    : <button type="button" disabled={!canEdit || !image} onClick={() => placeZone(zone)}>Place</button>}
                </header>
                {placed && zoneSelected ? (
                  <>
                    <label className="property-check">
                      <input type="checkbox" checked={notifications} disabled={!canEdit} onChange={(event) => void toggleZoneNotifications(zone.cameraId, zone.zoneName, event.target.checked)} />
                      Notifications
                    </label>
                    {proposal?.key === zoneKey ? (
                      <>
                        <div className="property-card-actions">
                          <button type="button" className="property-undo" onClick={() => void acceptProposal()}>Accept outline</button>
                          <button type="button" className="property-undo" onClick={clearProposal}>Discard</button>
                        </div>
                        <p className="property-hint">This outline is not saved yet. Drag a point to adjust it, then accept.</p>
                      </>
                    ) : (
                      <>
                        <div className="property-card-actions">
                          <button
                            type="button"
                            className="property-undo"
                            onClick={() => void conformZone(zone.cameraId, zone.zoneName)}
                            disabled={!canEdit || !image || Boolean(conformingKey) || (layout.zones.find((item) => item.camera_id === zone.cameraId && item.zone_name === zone.zoneName)?.points?.length || 0) < 3}
                          >
                            <Sparkles size={15} /> {conformingKey === zoneKey ? "Conforming…" : "Conform"}
                          </button>
                          <button type="button" className="property-undo" onClick={() => undoZonePoint(zoneKey)} disabled={!canEdit || !(pointHistory[zoneKey]?.length)}>
                            <Undo2 size={15} /> Undo Point
                          </button>
                        </div>
                        <p className="property-hint">Click the photo to add points in order. Drag a point to move it.</p>
                      </>
                    )}
                  </>
                ) : null}
              </article>
            );
          }) : <p>Configure detection zones on a camera before placing them here.</p>}
          {layout.zones.filter((zone) => overlayPresentation(zone, configuredZones).stale).map((zone) => {
            const zoneKey = zoneActivityKey(zone.camera_id, zone.zone_name);
            const zoneSelected = selected?.kind === "zone" && selected.id === zoneKey;
            return (
              <article className={`property-card${zoneSelected ? " is-selected" : ""}`} key={`stale:${zone.camera_id}:${zone.zone_name}`}>
                <header>
                  <button type="button" className="property-card-name" aria-expanded={zoneSelected} onClick={() => chooseSelection({ kind: "zone", id: zoneKey })}>
                    {zone.camera_id} · {zone.zone_name}
                  </button>
                  <button type="button" disabled={!canEdit} onClick={() => removeZone(zone.camera_id, zone.zone_name)}>Remove</button>
                </header>
                {zoneSelected ? <p className="property-stale">This zone is no longer configured.</p> : null}
              </article>
            );
          })}
        </section>
      </aside>
    </main>
  );
}
