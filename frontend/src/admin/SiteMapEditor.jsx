import React, { useRef, useState } from "react";
import { Trash2, Upload } from "lucide-react";
import { fetch } from "../shared/api.js";
import { PropertyMapCanvas, propertyMapImageUrl } from "../incidents/PropertyMapPanel.jsx";

function mergePlacement(camera, patch) {
  return {
    enabled: true,
    x: 0.5,
    y: 0.5,
    heading: 0,
    fov: 90,
    range: 0.18,
    ...(camera?.map_placement || {}),
    ...patch,
  };
}

export function SiteMapEditor({ config, cameras, onUpdatePlacement, onSiteMapChanged }) {
  const fileRef = useRef(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [selectedCameraId, setSelectedCameraId] = useState(cameras[0]?.id || "");
  const [mapVersion, setMapVersion] = useState(config?.site_map?.image_path || "");
  const imageUrl = propertyMapImageUrl(config, mapVersion);
  const selectedCamera = cameras.find((camera) => camera.id === selectedCameraId) || null;
  const placement = selectedCamera?.map_placement || {};

  async function uploadMap(event) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    setUploading(true);
    setUploadError("");
    try {
      const body = new FormData();
      body.append("file", file);
      const response = await fetch("/api/config/site-map", { method: "POST", body });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Unable to upload property map");
      setMapVersion(payload.image_path || String(Date.now()));
      onSiteMapChanged?.(payload);
    } catch (error) {
      setUploadError(error.message || "Unable to upload property map");
    } finally {
      setUploading(false);
    }
  }

  async function removeMap() {
    setUploading(true);
    setUploadError("");
    try {
      const response = await fetch("/api/config/site-map", { method: "DELETE" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Unable to remove property map");
      setMapVersion("");
      onSiteMapChanged?.(payload);
    } catch (error) {
      setUploadError(error.message || "Unable to remove property map");
    } finally {
      setUploading(false);
    }
  }

  function updatePlacement(patch) {
    if (!selectedCameraId || !onUpdatePlacement) return;
    onUpdatePlacement(selectedCameraId, mergePlacement(selectedCamera, patch));
  }

  return (
    <div className="site-map-editor">
      <div className="site-map-editor-actions">
        <input ref={fileRef} type="file" accept="image/jpeg,image/png,image/webp" hidden onChange={uploadMap} />
        <button type="button" onClick={() => fileRef.current?.click()} disabled={uploading}><Upload size={15} /> {uploading ? "Uploading…" : "Upload property map"}</button>
        {config?.site_map?.image_path ? <button type="button" className="danger compact" onClick={removeMap} disabled={uploading}><Trash2 size={14} /> Remove map</button> : null}
      </div>
      {uploadError ? <p className="admin-action-note error">{uploadError}</p> : null}
      <p className="admin-action-note">Upload a top-down image of your property. Place each camera on the map and set its field of view for the incident workspace.</p>
      <div className="site-map-editor-layout">
        <PropertyMapCanvas
          cameras={cameras}
          editable
          selectedCameraId={selectedCameraId}
          imageUrl={imageUrl}
          onPlacementChange={(cameraId, nextPlacement) => onUpdatePlacement?.(cameraId, nextPlacement)}
          className="admin-site-map-canvas"
        />
        <div className="site-map-editor-controls">
          <label>Camera<select value={selectedCameraId} onChange={(event) => setSelectedCameraId(event.target.value)}>
            {cameras.map((camera) => <option key={camera.id} value={camera.id}>{camera.name || camera.id}</option>)}
          </select></label>
          <label className="checkbox-row"><input type="checkbox" checked={placement.enabled !== false} onChange={(event) => updatePlacement({ enabled: event.target.checked })} /> Show on property map</label>
          <label>Heading (°)<input type="number" min="0" max="359" step="1" value={Math.round(Number(placement.heading) || 0)} onChange={(event) => updatePlacement({ heading: Number(event.target.value) })} /></label>
          <label>Field of view (°)<input type="number" min="10" max="180" step="1" value={Math.round(Number(placement.fov) || 90)} onChange={(event) => updatePlacement({ fov: Number(event.target.value) })} /></label>
          <label>Reach<input type="number" min="0.03" max="0.5" step="0.01" value={Number(placement.range ?? 0.18)} onChange={(event) => updatePlacement({ range: Number(event.target.value) })} /></label>
          <p className="admin-action-note">Click the map to place the selected camera. Use arrow keys to nudge placement while the map is focused.</p>
        </div>
      </div>
    </div>
  );
}

export function CameraMapPlacementEditor({ camera, onUpdatePlacement, config, cameras }) {
  const [mapVersion] = useState(config?.site_map?.image_path || "");
  const imageUrl = propertyMapImageUrl(config, mapVersion);
  const placement = camera?.map_placement || {};

  function updatePlacement(patch) {
    onUpdatePlacement?.(camera.id, mergePlacement(camera, patch));
  }

  if (!config?.site_map?.image_path) {
    return <p className="admin-action-note">Upload a property map under Admin → General → Property Map before placing this camera.</p>;
  }

  return (
    <div className="camera-map-editor">
      <PropertyMapCanvas
        cameras={cameras}
        editable
        selectedCameraId={camera.id}
        imageUrl={imageUrl}
        onPlacementChange={(cameraId, nextPlacement) => onUpdatePlacement?.(cameraId, nextPlacement)}
        className="admin-site-map-canvas"
      />
      <div className="admin-field-grid">
        <label className="checkbox-row"><input type="checkbox" checked={placement.enabled !== false} onChange={(event) => updatePlacement({ enabled: event.target.checked })} /> Show on property map</label>
        <label>Heading (°)<input type="number" min="0" max="359" step="1" value={Math.round(Number(placement.heading) || 0)} onChange={(event) => updatePlacement({ heading: Number(event.target.value) })} /></label>
        <label>Field of view (°)<input type="number" min="10" max="180" step="1" value={Math.round(Number(placement.fov) || 90)} onChange={(event) => updatePlacement({ fov: Number(event.target.value) })} /></label>
        <label>Reach<input type="number" min="0.03" max="0.5" step="0.01" value={Number(placement.range ?? 0.18)} onChange={(event) => updatePlacement({ range: Number(event.target.value) })} /></label>
      </div>
    </div>
  );
}
