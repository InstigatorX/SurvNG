import React, { useLayoutEffect, useMemo, useRef, useState } from "react";
import { MapPin } from "lucide-react";
import { appUrl } from "../shared/api.js";

function fovConePath(cx, cy, heading, fov, radius) {
  const half = (Number(fov) || 90) / 2;
  const start = ((Number(heading) - half - 90) * Math.PI) / 180;
  const end = ((Number(heading) + half - 90) * Math.PI) / 180;
  const x1 = cx + radius * Math.cos(start);
  const y1 = cy + radius * Math.sin(start);
  const x2 = cx + radius * Math.cos(end);
  const y2 = cy + radius * Math.sin(end);
  const largeArc = Number(fov) > 180 ? 1 : 0;
  return `M ${cx} ${cy} L ${x1} ${y1} A ${radius} ${radius} 0 ${largeArc} 1 ${x2} ${y2} Z`;
}

export function propertyMapImageUrl(appConfig, cacheKey = "") {
  if (!appConfig?.site_map?.image_path) return "";
  const suffix = cacheKey ? `?v=${encodeURIComponent(cacheKey)}` : "";
  return appUrl(`/api/site-map/image${suffix}`);
}

export function PropertyMapCanvas({
  cameras = [],
  activeCameraId = "",
  editable = false,
  selectedCameraId = "",
  imageUrl = "",
  onPlacementChange = null,
  className = "",
}) {
  const containerRef = useRef(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  useLayoutEffect(() => {
    const node = containerRef.current;
    if (!node) return undefined;
    const update = () => {
      const rect = node.getBoundingClientRect();
      setSize({ width: Math.max(0, rect.width), height: Math.max(0, rect.height) });
    };
    update();
    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", update);
      return () => window.removeEventListener("resize", update);
    }
    const observer = new ResizeObserver(update);
    observer.observe(node);
    return () => observer.disconnect();
  }, [imageUrl]);

  const placements = useMemo(
    () => cameras
      .map((camera) => ({
        camera,
        placement: camera.map_placement || {},
      }))
      .filter(({ placement }) => placement.enabled !== false && (editable || placement.enabled)),
    [cameras, editable],
  );

  const radius = Math.max(24, Math.min(size.width, size.height) * 0.22);

  function pointerToPlacement(event) {
    const rect = event.currentTarget.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(1, (event.clientX - rect.left) / Math.max(1, rect.width))),
      y: Math.max(0, Math.min(1, (event.clientY - rect.top) / Math.max(1, rect.height))),
    };
  }

  function handleCanvasClick(event) {
    if (!editable || !selectedCameraId || !onPlacementChange) return;
    const next = pointerToPlacement(event);
    onPlacementChange(selectedCameraId, { ...next, enabled: true });
  }

  return (
    <div ref={containerRef} className={`property-map-canvas ${className}`.trim()} onClick={handleCanvasClick} role={editable ? "button" : undefined} tabIndex={editable ? 0 : undefined} onKeyDown={editable ? (event) => {
      if (!selectedCameraId || !onPlacementChange) return;
      const step = event.shiftKey ? 0.05 : 0.01;
      const camera = cameras.find((item) => item.id === selectedCameraId);
      const current = camera?.map_placement || { x: 0.5, y: 0.5, enabled: true };
      if (event.key === "ArrowLeft") onPlacementChange(selectedCameraId, { ...current, enabled: true, x: Math.max(0, current.x - step) });
      if (event.key === "ArrowRight") onPlacementChange(selectedCameraId, { ...current, enabled: true, x: Math.min(1, current.x + step) });
      if (event.key === "ArrowUp") onPlacementChange(selectedCameraId, { ...current, enabled: true, y: Math.max(0, current.y - step) });
      if (event.key === "ArrowDown") onPlacementChange(selectedCameraId, { ...current, enabled: true, y: Math.min(1, current.y + step) });
    } : undefined}>
      {imageUrl ? <img src={imageUrl} alt="" className="property-map-image" draggable={false} /> : <div className="property-map-empty">Upload a property map in Admin to show camera placement here.</div>}
      {imageUrl && size.width > 0 && size.height > 0 ? (
        <svg className="property-map-overlay" viewBox={`0 0 ${size.width} ${size.height}`} preserveAspectRatio="none" aria-hidden="true">
          {placements.map(({ camera, placement }) => {
            const cx = placement.x * size.width;
            const cy = placement.y * size.height;
            const active = camera.id === activeCameraId;
            const selected = camera.id === selectedCameraId;
            const coneRadius = Math.max(18, Math.min(size.width, size.height) * (Number(placement.range) || 0.18));
            const showCone = active || selected || editable;
            return (
              <g key={camera.id} className={`property-map-camera ${active ? "active" : ""} ${selected ? "selected" : ""}`}>
                {showCone ? <path d={fovConePath(cx, cy, placement.heading ?? 0, placement.fov ?? 90, coneRadius)} className="property-map-fov" /> : null}
                <circle cx={cx} cy={cy} r={active ? 7 : 5.5} className="property-map-dot" />
              </g>
            );
          })}
        </svg>
      ) : null}
    </div>
  );
}

export function PropertyMapPanel({ cameras = [], activeCameraId = "", cameraNameById, appConfig, compact = false }) {
  const imageUrl = propertyMapImageUrl(appConfig, appConfig?.site_map?.image_path);
  const activeName = cameraNameById?.get(activeCameraId) || activeCameraId;
  const hasMap = Boolean(appConfig?.site_map?.image_path);

  return (
    <section className={`property-map-panel ${compact ? "compact" : ""}`.trim()} aria-label="Camera placement map">
      <div className="property-map-panel-head">
        <MapPin size={14} aria-hidden="true" />
        <strong>Camera placement</strong>
      </div>
      <PropertyMapCanvas
        cameras={cameras}
        activeCameraId={activeCameraId}
        imageUrl={imageUrl}
        className={hasMap ? "has-image" : "missing-image"}
      />
      <footer className="property-map-panel-foot">
        <span>{hasMap ? (activeName || "Select an incident") : "No property map configured"}</span>
      </footer>
    </section>
  );
}
