import React, { useEffect, useRef, useState } from "react";
import {
  Bike,
  Car,
  CircleDot,
  Package,
  PawPrint,
  Pentagon,
  User,
  Video,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { insertZonePointWithIndex } from "../zoneGeometry.mjs";
import {
  aimFromPointer,
  clampUnit,
  conePolygon,
  moveZonePoint,
  panPropertyView,
  PROPERTY_ZOOM_FACTOR,
  PROPERTY_ZOOM_MAX,
  PROPERTY_ZOOM_MIN,
  propertyViewAtRest,
  removeZonePoint,
  stackPropertyOverlays,
  translateZonePoints,
  zoneCentroid,
  zonePolygon,
  zoomPropertyView,
} from "./propertyView.mjs";

const ICONS = {
  person: User,
  vehicle: Car,
  animal: PawPrint,
  bicycle: Bike,
  package: Package,
  generic: CircleDot,
};

function pointList(points) {
  return points.map((point) => `${point.x},${point.y}`).join(" ");
}

export function PropertyCanvas({
  imageUrl,
  widthM,
  heightM,
  cameras,
  zones,
  canEdit,
  showCameras = true,
  showZones = true,
  selected = null,
  onSelect,
  onShowCameras,
  onShowZones,
  onChangeCamera,
  onChangeZone,
  onGestureEnd,
}) {
  const stageRef = useRef(null);
  const svgRef = useRef(null);
  const dragRef = useRef(null);
  const viewRef = useRef(propertyViewAtRest());
  const [view, setView] = useState(propertyViewAtRest);
  viewRef.current = view;

  useEffect(() => {
    setView(propertyViewAtRest());
  }, [imageUrl]);

  useEffect(() => {
    const node = stageRef.current;
    if (!node) return undefined;
    function onWheel(event) {
      event.preventDefault();
      const box = node.getBoundingClientRect();
      const delta = Math.max(-120, Math.min(120, event.deltaY));
      const factor = Math.exp(-delta * 0.0017);
      setView((current) => zoomPropertyView(current, box, {
        x: event.clientX - box.left - box.width / 2,
        y: event.clientY - box.top - box.height / 2,
      }, factor));
    }
    function onTouchStart(event) {
      if (event.touches.length !== 2) return;
      event.preventDefault();
      const box = node.getBoundingClientRect();
      dragRef.current = {
        kind: "pinch",
        distance: touchDistance(event.touches),
        view: viewRef.current,
        anchor: touchAnchor(event.touches, box),
        frame: box,
      };
    }
    function onTouchMove(event) {
      const drag = dragRef.current;
      if (drag?.kind !== "pinch" || event.touches.length !== 2 || !drag.distance) return;
      event.preventDefault();
      const distance = touchDistance(event.touches);
      setView(zoomPropertyView(drag.view, drag.frame, drag.anchor, distance / drag.distance));
    }
    function onTouchEnd(event) {
      if (dragRef.current?.kind === "pinch" && event.touches.length < 2) dragRef.current = null;
    }
    node.addEventListener("wheel", onWheel, { passive: false });
    node.addEventListener("touchstart", onTouchStart, { passive: false });
    node.addEventListener("touchmove", onTouchMove, { passive: false });
    node.addEventListener("touchend", onTouchEnd);
    node.addEventListener("touchcancel", onTouchEnd);
    return () => {
      node.removeEventListener("wheel", onWheel);
      node.removeEventListener("touchstart", onTouchStart);
      node.removeEventListener("touchmove", onTouchMove);
      node.removeEventListener("touchend", onTouchEnd);
      node.removeEventListener("touchcancel", onTouchEnd);
    };
  }, []);

  function zoomBy(factor) {
    const box = stageRef.current?.getBoundingClientRect();
    if (!box) return;
    setView((current) => zoomPropertyView(current, box, { x: 0, y: 0 }, factor));
  }

  function pointer(event) {
    const rect = svgRef.current.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left) / rect.width,
      y: (event.clientY - rect.top) / rect.height,
    };
  }

  function beginDrag(event, drag) {
    if (dragRef.current?.kind === "pinch") return;
    event.stopPropagation();
    if (drag.kind === "move-camera" || drag.kind === "aim-camera") onSelect?.({ kind: "camera", id: drag.cameraId });
    if (drag.kind === "move-zone" || drag.kind === "move-point") onSelect?.({ kind: "zone", id: drag.key });
    if (drag.kind !== "pan" && !canEdit) return;
    dragRef.current = { ...drag, origin: pointer(event), lastX: event.clientX, lastY: event.clientY };
    try {
      event.currentTarget?.setPointerCapture?.(event.pointerId);
    } catch {
      // The pointer can already be released when a second touch starts a pinch.
    }
  }

  function moveDrag(event) {
    const drag = dragRef.current;
    if (!drag) return;
    const point = pointer(event);
    if (drag.kind === "pan") {
      const box = stageRef.current?.getBoundingClientRect();
      if (!box) return;
      const dx = event.clientX - drag.lastX;
      const dy = event.clientY - drag.lastY;
      drag.lastX = event.clientX;
      drag.lastY = event.clientY;
      setView((current) => panPropertyView(current, box, dx, dy));
    } else if (drag.kind === "move-camera") {
      onChangeCamera(drag.cameraId, { x: clampUnit(point.x), y: clampUnit(point.y) });
    } else if (drag.kind === "aim-camera") {
      onChangeCamera(drag.cameraId, aimFromPointer(drag.anchor, point, widthM, heightM));
    } else if (drag.kind === "move-zone") {
      onChangeZone(drag.key, {
        points: translateZonePoints(drag.points, point.x - drag.origin.x, point.y - drag.origin.y),
      });
    } else if (drag.kind === "move-point") {
      onChangeZone(drag.key, { points: moveZonePoint(drag.points, drag.index, point) });
    }
  }

  function endDrag() {
    if (!dragRef.current || dragRef.current.kind === "pinch") return;
    const kind = dragRef.current.kind;
    dragRef.current = null;
    if (kind !== "pan" && kind !== "pinch") onGestureEnd();
  }

  const zoomed = view.scale > 1;
  const mark = 1 / view.scale;
  const visibleCameras = showCameras ? cameras : [];
  const visibleZones = showZones ? zones : [];
  const stack = stackPropertyOverlays(visibleCameras, visibleZones, selected);

  function addZonePoint(event, zone, points) {
    event.stopPropagation();
    event.preventDefault();
    if (!canEdit) return;
    const inserted = insertZonePointWithIndex(points, pointer(event));
    onSelect?.({ kind: "zone", id: zone.key });
    onChangeZone(zone.key, { points: inserted.points });
    onGestureEnd();
  }

  function deleteZonePoint(event, zone, points, index) {
    event.stopPropagation();
    event.preventDefault();
    if (!canEdit || points.length <= 3) return;
    onChangeZone(zone.key, { points: removeZonePoint(points, index) });
    onGestureEnd();
  }

  return (
    <div className={`property-stage${zoomed ? " is-zoomed" : ""}`} ref={stageRef}>
      <div
        className="property-view"
        style={{
          "--property-zoom": String(view.scale),
          "--property-pan-x": `${view.x}px`,
          "--property-pan-y": `${view.y}px`,
        }}
      >
      <img src={imageUrl} alt="Aerial view of the property" draggable="false" />
      <svg
        ref={svgRef}
        viewBox="0 0 1 1"
        preserveAspectRatio="none"
        onPointerDown={(event) => {
          if (dragRef.current?.kind === "pinch") return;
          if (event.target !== event.currentTarget || !zoomed) return;
          beginDrag(event, { kind: "pan" });
        }}
        onPointerMove={moveDrag}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
      >
        {stack.map((entry) => {
          if (entry.kind === "camera") {
            const camera = entry.item;
            const cone = conePolygon(camera, widthM, heightM);
            const tip = cone[Math.floor(cone.length / 2)];
            const active = selected?.kind === "camera" && selected.id === camera.camera_id;
            return (
              <g key={`camera:${camera.camera_id}`}>
                <polygon points={pointList(cone)} fill="rgba(15, 118, 110, 0.28)" stroke="#0f766e" strokeWidth={0.004 * mark} pointerEvents="none" />
                <circle
                  cx={camera.x}
                  cy={camera.y}
                  r={0.008 * mark}
                  fill="#0f766e"
                  stroke="#ffffff"
                  strokeWidth={0.002 * mark}
                  onPointerDown={(event) => beginDrag(event, { kind: "move-camera", cameraId: camera.camera_id })}
                >
                  <title>{camera.name || camera.camera_id}</title>
                </circle>
                {canEdit && active ? (
                  <circle
                    cx={tip.x}
                    cy={tip.y}
                    r={0.007 * mark}
                    fill="#ffffff"
                    stroke="#0f766e"
                    strokeWidth={0.002 * mark}
                    onPointerDown={(event) => beginDrag(event, {
                      kind: "aim-camera",
                      cameraId: camera.camera_id,
                      anchor: { x: camera.x, y: camera.y },
                    })}
                  >
                    <title>Aim and range</title>
                  </circle>
                ) : null}
              </g>
            );
          }
          const zone = entry.item;
          const points = zonePolygon(zone);
          const opacity = zone.opacity || 0;
          const active = selected?.kind === "zone" && selected.id === zone.key;
          return (
            <g key={`zone:${zone.key}`}>
              <polygon
                points={pointList(points)}
                fill={zone.color}
                fillOpacity={Math.max(opacity, zone.stale ? 0.08 : 0.16)}
                stroke={zone.color}
                strokeWidth={0.004 * mark}
                strokeDasharray={zone.stale ? `${0.02 * mark} ${0.012 * mark}` : undefined}
                onPointerDown={(event) => beginDrag(event, { kind: "move-zone", key: zone.key, points })}
                onDoubleClick={(event) => addZonePoint(event, zone, points)}
              >
                <title>{zone.label}. Drag to move. Double-click the outline to add a point.</title>
              </polygon>
              {canEdit && active ? points.map((point, index) => (
                <circle
                  key={index}
                  cx={point.x}
                  cy={point.y}
                  r={0.006 * mark}
                  fill="#ffffff"
                  stroke={zone.color}
                  strokeWidth={0.002 * mark}
                  onPointerDown={(event) => beginDrag(event, { kind: "move-point", key: zone.key, index, points })}
                  onDoubleClick={(event) => deleteZonePoint(event, zone, points, index)}
                >
                  <title>{points.length > 3 ? "Drag point. Double-click to remove." : "Drag point."}</title>
                </circle>
              )) : null}
            </g>
          );
        })}
      </svg>
      <div className="property-icons">
        {visibleZones.filter((zone) => zone.opacity > 0 && zone.icons?.length).map((zone) => {
          const center = zoneCentroid(zonePolygon(zone));
          return (
            <div key={zone.key} className="property-icon-group" style={{ left: `${center.x * 100}%`, top: `${center.y * 100}%`, opacity: Math.min(1, zone.opacity + 0.35) }}>
              {zone.icons.slice(0, 3).map((name) => {
                const Icon = ICONS[name] || CircleDot;
                return <Icon key={name} size={18} aria-label={name} />;
              })}
            </div>
          );
        })}
      </div>
      </div>
      <div className="property-hud" role="group" aria-label="Property overlay">
        <button type="button" aria-pressed={showCameras} onClick={() => onShowCameras?.(!showCameras)}>
          <Video size={14} />
          <span>Cameras</span>
        </button>
        <button type="button" aria-pressed={showZones} onClick={() => onShowZones?.(!showZones)}>
          <Pentagon size={14} />
          <span>Zones</span>
        </button>
      </div>
      <div className="property-zoom" role="group" aria-label="Aerial zoom">
        <button type="button" aria-label="Zoom in" onClick={() => zoomBy(PROPERTY_ZOOM_FACTOR)} disabled={view.scale >= PROPERTY_ZOOM_MAX}>
          <ZoomIn size={16} />
        </button>
        <button type="button" className="property-zoom-level" aria-label="Reset zoom" onClick={() => setView(propertyViewAtRest())} disabled={!zoomed}>
          {Math.round(view.scale * 100)}%
        </button>
        <button type="button" aria-label="Zoom out" onClick={() => zoomBy(1 / PROPERTY_ZOOM_FACTOR)} disabled={view.scale <= PROPERTY_ZOOM_MIN}>
          <ZoomOut size={16} />
        </button>
      </div>
    </div>
  );
}

function touchDistance(touches) {
  const dx = touches[0].clientX - touches[1].clientX;
  const dy = touches[0].clientY - touches[1].clientY;
  return Math.hypot(dx, dy);
}

function touchAnchor(touches, box) {
  const x = (touches[0].clientX + touches[1].clientX) / 2;
  const y = (touches[0].clientY + touches[1].clientY) / 2;
  return {
    x: x - box.left - box.width / 2,
    y: y - box.top - box.height / 2,
  };
}

