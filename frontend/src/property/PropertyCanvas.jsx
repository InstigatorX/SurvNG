import React, { useEffect, useRef, useState } from "react";
import {
  Bike,
  Car,
  CircleDot,
  Package,
  PawPrint,
  User,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import {
  aimFromPointer,
  clampUnit,
  conePolygon,
  panPropertyView,
  PROPERTY_ZOOM_FACTOR,
  PROPERTY_ZOOM_MAX,
  PROPERTY_ZOOM_MIN,
  propertyViewAtRest,
  rotationFromPointer,
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
    if (drag.kind !== "pan" && !canEdit) return;
    event.stopPropagation();
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
        x: clampUnit(drag.zone.x + point.x - drag.origin.x),
        y: clampUnit(drag.zone.y + point.y - drag.origin.y),
      });
    } else if (drag.kind === "rotate-zone") {
      onChangeZone(drag.key, { rotation_deg: rotationFromPointer(drag.anchor, point) });
    } else if (drag.kind === "resize-zone") {
      const local = localPoint(point, drag.anchor, drag.zone.rotation_deg);
      onChangeZone(drag.key, {
        width: Math.min(1, Math.max(0.04, Math.abs(local.x) * 2)),
        height: Math.min(1, Math.max(0.04, Math.abs(local.y) * 2)),
      });
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

  return (
    <div className={`property-stage${zoomed ? " is-zoomed" : ""}`} ref={stageRef}>
      <div
        className="property-view"
        style={{
          transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})`,
          "--property-zoom": String(view.scale),
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
        {cameras.map((camera) => {
          const cone = conePolygon(camera, widthM, heightM);
          const tip = cone[Math.floor(cone.length / 2)];
          return (
            <g key={camera.camera_id}>
              <polygon points={pointList(cone)} fill="rgba(15, 118, 110, 0.28)" stroke="#0f766e" strokeWidth={0.004 * mark} />
              <circle
                cx={camera.x}
                cy={camera.y}
                r={0.016 * mark}
                fill="#0f766e"
                stroke="#ffffff"
                strokeWidth={0.004 * mark}
                onPointerDown={(event) => beginDrag(event, { kind: "move-camera", cameraId: camera.camera_id })}
              >
                <title>{camera.name || camera.camera_id}</title>
              </circle>
              {canEdit ? (
                <circle
                  cx={tip.x}
                  cy={tip.y}
                  r={0.014 * mark}
                  fill="#ffffff"
                  stroke="#0f766e"
                  strokeWidth={0.004 * mark}
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
        })}
        {zones.map((zone) => {
          const halfW = zone.width / 2;
          const halfH = zone.height / 2;
          const opacity = zone.opacity || 0;
          return (
            <g key={zone.key} transform={`translate(${zone.x} ${zone.y}) rotate(${zone.rotation_deg})`}>
              <rect
                x={-halfW}
                y={-halfH}
                width={zone.width}
                height={zone.height}
                fill={zone.color}
                fillOpacity={Math.max(opacity, zone.stale ? 0.08 : 0.16)}
                stroke={zone.color}
                strokeWidth={0.005 * mark}
                strokeDasharray={zone.stale ? `${0.02 * mark} ${0.012 * mark}` : undefined}
                onPointerDown={(event) => beginDrag(event, {
                  kind: "move-zone",
                  key: zone.key,
                  zone,
                })}
              >
                <title>{zone.label}</title>
              </rect>
              {canEdit ? (
                <>
                  <circle
                    cx={halfW}
                    cy={halfH}
                    r={0.012 * mark}
                    fill="#ffffff"
                    stroke={zone.color}
                    strokeWidth={0.003 * mark}
                    onPointerDown={(event) => beginDrag(event, {
                      kind: "resize-zone",
                      key: zone.key,
                      zone,
                      anchor: { x: zone.x, y: zone.y },
                    })}
                  >
                    <title>Resize zone</title>
                  </circle>
                  <circle
                    cx="0"
                    cy={-halfH - 0.03 * mark}
                    r={0.012 * mark}
                    fill="#ffffff"
                    stroke={zone.color}
                    strokeWidth={0.003 * mark}
                    onPointerDown={(event) => beginDrag(event, {
                      kind: "rotate-zone",
                      key: zone.key,
                      anchor: { x: zone.x, y: zone.y },
                    })}
                  >
                    <title>Rotate zone</title>
                  </circle>
                </>
              ) : null}
            </g>
          );
        })}
      </svg>
      <div className="property-icons">
        {zones.filter((zone) => zone.opacity > 0 && zone.icons?.length).map((zone) => (
          <div key={zone.key} className="property-icon-group" style={{ left: `${zone.x * 100}%`, top: `${zone.y * 100}%`, opacity: Math.min(1, zone.opacity + 0.35) }}>
            {zone.icons.slice(0, 3).map((name) => {
              const Icon = ICONS[name] || CircleDot;
              return <Icon key={name} size={18} aria-label={name} />;
            })}
          </div>
        ))}
      </div>
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

function localPoint(point, origin, rotationDeg) {
  const dx = point.x - origin.x;
  const dy = point.y - origin.y;
  const rad = (-rotationDeg * Math.PI) / 180;
  return {
    x: dx * Math.cos(rad) - dy * Math.sin(rad),
    y: dx * Math.sin(rad) + dy * Math.cos(rad),
  };
}
