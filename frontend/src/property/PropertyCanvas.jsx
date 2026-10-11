import React, { useRef } from "react";
import {
  Bike,
  Car,
  CircleDot,
  Package,
  PawPrint,
  User,
} from "lucide-react";
import { aimFromPointer, clampUnit, conePolygon, rotationFromPointer } from "./propertyView.mjs";

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
  const svgRef = useRef(null);
  const dragRef = useRef(null);

  function pointer(event) {
    const rect = svgRef.current.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left) / rect.width,
      y: (event.clientY - rect.top) / rect.height,
    };
  }

  function beginDrag(event, drag) {
    if (!canEdit) return;
    event.stopPropagation();
    svgRef.current?.setPointerCapture?.(event.pointerId);
    dragRef.current = { ...drag, origin: pointer(event) };
  }

  function moveDrag(event) {
    const drag = dragRef.current;
    if (!drag) return;
    const point = pointer(event);
    if (drag.kind === "move-camera") {
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
    if (!dragRef.current) return;
    dragRef.current = null;
    onGestureEnd();
  }

  return (
    <div className="property-stage">
      <img src={imageUrl} alt="Aerial view of the property" draggable="false" />
      <svg
        ref={svgRef}
        viewBox="0 0 1 1"
        preserveAspectRatio="none"
        onPointerMove={moveDrag}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
      >
        {cameras.map((camera) => {
          const cone = conePolygon(camera, widthM, heightM);
          const tip = cone[Math.floor(cone.length / 2)];
          return (
            <g key={camera.camera_id}>
              <polygon points={pointList(cone)} fill="rgba(15, 118, 110, 0.28)" stroke="#0f766e" strokeWidth="0.004" />
              <circle
                cx={camera.x}
                cy={camera.y}
                r="0.016"
                fill="#0f766e"
                stroke="#ffffff"
                strokeWidth="0.004"
                onPointerDown={(event) => beginDrag(event, { kind: "move-camera", cameraId: camera.camera_id })}
              >
                <title>{camera.name || camera.camera_id}</title>
              </circle>
              {canEdit ? (
                <circle
                  cx={tip.x}
                  cy={tip.y}
                  r="0.014"
                  fill="#ffffff"
                  stroke="#0f766e"
                  strokeWidth="0.004"
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
                strokeWidth="0.005"
                strokeDasharray={zone.stale ? "0.02 0.012" : undefined}
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
                    r="0.012"
                    fill="#ffffff"
                    stroke={zone.color}
                    strokeWidth="0.003"
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
                    cy={-halfH - 0.03}
                    r="0.012"
                    fill="#ffffff"
                    stroke={zone.color}
                    strokeWidth="0.003"
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
  );
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
