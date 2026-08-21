import React from "react";
import { Camera } from "lucide-react";

/** Compact native camera select shown on mobile (max-width 760px) via `.mobile-camera-select`. */
export function MobileCameraSelect({
  cameras = [],
  value = "",
  onChange = () => {},
  className = "",
  ariaLabel = "Camera",
  allOption = null,
  disabled = false,
}) {
  return (
    <label className={`mobile-camera-select ${className}`.trim()}>
      <Camera size={15} aria-hidden="true" />
      <span className="sr-only">{ariaLabel}</span>
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        aria-label={ariaLabel}
        disabled={disabled}
      >
        {allOption ? <option value={allOption.value}>{allOption.label}</option> : null}
        {cameras.map((camera) => (
          <option key={camera.id} value={camera.id}>{camera.name || camera.id}</option>
        ))}
      </select>
    </label>
  );
}
