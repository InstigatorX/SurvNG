import React, { useMemo } from "react";
import { MobileCameraSelect } from "./MobileCameraSelect.jsx";
import { TimelineCameraPicker } from "../timeline/TimelinePages.jsx";

export function mergeCamerasWithRuntimeStatus(cameras = [], runtimeStatus = []) {
  const statusById = new Map(runtimeStatus.map((item) => [item.id, item]));
  return cameras.map((camera) => {
    const status = statusById.get(camera.id) || {};
    const running = Boolean(status.running);
    return {
      ...camera,
      name: camera.name || camera.id,
      running,
      recording: Boolean(status.recording) || running,
      sub_recording: Boolean(status.sub_recording),
    };
  });
}

/** Shared camera scope control: searchable picker on desktop, native select on mobile. */
export function CameraScopePicker({
  cameras = [],
  runtimeStatus = [],
  value = "",
  onChange = () => {},
  allOption = null,
  ariaLabel = "Select camera",
  disabled = false,
  className = "",
  mobileClassName = "",
}) {
  const enrichedCameras = useMemo(
    () => mergeCamerasWithRuntimeStatus(cameras, runtimeStatus),
    [cameras, runtimeStatus],
  );
  const resolvedValue = value || (allOption ? allOption.value : enrichedCameras[0]?.id || "");

  return (
    <>
      <div className={`camera-scope-picker ${className}`.trim()}>
        <TimelineCameraPicker
          cameras={enrichedCameras}
          value={resolvedValue}
          onChange={onChange}
          allOption={allOption}
          ariaLabel={ariaLabel}
        />
      </div>
      <MobileCameraSelect
        className={`camera-scope-picker-mobile ${mobileClassName}`.trim()}
        cameras={enrichedCameras}
        value={value}
        onChange={onChange}
        ariaLabel={ariaLabel}
        allOption={allOption}
        disabled={disabled}
      />
    </>
  );
}
