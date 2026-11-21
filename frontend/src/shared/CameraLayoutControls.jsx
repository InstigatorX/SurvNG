import React from "react";
import { Camera, ChevronLeft, ChevronRight, Grid2X2, GripVertical } from "lucide-react";
import { LIVE_DENSITY_OPTIONS } from "../liveWorkspace.mjs";

/**
 * Camera grid controls shared by Live and the Timeline's All cameras grid:
 * how many cameras per page (with paging), and the Automatic / Focus / Custom
 * layout. Each page decides what Custom means; extra controls go in children.
 */
export function CameraDensityControl({ density, page, pageCount, onDensityChange, onPageChange }) {
  return (
    <div className="live-density-control" role="group" aria-label="Visible camera density">
      {LIVE_DENSITY_OPTIONS.map((option) => <button type="button" key={option} className={density === option ? "active" : ""} aria-pressed={density === option} onClick={() => onDensityChange(option)}>{option === "fit" ? <Grid2X2 size={15} /> : option === "4" ? <><Grid2X2 size={13} /> 4</> : option}</button>)}
      {pageCount > 1 ? <span className="live-density-pages"><button type="button" onClick={() => onPageChange(page - 1)} disabled={page === 0} aria-label="Previous camera page"><ChevronLeft size={15} /></button><small>{page + 1}/{pageCount}</small><button type="button" onClick={() => onPageChange(page + 1)} disabled={page >= pageCount - 1} aria-label="Next camera page"><ChevronRight size={15} /></button></span> : null}
    </div>
  );
}

export function CameraLayoutModeControl({
  mode,
  customAvailable,
  onModeChange,
  ariaLabel,
  customTitle = "Arrange and resize cameras",
  children = null,
}) {
  return (
    <div className="live-layout-control" role="group" aria-label={ariaLabel}>
      <button type="button" className={mode === "auto" ? "active" : ""} aria-pressed={mode === "auto"} onClick={() => onModeChange("auto")}><Grid2X2 size={15} /> Automatic</button>
      <button type="button" className={mode === "focus" ? "active" : ""} aria-pressed={mode === "focus"} onClick={() => onModeChange("focus")} disabled={!customAvailable} title={customAvailable ? "Prioritize one camera in a desktop mosaic" : "Focus layout is available on desktop"}><Camera size={15} /> Focus</button>
      <button type="button" className={mode === "custom" ? "active" : ""} aria-pressed={mode === "custom"} onClick={() => onModeChange("custom")} disabled={!customAvailable} title={customAvailable ? customTitle : "Custom layout is available on desktop"}><GripVertical size={15} /> Custom</button>
      {children}
    </div>
  );
}
