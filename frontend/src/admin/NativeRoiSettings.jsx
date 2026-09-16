import React from "react";

export default function NativeRoiSettings({ camera, onChange }) {
  return <fieldset>
    <legend>Detection region</legend>
    <label className="check-field"><input type="checkbox" checked={camera.native_roi?.enabled || false} onChange={(event) => onChange(["native_roi", "enabled"], event.target.checked)} /> Focus detection around incident zones</label>
    <p>Uses one padded rectangle around the selected zones. Full-frame checks continue periodically. This can help small objects, but does not guarantee lower GPU usage. Zones use the uncropped live-stream picture.</p>
    {camera.native_roi?.enabled ? <>
      <label>Incident zones<select aria-label="Incident zones" multiple value={camera.native_roi?.zone_names || []} onChange={(event) => onChange(["native_roi", "zone_names"], Array.from(event.target.selectedOptions, (option) => option.value))}>
        {(camera.zones || []).filter((zone) => zone.enabled !== false && zone.behavior === "incident").map((zone) => <option key={zone.name} value={zone.name}>{zone.name}</option>)}
      </select><small>No selection uses all enabled incident zones. If none match, detection uses the full frame.</small></label>
      <label>Padding (% of frame)<input aria-label="Padding (% of frame)" type="number" min="0" max="50" step="1" value={Math.round((camera.native_roi?.padding ?? 0.15) * 100)} onChange={(event) => onChange(["native_roi", "padding"], Number(event.target.value) / 100)} /><small>Include enough space above a floor zone to see a whole person or vehicle.</small></label>
      <label>Full-frame check every N detections<input aria-label="Full-frame check every N detections" type="number" min="1" max="30" step="1" value={camera.native_roi?.full_frame_interval ?? 5} onChange={(event) => onChange(["native_roi", "full_frame_interval"], Number(event.target.value))} /><small>1 always checks the full frame. Detection frequency stays unchanged.</small></label>
    </> : null}
  </fieldset>;
}
