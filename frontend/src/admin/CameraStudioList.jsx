import React from "react";
import { Plus } from "lucide-react";
import "./studio.css";

/**
 * Camera Studio list: every camera with a status dot (the status text is in
 * the tooltip and for screen readers), plus how many unsaved changes it has.
 * Selecting a camera keeps the current settings tab.
 */
export function CameraStudioList({ cameras = [], selectedId = "", statusById = {}, unsavedById = {}, onSelect, onAdd, disabled = false }) {
  return (
    <nav className="camera-studio-list" aria-label="Cameras">
      <ul>
        {cameras.map((camera) => {
          const status = statusById[camera.id] || { label: "Not saved yet", tone: "muted" };
          const unsaved = Number(unsavedById[camera.id] || 0);
          const selected = camera.id === selectedId;
          return (
            <li key={camera.id}>
              <button type="button" title={status.label} className={selected ? "active" : undefined} aria-current={selected ? "page" : undefined} onClick={() => onSelect(camera.id)} disabled={disabled}>
                <span className="camera-studio-name">{camera.name || camera.id}<span className="sr-only">, {status.label}</span></span>
                {unsaved ? <span className="camera-studio-sub unsaved">{unsaved} unsaved</span> : null}
                <i className={`camera-studio-dot ${status.tone}`} title={status.label} aria-hidden="true" />
              </button>
            </li>
          );
        })}
      </ul>
      <button type="button" className="camera-studio-add" onClick={onAdd} disabled={disabled}><Plus size={15} aria-hidden="true" />Add camera</button>
    </nav>
  );
}
