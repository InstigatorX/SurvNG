import React, { useEffect, useState } from "react";
import {
  Camera,
  Copy,
  RefreshCcw,
  RotateCcw,
} from "lucide-react";
import { liveFramingStyle, normalizedLiveFraming } from "../liveFraming.mjs";
import { appUrl } from "../shared/api.js";
import { clearMaskedUrlPassword, clearMaskedSecret, secretInputValue, secretInputHint } from "../shared/secrets.js";
import { uniqueCameraId } from "../shared/cameras.js";

export function defaultCamera(cameras, seed = {}) {
  const id = uniqueCameraId(cameras, seed.id || seed.name || "camera");
  const host = seed.onvif?.host || seed.baichuan?.host || "";
  return {
    id,
    name: seed.name ? `${seed.name} Copy` : "New Camera",
    video_backend: seed.video_backend || "url",
    stream_url: clearMaskedUrlPassword(seed.stream_url),
    live_stream_url: clearMaskedUrlPassword(seed.live_stream_url),
    live_view: structuredClone(seed.live_view || {
      main: { fit: "cover", focal_x: 50, focal_y: 50, zoom: 1 },
      live: { fit: "cover", focal_x: 50, focal_y: 50, zoom: 1 },
    }),
    record: seed.record ?? true,
    record_sub: seed.record_sub ?? false,
    retention: {
      main_days: seed.retention?.main_days ?? null,
      live_days: seed.retention?.live_days ?? null,
    },
    require_incident_zone: seed.require_incident_zone ?? null,
    object_activity_attribution: seed.object_activity_attribution || "inherit",
    motion_qualification: {
      ...defaultCameraMotionQualification(),
      mode: seed.motion_qualification?.mode || "inherit",
      sensitivity: seed.motion_qualification?.sensitivity || "inherit",
      stationary_object_tolerance: seed.motion_qualification?.stationary_object_tolerance || "inherit",
      illumination_filter_enabled: seed.motion_qualification?.illumination_filter_enabled ?? null,
      frame_width: seed.motion_qualification?.frame_width ?? null,
      visual_backup_grace_seconds: seed.motion_qualification?.visual_backup_grace_seconds ?? null,
      visual_backup_min_score: seed.motion_qualification?.visual_backup_min_score ?? null,
      visual_backup_min_consecutive: seed.motion_qualification?.visual_backup_min_consecutive ?? null,
      visual_backup_cooldown_seconds: seed.motion_qualification?.visual_backup_cooldown_seconds ?? null,
      visual_backup_max_triggers_5m: seed.motion_qualification?.visual_backup_max_triggers_5m ?? null,
      borderline_rescue_enabled: seed.motion_qualification?.borderline_rescue_enabled ?? null,
      borderline_margin: seed.motion_qualification?.borderline_margin ?? null,
      suppression_verification_rate: seed.motion_qualification?.suppression_verification_rate ?? null,
      spatial_alignment: structuredClone(seed.motion_qualification?.spatial_alignment || {}),
      pipeline: structuredClone(seed.motion_qualification?.pipeline || {}),
    },
    zones: structuredClone(seed.zones || []),
    onvif: {
      enabled: seed.onvif?.enabled || false,
      host,
      port: seed.onvif?.port || 8000,
      username: seed.onvif?.username || "",
      password: clearMaskedSecret(seed.onvif?.password),
    },
    baichuan: {
      enabled: seed.baichuan?.enabled || false,
      host,
      port: seed.baichuan?.port || 9000,
      username: seed.baichuan?.username || "",
      password: clearMaskedSecret(seed.baichuan?.password),
      channel: seed.baichuan?.channel || 0,
    },
  };
}

export function CameraOnvifEditor({ camera, onChange }) {
  const update = (field, value) => onChange(["onvif", field], value);

  return <section className="sub-panel camera-onvif-panel" aria-labelledby={`camera-onvif-title-${camera.id}`}>
    <div className="camera-onvif-heading">
      <h3 id={`camera-onvif-title-${camera.id}`}>ONVIF</h3>
      <label className="check-field"><input type="checkbox" checked={camera.onvif?.enabled || false} onChange={(event) => update("enabled", event.target.checked)} /> Enabled</label>
    </div>
    <div className="onvif-field-grid">
      <label>Host<input value={camera.onvif?.host || ""} onChange={(event) => update("host", event.target.value)} /></label>
      <label>Port<input type="number" value={camera.onvif?.port || 8000} onChange={(event) => update("port", Number(event.target.value))} /></label>
      <label>Username<input value={camera.onvif?.username || ""} onChange={(event) => update("username", event.target.value)} /></label>
      <label>Password<input type="password" value={secretInputValue(camera.onvif?.password)} placeholder={secretInputHint(camera.onvif?.password)} onChange={(event) => update("password", event.target.value)} /></label>
    </div>
  </section>;
}

export function LiveViewFramingEditor({ camera, onChange }) {
  const [source, setSource] = useState(camera.live_stream_url ? "live" : "main");
  const [previewRevision, setPreviewRevision] = useState(() => Date.now());
  const framing = normalizedLiveFraming(camera, source);
  const sourceName = source === "main" ? "Main" : "Sub";
  const update = (field, value) => onChange(["live_view", source, field], value);
  const reset = () => onChange(["live_view", source], {
    fit: "cover",
    focal_x: 50,
    focal_y: 50,
    zoom: 1,
  });

  useEffect(() => {
    setSource(camera.live_stream_url ? "live" : "main");
    setPreviewRevision(Date.now());
  }, [camera.id]);

  return <section className="live-framing-editor" aria-labelledby={`live-framing-title-${camera.id}`}>
    <div className="live-framing-heading">
      <div>
        <h3 id={`live-framing-title-${camera.id}`}>Live view framing</h3>
        <p>Tile presentation only · recordings and detection are unchanged.</p>
      </div>
      <div className="segmented live-framing-source" role="group" aria-label="Stream to frame">
        <button type="button" className={source === "main" ? "active" : ""} aria-pressed={source === "main"} onClick={() => setSource("main")}>Main</button>
        <button type="button" className={source === "live" ? "active" : ""} aria-pressed={source === "live"} onClick={() => setSource("live")} disabled={!camera.live_stream_url}>Sub</button>
      </div>
    </div>
    <div className="live-framing-workspace">
      <div className="live-framing-preview" style={liveFramingStyle(camera, source)}>
        <img
          key={`${camera.id}:${source}:${previewRevision}`}
          src={appUrl(`/api/cameras/${encodeURIComponent(camera.id)}/snapshot.jpg?source=${source}&t=${previewRevision}`)}
          alt={`${camera.name} ${sourceName} stream framing preview`}
        />
        <span>{sourceName} preview</span>
      </div>
      <div className="live-framing-controls">
        <label><span>Tile fill</span>
          <select value={framing.fit} onChange={(event) => update("fit", event.target.value)}>
            <option value="cover">Fill frame (crop edges)</option>
            <option value="contain">Fit entire image</option>
          </select>
        </label>
        <label><span>Horizontal focus (%)</span>
          <input type="number" min="0" max="100" step="1" value={Math.round(framing.focalX)} onChange={(event) => update("focal_x", Number(event.target.value))} />
        </label>
        <label><span>Vertical focus (%)</span>
          <input type="number" min="0" max="100" step="1" value={Math.round(framing.focalY)} onChange={(event) => update("focal_y", Number(event.target.value))} />
        </label>
        <label><span>Display zoom (×)</span>
          <input type="number" min="1" max="3" step="0.05" value={framing.zoom.toFixed(2)} onChange={(event) => update("zoom", Number(event.target.value))} />
        </label>
        <div className="live-framing-actions">
          <button type="button" onClick={() => setPreviewRevision(Date.now())}><RefreshCcw size={15} /> Refresh preview</button>
          <button type="button" onClick={reset}><RotateCcw size={15} /> Reset {sourceName}</button>
        </div>
      </div>
    </div>
  </section>;
}

export function defaultCameraMotionQualification() {
  return {
    mode: "inherit",
    sensitivity: "inherit",
    stationary_object_tolerance: "inherit",
    illumination_filter_enabled: null,
    frame_width: null,
    visual_backup_grace_seconds: null,
    visual_backup_min_score: null,
    visual_backup_min_consecutive: null,
    visual_backup_cooldown_seconds: null,
    visual_backup_max_triggers_5m: null,
    borderline_rescue_enabled: null,
    borderline_margin: null,
    suppression_verification_rate: null,
    spatial_alignment: {},
    pipeline: {
      qualification: null,
      observation: null,
      fusion: null,
    },
  };
}

export function cameraMotionQualificationInherited(qualification) {
  const value = qualification || {};
  const alignment = value.spatial_alignment || {};
  const pipeline = value.pipeline || {};
  return (
    (value.mode == null || value.mode === "inherit")
    && (value.sensitivity == null || value.sensitivity === "inherit")
    && (value.stationary_object_tolerance == null || value.stationary_object_tolerance === "inherit")
    && value.illumination_filter_enabled == null
    && value.frame_width == null
    && value.visual_backup_grace_seconds == null
    && value.visual_backup_min_score == null
    && value.visual_backup_min_consecutive == null
    && value.visual_backup_cooldown_seconds == null
    && value.visual_backup_max_triggers_5m == null
    && value.borderline_rescue_enabled == null
    && value.borderline_margin == null
    && value.suppression_verification_rate == null
    && (alignment.mode == null || alignment.mode === "auto")
    && Number(alignment.confidence ?? 0) === 0
    && Number(alignment.scale_x ?? 1) === 1
    && Number(alignment.scale_y ?? 1) === 1
    && Number(alignment.offset_x ?? 0) === 0
    && Number(alignment.offset_y ?? 0) === 0
    && pipeline.qualification == null
    && pipeline.observation == null
    && pipeline.fusion == null
  );
}
