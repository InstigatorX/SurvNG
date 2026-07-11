import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  ArrowLeft,
  Camera,
  Copy,
  CircleDot,
  Clock3,
  Crop,
  Cog,
  Download,
  Cpu,
  Film,
  Gauge,
  HardDrive,
  Search,
  ListTree,
  Monitor,
  Moon,
  Plus,
  Pause,
  Play,
  Power,
  Radar,
  Radio,
  RefreshCcw,
  Save,
  ShieldCheck,
  Siren,
  SkipBack,
  SkipForward,
  Sun,
  Trash2,
  Video,
  X,
} from "lucide-react";
import "./styles.css";

const DEFAULT_TIME_ZONE = "America/New_York";
const US_TIME_ZONES = [
  ["America/New_York", "Eastern"],
  ["America/Chicago", "Central"],
  ["America/Denver", "Mountain"],
  ["America/Phoenix", "Arizona"],
  ["America/Los_Angeles", "Pacific"],
  ["America/Anchorage", "Alaska"],
  ["Pacific/Honolulu", "Hawaii"],
];

const THEMES = ["auto", "light", "dark"];
const THEME_META = {
  auto: { label: "Auto", icon: Monitor },
  light: { label: "Light", icon: Sun },
  dark: { label: "Dark", icon: Moon },
};

function apiFile(path) {
  return `/api/files?path=${encodeURIComponent(path)}`;
}

function useStoredState(key, initialValue) {
  const [value, setValue] = useState(() => localStorage.getItem(key) || initialValue);
  useEffect(() => {
    localStorage.setItem(key, value);
  }, [key, value]);
  return [value, setValue];
}

function formatDateTime(value, timeZone) {
  if (!value) return "--";
  const date = typeof value === "number" ? new Date(value * 1000) : new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat("en-US", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
    timeZoneName: "short",
  }).format(date);
}

function formatTimeOnly(value, timeZone) {
  if (!value) return "--";
  const date = typeof value === "number" ? new Date(value * 1000) : new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat("en-US", {
    timeZone,
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
  }).format(date);
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "--";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = Math.max(0, bytes);
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value >= 10 || unit === 0 ? value.toFixed(0) : value.toFixed(1)}${units[unit]}`;
}

function formatMilliseconds(value) {
  if (!Number.isFinite(value)) return "--";
  return `${value >= 10 ? value.toFixed(0) : value.toFixed(1)}ms`;
}

function formatRate(value) {
  if (!Number.isFinite(value)) return "--";
  return `${value >= 10 ? value.toFixed(0) : value.toFixed(1)}/s`;
}

function formatAge(seconds) {
  if (!Number.isFinite(seconds)) return "--";
  if (seconds < 60) return `${Math.max(0, Math.round(seconds))}s ago`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h ago`;
  return `${Math.round(seconds / 86400)}d ago`;
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 1) return "instant";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60);
  return rest ? `${minutes}m ${rest}s` : `${minutes}m`;
}

function formatClock(seconds) {
  if (!Number.isFinite(seconds)) return "00:00";
  const whole = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor((whole % 3600) / 60);
  const secs = whole % 60;
  return hours
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`
    : `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function formatDayClock(seconds) {
  const whole = Math.max(0, Math.min(24 * 60 * 60, Math.floor(Number(seconds) || 0)));
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor((whole % 3600) / 60);
  const secs = whole % 60;
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function dateKeyForTimeZone(value, timeZone) {
  const date = typeof value === "number" ? new Date(value * 1000) : new Date(value || Date.now());
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(date);
  const part = (type) => parts.find((item) => item.type === type)?.value || "";
  return `${part("year")}-${part("month")}-${part("day")}`;
}

function addDaysToDateKey(dateKey, days) {
  const [year, month, day] = dateKey.split("-").map(Number);
  const date = new Date(Date.UTC(year, month - 1, day + days));
  return date.toISOString().slice(0, 10);
}

function zonedDateSecondToEpoch(dateKey, seconds, timeZone) {
  const [year, month, day] = dateKey.split("-").map(Number);
  const desired = Date.UTC(year, month - 1, day, 0, 0, 0) + Math.max(0, seconds) * 1000;
  let guess = desired;
  const formatter = new Intl.DateTimeFormat("en-US", {
    timeZone,
    hourCycle: "h23",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  for (let index = 0; index < 3; index += 1) {
    const parts = formatter.formatToParts(new Date(guess));
    const value = (type) => Number(parts.find((item) => item.type === type)?.value || 0);
    const actual = Date.UTC(value("year"), value("month") - 1, value("day"), value("hour"), value("minute"), value("second"));
    guess += desired - actual;
  }
  return guess / 1000;
}

function slugify(value) {
  return String(value || "camera")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    || "camera";
}

function uniqueCameraId(cameras, base) {
  const existing = new Set(cameras.map((camera) => camera.id));
  let candidate = slugify(base);
  let index = 2;
  while (existing.has(candidate)) {
    candidate = `${slugify(base)}-${index}`;
    index += 1;
  }
  return candidate;
}

function streamUrlDefaults(url) {
  try {
    const parsed = new URL(url);
    if (!parsed.hostname) return {};
    return {
      scheme: parsed.protocol.replace(":", "").toLowerCase(),
      host: parsed.hostname,
      port: parsed.port ? Number(parsed.port) : null,
      username: decodeURIComponent(parsed.username || ""),
      password: decodeURIComponent(parsed.password || ""),
      channel: Number(parsed.searchParams.get("channel") || parsed.searchParams.get("chn") || 0),
    };
  } catch {
    return {};
  }
}

function inferredBackendLabel(camera) {
  const defaults = streamUrlDefaults(camera.stream_url || camera.live_stream_url || "");
  if (defaults.scheme === "reolink") return "Reolink Baichuan";
  if (defaults.scheme === "rtsp") return "RTSP";
  if (defaults.scheme) return defaults.scheme.toUpperCase();
  return "URL";
}

function cameraWithDerivedConnection(camera) {
  const defaults = streamUrlDefaults(camera.stream_url || camera.live_stream_url || "");
  if (!defaults.host) return camera;
  const isReolink = defaults.scheme === "reolink";
  const isRtsp = defaults.scheme === "rtsp";
  return {
    ...camera,
    video_backend: isReolink ? "baichuan_native" : isRtsp ? "url" : camera.video_backend,
    onvif: {
      ...camera.onvif,
      host: camera.onvif?.host || defaults.host,
      username: camera.onvif?.username || defaults.username,
      password: camera.onvif?.password || defaults.password,
    },
    baichuan: {
      ...camera.baichuan,
      enabled: isReolink,
      host: isReolink ? defaults.host : camera.baichuan?.host || defaults.host,
      port: isReolink ? defaults.port || 9000 : camera.baichuan?.port || 9000,
      username: isReolink ? defaults.username : camera.baichuan?.username || defaults.username,
      password: isReolink ? defaults.password : camera.baichuan?.password || defaults.password,
      channel: isReolink ? defaults.channel || 0 : camera.baichuan?.channel || 0,
    },
  };
}

function camerasWithGeneratedIds(cameras) {
  const used = new Set();
  return (cameras || []).map((camera) => {
    const base = slugify(camera.name || camera.id || "camera") || "camera";
    let id = base;
    let index = 2;
    while (used.has(id)) {
      id = `${base}-${index}`;
      index += 1;
    }
    used.add(id);
    return { ...cameraWithDerivedConnection(camera), id };
  });
}

function defaultCamera(cameras, seed = {}) {
  const id = uniqueCameraId(cameras, seed.id || seed.name || "camera");
  const host = seed.onvif?.host || seed.baichuan?.host || "";
  return {
    id,
    name: seed.name ? `${seed.name} Copy` : "New Camera",
    video_backend: seed.video_backend || "url",
    stream_url: seed.stream_url || "",
    live_stream_url: seed.live_stream_url || "",
    record: seed.record ?? true,
    onvif: {
      enabled: seed.onvif?.enabled || false,
      host,
      port: seed.onvif?.port || 8000,
      username: seed.onvif?.username || "",
      password: seed.onvif?.password || "",
    },
    baichuan: {
      enabled: seed.baichuan?.enabled || false,
      host,
      port: seed.baichuan?.port || 9000,
      username: seed.baichuan?.username || "",
      password: seed.baichuan?.password || "",
      channel: seed.baichuan?.channel || 0,
    },
  };
}

function Shell({ page, theme, children }) {
  const isRecordings = page === "recordings";
  const isConfig = page === "config";
  const isIncidents = page === "incidents";
  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <div className="brand-mark"><ShieldCheck size={22} /></div>
          <div>
            <h1>{isConfig ? "Config" : isRecordings ? "Recordings" : isIncidents ? "Incidents" : "SurvNG"}</h1>
            <p>{isConfig ? "Camera inventory, cloning, and capability detection" : isRecordings ? "Continuous review of saved camera history" : isIncidents ? "Motion and object incident review" : "Streams, events, recordings, and object detections"}</p>
          </div>
          {!isConfig && !isRecordings && !isIncidents ? <LiveHeaderStats /> : null}
        </div>
        <div className="top-actions">
          <nav className="topnav" aria-label="Primary">
            <a className="nav-button" href="/"><Video size={16} /> Live</a>
            <a className="nav-button incidents-nav" href="/incidents"><Siren size={16} /> Incidents</a>
            <a className="nav-button" href="/recordings"><Film size={16} /> Recordings</a>
            <a className="nav-button" href="/config"><Cog size={16} /> Config</a>
          </nav>
        </div>
      </header>
      {children}
    </div>
  );
}

function LiveHeaderStats() {
  const [stats, setStats] = useState({
    motion: "--",
    objects: "--",
    storage: null,
    detector: null,
    cameras: null,
  });

  async function load() {
    const [eventResponse, systemResponse] = await Promise.all([
      fetch("/api/events?limit=50"),
      fetch("/api/system/status"),
    ]);
    const events = await eventResponse.json();
    const system = systemResponse.ok ? await systemResponse.json() : {};
    setStats({
      motion: events.filter((event) => event.kind === "motion").length,
      objects: events.filter(hasDetectedObjects).length,
      storage: system.storage || null,
      detector: system.detector || null,
      cameras: system.cameras || null,
    });
  }

  useEffect(() => {
    load();
    const timer = window.setInterval(load, 5000);
    return () => window.clearInterval(timer);
  }, []);

  const detector = stats.detector || {};
  const runtime = detector.runtime || {};
  const detectorLoaded = detector.coreml_loaded || detector.openvino_loaded || detector.opencv_loaded;
  const detectorLabel = detector.enabled
    ? `${detector.loaded_backend || detector.configured_backend || "det"}${detector.loaded_device ? ` ${detector.loaded_device}` : ""}`
    : "off";
  const storageLabel = stats.storage ? `${formatBytes(stats.storage.free_bytes)} free` : "--";
  const cameraLabel = stats.cameras ? `${stats.cameras.recording}/${stats.cameras.total} rec` : "--";

  return (
    <div className="header-stats" aria-label="System and recent event summary">
      <span className="header-stat warn"><Siren size={15} /><small>Motion</small><strong>{stats.motion}</strong></span>
      <span className="header-stat hot"><Activity size={15} /><small>Objects</small><strong>{stats.objects}</strong></span>
      <span className="header-stat"><HardDrive size={15} /><small>Storage</small><strong>{storageLabel}</strong></span>
      <span className={`header-stat ${detectorLoaded ? "ok" : "warn"}`}><Gauge size={15} /><small>Detect</small><strong>{detectorLabel}</strong></span>
      <span className="header-stat"><Cpu size={15} /><small>Infer</small><strong>{formatMilliseconds(runtime.last_inference_ms)}</strong></span>
      <span className="header-stat"><Activity size={15} /><small>Avg</small><strong>{formatMilliseconds(runtime.average_inference_ms)}</strong></span>
      <span className="header-stat"><Gauge size={15} /><small>Det/s</small><strong>{formatRate(runtime.detection_fps)}</strong></span>
      <span className={runtime.queue_depth > 0 ? "header-stat warn" : "header-stat"}><ListTree size={15} /><small>Queue</small><strong>{Number.isFinite(runtime.queue_depth) ? runtime.queue_depth : "--"}</strong></span>
      <span className="header-stat hot"><Clock3 size={15} /><small>Last Hit</small><strong>{formatAge(runtime.last_detection_age_seconds)}</strong></span>
      <span className="header-stat"><Camera size={15} /><small>Cameras</small><strong>{cameraLabel}</strong></span>
    </div>
  );
}

function StatCard({ icon, label, value, tone = "default" }) {
  return (
    <section className={`bento-card stat-card ${tone}`}>
      <div className="icon-badge">{icon}</div>
      <span>{label}</span>
      <strong>{value}</strong>
    </section>
  );
}

function usePollingData() {
  const [cameras, setCameras] = useState([]);
  const [incidents, setIncidents] = useState([]);
  const [loading, setLoading] = useState(true);

  async function load() {
    const [cameraResponse, incidentResponse] = await Promise.all([
      fetch("/api/cameras"),
      fetch("/api/incidents?limit=250&gap_seconds=45"),
    ]);
    setCameras(await cameraResponse.json());
    setIncidents(await incidentResponse.json());
    setLoading(false);
  }

  useEffect(() => {
    load();
    const timer = window.setInterval(load, 5000);
    return () => window.clearInterval(timer);
  }, []);

  return { cameras, incidents, loading, refresh: load };
}

function isSafariBrowser() {
  const ua = navigator.userAgent || "";
  return /Safari/.test(ua) && !/Chrome|Chromium|CriOS|FxiOS|Edg|OPR/.test(ua);
}

const STREAM_MODES = ["auto", "mjpeg"];
const STREAM_LABELS = {
  auto: "Auto",
  mjpeg: "MJPEG",
};

function mediaAspect(element) {
  const width = element?.videoWidth || element?.naturalWidth || 0;
  const height = element?.videoHeight || element?.naturalHeight || 0;
  if (!width || !height) return "16 / 9";
  return `${width} / ${height}`;
}

function CameraTile({ camera, timeZone, refresh, onOpen, startDelayMs = 0, dragProps = {}, dragging = false }) {
  const [streamMode, setStreamMode] = useStoredState(`survng.streamMode.v2.${camera.id}`, "auto");
  const [sourceMode, setSourceMode] = useStoredState(`survng.sourceMode.${camera.id}`, "live");
  const normalizedStreamMode = STREAM_MODES.includes(streamMode) ? streamMode : "auto";
  const [aspect, setAspect] = useState("16 / 9");
  const [mjpegToken, setMjpegToken] = useState(() => String(Date.now()));
  const [snapshotToken, setSnapshotToken] = useState(() => String(Date.now()));
  const [streamReady, setStreamReady] = useState(false);
  const safariSnapshotMode = isSafariBrowser() && normalizedStreamMode === "auto";
  const shouldUseMjpegStream = camera.running && streamReady && !safariSnapshotMode;

  useEffect(() => {
    if (!STREAM_MODES.includes(streamMode)) setStreamMode("auto");
  }, [streamMode, setStreamMode]);

  useEffect(() => {
    setMjpegToken(String(Date.now()));
    setSnapshotToken(String(Date.now()));
    setStreamReady(false);
  }, [camera.id, sourceMode, normalizedStreamMode]);

  useEffect(() => {
    setStreamReady(false);
    setSnapshotToken(String(Date.now()));
    if (!camera.running) return undefined;
    const timer = window.setTimeout(() => setStreamReady(true), startDelayMs);
    return () => window.clearTimeout(timer);
  }, [camera.id, camera.running, sourceMode, normalizedStreamMode, startDelayMs]);

  useEffect(() => {
    if (!shouldUseMjpegStream) return undefined;
    const timer = window.setInterval(() => setMjpegToken(String(Date.now())), 30000);
    return () => window.clearInterval(timer);
  }, [shouldUseMjpegStream]);

  useEffect(() => {
    if (!safariSnapshotMode || !camera.running) return undefined;
    const timer = window.setInterval(() => setSnapshotToken(String(Date.now())), 2000);
    return () => window.clearInterval(timer);
  }, [safariSnapshotMode, camera.running]);

  async function post(action) {
    await fetch(`/api/cameras/${camera.id}/${action}`, { method: "POST" });
    refresh();
  }

  function cycleStreamMode() {
    const index = STREAM_MODES.indexOf(normalizedStreamMode);
    setStreamMode(STREAM_MODES[(index + 1) % STREAM_MODES.length]);
  }

  function toggleSourceMode() {
    setSourceMode(sourceMode === "main" ? "live" : "main");
  }

  const imageUrl = shouldUseMjpegStream
    ? `/api/cameras/${camera.id}/stream.mjpg?source=${sourceMode}&t=${mjpegToken}`
    : `/api/cameras/${camera.id}/snapshot.jpg?source=${sourceMode}&t=${snapshotToken}`;

  return (
    <article className={`bento-card camera-tile ${dragging ? "dragging" : ""}`} {...dragProps}>
      <div
        className="video-frame camera-open-target"
        style={{ "--media-aspect": aspect }}
        role="button"
        tabIndex={0}
        onClick={() => onOpen(camera)}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            onOpen(camera);
          }
        }}
        aria-label={`Open ${camera.name} live view`}
      >
        <img
          src={imageUrl}
          alt={`${camera.name} ${sourceMode === "main" ? "main" : "sub"} live stream`}
          onLoad={(event) => setAspect(mediaAspect(event.currentTarget))}
        />
        <div
          className="tile-header camera-hud"
          onClick={(event) => event.stopPropagation()}
          onKeyDown={(event) => event.stopPropagation()}
        >
          <div className="tile-title">
            <h2>{camera.name}</h2>
          </div>
          <div className="tile-controls" aria-label={`${camera.name} controls`}>
            <span className={`status-pill ${camera.running ? "ok" : "bad"}`} title={camera.running ? "Online" : "Offline"}>
              <CircleDot size={13} /> {camera.running ? "online" : "offline"}
            </span>
            <span className={`status-pill ${camera.recording ? "ok" : ""}`} title={camera.recording ? "Recording" : "Not recording"}>
              {camera.recording ? "rec" : "idle"}
            </span>
            <button type="button" className="tile-control-button" onClick={toggleSourceMode} title="Switch main/sub stream">
              <Radio size={15} /> {sourceMode === "main" ? "Main" : "Sub"}
            </button>
            <button type="button" className="tile-control-button" onClick={cycleStreamMode} title="Cycle transport: Auto, MJPEG">
              {STREAM_LABELS[normalizedStreamMode] || "Auto"}
            </button>
            <button
              type="button"
              className={`tile-control-button icon-only ${camera.running ? "danger" : ""}`}
              onClick={() => post(camera.running ? "camera/stop" : "camera/start")}
              title={camera.running ? "Stop camera" : "Start camera"}
              aria-label={`${camera.running ? "Stop" : "Start"} ${camera.name}`}
            >
              <Power size={16} />
            </button>
          </div>
        </div>
      </div>
    </article>
  );
}

function LiveCameraOverlay({ camera, onClose }) {
  const imageRef = useRef(null);
  const [aspect, setAspect] = useState("16 / 9");

  useEffect(() => {
    function onKey(event) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  function updateImageAspect() {
    setAspect(mediaAspect(imageRef.current));
  }

  return (
    <div className="live-overlay" role="dialog" aria-modal="true" aria-label={`${camera.name} full live view`}>
      <button className="live-overlay-backdrop" type="button" onClick={onClose} aria-label="Close live view" />
      <section className="live-overlay-panel" style={{ "--media-aspect": aspect }}>
        <div className="live-overlay-head">
          <div>
            <h2>{camera.name}</h2>
            <span>Shared main stream</span>
          </div>
          <button type="button" className="tile-control-button icon-only" onClick={onClose} aria-label="Close live view">
            <X size={18} />
          </button>
        </div>
        <div className="live-overlay-media">
          <img
            ref={imageRef}
            src={`/api/cameras/${camera.id}/stream.mjpg?source=main`}
            alt={`${camera.name} main live stream`}
            onLoad={updateImageAspect}
          />
        </div>
      </section>
    </div>
  );
}

function Segmented({ value, onChange, options }) {
  return (
    <div className="segmented">
      {options.map(([id, label]) => (
        <button key={id} className={value === id ? "active" : ""} onClick={() => onChange(id)}>{label}</button>
      ))}
    </div>
  );
}

function eventObjects(event) {
  return event.objects || [];
}

function incidentLabels(incident) {
  const labels = incident.labels?.length
    ? incident.labels
    : eventObjects(incident).map((object) => object.label).filter(Boolean);
  return Array.from(new Set(labels.filter(Boolean)));
}

function hasDetectedObjects(event) {
  return Boolean(event.has_objects) || eventObjects(event).some((object) => object.label) || incidentLabels(event).length > 0;
}


function objectBoxes(event) {
  return eventObjects(event)
    .map((object) => ({ object, box: object?.box }))
    .filter(({ object, box }) => object?.label && box && [box.x1, box.y1, box.x2, box.y2].every((value) => Number.isFinite(Number(value))))
    .map(({ object, box }) => ({
      label: object.label,
      confidence: object.confidence,
      x1: Number(box.x1),
      y1: Number(box.y1),
      x2: Number(box.x2),
      y2: Number(box.y2),
    }))
    .filter((box) => box.x2 > box.x1 && box.y2 > box.y1);
}

function SnapshotImage({ event, alt, iconSize = 24, className = "", layerStyle = null, allowObjectFocus = true, children }) {
  const boxes = objectBoxes(event);
  const frameRef = useRef(null);
  const [imageSize, setImageSize] = useState(null);
  const [frameSize, setFrameSize] = useState(null);
  const [objectFocused, setObjectFocused] = useState(false);
  const renderedImage = useMemo(() => {
    if (!imageSize?.width || !imageSize?.height || !frameSize?.width || !frameSize?.height) return null;
    const scale = Math.min(frameSize.width / imageSize.width, frameSize.height / imageSize.height);
    const width = imageSize.width * scale;
    const height = imageSize.height * scale;
    return {
      x: (frameSize.width - width) / 2,
      y: (frameSize.height - height) / 2,
      width,
      height,
      scale,
    };
  }, [frameSize, imageSize]);
  const canFocus = allowObjectFocus && boxes.length > 0 && renderedImage;

  useEffect(() => {
    setObjectFocused(false);
  }, [event?.id, event?.snapshot_path]);

  useEffect(() => {
    const frame = frameRef.current;
    if (!frame) return undefined;
    function updateFrameSize() {
      const rect = frame.getBoundingClientRect();
      if (rect.width && rect.height) setFrameSize({ width: rect.width, height: rect.height });
    }
    updateFrameSize();
    const observer = new ResizeObserver(updateFrameSize);
    observer.observe(frame);
    return () => observer.disconnect();
  }, []);

  function onImageLoad(loadEvent) {
    const image = loadEvent.currentTarget;
    if (image.naturalWidth && image.naturalHeight) {
      setImageSize({ width: image.naturalWidth, height: image.naturalHeight });
    }
  }

  const renderedBoxes = useMemo(() => {
    if (!renderedImage || !frameSize) return [];
    return boxes.map((box) => ({
      ...box,
      left: renderedImage.x + box.x1 * renderedImage.scale,
      top: renderedImage.y + box.y1 * renderedImage.scale,
      width: (box.x2 - box.x1) * renderedImage.scale,
      height: (box.y2 - box.y1) * renderedImage.scale,
    })).filter((box) => box.width > 0 && box.height > 0);
  }, [boxes, frameSize, renderedImage]);

  const focusStyle = useMemo(() => {
    if (!canFocus || !frameSize || !renderedBoxes.length) return null;
    const minX = Math.max(0, Math.min(...renderedBoxes.map((box) => box.left)));
    const minY = Math.max(0, Math.min(...renderedBoxes.map((box) => box.top)));
    const maxX = Math.min(frameSize.width, Math.max(...renderedBoxes.map((box) => box.left + box.width)));
    const maxY = Math.min(frameSize.height, Math.max(...renderedBoxes.map((box) => box.top + box.height)));
    const boxWidth = Math.max(1, maxX - minX);
    const boxHeight = Math.max(1, maxY - minY);
    const padX = Math.max(frameSize.width * 0.04, boxWidth * 0.35);
    const padY = Math.max(frameSize.height * 0.04, boxHeight * 0.35);
    const cropX1 = Math.max(0, minX - padX);
    const cropY1 = Math.max(0, minY - padY);
    const cropX2 = Math.min(frameSize.width, maxX + padX);
    const cropY2 = Math.min(frameSize.height, maxY + padY);
    const cropWidth = Math.max(1, cropX2 - cropX1);
    const cropHeight = Math.max(1, cropY2 - cropY1);
    const centerX = cropX1 + cropWidth / 2;
    const centerY = cropY1 + cropHeight / 2;
    const scale = Math.min(5.5, Math.max(1, Math.min((frameSize.width * 0.82) / cropWidth, (frameSize.height * 0.82) / cropHeight)));
    return {
      transform: `translate3d(${frameSize.width / 2 - centerX * scale}px, ${frameSize.height / 2 - centerY * scale}px, 0) scale(${scale})`,
      transformOrigin: "0 0",
    };
  }, [canFocus, frameSize, renderedBoxes]);

  const activeLayerStyle = objectFocused && focusStyle ? focusStyle : layerStyle;
  const aspect = imageSize ? `${imageSize.width} / ${imageSize.height}` : undefined;

  return (
    <div ref={frameRef} className={`snapshot-frame ${objectFocused ? "object-focused" : ""} ${className}`} style={aspect ? { "--snapshot-aspect": aspect } : undefined}>
      <div className="snapshot-layer" style={activeLayerStyle || undefined}>
        {event?.snapshot_path ? <img src={apiFile(event.snapshot_path)} alt={alt} onLoad={onImageLoad} /> : <div className="empty-thumb"><Camera size={iconSize} /></div>}
        {renderedBoxes.length ? (
          <div className="object-box-layer" aria-hidden="true">
            {renderedBoxes.map((box, index) => (
              <span
                className="object-box"
                key={`${box.label}-${index}-${box.x1}-${box.y1}`}
                style={{ left: `${box.left}px`, top: `${box.top}px`, width: `${box.width}px`, height: `${box.height}px` }}
              >
                <strong>{box.label}{box.confidence ? ` ${(box.confidence * 100).toFixed(0)}%` : ""}</strong>
              </span>
            ))}
          </div>
        ) : null}
      </div>
      {canFocus ? (
        <button
          type="button"
          className={`snapshot-focus-button ${objectFocused ? "active" : ""}`}
          onClick={(event) => {
            event.stopPropagation();
            setObjectFocused((focused) => !focused);
          }}
          title={objectFocused ? "Show full snapshot" : "Focus detected objects"}
          aria-label={objectFocused ? "Show full snapshot" : "Focus detected objects"}
        >
          <Crop size={14} />
        </button>
      ) : null}
      {children}
    </div>
  );
}

function IncidentCard({ incident, timeZone, expanded, onToggle, onSelect }) {
  const rawEvents = incident.events || [];
  const showSubEvents = rawEvents.length > 1;
  const [selectedPreview, setSelectedPreview] = useState(null);
  const [subEventsOpen, setSubEventsOpen] = useState(false);
  const preview = selectedPreview || incident;
  const labels = incidentLabels(incident);
  const eventCount = incident.event_count || rawEvents.length || 1;
  const countText = `${eventCount} ${eventCount === 1 ? "event" : "events"}`;
  const timeText = incident.start_at && incident.end_at && incident.start_at !== incident.end_at
    ? `${formatDateTime(incident.start_at, timeZone)} - ${formatDuration(incident.duration_seconds)}`
    : formatDateTime(incident.created_at, timeZone);
  const previewTimeText = preview.created_at ? formatDateTime(preview.created_at, timeZone) : timeText;

  useEffect(() => {
    if (!expanded) {
      setSelectedPreview(null);
      setSubEventsOpen(false);
    }
  }, [expanded]);

  useEffect(() => {
    setSelectedPreview(null);
    setSubEventsOpen(false);
  }, [incident.id]);

  function toggle() {
    onToggle(incident.id);
  }

  function onKey(keyEvent) {
    if (keyEvent.key === "Enter" || keyEvent.key === " ") {
      keyEvent.preventDefault();
      toggle();
    }
  }

  function openPreview(pointerEvent) {
    pointerEvent.stopPropagation();
    if (expanded) onSelect(preview);
    else toggle();
  }

  return (
    <article
      className={`incident-card ${hasDetectedObjects(incident) ? "has-objects" : ""} ${expanded ? "expanded" : ""}`}
      role="button"
      tabIndex={0}
      onClick={toggle}
      onKeyDown={onKey}
      title={`${incident.camera_id} ${timeText}`}
    >
      <div className="incident-preview" onClick={openPreview} aria-label={expanded ? "Open selected event snapshot" : "Expand incident"}>
        <SnapshotImage event={preview} alt="incident snapshot">
          <div className="incident-snapshot-hud">
            <div className="incident-snapshot-main">
              <strong>{incident.camera_id}</strong>
              <time>{expanded ? previewTimeText : timeText}</time>
            </div>
            <div className="pill-row compact incident-labels">
              {labels.length ? labels.slice(0, 3).map((item) => <span className="pill" key={item}>{item}</span>) : <span className="pill quiet">motion</span>}
            </div>
          </div>
          <span className="event-count">{countText}</span>
        </SnapshotImage>
      </div>
      {expanded && showSubEvents ? (
        <div className="incident-meta">
          <div className="incident-detail" onClick={(event) => event.stopPropagation()}>
            <button
              className="incident-events-toggle"
              type="button"
              onClick={() => setSubEventsOpen((open) => !open)}
              aria-expanded={subEventsOpen}
            >
              <span>{countText}</span>
              <strong>{subEventsOpen ? "Hide" : "Show"}</strong>
            </button>
            {subEventsOpen ? (
              <div className="incident-events">
                {rawEvents.map((event, index) => {
                  const eventLabels = incidentLabels(event);
                  const eventLabelText = eventLabels.length ? eventLabels.join(", ") : "motion";
                  const isActive = (preview.id || incident.id) === event.id && (preview.created_at || incident.created_at) === event.created_at;
                  return (
                    <button type="button" key={`${event.id || "event"}-${index}`} className={isActive ? "active" : ""} onClick={() => setSelectedPreview(event)}>
                      <span>{formatTimeOnly(event.created_at || incident.created_at, timeZone)}</span>
                      <strong>{eventLabelText}</strong>
                    </button>
                  );
                })}
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
    </article>
  );
}

function EventOverlay({ event, events, timeZone, onClose, onSelect, onRefresh }) {
  const clipVideoRef = useRef(null);
  const mediaRef = useRef(null);
  const gestureRef = useRef({ pointerId: null, startX: 0, startY: 0, panX: 0, panY: 0, moved: false, pinchDistance: 0, scale: 1 });
  const [clipInfo, setClipInfo] = useState(null);
  const [clipLoading, setClipLoading] = useState(false);
  const [clipError, setClipError] = useState("");
  const [videoActive, setVideoActive] = useState(false);
  const [zoom, setZoom] = useState({ scale: 1, x: 0, y: 0 });
  const [manualConfidence, setManualConfidence] = useStoredState("survng.manualDetectionConfidence.v1", "0.35");
  const [manualDetection, setManualDetection] = useState(null);
  const [manualLoading, setManualLoading] = useState(false);
  const [manualError, setManualError] = useState("");
  const displayedEvent = manualDetection ? { ...event, objects: manualDetection.objects || [] } : event;
  const objects = eventObjects(displayedEvent);
  const detectedObjects = objects.filter((object) => object.label);
  const manualConfidenceNumber = Number(manualConfidence);
  const safeManualConfidence = Number.isFinite(manualConfidenceNumber) ? Math.max(0.01, Math.min(0.99, manualConfidenceNumber)) : 0.35;
  const manualEventId = Number(event.representative_event_id || event.id);
  const downloadName = `survng-${String(event.camera_id || "camera")}-${String(event.created_at || event.id || "event").replace(/[^0-9A-Za-z_-]+/g, "-")}.mp4`;

  useEffect(() => {
    let cancelled = false;
    async function loadClipSettings() {
      const eventId = Number(event.representative_event_id || event.id);
      if (!Number.isFinite(eventId)) {
        setClipInfo(null);
        setClipLoading(false);
        setClipError("No event video available");
        return;
      }
      setClipInfo(null);
      setClipLoading(true);
      setClipError("");
      setVideoActive(false);
      let before = 5;
      let after = 5;
      try {
        const response = await fetch("/api/event-clip/settings");
        if (response.ok) {
          const settings = await response.json();
          before = Number(settings.before_seconds ?? before);
          after = Number(settings.after_seconds ?? after);
        }
      } catch {
        // Keep the default review window if settings are temporarily unavailable.
      }
      if (cancelled) return;
      const safeBefore = Number.isFinite(before) ? before : 5;
      const safeAfter = Number.isFinite(after) ? after : 5;
      setClipInfo({
        url: eventClipUrl(eventId, safeBefore, safeAfter),
        before: safeBefore,
        after: safeAfter,
        duration: safeBefore + safeAfter,
      });
      setClipLoading(false);
    }
    loadClipSettings();
    return () => { cancelled = true; };
  }, [event.id, event.representative_event_id]);

  useEffect(() => {
    const video = clipVideoRef.current;
    if (!video || !clipInfo || !videoActive) return undefined;
    setClipLoading(true);
    video.src = clipInfo.url;
    video.load();
    function onReady() {
      setClipLoading(false);
      setClipError("");
      video.play().catch(() => {});
    }
    function onError() {
      setClipLoading(false);
      setVideoActive(false);
      setClipError("No recording window found");
    }
    video.addEventListener("canplay", onReady, { once: true });
    video.addEventListener("error", onError);
    return () => {
      video.removeEventListener("canplay", onReady);
      video.removeEventListener("error", onError);
    };
  }, [clipInfo, videoActive]);

  function playEventClip() {
    if (!clipInfo || clipError) return;
    setVideoActive(true);
    const video = clipVideoRef.current;
    if (!video) return;
    if (video.ended || video.currentTime >= Math.max(0, video.duration - 0.1)) video.currentTime = 0;
    video.play().catch(() => {});
  }


  function clampZoom(nextZoom) {
    const scale = Math.max(1, Math.min(6, nextZoom.scale));
    if (scale === 1) return { scale: 1, x: 0, y: 0 };
    const limit = 46 * scale;
    return {
      scale,
      x: Math.max(-limit, Math.min(limit, nextZoom.x || 0)),
      y: Math.max(-limit, Math.min(limit, nextZoom.y || 0)),
    };
  }

  function zoomAround(clientX, clientY, factor) {
    const box = mediaRef.current?.getBoundingClientRect();
    if (!box) return;
    setVideoActive(false);
    setZoom((current) => {
      const nextScale = Math.max(1, Math.min(6, current.scale * factor));
      if (nextScale === 1) return { scale: 1, x: 0, y: 0 };
      const anchorX = clientX - box.left - box.width / 2;
      const anchorY = clientY - box.top - box.height / 2;
      const scaleRatio = nextScale / current.scale;
      return clampZoom({
        scale: nextScale,
        x: anchorX - (anchorX - current.x) * scaleRatio,
        y: anchorY - (anchorY - current.y) * scaleRatio,
      });
    });
  }

  function onMediaWheel(wheelEvent) {
    if (videoActive) return;
    wheelEvent.preventDefault();
    zoomAround(wheelEvent.clientX, wheelEvent.clientY, wheelEvent.deltaY < 0 ? 1.16 : 0.86);
  }

  function touchDistance(touches) {
    if (touches.length < 2) return 0;
    const dx = touches[0].clientX - touches[1].clientX;
    const dy = touches[0].clientY - touches[1].clientY;
    return Math.hypot(dx, dy);
  }

  function touchCenter(touches) {
    return {
      x: (touches[0].clientX + touches[1].clientX) / 2,
      y: (touches[0].clientY + touches[1].clientY) / 2,
    };
  }

  function onMediaTouchStart(touchEvent) {
    if (videoActive) return;
    if (touchEvent.touches.length === 2) {
      touchEvent.preventDefault();
      const center = touchCenter(touchEvent.touches);
      gestureRef.current = { ...gestureRef.current, pinchDistance: touchDistance(touchEvent.touches), scale: zoom.scale, startX: center.x, startY: center.y, panX: zoom.x, panY: zoom.y, moved: true };
    }
  }

  function onMediaTouchMove(touchEvent) {
    if (videoActive || touchEvent.touches.length !== 2) return;
    touchEvent.preventDefault();
    const distance = touchDistance(touchEvent.touches);
    const center = touchCenter(touchEvent.touches);
    const gesture = gestureRef.current;
    if (!gesture.pinchDistance) return;
    setZoom(clampZoom({
      scale: gesture.scale * (distance / gesture.pinchDistance),
      x: gesture.panX + center.x - gesture.startX,
      y: gesture.panY + center.y - gesture.startY,
    }));
  }

  function onMediaPointerDown(pointerEvent) {
    if (videoActive || pointerEvent.pointerType === "touch" || zoom.scale <= 1) return;
    pointerEvent.preventDefault();
    pointerEvent.currentTarget.setPointerCapture(pointerEvent.pointerId);
    gestureRef.current = { pointerId: pointerEvent.pointerId, startX: pointerEvent.clientX, startY: pointerEvent.clientY, panX: zoom.x, panY: zoom.y, moved: false };
  }

  function onMediaPointerMove(pointerEvent) {
    const gesture = gestureRef.current;
    if (gesture.pointerId !== pointerEvent.pointerId || zoom.scale <= 1) return;
    pointerEvent.preventDefault();
    const dx = pointerEvent.clientX - gesture.startX;
    const dy = pointerEvent.clientY - gesture.startY;
    if (Math.abs(dx) + Math.abs(dy) > 4) gesture.moved = true;
    setZoom((current) => clampZoom({ scale: current.scale, x: gesture.panX + dx, y: gesture.panY + dy }));
  }

  function onMediaPointerUp(pointerEvent) {
    if (gestureRef.current.pointerId === pointerEvent.pointerId) {
      gestureRef.current.pointerId = null;
    }
  }

  function onMediaClick() {
    if (gestureRef.current.moved) {
      gestureRef.current.moved = false;
      return;
    }
    if (zoom.scale > 1 || videoActive) return;
    if (clipInfo && !clipError) playEventClip();
  }

  function resetZoom() {
    setZoom({ scale: 1, x: 0, y: 0 });
  }

  useEffect(() => {
    resetZoom();
    setManualDetection(null);
    setManualError("");
    setManualLoading(false);
  }, [event.id]);

  async function runManualDetection() {
    if (!Number.isFinite(manualEventId)) {
      setManualError("No event id available");
      return;
    }
    setManualLoading(true);
    setManualError("");
    try {
      const response = await fetch(`/api/events/${manualEventId}/detect?confidence=${safeManualConfidence.toFixed(2)}`, { method: "POST" });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "Manual detection failed");
      setManualDetection(payload);
      if (payload.event) onSelect(payload.event);
      if (onRefresh) onRefresh();
      setVideoActive(false);
      resetZoom();
    } catch (error) {
      setManualError(error.message || "Manual detection failed");
    } finally {
      setManualLoading(false);
    }
  }

  useEffect(() => {
    function onKey(keyEvent) {
      if (keyEvent.key === "Escape") {
        onClose();
        return;
      }
      if (keyEvent.key !== "ArrowLeft" && keyEvent.key !== "ArrowRight") return;
      const navItems = events.flatMap((candidate) => [candidate, ...(candidate.events || [])]);
      if (!navItems.length) return;
      const currentIndex = navItems.findIndex((candidate) => candidate.id === event.id);
      if (currentIndex < 0) return;
      keyEvent.preventDefault();
      const direction = keyEvent.key === "ArrowRight" ? 1 : -1;
      const nextIndex = (currentIndex + direction + navItems.length) % navItems.length;
      onSelect(navItems[nextIndex]);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [event.id, events, onClose, onSelect]);

  return (
    <div className="event-overlay" role="dialog" aria-modal="true" aria-label="Event image">
      <button className="live-overlay-backdrop" type="button" onClick={onClose} aria-label="Close event image" />
      <section className="event-overlay-panel">
        <div className="event-detail-head">
          <div>
            <h2>{event.camera_id}</h2>
            <time>{formatDateTime(event.created_at, timeZone)}</time>
          </div>
          <div className="overlay-actions">
            {clipInfo && !clipError ? (
              <a className="tile-control-button icon-only" href={clipInfo.url} download={downloadName} title="Download event video" aria-label="Download event video">
                <Download size={18} />
              </a>
            ) : (
              <span className="tile-control-button icon-only disabled" title={clipError || "Event video not ready"} aria-label="Event video not ready">
                <Download size={18} />
              </span>
            )}
            <button type="button" className="tile-control-button icon-only" onClick={onClose} aria-label="Close event image">
              <X size={18} />
            </button>
          </div>
        </div>
        <div
          ref={mediaRef}
          className={`event-detail-media event-detail-play-target ${zoom.scale > 1 ? "zoomed" : ""}`}
          role="button"
          tabIndex={0}
          onClick={onMediaClick}
          onDoubleClick={resetZoom}
          onWheel={onMediaWheel}
          onTouchStart={onMediaTouchStart}
          onTouchMove={onMediaTouchMove}
          onPointerDown={onMediaPointerDown}
          onPointerMove={onMediaPointerMove}
          onPointerUp={onMediaPointerUp}
          onPointerCancel={onMediaPointerUp}
          onKeyDown={(keyEvent) => {
            if (keyEvent.key === "0") resetZoom();
            if ((keyEvent.key === "Enter" || keyEvent.key === " ") && clipInfo && !clipError && zoom.scale === 1) {
              keyEvent.preventDefault();
              playEventClip();
            }
          }}
          title={zoom.scale > 1 ? "Drag to pan. Double-click to reset zoom." : clipError || (clipLoading ? "Preparing event clip" : "Scroll or pinch to zoom. Click to play event video.")}
        >
          <SnapshotImage
            event={displayedEvent}
            alt="selected event snapshot"
            iconSize={42}
            className="event-snapshot-frame"
            layerStyle={{ transform: `translate3d(${zoom.x}px, ${zoom.y}px, 0) scale(${zoom.scale})` }}
            allowObjectFocus={zoom.scale === 1 && !videoActive}
          />
          {videoActive && clipInfo && !clipError ? (
            <video className="event-video-layer" ref={clipVideoRef} controls playsInline preload="metadata" onClick={(event) => event.stopPropagation()} />
          ) : null}
        </div>
        <div className="event-detail-body">
          <div>
            <span className="muted">Detection</span>
            <div className="pill-row">
              {detectedObjects.length ? detectedObjects.map((object, index) => (
                <span className="pill" key={`${object.label}-${index}`}>
                  {object.label}
                  {object.confidence ? ` ${(object.confidence * 100).toFixed(1)}%` : ""}
                </span>
              )) : <span className="pill quiet">motion only</span>}
            </div>
          </div>
          <div className="manual-detect-panel" onClick={(event) => event.stopPropagation()}>
            <div className="manual-detect-head">
              <div>
                <span className="muted">Manual OpenVINO</span>
                <strong>{Math.round(safeManualConfidence * 100)}% confidence</strong>
              </div>
              <button type="button" className="tile-control-button" onClick={runManualDetection} disabled={manualLoading || !Number.isFinite(manualEventId)}>
                <Search size={15} /> {manualLoading ? "Running" : "Run"}
              </button>
            </div>
            <input
              type="range"
              min="0.05"
              max="0.95"
              step="0.01"
              value={safeManualConfidence}
              onChange={(event) => setManualConfidence(event.target.value)}
              aria-label="Manual detection confidence"
            />
            {manualError ? <span className="manual-debug error">{manualError}</span> : null}
            {manualDetection ? (
              <div className="manual-debug">
                <span>{manualDetection.object_count} objects</span>
                <span>{manualDetection.elapsed_ms} ms</span>
                <span>{manualDetection.detector?.loaded_backend || "detector"}</span>
                <span>{manualDetection.detector?.loaded_device || manualDetection.detector?.configured_device || "device"}</span>
                <span>{manualDetection.snapshot_width}x{manualDetection.snapshot_height}</span>
                {manualDetection.labels?.length ? <code>{manualDetection.labels.join(", ")}</code> : <code>no labels</code>}
              </div>
            ) : null}
          </div>
        </div>
      </section>
    </div>
  );
}

function IncidentsPage({ timeZone }) {
  const { cameras, incidents, refresh } = usePollingData();
  const [eventFilter, setEventFilter] = useStoredState("survng.liveEventFilter.v2", "object");
  const [incidentCameraFilter, setIncidentCameraFilter] = useStoredState("survng.incidentCameraFilter.v1", "all");
  const [incidentObjectFilter, setIncidentObjectFilter] = useStoredState("survng.incidentObjectFilter.v1", "all");
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [expandedIncidentId, setExpandedIncidentId] = useState(null);
  const [incidentPage, setIncidentPage] = useState(0);
  const incidentsPerPage = 12;
  const cameraNameById = useMemo(() => new Map(cameras.map((camera) => [camera.id, camera.name || camera.id])), [cameras]);
  const incidentCameraOptions = useMemo(() => {
    const ids = Array.from(new Set(incidents.map((incident) => incident.camera_id).filter(Boolean)));
    return ids.sort((left, right) => (cameraNameById.get(left) || left).localeCompare(cameraNameById.get(right) || right));
  }, [incidents, cameraNameById]);
  const incidentObjectOptions = useMemo(() => {
    const labels = new Set();
    incidents.forEach((incident) => incidentLabels(incident).forEach((label) => labels.add(label)));
    return Array.from(labels).sort((left, right) => left.localeCompare(right));
  }, [incidents]);
  const visibleIncidents = useMemo(() => incidents.filter((incident) => {
    if (eventFilter === "object" && !hasDetectedObjects(incident)) return false;
    if (incidentCameraFilter !== "all" && incident.camera_id !== incidentCameraFilter) return false;
    if (incidentObjectFilter !== "all" && !incidentLabels(incident).includes(incidentObjectFilter)) return false;
    return true;
  }), [incidents, eventFilter, incidentCameraFilter, incidentObjectFilter]);
  const focusedIncident = visibleIncidents.find((incident) => incident.id === expandedIncidentId) || null;
  const galleryIncidents = focusedIncident
    ? visibleIncidents.filter((incident) => incident.id !== focusedIncident.id)
    : visibleIncidents;
  const incidentPageCount = Math.max(1, Math.ceil(galleryIncidents.length / incidentsPerPage));
  const clampedIncidentPage = Math.min(incidentPage, incidentPageCount - 1);
  const pagedIncidents = galleryIncidents.slice(clampedIncidentPage * incidentsPerPage, (clampedIncidentPage + 1) * incidentsPerPage);

  useEffect(() => {
    setIncidentPage(0);
  }, [eventFilter, incidentCameraFilter, incidentObjectFilter, expandedIncidentId]);

  useEffect(() => {
    if (incidentPage >= incidentPageCount) setIncidentPage(Math.max(0, incidentPageCount - 1));
  }, [incidentPage, incidentPageCount]);

  useEffect(() => {
    if (selectedEvent && !visibleIncidents.some((incident) => {
      if (incident.id === selectedEvent.id) return true;
      return (incident.events || []).some((event) => event.id === selectedEvent.id);
    })) {
      setSelectedEvent(null);
    }
    if (expandedIncidentId && !visibleIncidents.some((incident) => incident.id === expandedIncidentId)) {
      setExpandedIncidentId(null);
    }
  }, [selectedEvent, expandedIncidentId, visibleIncidents]);

  useEffect(() => {
    function onKey(keyEvent) {
      if (keyEvent.key === "Escape" && expandedIncidentId && !selectedEvent) {
        keyEvent.preventDefault();
        setExpandedIncidentId(null);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expandedIncidentId, selectedEvent]);

  function toggleIncident(incidentId) {
    setExpandedIncidentId((current) => current === incidentId ? null : incidentId);
  }

  return (
    <main className="bento-grid incidents-grid">
      <section className="bento-card events-zone incidents-page-zone">
        <div className="section-head compact incident-head">
          <div><h2>Incidents</h2></div>
          <div className="incident-head-actions">
            <div className="incident-filter-toggle compact" aria-label="Incident type filter">
              <button className={eventFilter === "object" ? "active" : ""} onClick={() => setEventFilter("object")}>Object</button>
              <button className={eventFilter === "motion" ? "active" : ""} onClick={() => setEventFilter("motion")}>Motion</button>
            </div>
            <span className="shown-bubble">{visibleIncidents.length} shown</span>
          </div>
        </div>
        <div className="event-filter incident-filter-panel" aria-label="Incident filters">
          <div className="incident-filter-selects">
            <label>
              <span>Camera</span>
              <select value={incidentCameraFilter} onChange={(event) => setIncidentCameraFilter(event.target.value)}>
                <option value="all">All cameras</option>
                {incidentCameraOptions.map((id) => <option value={id} key={id}>{cameraNameById.get(id) || id}</option>)}
              </select>
            </label>
            <label>
              <span>Object</span>
              <select value={incidentObjectFilter} onChange={(event) => setIncidentObjectFilter(event.target.value)}>
                <option value="all">All objects</option>
                {incidentObjectOptions.map((label) => <option value={label} key={label}>{label}</option>)}
              </select>
            </label>
          </div>
        </div>
        {focusedIncident ? (
          <div className="incident-focus">
            <IncidentCard
              incident={focusedIncident}
              timeZone={timeZone}
              expanded
              onToggle={toggleIncident}
              onSelect={setSelectedEvent}
            />
          </div>
        ) : null}
        <div className="incident-gallery">
          {visibleIncidents.length
            ? pagedIncidents.map((incident) => (
              <IncidentCard
                key={incident.id}
                incident={incident}
                timeZone={timeZone}
                expanded={false}
                onToggle={toggleIncident}
                onSelect={setSelectedEvent}
              />
            ))
            : <div className="empty-state">No incidents match the current filters.</div>}
        </div>
        {galleryIncidents.length > incidentsPerPage ? (
          <div className="incident-pager" aria-label="Incident pages">
            <button type="button" onClick={() => setIncidentPage((page) => Math.max(0, page - 1))} disabled={clampedIncidentPage === 0}>Prev</button>
            <span>{clampedIncidentPage + 1} / {incidentPageCount}</span>
            <button type="button" onClick={() => setIncidentPage((page) => Math.min(incidentPageCount - 1, page + 1))} disabled={clampedIncidentPage >= incidentPageCount - 1}>Next</button>
          </div>
        ) : null}
      </section>
      {selectedEvent ? <EventOverlay event={selectedEvent} events={visibleIncidents} timeZone={timeZone} onClose={() => setSelectedEvent(null)} onSelect={setSelectedEvent} onRefresh={refresh} /> : null}
    </main>
  );
}

function LivePage({ timeZone }) {
  const { cameras, incidents, refresh } = usePollingData();
  const [eventFilter, setEventFilter] = useStoredState("survng.liveEventFilter.v2", "object");
  const [incidentCameraFilter, setIncidentCameraFilter] = useStoredState("survng.incidentCameraFilter.v1", "all");
  const [incidentObjectFilter, setIncidentObjectFilter] = useStoredState("survng.incidentObjectFilter.v1", "all");
  const [cameraOrder, setCameraOrder] = useStoredState("survng.liveCameraOrder.v1", "[]");
  const [dragCameraId, setDragCameraId] = useState("");
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [expandedIncidentId, setExpandedIncidentId] = useState(null);
  const [expandedCamera, setExpandedCamera] = useState(null);
  const [incidentPage, setIncidentPage] = useState(0);
  const incidentsPerPage = 12;
  const orderedCameras = useMemo(() => {
    let order = [];
    try {
      order = JSON.parse(cameraOrder);
    } catch {
      order = [];
    }
    const cameraById = new Map(cameras.map((camera) => [camera.id, camera]));
    const sorted = order.map((id) => cameraById.get(id)).filter(Boolean);
    const seen = new Set(sorted.map((camera) => camera.id));
    return [...sorted, ...cameras.filter((camera) => !seen.has(camera.id))];
  }, [cameras, cameraOrder]);
  const cameraNameById = useMemo(() => new Map(cameras.map((camera) => [camera.id, camera.name || camera.id])), [cameras]);
  const incidentCameraOptions = useMemo(() => {
    const ids = Array.from(new Set(incidents.map((incident) => incident.camera_id).filter(Boolean)));
    return ids.sort((left, right) => (cameraNameById.get(left) || left).localeCompare(cameraNameById.get(right) || right));
  }, [incidents, cameraNameById]);
  const incidentObjectOptions = useMemo(() => {
    const labels = new Set();
    incidents.forEach((incident) => incidentLabels(incident).forEach((label) => labels.add(label)));
    return Array.from(labels).sort((left, right) => left.localeCompare(right));
  }, [incidents]);
  const visibleIncidents = useMemo(() => incidents.filter((incident) => {
    if (eventFilter === "object" && !hasDetectedObjects(incident)) return false;
    if (incidentCameraFilter !== "all" && incident.camera_id !== incidentCameraFilter) return false;
    if (incidentObjectFilter !== "all" && !incidentLabels(incident).includes(incidentObjectFilter)) return false;
    return true;
  }), [incidents, eventFilter, incidentCameraFilter, incidentObjectFilter]);
  const focusedIncident = visibleIncidents.find((incident) => incident.id === expandedIncidentId) || null;
  const galleryIncidents = focusedIncident
    ? visibleIncidents.filter((incident) => incident.id !== focusedIncident.id)
    : visibleIncidents;
  const incidentPageCount = Math.max(1, Math.ceil(galleryIncidents.length / incidentsPerPage));
  const clampedIncidentPage = Math.min(incidentPage, incidentPageCount - 1);
  const pagedIncidents = galleryIncidents.slice(clampedIncidentPage * incidentsPerPage, (clampedIncidentPage + 1) * incidentsPerPage);

  function moveCameraBefore(sourceId, targetId) {
    if (!sourceId || !targetId || sourceId === targetId) return;
    const ids = orderedCameras.map((camera) => camera.id);
    const withoutSource = ids.filter((id) => id !== sourceId);
    const targetIndex = withoutSource.indexOf(targetId);
    if (targetIndex < 0) return;
    withoutSource.splice(targetIndex, 0, sourceId);
    setCameraOrder(JSON.stringify(withoutSource));
  }

  useEffect(() => {
    setIncidentPage(0);
  }, [eventFilter, incidentCameraFilter, incidentObjectFilter, expandedIncidentId]);

  useEffect(() => {
    if (incidentPage >= incidentPageCount) setIncidentPage(Math.max(0, incidentPageCount - 1));
  }, [incidentPage, incidentPageCount]);

  useEffect(() => {
    if (selectedEvent && !visibleIncidents.some((incident) => {
      if (incident.id === selectedEvent.id) return true;
      return (incident.events || []).some((event) => event.id === selectedEvent.id);
    })) {
      setSelectedEvent(null);
    }
    if (expandedIncidentId && !visibleIncidents.some((incident) => incident.id === expandedIncidentId)) {
      setExpandedIncidentId(null);
    }
  }, [selectedEvent, expandedIncidentId, visibleIncidents]);

  useEffect(() => {
    function onKey(keyEvent) {
      if (keyEvent.key === "Escape" && expandedIncidentId && !selectedEvent) {
        keyEvent.preventDefault();
        setExpandedIncidentId(null);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expandedIncidentId, selectedEvent]);

  function toggleIncident(incidentId) {
    setExpandedIncidentId((current) => current === incidentId ? null : incidentId);
  }

  return (
    <main className="bento-grid live-grid">
      <section className="bento-card camera-zone">
        <div className="camera-grid">
          {orderedCameras.map((camera, index) => (
            <CameraTile
              key={camera.id}
              camera={camera}
              timeZone={timeZone}
              refresh={refresh}
              onOpen={setExpandedCamera}
              startDelayMs={index * 450}
              dragging={dragCameraId === camera.id}
              dragProps={{
                draggable: true,
                onDragStart: (event) => {
                  setDragCameraId(camera.id);
                  event.dataTransfer.effectAllowed = "move";
                  event.dataTransfer.setData("text/plain", camera.id);
                },
                onDragOver: (event) => {
                  if (!dragCameraId || dragCameraId === camera.id) return;
                  event.preventDefault();
                  event.dataTransfer.dropEffect = "move";
                },
                onDrop: (event) => {
                  event.preventDefault();
                  const sourceId = event.dataTransfer.getData("text/plain") || dragCameraId;
                  moveCameraBefore(sourceId, camera.id);
                  setDragCameraId("");
                },
                onDragEnd: () => setDragCameraId(""),
                title: "Drag to reorder camera tiles",
              }}
            />
          ))}
        </div>
      </section>
      <section className="bento-card events-zone">
        <div className="section-head compact incident-head">
          <div><h2>Incidents</h2></div>
          <div className="incident-head-actions">
            <div className="incident-filter-toggle compact" aria-label="Incident type filter">
              <button className={eventFilter === "object" ? "active" : ""} onClick={() => setEventFilter("object")}>Object</button>
              <button className={eventFilter === "motion" ? "active" : ""} onClick={() => setEventFilter("motion")}>Motion</button>
            </div>
            <span className="shown-bubble">{visibleIncidents.length} shown</span>
          </div>
        </div>
        <div className="event-filter incident-filter-panel" aria-label="Incident filters">
          <div className="incident-filter-selects">
            <label>
              <span>Camera</span>
              <select value={incidentCameraFilter} onChange={(event) => setIncidentCameraFilter(event.target.value)}>
                <option value="all">All cameras</option>
                {incidentCameraOptions.map((id) => <option value={id} key={id}>{cameraNameById.get(id) || id}</option>)}
              </select>
            </label>
            <label>
              <span>Object</span>
              <select value={incidentObjectFilter} onChange={(event) => setIncidentObjectFilter(event.target.value)}>
                <option value="all">All objects</option>
                {incidentObjectOptions.map((label) => <option value={label} key={label}>{label}</option>)}
              </select>
            </label>
          </div>
        </div>
        {focusedIncident ? (
          <div className="incident-focus">
            <IncidentCard
              incident={focusedIncident}
              timeZone={timeZone}
              expanded
              onToggle={toggleIncident}
              onSelect={setSelectedEvent}
            />
          </div>
        ) : null}
        <div className="incident-gallery">
          {visibleIncidents.length
            ? pagedIncidents.map((incident) => (
              <IncidentCard
                key={incident.id}
                incident={incident}
                timeZone={timeZone}
                expanded={false}
                onToggle={toggleIncident}
                onSelect={setSelectedEvent}
              />
            ))
            : <div className="empty-state">No incidents match the current filters.</div>}
        </div>
        {galleryIncidents.length > incidentsPerPage ? (
          <div className="incident-pager" aria-label="Incident pages">
            <button type="button" onClick={() => setIncidentPage((page) => Math.max(0, page - 1))} disabled={clampedIncidentPage === 0}>Prev</button>
            <span>{clampedIncidentPage + 1} / {incidentPageCount}</span>
            <button type="button" onClick={() => setIncidentPage((page) => Math.min(incidentPageCount - 1, page + 1))} disabled={clampedIncidentPage >= incidentPageCount - 1}>Next</button>
          </div>
        ) : null}
      </section>
      {selectedEvent ? <EventOverlay event={selectedEvent} events={visibleIncidents} timeZone={timeZone} onClose={() => setSelectedEvent(null)} onSelect={setSelectedEvent} onRefresh={refresh} /> : null}
      {expandedCamera ? <LiveCameraOverlay camera={expandedCamera} onClose={() => setExpandedCamera(null)} /> : null}
    </main>
  );
}

function eventClipUrl(eventId, before = 5, after = 5) {
  const params = new URLSearchParams({ before: before.toFixed(3), after: after.toFixed(3) });
  return `/api/events/${eventId}/clip.mp4?${params.toString()}`;
}

function recordingStreamUrl(cameraId, offset, duration = null) {
  const params = new URLSearchParams({ offset: offset.toFixed(3) });
  if (duration !== null && Number.isFinite(duration)) params.set("duration", duration.toFixed(3));
  return `/api/cameras/${cameraId}/recordings/stream.mp4?${params.toString()}`;
}

function recordingClipHlsUrl(cameraId, offset, duration = null) {
  const params = new URLSearchParams({ offset: offset.toFixed(3) });
  if (duration !== null && Number.isFinite(duration)) params.set("duration", duration.toFixed(3));
  return `/api/cameras/${cameraId}/recordings/clip.m3u8?${params.toString()}`;
}

function recordingHlsUrl(cameraId) {
  return `/api/cameras/${cameraId}/recordings/hls/index.m3u8`;
}

function RecordingsPage({ timeZone }) {
  const DAY_SECONDS = 24 * 60 * 60;
  const videoRef = useRef(null);
  const [cameras, setCameras] = useState([]);
  const [cameraId, setCameraId] = useStoredState("survng.recordingsCamera", "");
  const [selectedDate, setSelectedDate] = useStoredState("survng.recordingsDate", dateKeyForTimeZone(Date.now(), timeZone));
  const [recordings, setRecordings] = useState([]);
  const [events, setEvents] = useState([]);
  const [dayTime, setDayTimeState] = useStoredState("survng.recordingsDayTime", "0");
  const [streamBaseGlobal, setStreamBaseGlobal] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(true);
  const [dragValue, setDragValue] = useState(null);

  const activeCameraId = cameraId || cameras[0]?.id || "";
  const dayStartEpoch = useMemo(() => zonedDateSecondToEpoch(selectedDate, 0, timeZone), [selectedDate, timeZone]);
  const currentDayTime = Math.max(0, Math.min(DAY_SECONDS, dragValue ?? Number(dayTime || 0)));

  const { fullTimeline, dayTimeline } = useMemo(() => {
    let globalOffset = 0;
    const all = recordings
      .filter((clip) => Number.isFinite(clip.duration_seconds) && clip.duration_seconds > 0 && Number.isFinite(clip.start_epoch))
      .sort((a, b) => (a.start_epoch || 0) - (b.start_epoch || 0))
      .map((clip) => {
        const duration = Number(clip.duration_seconds || 0);
        const next = {
          ...clip,
          globalOffset,
          globalEndOffset: globalOffset + duration,
          start_epoch: Number(clip.start_epoch),
          end_epoch: Number(clip.end_epoch || (Number(clip.start_epoch) + duration)),
        };
        globalOffset += duration;
        return next;
      });
    const dayStart = dayStartEpoch;
    const dayEnd = dayStartEpoch + DAY_SECONDS;
    const day = all
      .filter((clip) => clip.end_epoch > dayStart && clip.start_epoch < dayEnd)
      .map((clip) => {
        const visibleStart = Math.max(clip.start_epoch, dayStart);
        const visibleEnd = Math.min(clip.end_epoch, dayEnd);
        return {
          ...clip,
          dayStartOffset: visibleStart - dayStart,
          dayEndOffset: visibleEnd - dayStart,
          visibleDuration: Math.max(0, visibleEnd - visibleStart),
        };
      });
    return { fullTimeline: all, dayTimeline: day };
  }, [recordings, dayStartEpoch]);

  const dayEvents = useMemo(() => events
    .map((event) => {
      const eventEpoch = new Date(event.created_at).getTime() / 1000;
      return { ...event, day_offset: eventEpoch - dayStartEpoch };
    })
    .filter((event) => event.day_offset >= 0 && event.day_offset < DAY_SECONDS), [events, dayStartEpoch]);

  const activeClip = dayTimeline.find((clip) => currentDayTime >= clip.dayStartOffset && currentDayTime < clip.dayEndOffset) || null;
  const activeCamera = cameras.find((camera) => camera.id === activeCameraId) || null;
  const objectEventCount = dayEvents.filter((event) => event.has_objects).length;
  const recordedSeconds = dayTimeline.reduce((total, clip) => total + Number(clip.visibleDuration || 0), 0);
  const wallClock = dayStartEpoch + currentDayTime;

  function globalOffsetToDayTime(globalOffset) {
    const clip = dayTimeline.find((item) => globalOffset >= item.globalOffset && globalOffset < item.globalEndOffset);
    if (!clip) return null;
    const epoch = clip.start_epoch + (globalOffset - clip.globalOffset);
    return epoch - dayStartEpoch;
  }

  function snapDayTimeToRecording(nextDayTime) {
    const clamped = Math.max(0, Math.min(DAY_SECONDS - 0.01, nextDayTime));
    if (!dayTimeline.length) return clamped;
    const active = dayTimeline.find((item) => clamped >= item.dayStartOffset && clamped < item.dayEndOffset);
    if (active) return clamped;
    let nearestTime = clamped;
    let nearestDistance = Infinity;
    dayTimeline.forEach((clip) => {
      const start = Math.max(0, Math.min(DAY_SECONDS - 0.01, clip.dayStartOffset));
      const end = Math.max(0, Math.min(DAY_SECONDS - 0.01, clip.dayEndOffset - 0.01));
      [start, end].forEach((candidate) => {
        const distance = Math.abs(candidate - clamped);
        if (distance < nearestDistance) {
          nearestDistance = distance;
          nearestTime = candidate;
        }
      });
    });
    return nearestTime;
  }

  function dayTimeToGlobalOffset(nextDayTime) {
    const clip = dayTimeline.find((item) => nextDayTime >= item.dayStartOffset && nextDayTime < item.dayEndOffset);
    if (!clip) return null;
    const epoch = dayStartEpoch + nextDayTime;
    return clip.globalOffset + Math.max(0, epoch - clip.start_epoch);
  }

  function canUseNativeHls() {
    return Boolean(videoRef.current?.canPlayType("application/vnd.apple.mpegurl"));
  }

  async function load() {
    setLoading(true);
    try {
      const cameraResponse = await fetch("/api/cameras");
      const nextCameras = await cameraResponse.json();
      setCameras(nextCameras);
      const nextCameraId = activeCameraId || nextCameras[0]?.id;
      if (!nextCameraId) return;
      if (!cameraId) setCameraId(nextCameraId);
      const recordingsResponse = await fetch(`/api/cameras/${nextCameraId}/recordings?limit=20000`);
      setRecordings(await recordingsResponse.json());
      const eventResponse = await fetch(`/api/cameras/${nextCameraId}/recordings/events?limit=5000`);
      setEvents(await eventResponse.json());
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, [activeCameraId]);

  useEffect(() => {
    if (!dayTimeline.length) return;
    const saved = Number(dayTime || 0);
    const hasSavedClip = dayTimeline.some((clip) => saved >= clip.dayStartOffset && saved < clip.dayEndOffset);
    setPlaybackTime(hasSavedClip ? saved : dayTimeline[0].dayStartOffset, false);
  }, [dayTimeline.length, activeCameraId, selectedDate]);

  function setPlaybackTime(nextTime, autoplay = playing) {
    if (!activeCameraId) return;
    const clamped = snapDayTimeToRecording(nextTime);
    const video = videoRef.current;
    setDayTimeState(String(clamped));
    setDragValue(null);
    const globalOffset = dayTimeToGlobalOffset(clamped);
    if (!video || globalOffset === null) {
      if (video) video.pause();
      return;
    }
    if (canUseNativeHls()) {
      const src = recordingHlsUrl(activeCameraId);
      if (video.getAttribute("src") !== src) {
        video.src = src;
        video.load();
      }
      video.currentTime = globalOffset;
      setStreamBaseGlobal(0);
    } else {
      video.src = recordingStreamUrl(activeCameraId, globalOffset);
      video.load();
      setStreamBaseGlobal(globalOffset);
    }
    if (autoplay) video.play().catch(() => {});
  }

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return undefined;
    function tick() {
      const globalOffset = canUseNativeHls() ? video.currentTime : streamBaseGlobal + video.currentTime;
      const nextDayTime = globalOffsetToDayTime(globalOffset);
      if (Number.isFinite(nextDayTime)) setDayTimeState(String(Math.max(0, Math.min(DAY_SECONDS, nextDayTime))));
    }
    function onPlay() { setPlaying(true); }
    function onPause() { setPlaying(false); }
    function onEnded() {
      const nextClip = dayTimeline.find((clip) => clip.dayStartOffset > currentDayTime + 1);
      if (nextClip) setPlaybackTime(nextClip.dayStartOffset, false);
    }
    video.addEventListener("timeupdate", tick);
    video.addEventListener("play", onPlay);
    video.addEventListener("pause", onPause);
    video.addEventListener("ended", onEnded);
    return () => {
      video.removeEventListener("timeupdate", tick);
      video.removeEventListener("play", onPlay);
      video.removeEventListener("pause", onPause);
      video.removeEventListener("ended", onEnded);
    };
  }, [streamBaseGlobal, dayTimeline, dayStartEpoch, currentDayTime]);

  function jumpBy(seconds) {
    setPlaybackTime(currentDayTime + seconds);
  }

  function jumpToEvent(event) {
    const offset = Number(event.day_offset);
    if (Number.isFinite(offset)) setPlaybackTime(offset, true);
  }

  function nudgeDay(days) {
    setSelectedDate(addDaysToDateKey(selectedDate, days));
  }

  return (
    <main className="recording-review-page">
      <section className="bento-card recording-review-shell">
        <div className="review-topbar">
          <div>
            <h2>{activeCamera?.name || "Recordings"}</h2>
            <p>{formatDateTime(wallClock, timeZone)}</p>
          </div>
          <div className="review-date-controls">
            <button type="button" onClick={() => nudgeDay(-1)}><SkipBack size={15} /> Day</button>
            <input type="date" value={selectedDate} onChange={(event) => setSelectedDate(event.target.value)} />
            <button type="button" onClick={() => nudgeDay(1)}>Day <SkipForward size={15} /></button>
          </div>
        </div>

        <div className="camera-day-rail" aria-label="Camera selection">
          {cameras.map((camera) => (
            <button key={camera.id} className={camera.id === activeCameraId ? "active" : ""} onClick={() => setCameraId(camera.id)}>
              <Camera size={15} />
              <span>{camera.name}</span>
              <small>{camera.recording ? "rec" : "idle"}</small>
            </button>
          ))}
        </div>

        <div className="daily-review-stage">
          <div className="review-player cohesive-player">
            <video ref={videoRef} controls playsInline preload="metadata" />
            {!dayTimeline.length && !loading ? <div className="review-empty"><Film size={34} />No recordings for this camera on this day.</div> : null}
          </div>

          <div className="day-summary-strip">
            <span><strong>{formatDayClock(currentDayTime)}</strong><small>current</small></span>
            <span><strong>{formatDuration(recordedSeconds)}</strong><small>recorded</small></span>
            <span><strong>{dayEvents.length}</strong><small>motion</small></span>
            <span><strong>{objectEventCount}</strong><small>objects</small></span>
            <span><strong>{activeClip ? "recorded" : dayTimeline.length ? "gap" : "empty"}</strong><small>position</small></span>
          </div>

          <RecordingScrubber
            timeline={dayTimeline}
            events={dayEvents}
            totalDuration={DAY_SECONDS}
            timeZone={timeZone}
            value={Math.round(currentDayTime)}
            onDrag={setDragValue}
            onCommit={() => setPlaybackTime(dragValue ?? currentDayTime)}
            onEventClick={jumpToEvent}
          />

          <div className="review-controls cohesive-controls">
            <button onClick={() => jumpBy(-300)}><SkipBack size={16} /> 5m</button>
            <button onClick={() => jumpBy(-10)}><SkipBack size={16} /> 10s</button>
            <button className="primary" onClick={() => videoRef.current?.paused ? videoRef.current?.play() : videoRef.current?.pause()} disabled={!dayTimeline.length}>{playing ? <Pause size={16} /> : <Play size={16} />}{playing ? "Pause" : "Play"}</button>
            <button onClick={() => jumpBy(10)}>10s <SkipForward size={16} /></button>
            <button onClick={() => jumpBy(300)}>5m <SkipForward size={16} /></button>
          </div>

          <div className="day-event-dock">
            {dayEvents.slice(0, 18).map((event) => (
              <button key={event.id} type="button" className={event.has_objects ? "has-object" : ""} onClick={() => jumpToEvent(event)}>
                <span>{formatDayClock(event.day_offset)}</span>
                <strong>{event.labels?.length ? event.labels.join(", ") : event.has_objects ? "object" : "motion"}</strong>
              </button>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}

function RecordingScrubber({ timeline, events, totalDuration, timeZone, value, onDrag, onCommit, onEventClick }) {
  return (
    <div className="merged-scrubber">
      <span>00:00</span>
      <div className="segment-track scrubber-track">
        {timeline.map((clip) => {
          const offset = Number.isFinite(clip.dayStartOffset) ? clip.dayStartOffset : clip.offset || 0;
          const duration = Number.isFinite(clip.visibleDuration) ? clip.visibleDuration : clip.duration_seconds || 0;
          return (
            <span
              key={`${clip.name}-${offset}`}
              className="recording-span"
              style={{ left: `${totalDuration ? (offset / totalDuration) * 100 : 0}%`, width: `${totalDuration ? (duration / totalDuration) * 100 : 0}%` }}
            />
          );
        })}
        {events.map((event) => {
          const offset = Math.max(0, Math.min(totalDuration, Number.isFinite(event.day_offset) ? event.day_offset : Number(event.timeline_offset) || 0));
          const labels = event.labels?.length ? `: ${event.labels.join(", ")}` : "";
          const title = `${event.has_objects ? "Object detected" : "Motion"}${labels} at ${formatDateTime(event.created_at, timeZone)}`;
          return (
            <button
              key={event.id}
              type="button"
              className={event.has_objects ? "object-event" : "motion-event"}
              title={title}
              style={{ left: `${totalDuration ? (offset / totalDuration) * 100 : 0}%`, width: `${totalDuration ? Math.max(0.25, (12 / totalDuration) * 100) : 0}%` }}
              onClick={() => onEventClick?.(event)}
            />
          );
        })}
        <input
          className="timeline-range"
          type="range"
          min="0"
          max={totalDuration}
          step="1"
          value={value}
          onChange={(event) => onDrag(Number(event.target.value))}
          onMouseUp={onCommit}
          onTouchEnd={onCommit}
          onKeyUp={onCommit}
          onBlur={onCommit}
          aria-label="Daily recording timeline"
        />
      </div>
      <span>24:00</span>
    </div>
  );
}

function ConfigPage({ timeZone, setTimeZone, theme, setTheme }) {
  const [config, setConfig] = useState(null);
  const [runtimeStatus, setRuntimeStatus] = useState([]);
  const [accelerator, setAccelerator] = useState(null);
  const [settingsTab, setSettingsTab] = useStoredState("survng.configTab", "general");
  const [selectedId, setSelectedId] = useState("");
  const [status, setStatus] = useState("");
  const [probe, setProbe] = useState(null);
  const [logLines, setLogLines] = useState([]);
  const [logFilter, setLogFilter] = useStoredState("survng.logFilter.v1", "");
  const [logLevel, setLogLevel] = useStoredState("survng.logLevel.v1", "INFO");

  async function load() {
    const [response, statusResponse, acceleratorResponse] = await Promise.all([
      fetch("/api/config"),
      fetch("/api/cameras"),
      fetch("/api/accelerator"),
    ]);
    const nextConfig = await response.json();
    setConfig(nextConfig);
    if (statusResponse.ok) setRuntimeStatus(await statusResponse.json());
    if (acceleratorResponse.ok) setAccelerator(await acceleratorResponse.json());
    setSelectedId((current) => nextConfig.cameras?.some((camera) => camera.id === current) ? current : nextConfig.cameras?.[0]?.id || "");
  }

  useEffect(() => {
    load();
  }, []);


  async function loadLogs() {
    const params = new URLSearchParams({ limit: "500", level: logLevel, q: logFilter });
    const response = await fetch(`/api/logs?${params.toString()}`);
    if (response.ok) {
      const payload = await response.json();
      setLogLines(payload.lines || []);
    }
  }

  useEffect(() => {
    if (settingsTab !== "logs") return undefined;
    loadLogs();
    const timer = window.setInterval(loadLogs, 2000);
    return () => window.clearInterval(timer);
  }, [settingsTab, logLevel, logFilter]);

  const cameras = config?.cameras || [];
  const selectedCamera = cameras.find((camera) => camera.id === selectedId) || cameras[0] || null;
  const selectedRuntimeStatus = runtimeStatus.find((camera) => camera.id === selectedCamera?.id);



  if (!config) {
    return <main className="bento-grid config-grid"><section className="bento-card config-editor"><div className="empty-state">Loading config...</div></section></main>;
  }

  function updateConfig(path, value) {
    setConfig((current) => {
      const next = structuredClone(current);
      let target = next;
      for (let index = 0; index < path.length - 1; index += 1) target = target[path[index]];
      target[path[path.length - 1]] = value;
      return next;
    });
  }

  function updateCamera(cameraId, path, value) {
    setConfig((current) => {
      const next = structuredClone(current);
      const camera = next.cameras.find((item) => item.id === cameraId);
      let target = camera;
      for (let index = 0; index < path.length - 1; index += 1) target = target[path[index]];
      target[path[path.length - 1]] = value;
      return next;
    });
  }



  function addCamera(seed = {}) {
    const camera = cameraWithDerivedConnection(defaultCamera(cameras, seed));
    setConfig((current) => ({ ...current, cameras: [...(current.cameras || []), camera] }));
    setSelectedId(camera.id);
    setProbe(null);
  }

  function cloneCamera(camera) {
    addCamera(camera);
  }

  function removeCamera(cameraId) {
    const nextCameras = cameras.filter((camera) => camera.id !== cameraId);
    setConfig((current) => ({ ...current, cameras: nextCameras }));
    setSelectedId(nextCameras[0]?.id || "");
    setProbe(null);
  }

  async function save() {
    const ids = new Set();
    const configToSave = {
      ...config,
      cameras: camerasWithGeneratedIds(config.cameras || []),
    };
    for (const camera of configToSave.cameras || []) {
      if (ids.has(camera.id)) {
        setStatus(`Duplicate camera ID "${camera.id}". Fix duplicates before saving.`);
        return;
      }
      ids.add(camera.id);
    }
    setStatus("Saving and reloading cameras...");
    const response = await fetch("/api/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(configToSave),
    });
    if (!response.ok) {
      setStatus(await response.text());
      return;
    }
    setStatus("Saved. Camera workers reloaded.");
    await load();
  }

  async function probeCamera(camera) {
    setProbe({ loading: true });
    const probeCameraConfig = cameraWithDerivedConnection(camera);
    if (probeCameraConfig !== camera) {
      setConfig((current) => ({
        ...current,
        cameras: (current.cameras || []).map((item) => item.id === camera.id ? probeCameraConfig : item),
      }));
    }
    const host = probeCameraConfig.onvif?.host || probeCameraConfig.baichuan?.host || "";
    const username = probeCameraConfig.onvif?.username || probeCameraConfig.baichuan?.username || "";
    const password = probeCameraConfig.onvif?.password || probeCameraConfig.baichuan?.password || "";
    const response = await fetch("/api/config/probe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        host,
        username,
        password,
        onvif_port: probeCameraConfig.onvif?.port || 8000,
        baichuan_port: probeCameraConfig.baichuan?.port || 9000,
      }),
    });
    const result = await response.json();
    setProbe(result);
    if (result.onvif?.reachable) updateCamera(camera.id, ["onvif", "enabled"], true);
    if (result.baichuan?.reachable) {
      updateCamera(camera.id, ["baichuan", "enabled"], true);
      updateCamera(camera.id, ["video_backend"], "baichuan_native");
    }
  }

  return (
    <main className="bento-grid config-grid settings-grid">
      <div className="settings-tabs" role="tablist" aria-label="Config sections">
        <button className={settingsTab === "general" ? "active" : ""} onClick={() => setSettingsTab("general")} role="tab" aria-selected={settingsTab === "general"}><Cog size={16} /> General</button>
        <button className={settingsTab === "cameras" ? "active" : ""} onClick={() => setSettingsTab("cameras")} role="tab" aria-selected={settingsTab === "cameras"}><Camera size={16} /> Camera Settings</button>
        <button className={settingsTab === "logs" ? "active" : ""} onClick={() => setSettingsTab("logs")} role="tab" aria-selected={settingsTab === "logs"}><ListTree size={16} /> Logs</button>
      </div>

      {settingsTab === "general" ? (
        <section className="bento-card config-editor settings-panel">
          <div className="section-head">
            <div><h2>General</h2><p>Application preferences and detector settings</p></div>
            <button className="primary" onClick={save}><Save size={16} /> Save</button>
          </div>
          <GeneralSettings
            config={config}
            updateConfig={updateConfig}
            timeZone={timeZone}
            setTimeZone={setTimeZone}
            theme={theme}
            setTheme={setTheme}
            accelerator={accelerator}
          />
          {status ? <div className="save-status settings-status">{status}</div> : null}
        </section>
      ) : settingsTab === "logs" ? (
        <section className="bento-card config-editor settings-panel log-panel">
          <div className="section-head">
            <div><h2>Logs</h2><p>Live application log stream</p></div>
            <button onClick={loadLogs}><RefreshCcw size={16} /> Refresh</button>
          </div>
          <LogViewer
            lines={logLines}
            filter={logFilter}
            setFilter={setLogFilter}
            level={logLevel}
            setLevel={setLogLevel}
            timeZone={timeZone}
          />
        </section>
      ) : (
        <>
      <section className="bento-card camera-tree config-tree">
        <div className="section-head compact">
          <div><h2>Cameras</h2><p>Add, clone, or select</p></div>
          <button onClick={() => addCamera()}><Plus size={16} /> Add</button>
        </div>
        <div className="tree-list">
          {cameras.map((camera) => (
            <button key={camera.id} className={camera.id === selectedCamera?.id ? "active" : ""} onClick={() => { setSelectedId(camera.id); setProbe(null); }}>
              <Camera size={16} />
              <span>{camera.name || camera.id}</span>
            </button>
          ))}
        </div>
      </section>
      <section className="bento-card config-editor">
        <div className="section-head">
          <div><h2>{selectedCamera ? selectedCamera.name : "Camera Config"}</h2><p>Changes save to config.json and reload camera workers</p></div>
          <button className="primary" onClick={save}><Save size={16} /> Save</button>
        </div>

        <div className="config-form">
          {selectedCamera ? (
            <>
              <div className="field-row">
                <label>Name<input value={selectedCamera.name} onChange={(event) => updateCamera(selectedCamera.id, ["name"], event.target.value)} /></label>
                <label>Generated Camera ID<input value={slugify(selectedCamera.name || selectedCamera.id || "camera")} readOnly /></label>
                <label>Detected Backend<input value={inferredBackendLabel(selectedCamera)} readOnly /></label>
              </div>
              <div className="field-row">
                <label>Main Stream URL<input value={selectedCamera.stream_url || ""} onChange={(event) => updateCamera(selectedCamera.id, ["stream_url"], event.target.value)} /></label>
                <label>Live/Sub Stream URL<input value={selectedCamera.live_stream_url || ""} onChange={(event) => updateCamera(selectedCamera.id, ["live_stream_url"], event.target.value)} /></label>
                <label className="check-field"><input type="checkbox" checked={selectedCamera.record} onChange={(event) => updateCamera(selectedCamera.id, ["record"], event.target.checked)} /> Record automatically</label>
              </div>

              <div className="config-panels">
                <div className="sub-panel">
                  <h3>ONVIF</h3>
                  <label className="check-field"><input type="checkbox" checked={selectedCamera.onvif?.enabled || false} onChange={(event) => updateCamera(selectedCamera.id, ["onvif", "enabled"], event.target.checked)} /> Enabled</label>
                  <label>Host<input value={selectedCamera.onvif?.host || ""} onChange={(event) => updateCamera(selectedCamera.id, ["onvif", "host"], event.target.value)} /></label>
                  <label>Port<input type="number" value={selectedCamera.onvif?.port || 8000} onChange={(event) => updateCamera(selectedCamera.id, ["onvif", "port"], Number(event.target.value))} /></label>
                  <label>Username<input value={selectedCamera.onvif?.username || ""} onChange={(event) => updateCamera(selectedCamera.id, ["onvif", "username"], event.target.value)} /></label>
                  <label>Password<input type="password" value={selectedCamera.onvif?.password || ""} onChange={(event) => updateCamera(selectedCamera.id, ["onvif", "password"], event.target.value)} /></label>
                </div>
              </div>

              <div className="actions">
                <button onClick={() => cloneCamera(selectedCamera)}><Copy size={16} /> Clone Camera</button>
                <button onClick={() => probeCamera(selectedCamera)}><Radar size={16} /> Auto-detect</button>
                <button onClick={() => removeCamera(selectedCamera.id)}><Trash2 size={16} /> Remove</button>
              </div>
              <RuntimeStatus status={selectedRuntimeStatus} timeZone={timeZone} />
              {probe ? <ProbeResult probe={probe} /> : null}
            </>
          ) : (
            <div className="empty-state">Add a camera to begin.</div>
          )}
          {status ? <div className="save-status">{status}</div> : null}
        </div>
      </section>
        </>
      )}
    </main>
  );
}

function LogViewer({ lines, filter, setFilter, level, setLevel, timeZone }) {
  return (
    <div className="log-viewer">
      <div className="log-toolbar">
        <label>Level<select value={level} onChange={(event) => setLevel(event.target.value)}>
          <option value="DEBUG">Debug+</option>
          <option value="INFO">Info+</option>
          <option value="WARNING">Warning+</option>
          <option value="ERROR">Error+</option>
          <option value="CRITICAL">Critical</option>
        </select></label>
        <label>Filter<input value={filter} onChange={(event) => setFilter(event.target.value)} placeholder="logger, text, error..." /></label>
      </div>
      <div className="log-lines" role="log" aria-live="polite">
        {lines.length ? lines.map((line, index) => (
          <div className={`log-line ${String(line.level || "").toLowerCase()}`} key={`${line.time}-${index}`}>
            <time>{formatTimeOnly(line.time, timeZone)}</time>
            <span>{line.level}</span>
            <strong>{line.logger}</strong>
            <code>{line.message}</code>
          </div>
        )) : <div className="empty-state">No log lines match the current filters.</div>}
      </div>
    </div>
  );
}

function ProbeResult({ probe }) {
  if (probe.loading) return <div className="probe-result">Probing camera capabilities...</div>;
  return (
    <div className="probe-result">
      <strong>Auto-detection</strong>
      <span>Reolink Baichuan: {probe.baichuan?.reachable ? `reachable on ${probe.baichuan.port}` : "not reachable"}</span>
      <span>ONVIF: {probe.onvif?.reachable ? `reachable on ${probe.onvif.port}` : "not reachable"}</span>
      {probe.onvif?.capabilities ? <span>Capabilities: {Object.entries(probe.onvif.capabilities).filter(([, value]) => value).map(([key]) => key).join(", ") || "none reported"}</span> : null}
      {probe.onvif?.error ? <span>{probe.onvif.error}</span> : null}
    </div>
  );
}

function GeneralSettings({ config, updateConfig, timeZone, setTimeZone, theme, setTheme, accelerator }) {
  const openvinoDevices = accelerator?.openvino_devices || [];
  const hasOpenvinoGpu = openvinoDevices.includes("GPU");
  const detectorBackend = config.detector?.backend || "openvino";
  const coremlLabel = accelerator?.is_macos
    ? accelerator?.coreml_available
      ? "Core ML available"
      : "Core ML not installed"
    : "Core ML is macOS only";
  const gpuLabel = accelerator?.is_apple_silicon
    ? "Mac GPU detected, OpenVINO GPU not available on Apple GPU"
    : accelerator?.has_nvidia
      ? "NVIDIA GPU detected"
      : hasOpenvinoGpu
        ? "OpenVINO GPU device available"
        : "No OpenVINO GPU device reported";
  const deviceOptions = [
    ["CPU", "CPU"],
    ["GPU", hasOpenvinoGpu ? "GPU" : "GPU (if OpenVINO plugin is available)"],
    ["AUTO", "AUTO"],
  ];

  return (
    <div className="config-form">
      <div className="config-panels">
        <div className="sub-panel">
          <h3>Preferences</h3>
          <label>Timezone<select value={timeZone} onChange={(event) => setTimeZone(event.target.value)}>
            {US_TIME_ZONES.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select></label>
          <label>Theme<select value={theme} onChange={(event) => setTheme(event.target.value)}>
            {THEMES.map((value) => <option key={value} value={value}>{THEME_META[value].label}</option>)}
          </select></label>
        </div>

        <div className="sub-panel">
          <h3>Storage</h3>
          <label>Storage Directory<input value={config.storage_dir || ""} onChange={(event) => updateConfig(["storage_dir"], event.target.value)} /></label>
          <label>FFmpeg Path<input value={config.ffmpeg_path || ""} onChange={(event) => updateConfig(["ffmpeg_path"], event.target.value)} /></label>
          <label>Event Clip Before<input type="number" min="0" max="30" step="1" value={config.event_clip_before_seconds ?? 5} onChange={(event) => updateConfig(["event_clip_before_seconds"], Number(event.target.value))} /></label>
          <label>Event Clip After<input type="number" min="0" max="30" step="1" value={config.event_clip_after_seconds ?? 5} onChange={(event) => updateConfig(["event_clip_after_seconds"], Number(event.target.value))} /></label>
          <label>Recording Segment Seconds<input type="number" min="2" max="300" step="1" value={config.recording_segment_seconds ?? 10} onChange={(event) => updateConfig(["recording_segment_seconds"], Number(event.target.value))} /></label>
        </div>
      </div>

      <div className="sub-panel">
        <h3>Object Detection</h3>
        <div className="field-row">
          <label className="check-field"><input type="checkbox" checked={config.detector?.enabled || false} onChange={(event) => updateConfig(["detector", "enabled"], event.target.checked)} /> Enable detector</label>
          <label>Backend<select value={detectorBackend} onChange={(event) => updateConfig(["detector", "backend"], event.target.value)}>
            <option value="openvino">OpenVINO / ONNX</option>
            <option value="coreml">Core ML (Mac)</option>
          </select></label>
          <label>OpenVINO Device<select value={config.detector?.device || "CPU"} onChange={(event) => updateConfig(["detector", "device"], event.target.value)}>
            {deviceOptions.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select></label>
          <label>Confidence<input type="number" min="0" max="1" step="0.01" value={config.detector?.confidence_threshold ?? 0.45} onChange={(event) => updateConfig(["detector", "confidence_threshold"], Number(event.target.value))} /></label>
        </div>
        <div className="field-row">
          <label>ONNX Model Path<input value={config.detector?.model_path || ""} onChange={(event) => updateConfig(["detector", "model_path"], event.target.value)} /></label>
          <label>OpenVINO XML<input value={config.detector?.model_xml || ""} onChange={(event) => updateConfig(["detector", "model_xml"], event.target.value)} /></label>
          <label>Labels Path<input value={config.detector?.labels_path || ""} onChange={(event) => updateConfig(["detector", "labels_path"], event.target.value)} /></label>
        </div>
        {detectorBackend === "coreml" ? (
          <div className="field-row">
            <label>Core ML Model Path<input value={config.detector?.coreml_model_path || ""} onChange={(event) => updateConfig(["detector", "coreml_model_path"], event.target.value)} placeholder="model.mlpackage or model.mlmodel" /></label>
          </div>
        ) : null}
        <div className="probe-result">
          <strong>Accelerator</strong>
          <span>System: {accelerator ? `${accelerator.system} ${accelerator.machine}` : "checking..."}</span>
          <span>Detector recommendation: {accelerator?.recommended_detector_backend === "coreml" ? "Core ML" : "OpenVINO / ONNX"}</span>
          <span>{coremlLabel}</span>
          <span>OpenVINO devices: {openvinoDevices.length ? openvinoDevices.join(", ") : "none reported"}</span>
          <span>{gpuLabel}</span>
          {accelerator?.recommended_openvino_device ? <span>Recommended OpenVINO device: {accelerator.recommended_openvino_device}</span> : null}
          {accelerator?.coreml_error ? <span>{accelerator.coreml_error}</span> : null}
          {accelerator?.openvino_error ? <span>{accelerator.openvino_error}</span> : null}
        </div>
      </div>
    </div>
  );
}

function RuntimeStatus({ status, timeZone }) {
  if (!status) {
    return <div className="probe-result"><strong>Runtime</strong><span>Save this camera to start workers.</span></div>;
  }
  return (
    <div className="probe-result runtime-result">
      <strong>Runtime</strong>
      <span>Stream worker: {status.running ? "running" : "not running"}</span>
      <span>Recording: {status.recording ? "running" : "stopped"}</span>
      <span>ONVIF: {status.onvif_enabled ? (status.onvif_connected ? "connected" : `not connected${status.onvif_last_error ? `: ${status.onvif_last_error}` : ""}`) : "disabled"}</span>
      {status.onvif_last_event_at ? <span>Last ONVIF message: {formatDateTime(status.onvif_last_event_at, timeZone)}</span> : null}
    </div>
  );
}

function App() {
  const [timeZone, setTimeZone] = useStoredState("survng.timeZone", DEFAULT_TIME_ZONE);
  const [theme, setTheme] = useStoredState("survng.theme", "auto");
  const page = window.location.pathname.startsWith("/config")
    ? "config"
    : window.location.pathname.startsWith("/recordings")
      ? "recordings"
      : window.location.pathname.startsWith("/incidents")
        ? "incidents"
        : "live";
  useEffect(() => {
    document.documentElement.dataset.theme = THEMES.includes(theme) ? theme : "auto";
  }, [theme]);
  return (
    <Shell page={page} theme={theme}>
      {page === "config"
        ? <ConfigPage timeZone={timeZone} setTimeZone={setTimeZone} theme={theme} setTheme={setTheme} />
        : page === "recordings"
          ? <RecordingsPage timeZone={timeZone} />
          : page === "incidents"
            ? <IncidentsPage timeZone={timeZone} />
            : <LivePage timeZone={timeZone} />}
    </Shell>
  );
}

createRoot(document.getElementById("root")).render(<App />);
