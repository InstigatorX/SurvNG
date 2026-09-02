import React, { useContext, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  Activity,
  ArrowLeft,
  ArrowRight,
  ArrowUpDown,
  Camera,
  Check,
  CircleAlert,
  ChevronLeft,
  ChevronRight,
  Copy,
  CircleDot,
  Clock3,
  Crop,
  Cog,
  Download,
  Cpu,
  Gauge,
  KeyRound,
  Layers,
  GripVertical,
  HardDrive,
  Search,
  ListTree,
  LayoutDashboard,
  Monitor,
  Moon,
  Pause,
  PanelLeftOpen,
  Plus,
  Power,
  Radar,
  Radio,
  RefreshCcw,
  RotateCcw,
  Save,
  ScanFace,
  ShieldCheck,
  Sparkles,
  Sun,
  Trash2,
  Undo2,
  Wrench,
  X,
} from "lucide-react";
import { buildMotionDecisionFusion, MOTION_BEHAVIOR_OPTIONS, motionBehaviorOption, motionBehaviorSettings, motionBehaviorValue, motionModeInfo, readMotionDecisionFusion } from "../motionDecisionConfig.mjs";
import { availableQualificationPresets, motionAnalysisPresetSelectionUseful, presetQualificationGraph, readMotionAnalysisPreset } from "../motionAnalysisConfig.mjs";
import { TUNEUP_PERIODS, TUNEUP_SETTING_NAMES, tuneupHistoryTitle, tuneupOutcome, tuneupRecommendationGroup, tuneupValue } from "../detectionTuneup.mjs";
import { formatServerUptime } from "../duration.mjs";
import { cameraCaptureConnectivity, cameraConnectivityClass, cameraConnectivityLabel } from "../cameraConnectivity.mjs";
import { browserStorage, readStoredValue } from "../storage.mjs";
import { logPayloadSignature } from "../pollingPolicy.mjs";
import { useVisiblePolling } from "../visibilityPolling.mjs";
import { motionAuditRegions } from "../motionAudit.mjs";
import { insertZonePointWithIndex } from "../zoneGeometry.mjs";
import { ADMIN_NAV_GROUPS, GENERAL_SECTION_LABELS, adminDestination, adminHomeDestinations, adminWorkspaceSearch, cameraConfigDirtyState, comparableCameraSettings, comparableSystemConfig, configValuesEqual, dirtyCameraCount, nextTabId, normalizeTelemetrySection, perCameraDirtyState, readAdminSubsection, readAdminWorkspace, telemetryLocationOptions } from "../adminWorkspace.mjs";
import { appUrl, mediaUrl, fetch } from "../shared/api.js";
import { MEDIA_STORAGE_ROLES, CAMERA_ADMIN_SECTIONS, TELEMETRY_ADMIN_SECTIONS, HEALTH_TELEMETRY_SECTIONS, GENERAL_ADMIN_SECTIONS, US_TIME_ZONES, THEMES } from "../shared/constants.js";
import { CameraScopePicker } from "../shared/CameraScopePicker.jsx";
import { secretInputValue, secretInputHint } from "../shared/secrets.js";
import { formatDateTime, formatTimeOnly, formatBytes, formatMilliseconds, formatAge, formatDuration, formatCompactDuration } from "../shared/format.js";
import { useStoredState, useStoredJsonState, useModalFocus } from "../shared/hooks.js";
import { mediaStorageConfigurationError, slugify, inferredBackendLabel, cameraWithDerivedConnection, camerasWithGeneratedIds } from "../shared/cameras.js";
import { defaultCamera, CameraOnvifEditor, LiveViewFramingEditor, defaultCameraMotionQualification, cameraMotionQualificationInherited } from "./cameraEditors.jsx";
import { AccessSettings } from "./AccessSettings.jsx";
import { ModelsAndHardwarePanel } from "./ModelsAndHardwarePanel.jsx";
import { AdminCommandBar, AdminCommandLabel } from "./AdminCommandBar.jsx";
import { DetectionOccupancyCard } from "./DetectionOccupancyCard.jsx";

export const ADMIN_DESTINATION_ICONS = {
  home: LayoutDashboard,
  cameras: Camera,
  detection: Cpu,
  storage: HardDrive,
  integrations: Radio,
  access: KeyRound,
  preferences: Cog,
  server: Cog,
  overview: Gauge,
  health: ShieldCheck,
  audit: Activity,
  logs: ListTree,
  tuneup: Sparkles,
  diagnostics: Wrench,
  maintenance: HardDrive,
  advisor: Sparkles,
};
export const THEME_META = {
  auto: { label: "Auto", icon: Monitor },
  light: { label: "Light", icon: Sun },
  dark: { label: "Dark", icon: Moon },
};

function normalizeDetectorModelPath(value) {
  return String(value || "").replaceAll("\\", "/").replace(/\/+/g, "/");
}

export function findDetectorModel(models, activePath) {
  const active = normalizeDetectorModelPath(activePath);
  if (!active) return undefined;
  return (models || []).find((model) => {
    const path = normalizeDetectorModelPath(model.path);
    return path === active || path.endsWith(active) || active.endsWith(path);
  });
}

export function MotionAnalysisPresetEditor({
  qualification,
  inherited = false,
  catalog,
  onSetInherited,
  onChange,
}) {
  const presets = availableQualificationPresets(catalog);
  const parsed = readMotionAnalysisPreset(qualification, catalog);
  const selectedValue = inherited
    ? "inherit"
    : parsed.custom
      ? "custom"
      : parsed.preset?.id || "";

  function selectPreset(value) {
    if (value === "inherit") {
      onSetInherited?.(true);
      return;
    }
    const preset = presets.find((candidate) => candidate.id === value);
    if (preset) onChange(presetQualificationGraph(preset));
  }

  if (!motionAnalysisPresetSelectionUseful(catalog)) {
    return parsed.custom && !inherited ? (
      <div className="motion-analysis-warning motion-analysis-custom-notice">
        <strong>Advanced custom motion pipeline</strong>
        <span>This externally configured pipeline remains active and protected from guided settings.</span>
      </div>
    ) : null;
  }

  return (
    <div className="motion-analysis-preset">
      <label>Motion analysis method<select value={selectedValue} onChange={(event) => selectPreset(event.target.value)} disabled={!presets.length}>
        {inherited ? <option value="inherit">Use global setting</option> : null}
        {parsed.custom && !inherited ? <option value="custom">Advanced custom pipeline</option> : null}
        {!presets.length ? <option value="">Loading available methods...</option> : null}
        {presets.map((preset) => (
          <option key={preset.id} value={preset.id}>{preset.label}{preset.recommended ? " (Recommended)" : ""}</option>
        ))}
      </select></label>
      {parsed.custom && !inherited ? <div className="motion-analysis-warning">This advanced pipeline is protected. Selecting another method will replace only the motion-analysis stages.</div> : null}
    </div>
  );
}

export function MotionDecisionEditor({
  fusion,
  mode,
  globalMode = "camera",
  inherited = false,
  inheritedFusion,
  onSetInherited,
  onModeChange,
  onChange,
  onRestoreDefaults,
  configurationInherited,
  cameraName,
}) {
  const parsed = readMotionDecisionFusion(fusion);
  const inheritedParsed = readMotionDecisionFusion(inheritedFusion);
  const effective = inherited ? inheritedParsed : parsed;
  const settings = effective.settings;
  const effectiveMode = mode === "inherit" ? globalMode : mode;
  const legacyMode = ["audit", "off", "enforce"].includes(effectiveMode);
  const fullyInherited = Boolean(onSetInherited && inherited && mode === "inherit");
  const custom = (!inherited && parsed.custom) || (fullyInherited && inheritedParsed.custom);
  const effectiveBehavior = motionBehaviorValue(effectiveMode, settings);
  const selectedBehavior = fullyInherited
    ? "inherit"
    : custom
      ? "custom"
      : legacyMode
        ? `legacy:${effectiveMode}`
        : effectiveBehavior;
  const behaviorInfo = custom
    ? {
      status: fullyInherited ? "Global advanced configuration" : "Advanced custom configuration",
      description: "This pipeline was created outside the guided editor. Selecting a standard behavior will replace its decision stages.",
    }
    : legacyMode
      ? motionModeInfo(effectiveMode)
      : motionBehaviorOption(effectiveBehavior);
  const statusLabel = onRestoreDefaults
    ? configurationInherited ? "Inherited" : "Custom"
    : fullyInherited ? "Inherited" : custom ? "Advanced" : legacyMode ? "Legacy" : parsed.usesDefaults ? "Recommended default" : "Customized";

  function updateSettings(patch) {
    onChange(buildMotionDecisionFusion({ ...settings, ...patch }));
  }

  function selectBehavior(value) {
    if (value === "inherit") {
      onModeChange("inherit");
      onSetInherited?.(true);
      return;
    }
    if (value === "custom" || value.startsWith("legacy:")) return;
    const next = motionBehaviorSettings(settings, value);
    onModeChange(next.mode);
    onSetInherited?.(false);
    onChange(buildMotionDecisionFusion(next.settings));
  }

  return (
    <div className={`motion-decision-editor${custom || legacyMode ? " motion-decision-custom" : ""}`}>
      <div className="motion-decision-heading">
        <div>
          <strong>Motion behavior</strong>
          <span>{cameraName ? `Choose what can start object detection for ${cameraName}.` : "Choose what can start object detection."}</span>
        </div>
        <div className="motion-decision-status-actions">
          <span className="motion-decision-status">{statusLabel}</span>
          {onRestoreDefaults ? <button type="button" className="motion-decision-status motion-defaults-action" onClick={onRestoreDefaults} title="Restore all motion settings for this camera to global inheritance">Defaults</button> : null}
        </div>
      </div>

      <div className="motion-behavior-row">
        <label>What starts object detection?<select value={selectedBehavior} onChange={(event) => selectBehavior(event.target.value)}>
          {onSetInherited ? <option value="inherit">Use global setting</option> : null}
          {custom ? <option value="custom">Advanced custom configuration</option> : null}
          {legacyMode ? <option value={`legacy:${effectiveMode}`}>{motionModeInfo(effectiveMode).label}</option> : null}
          {MOTION_BEHAVIOR_OPTIONS.map((option) => <option value={option.value} key={option.value}>{option.label}</option>)}
        </select></label>
        <div className={`motion-decision-mode mode-${effectiveMode}`}>
          <strong>{fullyInherited && !custom ? `Global · ${behaviorInfo.status}` : behaviorInfo.status}</strong>
          <span>{behaviorInfo.description}</span>
        </div>
      </div>

    </div>
  );
}

export const TelemetryInterruptionsContext = React.createContext([]);

export function TelemetryTrend({ title, history, series, timeZone, interruptions = null, maximum = null, valueFormatter = (value) => `${value}` }) {
  const [hoverState, setHoverState] = useState(null);
  const sharedInterruptions = useContext(TelemetryInterruptionsContext);
  const chartInterruptions = interruptions || sharedInterruptions;
  const numericValue = (raw) => {
    if (raw == null || raw === "") return null;
    const value = Number(raw);
    return Number.isFinite(value) ? value : null;
  };
  const values = series.flatMap((item) => history.map((point) => numericValue(point[item.key])).filter((value) => value != null));
  const top = maximum || Math.max(1, ...values) * 1.12;
  const width = 100;
  const height = 42;
  const sampleTimes = history.map((point) => new Date(point?.sampled_at || 0).getTime());
  const firstAt = Number.isFinite(sampleTimes[0]) ? sampleTimes[0] : 0;
  const lastAt = Number.isFinite(sampleTimes.at(-1)) ? sampleTimes.at(-1) : firstAt;
  const timeSpan = Math.max(0, lastAt - firstAt);
  const xForIndex = (index) => {
    if (history.length <= 1) return width;
    const sampledAt = sampleTimes[index];
    return timeSpan > 0 && Number.isFinite(sampledAt)
      ? ((sampledAt - firstAt) / timeSpan) * width
      : (index / (history.length - 1)) * width;
  };
  const segmentsFor = (key) => {
    const segments = [];
    let segment = [];
    history.forEach((point, index) => {
      const value = numericValue(point[key]);
      if (value == null) {
        if (segment.length) segments.push(segment);
        segment = [];
      }
      if (value != null) {
        const x = xForIndex(index);
        const y = height - Math.min(height, (Math.max(0, value) / top) * height);
        segment.push(`${x.toFixed(2)},${y.toFixed(2)}`);
      }
    });
    if (segment.length) segments.push(segment);
    return segments;
  };
  const xForTime = (value) => {
    const timestamp = new Date(value || 0).getTime();
    if (!Number.isFinite(timestamp) || timeSpan <= 0) return null;
    return Math.max(0, Math.min(width, ((timestamp - firstAt) / timeSpan) * width));
  };
  const visibleInterruptions = chartInterruptions.map((item) => {
    const startX = xForTime(item.start_at);
    const markerX = xForTime(item.marker_at || item.start_at);
    const endX = xForTime(item.end_at);
    const actualStartX = Math.min(startX ?? 0, endX ?? 0);
    const actualWidth = Math.abs((endX ?? 0) - (startX ?? 0));
    const displayWidth = Math.max(0.18, actualWidth);
    const displayStartX = Math.max(
      0,
      Math.min(
        width - displayWidth,
        actualWidth >= 0.18
          ? actualStartX
          : (markerX ?? actualStartX) - (displayWidth / 2),
      ),
    );
    return {
      ...item,
      startX,
      markerX,
      endX,
      displayStartX,
      displayWidth,
    };
  }).filter((item) => item.startX != null && item.endX != null && new Date(item.end_at).getTime() >= firstAt && new Date(item.start_at).getTime() <= lastAt);
  const coordinatesFor = (key, index) => {
    const value = numericValue(history[index]?.[key]);
    if (value == null) return null;
    return {
      value,
      x: xForIndex(index),
      y: height - Math.min(height, (Math.max(0, value) / top) * height),
    };
  };
  const latestValue = (key) => {
    for (let index = history.length - 1; index >= 0; index -= 1) {
      const value = numericValue(history[index]?.[key]);
      if (value != null) return value;
    }
    return null;
  };
  const formatBoundary = (value) => (
    lastAt - firstAt >= 24 * 60 * 60 * 1000
      ? formatDateTime(value, timeZone)
      : formatTimeOnly(value, timeZone)
  );
  const selectedIndex = Number.isInteger(hoverState?.index) && hoverState.index >= 0 && hoverState.index < history.length
    ? hoverState.index
    : null;
  const selectedPoint = selectedIndex == null ? null : history[selectedIndex];
  const selectedPointX = selectedIndex == null
    ? 0
    : xForIndex(selectedIndex);
  const hoverX = hoverState?.x ?? selectedPointX;
  const interruptionHitTolerance = hoverState?.hitToleranceX ?? 0.25;
  const hoverInterruption = visibleInterruptions.find((item) => (
    hoverX >= item.displayStartX - interruptionHitTolerance
    && hoverX <= item.displayStartX + item.displayWidth + interruptionHitTolerance
  )) || null;
  const tooltipAlignment = hoverX < 25 ? "start" : hoverX > 75 ? "end" : "center";
  const selectHoverIndex = (index) => {
    if (!history.length) return;
    const boundedIndex = Math.max(0, Math.min(history.length - 1, index));
    setHoverState({ index: boundedIndex, x: xForIndex(boundedIndex), hitToleranceX: 0.25 });
  };
  const updateHover = (event) => {
    if (!history.length) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (event.clientX - bounds.left) / Math.max(1, bounds.width)));
    if (timeSpan <= 0) {
      setHoverState({
        index: Math.round(ratio * Math.max(0, history.length - 1)),
        x: ratio * width,
        hitToleranceX: (6 / Math.max(1, bounds.width)) * width,
      });
      return;
    }
    const targetTime = firstAt + ratio * timeSpan;
    let nearestIndex = 0;
    for (let index = 1; index < sampleTimes.length; index += 1) {
      if (Math.abs(sampleTimes[index] - targetTime) < Math.abs(sampleTimes[nearestIndex] - targetTime)) {
        nearestIndex = index;
      }
    }
    setHoverState({
      index: nearestIndex,
      x: ratio * width,
      hitToleranceX: (6 / Math.max(1, bounds.width)) * width,
    });
  };
  const handleChartKeyDown = (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    if (event.key === "Home") selectHoverIndex(0);
    else if (event.key === "End") selectHoverIndex(history.length - 1);
    else {
      const currentIndex = Number.isInteger(hoverState?.index)
        ? hoverState.index
        : history.length - 1;
      selectHoverIndex(currentIndex + (event.key === "ArrowLeft" ? -1 : 1));
    }
  };
  return (
    <article className={`telemetry-trend${selectedPoint ? " has-tooltip" : ""}`}>
      <header><strong>{title}</strong><div className="telemetry-trend-values" aria-label="Chart lines">{series.map((item) => <span className={item.className || ""} key={item.key}><i /><b>{item.label}</b><em>{latestValue(item.key) == null ? "--" : valueFormatter(latestValue(item.key), item.key)}</em></span>)}</div></header>
      <div
        className="telemetry-trend-chart"
        tabIndex={history.length ? 0 : -1}
        aria-label={`${title}. Point, tap, or use the arrow keys to inspect values.`}
        onPointerMove={updateHover}
        onPointerDown={updateHover}
        onFocus={(event) => {
          if (hoverState == null && event.currentTarget.matches(":focus-visible")) {
            selectHoverIndex(history.length - 1);
          }
        }}
        onBlur={() => setHoverState(null)}
        onKeyDown={handleChartKeyDown}
        onPointerLeave={(event) => {
          if (event.pointerType === "mouse") setHoverState(null);
        }}
      >
        <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" role="img" aria-label={`${title} trend`}>
          <line x1="0" y1={height / 2} x2={width} y2={height / 2} />
          {visibleInterruptions.map((item, index) => <g className={`telemetry-interruption ${item.kind}`} key={`${item.start_at}-${index}`}>
            <rect x={item.displayStartX} y="0" width={item.displayWidth} height={height} />
          </g>)}
          {series.flatMap((item) => segmentsFor(item.key).map((points, index) => <polyline className={item.className || ""} key={`${item.key}-${index}`} points={points.join(" ")} />))}
          {selectedPoint ? <line className="telemetry-trend-cursor" x1={hoverX} y1="0" x2={hoverX} y2={height} /> : null}
          {selectedPoint ? series.map((item) => {
            const coordinates = coordinatesFor(item.key, selectedIndex);
            return coordinates ? <ellipse className={`telemetry-trend-point ${item.className || ""}`} key={item.key} cx={coordinates.x} cy={coordinates.y} rx="0.75" ry="1.4" /> : null;
          }) : null}
        </svg>
        {selectedPoint ? (
          <div className={`telemetry-trend-tooltip ${tooltipAlignment}`} style={{ left: `${hoverX}%` }} role="status">
            <time>{hoverInterruption ? `Restarted ${formatDateTime(hoverInterruption.marker_at, timeZone)}` : formatDateTime(selectedPoint.sampled_at, timeZone)}</time>
            {hoverInterruption ? <div className={`telemetry-trend-interruption-detail ${hoverInterruption.kind}`}><strong>{hoverInterruption.title}</strong><small>{formatDuration(hoverInterruption.duration_seconds || 0)} · {hoverInterruption.description}</small></div> : null}
            {series.map((item) => {
              const value = numericValue(selectedPoint[item.key]);
              return <span className={item.className || ""} key={item.key}><i />{item.label}<strong>{value == null ? "--" : valueFormatter(value, item.key)}</strong></span>;
            })}
          </div>
        ) : null}
      </div>
      <footer><time>{history[0] ? formatBoundary(history[0].sampled_at) : "Now"}</time><span>{history.length} sample{history.length === 1 ? "" : "s"}</span><time>{history.at(-1) ? formatBoundary(history.at(-1).sampled_at) : "Now"}</time></footer>
    </article>
  );
}

export function formatRecorderTimestampHealth(sources) {
  const entries = Object.entries(sources || {});
  if (!entries.length) return "Stable · no discontinuities";
  return entries.map(([source, health]) => {
    const details = [
      `${Number(health.discontinuities || 0)} jump${Number(health.discontinuities || 0) === 1 ? "" : "s"}`,
      `${Number(health.epoch_rollovers || 0)} recovered`,
    ];
    if (health.rollover_pending) details.push("recovery pending");
    if (Number(health.rollover_failures || 0)) details.push(`${health.rollover_failures} failed`);
    if (Number(health.rate_limited || 0)) details.push(`${health.rate_limited} rate-limited`);
    return `${source}: ${details.join(", ")}`;
  }).join(" · ");
}

export function TelemetryContinuity({ data }) {
  if (!data) return null;
  const summary = data.interruption_summary || {};
  const parts = [
    summary.controlled ? `${summary.controlled} controlled restart${summary.controlled === 1 ? "" : "s"}` : "",
    summary.unexpected ? `${summary.unexpected} unexpected restart${summary.unexpected === 1 ? "" : "s"}` : "",
    summary.unknown ? `${summary.unknown} unexplained gap${summary.unknown === 1 ? "" : "s"}` : "",
  ].filter(Boolean);
  return (
    <div className={`telemetry-interruption-summary telemetry-header-continuity ${summary.unexpected ? "danger" : summary.unknown ? "warning" : "healthy"}`}>
      <Clock3 size={16} />
      <span><strong>Service continuity · 24h</strong><em>{parts.length ? `${parts.join(" · ")} · ${formatDuration(summary.duration_seconds || 0)} unavailable` : "No interruptions"}</em></span>
    </div>
  );
}

export function TelemetryViewer({ data, cameraId, timeZone, config }) {
  if (!data) return <div className="empty-state">Waiting for telemetry...</div>;
  const selected = cameraId ? data.cameras?.find((camera) => camera.id === cameraId) : null;
  const activity = selected?.activity || data.activity;
  const lastHour = activity?.last_hour || {};
  const lastDay = activity?.last_24h || {};
  const runtime = data.detector?.runtime || {};
  const objectWorkers = data.detector?.workers?.object || data.detector?.isolation || {};
  const semantic = data.semantic_search || {};
  const faceRecognition = data.face_recognition || {};
  const gpu = data.gpu || {};
  const storage = data.system?.storage || {};
  const memory = data.system?.memory || {};
  const serviceMemory = data.system?.service_memory || {};
  const workerMemory = data.system?.worker_memory || {};
  const memoryMaintenance = data.system?.memory_maintenance || {};
  const recordedDecode = data.detector?.recorded_decode || {};
  const recordedDecodeErrors = data.detector?.recorded_decode?.camera_decoder_errors || {};
  const recordedHevcErrors = Object.values(recordedDecodeErrors).reduce((total, item) => total + Number(item?.hevc_error_lines || 0), 0);
  const hourly = activity?.hourly || [];
  const runtimeShort = data.runtime_history?.short || [];
  const runtimeLong = data.runtime_history?.long || [];
  const capacityShort = data.tracking_capacity_history?.short || [];
  const capacityLong = data.tracking_capacity_history?.long || [];
  const memoryShort = data.process_memory_history?.short || [];
  const memoryLong = data.process_memory_history?.long || [];
  const appearanceBackfill = data.appearance_backfill || {};
  const backfillCounts = appearanceBackfill.counts || {};
  const capacityTotals = capacityShort.reduce((total, point) => ({
    attempts: total.attempts + Number(point.attempts || 0),
    waited: total.waited + Number(point.waited || 0),
    skipped: total.skipped + Number(point.skipped || 0),
    waitMax: Math.max(total.waitMax, Number(point.wait_seconds_max || 0)),
  }), { attempts: 0, waited: 0, skipped: 0, waitMax: 0 });
  const maxHourly = Math.max(1, ...hourly.map((item) => Number(item.events) || 0));
  const topLabels = Object.entries(lastDay.labels || {}).sort((left, right) => right[1] - left[1]).slice(0, 5);
  const shownCameras = selected ? [selected] : (data.cameras || []);
  const activityAttribution = shownCameras.reduce((total, camera) => {
    const status = camera.object_tracking?.object_activity_attribution || {};
    total.evaluated += Number(status.evaluated || 0);
    total.active += Number(status.active || 0);
    total.sceneContext += Number(status.scene_context || 0);
    total.indeterminate += Number(status.indeterminate || 0);
    total.enforced += Number(status.enforced_suppressions || 0);
    total.detectorAdmissions += Number(status.detector_admissions || 0);
    total.confidenceRejections += Number(status.confidence_rejections || 0);
    total.zoneRejections += Number(status.zone_rejections || 0);
    total.temporalRejections += Number(status.temporal_rejections || 0);
    if (status.mode) total.modes.add(status.mode);
    return total;
  }, { evaluated: 0, active: 0, sceneContext: 0, indeterminate: 0, enforced: 0, detectorAdmissions: 0, confidenceRejections: 0, zoneRejections: 0, temporalRejections: 0, modes: new Set() });
  const selectedCapture = selected?.capture || {};
  const selectedReadFailures = [selectedCapture.live, selectedCapture.main]
    .reduce((total, source) => total + Number(source?.read_failures || 0), 0);
  const selectedOpenFailures = [selectedCapture.live, selectedCapture.main]
    .reduce((total, source) => total + Number(source?.open_failures || 0), 0);
  const runtimeTotals = runtimeShort.reduce((total, point) => ({
    analyzed: total.analyzed + Number(point.analysis_frames_sampled || 0),
    superseded: total.superseded + Number(point.analysis_frames_dropped || 0),
    interruptions: total.interruptions + Number(point.capture_interruptions || 0),
    eventLoss: total.eventLoss + Number(point.event_delivery_failures || 0),
    availabilitySum: total.availabilitySum + Number(point.camera_availability_percent ?? 0),
    availabilitySamples: total.availabilitySamples + (point.camera_availability_percent == null ? 0 : 1),
    minimumAvailability: Math.min(
      total.minimumAvailability,
      Number(point.camera_availability_percent ?? 100),
    ),
  }), {
    analyzed: 0,
    superseded: 0,
    interruptions: 0,
    eventLoss: 0,
    availabilitySum: 0,
    availabilitySamples: 0,
    minimumAvailability: 100,
  });
  const averageAvailability = runtimeTotals.availabilitySamples
    ? runtimeTotals.availabilitySum / runtimeTotals.availabilitySamples
    : null;
  const analysisTotal = runtimeTotals.analyzed + runtimeTotals.superseded;
  const analysisCoverage = analysisTotal
    ? (runtimeTotals.analyzed / analysisTotal) * 100
    : null;
  const formatCoverage = (value) => value == null
    ? "--"
    : `${Number(value).toFixed(Number(value) >= 99.95 ? 2 : 1)}%`;
  return (
    <TelemetryInterruptionsContext.Provider value={selected ? [] : (data.interruptions || [])}>
      <div className="telemetry-viewer">
        <div className={`telemetry-summary-grid${selected ? " camera-summary" : " overview-summary"}`}>
          <article><span>Events · 1h</span><strong>{Number(lastHour.events || 0).toLocaleString()}</strong><small>{Number(lastDay.events || 0).toLocaleString()} in the shown 24-hour window</small></article>
          <article><span>Object incidents · 24h</span><strong>{Number(lastDay.object_incidents || 0).toLocaleString()}</strong><small>{Number(lastDay.objects || 0).toLocaleString()} eligible object detections</small></article>
          {selected ? <>
            <article><span>Live video</span><strong>{cameraConnectivityLabel(cameraCaptureConnectivity(selected))}</strong><small>Last frame {formatAge(selected.last_frame_age_seconds)}{selected.live_pipeline?.source_element ? ` · ${selected.live_pipeline.source_element}` : ""}{Number(selected.capture_reconnects || 0) ? ` · ${Number(selected.capture_reconnects).toLocaleString()} reconnects since restart` : ""}{selected.last_error ? ` · ${selected.last_error}` : ""}</small></article>
            <article><span>Stream interruptions · since restart</span><strong>{(selectedReadFailures + selectedOpenFailures).toLocaleString()}</strong><small>{selectedReadFailures.toLocaleString()} interrupted reads · {selectedOpenFailures.toLocaleString()} failed connections</small></article>
            <article><span>Tracking · 2h</span><strong>{capacityTotals.skipped ? `${capacityTotals.skipped} skipped` : "No skips"}</strong><small>{capacityTotals.attempts} sessions · {capacityTotals.waited} waited · longest {capacityTotals.waitMax.toFixed(1)}s</small></article>
            <article><span>EMA coverage · 2h</span><strong>{analysisTotal ? formatCoverage(analysisCoverage) : "Not active"}</strong><small>{runtimeTotals.eventLoss ? `${runtimeTotals.eventLoss} events lost` : "No events lost"}</small></article>
            <article className="telemetry-memory-card"><span>Recorded decode memory</span><strong>{formatBytes(selected.recorded_decode?.reserved_bytes)}</strong><small>{selected.recorded_decode?.active_workflows ? `${selected.recorded_decode.active_workflows} active refinement${selected.recorded_decode.active_workflows === 1 ? "" : "s"}` : "No active refinement"}{selected.recorded_decoder_errors?.hevc_error_lines ? ` · ${selected.recorded_decoder_errors.hevc_error_lines} HEVC decode lines` : ""}</small></article>
          </> : <>
            <article><span>Camera uptime · 2h</span><strong>{formatCoverage(averageAvailability)}</strong><small>Lowest minute {formatCoverage(runtimeTotals.minimumAvailability)} · {runtimeTotals.interruptions ? `${runtimeTotals.interruptions.toLocaleString()} recovered stream issues` : "no stream interruptions"}</small></article>
            <article><span>EMA coverage · 2h</span><strong>{analysisTotal ? formatCoverage(analysisCoverage) : "Not active"}</strong><small>{analysisTotal ? (runtimeTotals.superseded ? `${runtimeTotals.superseded.toLocaleString()} stale frames skipped to stay current` : "Every sampled frame analyzed") : "No EMA samples in this window"}{runtimeTotals.eventLoss ? ` · ${runtimeTotals.eventLoss} events lost` : " · no events lost"}</small></article>
            <article><span>Detector response</span><strong>{formatMilliseconds(runtime.average_inference_ms)}</strong><small>{objectWorkers.alive_workers || (objectWorkers.worker_alive ? 1 : 0)}/{objectWorkers.configured_workers || 1} workers online · {Number(runtime.failed_inferences || 0) ? `${Number(runtime.failed_inferences).toLocaleString()} failures` : "no failures"}</small></article>
            <article><span>GPU</span><strong>{gpu.available ? "Available" : "Unavailable"}</strong><small>{Number.isFinite(gpu.utilization_percent) ? `${gpu.utilization_percent}% busy now` : "Collecting activity"}</small></article>
            <article><span>Storage free</span><strong>{formatBytes(storage.free_bytes)}</strong><small>{storage.used_percent || 0}% used of {formatBytes(storage.total_bytes)}</small></article>
            <article><span>Tracking · 2h</span><strong>{capacityTotals.skipped ? `${capacityTotals.skipped} skipped` : "No skips"}</strong><small>{capacityTotals.waited} delayed · {Number(backfillCounts.completed || 0).toLocaleString()} recovered · {Number(backfillCounts.queued || 0).toLocaleString()} waiting</small></article>
            <article><span>SurvNG uptime</span><strong>{formatServerUptime(Number(data.system?.uptime_seconds || 0))}</strong><small>Since the last service start</small></article>
            <article><span>CPU demand</span><strong>{data.system?.load_average?.one ?? "--"}</strong><small>Across {data.system?.cpu_count || 1} cores</small></article>
            <article className="telemetry-memory-card"><span>Host memory</span><strong>{formatBytes(memory.available_bytes)}</strong><small>{memory.used_percent || 0}% currently used</small></article>
            <article className="telemetry-memory-card"><span>Application memory</span><strong>{formatBytes(serviceMemory.application_bytes)}</strong><small>SurvNG and AI workers</small></article>
            <article className="telemetry-memory-card"><span>Recorded decode memory</span><strong>{formatBytes(recordedDecode.reserved_bytes)}</strong><small>{formatBytes(recordedDecode.memory_budget_bytes)} capacity · {recordedDecode.active_workflows || 0}/{recordedDecode.configured_processes || 0} active{recordedHevcErrors ? ` · ${recordedHevcErrors} HEVC decode lines` : ""}</small></article>
            <article className="telemetry-memory-card"><span>File cache</span><strong>{formatBytes(serviceMemory.reclaimable_file_cache_bytes)}</strong><small>Released automatically as needed</small></article>
            <article><span>Local databases</span><strong>{formatBytes(data.system?.database?.bytes)}</strong><small>Events, indexes, and runtime state</small></article>
          </>}
        </div>

        {!selected ? <details className="telemetry-technical telemetry-system-technical">
          <summary>Technical system diagnostics</summary>
          <dl className="telemetry-details">
            <div><dt>CPU load · 1 / 5 / 15 min</dt><dd>{data.system?.load_average?.one ?? "--"} / {data.system?.load_average?.five ?? "--"} / {data.system?.load_average?.fifteen ?? "--"}</dd></div>
            <div><dt>Working set / service total</dt><dd>{formatBytes(serviceMemory.working_set_bytes)} / {formatBytes(serviceMemory.total_bytes)}</dd></div>
            <div><dt>Main / inference-worker RSS</dt><dd>{formatBytes(data.system?.process_rss_bytes)} / {formatBytes(workerMemory.total_rss_bytes)}</dd></div>
            <div><dt>Allocator live / retained</dt><dd>{formatBytes(data.system?.process_memory?.malloc?.allocated_bytes)} / {formatBytes(data.system?.process_memory?.malloc?.free_bytes)}</dd></div>
            <div><dt>Allocator trims</dt><dd>{Number(memoryMaintenance.successful_trims || 0).toLocaleString()} <small>{formatBytes(memoryMaintenance.reclaimed_total_bytes)} reclaimed</small></dd></div>
            <div><dt>Threads / open files</dt><dd>{Number(data.system?.process_memory?.threads || 0).toLocaleString()} / {Number(data.system?.process_memory?.file_descriptors || 0).toLocaleString()}</dd></div>
            <div><dt>Detector backend / device</dt><dd>{data.detector?.loaded_backend || "Not loaded"} / {data.detector?.loaded_device || data.detector?.configured_device || "--"}</dd></div>
            <div><dt>Object detector processes</dt><dd>{(objectWorkers.worker_pids || [objectWorkers.worker_pid]).filter(Boolean).join(", ") || "None"}</dd></div>
            <div><dt>Per-detector response</dt><dd>{(runtime.workers || []).length ? runtime.workers.map((worker) => `#${worker.index} ${formatMilliseconds(worker.average_inference_ms)} · ${Number(worker.queue_depth || 0)} queued`).join(" · ") : "Waiting for samples"}</dd></div>
            <div><dt>Inference requests / object hits</dt><dd>{Number(runtime.total_inferences || 0).toLocaleString()} / {Number(runtime.object_hit_inferences || 0).toLocaleString()}</dd></div>
          </dl>
        </details> : null}

        <section className="telemetry-section">
          <div className="telemetry-section-head"><div><h3>Events by hour{selected ? ` · ${selected.name}` : ""}</h3></div></div>
          <div className="telemetry-hourly" aria-label="Events per hour">
            {hourly.map((item, index) => (
              <div className="telemetry-hour" key={item.started_at} title={`${formatDateTime(item.started_at, timeZone)}: ${item.events} events, ${item.object_incidents} object incidents`}>
                <div className="telemetry-hour-bars">
                  <i style={{ height: `${Math.max(3, (Number(item.events) / maxHourly) * 100)}%` }} />
                  <b style={{ height: `${Math.max(0, (Number(item.object_incidents) / maxHourly) * 100)}%` }} />
                </div>
                {(index % 4 === 0 || index === hourly.length - 1) ? <time>{formatTimeOnly(item.started_at, timeZone).replace(/:00(?=\s)/, "")}</time> : <time />}
              </div>
            ))}
          </div>
          <div className="telemetry-legend"><span><i /> Events</span><span><i className="objects" /> Object incidents</span></div>
        </section>

        <section className="telemetry-section">
          <div className="telemetry-section-head"><div><h3>{selected ? `${selected.name} object tracking` : "Object tracking"}</h3></div></div>
          <div className="telemetry-trend-grid two-column">
            <TelemetryTrend title="Tracking · 2 hours" history={capacityShort} timeZone={timeZone} valueFormatter={(value) => Math.round(value).toLocaleString()} series={[{ key: "attempts", label: "Requested", className: "rate" }, { key: "waited", label: "Delayed", className: "warning" }, { key: "skipped", label: "Skipped", className: "danger" }]} />
            <TelemetryTrend title="Tracking · 7 days" history={capacityLong} timeZone={timeZone} valueFormatter={(value) => Math.round(value).toLocaleString()} series={[{ key: "attempts", label: "Requested", className: "rate" }, { key: "waited", label: "Delayed", className: "warning" }, { key: "skipped", label: "Skipped", className: "danger" }]} />
          </div>
        </section>

        <section className="telemetry-section">
          <div className="telemetry-section-head"><div><h3>Camera reliability{selected ? ` · ${selected.name}` : ""}</h3></div></div>
          <div className="telemetry-trend-grid two-column">
            <TelemetryTrend title="Availability · 2 hours" history={runtimeShort} timeZone={timeZone} maximum={100} valueFormatter={formatCoverage} series={[{ key: "camera_availability_percent", label: "Available", className: "rate" }]} />
            <TelemetryTrend title="Stream interruptions · 2 hours" history={runtimeShort} timeZone={timeZone} valueFormatter={(value) => Math.round(value).toLocaleString()} series={[{ key: "capture_interruptions", label: "Interruptions", className: "danger" }]} />
            <TelemetryTrend title="Availability · 7 days" history={runtimeLong} timeZone={timeZone} maximum={100} valueFormatter={formatCoverage} series={[{ key: "camera_availability_percent", label: "Available", className: "rate" }]} />
            <TelemetryTrend title="Stream interruptions · 7 days" history={runtimeLong} timeZone={timeZone} valueFormatter={(value) => Math.round(value).toLocaleString()} series={[{ key: "capture_interruptions", label: "Interruptions", className: "danger" }]} />
          </div>
        </section>

        <section className="telemetry-section">
          <div className="telemetry-section-head"><div><h3>Enhanced motion analysis{selected ? ` · ${selected.name}` : ""}</h3></div></div>
          <div className="telemetry-trend-grid two-column">
            <TelemetryTrend title="EMA coverage · 2 hours" history={runtimeShort} timeZone={timeZone} maximum={100} valueFormatter={formatCoverage} series={[{ key: "analysis_coverage_percent", label: "Coverage", className: "rate" }]} />
            <TelemetryTrend title="EMA coverage · 7 days" history={runtimeLong} timeZone={timeZone} maximum={100} valueFormatter={formatCoverage} series={[{ key: "analysis_coverage_percent", label: "Coverage", className: "rate" }]} />
            <TelemetryTrend title="EMA rescue path · 2 hours" history={runtimeShort} timeZone={timeZone} valueFormatter={(value) => Math.round(value).toLocaleString()} series={[{ key: "ema_credible_episodes", label: "Credible", className: "secondary" }, { key: "object_checks_admitted", label: "Admitted", className: "warning" }, { key: "object_checks_completed", label: "Completed", className: "rate" }]} />
            <TelemetryTrend title="EMA rescue path · 7 days" history={runtimeLong} timeZone={timeZone} valueFormatter={(value) => Math.round(value).toLocaleString()} series={[{ key: "ema_credible_episodes", label: "Credible", className: "secondary" }, { key: "object_checks_admitted", label: "Admitted", className: "warning" }, { key: "object_checks_completed", label: "Completed", className: "rate" }]} />
          </div>
        </section>

        {!selected ? <section className="telemetry-section">
          <div className="telemetry-section-head"><div><h3>System performance</h3></div></div>
          <div className="telemetry-trend-grid two-column">
            <TelemetryTrend title="Host demand · 2 hours" history={runtimeShort} timeZone={timeZone} maximum={100} valueFormatter={(value) => `${value.toFixed(1)}%`} series={[{ key: "cpu_load_percent", label: "CPU", className: "cpu" }, { key: "memory_used_percent", label: "Memory", className: "memory" }]} />
            <TelemetryTrend title="Detector response · 2 hours" history={runtimeShort} timeZone={timeZone} valueFormatter={(value) => formatMilliseconds(value)} series={[{ key: "inference_ms", label: "Response", className: "inference" }]} />
            <TelemetryTrend title="Host demand · 7 days" history={runtimeLong} timeZone={timeZone} maximum={100} valueFormatter={(value) => `${value.toFixed(1)}%`} series={[{ key: "cpu_load_percent", label: "CPU", className: "cpu" }, { key: "memory_used_percent", label: "Memory", className: "memory" }]} />
            <TelemetryTrend title="Detector response · 7 days" history={runtimeLong} timeZone={timeZone} valueFormatter={(value) => formatMilliseconds(value)} series={[{ key: "inference_ms", label: "Response", className: "inference" }]} />
          </div>
        </section> : null}

        {!selected ? <section className="telemetry-section">
          <div className="telemetry-section-head"><div><h3>Memory stability</h3></div></div>
          <div className="telemetry-trend-grid two-column">
            <TelemetryTrend title="Application memory · 24 hours" history={memoryShort} timeZone={timeZone} valueFormatter={(value) => formatBytes(value)} series={[{ key: "rss_bytes", label: "SurvNG", className: "process-memory" }, { key: "worker_rss_bytes", label: "AI workers", className: "secondary" }]} />
            <TelemetryTrend title="Application memory · 7 days" history={memoryLong} timeZone={timeZone} valueFormatter={(value) => formatBytes(value)} series={[{ key: "rss_bytes", label: "SurvNG", className: "process-memory" }, { key: "worker_rss_bytes", label: "AI workers", className: "secondary" }]} />
          </div>
        </section> : null}

        <div className={`telemetry-activity-grid${selected ? " camera-only" : ""}`}>
          <section className="telemetry-section">
            <div className="telemetry-section-head"><div><h3>{selected ? `${selected.name} activity` : "Object activity"}</h3></div></div>
            <dl className="telemetry-details">
              <div><dt>Top labels · 24h</dt><dd>{topLabels.length ? topLabels.map(([label, count]) => `${label} ${count}`).join(" · ") : "None"}</dd></div>
              <div><dt>Activity attribution · since restart</dt><dd>{activityAttribution.evaluated.toLocaleString()} checked <small>{activityAttribution.active.toLocaleString()} active · {activityAttribution.sceneContext.toLocaleString()} scene context · {activityAttribution.indeterminate.toLocaleString()} uncertain</small></dd></div>
              <div><dt>Context prevented from labeling incidents</dt><dd>{activityAttribution.enforced.toLocaleString()} <small>{activityAttribution.modes.size ? [...activityAttribution.modes].join(" / ").replaceAll("_", " ") : "waiting for detections"}</small></dd></div>
              <div><dt>Object admission · since restart</dt><dd>{activityAttribution.detectorAdmissions.toLocaleString()} detector-eligible <small>{activityAttribution.confidenceRejections.toLocaleString()} low confidence · {activityAttribution.zoneRejections.toLocaleString()} zone-rejected · {activityAttribution.temporalRejections.toLocaleString()} unconfirmed · {activityAttribution.enforced.toLocaleString()} scene context</small></dd></div>
              <div><dt>Depth-shadow health · 24h</dt><dd><DepthShadowPerformance cameraId={selected?.id || ""} label={selected ? "Depth shadow · this camera · last 24 hours" : "Depth shadow · all cameras · last 24 hours"} /></dd></div>
            </dl>
          </section>
          {!selected ? <section className="telemetry-section">
            <div className="telemetry-section-head"><div><h3>Semantic search</h3></div></div>
            <dl className="telemetry-details">
              <div><dt>Status</dt><dd>{String(semantic.state || (semantic.enabled ? "starting" : "disabled")).replaceAll("_", " ")}{semantic.device ? ` · ${semantic.device}` : ""}</dd></div>
              <div><dt>Indexed incidents</dt><dd>{Number(semantic.event_count || 0).toLocaleString()}</dd></div>
              <div><dt>Search evidence</dt><dd>{Number(semantic.evidence_count || 0).toLocaleString()} <small>whole images and object crops</small></dd></div>
              <div><dt>Queue / added since restart</dt><dd>{Number(semantic.queue_depth || 0).toLocaleString()} / {Number(semantic.indexed_since_start || 0).toLocaleString()}</dd></div>
              {semantic.error || semantic.reason ? <div><dt>Last issue</dt><dd>{semantic.error || semantic.reason}</dd></div> : null}
            </dl>
          </section> : null}
          {!selected ? <section className="telemetry-section">
            <div className="telemetry-section-head"><div><h3>Face recognition</h3></div></div>
            <dl className="telemetry-details">
              <div><dt>Recognizable faces</dt><dd>{Number(faceRecognition.actionable_observations || 0).toLocaleString()} <small>{Number(faceRecognition.known || 0).toLocaleString()} identified · {Number(faceRecognition.unknown || 0).toLocaleString()} unknown</small></dd></div>
              <div><dt>Identification rate</dt><dd>{Number(faceRecognition.identified_percent || 0).toFixed(1)}%</dd></div>
              <div><dt>Unusable faces</dt><dd>{Number(faceRecognition.too_small || 0).toLocaleString()} <small>{Number(faceRecognition.processing_failed || 0).toLocaleString()} failures</small></dd></div>
              <div><dt>Candidate frames / multi-frame tracks</dt><dd>{Number(faceRecognition.candidate_frames || 0).toLocaleString()} / {Number(faceRecognition.multi_frame_tracks || 0).toLocaleString()}</dd></div>
              <div><dt>Recognition queue</dt><dd>{Number(faceRecognition.recognition?.queue_depth || 0).toLocaleString()} <small>{Number(faceRecognition.recognition?.pending || 0).toLocaleString()} pending · {Number(faceRecognition.recognition?.failed || 0).toLocaleString()} failed</small></dd></div>
            </dl>
          </section> : null}
        </div>

        {selected ? <section className="telemetry-section">
          <div className="telemetry-section-head"><div><h3>Camera configuration &amp; storage</h3></div></div>
          <div className="telemetry-camera-grid">
            {shownCameras.map((camera) => {
              const analysisRuntime = camera.motion?.analysis_runtime || {};
              const performance = camera.performance || {};
              const analyzed = Number(analysisRuntime.frames_sampled || 0);
              const superseded = Number(analysisRuntime.mailbox_replacements || camera.motion?.analysis_frames_dropped || 0);
              const objectActivity = camera.object_tracking?.object_activity_attribution || {};
              const onvifIssues = Number(camera.onvif?.poll_errors || 0) + Number(camera.onvif?.poll_timeouts || 0) + Number(camera.onvif?.renewal_errors || 0);
              const expected = camera.expected_enabled ?? (camera.lifecycle?.enabled !== false);
              const connectivity = cameraCaptureConnectivity(camera);
              const cameraEventStatus = !camera.onvif?.enabled
                ? "Disabled"
                : !camera.onvif?.connected
                  ? "Unavailable"
                  : onvifIssues
                    ? `Connected · ${onvifIssues.toLocaleString()} recovered issues`
                    : "Healthy";
              const statusClass = !expected ? "disabled" : cameraConnectivityClass(connectivity);
              const statusLabel = !expected ? "Paused" : cameraConnectivityLabel(connectivity);
              return <article className="telemetry-camera-card" key={camera.id}>
                <header><div><strong>{camera.name}</strong><small>{camera.id}</small></div><span className={statusClass}>{statusLabel}</span></header>
                <dl>
                  <div><dt>Recording / detection</dt><dd>{camera.recording ? "On" : "Off"} / {camera.detection_enabled ? "On" : "Off"}</dd></div>
                  <div><dt>Recording timeline</dt><dd>{formatRecorderTimestampHealth(camera.recording_timestamps)}</dd></div>
                  <div><dt>Used-Recordings</dt><dd>{formatBytes(camera.storage?.recording_bytes)}</dd></div>
                  <div><dt>Used-Snapshots</dt><dd>{formatBytes(camera.storage?.snapshot_bytes)}</dd></div>
                  <div><dt>Processing health</dt><dd>{performance.summary || "Collecting a representative processing sample"}</dd></div>
                  <div><dt>Camera event connection</dt><dd>{cameraEventStatus}</dd></div>
                  <div><dt>Recorded decode memory</dt><dd>{camera.recorded_decode?.active_workflows ? `${formatBytes(camera.recorded_decode.reserved_bytes)} · ${camera.recorded_decode.active_workflows} active` : "Idle"}</dd></div>
                </dl>
                <details className="telemetry-technical">
                  <summary>Technical diagnostics</summary>
                  <dl>
                    <div><dt>Lifecycle / workers</dt><dd>{camera.lifecycle?.phase || "unknown"} · {camera.lifecycle?.active_worker_count || 0} active</dd></div>
                    <div><dt>Live decoded FPS</dt><dd>{Number(camera.capture?.live?.fps || 0).toFixed(1)}</dd></div>
                    <div><dt>Live source element</dt><dd>{camera.live_pipeline?.source_element || "Unknown"}</dd></div>
                    <div><dt>Shared gvadetect</dt><dd>{camera.live_pipeline?.model_instance_id || "not reported yet"}</dd></div>
                    <div><dt>Recorded decode reservation</dt><dd>{camera.recorded_decode?.active_workflows ? `${formatBytes(camera.recorded_decode.reserved_bytes)} · ${formatBytes(camera.recorded_decode.frame_bytes)} × ${camera.recorded_decode.frames || 0} frames` : "None"}</dd></div>
                    <div><dt>Main decoder starts</dt><dd>{Number(camera.capture?.main?.starts || 0).toLocaleString()}</dd></div>
                    <div><dt>Read / open failures</dt><dd>{Number(camera.capture?.live?.read_failures || 0) + Number(camera.capture?.main?.read_failures || 0)} / {Number(camera.capture?.live?.open_failures || 0) + Number(camera.capture?.main?.open_failures || 0)}</dd></div>
                    <div><dt>Capture-to-analysis p95 / p99</dt><dd>{formatMilliseconds(analysisRuntime.capture_to_analysis_p95_ms)} / {formatMilliseconds(analysisRuntime.capture_to_analysis_p99_ms)}</dd></div>
                    <div><dt>Performance gates</dt><dd>{(performance.checks || []).map((check) => `${check.label}: ${Number(check.value || 0).toFixed(check.unit === "%" ? 1 : 2)}${check.unit}`).join(" · ") || "Waiting for samples"}</dd></div>
                    <div><dt>Analyzed / stale skipped / deferred</dt><dd>{analyzed.toLocaleString()} / {superseded.toLocaleString()} / {Number(analysisRuntime.analysis_slot_deferrals || 0).toLocaleString()}</dd></div>
                    <div><dt>Motion passed / rejected / suppressed</dt><dd>{camera.motion?.passed || 0} / {camera.motion?.rejected || 0} / {camera.motion?.suppressed || 0}</dd></div>
                    <div><dt>Temporal filter</dt><dd>{(() => {
                      const threshold = Number(config?.motion_qualification?.temporal_filter_threshold ?? 0.005);
                      return threshold > 0
                        ? `Active · ${Number(camera.motion?.analysis_runtime?.temporal_filter_skips || 0).toLocaleString()} skips @ ${threshold.toFixed(3)}`
                        : `Inactive · threshold ${threshold.toFixed(3)}`;
                    })()}</dd></div>
                    <div><dt>Object admission / confidence / zone / confirmation / context</dt><dd>{Number(objectActivity.detector_admissions || 0).toLocaleString()} / {Number(objectActivity.confidence_rejections || 0).toLocaleString()} / {Number(objectActivity.zone_rejections || 0).toLocaleString()} / {Number(objectActivity.temporal_rejections || 0).toLocaleString()} / {Number(objectActivity.enforced_suppressions || 0).toLocaleString()}</dd></div>
                    <div><dt>Event queue peak / evicted / rejected / retry lost</dt><dd>{camera.motion?.event_runtime?.queue_high_water || 0} / {camera.motion?.event_runtime?.evicted || 0} / {camera.motion?.event_runtime?.rejected || 0} / {camera.motion?.event_runtime?.retries_dropped || 0}</dd></div>
                    <div><dt>EMA requests · admitted / merged / failed</dt><dd>{camera.motion?.event_runtime?.episode?.decision_counts?.request_admitted || 0} / {camera.motion?.event_runtime?.episode?.decision_counts?.merged_with_request || 0} / {camera.motion?.event_runtime?.episode?.decision_counts?.detector_failed || 0}</dd></div>
                    <div><dt>ONVIF notices / renewals / issues</dt><dd>{camera.onvif?.notifications || 0} / {camera.onvif?.renewals || 0} / {onvifIssues}</dd></div>
                    <div><dt>Tracking waits / longest / timeouts</dt><dd>{camera.tracking?.capacity_waits || 0} / {Number(camera.tracking?.capacity_wait_seconds_max || 0).toFixed(1)}s / {camera.tracking?.capacity_timeouts || 0}</dd></div>
                    <div><dt>ReID checks / recoveries / failures</dt><dd>{camera.tracking?.reid_attempts || 0} / {camera.tracking?.reid_recoveries || 0} / {camera.tracking?.reid_failures || 0}</dd></div>
                  </dl>
                </details>
              </article>
            })}
          </div>
        </section> : null}
        <p className="telemetry-footnote">Availability, interruptions, EMA coverage, event delivery, and tracking capacity are the primary health signals. “Stale skipped” means a newer frame replaced an older pending sample so analysis stayed current; it matters only when coverage drops persistently. One-minute detail is retained for 48 hours, with compact summaries retained longer.</p>
      </div>
    </TelemetryInterruptionsContext.Provider>
  );
}

export function MaintenanceViewer({ state }) {
  if (!state || state.status === "idle") {
    return <div className="empty-state">Run a storage scan to compare files on disk with SurvNG’s local databases.</div>;
  }
  if (state.status === "running" || state.status === "cancelling") {
    const progress = state.progress || {};
    const percent = Number.isFinite(progress.total) && progress.total > 0 ? Math.min(100, Math.round(Number(progress.current || 0) / progress.total * 100)) : null;
    return <div className="maintenance-running" role="status"><RefreshCcw className="spin" size={20} /><div><strong>{state.status === "cancelling" ? "Cancelling safely…" : state.mode === "repair" ? "Repairing storage records…" : state.full ? "Running full storage scan…" : "Running quick storage check…"}</strong><span>{progress.phase || "Starting"}{percent != null ? ` · ${percent}%` : progress.current ? ` · ${Number(progress.current).toLocaleString()} checked` : ""}</span></div></div>;
  }
  if (state.status === "cancelled") {
    return <div className="maintenance-result-banner warning"><CircleAlert size={20} /><div><strong>Maintenance cancelled</strong><span>No media files were deleted. Run a quick check whenever you are ready.</span></div></div>;
  }
  if (state.status === "failed") {
    return <div className="error-banner"><strong>Maintenance failed</strong><span>{state.error || "Check Logs for details."}</span></div>;
  }
  const result = state.result || {};
  const summary = result.summary || {};
  const repairs = result.repairs || {};
  const missingReferences = (summary.missing_event_snapshots || 0) + (summary.missing_event_recordings || 0) + (summary.missing_motion_snapshots || 0) + (summary.missing_face_snapshots || 0);
  const databaseIssues = (summary.missing_index_rows || 0) + (summary.unindexed_recording_files || 0) + missingReferences;
  const fullScan = result.full === true;
  const cameraRows = Object.entries(summary.per_camera || {});
  const repaired = Object.values(repairs).reduce((total, value) => total + (Number(value) || 0), 0);
  return (
    <div className="maintenance-viewer">
      <div className={`maintenance-result-banner ${databaseIssues ? "warning" : "healthy"}`}>
        {databaseIssues ? <CircleAlert size={20} /> : <CircleDot size={20} />}
        <div><strong>{databaseIssues ? `${databaseIssues.toLocaleString()} database mismatch${databaseIssues === 1 ? "" : "es"} found` : fullScan ? "Storage records are consistent" : "Quick check found no mismatches"}</strong><span>{fullScan ? "Full library checked." : `${Number(summary.index_rows_scanned || 0).toLocaleString()} newest index rows and ${Number(summary.recording_hours_scanned || 0)} recent recording hours checked.`} {result.note}</span></div>
      </div>
      <div className="telemetry-summary-grid maintenance-summary-grid">
        <article><span>{fullScan ? "Recording files" : "Recent files checked"}</span><strong>{Number(summary.recording_files || 0).toLocaleString()}</strong><small>{Number(summary.indexed_recordings || 0).toLocaleString()} total indexed · {Number(summary.recent_recording_files || 0).toLocaleString()} active/recent protected</small></article>
        <article><span>Recording index</span><strong>{Number(summary.missing_index_rows || 0).toLocaleString()} missing</strong><small>{Number(summary.unindexed_recording_files || 0).toLocaleString()} files need indexing</small></article>
        <article><span>Missing incident media</span><strong>{missingReferences.toLocaleString()}</strong><small>{summary.missing_event_snapshots || 0} incident · {summary.missing_motion_snapshots || 0} motion · {summary.missing_face_snapshots || 0} face images</small></article>
        <article><span>Unlinked media</span><strong>{fullScan ? Number(summary.orphan_media_files || 0).toLocaleString() : "Full scan"}</strong><small>{fullScan ? `${formatBytes(summary.orphan_media_bytes)} reported only; never auto-deleted` : "Not walked during the bounded quick check"}</small></article>
        <article><span>Regenerable cache</span><strong>{fullScan ? formatBytes(summary.regenerable_cache_bytes) : "Full scan"}</strong><small>Playback, event clip, and HLS working files</small></article>
        <article><span>Storage free</span><strong>{formatBytes(summary.storage_free_bytes)}</strong><small>{formatBytes(summary.storage_used_bytes)} used of {formatBytes(summary.storage_total_bytes)}</small></article>
      </div>
      {result.mode === "repair" ? <section className="telemetry-section"><div className="telemetry-section-head"><div><h3>Last repair</h3><p>{repaired.toLocaleString()} records updated; no incidents or media files were deleted.</p></div></div><dl className="telemetry-details maintenance-repair-details"><div><dt>Recording rows removed / added</dt><dd>{repairs.stale_index_rows_removed || 0} / {repairs.recordings_reindexed || 0}</dd></div><div><dt>Recordings validated / fingerprinted</dt><dd>{repairs.recordings_validated || 0} / {repairs.recording_fingerprints_added || 0}</dd></div><div><dt>Incident media links cleared</dt><dd>{repairs.event_media_references_cleared || 0}</dd></div><div><dt>Motion / face links cleared</dt><dd>{repairs.motion_sample_references_cleared || 0} / {repairs.face_media_references_cleared || 0}</dd></div></dl></section> : null}
      {cameraRows.length ? <section className="telemetry-section"><div className="telemetry-section-head"><div><h3>Affected cameras</h3><p>Missing media references grouped by camera.</p></div></div><div className="maintenance-camera-list">{cameraRows.map(([cameraId, counts]) => <div key={cameraId}><strong>{cameraId}</strong><span>{Object.entries(counts).map(([kind, count]) => `${String(kind).replaceAll("_", " ")} ${count}`).join(" · ")}</span></div>)}</div></section> : null}
      {(summary.missing_reference_samples?.length || summary.orphan_media_samples?.length || summary.missing_index_samples?.length || summary.unindexed_samples?.length) ? <details className="maintenance-details"><summary>Technical details and sample paths</summary><div>{summary.missing_reference_samples?.length ? <><h4>Missing media references</h4><pre>{summary.missing_reference_samples.map((item) => `${item.camera_id} · ${item.kind} · ${item.path}`).join("\n")}</pre></> : null}{summary.missing_index_samples?.length ? <><h4>Missing recording files still indexed</h4><pre>{summary.missing_index_samples.join("\n")}</pre></> : null}{summary.unindexed_samples?.length ? <><h4>Recording files not indexed</h4><pre>{summary.unindexed_samples.join("\n")}</pre></> : null}{summary.orphan_media_samples?.length ? <><h4>Unlinked media (report only)</h4><pre>{summary.orphan_media_samples.join("\n")}</pre></> : null}</div></details> : null}
    </div>
  );
}

export function CalibrationLab({ cameras, runtimeStatus = [], timeZone, onCommandBarChange = null }) {
  const [runs, setRuns] = useState([]);
  const [changeSets, setChangeSets] = useState([]);
  const [section, setSection] = useState("tuneup");
  const [wizardStep, setWizardStep] = useStoredJsonState("survng.detectionTuneup.step.v1", 1);
  const [selectedRunId, setSelectedRunId] = useStoredJsonState("survng.detectionTuneup.run.v1", null);
  const [selectedRecommendations, setSelectedRecommendations] = useStoredJsonState("survng.detectionTuneup.recommendations.v1", []);
  const [selectedCameras, setSelectedCameras] = useStoredJsonState("survng.detectionTuneup.cameras.v1", cameras.map((camera) => camera.id));
  const [cameraChoice, setCameraChoice] = useStoredState("survng.detectionTuneup.cameraChoice.v1", "all");
  const [mode, setMode] = useStoredState("survng.detectionTuneup.period.v1", "standard");
  const [evaluationHours, setEvaluationHours] = useStoredJsonState("survng.detectionTuneup.monitorHours.v1", 72);
  const [preview, setPreview] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const selectedRun = runs.find((run) => run.id === selectedRunId) || runs[0] || null;
  const statuses = new Map(runtimeStatus.map((item) => [item.id, item]));
  const attentionCameras = cameras.filter((camera) => {
    const status = statuses.get(camera.id) || {};
    return status.running === false || status.frame_fresh === false || Boolean(status.last_error) || Number(status.analysis_frames_dropped || 0) > 0;
  });
  const activeRun = runs.find((run) => ["queued", "running", "cancelling"].includes(run.status));
  const monitoringSets = changeSets.filter((item) => item.action === "apply" && ["collecting", "reviewing", "evaluation_failed", "evaluated"].includes(item.status));

  async function loadCalibration() {
    try {
      const [runResponse, changeResponse] = await Promise.all([
        fetch("/api/calibration/runs?limit=20"),
        fetch("/api/calibration/change-sets?limit=50"),
      ]);
      if (!runResponse.ok || !changeResponse.ok) throw new Error("Calibration history could not be loaded");
      const [runPayload, changePayload] = await Promise.all([runResponse.json(), changeResponse.json()]);
      setRuns((current) => (runPayload.runs || []).map((run) => {
        const existing = current.find((item) => item.id === run.id);
        return existing?.result && Object.keys(existing.result).length
          ? { ...run, result: existing.result }
          : run;
      }));
      setChangeSets(changePayload.change_sets || []);
      setSelectedRunId((current) => current || runPayload.runs?.[0]?.id || null);
      setError("");
    } catch (loadError) {
      setError(loadError.message || "Calibration history could not be loaded");
    }
  }

  async function loadCalibrationRun(runId) {
    if (!runId) return;
    const response = await fetch(`/api/calibration/runs/${runId}`);
    if (!response.ok) throw new Error("Calibration run details could not be loaded");
    const run = await response.json();
    setRuns((current) => current.some((item) => item.id === run.id)
      ? current.map((item) => item.id === run.id ? run : item)
      : [run, ...current]);
  }

  useEffect(() => { void loadCalibration(); }, []);
  useEffect(() => {
    if (!selectedRunId) return;
    void loadCalibrationRun(selectedRunId).catch((loadError) => setError(loadError.message));
  }, [selectedRunId]);
  const calibrationRunActive = runs.some((run) => ["queued", "running", "cancelling"].includes(run.status));
  const calibrationEvaluationActive = changeSets.some((item) => ["collecting", "reviewing"].includes(item.status));
  useVisiblePolling(async () => {
    try {
      await loadCalibration();
      if (selectedRunId) await loadCalibrationRun(selectedRunId);
    } catch (loadError) {
      setError(loadError.message || "Calibration status could not be refreshed");
    }
  }, calibrationRunActive ? 2000 : 10000, calibrationRunActive || calibrationEvaluationActive, { immediate: false, restartKey: selectedRunId || "" });
  useEffect(() => {
    const ids = new Set(cameras.map((camera) => camera.id));
    setSelectedCameras((current) => current.filter((id) => ids.has(id)));
  }, [cameras]);
  useEffect(() => { setPreview(null); }, [selectedRunId, selectedRecommendations]);

  const calibrationSectionLabel = section === "monitoring" ? "Monitoring" : section === "history" ? "Tune-Up History" : "Detection Tune-Up";
  useEffect(() => {
    onCommandBarChange?.({
      sectionLabel: calibrationSectionLabel,
      refresh: () => { void loadCalibration(); },
    });
    return () => onCommandBarChange?.(null);
  }, [calibrationSectionLabel, onCommandBarChange]);

  function chooseCameraScope(choice) {
    setCameraChoice(choice);
    if (choice === "all") setSelectedCameras(cameras.map((camera) => camera.id));
    if (choice === "attention") setSelectedCameras(attentionCameras.map((camera) => camera.id));
  }

  async function startRun(override = false) {
    if (!selectedCameras.length) return setError("Select at least one camera.");
    if (mode === "deep" && !override && !window.confirm(`Deep analysis can review up to 40 images for each of ${selectedCameras.length} selected cameras and may use substantial AI API capacity. Continue?`)) return;
    setBusy(true); setError("");
    try {
      let response = await fetch("/api/calibration/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ camera_ids: selectedCameras, mode, override_active_evaluation: override }),
      });
      let payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        if (response.status === 409 && !override && window.confirm(`${payload.detail}\n\nStart a new analysis anyway?`)) {
          response = await fetch("/api/calibration/runs", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ camera_ids: selectedCameras, mode, override_active_evaluation: true }),
          });
          payload = await response.json().catch(() => ({}));
        }
      }
      if (!response.ok) {
        throw new Error(typeof payload.detail === "string" ? payload.detail : "Calibration could not start");
      }
      setSelectedRunId(payload.id);
      setSelectedRecommendations([]);
      setWizardStep(3);
      await loadCalibration();
    } catch (runError) { setError(runError.message || "Calibration could not start"); }
    finally { setBusy(false); }
  }

  async function previewSelected() {
    if (!selectedRun) return;
    if (!selectedRecommendations.length) {
      setError("Select at least one suggested change before continuing.");
      return;
    }
    setBusy(true); setError("");
    try {
      const response = await fetch(`/api/calibration/runs/${selectedRun.id}/preview`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ recommendation_ids: selectedRecommendations, configuration_fingerprint: selectedRun.result?.configuration_fingerprint || selectedRun.configuration_fingerprint }) });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : "Selected changes are not ready to apply");
      setPreview(payload); setWizardStep(5);
    } catch (previewError) { setError(previewError.message || "Selected changes are not ready to apply"); }
    finally { setBusy(false); }
  }

  async function applySelected() {
    if (!selectedRun || !selectedRecommendations.length) return;
    if (!window.confirm(`Apply ${selectedRecommendations.length} selected calibration change${selectedRecommendations.length === 1 ? "" : "s"}? SurvNG will validate one candidate configuration and reload only affected services.`)) return;
    setBusy(true); setError("");
    try {
      const response = await fetch(`/api/calibration/runs/${selectedRun.id}/apply`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ recommendation_ids: selectedRecommendations, confirmed: true, configuration_fingerprint: selectedRun.result?.configuration_fingerprint || selectedRun.configuration_fingerprint, evaluation_hours: evaluationHours }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : "Calibration changes could not be applied");
      setSelectedRecommendations([]);
      await loadCalibration();
      setSection("monitoring"); setWizardStep(6);
    } catch (applyError) { setError(applyError.message || "Calibration changes could not be applied"); }
    finally { setBusy(false); }
  }

  async function simpleAction(url, fallback) {
    setBusy(true); setError("");
    try {
      const response = await fetch(url, { method: "POST" });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : fallback);
      if (payload.id) setSelectedRunId(payload.id);
      await loadCalibration();
      return payload;
    } catch (actionError) { setError(actionError.message || fallback); return null; }
    finally { setBusy(false); }
  }

  function runAnotherTuneup() {
    setSection("tuneup"); setWizardStep(1); setSelectedRunId(null); setSelectedRecommendations([]); setPreview(null);
  }

  function setRecommendationSelected(recommendationId, selected) {
    setError("");
    setSelectedRecommendations((current) => selected
      ? [...new Set([...current, recommendationId])]
      : current.filter((id) => id !== recommendationId));
  }

  function toggleRecommendationFromCard(event, recommendationId) {
    if (event.target.closest("a, button, input, label, details, summary")) return;
    setRecommendationSelected(recommendationId, !selectedRecommendations.includes(recommendationId));
  }

  async function rollback(changeSet, { changeIds = [], cameraIds = [] } = {}) {
    const scopeLabel = changeIds.length ? "this setting" : cameraIds.length ? "this camera's settings" : "all settings";
    if (!window.confirm(`Roll back ${scopeLabel} from change set #${changeSet.id}? Newer conflicting values will be preserved.`)) return;
    setBusy(true); setError("");
    try {
      let response = await fetch(`/api/calibration/change-sets/${changeSet.id}/rollback`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ confirmed: true, change_ids: changeIds, camera_ids: cameraIds, force_conflicts: false }) });
      let payload = await response.json().catch(() => ({}));
      if (response.status === 409 && payload.detail?.conflicts?.length && window.confirm(`${payload.detail.message}. Replace the ${payload.detail.conflicts.length} newer conflicting value${payload.detail.conflicts.length === 1 ? "" : "s"} anyway?`)) {
        response = await fetch(`/api/calibration/change-sets/${changeSet.id}/rollback`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ confirmed: true, change_ids: changeIds, camera_ids: cameraIds, force_conflicts: true }) });
        payload = await response.json().catch(() => ({}));
      }
      if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : payload.detail?.message || "Rollback could not be completed");
      await loadCalibration();
    } catch (rollbackError) { setError(rollbackError.message || "Rollback could not be completed"); }
    finally { setBusy(false); }
  }

  async function evaluate(changeSet) {
    setBusy(true); setError("");
    try {
      const response = await fetch(`/api/calibration/change-sets/${changeSet.id}/evaluate`, { method: "POST" });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : "Evaluation could not start");
      await loadCalibration();
    } catch (evaluationError) { setError(evaluationError.message || "Evaluation could not start"); }
    finally { setBusy(false); }
  }

  const recommendations = selectedRun?.result?.recommendations || [];
  const recommendationGroups = recommendations.reduce((groups, item) => {
    const group = tuneupRecommendationGroup(item);
    return { ...groups, [group]: [...(groups[group] || []), item] };
  }, {});
  const completed = Number(selectedRun?.result?.progress?.completed || 0);
  const total = Number(selectedRun?.result?.progress?.total || selectedRun?.camera_ids?.length || 0);
  return <section id="admin-panel-calibration" className="bento-card config-editor settings-panel settings-panel-wide calibration-panel" aria-labelledby="admin-destination-tuneup">
    <div className="calibration-tabs-shell">
      <div className="tree-list tuneup-section-list admin-section-tabs camera-section-tabs detection-subsection-tabs" role="tablist" aria-label="Detection Tune-Up sections" onKeyDown={(event) => { const next = nextTabId(["tuneup", "monitoring", "history"], section, event.key); if (!next) return; event.preventDefault(); setSection(next); window.requestAnimationFrame(() => document.getElementById(`tuneup-tab-${next}`)?.focus()); }}>
        <button id="tuneup-tab-tuneup" type="button" tabIndex={section === "tuneup" ? 0 : -1} aria-controls="tuneup-section-panel" className={section === "tuneup" ? "active" : ""} onClick={() => setSection("tuneup")} role="tab" aria-selected={section === "tuneup"}><Sparkles size={16} /><span>Tune-Up</span></button>
        <button id="tuneup-tab-monitoring" type="button" tabIndex={section === "monitoring" ? 0 : -1} aria-controls="tuneup-section-panel" className={section === "monitoring" ? "active" : ""} onClick={() => setSection("monitoring")} role="tab" aria-selected={section === "monitoring"}><Activity size={16} /><span>Monitoring{monitoringSets.length ? <em>{monitoringSets.length}</em> : null}</span></button>
        <button id="tuneup-tab-history" type="button" tabIndex={section === "history" ? 0 : -1} aria-controls="tuneup-section-panel" className={section === "history" ? "active" : ""} onClick={() => setSection("history")} role="tab" aria-selected={section === "history"}><Clock3 size={16} /><span>History</span></button>
      </div>
      {activeRun ? <button type="button" className="tuneup-resume-card" onClick={() => { setSelectedRunId(activeRun.id); setSection("tuneup"); setWizardStep(3); }}><RefreshCcw className="spin" size={16} /><span><strong>Review in progress</strong><small>{tuneupHistoryTitle(activeRun, cameras)}</small></span></button> : null}
    </div>
      {error ? <div className="error-banner">{error}</div> : null}
      <div id="tuneup-section-panel" role="tabpanel" aria-labelledby={`tuneup-tab-${section}`}>
        {section === "tuneup" ? <div className="tuneup-workflow">
          <nav className="tuneup-steps" aria-label="Tune-Up progress">{["Choose cameras", "Review period", "Review performance", "Choose changes", "Confirm", "Monitor", "Results"].map((label, index) => <span className={wizardStep === index + 1 ? "active" : wizardStep > index + 1 ? "done" : ""} key={label}><b>{wizardStep > index + 1 ? <Check size={13} /> : index + 1}</b>{label}</span>)}</nav>
          {wizardStep === 1 ? <div className="tuneup-stage"><header><span>Step 1 of 7</span><h3>Which cameras should SurvNG review?</h3></header><div className="tuneup-choice-grid">
            <button type="button" className={cameraChoice === "all" ? "selected" : ""} onClick={() => chooseCameraScope("all")}><ShieldCheck size={22} /><strong>All cameras</strong><em>Recommended</em><small>Looks for system-wide patterns and camera-specific exceptions.</small></button>
            <button type="button" className={cameraChoice === "attention" ? "selected" : ""} onClick={() => chooseCameraScope("attention")} disabled={!attentionCameras.length}><CircleAlert size={22} /><strong>Cameras needing attention</strong><small>{attentionCameras.length ? `${attentionCameras.length} cameras have current health or analysis concerns.` : "No cameras currently need attention."}</small></button>
            <button type="button" className={cameraChoice === "custom" ? "selected" : ""} onClick={() => setCameraChoice("custom")}><Camera size={22} /><strong>Choose cameras</strong><small>Review only the scenes you select.</small></button>
          </div>{cameraChoice === "custom" ? <div className="calibration-camera-list tuneup-camera-list">{cameras.map((camera) => <label key={camera.id}><input type="checkbox" checked={selectedCameras.includes(camera.id)} onChange={(event) => setSelectedCameras((current) => event.target.checked ? [...new Set([...current, camera.id])] : current.filter((id) => id !== camera.id))} /><span>{camera.name || camera.id}</span></label>)}</div> : null}<footer><span>{selectedCameras.length} of {cameras.length} cameras selected</span><button className="primary" disabled={!selectedCameras.length} onClick={() => setWizardStep(2)}>Continue <ArrowRight size={16} /></button></footer></div> : null}
          {wizardStep === 2 ? <div className="tuneup-stage"><header><span>Step 2 of 7</span><h3>How much history should be reviewed?</h3><p>Longer periods see more scene conditions but take longer and use more AI analysis.</p></header><div className="tuneup-period-grid">{Object.entries(TUNEUP_PERIODS).map(([value, period]) => <button type="button" className={mode === value ? "selected" : ""} onClick={() => setMode(value)} key={value}><Clock3 size={21} /><strong>{period.label}</strong>{value === "standard" ? <em>Recommended</em> : null}<small>{period.detail}</small></button>)}</div><footer><button onClick={() => setWizardStep(1)}><ArrowLeft size={16} />Back</button><button className="primary" onClick={() => void startRun()} disabled={busy || Boolean(activeRun)}>{busy ? <RefreshCcw className="spin" size={16} /> : <Sparkles size={16} />}Start review</button></footer></div> : null}
          {wizardStep === 3 ? <div className="tuneup-stage tuneup-reviewing"><header><span>Step 3 of 7</span><h3>{["queued", "running", "cancelling"].includes(selectedRun?.status) ? `Reviewing ${Math.min(completed + 1, total || 1)} of ${total} cameras` : selectedRun?.status === "completed" ? "Review complete" : "Review could not be completed"}</h3><p>{["queued", "running", "cancelling"].includes(selectedRun?.status) ? "It is safe to leave this page. SurvNG saves progress and this workflow will resume when you return." : selectedRun?.error}</p></header><div className="tuneup-progress-track"><span style={{ width: `${total ? Math.round((completed / total) * 100) : 0}%` }} /></div><div className="tuneup-camera-progress">{(selectedRun?.camera_ids || selectedCameras).map((cameraId, index) => { const failed = selectedRun?.result?.camera_errors?.[cameraId]; return <div className={failed ? "failed" : index < completed ? "complete" : index === completed && selectedRun?.status === "running" ? "active" : "pending"} key={cameraId}>{index < completed && !failed ? <Check size={15} /> : failed ? <CircleAlert size={15} /> : index === completed && selectedRun?.status === "running" ? <RefreshCcw className="spin" size={15} /> : <CircleDot size={15} />}<span><strong>{cameras.find((camera) => camera.id === cameraId)?.name || cameraId}</strong>{failed ? <small>{failed}</small> : null}</span></div>; })}</div><footer>{["queued", "running", "cancelling"].includes(selectedRun?.status) ? <button onClick={() => void simpleAction(`/api/calibration/runs/${selectedRun.id}/cancel`, "Analysis could not be cancelled")} disabled={busy || selectedRun.status === "cancelling"}>{selectedRun.status === "cancelling" ? "Stopping…" : "Cancel review"}</button> : null}{selectedRun?.result?.camera_errors && Object.keys(selectedRun.result.camera_errors).length ? <button onClick={() => void simpleAction(`/api/calibration/runs/${selectedRun.id}/retry`, "Failed cameras could not be retried")} disabled={busy}>Retry failed cameras</button> : null}{selectedRun?.status === "completed" ? <button className="primary" onClick={() => setWizardStep(4)}>Review suggestions <ArrowRight size={16} /></button> : null}</footer></div> : null}
          {wizardStep === 4 ? <div className="tuneup-stage"><header><span>Step 4 of 7</span><h3>Choose suggested changes</h3><p>Select only the improvements you want SurvNG to make. Click anywhere on a suggestion card to select it.</p></header><div className="tuneup-summary-line"><ShieldCheck size={20} /><span><strong>{selectedRun?.result?.summary}</strong><small>{recommendations.length} bounded suggestion{recommendations.length === 1 ? "" : "s"}</small></span>{recommendations.length ? <div className="tuneup-selection-actions"><button type="button" onClick={() => { setError(""); setSelectedRecommendations(recommendations.map((item) => item.id)); }}>Select all</button><button type="button" onClick={() => { setError(""); setSelectedRecommendations([]); }}>Clear</button></div> : null}</div>{Object.entries(recommendationGroups).map(([group, items]) => <section className="tuneup-recommendation-group" key={group}><h4>{group}</h4>{items.map((item) => <article className={`${selectedRecommendations.includes(item.id) ? "selected" : ""} selectable`} key={item.id} onClick={(event) => toggleRecommendationFromCard(event, item.id)}><label><input type="checkbox" aria-label={`Select ${TUNEUP_SETTING_NAMES[item.setting] || item.setting}`} checked={selectedRecommendations.includes(item.id)} onChange={(event) => setRecommendationSelected(item.id, event.target.checked)} /><span><strong>{selectedRecommendations.includes(item.id) ? "Selected" : "Select change"} · {item.scope === "global" ? "All applicable cameras" : cameras.find((camera) => camera.id === item.camera_id)?.name || item.camera_id}</strong><small>{TUNEUP_SETTING_NAMES[item.setting] || String(item.setting || "Setting").split(".").pop().replaceAll("_", " ")}</small></span></label><div className="tuneup-before-after"><span><small>Now</small><b>{tuneupValue(item.current_effective ?? item.current)}</b></span><ArrowRight size={17} /><span><small>Suggested</small><b>{tuneupValue(item.proposed)}</b></span></div><p>{item.expected_benefit}</p><small className="tuneup-tradeoff"><b>Tradeoff:</b> {item.downside}</small>{item.evidence?.length ? <div className="calibration-evidence">{item.evidence.slice(0, 6).map((evidence, index) => evidence.image_url ? <a href={evidence.event_id ? appUrl(`/incidents?event_ids=${evidence.event_id}`) : appUrl(evidence.image_url)} key={`${evidence.record_id || evidence.id || index}-${index}`} title={`Open exact ${evidence.event_id ? "incident" : "motion audit"}`}><img src={appUrl(evidence.image_url)} alt={`Evidence ${index + 1} for ${item.camera_id || "all cameras"}`} loading="lazy" /><span>{evidence.event_id ? "Incident" : "Motion audit"}</span></a> : null)}</div> : null}<details><summary>Technical details</summary><dl><div><dt>Setting</dt><dd><code>{item.setting}</code></dd></div><div><dt>Evidence</dt><dd>{item.evidence_strength} · {item.support_count || 0} samples</dd></div><div><dt>Processing impact</dt><dd>{item.compute_impact}</dd></div></dl>{item.effective_preview?.length > 1 ? <div className="calibration-effective-preview">{item.effective_preview.map((camera) => <div key={camera.camera_id}><span>{cameras.find((entry) => entry.id === camera.camera_id)?.name || camera.camera_id}</span><code>{JSON.stringify(camera.current)} → {JSON.stringify(camera.proposed)}</code></div>)}</div> : null}</details></article>)}</section>)}{!recommendations.length ? <div className="empty-state">No safe setting change was supported by the reviewed evidence.</div> : null}<details className="calibration-camera-findings"><summary>Camera review notes ({selectedRun?.result?.camera_summaries?.length || 0})</summary>{selectedRun?.result?.camera_summaries?.map((camera) => <article key={camera.camera_id}><strong>{camera.camera_name}</strong><span>{camera.summary}</span><small>{camera.analyzed} reviewed · {camera.failed} failed</small></article>)}</details><footer><button onClick={() => setWizardStep(3)}><ArrowLeft size={16} />Back</button><span>{selectedRecommendations.length ? `${selectedRecommendations.length} selected` : "Select at least one change to continue"}</span><button className="primary" onClick={() => void previewSelected()} disabled={busy}>Review selected changes <ArrowRight size={16} /></button></footer></div> : null}
          {wizardStep === 5 ? <div className="tuneup-stage"><header><span>Step 5 of 7</span><h3>Confirm and apply</h3><p>Only the changes below will be applied. SurvNG validated them together against the current configuration, and every change is reversible.</p></header>{preview?.ready ? <div className="tuneup-readiness"><ShieldCheck size={22} /><span><strong>Ready to apply</strong><small>No configuration drift or recommendation conflicts were found.</small></span></div> : null}<div className="tuneup-confirm-list">{preview?.changes?.map((change) => <div key={`${change.camera_id}-${change.setting}`}><span><strong>{change.camera_id ? cameras.find((camera) => camera.id === change.camera_id)?.name || change.camera_id : "System default"}</strong><small>{TUNEUP_SETTING_NAMES[change.setting] || String(change.setting).split(".").pop().replaceAll("_", " ")}</small></span><b>{tuneupValue(change.before)} <ArrowRight size={14} /> {tuneupValue(change.after)}</b></div>)}</div><label className="tuneup-monitor-duration"><span><strong>Monitor results for</strong><small>SurvNG will compare matched evidence after this observation period.</small></span><select value={evaluationHours} onChange={(event) => setEvaluationHours(Number(event.target.value))}><option value={24}>24 hours</option><option value={72}>3 days</option><option value={168}>7 days</option></select></label><div className="tuneup-warning"><CircleAlert size={18} /><span><strong>Expected tradeoffs</strong><small>{selectedRecommendations.map((id) => recommendations.find((item) => item.id === id)?.downside).filter(Boolean).join(" ")}</small></span></div><footer><button onClick={() => setWizardStep(4)}><ArrowLeft size={16} />Back</button><button className="primary" onClick={() => void applySelected()} disabled={busy || !preview?.ready}><Check size={16} />Apply {preview?.change_count || selectedRecommendations.length} changes</button></footer></div> : null}
        </div> : null}
        {section === "monitoring" ? <div className="tuneup-monitoring">{monitoringSets.length ? monitoringSets.map((item) => { const rolledBack = new Set(item.rolled_back_change_ids || []); const remaining = (item.changes || []).filter((change) => !rolledBack.has(change.id)); const [outcome, tone] = tuneupOutcome(item); const affected = [...new Set((item.changes || []).flatMap((change) => change.camera_id ? [change.camera_id] : (runs.find((run) => run.id === item.run_id)?.camera_ids || [])))]; return <article className="tuneup-monitor-card" key={item.id}><header><span><strong>{item.status === "collecting" ? "Monitoring changes" : item.status === "reviewing" ? "Reviewing results" : outcome}</strong><small>{formatDateTime(item.created_at, timeZone)} · {item.changes?.length || 0} changes</small></span><em className={tone}>{String(item.status).replaceAll("_", " ")}</em></header>{item.status === "collecting" ? <div className="tuneup-countdown"><Clock3 size={19} /><span><strong>{item.seconds_until_ready > 86400 ? `${Math.ceil(item.seconds_until_ready / 86400)} days remaining` : item.seconds_until_ready > 3600 ? `${Math.ceil(item.seconds_until_ready / 3600)} hours remaining` : "Ready for review"}</strong><small>SurvNG is collecting matched follow-up evidence.</small></span></div> : null}<div className="tuneup-health-list">{affected.map((cameraId) => { const status = statuses.get(cameraId) || {}; const healthy = status.running !== false && status.frame_fresh !== false; return <span className={healthy ? "healthy" : "unhealthy"} key={cameraId}><CircleDot size={13} />{cameras.find((camera) => camera.id === cameraId)?.name || cameraId}</span>; })}</div>{item.evaluation?.summary ? <p>{item.evaluation.summary}</p> : null}<details><summary>Applied changes</summary>{remaining.map((change) => <div className="tuneup-change-row" key={change.id}><span>{TUNEUP_SETTING_NAMES[change.setting] || String(change.setting).split(".").pop().replaceAll("_", " ")}</span><b>{tuneupValue(change.before)} → {tuneupValue(change.after)}</b><button onClick={() => void rollback(item, { changeIds: [change.id] })} disabled={busy}><Undo2 size={14} />Undo</button></div>)}</details><footer>{item.status === "collecting" && item.seconds_until_ready <= 0 ? <button onClick={() => void evaluate(item)} disabled={busy}><Activity size={15} />Review now</button> : null}{item.status === "evaluated" ? <><button onClick={runAnotherTuneup}><Plus size={15} />Run another</button><button className="primary" onClick={() => void simpleAction(`/api/calibration/change-sets/${item.id}/keep`, "Changes could not be marked as kept")} disabled={busy}><Check size={15} />Keep changes</button></> : null}{remaining.length ? <button onClick={() => void rollback(item)} disabled={busy}><Undo2 size={15} />Undo changes</button> : null}</footer></article>; }) : <div className="empty-state"><ShieldCheck size={28} /><strong>No tune-up is being monitored</strong><span>Apply a recommendation to begin a before-and-after review.</span><button className="primary" onClick={runAnotherTuneup}>Run a tune-up</button></div>}</div> : null}
        {section === "history" ? <div className="tuneup-history"><div className="tuneup-history-actions"><span>{runs.length} recent review{runs.length === 1 ? "" : "s"}</span><button className="primary" onClick={runAnotherTuneup}><Plus size={15} />Run another tune-up</button></div>{runs.map((run) => { const applied = changeSets.filter((item) => item.run_id === run.id && item.action === "apply"); return <article key={run.id}><button type="button" onClick={() => { setSelectedRunId(run.id); setSection("tuneup"); setWizardStep(run.status === "completed" ? 4 : 3); }}><span><strong>{tuneupHistoryTitle(run, cameras)}</strong><small>{formatDateTime(run.created_at, timeZone)} · {String(run.status).replaceAll("_", " ")}</small></span><ArrowRight size={16} /></button><div><span>{applied.reduce((count, item) => count + Number(item.changes?.length || 0), 0)} changes applied</span>{applied.map((item) => <em key={item.id}>{item.evaluation?.summary || String(item.status).replaceAll("_", " ")}</em>)}</div>{applied.map((item) => { const rolledBack = new Set(item.rolled_back_change_ids || []); const remaining = (item.changes || []).filter((change) => !rolledBack.has(change.id)); return remaining.length ? <details className="tuneup-history-changes" key={item.id}><summary>Review or undo {remaining.length} applied change{remaining.length === 1 ? "" : "s"}</summary>{remaining.map((change) => <div className="tuneup-change-row" key={change.id}><span>{change.camera_id ? cameras.find((camera) => camera.id === change.camera_id)?.name || change.camera_id : "System default"} · {TUNEUP_SETTING_NAMES[change.setting] || String(change.setting).split(".").pop().replaceAll("_", " ")}</span><b>{tuneupValue(change.before)} → {tuneupValue(change.after)}</b><button onClick={() => void rollback(item, { changeIds: [change.id] })} disabled={busy}><Undo2 size={14} />Undo</button></div>)}</details> : null; })}</article>; })}{!runs.length ? <div className="empty-state">No tune-ups have been run yet.</div> : null}</div> : null}
      </div>
  </section>;
}

export function ConfigPage({ timeZone, setTimeZone, theme, setTheme, onAssistantContextChange }) {
  const initialAdminSearch = useMemo(() => window.location.search, []);
  const initialAdminParams = useMemo(() => new URLSearchParams(initialAdminSearch), [initialAdminSearch]);
  const initialAdminWorkspace = useMemo(() => readAdminWorkspace(
    initialAdminSearch,
    readStoredValue(browserStorage(window), "survng.configTab", "general"),
  ), [initialAdminSearch]);
  const [config, setConfig] = useState(null);
  const [runtimeStatus, setRuntimeStatus] = useState([]);
  const [accelerator, setAccelerator] = useState(null);
  const [detectorModels, setDetectorModels] = useState([]);
  const [recordingCache, setRecordingCache] = useState(null);
  const [retentionStatus, setRetentionStatus] = useState(null);
  const [retentionError, setRetentionError] = useState("");
  const [mqttStatus, setMqttStatus] = useState(null);
  const [detectorStatus, setDetectorStatus] = useState(null);
  const [motionCatalog, setMotionCatalog] = useState(null);
  const [settingsTab, setSettingsTab] = useStoredState("survng.configTab", initialAdminWorkspace, { preferInitial: initialAdminParams.has("section") });
  const [generalSection, setGeneralSection] = useStoredState("survng.generalSection.v1", readAdminSubsection(initialAdminSearch, GENERAL_ADMIN_SECTIONS, "general"), { preferInitial: initialAdminWorkspace === "general" && initialAdminParams.has("subsection") });
  const [cameraSection, setCameraSection] = useStoredState("survng.cameraSection.v1", readAdminSubsection(initialAdminSearch, CAMERA_ADMIN_SECTIONS, "settings"), { preferInitial: initialAdminWorkspace === "cameras" && initialAdminParams.has("subsection") });
  const [selectedId, setSelectedId] = useState(() => new URLSearchParams(initialAdminSearch).get("camera") || "");
  const [saveNotice, setSaveNotice] = useState(null);
  const [configLoadError, setConfigLoadError] = useState("");
  const [generalSaving, setGeneralSaving] = useState(false);
  const [zonesSaving, setZonesSaving] = useState(false);
  const [cameraSaving, setCameraSaving] = useState(false);
  const [cameraOrderEditing, setCameraOrderEditing] = useState(false);
  const [cameraOrderSaving, setCameraOrderSaving] = useState(false);
  const [dragConfigCameraId, setDragConfigCameraId] = useState("");
  const [dragConfigCameraTarget, setDragConfigCameraTarget] = useState("");
  const [dragConfigCameraAfter, setDragConfigCameraAfter] = useState(false);
  const cameraOrderOriginalRef = useRef([]);
  const [probe, setProbe] = useState(null);
  const [logLines, setLogLines] = useState([]);
  const logSignatureRef = useRef("");
  const logRequestRef = useRef({ generation: 0, controller: null });
  const [logFilter, setLogFilter] = useStoredState("survng.logFilter.v1", "");
  const [logLevel, setLogLevel] = useStoredState("survng.logLevel.v1", "INFO");
  const [logOrder, setLogOrder] = useStoredState("survng.logOrder.v1", "newest");
  const [debouncedLogQuery, setDebouncedLogQuery] = useState(() => ({ level: logLevel, filter: logFilter }));
  const [auditItems, setAuditItems] = useState([]);
  const [auditTotal, setAuditTotal] = useState(0);
  const [auditPage, setAuditPage] = useState(0);
  const [auditCamera, setAuditCamera] = useState(() => {
    const fromUrl = initialAdminParams.get("camera") || "";
    if (fromUrl) return fromUrl;
    return readStoredValue(browserStorage(window), "survng.motionAuditCamera.v1", "");
  });
  const [auditCategory, setAuditCategory] = useStoredState("survng.motionAuditCategory.v1", "all");
  const [auditOutcome, setAuditOutcome] = useStoredState("survng.motionAuditOutcome.v1", "all");
  const [auditLoading, setAuditLoading] = useState(false);
  const [auditError, setAuditError] = useState("");
  const [selectedAuditId, setSelectedAuditId] = useState(null);
  const [linkedAudit, setLinkedAudit] = useState(null);
  const [telemetry, setTelemetry] = useState(null);
  const [telemetryLoading, setTelemetryLoading] = useState(false);
  const [telemetryError, setTelemetryError] = useState("");
  const [telemetrySection, setTelemetrySection] = useStoredState("survng.telemetrySection.v1", normalizeTelemetrySection(readAdminSubsection(
    initialAdminSearch,
    TELEMETRY_ADMIN_SECTIONS,
    "health",
  )), { preferInitial: initialAdminWorkspace === "telemetry" && (initialAdminParams.has("subsection") || initialAdminParams.has("camera")) });
  const [telemetryCamera, setTelemetryCamera] = useStoredState("survng.telemetryCamera.v1", initialAdminParams.get("camera") || "", { preferInitial: initialAdminWorkspace === "telemetry" && initialAdminParams.has("camera") });
  const [diagnosticScope, setDiagnosticScope] = useState("system");
  const [diagnosticDuration, setDiagnosticDuration] = useState("3600");
  const [maintenance, setMaintenance] = useState(null);
  const [maintenanceError, setMaintenanceError] = useState("");
  const [adminNavOpen, setAdminNavOpen] = useState(false);
  const [calibrationCommandBar, setCalibrationCommandBar] = useState(null);
  const [generalViewNonce, setGeneralViewNonce] = useState(0);
  const [detectionSection, setDetectionSection] = useState("object");
  const [calibrationViewNonce, setCalibrationViewNonce] = useState(0);
  const [apiTokenSecretVisible, setApiTokenSecretVisible] = useState(false);
  const configLoadSequence = useRef(0);
  const adminHistoryWriteRef = useRef(true);
  const adminDirtyRef = useRef(false);
  const apiTokenSecretVisibleRef = useRef(apiTokenSecretVisible);
  const acceptedAdminLocationRef = useRef(`${window.location.pathname}${window.location.search}${window.location.hash}`);
  const baselineConfigRef = useRef(null);
  const [baselineRevision, setBaselineRevision] = useState(0);
  const auditPageSize = 24;

  function adminLocationOptions(section = settingsTab) {
    if (section === "general") return {
      subsection: generalSection === "general" ? "" : generalSection,
      camera: generalSection === "motion-review" ? selectedId : "",
    };
    if (section === "cameras") return { subsection: cameraSection === "settings" ? "" : cameraSection, camera: selectedId };
    if (section === "telemetry") return telemetryLocationOptions(telemetrySection, telemetryCamera);
    if (section === "audit") return { camera: auditCamera };
    return {};
  }

  useEffect(() => {
    onAssistantContextChange?.({
      page: "admin",
      camera_id: settingsTab === "cameras" ? selectedId : "",
      filters: { section: settingsTab, general_section: generalSection, camera_section: cameraSection },
    });
  }, [cameraSection, generalSection, onAssistantContextChange, selectedId, settingsTab]);

  useEffect(() => {
    if (!adminHistoryWriteRef.current) {
      adminHistoryWriteRef.current = true;
      return;
    }
    const search = adminWorkspaceSearch(settingsTab, window.location.search, adminLocationOptions());
    const location = appUrl(`/admin${search}`);
    window.history.replaceState(window.history.state, "", location);
    acceptedAdminLocationRef.current = location;
  }, [auditCamera, cameraSection, generalSection, selectedId, settingsTab, telemetryCamera, telemetrySection]);

  useEffect(() => {
    function restoreAdminWorkspace() {
      const requestedLocation = `${window.location.pathname}${window.location.search}${window.location.hash}`;
      if (adminDirtyRef.current || apiTokenSecretVisibleRef.current) {
        const warning = adminDirtyRef.current && apiTokenSecretVisibleRef.current
          ? "Leave this Admin view, discard unsaved changes, and discard the one-time API token secret?"
          : apiTokenSecretVisibleRef.current
            ? "The new API token secret is shown only once. Leave this Admin view and discard the displayed secret?"
            : "Leave this Admin view and discard unsaved changes?";
        if (!window.confirm(warning)) {
          window.history.pushState(window.history.state, "", acceptedAdminLocationRef.current);
          return;
        }
        if (apiTokenSecretVisibleRef.current) setApiTokenSecretVisible(false);
        if (adminDirtyRef.current) discardAdminChanges();
      }
      adminHistoryWriteRef.current = false;
      const nextSection = readAdminWorkspace(window.location.search, settingsTab);
      setSettingsTab(nextSection);
      if (nextSection === "general") setGeneralSection(readAdminSubsection(window.location.search, GENERAL_ADMIN_SECTIONS, generalSection));
      if (nextSection === "cameras") {
        setCameraSection(readAdminSubsection(window.location.search, CAMERA_ADMIN_SECTIONS, cameraSection));
        setSelectedId(new URLSearchParams(window.location.search).get("camera") || selectedId);
      }
      if (nextSection === "telemetry") {
        setTelemetrySection(normalizeTelemetrySection(readAdminSubsection(window.location.search, TELEMETRY_ADMIN_SECTIONS, telemetrySection)));
        setTelemetryCamera(new URLSearchParams(window.location.search).get("camera") || telemetryCamera);
      }
      if (nextSection === "audit") {
        setAuditCamera(new URLSearchParams(window.location.search).get("camera") || "");
      }
      acceptedAdminLocationRef.current = requestedLocation;
    }
    window.addEventListener("popstate", restoreAdminWorkspace);
    return () => window.removeEventListener("popstate", restoreAdminWorkspace);
  }, [cameraSection, generalSection, selectedId, setSettingsTab, settingsTab, telemetryCamera, telemetrySection]);

  function openOccupancySetting(setting = {}) {
    if (!confirmDiscardAdminChanges("Switch sections and discard unsaved changes?")) return;
    const cameraId = setting.cameraId || "";
    if (setting.workspace === "audit") {
      if (cameraId) setAuditCamera(cameraId);
      setAuditCategory(setting.category || "visual_backup");
      setAuditOutcome("all");
      setAuditPage(0);
      setSelectedAuditId(null);
      setSettingsTab("audit");
      const search = adminWorkspaceSearch("audit", window.location.search, { camera: cameraId });
      const location = appUrl(`/admin${search}`);
      window.history.pushState({ ...(window.history.state || {}), survngAdminSection: "audit" }, "", location);
      acceptedAdminLocationRef.current = location;
      adminHistoryWriteRef.current = false;
      setAdminNavOpen(false);
      return;
    }
    if (setting.workspace === "cameras") {
      const nextCameraId = cameraId || selectedId || cameras[0]?.id || "";
      if (nextCameraId) setSelectedId(nextCameraId);
      const subsection = setting.subsection || "motion";
      setCameraSection(subsection);
      setSettingsTab("cameras");
      const search = adminWorkspaceSearch("cameras", window.location.search, {
        subsection: subsection === "settings" ? "" : subsection,
        camera: nextCameraId,
      });
      const location = appUrl(`/admin${search}`);
      window.history.pushState({ ...(window.history.state || {}), survngAdminSection: "cameras" }, "", location);
      acceptedAdminLocationRef.current = location;
      adminHistoryWriteRef.current = false;
      setAdminNavOpen(false);
      return;
    }
    setGeneralSection("detection");
    setDetectionSection(setting.detectionSection || "motion");
    setSettingsTab("general");
    const search = adminWorkspaceSearch("general", window.location.search, { subsection: "detection" });
    const location = appUrl(`/admin${search}`);
    window.history.pushState({ ...(window.history.state || {}), survngAdminSection: "general" }, "", location);
    acceptedAdminLocationRef.current = location;
    adminHistoryWriteRef.current = false;
    setAdminNavOpen(false);
  }

  function selectAdminDestination(destination) {
    const nextWorkspace = destination.workspace;
    const nextSubsection = destination.subsection || "";
    const alreadySelected = nextWorkspace === settingsTab && (
      nextWorkspace === "general"
        ? nextSubsection === "detection"
          ? false
          : nextSubsection === "motion-review"
            ? generalSection === "motion-review" && selectedId === (cameras[0]?.id || "")
            : (nextSubsection || "general") === generalSection
        : nextWorkspace === "cameras"
          ? cameraSection === "settings" && selectedId === (cameras[0]?.id || "")
        : nextWorkspace === "telemetry"
          ? (nextSubsection === "diagnostics") === (normalizeTelemetrySection(telemetrySection) === "diagnostics")
            && (normalizeTelemetrySection(nextSubsection || "health") !== "health" || !telemetryCamera)
        : nextWorkspace === "calibration"
          ? false
          : nextWorkspace === "audit"
            ? auditCamera === (cameras[0]?.id || "") && auditCategory === "all" && auditOutcome === "all" && auditPage === 0
          : true
    );
    if (alreadySelected) {
      setAdminNavOpen(false);
      return;
    }
    if (apiTokenSecretVisible && settingsTab === "general") {
      if (!window.confirm("The new API token secret is shown only once. Leave this section and discard the displayed secret?")) return;
      setApiTokenSecretVisible(false);
    }
    if (!confirmDiscardAdminChanges("Switch sections and discard unsaved changes?")) return;
    if (nextWorkspace === "general") {
      setGeneralSection(nextSubsection || "general");
      if (nextSubsection === "detection") setGeneralViewNonce((current) => current + 1);
      if (nextSubsection === "motion-review") setSelectedId(cameras[0]?.id || "");
    }
    if (nextWorkspace === "cameras") {
      setCameraSection("settings");
      setSelectedId(cameras[0]?.id || "");
    }
    if (nextWorkspace === "telemetry") {
      const telemetryDestination = normalizeTelemetrySection(nextSubsection || "health");
      setTelemetrySection(telemetryDestination);
      if (telemetryDestination === "health") setTelemetryCamera("");
      if (telemetryDestination === "diagnostics") {
        setDiagnosticScope("system");
        setDiagnosticDuration("3600");
      }
    }
    if (nextWorkspace === "calibration") setCalibrationViewNonce((current) => current + 1);
    if (nextWorkspace === "audit") {
      setAuditCamera(cameras[0]?.id || "");
      setAuditCategory("all");
      setAuditOutcome("all");
      setAuditPage(0);
      setSelectedAuditId(null);
    }
    const options = nextWorkspace === "general"
      ? {
        subsection: nextSubsection === "general" ? "" : nextSubsection,
        camera: nextSubsection === "motion-review" ? cameras[0]?.id || "" : "",
      }
      : nextWorkspace === "cameras"
        ? { camera: cameras[0]?.id || "" }
      : nextWorkspace === "telemetry"
        ? nextSubsection === "diagnostics"
          ? { subsection: "diagnostics" }
          : telemetryLocationOptions(nextSubsection || "health", "")
        : nextWorkspace === "audit"
          ? { camera: cameras[0]?.id || "" }
        : adminLocationOptions(nextWorkspace);
    const search = adminWorkspaceSearch(nextWorkspace, window.location.search, options);
    const location = appUrl(`/admin${search}`);
    window.history.pushState({ ...(window.history.state || {}), survngAdminSection: nextWorkspace }, "", location);
    acceptedAdminLocationRef.current = location;
    adminHistoryWriteRef.current = false;
    setSettingsTab(nextWorkspace);
    setAdminNavOpen(false);
  }

  function selectAdminSubsection(nextSubsection, setter, section = settingsTab, cameraOverride = "") {
    const options = section === "general"
      ? { subsection: nextSubsection === "general" ? "" : nextSubsection }
      : section === "cameras"
        ? { subsection: nextSubsection === "settings" ? "" : nextSubsection, camera: cameraOverride || selectedId }
        : section === "telemetry"
          ? telemetryLocationOptions(nextSubsection, cameraOverride || telemetryCamera)
          : { camera: cameraOverride || telemetryCamera };
    const search = adminWorkspaceSearch(section, window.location.search, options);
    const location = appUrl(`/admin${search}`);
    window.history.pushState({ ...(window.history.state || {}), survngAdminSubsection: nextSubsection }, "", location);
    acceptedAdminLocationRef.current = location;
    adminHistoryWriteRef.current = false;
    setter(nextSubsection);
  }

  async function load() {
    const sequence = ++configLoadSequence.current;
    setConfigLoadError("");
    try {
      const response = await fetch("/api/config");
      if (!response.ok) throw new Error(`Configuration failed to load (${response.status})`);
      const nextConfig = await response.json();
      if (sequence !== configLoadSequence.current) return false;
      baselineConfigRef.current = structuredClone(nextConfig);
      setBaselineRevision((current) => current + 1);
      setConfig(nextConfig);
      setSelectedId((current) => nextConfig.cameras?.some((camera) => camera.id === current) ? current : nextConfig.cameras?.[0]?.id || "");

      // These values enrich individual cards but are not required to render
      // editable configuration. Load them independently so a slow storage or
      // hardware status probe cannot strand the entire Admin page.
      const optionalPayload = async (path, timeoutMs = 5000) => {
        const controller = new AbortController();
        const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
        try {
          const optionalResponse = await fetch(path, { signal: controller.signal });
          if (!optionalResponse.ok) return null;
          return await optionalResponse.json();
        } catch {
          return null;
        } finally {
          window.clearTimeout(timeout);
        }
      };
      void Promise.all([
        optionalPayload("/api/cameras"),
        optionalPayload("/api/accelerator"),
        optionalPayload("/api/detector/models"),
        optionalPayload("/api/recordings/cache/status"),
        optionalPayload("/api/system/status"),
        optionalPayload("/api/motion/pipeline/catalog"),
        optionalPayload("/api/retention/status"),
      ]).then(([status, acceleratorPayload, models, cache, system, catalog, retention]) => {
        if (sequence !== configLoadSequence.current) return;
        if (Array.isArray(status)) setRuntimeStatus(status);
        if (acceleratorPayload) setAccelerator(acceleratorPayload);
        if (models) setDetectorModels(models.models || []);
        if (cache) setRecordingCache(cache);
        if (system) {
          setMqttStatus(system.mqtt || null);
          setDetectorStatus(system.detector || null);
        }
        if (catalog) setMotionCatalog(catalog);
        if (retention) setRetentionStatus(retention);
      });
      return true;
    } catch (error) {
      if (sequence === configLoadSequence.current) setConfigLoadError(error.message || "Configuration failed to load");
      return false;
    }
  }

  useEffect(() => {
    void load();
    return () => { configLoadSequence.current += 1; };
  }, []);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get("section") !== "audit") return;
    setSettingsTab("audit");
    const auditId = Number(params.get("audit_id"));
    if (!Number.isInteger(auditId) || auditId <= 0) return;
    let active = true;
    fetch(`/api/motion-audit/${auditId}`)
      .then((response) => {
        if (!response.ok) throw new Error(`Motion audit failed to load (${response.status})`);
        return response.json();
      })
      .then((item) => {
        if (!active) return;
        setLinkedAudit(item);
        setSelectedAuditId(item.id);
      })
      .catch((error) => {
        if (active) setAuditError(error.message || "Unable to open the selected motion audit.");
      });
    return () => { active = false; };
  }, [setSettingsTab]);

  async function loadRetention() {
    try {
      const response = await fetch("/api/retention/status");
      if (!response.ok) throw new Error(`Retention status failed (${response.status})`);
      setRetentionStatus(await response.json());
      setRetentionError("");
    } catch (error) {
      setRetentionError(error.message || "Unable to load retention status.");
    }
  }

  async function runRetention(apply = false) {
    if (apply && !window.confirm("Apply the current retention plan? This deletes eligible continuous recordings and incident snapshots older than their configured retention. Pinned face references, databases, motion-audit evidence, and the newest five minutes of recordings remain protected.")) return;
    setRetentionError("");
    try {
      const response = await fetch("/api/retention/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ apply }),
      });
      if (!response.ok) throw new Error(`Retention run failed (${response.status})`);
      setRetentionStatus(await response.json());
    } catch (error) {
      setRetentionError(error.message || "Unable to start retention.");
    }
  }

  useVisiblePolling(loadRetention, 5000, settingsTab === "general" && generalSection === "storage");

  async function refreshCameraRuntime() {
    try {
      const response = await fetch("/api/cameras");
      if (response.ok) setRuntimeStatus(await response.json());
    } catch {
      // Keep the last known runtime state and retry on the next interval.
    }
  }

  useVisiblePolling(refreshCameraRuntime, 5000, settingsTab === "cameras");


  async function loadLogs() {
    if (document.hidden) return;
    const generation = logRequestRef.current.generation + 1;
    logRequestRef.current.controller?.abort();
    const controller = new AbortController();
    logRequestRef.current = { generation, controller };
    try {
      const params = new URLSearchParams({ limit: "500", level: debouncedLogQuery.level, q: debouncedLogQuery.filter });
      const response = await fetch(`/api/logs?${params.toString()}`, { signal: controller.signal });
      if (response.ok && logRequestRef.current.generation === generation) {
        const payload = await response.json();
        if (logRequestRef.current.generation !== generation) return;
        const lines = payload.lines || [];
        const signature = logPayloadSignature(lines);
        if (signature !== logSignatureRef.current) {
          logSignatureRef.current = signature;
          setLogLines(lines);
        }
      }
    } catch (error) {
      if (error?.name === "AbortError") return;
      // Preserve the current log view; polling retries automatically.
    }
  }

  useEffect(() => {
    const timer = window.setTimeout(() => {
      logSignatureRef.current = "";
      setDebouncedLogQuery({ level: logLevel, filter: logFilter });
    }, 250);
    return () => {
      window.clearTimeout(timer);
    };
  }, [logLevel, logFilter]);
  useEffect(() => () => logRequestRef.current.controller?.abort(), []);
  useVisiblePolling(
    loadLogs,
    2000,
    settingsTab === "logs",
    { restartKey: `${debouncedLogQuery.level}\u0000${debouncedLogQuery.filter}` },
  );

  async function loadMotionAudit(page = auditPage) {
    setAuditLoading(true);
    setAuditError("");
    try {
      const params = new URLSearchParams({
        limit: String(auditPageSize),
        offset: String(page * auditPageSize),
        outcome: auditOutcome,
        category: auditCategory,
      });
      if (auditCamera) params.set("camera_id", auditCamera);
      const response = await fetch(`/api/motion-audit?${params.toString()}`);
      if (!response.ok) throw new Error(await response.text());
      const payload = await response.json();
      setAuditItems(payload.items || []);
      setAuditTotal(Number(payload.total) || 0);
    } catch (error) {
      setAuditError(error.message || "Unable to load motion audit entries.");
    } finally {
      setAuditLoading(false);
    }
  }

  useVisiblePolling(
    () => loadMotionAudit(0),
    10000,
    settingsTab === "audit" && auditPage === 0,
    { restartKey: `${auditCamera}\u0000${auditCategory}\u0000${auditOutcome}` },
  );
  useEffect(() => {
    if (settingsTab === "audit" && auditPage !== 0) void loadMotionAudit(auditPage);
  }, [settingsTab, auditPage, auditCamera, auditCategory, auditOutcome]);

  async function loadTelemetry() {
    setTelemetryLoading(true);
    setTelemetryError("");
    try {
      const params = new URLSearchParams({ hours: "24" });
      // An empty scope means the All cameras view. Do not substitute the first
      // camera here: system histories are intentionally unavailable for a
      // camera-scoped telemetry request.
      const cameraId = telemetryCamera;
      if (telemetrySection !== "diagnostics" && cameraId) params.set("camera_id", cameraId);
      const response = await fetch(`/api/telemetry?${params.toString()}`);
      if (!response.ok) throw new Error(`Telemetry failed to load (${response.status})`);
      setTelemetry(await response.json());
    } catch (error) {
      setTelemetryError(error.message || "Unable to load telemetry.");
    } finally {
      setTelemetryLoading(false);
    }
  }

  async function startTelemetryDiagnostics() {
    const cameraId = telemetryCamera || config?.cameras?.[0]?.id || "";
    const response = await fetch("/api/telemetry/diagnostics", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scope: diagnosticScope,
        camera_id: diagnosticScope === "camera" ? cameraId : "",
        duration_seconds: Number(diagnosticDuration),
      }),
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      setTelemetryError(payload.detail || `Unable to start diagnostics (${response.status})`);
      return;
    }
    await loadTelemetry();
  }

  async function stopTelemetryDiagnostics(sessionId) {
    const response = await fetch(`/api/telemetry/diagnostics/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
    if (!response.ok) {
      setTelemetryError(`Unable to stop diagnostics (${response.status})`);
      return;
    }
    await loadTelemetry();
  }

  useVisiblePolling(
    loadTelemetry,
    10000,
    settingsTab === "telemetry",
    { restartKey: `${telemetrySection}\u0000${telemetryCamera}` },
  );

  async function loadMaintenance() {
    try {
      const response = await fetch("/api/maintenance/storage");
      if (!response.ok) throw new Error(`Maintenance status failed to load (${response.status})`);
      setMaintenance(await response.json());
      setMaintenanceError("");
    } catch (error) {
      setMaintenanceError(error.message || "Unable to load maintenance status.");
    }
  }

  async function startMaintenance(apply = false, full = false) {
    if (full && !apply && !window.confirm("A full scan walks the entire NFS media library and may take a long time. You can cancel it at any point. Continue?")) return;
    if (apply && !window.confirm(`Repair the ${full ? "full" : "recent"} database findings now? Incident history and media files will not be deleted.`)) return;
    setMaintenanceError("");
    try {
      const response = await fetch("/api/maintenance/storage", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ apply, full }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || `Maintenance could not start (${response.status})`);
      }
      setMaintenance(await response.json());
    } catch (error) {
      setMaintenanceError(error.message || "Unable to start storage maintenance.");
    }
  }

  async function cancelMaintenance() {
    try {
      const response = await fetch("/api/maintenance/storage", { method: "DELETE" });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || `Maintenance could not be cancelled (${response.status})`);
      }
      setMaintenance(await response.json());
    } catch (error) {
      setMaintenanceError(error.message || "Unable to cancel storage maintenance.");
    }
  }

  useVisiblePolling(
    loadMaintenance,
    ["running", "cancelling"].includes(maintenance?.status) ? 1000 : 5000,
    settingsTab === "maintenance",
  );

  const cameras = config?.cameras || [];
  const selectedTelemetryCamera = cameras.some((camera) => camera.id === telemetryCamera)
    ? telemetryCamera
    : cameras[0]?.id || "";
  const selectedCamera = cameras.find((camera) => camera.id === selectedId) || cameras[0] || null;
  const baselineConfig = baselineConfigRef.current;
  const baselineCamera = baselineConfig?.cameras?.find((camera) => camera.id === selectedCamera?.id) || null;
  const generalDirty = Boolean(config && baselineConfig) && !configValuesEqual(comparableSystemConfig(config), comparableSystemConfig(baselineConfig));
  const selectedCameraSettingsDirty = Boolean(selectedCamera) && !configValuesEqual(comparableCameraSettings(selectedCamera), comparableCameraSettings(baselineCamera));
  const selectedZonesDirty = Boolean(selectedCamera) && !configValuesEqual(selectedCamera.zones || [], baselineCamera?.zones || []);
  const cameraDirtyState = cameraConfigDirtyState(config?.cameras || [], baselineConfig?.cameras || []);
  const perCameraDirty = perCameraDirtyState(config?.cameras || [], baselineConfig?.cameras || []);
  const dirtyCamerasCount = dirtyCameraCount(perCameraDirty);
  const runtimeStatusById = useMemo(
    () => new Map(runtimeStatus.map((item) => [item.id, item])),
    [runtimeStatus],
  );
  const cameraSettingsDirty = Boolean(config && baselineConfig) && cameraDirtyState.settings;
  const zonesDirty = Boolean(config && baselineConfig) && cameraDirtyState.zones;
  const cameraOrderDirty = Boolean(config && baselineConfig) && !configValuesEqual(
    (config.cameras || []).map((camera) => camera.id),
    (baselineConfig.cameras || []).map((camera) => camera.id),
  );
  const adminDirty = generalDirty || cameraSettingsDirty || zonesDirty || cameraOrderDirty;
  const activeAdminDestination = adminDestination(settingsTab, { generalSection, telemetrySection });
  const currentAdminDirty = settingsTab === "general"
    ? generalDirty
    : settingsTab === "cameras"
      ? cameraSettingsDirty || zonesDirty || cameraOrderDirty
      : false;
  const currentAdminSaving = generalSaving || cameraSaving || zonesSaving || cameraOrderSaving;
  const adminSaveStatusTitle = currentAdminSaving
    ? "Saving changes"
    : currentAdminDirty
      ? settingsTab === "cameras" && dirtyCamerasCount > 1
        ? `${dirtyCamerasCount} cameras with unsaved changes`
        : "Unsaved changes"
      : saveNotice?.state === "error"
        ? "Save failed"
        : "Changes saved";
  const adminSaveAvailable = settingsTab === "general" || settingsTab === "cameras";
  const adminSaveImpact = settingsTab === "cameras"
    ? dirtyCamerasCount > 1
      ? "Saving will apply changes to each edited camera independently."
      : selectedCameraSettingsDirty
        ? "Camera settings save independently; only structural changes reload the affected camera worker."
        : selectedZonesDirty
          ? "Zone changes apply without restarting camera workers."
          : cameraOrderDirty
            ? "Camera order changes the interface only and does not restart camera workers."
            : "Camera changes use scoped saves and avoid unnecessary worker restarts."
    : "Settings apply selectively; the confirmation will identify any subsystem or camera worker that reloaded.";
  adminDirtyRef.current = adminDirty;
  apiTokenSecretVisibleRef.current = apiTokenSecretVisible;
  const selectedRuntimeStatus = runtimeStatus.find((camera) => camera.id === selectedCamera?.id);
  const selectedAudit = auditItems.find((item) => item.id === selectedAuditId)
    || (linkedAudit?.id === selectedAuditId ? linkedAudit : null);
  const selectedAuditItems = selectedAudit && !auditItems.some((item) => item.id === selectedAudit.id)
    ? [selectedAudit, ...auditItems]
    : auditItems;
  const activeDetectorPath = config?.detector?.model_path || config?.detector?.model_xml || "";
  const activeDetectorModel = findDetectorModel(detectorModels, activeDetectorPath);
  const zoneClassOptions = activeDetectorModel?.classes?.length
    ? activeDetectorModel.classes
    : config?.detector?.labels || [];

  useEffect(() => {
    function warnBeforeUnload(event) {
      if (!adminDirty && !apiTokenSecretVisible) return;
      event.preventDefault();
      event.returnValue = "";
    }
    window.addEventListener("beforeunload", warnBeforeUnload);
    return () => window.removeEventListener("beforeunload", warnBeforeUnload);
  }, [adminDirty, apiTokenSecretVisible]);



  function discardAdminChanges() {
    if (!baselineConfigRef.current) return;
    setConfig(structuredClone(baselineConfigRef.current));
    setSelectedId((current) => baselineConfigRef.current.cameras?.some((camera) => camera.id === current)
      ? current
      : baselineConfigRef.current.cameras?.[0]?.id || "");
    setCameraOrderEditing(false);
    setSaveNotice({ state: "saved", text: "Unsaved changes discarded." });
  }

  function confirmDiscardAdminChanges(message) {
    if (!adminDirty) return true;
    if (!window.confirm(message)) return false;
    discardAdminChanges();
    return true;
  }

  function moveTabFocus(event, ids, selected, select) {
    const next = nextTabId(ids, selected, event.key);
    if (!next) return;
    event.preventDefault();
    const tabListId = event.currentTarget.id;
    select(next);
    window.requestAnimationFrame(() => document.getElementById(tabListId)?.querySelector(`[data-tab-id="${next}"]`)?.focus());
  }

  function selectConfigCamera(cameraId) {
    if (cameraId === selectedCamera?.id) return;
    setSelectedId(cameraId);
    setProbe(null);
    const search = adminWorkspaceSearch("cameras", window.location.search, { subsection: cameraSection === "settings" ? "" : cameraSection, camera: cameraId });
    const location = appUrl(`/admin${search}`);
    window.history.pushState({ ...(window.history.state || {}), survngAdminCamera: cameraId }, "", location);
    acceptedAdminLocationRef.current = location;
    adminHistoryWriteRef.current = false;
  }

  function selectAuditCamera(nextCameraId) {
    setAuditCamera(nextCameraId);
    setAuditPage(0);
    browserStorage(window)?.setItem("survng.motionAuditCamera.v1", nextCameraId);
  }

  function selectTelemetryCamera(nextCameraId) {
    setTelemetryCamera(nextCameraId);
    if (telemetrySection === "diagnostics") setTelemetrySection("health");
    const search = adminWorkspaceSearch("telemetry", window.location.search, telemetryLocationOptions(telemetrySection === "diagnostics" ? "health" : telemetrySection, nextCameraId));
    const location = appUrl(`/admin${search}`);
    window.history.pushState({ ...(window.history.state || {}), survngTelemetryCamera: nextCameraId }, "", location);
    acceptedAdminLocationRef.current = location;
    adminHistoryWriteRef.current = false;
  }

  function updateConfig(path, value) {
    setSaveNotice(null);
    setConfig((current) => {
      const next = structuredClone(current);
      let target = next;
      for (let index = 0; index < path.length - 1; index += 1) target = target[path[index]];
      target[path[path.length - 1]] = value;
      return next;
    });
  }

  function commitImmediateConfig(path, value) {
    const applyValue = (targetConfig) => {
      const next = structuredClone(targetConfig);
      let target = next;
      for (let index = 0; index < path.length - 1; index += 1) target = target[path[index]];
      target[path[path.length - 1]] = value;
      return next;
    };
    setConfig((current) => applyValue(current));
    if (baselineConfigRef.current) {
      baselineConfigRef.current = applyValue(baselineConfigRef.current);
      setBaselineRevision((current) => current + 1);
    }
  }

  function updateCamera(cameraId, path, value) {
    setSaveNotice(null);
    setConfig((current) => {
      const next = structuredClone(current);
      const camera = next.cameras.find((item) => item.id === cameraId);
      let target = camera;
      for (let index = 0; index < path.length - 1; index += 1) {
        const key = path[index];
        if (!target[key] || typeof target[key] !== "object") target[key] = {};
        target = target[key];
      }
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

  function startCameraOrderEdit() {
    cameraOrderOriginalRef.current = cameras.map((camera) => camera.id);
    setCameraOrderEditing(true);
    setSaveNotice(null);
  }

  function cancelCameraOrderEdit() {
    const originalOrder = cameraOrderOriginalRef.current;
    setConfig((current) => {
      const cameraById = new Map((current.cameras || []).map((camera) => [camera.id, camera]));
      const ordered = originalOrder.map((cameraId) => cameraById.get(cameraId)).filter(Boolean);
      const seen = new Set(ordered.map((camera) => camera.id));
      return { ...current, cameras: [...ordered, ...(current.cameras || []).filter((camera) => !seen.has(camera.id))] };
    });
    setCameraOrderEditing(false);
    setDragConfigCameraId("");
    setDragConfigCameraTarget("");
    setSaveNotice(null);
  }

  function moveConfigCamera(sourceId, targetId, after) {
    if (!sourceId || !targetId || sourceId === targetId) return;
    setConfig((current) => {
      const nextCameras = [...(current.cameras || [])];
      const sourceIndex = nextCameras.findIndex((camera) => camera.id === sourceId);
      if (sourceIndex < 0) return current;
      const [source] = nextCameras.splice(sourceIndex, 1);
      const targetIndex = nextCameras.findIndex((camera) => camera.id === targetId);
      if (targetIndex < 0) return current;
      nextCameras.splice(targetIndex + (after ? 1 : 0), 0, source);
      return { ...current, cameras: nextCameras };
    });
  }

  function moveSelectedCameraBy(offset) {
    const currentIndex = cameras.findIndex((camera) => camera.id === selectedCamera?.id);
    const targetIndex = currentIndex + Number(offset || 0);
    if (currentIndex < 0 || targetIndex < 0 || targetIndex >= cameras.length) return;
    moveConfigCamera(selectedCamera.id, cameras[targetIndex].id, offset > 0);
    setSaveNotice({ state: "pending", text: `${selectedCamera.name || selectedCamera.id} moved to position ${targetIndex + 1}. Save order to apply.` });
  }

  async function saveCameraOrder() {
    if (cameraOrderSaving) return false;
    setCameraOrderSaving(true);
    setSaveNotice({ state: "saving", text: "Saving default camera order..." });
    try {
      const response = await fetch("/api/config/cameras/order", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(cameras.map((camera) => camera.id)),
      });
      if (!response.ok) throw new Error(await response.text());
      cameraOrderOriginalRef.current = cameras.map((camera) => camera.id);
      if (baselineConfigRef.current) {
        const byId = new Map((baselineConfigRef.current.cameras || []).map((camera) => [camera.id, camera]));
        baselineConfigRef.current.cameras = cameras.map((camera) => structuredClone(byId.get(camera.id) || camera));
        setBaselineRevision((current) => current + 1);
      }
      setCameraOrderEditing(false);
      setSaveNotice({ state: "saved", text: "Default live-view order saved." });
      return true;
    } catch (error) {
      setSaveNotice({ state: "error", text: error.message || "Unable to save camera order." });
      return false;
    } finally {
      setCameraOrderSaving(false);
      setDragConfigCameraId("");
      setDragConfigCameraTarget("");
    }
  }

  async function save() {
    if (generalSaving) return false;
    const ids = new Set();
    const configToSave = {
      ...config,
      cameras: camerasWithGeneratedIds(config.cameras || []),
    };
    for (const camera of configToSave.cameras || []) {
      if (ids.has(camera.id)) {
        setSaveNotice({ state: "error", text: `Duplicate camera ID "${camera.id}". Fix duplicates before saving.` });
        return false;
      }
      ids.add(camera.id);
    }
    const mediaStorageError = mediaStorageConfigurationError(configToSave.media_storage);
    if (mediaStorageError) {
      setSaveNotice({ state: "error", text: mediaStorageError });
      return false;
    }
    setGeneralSaving(true);
    setSaveNotice({ state: "saving", text: "Saving settings..." });
    try {
      const response = await fetch("/api/config", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(configToSave),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        const detail = typeof payload.detail === "string"
          ? payload.detail
          : `Configuration could not be saved (${response.status}).`;
        throw new Error(detail);
      }
      const payload = await response.json();
      const reloaded = await load();
      setSaveNotice(reloaded
        ? {
          state: "saved",
          text: payload.camera_workers_restarted
            ? payload.camera_ids_restarted?.length === 1
              ? `Saved. ${payload.camera_ids_restarted[0]} motion runtime reloaded; other cameras kept running.`
              : `Saved. ${payload.camera_ids_restarted?.length || "Affected"} camera motion runtimes reloaded.`
            : payload.subsystems_restarted?.includes("recorders") && payload.subsystems_restarted?.includes("mqtt")
              ? "Saved. Recorders restarted and MQTT reconnected; cameras kept running."
              : payload.subsystems_restarted?.includes("recorders")
                ? "Saved. Recorder processes restarted; cameras kept running."
                : payload.subsystems_restarted?.includes("mqtt")
                  ? "Saved. MQTT reconnected; cameras kept running."
                  : payload.subsystems_restarted?.some((name) => (
                    name === "tracking_sessions" || name.endsWith("_inference")
                  ))
                    ? "Saved. Detection services refreshed; camera streams kept running."
                    : "Saved without interrupting cameras.",
        }
        : { state: "error", text: "Saved, but the refreshed configuration could not be loaded. Retry this page." });
      return reloaded;
    } catch (error) {
      setSaveNotice({ state: "error", text: error.message || "Unable to save general settings." });
      return false;
    } finally {
      setGeneralSaving(false);
    }
  }

  async function saveZones(camera) {
    if (!camera || zonesSaving) return false;
    setZonesSaving(true);
    setSaveNotice({ state: "saving", text: "Saving zones..." });
    try {
      const response = await fetch(`/api/config/cameras/${encodeURIComponent(camera.id)}/zones`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(camera.zones || []),
      });
      if (!response.ok) throw new Error(await response.text());
      const payload = await response.json();
      updateCamera(camera.id, ["zones"], payload.zones || []);
      const baselineCameraToUpdate = baselineConfigRef.current?.cameras?.find((item) => item.id === camera.id);
      if (baselineCameraToUpdate) {
        baselineCameraToUpdate.zones = structuredClone(payload.zones || []);
        setBaselineRevision((current) => current + 1);
      }
      setSaveNotice({ state: "saved", text: "Zones saved without restarting cameras." });
      const statusResponse = await fetch("/api/cameras");
      if (statusResponse.ok) setRuntimeStatus(await statusResponse.json());
      return true;
    } catch (error) {
      setSaveNotice({ state: "error", text: error.message || "Unable to save zones." });
      return false;
    } finally {
      setZonesSaving(false);
    }
  }

  async function saveCamera(camera) {
    if (!camera || cameraSaving) return false;
    setCameraSaving(true);
    setSaveNotice({ state: "saving", text: "Saving camera settings..." });
    try {
      const response = await fetch(`/api/config/cameras/${encodeURIComponent(camera.id)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(camera),
      });
      if (!response.ok) throw new Error(await response.text());
      const payload = await response.json();
      const savedCamera = payload.camera;
      setConfig((current) => ({
        ...current,
        cameras: current.cameras.map((item) => item.id === camera.id
          ? { ...savedCamera, zones: item.zones || [] }
          : item),
      }));
      setSelectedId(savedCamera.id);
      if (baselineConfigRef.current) {
        const baselineCameras = baselineConfigRef.current.cameras || [];
        const baselineMatch = baselineCameras.find((item) => item.id === camera.id);
        const savedBaselineCamera = {
          ...structuredClone(savedCamera),
          zones: structuredClone(baselineMatch?.zones || camera.zones || []),
        };
        baselineConfigRef.current.cameras = baselineMatch
          ? baselineCameras.map((item) => item.id === camera.id ? savedBaselineCamera : item)
          : [...baselineCameras, savedBaselineCamera];
        setBaselineRevision((current) => current + 1);
      }
      setSaveNotice({
        state: "saved",
        text: payload.camera_workers_restarted
          ? `${savedCamera.name || savedCamera.id} motion runtime reloaded; other cameras kept running.`
          : "Camera settings saved without interrupting camera streams.",
      });
      const statusResponse = await fetch("/api/cameras");
      if (statusResponse.ok) setRuntimeStatus(await statusResponse.json());
      return true;
    } catch (error) {
      setSaveNotice({ state: "error", text: error.message || "Unable to save camera settings." });
      return false;
    } finally {
      setCameraSaving(false);
    }
  }

  async function saveCurrentAdminChanges() {
    if (settingsTab === "general") {
      if (generalDirty) await save();
      return;
    }
    if (settingsTab !== "cameras") return;
    const dirtyMap = perCameraDirtyState(config?.cameras || [], baselineConfigRef.current?.cameras || []);
    for (const camera of config?.cameras || []) {
      const dirty = dirtyMap[camera.id];
      if (!dirty) continue;
      if (dirty.settings && !await saveCamera(camera)) return;
      if (dirty.zones && !await saveZones(camera)) return;
    }
    if (cameraOrderDirty) await saveCameraOrder();
  }

  async function deleteCamera(camera) {
    if (!camera || cameraSaving || !window.confirm(`Remove ${camera.name || camera.id}?`)) return;
    const isPersisted = runtimeStatus.some((item) => item.id === camera.id);
    if (!isPersisted) {
      removeCamera(camera.id);
      setSaveNotice({ state: "saved", text: "Unsaved camera removed." });
      return;
    }
    setCameraSaving(true);
    setSaveNotice({ state: "saving", text: "Removing camera..." });
    try {
      const response = await fetch(`/api/config/cameras/${encodeURIComponent(camera.id)}`, { method: "DELETE" });
      if (!response.ok) throw new Error(await response.text());
      removeCamera(camera.id);
      if (baselineConfigRef.current) {
        baselineConfigRef.current.cameras = (baselineConfigRef.current.cameras || []).filter((item) => item.id !== camera.id);
        setBaselineRevision((current) => current + 1);
      }
      setSaveNotice({ state: "saved", text: "Camera removed. Workers reloaded." });
      const statusResponse = await fetch("/api/cameras");
      if (statusResponse.ok) setRuntimeStatus(await statusResponse.json());
    } catch (error) {
      setSaveNotice({ state: "error", text: error.message || "Unable to remove camera." });
    } finally {
      setCameraSaving(false);
    }
  }

  async function probeCamera(camera) {
    setProbe({ loading: true });
    setCameraSection("info");
    const probeCameraConfig = cameraWithDerivedConnection(camera);
    if (probeCameraConfig !== camera) {
      setConfig((current) => ({
        ...current,
        cameras: (current.cameras || []).map((item) => item.id === camera.id ? probeCameraConfig : item),
      }));
    }
    const host = probeCameraConfig.onvif?.host || "";
    const username = probeCameraConfig.onvif?.username || "";
    const password = probeCameraConfig.onvif?.password || "";
    try {
      const response = await fetch("/api/config/probe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          camera_id: probeCameraConfig.id,
          host,
          username,
          password,
          onvif_port: probeCameraConfig.onvif?.port || 8000,
        }),
      });
      if (!response.ok) throw new Error(`Camera probe failed (${response.status})`);
      const result = await response.json();
      setProbe(result);
      if (result.onvif?.reachable) updateCamera(camera.id, ["onvif", "enabled"], true);
    } catch (error) {
      setProbe({ loading: false, error: error.message || "Camera probe failed" });
    }
  }

  const auditCategoryTitle = auditCategory === "visual_backup"
    ? "Visual Backup"
    : auditCategory === "active_followup"
      ? "Active-Event Follow-Up"
      : auditCategory === "qualification"
        ? "Filtered Motion"
        : "Motion Decisions";
  const generalDestinationIcon = ADMIN_DESTINATION_ICONS[activeAdminDestination.id] || Cog;
  const adminCommandBar = useMemo(() => {
    if (settingsTab === "cameras") {
      return {
        scope: (
          <CameraScopePicker
            className="section-title-picker"
            cameras={cameras}
            runtimeStatus={runtimeStatus}
            value={selectedCamera?.id || ""}
            onChange={selectConfigCamera}
            ariaLabel="Configure camera"
          />
        ),
        actions: (
          <div className="camera-command-bar">
            {cameraOrderEditing ? (
              <>
                <button type="button" onClick={() => moveSelectedCameraBy(-1)} disabled={cameraOrderSaving || cameras.findIndex((camera) => camera.id === selectedCamera?.id) <= 0}>Up</button>
                <button type="button" onClick={() => moveSelectedCameraBy(1)} disabled={cameraOrderSaving || cameras.findIndex((camera) => camera.id === selectedCamera?.id) >= cameras.length - 1}>Down</button>
                <button type="button" onClick={cancelCameraOrderEdit} disabled={cameraOrderSaving}>Done</button>
              </>
            ) : (
              <>
                <button type="button" onClick={startCameraOrderEdit}><GripVertical size={16} /> Edit Order</button>
                <button type="button" onClick={() => addCamera()}><Plus size={16} /> Add</button>
              </>
            )}
            {selectedCamera ? <>
              <button type="button" onClick={() => cloneCamera(selectedCamera)} disabled={cameraSaving}><Copy size={16} /> Clone</button>
              <button type="button" onClick={() => probeCamera(selectedCamera)} disabled={cameraSaving}><Radar size={16} /> Auto-detect</button>
              <button type="button" className="danger" onClick={() => deleteCamera(selectedCamera)} disabled={cameraSaving}><Trash2 size={16} /> Remove</button>
            </> : null}
          </div>
        ),
      };
    }
    if (settingsTab === "telemetry") {
      if (telemetrySection === "diagnostics") {
        return {
          scope: <AdminCommandLabel icon={Wrench}>Diagnostics</AdminCommandLabel>,
          actions: <button type="button" onClick={() => void loadTelemetry()} disabled={telemetryLoading}><RefreshCcw className={telemetryLoading ? "spin" : ""} size={16} /> Refresh</button>,
        };
      }
      return {
        scope: (
          <CameraScopePicker
            className="section-title-picker"
            cameras={cameras}
            runtimeStatus={runtimeStatus}
            value={telemetryCamera}
            onChange={selectTelemetryCamera}
            allOption={{ value: "", label: "All cameras" }}
            ariaLabel="Health camera scope"
          />
        ),
        meta: <TelemetryContinuity data={telemetry} />,
        actions: <button type="button" onClick={() => void loadTelemetry()} disabled={telemetryLoading}><RefreshCcw className={telemetryLoading ? "spin" : ""} size={16} /> Refresh</button>,
      };
    }
    if (settingsTab === "general") {
      if (generalSection === "motion-review") {
        return {
          scope: (
            <CameraScopePicker
              className="section-title-picker"
              cameras={cameras}
              runtimeStatus={runtimeStatus}
              value={selectedId}
              onChange={setSelectedId}
              ariaLabel="Camera Advisor camera"
            />
          ),
          meta: <span className="admin-action-kind">Advisor actions apply immediately</span>,
        };
      }
      return {
        scope: <AdminCommandLabel icon={generalDestinationIcon}>{GENERAL_SECTION_LABELS[generalSection] || "Server"}</AdminCommandLabel>,
      };
    }
    if (settingsTab === "audit") {
      return {
        scope: (
          <>
            <CameraScopePicker className="section-title-picker" cameras={cameras} runtimeStatus={runtimeStatus} value={auditCamera} onChange={selectAuditCamera} allOption={{ value: "", label: "All cameras" }} ariaLabel="Motion Audit camera" />
            <label className="admin-command-filter">Category<select value={auditCategory} onChange={(event) => { setAuditCategory(event.target.value); setAuditPage(0); }}><option value="all">All categories</option><option value="visual_backup">Visual backup</option><option value="active_followup">Active-event follow-up</option><option value="qualification">Filtered motion</option></select></label>
            <label className="admin-command-filter">Outcome<select value={auditOutcome} onChange={(event) => { setAuditOutcome(event.target.value); setAuditPage(0); }}><option value="all">All outcomes</option><option value="object">Object found</option><option value="clear">No object</option><option value="not_run">Detection skipped</option></select></label>
          </>
        ),
        meta: <AdminCommandLabel>{auditCategoryTitle}</AdminCommandLabel>,
        actions: <button type="button" onClick={() => loadMotionAudit(auditPage)} disabled={auditLoading}><RefreshCcw className={auditLoading ? "spin" : ""} size={16} /> Refresh</button>,
      };
    }
    if (settingsTab === "maintenance") {
      return {
        scope: <AdminCommandLabel icon={Wrench}>Storage Reconciliation</AdminCommandLabel>,
        actions: (
          <div className="maintenance-actions">
            {["running", "cancelling"].includes(maintenance?.status) ? <button type="button" onClick={() => void cancelMaintenance()} disabled={maintenance?.status === "cancelling"}><X size={16} /> {maintenance?.status === "cancelling" ? "Cancelling" : "Cancel"}</button> : <><button type="button" onClick={() => void startMaintenance(false, false)}><RefreshCcw size={16} /> Quick Check</button><button type="button" onClick={() => void startMaintenance(false, true)}><Search size={16} /> Full Scan</button><button type="button" className="primary" onClick={() => void startMaintenance(true, maintenance?.result?.full === true)}><Wrench size={16} /> {maintenance?.result?.full === true ? "Repair Full Findings" : "Repair Recent Findings"}</button></>}
          </div>
        ),
      };
    }
    if (settingsTab === "logs") {
      return {
        scope: <AdminCommandLabel icon={ListTree}>Logs</AdminCommandLabel>,
        actions: (
          <div className="log-command-controls">
            <label>Minimum severity<select value={logLevel} onChange={(event) => setLogLevel(event.target.value)}>{[["DEBUG", "Debug+"], ["INFO", "Info+"], ["WARNING", "Warning+"], ["ERROR", "Error+"], ["CRITICAL", "Critical"]].map(([value, label]) => <option value={value} key={value}>{label}</option>)}</select></label>
            <button type="button" onClick={loadLogs}><RefreshCcw size={16} /> Refresh</button>
          </div>
        ),
      };
    }
    if (settingsTab === "calibration" && calibrationCommandBar) {
      return {
        scope: <AdminCommandLabel icon={Sparkles}>{calibrationCommandBar.sectionLabel}</AdminCommandLabel>,
        actions: <button type="button" onClick={() => calibrationCommandBar.refresh()}><RefreshCcw size={16} /> Refresh</button>,
      };
    }
    return null;
  }, [
    auditCamera,
    auditCategoryTitle,
    auditLoading,
    auditPage,
    calibrationCommandBar,
    cameraOrderEditing,
    cameraOrderSaving,
    cameraSaving,
    cameras,
    generalDestinationIcon,
    generalSection,
    loadLogs,
    loadMotionAudit,
    loadTelemetry,
    maintenance,
    runtimeStatus,
    selectedCamera,
    selectedId,
    settingsTab,
    telemetry,
    telemetryCamera,
    telemetryLoading,
    telemetrySection,
  ]);

  if (!config) {
    return <main className="bento-grid config-grid"><section className="bento-card config-editor"><div className="empty-state">{configLoadError || "Loading config..."}{configLoadError ? <button type="button" onClick={() => void load()}><RefreshCcw size={15} /> Retry</button> : null}</div></section></main>;
  }

  return (
    <main className="bento-grid config-grid settings-grid">
      <button type="button" className={`admin-navigation-backdrop${adminNavOpen ? " open" : ""}`} aria-label="Close Admin menu" onClick={() => setAdminNavOpen(false)} />
      <aside id="admin-navigation" className={`admin-navigation${adminNavOpen ? " open" : ""}`} aria-label="Admin navigation">
        <div className="admin-navigation-head">
          <span><Cog size={18} /><span><strong>Admin</strong><small>System administration</small></span></span>
          <button type="button" className="admin-navigation-close" aria-label="Close Admin menu" onClick={() => setAdminNavOpen(false)}><X size={18} /></button>
        </div>
        <nav>
          {ADMIN_NAV_GROUPS.map((group) => <section key={group.id} aria-labelledby={`admin-group-${group.id}`}>
            <h2 id={`admin-group-${group.id}`}>{group.label}</h2>
            {group.items.map((destination) => {
              const Icon = ADMIN_DESTINATION_ICONS[destination.id] || Cog;
              const active = activeAdminDestination.id === destination.id;
              return <button id={`admin-destination-${destination.id}`} type="button" className={`${active ? "active" : ""}${destination.secondary ? " secondary" : ""}`} aria-current={active ? "page" : undefined} onClick={() => selectAdminDestination(destination)} key={destination.id} title={destination.description}><Icon size={16} /><span>{destination.label}</span></button>;
            })}
          </section>)}
        </nav>
      </aside>
      <header className="admin-mobile-toolbar">
        <button type="button" aria-controls="admin-navigation" aria-expanded={adminNavOpen} onClick={() => setAdminNavOpen(true)}><PanelLeftOpen size={18} />Menu</button>
        <span><small>Admin</small><strong>{activeAdminDestination.label}</strong></span>
        {currentAdminDirty ? <em>Unsaved</em> : null}
      </header>
      <AdminCommandBar scope={adminCommandBar?.scope} meta={adminCommandBar?.meta} actions={adminCommandBar?.actions} />

      <div className={`admin-workspace-surface admin-workspace-${settingsTab}`}>

        {settingsTab === "home" ? (
          <section id="admin-panel-home" className="bento-card config-editor settings-panel admin-config-home" aria-labelledby="admin-home-title">
            <div className="admin-home-inner">
              <header className="admin-home-hero">
                <div><span className="admin-eyebrow">Configure</span><h1 id="admin-home-title">Make SurvNG work for your scenes</h1><p>Start with a camera, then tune intelligence, storage, and connections as your system grows.</p></div>
                <ShieldCheck size={34} aria-hidden="true" />
              </header>
              <div className="admin-home-summary" aria-label="System setup summary">
                <article><span className="admin-home-summary-icon"><Camera size={18} /></span><div><strong>{cameras.length}</strong><small>{cameras.length === 1 ? "Camera configured" : "Cameras configured"}</small></div><em className={cameras.length ? "good" : "attention"}>{cameras.length ? "Ready" : "Start here"}</em></article>
                <article><span className="admin-home-summary-icon"><ShieldCheck size={18} /></span><div><strong>{runtimeStatus.filter((item) => item.running !== false).length}/{cameras.length}</strong><small>Camera workers running</small></div><em className={cameras.length && runtimeStatus.filter((item) => item.running !== false).length === cameras.length ? "good" : "attention"}>{cameras.length && runtimeStatus.filter((item) => item.running !== false).length === cameras.length ? "Healthy" : "Review"}</em></article>
                <article><span className="admin-home-summary-icon"><Cpu size={18} /></span><div><strong>{detectorStatus?.running === false ? "Offline" : config.detector ? "Enabled" : "Not set"}</strong><small>Detection engine</small></div><em className={detectorStatus?.running === false ? "attention" : "good"}>{detectorStatus?.running === false ? "Check" : "Ready"}</em></article>
                <article><span className="admin-home-summary-icon"><HardDrive size={18} /></span><div><strong>{retentionStatus?.state ? String(retentionStatus.state).replaceAll("_", " ") : "Calculating"}</strong><small>Storage plan</small></div><em className={retentionStatus?.state === "error" ? "attention" : "good"}>{retentionStatus?.state === "error" ? "Review" : "Tracked"}</em></article>
              </div>
              <div className="admin-home-section-head"><div><h2>Choose a task</h2><p>Configuration is organized by the outcome you want, with advanced controls inside each area.</p></div></div>
              <div className="admin-home-actions">
                {adminHomeDestinations().map((destination) => { const Icon = ADMIN_DESTINATION_ICONS[destination.id] || Cog; return <button type="button" key={destination.id} onClick={() => selectAdminDestination(destination)}><Icon size={20} /><span><strong>{destination.label}</strong><small>{destination.description}</small></span><ChevronRight size={16} /></button>; })}
              </div>
              <div className="admin-home-footer"><span><CircleDot size={14} />Changes are scoped to the area you edit.</span><button type="button" onClick={() => selectAdminDestination(ADMIN_NAV_GROUPS.find((group) => group.id === "observe")?.items[0])}>View system health <ArrowRight size={15} /></button></div>
            </div>
          </section>
        ) : settingsTab === "general" ? (
          <>
            <section className="bento-card camera-tree config-tree settings-section-tree">
              <div className="tree-list">
                <span className="tree-group-label">System</span>
                <button type="button" aria-current={generalSection === "general" ? "page" : undefined} className={generalSection === "general" ? "active" : ""} onClick={() => selectAdminSubsection("general", setGeneralSection, "general")}><Cog size={16} /><span>Server</span></button>
                <button type="button" aria-current={generalSection === "storage" ? "page" : undefined} className={generalSection === "storage" ? "active" : ""} onClick={() => selectAdminSubsection("storage", setGeneralSection, "general")}><HardDrive size={16} /><span>Storage &amp; Retention</span></button>
                <button type="button" aria-current={generalSection === "mqtt" ? "page" : undefined} className={generalSection === "mqtt" ? "active" : ""} onClick={() => selectAdminSubsection("mqtt", setGeneralSection, "general")}><Radio size={16} /><span>API &amp; MQTT</span></button>
                <button type="button" aria-current={generalSection === "access" ? "page" : undefined} className={generalSection === "access" ? "active" : ""} onClick={() => selectAdminSubsection("access", setGeneralSection, "general")}><KeyRound size={16} /><span>Access</span></button>
                <span className="tree-group-label">Intelligence</span>
                <button type="button" aria-current={generalSection === "detection" ? "page" : undefined} className={generalSection === "detection" ? "active" : ""} onClick={() => selectAdminSubsection("detection", setGeneralSection, "general")}><Cpu size={16} /><span>Object Detection</span></button>
                <span className="tree-group-label">Tools</span>
                <button type="button" aria-current={generalSection === "motion-review" ? "page" : undefined} className={generalSection === "motion-review" ? "active" : ""} onClick={() => selectAdminSubsection("motion-review", setGeneralSection, "general")}><Sparkles size={16} /><span>Camera Advisor</span></button>
              </div>
            </section>
            <section id="admin-panel-general" className="bento-card config-editor settings-panel" aria-labelledby={`admin-destination-${activeAdminDestination.id}`}>
              <GeneralSettings
                key={`general-${generalSection}-${generalViewNonce}`}
                config={config}
                updateConfig={updateConfig}
                commitImmediateConfig={commitImmediateConfig}
                onTokenSecretVisibleChange={setApiTokenSecretVisible}
                onOpenApiTokens={() => selectAdminSubsection("mqtt", setGeneralSection, "general")}
                timeZone={timeZone}
                setTimeZone={setTimeZone}
                theme={theme}
                setTheme={setTheme}
                accelerator={accelerator}
                detectorModels={detectorModels}
                recordingCache={recordingCache}
                retentionStatus={retentionStatus}
                retentionError={retentionError}
                runRetention={runRetention}
                mqttStatus={mqttStatus}
                detectorStatus={detectorStatus}
                motionCatalog={motionCatalog}
                runtimeStatus={runtimeStatus}
                advisorCameraId={selectedId}
                onAdvisorCameraIdChange={setSelectedId}
                section={generalSection}
                detectionSection={detectionSection}
                onDetectionSectionChange={setDetectionSection}
              />
            </section>
          </>
        ) : settingsTab === "audit" ? (
          <section id="admin-panel-audit" className="bento-card config-editor settings-panel settings-panel-wide motion-audit-panel" aria-labelledby="admin-destination-audit">
              <MotionAuditViewer
                items={auditItems}
                total={auditTotal}
                page={auditPage}
                pageSize={auditPageSize}
                setPage={setAuditPage}
                loading={auditLoading}
                error={auditError}
                timeZone={timeZone}
                onOpen={(item) => setSelectedAuditId(item.id)}
              />
          </section>
        ) : settingsTab === "calibration" ? (
          <CalibrationLab key={`calibration-${calibrationViewNonce}`} cameras={cameras} runtimeStatus={runtimeStatus} timeZone={timeZone} onCommandBarChange={setCalibrationCommandBar} />
        ) : settingsTab === "telemetry" ? (
          <>
            <section id="admin-panel-telemetry" className="bento-card config-editor settings-panel telemetry-panel settings-panel-wide subsection-workspace" aria-labelledby={`admin-destination-${activeAdminDestination.id}`}>
              {telemetryError ? <div className="error-banner telemetry-error">{telemetryError}</div> : null}
              {telemetrySection === "diagnostics" ? <div id="telemetry-view-panel" className="telemetry-tab-panel" role="tabpanel"><div className="telemetry-diagnostics">
                <ModelsAndHardwarePanel config={config} updateConfig={updateConfig} detectorModels={detectorModels} accelerator={accelerator} />
                <section className="telemetry-section support-bundle-section">
                  <div className="telemetry-section-head"><div><h3>Support bundle</h3><p>Download one redacted system report to share when you need help troubleshooting. It includes software and runtime status, safe configuration, recent health events, diagnostics, and logs—never video, images, passwords, tokens, cookies, private keys, or camera stream URLs.</p></div><a className="button primary" href={appUrl("/api/support-bundle")} download="survng-support-bundle.json"><Download size={15} />Download support bundle</a></div>
                </section>
                <section className="telemetry-section">
                  <div className="telemetry-section-head"><div><h3>Temporary diagnostics</h3><p>Capture detailed troubleshooting data for a limited time. Sessions stop automatically and never include images, video, or credentials.</p></div></div>
                  <div className="telemetry-diagnostic-controls">
                    <label><span>Scope</span><select value={diagnosticScope} onChange={(event) => setDiagnosticScope(event.target.value)}><option value="system">Entire system</option><option value="detector">Object detector</option><option value="storage">Storage</option><option value="camera">One camera</option></select></label>
                    {diagnosticScope === "camera" ? (
                      <CameraScopePicker
                        cameras={cameras}
                        runtimeStatus={runtimeStatus}
                        value={selectedTelemetryCamera}
                        onChange={setTelemetryCamera}
                        ariaLabel="Diagnostics camera"
                      />
                    ) : null}
                    <label><span>Duration</span><select value={diagnosticDuration} onChange={(event) => setDiagnosticDuration(event.target.value)}><option value="900">15 minutes</option><option value="3600">1 hour</option><option value="21600">6 hours</option><option value="86400">24 hours</option></select></label>
                    <button type="button" className="primary" onClick={() => void startTelemetryDiagnostics()} disabled={diagnosticScope === "camera" && !selectedTelemetryCamera}>Start diagnostics</button>
                  </div>
                  {(telemetry?.diagnostics?.active || []).length ? <div className="telemetry-diagnostic-list">{telemetry.diagnostics.active.map((session) => {
                    const camera = cameras.find((item) => item.id === session.camera_id);
                    return <article className="telemetry-diagnostic-card" key={session.id}><div><strong>{session.scope === "camera" ? camera?.name || session.camera_id : String(session.scope).replaceAll("_", " ")} diagnostics</strong><span>Active until {formatDateTime(session.expires_at, timeZone)}</span></div><div className="button-row"><a className="button" href={appUrl(`/api/telemetry/diagnostics/${encodeURIComponent(session.id)}`)} download={`survng-diagnostics-${session.id}.json`}><Download size={14} />Download</a><button type="button" onClick={() => void stopTelemetryDiagnostics(session.id)}>Stop</button></div></article>;
                  })}</div> : <p className="telemetry-diagnostic-empty">No diagnostic capture is active.</p>}
                </section>
                {(telemetry?.diagnostics?.recent || []).some((session) => session.stopped_at || new Date(session.expires_at).getTime() <= Date.now()) ? <section className="telemetry-section"><details className="telemetry-technical"><summary>Recent diagnostic reports</summary><div className="telemetry-diagnostic-list">{telemetry.diagnostics.recent.filter((session) => session.stopped_at || new Date(session.expires_at).getTime() <= Date.now()).map((session) => <article className="telemetry-diagnostic-card" key={session.id}><div><strong>{session.scope === "camera" ? cameras.find((camera) => camera.id === session.camera_id)?.name || session.camera_id : String(session.scope).replaceAll("_", " ")} diagnostics</strong><span>{formatDateTime(session.started_at, timeZone)}</span></div><a className="button" href={appUrl(`/api/telemetry/diagnostics/${encodeURIComponent(session.id)}`)} download={`survng-diagnostics-${session.id}.json`}><Download size={14} />Download</a></article>)}</div></details></section> : null}
                {(telemetry?.operational_events || []).length ? <section className="telemetry-section"><details className="telemetry-technical"><summary>Recent health events</summary><div className="telemetry-health-event-list">{telemetry.operational_events.slice(0, 10).map((event) => <div key={event.id}><span>{event.summary}{Number(event.count || 1) > 1 ? ` · ${event.count} occurrences` : ""}</span><time>{formatDateTime(event.occurred_at, timeZone)}</time></div>)}</div></details></section> : null}
              </div></div> : (
                <div className="detection-settings subsection-workspace health-subsection-workspace">
                  <nav id="health-section-tabs" className="admin-section-tabs camera-section-tabs detection-subsection-tabs" role="tablist" aria-label="Health sections" onKeyDown={(event) => moveTabFocus(event, HEALTH_TELEMETRY_SECTIONS, telemetrySection === "occupancy" ? "occupancy" : "health", (next) => selectAdminSubsection(next, setTelemetrySection, "telemetry"))}>
                    <button id="health-tab-health" data-tab-id="health" type="button" tabIndex={telemetrySection === "occupancy" ? -1 : 0} aria-controls="telemetry-view-panel" className={telemetrySection === "occupancy" ? "" : "active"} onClick={() => selectAdminSubsection("health", setTelemetrySection, "telemetry")} role="tab" aria-selected={telemetrySection !== "occupancy"}><Gauge size={15} />Telemetry</button>
                    <button id="health-tab-occupancy" data-tab-id="occupancy" type="button" tabIndex={telemetrySection === "occupancy" ? 0 : -1} aria-controls="telemetry-view-panel" className={telemetrySection === "occupancy" ? "active" : ""} onClick={() => selectAdminSubsection("occupancy", setTelemetrySection, "telemetry")} role="tab" aria-selected={telemetrySection === "occupancy"}><Cpu size={15} />Detection at a glance</button>
                  </nav>
                  <div id="telemetry-view-panel" className="detection-settings-content health-subsection-content telemetry-tab-panel" role="tabpanel" aria-labelledby={telemetrySection === "occupancy" ? "health-tab-occupancy" : "health-tab-health"}>
                    {telemetrySection === "occupancy" ? (
                      <DetectionOccupancyCard telemetry={telemetry} cameraId={telemetryCamera} config={config} onOpenSetting={openOccupancySetting} />
                    ) : (
                      <TelemetryViewer data={telemetry} cameraId={telemetryCamera} timeZone={timeZone} config={config} />
                    )}
                  </div>
                </div>
              )}
            </section>
          </>
        ) : settingsTab === "maintenance" ? (
          <section id="admin-panel-maintenance" className="bento-card config-editor settings-panel settings-panel-wide maintenance-panel" aria-labelledby="admin-destination-maintenance">
            <details className="maintenance-explanation"><summary>What these checks do</summary><div><p>Quick Check is bounded to recent media and the newest index rows, so it will not saturate network storage. Repeated quick repairs can still leave older index batches for a later check.</p><p>Full Scan checks the entire library, reports progress, and can be cancelled. Repair only clears confirmed-missing index rows and media links; flaky or offline storage never deletes valid records. Repair also checks a small bounded batch of older recording metadata. Repairs never delete media or incident history.</p></div></details>
            {maintenanceError ? <div className="error-banner">{maintenanceError}</div> : null}
            <MaintenanceViewer state={maintenance} />
          </section>
        ) : settingsTab === "logs" ? (
          <section id="admin-panel-logs" className="bento-card config-editor settings-panel settings-panel-wide log-panel" aria-labelledby="admin-destination-logs">
            <LogViewer
              lines={logLines}
              filter={logFilter}
              setFilter={setLogFilter}
              order={logOrder}
              setOrder={setLogOrder}
              timeZone={timeZone}
            />
          </section>
        ) : (
          <>
            <section id="admin-panel-cameras" className="bento-card config-editor settings-panel settings-panel-wide admin-workspace-cameras subsection-workspace">

              {cameraOrderEditing ? <section className="sub-panel camera-order-panel" aria-label="Camera order">
                <div className="tree-list">
                  {cameras.map((camera) => (
                    <button
                      type="button"
                      aria-current={camera.id === selectedCamera?.id ? "page" : undefined}
                      key={camera.id}
                      draggable
                      className={`ordering ${camera.id === selectedCamera?.id ? "active" : ""} ${dragConfigCameraTarget === camera.id ? (dragConfigCameraAfter ? "drag-after" : "drag-before") : ""}`}
                      onClick={() => setSelectedId(camera.id)}
                      onKeyDown={(event) => {
                        if (!["ArrowUp", "ArrowDown"].includes(event.key)) return;
                        event.preventDefault();
                        setSelectedId(camera.id);
                        const offset = event.key === "ArrowUp" ? -1 : 1;
                        const currentIndex = cameras.findIndex((item) => item.id === camera.id);
                        const targetIndex = currentIndex + offset;
                        if (targetIndex >= 0 && targetIndex < cameras.length) {
                          moveConfigCamera(camera.id, cameras[targetIndex].id, offset > 0);
                          setSaveNotice({ state: "pending", text: `${camera.name || camera.id} moved to position ${targetIndex + 1}. Save order to apply.` });
                        }
                      }}
                      onDragStart={(event) => {
                        setDragConfigCameraId(camera.id);
                        event.dataTransfer.effectAllowed = "move";
                        event.dataTransfer.setData("text/plain", camera.id);
                      }}
                      onDragOver={(event) => {
                        if (!dragConfigCameraId || dragConfigCameraId === camera.id) return;
                        event.preventDefault();
                        const bounds = event.currentTarget.getBoundingClientRect();
                        setDragConfigCameraTarget(camera.id);
                        setDragConfigCameraAfter(event.clientY > bounds.top + bounds.height / 2);
                      }}
                      onDrop={(event) => {
                        event.preventDefault();
                        const sourceId = event.dataTransfer.getData("text/plain") || dragConfigCameraId;
                        const bounds = event.currentTarget.getBoundingClientRect();
                        moveConfigCamera(sourceId, camera.id, event.clientY > bounds.top + bounds.height / 2);
                        setDragConfigCameraId("");
                        setDragConfigCameraTarget("");
                      }}
                      onDragEnd={() => {
                        setDragConfigCameraId("");
                        setDragConfigCameraTarget("");
                      }}
                    >
                      <GripVertical size={16} />
                      <span>{camera.name || camera.id}</span>
                      {perCameraDirty[camera.id]?.settings || perCameraDirty[camera.id]?.zones
                        ? <em className="camera-dirty-badge" aria-label="Unsaved changes">Edited</em>
                        : null}
                    </button>
                  ))}
                </div>
              </section> : null}

              {selectedCamera ? <div id="camera-section-tabs" className="admin-section-tabs camera-section-tabs detection-subsection-tabs" role="tablist" aria-label={`${selectedCamera.name} settings sections`} onKeyDown={(event) => moveTabFocus(event, CAMERA_ADMIN_SECTIONS, cameraSection, (next) => selectAdminSubsection(next, setCameraSection, "cameras"))}>
                <button id="camera-tab-settings" data-tab-id="settings" tabIndex={cameraSection === "settings" ? 0 : -1} aria-controls="camera-settings-panel" type="button" className={cameraSection === "settings" ? "active" : ""} onClick={() => selectAdminSubsection("settings", setCameraSection, "cameras")} role="tab" aria-selected={cameraSection === "settings"}><Cog size={15} />Settings</button>
                <button id="camera-tab-motion" data-tab-id="motion" tabIndex={cameraSection === "motion" ? 0 : -1} aria-controls="camera-settings-panel" type="button" className={cameraSection === "motion" ? "active" : ""} onClick={() => selectAdminSubsection("motion", setCameraSection, "cameras")} role="tab" aria-selected={cameraSection === "motion"}><Activity size={15} />Motion/Object</button>
                <button id="camera-tab-zones" data-tab-id="zones" tabIndex={cameraSection === "zones" ? 0 : -1} aria-controls="camera-settings-panel" type="button" className={cameraSection === "zones" ? "active" : ""} onClick={() => selectAdminSubsection("zones", setCameraSection, "cameras")} role="tab" aria-selected={cameraSection === "zones"}><Crop size={15} />Zones</button>
                <button id="camera-tab-info" data-tab-id="info" tabIndex={cameraSection === "info" ? 0 : -1} aria-controls="camera-settings-panel" type="button" className={cameraSection === "info" ? "active" : ""} onClick={() => selectAdminSubsection("info", setCameraSection, "cameras")} role="tab" aria-selected={cameraSection === "info"}><Gauge size={15} />Info</button>
              </div> : null}

              <div id="camera-settings-panel" className="config-form" role="tabpanel" aria-labelledby={`camera-tab-${cameraSection}`}>
                {selectedCamera ? (
                  <>
                    {cameraSection === "settings" ? <>
                      <div className="field-row camera-identity-fields">
                        <label>Name<input value={selectedCamera.name} onChange={(event) => updateCamera(selectedCamera.id, ["name"], event.target.value)} /></label>
                      </div>
                    </> : null}

                    {cameraSection === "motion" ? <div className="field-row camera-object-policy-fields">
                      <label>Incident eligibility<select value={selectedCamera.require_incident_zone == null ? "" : String(selectedCamera.require_incident_zone)} onChange={(event) => updateCamera(selectedCamera.id, ["require_incident_zone"], event.target.value === "" ? null : event.target.value === "true")}>
                        <option value="">Use global ({(config.detector?.require_incident_zone ?? true) ? "Zones" : "Zones + Full Frame"})</option>
                        <option value="true">Zones</option>
                        <option value="false">Zones + Full Frame</option>
                      </select><small>Ignore zones always suppress their matching object classes.</small></label>
                      <label>Repeated scene context<select value={selectedCamera.object_activity_attribution || "inherit"} onChange={(event) => updateCamera(selectedCamera.id, ["object_activity_attribution"], event.target.value)}>
                        <option value="inherit">Use global ({config.detector?.object_activity_attribution === "shadow" ? "Observe" : config.detector?.object_activity_attribution === "off" ? "Off" : "Prevent labels"})</option>
                        <option value="enforce">Prevent false incident labels</option>
                        <option value="shadow">Observe only</option>
                        <option value="off">Off</option>
                      </select><small>Controls whether stable objects repeatedly seen in one location can remain evidence without labeling the incident.</small></label>
                    </div> : null}

                    {cameraSection === "info" ? <div className="field-row camera-info-fields">
                      <label>Generated Camera ID<input value={slugify(selectedCamera.name || selectedCamera.id || "camera")} readOnly /></label>
                      <label>Detected Backend<input value={inferredBackendLabel(selectedCamera)} readOnly /></label>
                    </div> : null}

                    {cameraSection === "settings" ? <>
                      <div className="camera-connectivity-grid">
                        <section className="sub-panel camera-stream-panel" aria-labelledby={`camera-stream-title-${selectedCamera.id}`}>
                          <h3 id={`camera-stream-title-${selectedCamera.id}`} className="section-heading-with-icon"><span className="section-heading-icon"><Camera size={16} /></span>Streams</h3>
                          <div className="field-row stream-field-row camera-stream-url-fields">
                            <div className="stream-field">
                              <div className="stream-field-head">
                                <label htmlFor={`main-stream-${selectedCamera.id}`}>Main Stream URL</label>
                                <label className="stream-record-toggle"><input type="checkbox" checked={selectedCamera.record} onChange={(event) => updateCamera(selectedCamera.id, ["record"], event.target.checked)} /> Record</label>
                              </div>
                              <input id={`main-stream-${selectedCamera.id}`} value={selectedCamera.stream_url || ""} onChange={(event) => updateCamera(selectedCamera.id, ["stream_url"], event.target.value)} />
                            </div>
                            <div className="stream-field">
                              <div className="stream-field-head">
                                <label htmlFor={`sub-stream-${selectedCamera.id}`}>Live/Sub Stream URL</label>
                                <label className="stream-record-toggle"><input type="checkbox" checked={selectedCamera.record_sub || false} onChange={(event) => updateCamera(selectedCamera.id, ["record_sub"], event.target.checked)} /> Record</label>
                              </div>
                              <input id={`sub-stream-${selectedCamera.id}`} value={selectedCamera.live_stream_url || ""} onChange={(event) => updateCamera(selectedCamera.id, ["live_stream_url"], event.target.value)} />
                            </div>
                          </div>
                        </section>
                        <CameraOnvifEditor camera={selectedCamera} onChange={(path, value) => updateCamera(selectedCamera.id, path, value)} />
                      </div>
                      <LiveViewFramingEditor camera={selectedCamera} onChange={(path, value) => updateCamera(selectedCamera.id, path, value)} />
                      <details className="camera-retention-details">
                        <summary>Camera recording retention</summary>
                        <div className="field-row">
                          <label>Main stream history<input type="number" min="1" max="3650" step="1" placeholder={`Global: ${config.retention?.main_days ?? 7} days`} value={selectedCamera.retention?.main_days ?? ""} onChange={(event) => updateCamera(selectedCamera.id, ["retention", "main_days"], event.target.value === "" ? null : Number(event.target.value))} /><small>Leave blank to inherit the global policy.</small></label>
                          <label>Substream history<input type="number" min="1" max="3650" step="1" placeholder={`Global: ${config.retention?.live_days ?? 21} days`} value={selectedCamera.retention?.live_days ?? ""} onChange={(event) => updateCamera(selectedCamera.id, ["retention", "live_days"], event.target.value === "" ? null : Number(event.target.value))} /><small>Leave blank to inherit the global policy.</small></label>
                        </div>
                      </details>
                    </> : null}

                    <div className="config-panels">
                      {cameraSection === "motion" ? <div className="sub-panel">
                        <h3>Motion Triggers &amp; Filtering</h3>
                        <MotionDecisionEditor
                          cameraName={selectedCamera.name}
                          fusion={selectedCamera.motion_qualification?.pipeline?.fusion}
                          mode={selectedCamera.motion_qualification?.mode || "inherit"}
                          globalMode={config.motion_qualification?.mode || "camera_rescue"}
                          inherited={selectedCamera.motion_qualification?.pipeline?.fusion == null}
                          inheritedFusion={config.motion_qualification?.pipeline?.fusion}
                          onModeChange={(mode) => updateCamera(selectedCamera.id, ["motion_qualification", "mode"], mode)}
                          onSetInherited={(shouldInherit) => {
                            const pipeline = { ...(selectedCamera.motion_qualification?.pipeline || {}) };
                            pipeline.fusion = shouldInherit
                              ? null
                              : buildMotionDecisionFusion(
                                readMotionDecisionFusion(config.motion_qualification?.pipeline?.fusion).settings,
                              );
                            updateCamera(selectedCamera.id, ["motion_qualification", "pipeline"], pipeline);
                          }}
                          onChange={(fusion) => updateCamera(
                            selectedCamera.id,
                            ["motion_qualification", "pipeline"],
                            { ...(selectedCamera.motion_qualification?.pipeline || {}), fusion },
                          )}
                          onRestoreDefaults={() => updateCamera(
                            selectedCamera.id,
                            ["motion_qualification"],
                            defaultCameraMotionQualification(),
                          )}
                          configurationInherited={cameraMotionQualificationInherited(selectedCamera.motion_qualification)}
                        />
                        <MotionAnalysisPresetEditor
                          qualification={selectedCamera.motion_qualification?.pipeline?.qualification}
                          inherited={selectedCamera.motion_qualification?.pipeline?.qualification == null}
                          catalog={motionCatalog}
                          onSetInherited={() => updateCamera(
                            selectedCamera.id,
                            ["motion_qualification", "pipeline"],
                            { ...(selectedCamera.motion_qualification?.pipeline || {}), qualification: null },
                          )}
                          onChange={(qualification) => updateCamera(
                            selectedCamera.id,
                            ["motion_qualification", "pipeline"],
                            { ...(selectedCamera.motion_qualification?.pipeline || {}), qualification },
                          )}
                        />
                        <details className="motion-tuning-details">
                          <summary>Advanced camera tuning</summary>
                          <div className="motion-camera-tuning">
                            <label>Sensitivity<select value={selectedCamera.motion_qualification?.sensitivity || "inherit"} onChange={(event) => updateCamera(selectedCamera.id, ["motion_qualification", "sensitivity"], event.target.value)}>
                              <option value="inherit">Use global setting</option>
                              <option value="high">High</option>
                              <option value="balanced">Balanced</option>
                              <option value="low">Low</option>
                            </select></label>
                            <label>Stationary object policy<select value={selectedCamera.motion_qualification?.stationary_object_tolerance || "inherit"} onChange={(event) => updateCamera(selectedCamera.id, ["motion_qualification", "stationary_object_tolerance"], event.target.value)}>
                              <option value="inherit">Use global setting</option>
                              <option value="low">Light</option>
                              <option value="balanced">Standard</option>
                              <option value="high">Strong</option>
                            </select><small>Controls how aggressively EMA rejects confined outline shimmer and reflections before object detection. Strong may ignore unusually slow or distant movement.</small></label>
                            <label>Light and shadow filtering<select value={selectedCamera.motion_qualification?.illumination_filter_enabled == null ? "" : String(selectedCamera.motion_qualification.illumination_filter_enabled)} onChange={(event) => updateCamera(selectedCamera.id, ["motion_qualification", "illumination_filter_enabled"], event.target.value === "" ? null : event.target.value === "true")}><option value="">Use global setting</option><option value="true">Enabled</option><option value="false">Disabled</option></select><small>Ignores clear moving illumination while uncertain motion continues to object detection.</small></label>
                            <label>Analysis size<select value={selectedCamera.motion_qualification?.frame_width ?? ""} onChange={(event) => updateCamera(selectedCamera.id, ["motion_qualification", "frame_width"], event.target.value ? Number(event.target.value) : null)}>
                              <option value="">Use global setting</option>
                              <option value="320">320 px</option>
                              <option value="480">480 px</option>
                              <option value="640">640 px</option>
                              <option value="720">720 px</option>
                              <option value="800">800 px</option>
                            </select></label>
                            <label>Visual confidence<input type="number" min="0" max="1" step="0.01" placeholder={`Global: ${config.motion_qualification?.visual_backup_min_score ?? 0.7}`} value={selectedCamera.motion_qualification?.visual_backup_min_score ?? ""} onChange={(event) => updateCamera(selectedCamera.id, ["motion_qualification", "visual_backup_min_score"], event.target.value === "" ? null : Number(event.target.value))} /><small>Leave blank to inherit. Higher values require stronger visual motion before camera-notification rescue runs detection.</small></label>
                            <label>Strong samples<input type="number" min="2" max="10" step="1" placeholder={`Global: ${config.motion_qualification?.visual_backup_min_consecutive ?? 3}`} value={selectedCamera.motion_qualification?.visual_backup_min_consecutive ?? ""} onChange={(event) => updateCamera(selectedCamera.id, ["motion_qualification", "visual_backup_min_consecutive"], event.target.value === "" ? null : Number(event.target.value))} /><small>Consecutive qualifying samples required before rescue.</small></label>
                            <label>Visual grace<input type="number" min="0" max="5" step="0.1" placeholder={`Global: ${config.motion_qualification?.visual_backup_grace_seconds ?? 1.5}s`} value={selectedCamera.motion_qualification?.visual_backup_grace_seconds ?? ""} onChange={(event) => updateCamera(selectedCamera.id, ["motion_qualification", "visual_backup_grace_seconds"], event.target.value === "" ? null : Number(event.target.value))} /><small>How long strong motion must persist. Leave blank to inherit.</small></label>
                            <label>Rescue cooldown<input type="number" min="5" max="300" step="5" placeholder={`Global: ${config.motion_qualification?.visual_backup_cooldown_seconds ?? 20}s`} value={selectedCamera.motion_qualification?.visual_backup_cooldown_seconds ?? ""} onChange={(event) => updateCamera(selectedCamera.id, ["motion_qualification", "visual_backup_cooldown_seconds"], event.target.value === "" ? null : Number(event.target.value))} /><small>Minimum seconds between visual rescue attempts.</small></label>
                            <label>Rescues per 5 minutes<input type="number" min="1" max="30" step="1" placeholder={`Global: ${config.motion_qualification?.visual_backup_max_triggers_5m ?? 3}`} value={selectedCamera.motion_qualification?.visual_backup_max_triggers_5m ?? ""} onChange={(event) => updateCamera(selectedCamera.id, ["motion_qualification", "visual_backup_max_triggers_5m"], event.target.value === "" ? null : Number(event.target.value))} /><small>Per-camera ceiling for visual rescue detection attempts.</small></label>
                            <label>Borderline Rescue<select value={selectedCamera.motion_qualification?.borderline_rescue_enabled == null ? "" : String(selectedCamera.motion_qualification.borderline_rescue_enabled)} onChange={(event) => updateCamera(selectedCamera.id, ["motion_qualification", "borderline_rescue_enabled"], event.target.value === "" ? null : event.target.value === "true")}>
                              <option value="">Use global setting</option>
                              <option value="true">Enabled</option>
                              <option value="false">Disabled</option>
                            </select></label>
                            <label>Rescue Margin<input type="number" min="0" max="0.1" step="0.005" placeholder="Global" value={selectedCamera.motion_qualification?.borderline_margin ?? ""} onChange={(event) => updateCamera(selectedCamera.id, ["motion_qualification", "borderline_margin"], event.target.value === "" ? null : Number(event.target.value))} /></label>
                            <label>Double-check filtered motion<select value={selectedCamera.motion_qualification?.suppression_verification_rate == null ? "" : String(selectedCamera.motion_qualification.suppression_verification_rate)} onChange={(event) => updateCamera(selectedCamera.id, ["motion_qualification", "suppression_verification_rate"], event.target.value === "" ? null : Number(event.target.value))}><option value="">Use global setting</option><option value="0">Off</option><option value="0.01">About 1 in 100</option><option value="0.05">About 1 in 20</option><option value="0.1">About 1 in 10</option></select><small>Runs object detection on a small sample that visual motion would filter. A configured object safely restores the incident.</small></label>
                          </div>
                        </details>
                      </div> : null}
                    </div>

                    {cameraSection === "zones" ? <ZoneEditor
                      camera={selectedCamera}
                      classOptions={zoneClassOptions}
                      onChange={(zones) => updateCamera(selectedCamera.id, ["zones"], zones)}
                    /> : null}

                    {cameraSection === "info" ? <>
                      <RuntimeStatus status={selectedRuntimeStatus} timeZone={timeZone} motionCatalog={motionCatalog} />
                      <MotionDebugViewer cameraId={selectedCamera.id} timeZone={timeZone} />
                      {probe ? <ProbeResult probe={probe} /> : null}
                    </> : null}
                  </>
                ) : (
                  <div className="empty-state">Add a camera to begin.</div>
                )}
              </div>
            </section>
          </>
        )}
      </div>
      {adminSaveAvailable ? <div className={`admin-save-bar${currentAdminDirty ? " dirty" : ""}`} role="region" aria-label="Save Admin changes">
        <div className="admin-save-state" role="status" aria-live="polite">
          {saveNotice?.state === "error" ? <CircleAlert size={17} /> : currentAdminSaving ? <RefreshCcw className="spin" size={17} /> : <CircleDot size={17} />}
          <span><strong>{adminSaveStatusTitle}</strong><small>{saveNotice?.state === "error" ? saveNotice.text : currentAdminDirty ? adminSaveImpact : saveNotice?.text || adminSaveImpact}</small></span>
        </div>
        <div className="admin-save-actions">
          <button type="button" onClick={discardAdminChanges} disabled={!currentAdminDirty || currentAdminSaving}>Discard</button>
          <button type="button" className="primary" onClick={() => void saveCurrentAdminChanges()} disabled={!currentAdminDirty || currentAdminSaving}>{currentAdminSaving ? <RefreshCcw className="spin" size={16} /> : <Save size={16} />}{currentAdminSaving ? "Saving…" : "Save changes"}</button>
        </div>
      </div> : null}
      {selectedAudit ? (
        <MotionAuditOverlay
          item={selectedAudit}
          items={selectedAuditItems}
          timeZone={timeZone}
          onClose={() => setSelectedAuditId(null)}
          onSelect={(item) => setSelectedAuditId(item.id)}
        />
      ) : null}
    </main>
  );
}

export function ZoneEditor({ camera, classOptions = [], onChange }) {
  const zones = camera.zones || [];
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [dragPoint, setDragPoint] = useState(null);
  const [pointAddHistory, setPointAddHistory] = useState({});
  const [snapshotSize, setSnapshotSize] = useState(null);
  const [canvasSize, setCanvasSize] = useState(null);
  const canvasRef = useRef(null);
  const snapshotUrl = useMemo(() => appUrl(`/api/cameras/${camera.id}/zone-snapshot.jpg?source=live&t=${Date.now()}`), [camera.id]);
  const selectedZone = zones[selectedIndex] || null;

  useEffect(() => {
    setSelectedIndex(0);
    setDragPoint(null);
    setPointAddHistory({});
    setSnapshotSize(null);
  }, [camera.id]);

  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const updateSize = () => {
      setCanvasSize((current) => {
        const next = {
          width: Math.max(0, canvas.clientWidth),
          height: Math.max(0, canvas.clientHeight),
        };
        return current && Math.abs(current.width - next.width) < 0.5 && Math.abs(current.height - next.height) < 0.5
          ? current
          : next;
      });
    };
    updateSize();
    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", updateSize);
      return () => window.removeEventListener("resize", updateSize);
    }
    const observer = new ResizeObserver(updateSize);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [camera.id]);

  const mediaSize = useMemo(() => {
    if (!snapshotSize?.width || !snapshotSize?.height || !canvasSize?.width || !canvasSize?.height) return null;
    const scale = Math.min(canvasSize.width / snapshotSize.width, canvasSize.height / snapshotSize.height);
    return {
      width: Math.max(1, Math.floor(snapshotSize.width * scale)),
      height: Math.max(1, Math.floor(snapshotSize.height * scale)),
    };
  }, [canvasSize, snapshotSize]);

  useEffect(() => {
    if (!dragPoint) return undefined;
    const preventSelection = (event) => event.preventDefault();
    document.documentElement.classList.add("zone-vertex-dragging");
    document.addEventListener("selectstart", preventSelection);
    return () => {
      document.documentElement.classList.remove("zone-vertex-dragging");
      document.removeEventListener("selectstart", preventSelection);
      window.getSelection()?.removeAllRanges();
    };
  }, [dragPoint]);

  function replaceZone(index, patch) {
    onChange(zones.map((zone, zoneIndex) => zoneIndex === index ? { ...zone, ...patch } : zone));
  }

  function addZone() {
    const next = [...zones, {
      name: `Zone ${zones.length + 1}`,
      color: ["#22c55e", "#38bdf8", "#f59e0b", "#e879f9"][zones.length % 4],
      enabled: true,
      points: [],
      object_classes: [],
      confidence_threshold: null,
      behavior: "incident",
      exclude_from_ema: false,
      trigger: "bottom_center",
    }];
    onChange(next);
    setSelectedIndex(next.length - 1);
  }

  function removeZone(index) {
    onChange(zones.filter((_, zoneIndex) => zoneIndex !== index));
    setPointAddHistory((current) => Object.fromEntries(
      Object.entries(current).flatMap(([zoneIndex, history]) => {
        const numericIndex = Number(zoneIndex);
        if (numericIndex === index) return [];
        return [[numericIndex > index ? numericIndex - 1 : numericIndex, history]];
      }),
    ));
    setSelectedIndex((current) => Math.max(0, Math.min(current, zones.length - 2)));
  }

  function undoPoint() {
    const history = pointAddHistory[selectedIndex] || [];
    const insertionIndex = history.at(-1);
    const points = selectedZone?.points || [];
    if (!selectedZone || insertionIndex == null || insertionIndex >= points.length) return;
    replaceZone(selectedIndex, {
      points: points.filter((_, pointIndex) => pointIndex !== insertionIndex),
    });
    setPointAddHistory((current) => {
      const remaining = (current[selectedIndex] || [])
        .slice(0, -1)
        .map((pointIndex) => pointIndex > insertionIndex ? pointIndex - 1 : pointIndex);
      return { ...current, [selectedIndex]: remaining };
    });
  }

  function pointerPosition(event) {
    const rect = event.currentTarget.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
      y: Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)),
    };
  }

  function addPoint(event) {
    if (!selectedZone || event.target !== event.currentTarget) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const point = pointerPosition(event);
    const inserted = insertZonePointWithIndex(
      selectedZone.points,
      point,
      { x: rect.width, y: rect.height },
    );
    setPointAddHistory((current) => {
      const shifted = (current[selectedIndex] || [])
        .map((pointIndex) => pointIndex >= inserted.insertionIndex ? pointIndex + 1 : pointIndex);
      return { ...current, [selectedIndex]: [...shifted, inserted.insertionIndex] };
    });
    replaceZone(selectedIndex, {
      points: inserted.points,
    });
  }

  function movePoint(event) {
    if (!dragPoint || dragPoint.zoneIndex !== selectedIndex || !selectedZone) return;
    const points = [...(selectedZone.points || [])];
    points[dragPoint.pointIndex] = pointerPosition(event);
    replaceZone(selectedIndex, { points });
  }

  return (
    <div className="sub-panel zone-settings">
      <div className="zone-settings-head">
        <div><h3>Detection Zones</h3><p>Objects match using the bottom-center of their detection box.</p></div>
        <div className="zone-settings-actions">
          <button type="button" onClick={undoPoint} disabled={!(pointAddHistory[selectedIndex]?.length)} title="Remove the last point added"><Undo2 size={15} /> Undo Point</button>
          <button type="button" onClick={addZone}><Plus size={15} /> Add Zone</button>
        </div>
      </div>
      <div className="zone-editor-layout">
        <div className="zone-canvas" ref={canvasRef}>
          <div className="zone-canvas-media" style={mediaSize || undefined}>
            <img
              src={snapshotUrl}
              alt={`${camera.name} zone editor`}
              onLoad={(event) => setSnapshotSize({
                width: event.currentTarget.naturalWidth,
                height: event.currentTarget.naturalHeight,
              })}
            />
            <svg
              viewBox="0 0 100 100"
              preserveAspectRatio="none"
              onPointerDown={addPoint}
              onPointerMove={movePoint}
              onPointerUp={(event) => {
                movePoint(event);
                setDragPoint(null);
              }}
              onPointerCancel={() => setDragPoint(null)}
              onDragStart={(event) => event.preventDefault()}
              aria-label="Zone polygon editor"
            >
              {zones.map((zone, zoneIndex) => {
                const points = (zone.points || []).map((point) => `${point.x * 100},${point.y * 100}`).join(" ");
                return (
                  <g key={`${zone.name}-${zoneIndex}`} opacity={zone.enabled === false ? 0.35 : 1}>
                    {zone.points?.length >= 3 ? <polygon points={points} fill={`${zone.color || "#22c55e"}33`} stroke={zone.color || "#22c55e"} strokeWidth="0.55" vectorEffect="non-scaling-stroke" pointerEvents="none" /> : null}
                    {zone.points?.length === 2 ? <polyline points={points} fill="none" stroke={zone.color || "#22c55e"} strokeWidth="0.55" vectorEffect="non-scaling-stroke" pointerEvents="none" /> : null}
                    {zoneIndex === selectedIndex ? (zone.points || []).map((point, pointIndex) => (
                      <circle
                        key={pointIndex}
                        cx={point.x * 100}
                        cy={point.y * 100}
                        r="0.85"
                        fill="#fff"
                        stroke={zone.color || "#22c55e"}
                        strokeWidth="0.35"
                        vectorEffect="non-scaling-stroke"
                        onPointerDown={(event) => {
                          event.stopPropagation();
                          event.currentTarget.setPointerCapture(event.pointerId);
                          setDragPoint({ zoneIndex, pointIndex });
                        }}
                      />
                    )) : null}
                  </g>
                );
              })}
            </svg>
          </div>
          {!selectedZone ? <div className="zone-canvas-empty">Add a zone to begin</div> : selectedZone.points?.length < 3 ? <div className="zone-canvas-hint">Click at least three points</div> : null}
        </div>
        <aside className="zone-sidebar">
          <div className="zone-sidebar-panel">
            <h4>Zones</h4>
            <div className="zone-list">
              {zones.map((zone, index) => (
                <button type="button" key={`${zone.name}-${index}`} className={index === selectedIndex ? "active" : ""} onClick={() => setSelectedIndex(index)}>
                  <span className="zone-swatch" style={{ background: zone.color || "#22c55e" }} />
                  <span>{zone.name || `Zone ${index + 1}`}</span>
                  <small>{zone.behavior === "none" ? "no object effect" : zone.behavior}{zone.exclude_from_ema ? " · EMA excluded" : ""}</small>
                </button>
              ))}
              {!zones.length ? <div className="empty-state compact">No zones configured.</div> : null}
            </div>
          </div>
          {selectedZone ? (
            <div className="zone-sidebar-panel zone-config-panel">
              <h4>Zone settings</h4>
              <div className="zone-fields">
                <label className="zone-field-name">Name<input value={selectedZone.name || ""} onChange={(event) => replaceZone(selectedIndex, { name: event.target.value })} /></label>
                <label className="zone-field-color">Color<input className="zone-color-input" type="color" value={selectedZone.color || "#22c55e"} onChange={(event) => replaceZone(selectedIndex, { color: event.target.value })} /></label>
                <label className="zone-field-behavior">Behavior<select value={selectedZone.behavior || "incident"} onChange={(event) => replaceZone(selectedIndex, { behavior: event.target.value })}><option value="incident">Incident</option><option value="ignore">Ignore</option><option value="none">No object effect</option></select></label>
                <div className="zone-class-field">
                  <span>Object Classes</span>
                  <details className={`zone-class-dropdown${selectedZone.behavior === "none" ? " disabled" : ""}`}>
                    <summary>{selectedZone.behavior === "none" ? "Not used" : selectedZone.object_classes?.length ? selectedZone.object_classes.join(", ") : "All classes"}</summary>
                    <div className="zone-class-menu">
                      <label>
                        <input type="checkbox" checked={!selectedZone.object_classes?.length} onChange={() => replaceZone(selectedIndex, { object_classes: [] })} />
                        All classes
                      </label>
                      {classOptions.map((className) => {
                        const selectedClasses = selectedZone.object_classes || [];
                        const checked = selectedClasses.includes(className);
                        return (
                          <label key={className}>
                            <input
                              type="checkbox"
                              checked={checked}
                              onChange={() => replaceZone(selectedIndex, {
                                object_classes: checked
                                  ? selectedClasses.filter((item) => item !== className)
                                  : [...selectedClasses, className],
                              })}
                            />
                            {className}
                          </label>
                        );
                      })}
                      {!classOptions.length ? <small>No model classes reported</small> : null}
                    </div>
                  </details>
                </div>
                <label className="zone-field-confidence">Confidence<input type="number" min="0.01" max="0.99" step="0.01" placeholder={selectedZone.behavior === "none" ? "N/A" : "Global"} disabled={selectedZone.behavior === "none"} value={selectedZone.confidence_threshold ?? ""} onChange={(event) => replaceZone(selectedIndex, { confidence_threshold: event.target.value === "" ? null : Number(event.target.value) })} /></label>
                <div className="zone-toggle-stack">
                  <label title="Motion inside this zone will not validate or trigger EMA activity. Object incident rules remain unchanged."><input type="checkbox" checked={selectedZone.exclude_from_ema === true} onChange={(event) => replaceZone(selectedIndex, { exclude_from_ema: event.target.checked })} /> Exclude from EMA</label>
                  <label><input type="checkbox" checked={selectedZone.enabled !== false} onChange={(event) => replaceZone(selectedIndex, { enabled: event.target.checked })} /> Enabled</label>
                </div>
                <button type="button" className="danger zone-remove-button" onClick={() => removeZone(selectedIndex)}><Trash2 size={15} /> Remove Zone</button>
              </div>
            </div>
          ) : null}
        </aside>
      </div>
    </div>
  );
}

export function LogViewer({ lines, filter, setFilter, order, setOrder, timeZone }) {
  const displayedLines = order === "oldest" ? lines : [...lines].reverse();
  return (
    <div className="log-viewer">
      <div className="log-toolbar">
        <div className="log-filter-control">
          <label htmlFor="log-filter-input">Filter</label>
          <div className="log-filter-row">
            <input id="log-filter-input" value={filter} onChange={(event) => setFilter(event.target.value)} placeholder="logger, text, error..." />
            <button
              type="button"
              className="log-order-button"
              onClick={() => setOrder(order === "newest" ? "oldest" : "newest")}
              title={order === "newest" ? "Show oldest messages first" : "Show newest messages first"}
              aria-label={order === "newest" ? "Newest messages first; switch to oldest first" : "Oldest messages first; switch to newest first"}
            >
              <ArrowUpDown size={15} /> {order === "newest" ? "Newest first" : "Oldest first"}
            </button>
          </div>
        </div>
      </div>
      <div className="log-lines" role="log" aria-live="polite">
        {displayedLines.length ? displayedLines.map((line, index) => (
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

export function motionAuditOutcome(item) {
  const visualBackup = item.category === "visual_backup";
  const activeFollowup = item.category === "active_followup";
  if (item.reason === "illumination_change") return { label: "Filtered · light or shadow change", className: "not-run" };
  if (item.features?.illumination_verification_probe) return { label: item.object_detected ? "Light filter check · object rescued" : "Light filter check · no object", className: item.object_detected ? "object" : "clear" };
  if (item.interpretation?.category === "visual_backup_scene_learning") return { label: "Visual backup · scene learning", className: "not-run" };
  if (item.interpretation?.category === "visual_backup_below_threshold") return { label: "Credible EMA motion · below backup threshold", className: "not-run" };
  if (item.interpretation?.category === "object_not_motion_correlated") return { label: "Motion confirmed · detected object outside motion area", className: "clear" };
  if (item.interpretation?.category === "duplicate_active_event") return { label: "Duplicate · event active", className: "not-run" };
  if (item.interpretation?.category === "duplicate_event_cooldown") return { label: "Duplicate · cooldown", className: "not-run" };
  if (item.interpretation?.category === "filtered_before_object_detection") return { label: "Filtered before detection", className: "not-run" };
  if (item.object_detected === true) return { label: visualBackup ? "Visual backup · object found" : activeFollowup ? "Active follow-up · object found" : "Object found", className: "object" };
  if (item.object_detected === false) return { label: visualBackup ? "Visual backup · no object" : activeFollowup ? "Active follow-up · no object" : "No object", className: "clear" };
  return { label: visualBackup ? "Visual backup · incomplete" : activeFollowup ? "Active follow-up · incomplete" : "Not run", className: "not-run" };
}

export function MotionAuditAnnotatedImage({ item, alt, loading, onImageSize, interactive = false }) {
  const frameRef = useRef(null);
  const [frameSize, setFrameSize] = useState(null);
  const [imageSize, setImageSize] = useState(null);
  const [zoom, setZoom] = useState({ scale: 1, x: 0, y: 0 });
  const zoomRef = useRef(zoom);
  const pointersRef = useRef(new Map());
  const gestureRef = useRef(null);
  const regions = motionAuditRegions(item.features);
  const renderedImage = useMemo(() => {
    if (!frameSize?.width || !frameSize?.height || !imageSize?.width || !imageSize?.height) return null;
    const scale = Math.min(frameSize.width / imageSize.width, frameSize.height / imageSize.height);
    const width = imageSize.width * scale;
    const height = imageSize.height * scale;
    return { left: (frameSize.width - width) / 2, top: (frameSize.height - height) / 2, width, height };
  }, [frameSize, imageSize]);

  useEffect(() => {
    const reset = { scale: 1, x: 0, y: 0 };
    zoomRef.current = reset;
    setZoom(reset);
    setImageSize(null);
    pointersRef.current.clear();
    gestureRef.current = null;
  }, [item.id]);

  useLayoutEffect(() => {
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

  function imageLoaded(event) {
    const size = { width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight };
    setImageSize(size);
    onImageSize?.(size);
  }

  function updateZoom(next) {
    const scale = Math.max(1, Math.min(8, Number(next.scale) || 1));
    const xLimit = renderedImage ? renderedImage.width * (scale - 1) / 2 : 0;
    const yLimit = renderedImage ? renderedImage.height * (scale - 1) / 2 : 0;
    const value = {
      scale,
      x: scale === 1 ? 0 : Math.max(-xLimit, Math.min(xLimit, Number(next.x) || 0)),
      y: scale === 1 ? 0 : Math.max(-yLimit, Math.min(yLimit, Number(next.y) || 0)),
    };
    zoomRef.current = value;
    setZoom(value);
  }

  function zoomAt(clientX, clientY, scale) {
    if (!interactive || !renderedImage || !frameRef.current) return;
    const current = zoomRef.current;
    const rect = frameRef.current.getBoundingClientRect();
    const nextScale = Math.max(1, Math.min(8, scale));
    const localX = clientX - rect.left - frameSize.width / 2;
    const localY = clientY - rect.top - frameSize.height / 2;
    const ratio = nextScale / current.scale;
    updateZoom({
      scale: nextScale,
      x: localX - (localX - current.x) * ratio,
      y: localY - (localY - current.y) * ratio,
    });
  }

  function onWheel(event) {
    if (!interactive) return;
    event.preventDefault();
    zoomAt(event.clientX, event.clientY, zoomRef.current.scale * Math.exp(-event.deltaY * 0.0017));
  }

  function onPointerDown(event) {
    if (!interactive) return;
    event.currentTarget.setPointerCapture?.(event.pointerId);
    pointersRef.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
    const pointers = [...pointersRef.current.values()];
    if (pointers.length === 2) {
      gestureRef.current = {
        mode: "pinch",
        distance: Math.hypot(pointers[1].x - pointers[0].x, pointers[1].y - pointers[0].y),
        scale: zoomRef.current.scale,
        centerX: (pointers[0].x + pointers[1].x) / 2,
        centerY: (pointers[0].y + pointers[1].y) / 2,
      };
    } else {
      gestureRef.current = { mode: "pan", pointerId: event.pointerId, x: event.clientX, y: event.clientY, panX: zoomRef.current.x, panY: zoomRef.current.y };
    }
  }

  function onPointerMove(event) {
    if (!interactive || !pointersRef.current.has(event.pointerId)) return;
    pointersRef.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
    const pointers = [...pointersRef.current.values()];
    const gesture = gestureRef.current;
    if (pointers.length === 2) {
      const distance = Math.hypot(pointers[1].x - pointers[0].x, pointers[1].y - pointers[0].y);
      const centerX = (pointers[0].x + pointers[1].x) / 2;
      const centerY = (pointers[0].y + pointers[1].y) / 2;
      if (!gesture || gesture.mode !== "pinch") {
        gestureRef.current = { mode: "pinch", distance, scale: zoomRef.current.scale, centerX, centerY };
        return;
      }
      zoomAt(centerX, centerY, gesture.scale * distance / Math.max(1, gesture.distance));
      return;
    }
    if (gesture?.mode === "pan" && gesture.pointerId === event.pointerId && zoomRef.current.scale > 1) {
      updateZoom({ scale: zoomRef.current.scale, x: gesture.panX + event.clientX - gesture.x, y: gesture.panY + event.clientY - gesture.y });
    }
  }

  function onPointerEnd(event) {
    pointersRef.current.delete(event.pointerId);
    const remaining = [...pointersRef.current.entries()];
    if (remaining.length === 1) {
      const [pointerId, point] = remaining[0];
      gestureRef.current = { mode: "pan", pointerId, x: point.x, y: point.y, panX: zoomRef.current.x, panY: zoomRef.current.y };
    } else if (!remaining.length) {
      gestureRef.current = null;
    }
  }

  const canvasStyle = renderedImage ? {
    left: `${renderedImage.left}px`,
    top: `${renderedImage.top}px`,
    width: `${renderedImage.width}px`,
    height: `${renderedImage.height}px`,
    transform: `translate3d(${zoom.x}px, ${zoom.y}px, 0) scale(${zoom.scale})`,
  } : undefined;

  return (
    <div
      className={`motion-audit-annotated-image ${interactive ? "interactive" : ""} ${zoom.scale > 1 ? "zoomed" : ""}`}
      ref={frameRef}
      onWheel={onWheel}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerEnd}
      onPointerCancel={onPointerEnd}
      onDoubleClick={(event) => interactive && (zoom.scale > 1 ? updateZoom({ scale: 1, x: 0, y: 0 }) : zoomAt(event.clientX, event.clientY, 2))}
    >
      <div className="motion-audit-image-canvas" style={canvasStyle}>
        <img src={appUrl(`/api/motion-audit/${item.id}/snapshot.jpg`)} alt={alt} loading={loading} onLoad={imageLoaded} draggable="false" />
        {renderedImage && regions.length ? <div className="motion-audit-region-layer" aria-hidden="true">
          {regions.map(([x1, y1, x2, y2], index) => <span
            className="motion-audit-region"
            key={`${x1}-${y1}-${x2}-${y2}-${index}`}
            style={{ left: `${x1 * 100}%`, top: `${y1 * 100}%`, width: `${(x2 - x1) * 100}%`, height: `${(y2 - y1) * 100}%` }}
          >{index === regions.length - 1 ? <strong>motion</strong> : null}</span>)}
        </div> : null}
      </div>
      {interactive && zoom.scale > 1 ? <button type="button" className="motion-audit-zoom-reset" onClick={(event) => { event.stopPropagation(); updateZoom({ scale: 1, x: 0, y: 0 }); }}>Reset {zoom.scale.toFixed(1)}×</button> : null}
    </div>
  );
}

export function MotionAuditViewer({ items, total, page, pageSize, setPage, loading, error, timeZone, onOpen }) {
  const pageCount = Math.max(1, Math.ceil(total / pageSize));
  return (
    <div className="motion-audit-viewer">
      {error ? <div className="save-status motion-audit-error">{error}</div> : null}
      <div className="motion-audit-grid">
        {items.map((item) => {
          const outcome = motionAuditOutcome(item);
          const features = Object.entries(item.features || {}).filter(([name, value]) => (
            typeof value === "number"
            && Number.isFinite(value)
          ));
          return (
            <article className="motion-audit-card" key={item.id}>
              <button type="button" className="motion-audit-media" onClick={() => onOpen(item)} aria-label={`Open ${item.camera_id} motion audit image`}>
                {item.has_snapshot
                  ? <MotionAuditAnnotatedImage item={item} alt={`${item.camera_id} motion decision`} loading="lazy" />
                  : <div className="empty-thumb"><Camera size={28} /><span>Audit image unavailable</span></div>}
                <span className={`motion-audit-outcome ${outcome.className}`}>{outcome.label}</span>
              </button>
              <div className="motion-audit-body">
                <div className="motion-audit-title"><strong>{item.camera_id}</strong><time>{formatDateTime(item.created_at, timeZone)}</time></div>
                <div className="motion-audit-decision">
                  <span>{String(item.reason || "rejected").replaceAll("_", " ")}</span>
                  <strong>{Number(item.score || 0).toFixed(3)} / {Number(item.threshold || 0).toFixed(3)}</strong>
                </div>
                <div className="motion-audit-meter" aria-label={`Score ${item.score}, threshold ${item.threshold}`}>
                  <i style={{ width: `${Math.max(0, Math.min(100, Number(item.score || 0) * 100))}%` }} />
                  <b style={{ left: `${Math.max(0, Math.min(100, Number(item.threshold || 0) * 100))}%` }} />
                </div>
                <div className="motion-audit-features">
                  {features.map(([name, value]) => <span key={name}>{name.replaceAll("_", " ")} <strong>{Number(value).toFixed(2)}</strong></span>)}
                </div>
                <div className="motion-audit-meta"><span>{item.mode} / {item.sensitivity}</span><span>{item.trigger_count} trigger{item.trigger_count === 1 ? "" : "s"}</span></div>
              </div>
            </article>
          );
        })}
        {!items.length && !loading ? <div className="empty-state">No motion decisions match these filters.</div> : null}
      </div>
      <div className="motion-audit-pagination">
        <button type="button" aria-label="Previous audit page" onClick={() => setPage(Math.max(0, page - 1))} disabled={page <= 0 || loading}><ChevronLeft size={16} /></button>
        <span>{total ? `${page * pageSize + 1}-${Math.min(total, (page + 1) * pageSize)} of ${total}` : "0 entries"}</span>
        <button type="button" aria-label="Next audit page" onClick={() => setPage(Math.min(pageCount - 1, page + 1))} disabled={page >= pageCount - 1 || loading}><ChevronRight size={16} /></button>
      </div>
    </div>
  );
}

export const motionAiSettingLabels = {
  analysis_preset: "Motion analysis method",
  stationary_object_tolerance: "Stationary object policy",
};

export function formatMotionAiValue(setting, value) {
  if (setting === "analysis_preset") return value === "adaptive" ? "Enhanced Motion Analysis (EMA)" : String(value);
  if (setting === "stationary_object_tolerance") return ({ low: "Light", balanced: "Standard", high: "Strong", inherit: "Use global setting" })[value] || String(value);
  return String(value);
}

export function MotionAuditPipeline({ telemetry }) {
  const graphs = telemetry?.graphs && typeof telemetry.graphs === "object" ? telemetry.graphs : null;
  if (!graphs) return null;
  const graphLabels = { qualification: "Frame analysis", observation: "Supporting sources", fusion: "Final decision" };
  return (
    <details className="motion-audit-pipeline">
      <summary>Processing used for this decision</summary>
      <div>
        {Object.entries(graphs).map(([name, graph]) => {
          const configuration = Array.isArray(graph?.configuration) ? graph.configuration : [];
          const timings = graph?.invocation_timings && typeof graph.invocation_timings === "object" ? Object.values(graph.invocation_timings) : [];
          const duration = timings.reduce((total, timing) => total + Number(timing?.duration_ms || 0), 0);
          return (
            <section key={name}>
              <span>{graphLabels[name] || name}</span>
              <strong>{configuration.length} step{configuration.length === 1 ? "" : "s"}{timings.length ? ` · ${duration.toFixed(1)} ms` : " · continuous"}</strong>
              <small>{telemetry.origins?.[name] || "default"} configuration · {configuration.map((stage) => stage.implementation).join(" → ") || "No stage details"}</small>
            </section>
          );
        })}
      </div>
    </details>
  );
}

export function MotionAuditOverlay({ item, items, timeZone, onClose, onSelect }) {
  const modalRef = useModalFocus(onClose);
  const outcome = motionAuditOutcome(item);
  const currentIndex = items.findIndex((candidate) => candidate.id === item.id);
  const [aiAdvice, setAiAdvice] = useState(null);
  const [aiError, setAiError] = useState("");
  const [aiLoading, setAiLoading] = useState(false);
  const [aiApplying, setAiApplying] = useState(false);
  const pipelineTelemetry = item.features?.pipeline_telemetry;

  useEffect(() => {
    setAiAdvice(null);
    setAiError("");
    setAiLoading(false);
    setAiApplying(false);
  }, [item.id]);

  async function analyzeWithAi() {
    if (aiLoading || !item.has_snapshot) return;
    setAiLoading(true);
    setAiError("");
    try {
      const response = await fetch(`/api/motion-audit/${item.id}/ai-analyze`, { method: "POST" });
      if (!response.ok) throw new Error(await response.text());
      setAiAdvice(await response.json());
    } catch (error) {
      setAiError(error.message || "Unable to analyze this audit image.");
    } finally {
      setAiLoading(false);
    }
  }

  async function applyAiChanges() {
    const changes = aiAdvice?.advice?.changes || [];
    if (!changes.length || aiApplying) return;
    if (!window.confirm(`Apply ${changes.length} AI-recommended motion setting${changes.length === 1 ? "" : "s"}? Camera workers will restart.`)) return;
    setAiApplying(true);
    setAiError("");
    try {
      const response = await fetch(`/api/motion-audit/${item.id}/ai-apply`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          changes,
          confirmed: true,
          configuration_fingerprint: aiAdvice?.configuration_fingerprint || "",
          recommendation_proof: aiAdvice?.recommendation_proof || "",
        }),
      });
      if (!response.ok) throw new Error(await response.text());
      const result = await response.json();
      setAiAdvice((current) => ({ ...current, applied: result.applied || [] }));
    } catch (error) {
      setAiError(error.message || "Unable to apply AI recommendations.");
    } finally {
      setAiApplying(false);
    }
  }

  function move(direction) {
    if (currentIndex < 0 || items.length < 2) return;
    onSelect(items[(currentIndex + direction + items.length) % items.length]);
  }

  useEffect(() => {
    function onKeyDown(event) {
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        move(-1);
      }
      if (event.key === "ArrowRight") {
        event.preventDefault();
        move(1);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [currentIndex, item.id, items, onClose, onSelect]);

  return createPortal((
    <div ref={modalRef} className="motion-audit-overlay" role="dialog" aria-modal="true" aria-label="Motion audit image">
      <button className="live-overlay-backdrop" type="button" onClick={onClose} aria-label="Close motion audit image" />
      <section className="motion-audit-overlay-panel">
        <header className="motion-audit-overlay-head">
          <div><h2>{item.camera_id}</h2><time>{formatDateTime(item.created_at, timeZone)}</time></div>
          <div className="overlay-actions">
            <button type="button" className="icon-only" onClick={() => move(-1)} disabled={items.length < 2} aria-label="Previous audit image"><ChevronLeft size={19} /></button>
            <span>{currentIndex + 1} / {items.length}</span>
            <button type="button" className="icon-only" onClick={() => move(1)} disabled={items.length < 2} aria-label="Next audit image"><ChevronRight size={19} /></button>
            <button type="button" className="icon-only" data-modal-initial onClick={onClose} aria-label="Close motion audit image"><X size={19} /></button>
          </div>
        </header>
        <div className="motion-audit-overlay-content">
          <div className="motion-audit-overlay-media">
            {item.has_snapshot
              ? <MotionAuditAnnotatedImage item={item} alt={`${item.camera_id} rejected motion`} interactive />
              : <div className="empty-thumb"><Camera size={42} /><span>Audit image unavailable</span></div>}
          </div>
          <aside className="motion-audit-overlay-details">
            <span className={`motion-audit-outcome ${outcome.className}`}>{outcome.label}</span>
            <div className="motion-audit-overlay-score"><span>{String(item.reason || "rejected").replaceAll("_", " ")}</span><strong>{Number(item.score || 0).toFixed(3)} / {Number(item.threshold || 0).toFixed(3)}</strong></div>
            {item.interpretation?.explanation ? <div className="motion-analysis-warning">{item.interpretation.explanation}</div> : null}
            <div className="motion-audit-meter"><i style={{ width: `${Math.max(0, Math.min(100, Number(item.score || 0) * 100))}%` }} /><b style={{ left: `${Math.max(0, Math.min(100, Number(item.threshold || 0) * 100))}%` }} /></div>
            <dl>
              {Object.entries(item.features || {}).filter(([, value]) => typeof value === "number").map(([name, value]) => <div key={name}><dt>{name.replaceAll("_", " ")}</dt><dd>{Number(value).toFixed(3)}</dd></div>)}
              <div><dt>Mode</dt><dd>{item.mode}</dd></div>
              <div><dt>Sensitivity</dt><dd>{item.sensitivity}</dd></div>
              <div><dt>Triggers</dt><dd>{item.trigger_count}</dd></div>
            </dl>
            <MotionAuditPipeline telemetry={pipelineTelemetry} />
            <div className="motion-audit-ai">
              <div className="motion-audit-ai-head">
                <strong><Sparkles size={15} /> AI Advisor</strong>
                <button type="button" onClick={analyzeWithAi} disabled={aiLoading || aiApplying || !item.has_snapshot} title={item.has_snapshot ? "Analyze this motion decision audit image" : "AI analysis requires a saved audit image"}><Sparkles size={15} /> {aiLoading ? "Analyzing..." : "Analyze"}</button>
              </div>
              {!item.has_snapshot ? <span className="motion-audit-ai-none">AI analysis requires an audit image. This older audit was not sampled or has passed the retention limit.</span> : null}
              {aiError ? <div className="motion-audit-ai-error">{aiError}</div> : null}
              {aiAdvice?.advice ? (
                <div className="motion-audit-ai-result">
                  {aiAdvice.motion_paradigm ? <small>Analyzed as {String(aiAdvice.motion_paradigm.paradigm || "motion decision").replaceAll("_", " ")} · {String(aiAdvice.motion_paradigm.automatic_trigger?.source || "configured trigger").replaceAll("_", " ")}</small> : null}
                  <div className="motion-audit-ai-verdict"><span>{aiAdvice.advice.verdict.replaceAll("_", " ")}</span><strong>{Math.round(Number(aiAdvice.advice.confidence || 0) * 100)}%</strong></div>
                  <p>{aiAdvice.advice.summary}</p>
                  {aiAdvice.advice.visible_subjects?.length ? <div className="motion-audit-ai-subjects">{aiAdvice.advice.visible_subjects.map((subject) => <span key={subject}>{subject}</span>)}</div> : null}
                  {aiAdvice.advice.explanation?.length ? <ul>{aiAdvice.advice.explanation.map((line) => <li key={line}>{line}</li>)}</ul> : null}
                  {aiAdvice.advice.changes?.length ? (
                    <>
                      <div className="motion-audit-ai-changes">
                        {aiAdvice.advice.changes.map((change, index) => <div key={`${change.scope}-${change.setting}-${index}`}><strong>{change.scope} · {motionAiSettingLabels[change.setting] || change.setting.replaceAll("_", " ")}</strong><code>{formatMotionAiValue(change.setting, change.value)}</code><small>{change.reason}</small></div>)}
                      </div>
                      <button type="button" className="primary" onClick={applyAiChanges} disabled={aiApplying || Boolean(aiAdvice.applied)}><Save size={15} /> {aiAdvice.applied ? "Applied" : aiApplying ? "Applying..." : "Apply Recommendations"}</button>
                    </>
                  ) : <span className="motion-audit-ai-none">No setting changes recommended.</span>}
                </div>
              ) : null}
            </div>
          </aside>
        </div>
      </section>
    </div>
  ), document.body);
}

export function ProbeResult({ probe }) {
  if (probe.loading) return <div className="probe-result">Probing camera capabilities...</div>;
  if (probe.error) return <div className="probe-result"><strong>Auto-detection failed</strong><span>{probe.error}</span></div>;
  return (
    <div className="probe-result">
      <strong>Auto-detection</strong>
      <span>ONVIF: {probe.onvif?.reachable ? `reachable on ${probe.onvif.port}` : "not reachable"}</span>
      {probe.onvif?.capabilities ? <span>Capabilities: {Object.entries(probe.onvif.capabilities).filter(([, value]) => value).map(([key]) => key).join(", ") || "none reported"}</span> : null}
      {probe.onvif?.error ? <span>{probe.onvif.error}</span> : null}
    </div>
  );
}

export function MotionAiReviewPanel({ cameras, runtimeStatus = [], advisorEnabled, cameraId: controlledCameraId = "", onCameraIdChange = null, hideScopePicker = false }) {
  const [cameraId, setCameraId] = useState(controlledCameraId || cameras[0]?.id || "");
  const [hours, setHours] = useState(24);
  const [imageLimit, setImageLimit] = useState(12);
  const [evaluationHours, setEvaluationHours] = useState(24);
  const [review, setReview] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [applying, setApplying] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  function selectCamera(nextCameraId) {
    setCameraId(nextCameraId);
    onCameraIdChange?.(nextCameraId);
  }

  useEffect(() => {
    if (controlledCameraId && cameras.some((camera) => camera.id === controlledCameraId)) {
      if (controlledCameraId !== cameraId) setCameraId(controlledCameraId);
      return;
    }
    if (!cameraId && cameras[0]?.id) {
      setCameraId(cameras[0].id);
      onCameraIdChange?.(cameras[0].id);
      return;
    }
    if (cameraId && !cameras.some((camera) => camera.id === cameraId)) {
      const next = cameras[0]?.id || "";
      setCameraId(next);
      onCameraIdChange?.(next);
    }
  }, [cameraId, cameras, controlledCameraId, onCameraIdChange]);

  async function loadReview(selectedCameraId, quiet = false) {
    if (!selectedCameraId) return;
    if (!quiet) setLoading(true);
    try {
      const response = await fetch(`/api/motion-ai-reviews/latest?camera_id=${encodeURIComponent(selectedCameraId)}`);
      if (!response.ok) throw new Error(await response.text());
      setReview(await response.json());
      setError("");
    } catch (loadError) {
      setError(loadError.message || "Unable to load the latest camera review.");
    } finally {
      if (!quiet) setLoading(false);
    }
  }

  async function loadEvaluation(selectedCameraId, quiet = false) {
    if (!selectedCameraId) return;
    try {
      const response = await fetch(`/api/camera-intelligence/evaluations/latest?camera_id=${encodeURIComponent(selectedCameraId)}`);
      if (!response.ok) throw new Error(await response.text());
      setEvaluation(await response.json());
    } catch (loadError) {
      if (!quiet) setError(loadError.message || "Unable to load the latest effectiveness check.");
    }
  }

  useEffect(() => {
    setReview(null);
    setEvaluation(null);
    setError("");
    setNotice("");
    void loadReview(cameraId);
    void loadEvaluation(cameraId);
  }, [cameraId]);

  useVisiblePolling(
    () => loadEvaluation(cameraId, true),
    2000,
    Boolean(cameraId) && evaluation?.status === "reviewing",
    { immediate: false, restartKey: cameraId || "" },
  );

  useVisiblePolling(
    () => loadReview(cameraId, true),
    2000,
    Boolean(cameraId) && ["queued", "running"].includes(review?.status),
    { immediate: false, restartKey: cameraId || "" },
  );

  async function startReview() {
    if (!cameraId || loading) return;
    setLoading(true);
    setError("");
    setNotice("");
    try {
      const response = await fetch("/api/motion-ai-reviews", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ camera_id: cameraId, hours, record_limit: 100, image_limit: imageLimit }),
      });
      if (!response.ok) throw new Error(await response.text());
      setReview(await response.json());
    } catch (startError) {
      setError(startError.message || "Unable to start the camera review.");
    } finally {
      setLoading(false);
    }
  }

  async function applyRecommendations() {
    const recommendations = report.recommendations || [];
    if (!review?.id || !recommendations.length || applying) return;
    const cameraName = selectedCamera?.name || cameraId;
    if (!window.confirm(`Apply ${recommendations.length} reviewed setting change${recommendations.length === 1 ? "" : "s"} to ${cameraName}? SurvNG will validate the changes and reload only the affected camera services.`)) return;
    setApplying(true);
    setError("");
    setNotice("");
    try {
      const response = await fetch(`/api/motion-ai-reviews/${review.id}/apply`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          confirmed: true,
          configuration_fingerprint: report.configuration_fingerprint,
          evaluation_hours: evaluationHours,
          changes: recommendations.map((recommendation) => ({
            scope: "camera",
            setting: recommendation.setting,
            value: recommendation.proposed ?? recommendation.value,
            reason: recommendation.reasons?.[0] || recommendation.reason || "Repeated review evidence supports this change.",
          })),
        }),
      });
      if (!response.ok) throw new Error(await response.text());
      const applied = await response.json();
      setEvaluation(applied.effectiveness_evaluation || null);
      setReview((current) => current ? ({ ...current, result: { ...(current.result || {}), can_apply: false } }) : current);
      setNotice("Recommended camera settings were applied successfully.");
    } catch (applyError) {
      setError(applyError.message || "Unable to apply the reviewed settings.");
    } finally {
      setApplying(false);
    }
  }

  async function runFollowup() {
    if (!evaluation?.id || evaluation.status !== "ready" || loading) return;
    setLoading(true);
    setError("");
    setNotice("");
    try {
      const response = await fetch(`/api/camera-intelligence/evaluations/${evaluation.id}/follow-up`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_limit: imageLimit }),
      });
      if (!response.ok) throw new Error(await response.text());
      setEvaluation(await response.json());
    } catch (followupError) {
      setError(followupError.message || "Unable to start the effectiveness check.");
    } finally {
      setLoading(false);
    }
  }

  const running = ["queued", "running"].includes(review?.status);
  const report = review?.result || {};
  const completedWork = Number(review?.analyzed || 0) + Number(review?.failed || 0);
  const selectedCamera = cameras.find((camera) => camera.id === cameraId);
  const isCameraIntelligence = report.review_type === "camera_intelligence";
  const verdictLabels = {
    consistent: "Looks correct",
    likely_miss: "Likely missed subject",
    likely_false_alarm: "Likely nuisance alert",
    likely_misclassification: "Likely wrong label",
    uncertain: "Uncertain",
  };
  const categoryLabels = {
    possible_miss: "Possible miss",
    visual_backup: "Visual backup",
    motion_filtered: "Filtered motion",
    motion_only_incident: "Motion-only incident",
    recognized_incident: "Recognized incident",
    other: "Other",
  };

  return (
    <div className="sub-panel motion-ai-review-panel">
      <h3>Camera Intelligence</h3>
      <p className="settings-help">Review how one camera has performed across recent incidents and motion decisions. SurvNG deliberately samples successes, possible misses, visual rescues, and filtered motion, then recommends a change only when multiple images support it. Nothing is applied automatically.</p>
      <div className="field-row motion-ai-review-controls">
        {hideScopePicker ? null : (
          <CameraScopePicker
            cameras={cameras}
            runtimeStatus={runtimeStatus}
            value={cameraId}
            onChange={selectCamera}
            ariaLabel="Camera Advisor camera"
            disabled={running}
          />
        )}
        <label>Review period<select value={hours} onChange={(event) => setHours(Number(event.target.value))} disabled={running}>
          <option value={24}>Last 24 hours</option>
          <option value={72}>Last 3 days</option>
          <option value={168}>Last 7 days</option>
        </select></label>
        <label>Images to inspect<select value={imageLimit} onChange={(event) => setImageLimit(Number(event.target.value))} disabled={running}>
          <option value={8}>8 · Lower cost</option>
          <option value={12}>12 · Balanced</option>
          <option value={16}>16 · More evidence</option>
          <option value={24}>24 · Most thorough</option>
        </select></label>
        <button type="button" className="primary" onClick={startReview} disabled={!cameraId || !advisorEnabled || running || loading}>
          {running ? <RefreshCcw className="spin" size={16} /> : <Sparkles size={16} />}
          {running ? "Reviewing..." : "Review camera"}
        </button>
        <button type="button" onClick={() => { void loadReview(cameraId); void loadEvaluation(cameraId, true); }} disabled={!cameraId || loading}><RefreshCcw className={loading ? "spin" : ""} size={16} /> Refresh</button>
      </div>
      {!advisorEnabled ? <div className="save-status motion-audit-error">Enable and save AI analysis under Object Detection before running a review.</div> : null}
      <div className="probe-result">
        <strong>What this uses</strong>
        <span>Up to 100 recent records for {selectedCamera?.name || cameraId || "the selected camera"}, balanced across different outcomes instead of simply choosing the newest images.</span>
        <span>At most {imageLimit} images are sent for analysis. Each image is one provider request; missing or expired images are skipped.</span>
      </div>
      {error ? <div className="save-status motion-audit-error">{error}</div> : null}
      {notice ? <div className="save-status">{notice}</div> : null}
      {review?.status && review.status !== "never" ? (
        <section className="motion-ai-review-report">
          <header>
            <div><strong>{selectedCamera?.name || review.camera_id}</strong><span>Review #{review.id} · {String(review.status).replaceAll("_", " ")}</span></div>
            <time>{review.updated_at ? formatDateTime(review.updated_at) : ""}</time>
          </header>
          {running ? (
            <div className="motion-ai-review-progress">
              <div><i style={{ width: `${Math.min(100, Number(review.images_available || 0) ? completedWork / Number(review.images_available) * 100 : 0)}%` }} /></div>
              <span>{review.analyzed || 0} reviewed · {review.failed || 0} unavailable · {review.images_available || 0} selected images from {review.audits_considered || 0} recent records</span>
            </div>
          ) : null}
          {review.error ? <div className="motion-runtime-warning">{review.error}</div> : null}
          {review.status === "completed" ? (
            <>
              <p>{report.summary}</p>
              {!isCameraIntelligence && report.review_context?.motion_paradigm ? (
                <div className="probe-result">
                  <strong>Configuration analyzed</strong>
                  <span>{report.review_context.motion_paradigm.paradigm === "camera_triggered" ? "ONVIF-triggered" : report.review_context.motion_paradigm.paradigm === "camera_triggered_with_visual_backup" ? "ONVIF + EMA backup" : report.review_context.motion_paradigm.paradigm === "visual_triggered" ? "EMA-triggered" : "Legacy trigger mode"} · {report.review_context.effective_settings?.incident_eligibility_policy === "zones_only" ? "Zones only" : "Zones + Full Frame"}</span>
                </div>
              ) : null}
              <div className="motion-ai-review-stats">
                <span><strong>{report.verdict_counts?.likely_miss ?? report.verdict_counts?.real_motion ?? 0}</strong>{isCameraIntelligence ? " likely missed" : " likely real motion"}</span>
                <span><strong>{report.verdict_counts?.likely_false_alarm ?? report.verdict_counts?.noise ?? 0}</strong> likely nuisance</span>
                {isCameraIntelligence ? <span><strong>{report.verdict_counts?.consistent || 0}</strong> looks correct</span> : null}
                <span><strong>{report.verdict_counts?.uncertain || 0}</strong> uncertain</span>
              </div>
              {isCameraIntelligence && report.samples?.length ? (
                <>
                  <h4>Images reviewed</h4>
                  <div className="camera-intelligence-samples">
                    {report.samples.map((sample) => (
                      <article key={`${sample.kind}-${sample.record_id}`}>
                        {sample.image_url?.startsWith("/api/") ? <img src={mediaUrl(sample.image_url)} alt={`${selectedCamera?.name || cameraId} review sample`} loading="lazy" /> : <div className="camera-intelligence-image-missing">Image unavailable</div>}
                        <div><strong>{verdictLabels[sample.verdict] || String(sample.verdict || "Uncertain").replaceAll("_", " ")}</strong><span>{categoryLabels[sample.category] || String(sample.category || "Other").replaceAll("_", " ")}</span></div>
                        <p>{sample.summary}</p>
                      </article>
                    ))}
                  </div>
                </>
              ) : null}
              <h4>Suggested camera changes</h4>
              {report.recommendations?.length ? (
                <div className="motion-ai-review-recommendations">
                  {report.recommendations.map((recommendation) => (
                    <article key={`${recommendation.setting}-${JSON.stringify(recommendation.value)}`}>
                      <div><strong>{motionAiSettingLabels[recommendation.setting] || recommendation.setting.replaceAll("_", " ")}</strong><code>{(recommendation.current ?? recommendation.current_value) == null ? "Current unavailable" : formatMotionAiValue(recommendation.setting, recommendation.current ?? recommendation.current_value)} → {formatMotionAiValue(recommendation.setting, recommendation.proposed ?? recommendation.value)}</code></div>
                      <span>Supported by {recommendation.support_count} analyzed image{recommendation.support_count === 1 ? "" : "s"} · {Math.round(Number(recommendation.average_confidence || 0) * 100)}% average confidence</span>
                      <p>{recommendation.reasons?.[0]}</p>
                      {recommendation.evidence_audit_ids?.length ? <small>Evidence: audit {recommendation.evidence_audit_ids.join(", ")}</small> : null}
                    </article>
                  ))}
                </div>
              ) : <span>No setting change was recommended consistently enough across the analyzed images.</span>}
              {isCameraIntelligence && report.recommendations?.length && !notice ? <div className="camera-intelligence-apply-row"><label>Check results after<select value={evaluationHours} onChange={(event) => setEvaluationHours(Number(event.target.value))}><option value={24}>24 hours</option><option value={168}>7 days</option></select></label><button type="button" className="primary camera-intelligence-apply" onClick={applyRecommendations} disabled={!report.can_apply || applying}>{applying ? <RefreshCcw className="spin" size={16} /> : <Check size={16} />}{report.can_apply ? (applying ? "Applying..." : "Review and apply suggestions") : "Applying AI suggestions is disabled"}</button></div> : null}
            </>
          ) : null}
        </section>
      ) : <div className="probe-result"><strong>No review yet</strong><span>Choose a camera and run its first manual review.</span></div>}
      {evaluation?.status && evaluation.status !== "never" ? (
        <section className={`camera-intelligence-effectiveness ${evaluation.comparison?.outcome || evaluation.status}`}>
          <header><div><strong>Did the change help?</strong><span>Applied {formatDateTime(evaluation.applied_at)}</span></div><span>{String(evaluation.status).replaceAll("_", " ")}</span></header>
          {evaluation.applied_changes?.length ? <div className="camera-intelligence-applied">{evaluation.applied_changes.map((change) => <span key={change.setting}>{motionAiSettingLabels[change.setting] || change.setting.replaceAll("_", " ")}: {formatMotionAiValue(change.setting, change.current)} → {formatMotionAiValue(change.setting, change.proposed)}</span>)}</div> : null}
          {evaluation.status === "collecting" ? <p>SurvNG is gathering normal camera activity. The follow-up becomes available {evaluation.ready_at ? formatDateTime(evaluation.ready_at) : "after the selected period"}.</p> : null}
          {evaluation.status === "ready" ? <><p>Enough time has passed to compare a new balanced image sample against the review made before the change.</p><button type="button" className="primary" onClick={runFollowup} disabled={loading}><Sparkles size={16} />Run follow-up review</button></> : null}
          {evaluation.status === "reviewing" ? <p><RefreshCcw className="spin" size={16} /> Reviewing post-change camera activity…</p> : null}
          {evaluation.status === "completed" && evaluation.comparison ? <><p className="camera-intelligence-outcome">{evaluation.comparison.summary}</p><div className="camera-intelligence-comparison">{evaluation.comparison.metrics?.map((metric) => <article key={metric.key}><span>{metric.label}</span><strong>{Math.round(Number(metric.before_rate || 0) * 100)}% → {Math.round(Number(metric.after_rate || 0) * 100)}%</strong><small>{Number(metric.change_points || 0) > 0 ? "+" : ""}{metric.change_points} percentage points</small></article>)}</div><small>{evaluation.comparison.caution}</small></> : null}
          {evaluation.error ? <div className="motion-runtime-warning">{evaluation.error}</div> : null}
        </section>
      ) : null}
    </div>
  );
}

export function RetentionSummary({ status }) {
  const plan = status.plan || {};
  const storage = plan.storage || {};
  const indexed = plan.indexed || {};
  const reclaim = plan.reclaim || {};
  const lastRun = status.last_run || null;
  const cameraStorageRows = plan.per_camera_storage || [];
  const snapshots = plan.snapshots || {};
  const headroom = indexed.days_to_minimum_free;
  return (
    <div className="retention-summary">
      <div className={`retention-alert ${storage.emergency ? "critical" : Number(reclaim.planned_bytes || 0) > 0 ? "warning" : "healthy"}`}>
        {storage.emergency || Number(reclaim.planned_bytes || 0) > 0 ? <CircleAlert size={20} /> : <CircleDot size={20} />}
        <div>
          <strong>{storage.emergency ? "Storage is critically low" : Number(reclaim.planned_bytes || 0) > 0 ? `${formatBytes(reclaim.planned_bytes)} eligible for cleanup` : "Storage is within the configured policy"}</strong>
          <span>{Number(storage.free_percent || 0).toFixed(1)}% free · {formatBytes(storage.free_bytes)} available{headroom == null ? "" : ` · approximately ${headroom} days to the cleanup threshold`}</span>
        </div>
      </div>
      <div className="retention-metrics">
        <article><span>Continuous recordings</span><strong>{formatBytes(indexed.bytes)}</strong><small>{Number(indexed.file_count || 0).toLocaleString()} indexed segments</small></article>
        <article><span>Incident snapshots</span><strong>{formatBytes(snapshots.bytes)}</strong><small>{Number(snapshots.file_count || 0).toLocaleString()} indexed images</small></article>
        <article><span>Current growth</span><strong>{formatBytes(indexed.bytes_per_day)}/day</strong><small>Estimated from indexed history</small></article>
        <article><span>Age-expired</span><strong>{formatBytes(reclaim.expired_bytes)}</strong><small>{Number(reclaim.expired_files || 0).toLocaleString()} segments</small></article>
        <article><span>Capacity pressure</span><strong>{formatBytes(Math.max(Number(reclaim.quota_bytes || 0), Number(reclaim.free_space_bytes || 0)))}</strong><small>{(reclaim.reasons || []).length ? reclaim.reasons.join(" + ").replaceAll("_", " ") : "None"}</small></article>
      </div>
      {lastRun ? <div className="retention-last-run"><strong>Last cleanup</strong><span>{Number(lastRun.deleted_files || 0).toLocaleString()} files · {formatBytes(lastRun.deleted_bytes)} reclaimed{lastRun.failed_files ? ` · ${lastRun.failed_files} failed` : ""}</span></div> : null}
      {cameraStorageRows.length ? <details className="retention-camera-details"><summary>Per-camera storage</summary><div className="retention-camera-table retention-camera-storage-table">
        <div className="heading"><span>Camera</span><span>Used-Recordings</span><span>Used-Snapshots</span><span>Recording files</span><span>Snapshot files</span></div>
        {cameraStorageRows.map((row) => <div key={row.camera_id}><strong>{row.camera_id}</strong><span>{formatBytes(row.recording_bytes)}</span><span>{formatBytes(row.snapshot_bytes)}</span><span>{Number(row.recording_files || 0).toLocaleString()}</span><span>{Number(row.snapshot_files || 0).toLocaleString()}</span></div>)}
      </div></details> : null}
    </div>
  );
}

export function GeneralSettings({ config, updateConfig, commitImmediateConfig, onTokenSecretVisibleChange, onOpenApiTokens = null, timeZone, setTimeZone, theme, setTheme, accelerator, detectorModels, recordingCache, retentionStatus, retentionError, runRetention, mqttStatus, detectorStatus, motionCatalog, runtimeStatus = [], advisorCameraId = "", onAdvisorCameraIdChange = null, section, detectionSection = "object", onDetectionSectionChange = null }) {
  const [liveOrderReset, setLiveOrderReset] = useState(false);
  const [serverRestart, setServerRestart] = useState({ state: "idle", text: "" });
  const [productUpdate, setProductUpdate] = useState(null);
  const [productUpdateBusy, setProductUpdateBusy] = useState(false);
  const [productUpdateError, setProductUpdateError] = useState("");
  const [productUpdateBranch, setProductUpdateBranch] = useState("");
  const [apiTokenDraft, setApiTokenDraft] = useState({ id: "", name: "", scopes: ["read"] });
  const [apiTokenSecret, setApiTokenSecret] = useState("");
  const [apiTokenBusy, setApiTokenBusy] = useState(false);
  const [apiTokenError, setApiTokenError] = useState("");
  const activeModelPath = config.detector?.model_path || config.detector?.model_xml || "";
  const setDetectionSection = onDetectionSectionChange || (() => {});
  const [serverPreferencesSection, setServerPreferencesSection] = useState("general");
  const [storageSection, setStorageSection] = useState("locations");
  const [apiSection, setApiSection] = useState("tokens");
  const mediaLocations = config.media_storage?.locations || [];
  const reidStatus = detectorStatus?.reid || null;
  const cameraTransitionRoutes = config.detector?.tracking?.camera_transition_routes || [];
  const routeCameras = config.cameras || [];
  const updateCameraRoute = (index, field, value) => updateConfig(
    ["detector", "tracking", "camera_transition_routes"],
    cameraTransitionRoutes.map((route, routeIndex) => routeIndex === index ? { ...route, [field]: value } : route),
  );
  const addCameraRoute = () => {
    if (routeCameras.length < 2) return;
    const existing = new Set(cameraTransitionRoutes.map((route) => `${route.from_camera}->${route.to_camera}`));
    let pair = null;
    for (const from of routeCameras) {
      for (const to of routeCameras) {
        if (from.id !== to.id && !existing.has(`${from.id}->${to.id}`)) {
          pair = [from.id, to.id];
          break;
        }
      }
      if (pair) break;
    }
    if (!pair) return;
    updateConfig(["detector", "tracking", "camera_transition_routes"], [
      ...cameraTransitionRoutes,
      { from_camera: pair[0], to_camera: pair[1], min_seconds: 0, max_seconds: 30, bidirectional: false, enabled: true, name: "" },
    ]);
  };
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
  const ffmpegAcceleration = accelerator?.ffmpeg_hardware_acceleration || {};
  const vaapi = ffmpegAcceleration.vaapi || {};
  const qsv = ffmpegAcceleration.qsv || {};
  const vaapiLabel = vaapi.available
    ? `VAAPI available (${(vaapi.encoders || []).join(", ") || "encoders detected"})`
    : vaapi.listed
      ? "VAAPI listed by FFmpeg but runtime init failed"
      : "VAAPI not available to FFmpeg";
  const qsvLabel = qsv.available
    ? `Intel QSV available (${(qsv.encoders || []).join(", ") || "encoders detected"})`
    : qsv.listed
      ? "Intel QSV listed by FFmpeg but runtime init failed"
      : "Intel QSV not available to FFmpeg";
  const activeModel = findDetectorModel(detectorModels, activeModelPath);
  const eventClassConfirmations = config.detector?.event_class_confirmation_frames || {};
  const eventClassConfidences = config.detector?.event_class_confidence_thresholds || {};
  const eventConfirmationClasses = [...new Set([
    ...(activeModel?.classes || []),
    ...Object.keys(eventClassConfirmations),
    ...Object.keys(eventClassConfidences),
  ].map((label) => String(label).trim().toLowerCase()).filter(Boolean))]
    .sort((left, right) => left.localeCompare(right));
  const trackingExcludedLabels = config.detector?.tracking?.excluded_labels || ["face"];
  const trackingClassOptions = [...new Set([
    ...(activeModel?.classes || []),
    ...trackingExcludedLabels,
  ].map((label) => String(label).trim().toLowerCase()).filter(Boolean))]
    .sort((left, right) => left.localeCompare(right));
  const refinementStagePresets = {
    full: [[-1, -0.5, 0, 0.5, 1], [1.5, 2, 2.5, 3], [3.5, 4, 4.5], [8, 8.5], [12, 12.5]],
    compact: [[-0.5, 0, 0.5], [4, 4.5], [8, 8.5]],
    tight: [[-0.5, 0, 0.5]],
  };
  const refinementStagesKey = JSON.stringify(config.detector?.event_refinement_stages || refinementStagePresets.full);
  const refinementStagePreset = Object.entries(refinementStagePresets).find(([, stages]) => JSON.stringify(stages) === refinementStagesKey)?.[0] || "custom";

  function selectOpenvinoModel(path) {
    updateConfig(["detector", "model_path"], path);
    updateConfig(["detector", "model_xml"], "");
    if (path.endsWith(".xml")) updateConfig(["detector", "labels_path"], "");
  }

  function setEventClassConfirmation(label, value) {
    const next = { ...eventClassConfirmations };
    if (value === "") delete next[label];
    else next[label] = Number(value);
    updateConfig(["detector", "event_class_confirmation_frames"], next);
  }

  function setEventClassConfidence(label, value) {
    const next = { ...eventClassConfidences };
    if (value === "") delete next[label];
    else next[label] = Number(value);
    updateConfig(["detector", "event_class_confidence_thresholds"], next);
  }

  function resetLiveCameraOrder() {
    localStorage.removeItem("survng.liveCameraOrder.v1");
    setLiveOrderReset(true);
  }

  function updateMediaLocation(index, field, value) {
    updateConfig(["media_storage", "locations"], mediaLocations.map((item, itemIndex) => (
      itemIndex === index ? { ...item, [field]: value } : item
    )));
  }

  function toggleMediaRole(index, role, enabled) {
    const current = mediaLocations[index]?.roles || [];
    const roles = enabled ? [...new Set([...current, role])] : current.filter((item) => item !== role);
    if (!roles.length) return;
    updateMediaLocation(index, "roles", roles);
  }

  function addMediaLocation() {
    const index = mediaLocations.length + 1;
    updateConfig(["media_storage", "locations"], [...mediaLocations, {
      id: index === 1 ? "primary" : `media-${index}`,
      name: index === 1 ? "Primary media" : `Media ${index}`,
      path: index === 1 ? (config.storage_dir || "") : "",
      enabled: true,
      roles: ["recordings", "snapshots", "motion_audits", "clips", "exports"],
      reserve_percent: 15,
      priority: 100,
      require_mount: false,
    }]);
  }

  async function restartServer() {
    if (serverRestart.state === "requesting" || serverRestart.state === "waiting") return;
    if (!window.confirm("Restart SurvNG now? Live view, recording playback, and detection will be briefly unavailable.")) return;
    setServerRestart({ state: "requesting", text: "Requesting restart..." });
    try {
      const response = await fetch("/api/system/restart", { method: "POST" });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "Unable to restart SurvNG.");
      const previousInstance = String(payload.instance_id || "");
      setServerRestart({ state: "waiting", text: "Restarting SurvNG..." });
      const deadline = Date.now() + 90_000;
      while (Date.now() < deadline) {
        await new Promise((resolve) => window.setTimeout(resolve, 1_500));
        try {
          const statusResponse = await fetch(`/api/system/status?restart_check=${Date.now()}`, { cache: "no-store" });
          if (!statusResponse.ok) continue;
          const status = await statusResponse.json();
          if (previousInstance && String(status.instance_id || "") === previousInstance) continue;
          window.location.reload();
          return;
        } catch {
          // The expected unavailable window while the service restarts.
        }
      }
      setServerRestart({ state: "error", text: "Restart is taking longer than expected. Refresh this page shortly." });
    } catch (error) {
      setServerRestart({ state: "error", text: error.message || "Unable to restart SurvNG." });
    }
  }

  async function loadProductUpdate(refreshRemote = false, branch = productUpdateBranch) {
    setProductUpdateError("");
    try {
      const params = new URLSearchParams({
        refresh_remote: refreshRemote ? "true" : "false",
      });
      const selectedBranch = String(branch || "").trim();
      if (selectedBranch) params.set("branch", selectedBranch);
      const response = await fetch(`/api/system/update?${params.toString()}`, { cache: "no-store" });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "Unable to load update status.");
      setProductUpdate(payload);
      const nextBranch = String(payload.target_branch || payload.branch || selectedBranch || "").trim();
      if (nextBranch && nextBranch !== productUpdateBranch) {
        setProductUpdateBranch(nextBranch);
      }
      return payload;
    } catch (error) {
      setProductUpdateError(error.message || "Unable to load update status.");
      return null;
    }
  }

  async function waitForUpdatedInstance(previousInstance) {
    const deadline = Date.now() + 180_000;
    while (Date.now() < deadline) {
      await new Promise((resolve) => window.setTimeout(resolve, 2_000));
      try {
        const statusResponse = await fetch(`/api/system/status?update_check=${Date.now()}`, { cache: "no-store" });
        if (!statusResponse.ok) continue;
        const status = await statusResponse.json();
        if (previousInstance && String(status.instance_id || "") === previousInstance) continue;
        window.location.reload();
        return true;
      } catch {
        // Expected while SurvNG restarts after applying the update.
      }
    }
    return false;
  }

  async function applyProductUpdate() {
    if (productUpdateBusy) return;
    const pending = Number(productUpdate?.behind_count || 0);
    const targetBranch = String(productUpdateBranch || productUpdate?.target_branch || productUpdate?.branch || "").trim();
    const switching = Boolean(productUpdate?.needs_checkout);
    let confirmText = "Update SurvNG from Git and restart? Live view and detection will be briefly unavailable.";
    if (switching && pending > 0) {
      confirmText = `Switch to ${targetBranch} and apply ${pending} commit${pending === 1 ? "" : "s"} from Git, then restart? Live view and detection will be briefly unavailable.`;
    } else if (switching) {
      confirmText = `Switch SurvNG to branch ${targetBranch} and restart? Live view and detection will be briefly unavailable.`;
    } else if (pending > 0) {
      confirmText = `Update SurvNG with ${pending} commit${pending === 1 ? "" : "s"} from ${targetBranch || "Git"}, then restart? Live view and detection will be briefly unavailable.`;
    }
    if (!window.confirm(confirmText)) return;
    setProductUpdateBusy(true);
    setProductUpdateError("");
    try {
      const previousInstanceResponse = await fetch("/api/system/status", { cache: "no-store" });
      const previousStatus = previousInstanceResponse.ok ? await previousInstanceResponse.json() : {};
      const previousInstance = String(previousStatus.instance_id || "");
      const response = await fetch("/api/system/update", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(targetBranch ? { branch: targetBranch } : {}),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "Unable to start product update.");
      setProductUpdate(payload);
      const deadline = Date.now() + 900_000;
      while (Date.now() < deadline) {
        await new Promise((resolve) => window.setTimeout(resolve, 1_500));
        const latest = await loadProductUpdate(false, targetBranch);
        const jobStatus = latest?.status || latest?.job?.status;
        if (jobStatus === "restarting") {
          const restarted = await waitForUpdatedInstance(previousInstance);
          if (!restarted) setProductUpdateError("Update applied, but SurvNG is taking longer than expected to come back. Refresh this page shortly.");
          return;
        }
        if (jobStatus === "complete") {
          window.location.reload();
          return;
        }
        if (jobStatus === "failed") {
          throw new Error(latest?.job?.error || latest?.message || "Product update failed.");
        }
      }
      setProductUpdateError("Update is taking longer than expected. Check Admin → Logs, then refresh.");
    } catch (error) {
      setProductUpdateError(error.message || "Unable to update SurvNG.");
    } finally {
      setProductUpdateBusy(false);
    }
  }

  useEffect(() => {
    if (section !== "general") return undefined;
    let cancelled = false;
    void loadProductUpdate(false).then((payload) => {
      if (cancelled || !payload) return;
    });
    return () => { cancelled = true; };
  }, [section]);

  function toggleApiTokenScope(scope) {
    setApiTokenDraft((current) => ({
      ...current,
      scopes: current.scopes.includes(scope)
        ? current.scopes.filter((item) => item !== scope)
        : [...current.scopes, scope],
    }));
  }

  async function createApiToken() {
    if (apiTokenBusy || !apiTokenDraft.id.trim() || !apiTokenDraft.name.trim() || !apiTokenDraft.scopes.length) return;
    setApiTokenBusy(true);
    setApiTokenError("");
    setApiTokenSecret("");
    onTokenSecretVisibleChange?.(false);
    try {
      const response = await fetch("/api/config/api-tokens", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          id: apiTokenDraft.id.trim(),
          name: apiTokenDraft.name.trim(),
          scopes: apiTokenDraft.scopes,
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "Could not create API token");
      commitImmediateConfig(["api_auth", "tokens"], [
        ...(config.api_auth?.tokens || []),
        { ...payload.credential, token_hash: "__SURVNG_SECRET_SET__" },
      ]);
      setApiTokenSecret(payload.token || "");
      onTokenSecretVisibleChange?.(Boolean(payload.token));
      setApiTokenDraft({ id: "", name: "", scopes: ["read"] });
    } catch (error) {
      setApiTokenError(error.message || "Could not create API token");
    } finally {
      setApiTokenBusy(false);
    }
  }

  async function deleteApiToken(tokenId) {
    if (apiTokenBusy || !window.confirm(`Delete API token “${tokenId}”? Clients using it will stop working immediately.`)) return;
    setApiTokenBusy(true);
    setApiTokenError("");
    try {
      const response = await fetch(`/api/config/api-tokens/${encodeURIComponent(tokenId)}`, { method: "DELETE" });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "Could not delete API token");
      commitImmediateConfig(["api_auth", "tokens"], (config.api_auth?.tokens || []).filter((token) => token.id !== tokenId));
      if (!payload.enabled) commitImmediateConfig(["api_auth", "enabled"], false);
      setApiTokenSecret("");
      onTokenSecretVisibleChange?.(false);
    } catch (error) {
      setApiTokenError(error.message || "Could not delete API token");
    } finally {
      setApiTokenBusy(false);
    }
  }

  return (
    <div className={`general-settings-content config-form${section === "detection" ? " detection-settings-content" : ""}${["storage", "access", "mqtt"].includes(section) ? " subsection-settings-content" : ""}`}>
      {section === "general" ? (
        <>
        <nav className="admin-section-tabs camera-section-tabs detection-subsection-tabs" aria-label="Server preferences settings">
          {[["general", "General", Cog], ["custom", "Custom", Wrench]].map(([value, label, Icon]) => <button type="button" className={serverPreferencesSection === value ? "active" : ""} aria-pressed={serverPreferencesSection === value} onClick={() => setServerPreferencesSection(value)} key={value}><Icon size={15} />{label}</button>)}
        </nav>
        {serverPreferencesSection === "general" ? (
        <div className="sub-panel general-preferences-panel">
          <h3>Server Preferences</h3>
          <label>Timezone<select value={timeZone} onChange={(event) => setTimeZone(event.target.value)}>
            {US_TIME_ZONES.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select></label>
          <label>Theme<select value={theme} onChange={(event) => setTheme(event.target.value)}>
            {THEMES.map((value) => <option key={value} value={value}>{THEME_META[value].label}</option>)}
          </select></label>
          <p className="admin-action-note">Timezone and theme apply immediately in this browser. Web Base Path is included in Save settings.</p>
          <label>Web Base Path<input value={config.base_path ?? "/survng"} onChange={(event) => updateConfig(["base_path"], event.target.value)} placeholder="/survng" /></label>
          <div className="preference-action general-server-actions">
            <span>
              <strong>Browser &amp; server actions</strong>
              <small>
                These actions apply immediately and are not included in Save settings.
                {productUpdate?.current_short_sha
                  ? ` Running ${productUpdate.current_short_sha}${productUpdate.branch ? ` on ${productUpdate.branch}` : ""}.`
                  : ""}
              </small>
            </span>
            <div className="preference-action-buttons">
              <button type="button" onClick={resetLiveCameraOrder}><RotateCcw size={15} /> Reset Order</button>
              <label className="product-update-branch">
                <span>Update branch</span>
                <select
                  value={productUpdateBranch || productUpdate?.target_branch || productUpdate?.branch || ""}
                  onChange={(event) => {
                    const next = event.target.value;
                    setProductUpdateBranch(next);
                    // Refresh so --single-branch checkouts fetch the selected tip.
                    void loadProductUpdate(true, next);
                  }}
                  disabled={productUpdateBusy || ["running", "restarting"].includes(productUpdate?.status)}
                  title="Git branch used for Check for Updates / Update"
                >
                  {(productUpdate?.branches || []).length
                    ? Array.from(new Set([
                      ...(productUpdateBranch && !productUpdate.branches.includes(productUpdateBranch) ? [productUpdateBranch] : []),
                      ...productUpdate.branches,
                    ])).map((name) => <option key={name} value={name}>{name}</option>)
                    : <option value={productUpdateBranch || productUpdate?.branch || ""}>{productUpdateBranch || productUpdate?.branch || "Check for Updates…"}</option>}
                </select>
              </label>
              <button
                type="button"
                onClick={() => void loadProductUpdate(true, productUpdateBranch)}
                disabled={productUpdateBusy || ["running", "restarting"].includes(productUpdate?.status)}
              >
                <RefreshCcw size={15} />
                Check for Updates
              </button>
              <button
                type="button"
                onClick={() => void applyProductUpdate()}
                disabled={productUpdateBusy || !productUpdate?.can_update}
                title={productUpdate?.message || "Update SurvNG from Git"}
              >
                {productUpdateBusy || ["running", "restarting"].includes(productUpdate?.status)
                  ? <RefreshCcw className="spin" size={15} />
                  : <Download size={15} />}
                {productUpdateBusy || ["running", "restarting"].includes(productUpdate?.status)
                  ? (productUpdate?.job?.phase || "Updating...")
                  : productUpdate?.behind_count
                    ? `Update (${productUpdate.behind_count})`
                    : productUpdate?.needs_checkout
                      ? "Switch Branch"
                      : "Update"}
              </button>
              <button type="button" className="danger" onClick={restartServer} disabled={["requesting", "waiting"].includes(serverRestart.state)}>
                {serverRestart.state === "requesting" || serverRestart.state === "waiting" ? <RefreshCcw className="spin" size={15} /> : <Power size={15} />}
                Restart Server
              </button>
            </div>
          </div>
          {liveOrderReset ? <span className="preference-status"><CircleDot size={13} /> Reset for this browser</span> : null}
          {productUpdate?.message ? <span className="preference-status" role="status">{productUpdate.message}</span> : null}
          {productUpdate?.commits_behind?.length ? (
            <ul className="preference-status product-update-commits">
              {productUpdate.commits_behind.slice(0, 5).map((commit) => (
                <li key={commit.sha}><code>{commit.sha}</code> {commit.subject}</li>
              ))}
            </ul>
          ) : null}
          {productUpdateError ? <span className="preference-status error" role="status">{productUpdateError}</span> : null}
          {serverRestart.text ? <span className={`preference-status ${serverRestart.state === "error" ? "error" : ""}`} role="status">{serverRestart.text}</span> : null}
        </div>
        ) : (
          <section className="detection-settings-card primary">
            <header className="detection-settings-card-head">
              <div className="detection-settings-card-icon"><Wrench size={18} /></div>
              <div><h3>Custom server settings</h3><p>Use a custom FFmpeg build or adjust RTSP transport behavior for camera capture.</p></div>
              <span className="admin-action-kind">Save settings to apply</span>
            </header>
            <div className="detection-field-grid">
              <label className="wide-field">Custom FFmpeg Path<input value={config.ffmpeg_path || ""} onChange={(event) => updateConfig(["ffmpeg_path"], event.target.value)} placeholder="ffmpeg" /><small>Used by SurvNG recording, probes, clips, and recorded-frame decoding. Leave as <code>ffmpeg</code> to use the system path.</small></label>
              <label>Hardware Acceleration<select value={config.hardware_acceleration || "auto"} onChange={(event) => updateConfig(["hardware_acceleration"], event.target.value)}><option value="auto">Auto (VAAPI preferred)</option><option value="vaapi">VAAPI</option><option value="qsv">Intel QSV</option><option value="off">Off</option></select></label>
              <label>RTSP Capture Transport<select value={config.capture_rtsp_transport || "tcp"} onChange={(event) => updateConfig(["capture_rtsp_transport"], event.target.value)}><option value="tcp">TCP (recommended)</option><option value="udp">UDP</option></select><small>TCP prevents packet-loss decoder errors on most camera networks. UDP is available for cameras or relays that require it.</small></label>
            </div>
            <p className="admin-action-note">Changing RTSP transport reloads camera capture workers. Existing custom <code>OPENCV_FFMPEG_CAPTURE_OPTIONS</code> remains an advanced environment override.</p>
          </section>
        )}
        </>
      ) : null}

      {section === "storage" ? (
        <div className="sub-panel subsection-workspace">
          <nav className="admin-section-tabs camera-section-tabs detection-subsection-tabs storage-subsection-tabs" aria-label="Storage and retention settings">
            {[['locations', 'Locations', HardDrive], ['media', 'Media', Monitor], ['retention', 'Retention & Cleanup', Clock3]].map(([value, label, Icon]) => (
              <button type="button" key={value} className={storageSection === value ? "active" : ""} aria-pressed={storageSection === value} onClick={() => setStorageSection(value)}><Icon size={15} />{label}</button>
            ))}
          </nav>
          <div className="subsection-workspace-content">
          <div className="admin-field-grid" hidden={storageSection !== "locations"}>
            <label>Storage Directory<input value={config.storage_dir || ""} onChange={(event) => updateConfig(["storage_dir"], event.target.value)} /></label>
            <label>Metadata Database Directory<input value={config.database_dir || ""} onChange={(event) => updateConfig(["database_dir"], event.target.value)} placeholder="Defaults to storage directory" /></label>
            <label>Recording Index Directory<input value={config.recording_index_dir || ""} onChange={(event) => updateConfig(["recording_index_dir"], event.target.value)} placeholder="Defaults to storage directory" /></label>
          </div>
          <section className="media-storage-settings" hidden={storageSection !== "locations"}>
            <div className="retention-heading">
              <div><h4 className="section-heading-with-icon"><span className="section-heading-icon"><HardDrive size={16} /></span>Media locations</h4><p>Place recordings and related media on one or more independently managed filesystems. At least one location is required.</p></div>
              <button type="button" onClick={addMediaLocation}><Plus size={15} /> Add location</button>
            </div>
            <div className="admin-field-grid">
              <label>Placement<select value={config.media_storage?.placement || "balanced"} onChange={(event) => updateConfig(["media_storage", "placement"], event.target.value)}><option value="balanced">Balanced free space</option><option value="priority">Location priority</option></select></label>
            </div>
            {!mediaLocations.length ? <div className="probe-result"><strong>Media location required</strong><span>Add a primary filesystem path for recordings, snapshots, clips, and exports. Storage Directory remains the portable-path anchor for metadata under that root.</span></div> : null}
            <div className="media-location-list">
              {mediaLocations.map((location, index) => {
                const candidateStatus = retentionStatus?.plan?.storage?.locations?.find((item) => item.id === location.id);
                const normalizePath = (value) => String(value || "").replace(/\/+$/, "");
                const status = candidateStatus && normalizePath(candidateStatus.path) === normalizePath(location.path)
                  ? candidateStatus
                  : null;
                return <article className="media-location-card" key={index}>
                  <header><strong>{location.name || location.id || `Location ${index + 1}`}</strong><span className={`retention-state ${status?.state === "online" ? "running" : status?.state || "idle"}`}>{status?.state || "save to inspect"}</span><button type="button" className="danger compact" aria-label={`Remove ${location.name || location.id}`} disabled={mediaLocations.length <= 1} onClick={() => updateConfig(["media_storage", "locations"], mediaLocations.filter((_item, itemIndex) => itemIndex !== index))}><Trash2 size={14} /></button></header>
                  <div className="admin-field-grid">
                    <label>ID<input value={location.id || ""} onChange={(event) => updateMediaLocation(index, "id", event.target.value)} /></label>
                    <label>Name<input value={location.name || ""} onChange={(event) => updateMediaLocation(index, "name", event.target.value)} /></label>
                    <label className="wide-field">Filesystem path<input value={location.path || ""} onChange={(event) => updateMediaLocation(index, "path", event.target.value)} placeholder="/mnt/survng-media-2" /></label>
                    <label>Reserve free space<input type="number" min="0" max="95" step="1" value={location.reserve_percent ?? 15} onChange={(event) => updateMediaLocation(index, "reserve_percent", Number(event.target.value))} /></label>
                    <label>Priority<input type="number" min="1" max="1000" step="1" value={location.priority ?? 100} onChange={(event) => updateMediaLocation(index, "priority", Number(event.target.value))} /></label>
                  </div>
                  <fieldset className="media-location-roles"><legend>Media roles</legend>{MEDIA_STORAGE_ROLES.map(([role, label]) => <label key={role}><input type="checkbox" checked={(location.roles || []).includes(role)} onChange={(event) => toggleMediaRole(index, role, event.target.checked)} />{label}</label>)}</fieldset>
                  <fieldset className="media-location-flags"><legend>Availability</legend><label><input type="checkbox" checked={location.enabled ?? true} onChange={(event) => updateMediaLocation(index, "enabled", event.target.checked)} />Accept new media</label><label><input type="checkbox" checked={location.require_mount ?? false} onChange={(event) => updateMediaLocation(index, "require_mount", event.target.checked)} />Require a real mount</label></fieldset>
                  {status ? <small>{formatBytes(status.free_bytes)} free of {formatBytes(status.total_bytes)} · {Number(status.free_percent || 0).toFixed(1)}% free</small> : null}
                </article>;
              })}
            </div>
            <p className="retention-protection"><ShieldCheck size={15} /> Databases, indexes, models, and playback cache remain on local SurvNG storage. A required mount is never replaced by its empty mountpoint.</p>
          </section>
          <div className="prewarm-setting" hidden={storageSection !== "media"}>
            <h4>Evidence image storage</h4>
            <div className="field-row">
              <label>Format<select value={config.image_storage?.format || "webp"} onChange={(event) => updateConfig(["image_storage", "format"], event.target.value)}><option value="webp">WebP (recommended)</option><option value="jpeg">JPEG</option></select></label>
              <label>Quality<input type="number" min="1" max="100" step="1" value={config.image_storage?.quality ?? 95} onChange={(event) => updateConfig(["image_storage", "quality"], Number(event.target.value))} /></label>
            </div>
            <p>Controls newly saved incident and motion-audit images. Higher quality preserves more forensic detail but uses more space. Existing images are left unchanged, and live snapshots remain JPEG for compatibility.</p>
          </div>
          <div className="admin-field-grid" hidden={storageSection !== "media"}>
            <label>Recording Segment Seconds<input type="number" min="2" max="300" step="1" value={config.recording_segment_seconds ?? 10} onChange={(event) => updateConfig(["recording_segment_seconds"], Number(event.target.value))} /></label>
            <label>Event Clip Before<input type="number" min="0" max="30" step="1" value={config.event_clip_before_seconds ?? 5} onChange={(event) => updateConfig(["event_clip_before_seconds"], Number(event.target.value))} /></label>
            <label>Event Clip After<input type="number" min="0" max="30" step="1" value={config.event_clip_after_seconds ?? 5} onChange={(event) => updateConfig(["event_clip_after_seconds"], Number(event.target.value))} /></label>
            <label>Incident thumbnail object focus<select className="incident-thumbnail-focus-select" value={config.incident_thumbnail_object_focus || "off"} onChange={(event) => updateConfig(["incident_thumbnail_object_focus"], event.target.value)}>
              <option value="off">Off (full frame)</option>
              <option value="auto">Auto crop to objects</option>
              <option value="button">Manual crop button</option>
            </select></label>
            <label>Object focus zoom<input type="number" min="0.25" max="5.5" step="0.05" value={config.incident_thumbnail_object_focus_zoom ?? 1} disabled={(config.incident_thumbnail_object_focus || "off") === "off"} onChange={(event) => updateConfig(["incident_thumbnail_object_focus_zoom"], Number(event.target.value))} /><small>1 fits objects with padding; below 1 shows more context; above 1 zooms tighter. Compact thumbs are cropped server-side near tile size.</small></label>
            <label>Playback Cache GB<input type="number" min="0.5" max="100" step="0.5" value={config.recording_cache_max_gb ?? 5} onChange={(event) => updateConfig(["recording_cache_max_gb"], Number(event.target.value))} /></label>
            <label>Playback Cache Days<input type="number" min="1" max="90" step="1" value={config.recording_cache_max_days ?? 7} onChange={(event) => updateConfig(["recording_cache_max_days"], Number(event.target.value))} /></label>
          </div>
          <div className="prewarm-setting" hidden={storageSection !== "media"}>
            <label className="check-field"><input type="checkbox" checked={config.incident_thumbnail_annotations ?? false} onChange={(event) => updateConfig(["incident_thumbnail_annotations"], event.target.checked)} /> Show detection boxes on incident thumbnails</label>
            <p>Draws stored object boxes on compact incident thumbnails in Incidents and Live Recent Activity. Object crop/zoom works independently of this overlay.</p>
          </div>
          <div className="prewarm-setting" hidden={storageSection !== "media"}>
            <label className="check-field"><input type="checkbox" checked={config.recording_cache_prewarm ?? true} onChange={(event) => updateConfig(["recording_cache_prewarm"], event.target.checked)} /> Prewarm finalized recordings</label>
            <p>Prepares each completed recording in the background so it opens faster on iPhone and in browsers. It trades additional remux work and playback-cache space for a shorter initial loading delay.</p>
          </div>
          {recordingCache ? <div className="probe-result" hidden={storageSection !== "media"}><strong>Playback Cache</strong><span>{formatBytes(recordingCache.bytes)} used across {recordingCache.entries} fragments</span><span>{formatBytes(recordingCache.max_bytes)} limit, {recordingCache.max_days} day maximum age</span><span>{recordingCache.metrics?.playback_hits || 0} hits / {recordingCache.metrics?.playback_misses || 0} misses, {recordingCache.metrics?.playback_avg_remux_ms || 0} ms average remux</span></div> : null}
          <div className="retention-settings" hidden={storageSection !== "retention"}>
            <div className="retention-heading">
              <div><h4 className="section-heading-with-icon"><span className="section-heading-icon"><Clock3 size={16} /></span>Media retention</h4><p>Daily recording and incident-snapshot planning with lightweight cleanup checks every 15 minutes.</p></div>
              <span className={`retention-state ${retentionStatus?.state || "starting"}`}>
                {String(retentionStatus?.state || "calculating").replaceAll("_", " ")}
                {retentionStatus?.progress && Number(retentionStatus.progress.initial_bytes || 0) > 0
                  ? ` · ${Number(retentionStatus.progress.percent || 0).toFixed(1)}% · ${retentionStatus.progress.eta_seconds == null ? "calculating" : `~${formatCompactDuration(retentionStatus.progress.eta_seconds)} left`}`
                  : ""}
              </span>
            </div>
            <div className="retention-fields">
              <label className="check-field"><input type="checkbox" checked={config.retention?.enabled ?? true} onChange={(event) => updateConfig(["retention", "enabled"], event.target.checked)} /> Monitor storage retention</label>
              <label className="check-field"><input type="checkbox" checked={config.retention?.automatic_cleanup ?? false} onChange={(event) => updateConfig(["retention", "automatic_cleanup"], event.target.checked)} /> Automatically remove expired recordings and snapshots</label>
              <label>SurvNG storage limit<input type="number" min="0.1" max="1000" step="0.5" value={config.retention?.storage_limit_tb ?? 13} onChange={(event) => updateConfig(["retention", "storage_limit_tb"], Number(event.target.value))} /><small>TiB allocated to indexed continuous recordings.</small></label>
              <label>Main stream history<input type="number" min="1" max="3650" step="1" value={config.retention?.main_days ?? 7} onChange={(event) => updateConfig(["retention", "main_days"], Number(event.target.value))} /><small>Days of high-resolution continuous video.</small></label>
              <label>Substream history<input type="number" min="1" max="3650" step="1" value={config.retention?.live_days ?? 21} onChange={(event) => updateConfig(["retention", "live_days"], Number(event.target.value))} /><small>Days of lower-bandwidth continuous video.</small></label>
              <label>Incident snapshot history<input type="number" min="1" max="3650" step="1" value={config.retention?.snapshot_days ?? 1095} onChange={(event) => updateConfig(["retention", "snapshot_days"], Number(event.target.value))} /><small>Days to keep clean incident evidence images. Default: 1,095 days.</small></label>
              <label>Start cleanup below<input type="number" min="1" max="95" step="1" value={config.retention?.minimum_free_percent ?? 15} onChange={(event) => updateConfig(["retention", "minimum_free_percent"], Number(event.target.value))} /><small>Percent free on the entire storage filesystem.</small></label>
              <label>Clean back to<input type="number" min="2" max="99" step="1" value={config.retention?.target_free_percent ?? 20} onChange={(event) => updateConfig(["retention", "target_free_percent"], Number(event.target.value))} /><small>Higher than the start threshold to prevent repeated cycling.</small></label>
              <label>Emergency threshold<input type="number" min="0.5" max="50" step="0.5" value={config.retention?.emergency_free_percent ?? 5} onChange={(event) => updateConfig(["retention", "emergency_free_percent"], Number(event.target.value))} /><small>Raises a critical storage state.</small></label>
            </div>
            {retentionStatus?.plan ? <RetentionSummary status={retentionStatus} /> : <div className="probe-result"><strong>Calculating retention projection</strong><span>The first index-only plan normally appears within a few seconds.</span></div>}
            {retentionError ? <div className="error-banner">{retentionError}</div> : null}
            <div className="retention-actions">
              <span className="admin-action-kind">Background task · applies immediately</span>
              <button type="button" onClick={() => runRetention(false)}><RefreshCcw size={15} /> Recalculate</button>
              <button type="button" className="danger" onClick={() => runRetention(true)} disabled={["queued", "planning", "cleaning", "waiting"].includes(retentionStatus?.state)}><Trash2 size={15} /> Clean Up Now</button>
            </div>
            <p className="retention-protection"><ShieldCheck size={15} /> Incident snapshots expire only after their configured history; face-reference images, incident clips, metadata databases, and the newest five minutes of recording remain protected.</p>
          </div>
          </div>
        </div>
      ) : null}

      {section === "access" ? (
        <AccessSettings config={config} updateConfig={updateConfig} commitImmediateConfig={commitImmediateConfig} onOpenApiTokens={onOpenApiTokens} />
      ) : null}


      {section === "mqtt" ? (
        <div className="sub-panel subsection-workspace">
          <nav className="admin-section-tabs camera-section-tabs detection-subsection-tabs" aria-label="API and MQTT settings">
            {[["tokens", "API Tokens", KeyRound], ["mqtt", "MQTT", Radio], ["ai", "AI Provider", Sparkles]].map(([value, label, Icon]) => <button type="button" key={value} className={apiSection === value ? "active" : ""} aria-pressed={apiSection === value} onClick={() => setApiSection(value)}><Icon size={15} />{label}</button>)}
          </nav>
          <div className="subsection-workspace-content">
          <section className="api-access-settings api-token-settings" hidden={apiSection !== "tokens"}>
            <div className="detection-settings-subhead">
              <div><strong className="section-heading-with-icon"><span className="section-heading-icon"><KeyRound size={16} /></span>API access tokens</strong><small>Long-lived credentials for Home Assistant and other integrations. Secrets are displayed only once and are never stored in readable form.</small></div>
              <div className="admin-action-status"><span className="admin-action-kind">Save settings to apply</span><span className={`retention-state ${config.api_auth?.enabled ? "running" : "idle"}`}>{config.api_auth?.enabled ? "Enforced" : "Not enforced"}</span></div>
            </div>
            <div className="api-auth-toggle">
              <label className="check-field">
                <input
                  type="checkbox"
                  checked={Boolean(config.api_auth?.enabled)}
                  disabled={!config.api_auth?.enabled && !(config.api_auth?.tokens || []).length}
                  onChange={(event) => updateConfig(["api_auth", "enabled"], event.target.checked)}
                />
                Require API authentication
              </label>
              <p className="settings-help">
                When enabled, API clients must send a valid scoped <code>Authorization: Bearer</code> token. Create at least one token before enabling this setting; Home Assistant will not connect while API authentication is disabled.
              </p>
              {!(config.api_auth?.tokens || []).length ? <p className="settings-help">Create a token below first. The control becomes available after the token is saved.</p> : null}
            </div>
            <div className="api-token-list">
              {(config.api_auth?.tokens || []).map((token) => (
                <article key={token.id}>
                  <div><strong>{token.name}</strong><code>{token.id}</code><small>{(token.scopes || []).join(" · ")}</small></div>
                  <button type="button" className="danger" onClick={() => deleteApiToken(token.id)} disabled={apiTokenBusy}><Trash2 size={14} /> Delete</button>
                </article>
              ))}
              {!(config.api_auth?.tokens || []).length ? <p className="settings-help">No integration tokens configured.</p> : null}
            </div>
            <div className="api-token-create">
              <label>Token ID<input value={apiTokenDraft.id} onChange={(event) => setApiTokenDraft((current) => ({ ...current, id: event.target.value }))} placeholder="home-assistant" /></label>
              <label>Name<input value={apiTokenDraft.name} onChange={(event) => setApiTokenDraft((current) => ({ ...current, name: event.target.value }))} placeholder="Home Assistant" /></label>
              <fieldset className="api-token-scopes"><legend>Token scopes</legend>
                {[["read", "Read"], ["camera:control", "Camera control"], ["admin", "Admin"]].map(([value, label]) => <label className="check-field" key={value}><input type="checkbox" checked={apiTokenDraft.scopes.includes(value)} onChange={() => toggleApiTokenScope(value)} /> {label}</label>)}
              </fieldset>
              <button type="button" className="primary" onClick={createApiToken} disabled={apiTokenBusy || !apiTokenDraft.id.trim() || !apiTokenDraft.name.trim() || !apiTokenDraft.scopes.length}>{apiTokenBusy ? <RefreshCcw className="spin" size={15} /> : <Plus size={15} />} Create token</button>
            </div>
            {apiTokenSecret ? <div className="api-token-secret" role="status"><strong>Copy this token now</strong><code>{apiTokenSecret}</code><button type="button" onClick={() => navigator.clipboard?.writeText(apiTokenSecret)}><Copy size={14} /> Copy</button><small>It cannot be displayed again after you leave this page.</small></div> : null}
            {apiTokenError ? <div className="error-banner">{apiTokenError}</div> : null}
          </section>
          <section className="api-access-settings mqtt-access-settings" hidden={apiSection !== "mqtt"}>
            <div className="detection-settings-subhead">
              <div><strong className="section-heading-with-icon"><span className="section-heading-icon"><Radio size={16} /></span>MQTT</strong><small>Broker connection, Home Assistant discovery, incident publishing, and server telemetry.</small></div>
              <div className="admin-action-status"><span className="admin-action-kind">Save settings to apply</span><span className={`retention-state ${mqttStatus?.connected ? "running" : "idle"}`}>{mqttStatus?.connected ? "Connected" : config.mqtt?.enabled ? "Disconnected" : "Disabled"}</span></div>
            </div>
            <div className="admin-field-grid">
              <label className="check-field"><input type="checkbox" checked={config.mqtt?.enabled || false} onChange={(event) => updateConfig(["mqtt", "enabled"], event.target.checked)} /> Enabled</label>
              <label>Broker Host<input value={config.mqtt?.host || ""} onChange={(event) => updateConfig(["mqtt", "host"], event.target.value)} placeholder="mqtt.local" /></label>
              <label>Port<input type="number" min="1" max="65535" value={config.mqtt?.port || 1883} onChange={(event) => updateConfig(["mqtt", "port"], Number(event.target.value))} /></label>
              <label>Username<input value={config.mqtt?.username || ""} onChange={(event) => updateConfig(["mqtt", "username"], event.target.value)} /></label>
              <label>Password<input type="password" value={secretInputValue(config.mqtt?.password)} placeholder={secretInputHint(config.mqtt?.password)} onChange={(event) => updateConfig(["mqtt", "password"], event.target.value)} /></label>
              <label>Client ID<input value={config.mqtt?.client_id || "survng"} onChange={(event) => updateConfig(["mqtt", "client_id"], event.target.value)} /></label>
              <label>Topic Prefix<input value={config.mqtt?.topic_prefix || "survng"} onChange={(event) => updateConfig(["mqtt", "topic_prefix"], event.target.value)} /></label>
              <label className="check-field"><input type="checkbox" checked={config.mqtt?.incident_events_enabled ?? true} onChange={(event) => updateConfig(["mqtt", "incident_events_enabled"], event.target.checked)} /> Publish incident events</label>
              <label className="check-field"><input type="checkbox" checked={config.mqtt?.discovery_enabled ?? true} onChange={(event) => updateConfig(["mqtt", "discovery_enabled"], event.target.checked)} /> Home Assistant Discovery</label>
              <label>Discovery Prefix<input value={config.mqtt?.discovery_prefix || "homeassistant"} onChange={(event) => updateConfig(["mqtt", "discovery_prefix"], event.target.value)} disabled={config.mqtt?.discovery_enabled === false} /></label>
              <label className="check-field"><input type="checkbox" checked={config.mqtt?.server_status_enabled ?? true} onChange={(event) => updateConfig(["mqtt", "server_status_enabled"], event.target.checked)} /> Publish SurvNG server status</label>
              <label>Server Device Name<input value={config.mqtt?.server_name || "SurvNG Server"} onChange={(event) => updateConfig(["mqtt", "server_name"], event.target.value)} disabled={config.mqtt?.server_status_enabled === false} /></label>
              <label>Server Metrics Interval<input type="number" min="10" max="3600" step="5" value={config.mqtt?.server_metrics_interval_seconds ?? 30} onChange={(event) => updateConfig(["mqtt", "server_metrics_interval_seconds"], Number(event.target.value))} disabled={config.mqtt?.server_status_enabled === false} /><small>Seconds between retained system, camera, detector, and storage updates.</small></label>
              <label>QoS<select value={config.mqtt?.qos ?? 0} onChange={(event) => updateConfig(["mqtt", "qos"], Number(event.target.value))}><option value={0}>0</option><option value={1}>1</option><option value={2}>2</option></select></label>
              <label className="check-field"><input type="checkbox" checked={config.mqtt?.tls || false} onChange={(event) => updateConfig(["mqtt", "tls"], event.target.checked)} /> TLS</label>
            </div>
            {mqttStatus ? <div className={`probe-result ${mqttStatus.connected ? "ok" : ""}`}><strong>Connection details</strong><span>{mqttStatus.host || "No broker"}:{mqttStatus.port || 1883}</span><span>{mqttStatus.messages_published || 0} published · {mqttStatus.publish_failures || 0} publish failures</span><span>Commands: {mqttStatus.command_subscriptions_active ? "ready" : mqttStatus.connected ? "not subscribed" : "offline"} · {mqttStatus.commands_received || 0} accepted · {mqttStatus.commands_rejected || 0} rejected · {mqttStatus.command_errors || 0} failed · {mqttStatus.command_queue_depth || 0} queued</span>{mqttStatus.server_status_enabled ? <span>Server: {mqttStatus.server_lifecycle || "starting"} · {mqttStatus.server_state?.health || "pending"} · {mqttStatus.server_state?.activity || "idle"} · every {mqttStatus.server_metrics_interval_seconds || 30}s · {mqttStatus.server_state_topic}</span> : null}{mqttStatus.incident_events_enabled ? <span>Incidents: {mqttStatus.incident_topic} ({mqttStatus.pending_incidents || 0} pending)</span> : null}{mqttStatus.server_status_error ? <span>Server metrics: {mqttStatus.server_status_error}</span> : null}{mqttStatus.last_error ? <span>{mqttStatus.last_error}</span> : null}</div> : null}
          </section>
          <section className="api-access-settings ai-provider-settings" id="ai-provider-settings" hidden={apiSection !== "ai"}>
            <div className="detection-settings-subhead">
              <div><strong className="section-heading-with-icon"><span className="section-heading-icon"><Sparkles size={16} /></span>AI Provider</strong><small>Shared provider for the assistant, Motion Audit reviews, and Camera Advisor.</small></div>
              <span className="admin-action-kind">Save settings to apply</span>
            </div>
            <div className="detection-field-grid">
              <label className="compact-toggle"><input type="checkbox" checked={config.audit_ai?.enabled ?? false} onChange={(event) => updateConfig(["audit_ai", "enabled"], event.target.checked)} /><span>AI features enabled</span></label>
              <label className="compact-toggle"><input type="checkbox" checked={config.audit_ai?.assistant_enabled ?? true} onChange={(event) => updateConfig(["audit_ai", "assistant_enabled"], event.target.checked)} disabled={!config.audit_ai?.enabled} /><span>SurvNG Assistant enabled</span></label>
              <label>Provider<select value={config.audit_ai?.provider || "openai"} onChange={(event) => updateConfig(["audit_ai", "provider"], event.target.value)}>
                <option value="openai">OpenAI</option>
                <option value="gemini">Google Gemini</option>
                <option value="openai_compatible">OpenAI compatible</option>
              </select></label>
              <label>Everyday AI model<input value={config.audit_ai?.model || ""} onChange={(event) => updateConfig(["audit_ai", "model"], event.target.value)} placeholder={config.audit_ai?.provider === "gemini" ? "gemini-2.5-flash" : "gpt-4.1-mini"} /><small>Used for Motion Audit reviews, finding incidents, status questions, and straightforward answers.</small></label>
              <label>Detailed analysis model<input value={config.audit_ai?.assistant_reasoning_model || ""} onChange={(event) => updateConfig(["audit_ai", "assistant_reasoning_model"], event.target.value)} placeholder="Leave blank to use the everyday model" /><small>Optional second model for visual incident reviews, difficult diagnoses, comparisons, and tuning advice.</small></label>
              <label>API Key<input type="password" value={secretInputValue(config.audit_ai?.api_key)} placeholder={secretInputHint(config.audit_ai?.api_key)} onChange={(event) => updateConfig(["audit_ai", "api_key"], event.target.value)} autoComplete="new-password" /></label>
              <label>Base URL<input value={config.audit_ai?.base_url || ""} onChange={(event) => updateConfig(["audit_ai", "base_url"], event.target.value)} placeholder={config.audit_ai?.provider === "gemini" ? "https://generativelanguage.googleapis.com/v1beta" : config.audit_ai?.provider === "openai_compatible" ? "http://localhost:11434/v1" : "https://api.openai.com/v1"} /></label>
              <label>Timeout Seconds<input type="number" min="5" max="120" step="1" value={config.audit_ai?.timeout_seconds ?? 45} onChange={(event) => updateConfig(["audit_ai", "timeout_seconds"], Number(event.target.value))} /></label>
              <label className="compact-toggle"><input type="checkbox" checked={config.audit_ai?.allow_apply_recommendations ?? false} onChange={(event) => updateConfig(["audit_ai", "allow_apply_recommendations"], event.target.checked)} /><span>Allow confirmed changes</span></label>
            </div>
          </section>
          </div>
        </div>
      ) : null}

      {section === "detection" ? (
        <div className="detection-settings subsection-workspace">
          <nav className="admin-section-tabs camera-section-tabs detection-subsection-tabs" aria-label="Intelligence and detection settings">
            {[["object", "Object Detection", Cpu], ["tracking", "Tracking & ReID", Activity], ["depth", "Depth Estimation", Layers], ["search", "Smart Search", Search], ["motion", "Motion Validation", Gauge], ["faces", "Face Recognition", ScanFace]].map(([value, label, Icon]) => <button type="button" className={detectionSection === value ? "active" : ""} aria-pressed={detectionSection === value} onClick={() => setDetectionSection(value)} key={value}><Icon size={15} />{label}</button>)}
          </nav>
          <div className="detection-settings-content">
          {detectionSection === "object" ? <section className="detection-settings-card primary">
            <header className="detection-settings-card-head">
              <div className="detection-settings-card-icon"><ScanFace size={18} /></div>
              <div><h3>Detection</h3><p>Choose the model, accelerator, and rules that turn motion into object incidents.</p></div>
              <label className="compact-toggle"><input type="checkbox" checked={config.detector?.enabled || false} onChange={(event) => updateConfig(["detector", "enabled"], event.target.checked)} /><span>Detector enabled</span></label>
            </header>
            <div className="detection-field-grid">
              <label>Backend<select value={detectorBackend} onChange={(event) => updateConfig(["detector", "backend"], event.target.value)}>
                <option value="openvino">OpenVINO / ONNX</option>
                <option value="coreml">Core ML (Mac)</option>
              </select></label>
              <label>OpenVINO Device<select value={config.detector?.device || "CPU"} onChange={(event) => updateConfig(["detector", "device"], event.target.value)}>
                {deviceOptions.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
              </select></label>
              <label>Parallel detectors<select value={String(config.detector?.object_worker_count ?? 2)} onChange={(event) => updateConfig(["detector", "object_worker_count"], Number(event.target.value))} disabled={detectorBackend !== "openvino"}>
                <option value="1" disabled={config.detector?.tracking?.enabled !== false}>1 detector</option>
                <option value="2">2 detectors</option>
                <option value="3">3 detectors</option>
                <option value="4">4 detectors</option>
              </select><small>Independent OpenVINO workers process simultaneous camera events. Tracking keeps at least 2 so a live incident check is not stuck behind overlay work. More workers use more accelerator and memory.</small></label>
              <label>Recorded decode processes<input type="number" min="1" max="16" step="1" value={config.detector?.recorded_decode_max_processes ?? 2} onChange={(event) => updateConfig(["detector", "recorded_decode_max_processes"], Number(event.target.value))} /><small>The shared decoded-frame budget is calculated from this count, the active refinement window, and each recording’s video dimensions.</small></label>
              <label>Incident confidence<input type="number" min="0.01" max="0.99" step="0.01" value={config.detector?.confidence_threshold ?? 0.45} onChange={(event) => updateConfig(["detector", "confidence_threshold"], Number(event.target.value))} /><small>A single detection must meet this confidence. Repeated candidates can still qualify through confirmation.</small></label>
              <label>Candidate confidence<input type="number" min="0.01" max="0.95" step="0.01" value={config.detector?.event_candidate_confidence_threshold ?? 0.25} onChange={(event) => updateConfig(["detector", "event_candidate_confidence_threshold"], Number(event.target.value))} /><small>Retains weaker detections only as temporal evidence; they require at least three consistent frames.</small></label>
              <label>Object confirmation<select value={String(config.detector?.event_confirmation_frames ?? 2)} onChange={(event) => updateConfig(["detector", "event_confirmation_frames"], Number(event.target.value))}><option value="1">Immediate (1 frame)</option><option value="2">Confirmed (2 frames)</option><option value="3">Strong (3 frames)</option><option value="4">Very strict (4 frames)</option><option value="5">Maximum (5 frames)</option></select><small>Requires the same label across this many recorded samples. Refinement stops early once confirmation is met, so lower counts also spend less detector time.</small></label>
              <label>Refinement window<select value={refinementStagePreset} onChange={(event) => {
                const preset = refinementStagePresets[event.target.value];
                if (preset) updateConfig(["detector", "event_refinement_stages"], preset);
              }}>
                <option value="full">Full (−1…+4.5s bridge, then +8/+12s)</option>
                <option value="compact">Compact (−0.5…+0.5s, then +4/+8s)</option>
                <option value="tight">Tight (−0.5…+0.5s only)</option>
                {refinementStagePreset === "custom" ? <option value="custom">Custom stages</option> : null}
              </select><small>Smaller windows free the detector sooner after each event. Full remains the default evidence profile.</small></label>
              <label>Refinement retry budget<input type="number" min="0" max="120" step="1" value={config.detector?.event_refinement_retry_seconds ?? 24} onChange={(event) => updateConfig(["detector", "event_refinement_retry_seconds"], Number(event.target.value))} /><small>Seconds spent waiting for finalized recordings and delayed discovery stages.</small></label>
              <label>Incident eligibility<select value={String(config.detector?.require_incident_zone ?? true)} onChange={(event) => updateConfig(["detector", "require_incident_zone"], event.target.value === "true")}>
                <option value="true">Zones</option>
                <option value="false">Zones + Full Frame</option>
              </select><small>Default for cameras using the global rule.</small></label>
              <label className="wide-field">Model<select value={activeModel?.path || ""} onChange={(event) => selectOpenvinoModel(event.target.value)}>
                <option value="">Custom path</option>
                {detectorModels.map((model) => {
                  const directory = String(model.path || "").split("/").slice(0, -1).pop();
                  return <option key={model.path} value={model.path} disabled={!model.valid}>{directory ? `${directory} / ` : ""}{model.name} ({model.task || "detect"}, {model.valid ? "ready" : "incomplete"})</option>;
                })}
              </select></label>
            </div>
            <details className="detection-compact-details">
              <summary>Model paths and startup options</summary>
              <div className="detection-field-grid">
                <label className="wide-field">OpenVINO / ONNX path<input value={activeModelPath} onChange={(event) => selectOpenvinoModel(event.target.value)} placeholder="openvino_model/best.xml or best.onnx" /></label>
                <label>Labels path<input value={config.detector?.labels_path || ""} onChange={(event) => updateConfig(["detector", "labels_path"], event.target.value)} placeholder="Automatic from metadata" /></label>
                <label>Compiled model cache<input value={config.detector?.cache_dir || ".cache/openvino"} onChange={(event) => updateConfig(["detector", "cache_dir"], event.target.value)} disabled={config.detector?.cache_enabled === false} /></label>
                <label className="compact-toggle"><input type="checkbox" checked={config.detector?.cache_enabled ?? true} onChange={(event) => updateConfig(["detector", "cache_enabled"], event.target.checked)} /><span>Cache compiled model</span></label>
                <label className="compact-toggle"><input type="checkbox" checked={config.detector?.warmup_enabled ?? true} onChange={(event) => updateConfig(["detector", "warmup_enabled"], event.target.checked)} /><span>Warm up at startup</span></label>
              </div>
            </details>
            <details className="detection-compact-details">
              <summary>Per-object confirmation and confidence</summary>
              <p className="settings-help">Tune how often and how confidently each object must be recognized. Higher confirmation reduces one-frame mistakes; higher confidence rejects weaker matches. Leaving either setting on global uses the values above.</p>
              {eventConfirmationClasses.length ? <div className="per-object-detection-grid">
                {eventConfirmationClasses.map((label) => <div className="per-object-detection-row" key={label}>
                  <strong>{label.replaceAll("_", " ")}</strong>
                  <label>Confirmation<select value={eventClassConfirmations[label] == null ? "" : String(eventClassConfirmations[label])} onChange={(event) => setEventClassConfirmation(label, event.target.value)}><option value="">Global ({config.detector?.event_confirmation_frames ?? 2} frames)</option><option value="1">1 frame</option><option value="2">2 frames</option><option value="3">3 frames</option><option value="4">4 frames</option><option value="5">5 frames</option></select></label>
                  <label>Confidence<input type="number" min="0.01" max="0.99" step="0.01" placeholder={`Global (${config.detector?.confidence_threshold ?? 0.45})`} value={eventClassConfidences[label] == null ? "" : String(eventClassConfidences[label])} onChange={(event) => setEventClassConfidence(label, event.target.value)} /></label>
                </div>)}
              </div> : <span className="settings-help">Select a model with class metadata to configure per-object overrides.</span>}
            </details>
          </section> : null}


          {detectionSection === "object" ? <section className="detection-settings-card detection-feature-card wide-card">
            <header className="detection-settings-card-head">
              <div className="detection-settings-card-icon"><Activity size={18} /></div>
              <div><h3>Stationary objects &amp; scene context</h3><p>Separate visual-motion filtering from object-level incident attribution.</p></div>
            </header>
            <div className="detection-field-grid">
              <label>Stationary object policy<select value={config.motion_qualification?.stationary_object_tolerance || "balanced"} onChange={(event) => updateConfig(["motion_qualification", "stationary_object_tolerance"], event.target.value)}><option value="low">Light</option><option value="balanced">Standard</option><option value="high">Strong</option></select><small>Coordinates EMA background learning, stationary-motion scoring, and parked-object scene memory. Strong may ignore unusually slow or distant travel.</small></label>
              <label>Repeated scene context<select value={config.detector?.object_activity_attribution || "enforce"} onChange={(event) => updateConfig(["detector", "object_activity_attribution"], event.target.value)}>
                <option value="enforce">Prevent false incident labels</option>
                <option value="shadow">Observe without changing incidents</option>
                <option value="off">Off</option>
              </select><small>Runs after object detection. Repeated stable objects remain stored as evidence without being treated as the cause; moving or uncertain objects remain eligible.</small></label>
              <div className="detection-settings-subhead"><strong>Fixed areas remain explicit</strong><small>Object Ignore zones suppress only their matching classes. “Exclude from EMA” independently removes all visual motion in that polygon.</small></div>
            </div>
          </section> : null}

          {detectionSection === "tracking" ? <section className="detection-settings-card wide-card">
            <header className="detection-settings-card-head">
              <div className="detection-settings-card-icon"><Activity size={18} /></div>
              <div><h3>Continuous tracking</h3><p>Identification and path overlays after an incident is confirmed. Tracking does not decide whether an incident is kept.</p></div>
            </header>
            <div className="detection-field-grid">
              <label>Tracking detail<select value={String(config.detector?.tracking?.sample_fps ?? 2)} onChange={(event) => updateConfig(["detector", "tracking", "sample_fps"], Number(event.target.value))}><option value="1">Lower CPU (1 frame/sec)</option><option value="2">Balanced (2 frames/sec)</option><option value="3">Smoother (3 frames/sec)</option><option value="5">Maximum detail (5 frames/sec)</option></select><small>OpenVINO runs once for every analyzed tracking frame.</small></label>
              <div className="zone-class-field tracking-class-field">
                <span>Do not track</span>
                <details className="zone-class-dropdown">
                  <summary>{trackingExcludedLabels.length ? trackingExcludedLabels.join(", ") : "Track all classes"}</summary>
                  <div className="zone-class-menu">
                    <label><input type="checkbox" checked={!trackingExcludedLabels.length} onChange={() => updateConfig(["detector", "tracking", "excluded_labels"], [])} /> Track all classes</label>
                    {trackingClassOptions.map((label) => {
                      const checked = trackingExcludedLabels.includes(label);
                      return <label key={label}><input type="checkbox" checked={checked} onChange={() => updateConfig(["detector", "tracking", "excluded_labels"], checked ? trackingExcludedLabels.filter((item) => item !== label) : [...trackingExcludedLabels, label])} /> {label}</label>;
                    })}
                  </div>
                </details>
                <small>Select classes to exclude. Face detection and recognition continue normally; excluded classes simply do not receive track IDs.</small>
              </div>
              <label>Maximum duration<input type="number" min="3" max="120" step="1" value={config.detector?.tracking?.max_session_seconds ?? 15} onChange={(event) => updateConfig(["detector", "tracking", "max_session_seconds"], Number(event.target.value))} /><small>Seconds after initial detection.</small></label>
              <label>Lost-object grace<input type="number" min="0.5" max="15" step="0.5" value={config.detector?.tracking?.lost_timeout_seconds ?? 3} onChange={(event) => updateConfig(["detector", "tracking", "lost_timeout_seconds"], Number(event.target.value))} /><small>Seconds to retain an obstructed object.</small></label>
              <label>Baseline camera limit<input type="number" min="1" max="16" step="1" value={config.detector?.tracking?.max_active_cameras ?? 2} onChange={(event) => updateConfig(["detector", "tracking", "max_active_cameras"], Number(event.target.value))} /><small>Normal simultaneous tracking sessions.</small></label>
              <label className="compact-toggle"><input type="checkbox" checked={config.detector?.tracking?.adaptive_burst_enabled ?? true} onChange={(event) => updateConfig(["detector", "tracking", "adaptive_burst_enabled"], event.target.checked)} /><span>Allow an extra tracker when healthy</span><small>Temporarily uses the burst limit only while inference has no backlog and system memory is healthy.</small></label>
              <label>Burst camera limit<input type="number" min={config.detector?.tracking?.max_active_cameras ?? 2} max="16" step="1" value={config.detector?.tracking?.burst_max_active_cameras ?? 3} onChange={(event) => updateConfig(["detector", "tracking", "burst_max_active_cameras"], Number(event.target.value))} /><small>Maximum only during a healthy short burst.</small></label>
              <label>Wait for tracking capacity<input type="number" min="0" max="30" step="0.5" value={config.detector?.tracking?.capacity_wait_seconds ?? 5} onChange={(event) => updateConfig(["detector", "tracking", "capacity_wait_seconds"], Number(event.target.value))} /><small>Wait briefly for a busy tracking slot, then recover the gap from recordings. Zero skips immediately.</small></label>
            </div>
            <details className="detection-compact-details">
              <summary>Association tuning</summary>
              <div className="detection-field-grid advanced-tracking-grid">
                <label className="compact-toggle"><input type="checkbox" checked={config.detector?.tracking?.enabled ?? true} onChange={(event) => updateConfig(["detector", "tracking", "enabled"], event.target.checked)} /><span>Enable core tracking</span><small>Runs after recorded confirmation. Live ticks reuse gvadetect boxes so the Intel GPU stays on capture. Recorded catch-up and cover checks still use OpenVINO.</small></label>
                <div className="detection-settings-subhead"><strong>SurvNG Hybrid tracking</strong><small>Production tracking uses SurvNG’s timestamp-aware geometry and selective appearance recovery. FastTrack is available only through the incident Compare tool.</small></div>
                <label>Confirm after detections<input type="number" min="1" max="10" step="1" value={config.detector?.tracking?.min_confirmations ?? 2} onChange={(event) => updateConfig(["detector", "tracking", "min_confirmations"], Number(event.target.value))} /><small>New objects found during an active session need this many matching observations. Incident-starting objects have already passed the event-frame confirmation above.</small></label>
                <label>Tracking confidence floor<input type="number" min="0.01" max="0.95" step="0.01" value={config.detector?.tracking?.low_confidence_threshold ?? 0.25} onChange={(event) => updateConfig(["detector", "tracking", "low_confidence_threshold"], Number(event.target.value))} /><small>Allows an existing track to survive weaker detections without creating a new incident object.</small></label>
                <label>Box match overlap<input type="number" min="0.05" max="0.9" step="0.05" value={config.detector?.tracking?.match_iou_threshold ?? 0.2} onChange={(event) => updateConfig(["detector", "tracking", "match_iou_threshold"], Number(event.target.value))} /><small>How much predicted and detected boxes must overlap to retain an ID.</small></label>
                <label>Movement match distance<input type="number" min="0.1" max="2" step="0.05" value={config.detector?.tracking?.match_center_distance_ratio ?? 0.65} onChange={(event) => updateConfig(["detector", "tracking", "match_center_distance_ratio"], Number(event.target.value))} /><small>Reconnects nearby boxes when overlap changes because someone moves quickly or approaches the camera.</small></label>
                <label>Maximum tracks per incident<input type="number" min="1" max="1000" step="10" value={config.detector?.tracking?.max_tracks_per_session ?? 100} onChange={(event) => updateConfig(["detector", "tracking", "max_tracks_per_session"], Number(event.target.value))} /><small>Safety limit for unusually noisy detector output.</small></label>
              </div>
            </details>
            <details className="detection-compact-details">
              <summary>Appearance matching (ReID)</summary>
              <div className="detection-field-grid advanced-tracking-grid">
                <div className="detection-settings-subhead"><strong>Person appearance matching</strong><small>Reconnect a person after geometry briefly loses them.</small></div>
                <label className="compact-toggle"><input type="checkbox" checked={config.detector?.tracking?.reid_enabled ?? false} onChange={(event) => updateConfig(["detector", "tracking", "reid_enabled"], event.target.checked)} /><span>Person ReID enabled</span></label>
                <label>Person ReID model<input value={config.detector?.tracking?.reid_model_path ?? ""} onChange={(event) => updateConfig(["detector", "tracking", "reid_model_path"], event.target.value)} placeholder="person-reidentification-retail-0286.xml" /><small>OpenVINO whole-person embedding model. Intel's 0286 model is the recommended accuracy-focused option; face-recognition models are not compatible.</small></label>
                <label>ReID device<input value={config.detector?.tracking?.reid_device ?? "CPU"} onChange={(event) => updateConfig(["detector", "tracking", "reid_device"], event.target.value)} /><small>CPU by default so live gvadetect keeps the Intel GPU. AUTO is treated as CPU. Explicit GPU remains available.</small></label>
                <label>Appearance similarity<input type="number" min="0" max="1" step="0.01" value={config.detector?.tracking?.reid_match_threshold ?? 0.7} onChange={(event) => updateConfig(["detector", "tracking", "reid_match_threshold"], Number(event.target.value))} /><small>0.70 is the conservative default. Higher values reduce accidental joins but make lost identities harder to recover.</small></label>
                <label>Remember lost appearance<input type="number" min="1" max="300" step="1" value={config.detector?.tracking?.reid_max_age_seconds ?? 30} onChange={(event) => updateConfig(["detector", "tracking", "reid_max_age_seconds"], Number(event.target.value))} /><small>Seconds a lost person can recover the same track ID.</small></label>
                <div className="detection-settings-subhead"><strong>Vehicle appearance matching</strong><small>Use vehicle appearance to recover car, truck, bus, and motorcycle identities.</small></div>
                <label className="compact-toggle"><input type="checkbox" checked={config.detector?.tracking?.vehicle_reid_enabled ?? false} onChange={(event) => updateConfig(["detector", "tracking", "vehicle_reid_enabled"], event.target.checked)} /><span>Vehicle ReID enabled</span></label>
                <label>Vehicle ReID model<input value={config.detector?.tracking?.vehicle_reid_model_path ?? ""} onChange={(event) => updateConfig(["detector", "tracking", "vehicle_reid_model_path"], event.target.value)} placeholder="vehicle-reid-0001.xml" /><small>OpenVINO whole-vehicle embedding model. This is separate from the person model.</small></label>
                <label>Vehicle labels<input value={(config.detector?.tracking?.vehicle_reid_labels || ["car", "truck", "bus", "motorcycle"]).join(", ")} onChange={(event) => updateConfig(["detector", "tracking", "vehicle_reid_labels"], event.target.value.split(",").map((value) => value.trim().toLowerCase()).filter(Boolean))} /><small>Comma-separated detector labels that use vehicle appearance matching.</small></label>
                <label>Vehicle ReID device<input value={config.detector?.tracking?.vehicle_reid_device ?? "CPU"} onChange={(event) => updateConfig(["detector", "tracking", "vehicle_reid_device"], event.target.value)} /><small>Shares the isolated appearance worker. CPU by default so live gvadetect keeps the Intel GPU. AUTO is treated as CPU.</small></label>
                <label>Vehicle appearance similarity<input type="number" min="0" max="1" step="0.01" value={config.detector?.tracking?.vehicle_reid_match_threshold ?? 0.8} onChange={(event) => updateConfig(["detector", "tracking", "vehicle_reid_match_threshold"], Number(event.target.value))} /><small>Higher values reduce accidental merging of similar-looking vehicles.</small></label>
                <label>Maximum appearance checks<input type="number" min="1" max="64" step="1" value={config.detector?.tracking?.reid_max_embeddings_per_frame ?? 8} onChange={(event) => updateConfig(["detector", "tracking", "reid_max_embeddings_per_frame"], Number(event.target.value))} /><small>Bounds combined person and vehicle ReID work in a crowded frame.</small></label>
                <label>Refresh appearance every<input type="number" min="1" max="120" step="1" value={config.detector?.tracking?.reid_refresh_interval_frames ?? 8} onChange={(event) => updateConfig(["detector", "tracking", "reid_refresh_interval_frames"], Number(event.target.value))} /><small>Matched samples between appearance refreshes. Geometry handles the intervening frames; lower values use more CPU.</small></label>
                <div className="detection-settings-subhead"><strong>Missed-session recovery</strong><small>Recover durable appearance evidence from the saved incident image after full tracking finishes or is skipped.</small></div>
                <label className="compact-toggle"><input type="checkbox" checked={config.detector?.tracking?.deferred_reid_enabled ?? true} onChange={(event) => updateConfig(["detector", "tracking", "deferred_reid_enabled"], event.target.checked)} /><span>Recover missed appearance evidence</span></label>
                <label>Recovery delay<input type="number" min="0" max="300" step="1" value={config.detector?.tracking?.deferred_reid_delay_seconds ?? 20} onChange={(event) => updateConfig(["detector", "tracking", "deferred_reid_delay_seconds"], Number(event.target.value))} /><small>Waits for stronger multi-frame tracking evidence before using a single saved snapshot.</small></label>
                <label>Nearby-camera window<input type="number" min="1" max="300" step="1" value={config.detector?.tracking?.related_sequence_window_seconds ?? 30} onChange={(event) => updateConfig(["detector", "tracking", "related_sequence_window_seconds"], Number(event.target.value))} /><small>Seconds on either side used to show clearly labeled sequence candidates. Time alone never claims identity.</small></label>
                <div className="detection-settings-subhead camera-route-heading"><div><strong>Expected camera routes</strong><small>Describe physically plausible camera-to-camera movement. Direction follows event time; routes strengthen ordering but never establish identity by themselves.</small></div><button type="button" onClick={addCameraRoute} disabled={routeCameras.length < 2}>Add route</button></div>
                <div className="camera-route-list">
                  {cameraTransitionRoutes.length ? cameraTransitionRoutes.map((route, index) => <div className="camera-route-row" key={`${route.from_camera}-${route.to_camera}-${index}`}>
                    <label>From<select value={route.from_camera} onChange={(event) => updateCameraRoute(index, "from_camera", event.target.value)}>{routeCameras.map((camera) => <option value={camera.id} key={camera.id}>{camera.name || camera.id}</option>)}</select></label>
                    <span className="camera-route-arrow">→</span>
                    <label>To<select value={route.to_camera} onChange={(event) => updateCameraRoute(index, "to_camera", event.target.value)}>{routeCameras.map((camera) => <option value={camera.id} key={camera.id}>{camera.name || camera.id}</option>)}</select></label>
                    <label>Earliest<input type="number" min="0" max="299" step="1" value={route.min_seconds ?? 0} onChange={(event) => updateCameraRoute(index, "min_seconds", Number(event.target.value))} /><small>seconds</small></label>
                    <label>Latest<input type="number" min="1" max="300" step="1" value={route.max_seconds ?? 30} onChange={(event) => updateCameraRoute(index, "max_seconds", Number(event.target.value))} /><small>seconds</small></label>
                    <label className="compact-toggle"><input type="checkbox" checked={route.bidirectional ?? false} onChange={(event) => updateCameraRoute(index, "bidirectional", event.target.checked)} /><span>Both directions</span></label>
                    <label className="compact-toggle"><input type="checkbox" checked={route.enabled ?? true} onChange={(event) => updateCameraRoute(index, "enabled", event.target.checked)} /><span>Enabled</span></label>
                    <button type="button" className="danger" onClick={() => updateConfig(["detector", "tracking", "camera_transition_routes"], cameraTransitionRoutes.filter((_item, routeIndex) => routeIndex !== index))}>Remove</button>
                  </div>) : <p className="settings-help">No expected routes yet. Nearby incidents still appear as general sequence candidates.</p>}
                </div>
              </div>
              {config.detector?.tracking?.reid_enabled ? (
                reidStatus?.enabled ? (
                  <div className={`probe-result ${(reidStatus.person?.ready ?? reidStatus.ready) ? "ok" : "bad"}`}>
                    <strong>{(reidStatus.person?.ready ?? reidStatus.ready) ? "Person appearance matching is ready" : "Person appearance matching is unavailable"}</strong>
                    <span>{(reidStatus.person?.ready ?? reidStatus.ready) ? `${reidStatus.person?.device || reidStatus.device || "AUTO"} · ${reidStatus.person?.embedding_size || reidStatus.embedding_size || 0}-value appearance signature` : reidStatus.person?.error || reidStatus.error || "The isolated ReID worker did not start."}</span>
                    {(reidStatus.person?.ready ?? reidStatus.ready) && (reidStatus.person?.model_load_ms ?? reidStatus.model_load_ms) != null ? <span>Model loaded in {Math.round(reidStatus.person?.model_load_ms ?? reidStatus.model_load_ms)} ms</span> : null}
                  </div>
                ) : <div className="probe-result"><strong>Person appearance matching is not active yet</strong><span>Save the configuration and restart SurvNG to start its isolated model worker.</span></div>
              ) : null}
              {config.detector?.tracking?.vehicle_reid_enabled ? (
                reidStatus?.enabled ? (
                  <div className={`probe-result ${reidStatus.vehicle?.ready ? "ok" : "bad"}`}>
                    <strong>{reidStatus.vehicle?.ready ? "Vehicle appearance matching is ready" : "Vehicle appearance matching is unavailable"}</strong>
                    <span>{reidStatus.vehicle?.ready ? `${reidStatus.vehicle.device || "AUTO"} · ${reidStatus.vehicle.embedding_size || 0}-value vehicle signature · ${(reidStatus.vehicle.labels || []).join(", ")}` : reidStatus.vehicle?.error || "The vehicle ReID model did not start."}</span>
                    {reidStatus.vehicle?.ready && reidStatus.vehicle.model_load_ms != null ? <span>Model loaded in {Math.round(reidStatus.vehicle.model_load_ms)} ms</span> : null}
                  </div>
                ) : <div className="probe-result"><strong>Vehicle appearance matching is not active yet</strong><span>Save the configuration and restart SurvNG to start the model.</span></div>
              ) : null}
            </details>
          </section> : null}

          {detectionSection === "search" ? <details className="detection-settings-card detection-feature-card wide-card" open>
            <summary><span className="detection-settings-card-icon"><Search size={18} /></span><span><strong>Smart Search</strong><small>Find indexed incidents by describing visible details in plain language.</small></span></summary>
            <div className="detection-feature-body detection-field-grid">
              <label className="compact-toggle"><input type="checkbox" checked={config.semantic_search?.enabled ?? false} onChange={(event) => updateConfig(["semantic_search", "enabled"], event.target.checked)} /><span>Smart Search enabled</span></label>
              <label>Model package<input value={config.semantic_search?.model_dir ?? ""} onChange={(event) => updateConfig(["semantic_search", "model_dir"], event.target.value)} placeholder="/path/to/SurvNG/models/mobileclip2-b-openvino-fp16" /><small>Use the host path for systemd or the mounted container path for Docker. The package contains semantic_model.json, both encoders, and tokenizer assets.</small></label>
              <label>Inference device<input value={config.semantic_search?.device ?? "GPU"} onChange={(event) => updateConfig(["semantic_search", "device"], event.target.value)} /><small>GPU is recommended on Intel systems. This does not share the object detector queue.</small></label>
              <label>Historical batch size<input type="number" min="1" max="250" step="1" value={config.semantic_search?.backfill_batch_size ?? 25} onChange={(event) => updateConfig(["semantic_search", "backfill_batch_size"], Number(event.target.value))} /><small>How many older incidents are scheduled at a time. Existing indexed generations are skipped.</small></label>
              <label>Historical pacing<input type="number" min="0.01" max="5" step="0.05" value={config.semantic_search?.backfill_pause_seconds ?? 0.25} onChange={(event) => updateConfig(["semantic_search", "backfill_pause_seconds"], Number(event.target.value))} /><small>Pause between older incidents so object detection and new Smart Search evidence retain priority.</small></label>
              <label className="compact-toggle"><input type="checkbox" checked={config.semantic_search?.index_full_frame ?? true} onChange={(event) => updateConfig(["semantic_search", "index_full_frame"], event.target.checked)} /><span>Index whole incident image</span></label>
              <label className="compact-toggle"><input type="checkbox" checked={config.semantic_search?.index_object_crops ?? true} onChange={(event) => updateConfig(["semantic_search", "index_object_crops"], event.target.checked)} /><span>Index detected object crops</span></label>
              <label>Object crops per incident<input type="number" min="1" max="100" step="1" value={config.semantic_search?.max_object_crops_per_event ?? 24} onChange={(event) => updateConfig(["semantic_search", "max_object_crops_per_event"], Number(event.target.value))} /><small>Caps crop inference and memory for unusually busy incidents; highest-confidence detections are indexed first.</small></label>
            </div>
          </details> : null}

          {detectionSection === "motion" ? <details className="detection-settings-card detection-feature-card wide-card" open>
            <summary><span className="detection-settings-card-icon"><Gauge size={18} /></span><span><strong>Motion validation</strong><small>How camera and visual motion decide when object detection runs.</small></span></summary>
            <div className="detection-feature-body">
              <MotionAnalysisPresetEditor
                qualification={config.motion_qualification?.pipeline?.qualification || []}
                catalog={motionCatalog}
                onChange={(qualification) => updateConfig(
                  ["motion_qualification", "pipeline"],
                  { ...(config.motion_qualification?.pipeline || {}), qualification },
                )}
              />
              <details className="motion-tuning-details">
                <summary>Advanced motion tuning</summary>
                <div className="field-row">
                  <label>Sensitivity<select value={config.motion_qualification?.sensitivity || "balanced"} onChange={(event) => updateConfig(["motion_qualification", "sensitivity"], event.target.value)}><option value="high">High</option><option value="balanced">Balanced</option><option value="low">Low</option></select></label>
                  <label>Light and shadow filtering<select value={String(config.motion_qualification?.illumination_filter_enabled ?? false)} onChange={(event) => updateConfig(["motion_qualification", "illumination_filter_enabled"], event.target.value === "true")}><option value="false">Disabled</option><option value="true">Enabled</option></select><small>Ignores clear moving illumination while uncertain motion continues to object detection. Disabled still records evidence for evaluation.</small></label>
                  <label>Analysis size<select value={config.motion_qualification?.frame_width ?? 320} onChange={(event) => updateConfig(["motion_qualification", "frame_width"], Number(event.target.value))}><option value="320">320 px</option><option value="480">480 px</option><option value="640">640 px</option><option value="720">720 px</option><option value="800">800 px</option></select><small>Maximum image edge used by EMA; portrait cameras no longer expand beyond this size.</small></label>
                  <label>Frame stability filter<input type="number" min="0" max="1" step="0.001" value={config?.motion_qualification?.temporal_filter_threshold ?? 0.005} onChange={(event) => updateConfig(["motion_qualification", "temporal_filter_threshold"], Number(event.target.value))} /><small>Skip analysis if pixel change is below this ratio (0.005 = 0.5%). Lower = more skips, higher = more analysis. Skips: check telemetry for per-camera stats.</small></label>
                  <label>Sample FPS<input type="number" min="2" max="10" step="1" value={config.motion_qualification?.sample_fps ?? 5} onChange={(event) => updateConfig(["motion_qualification", "sample_fps"], Number(event.target.value))} /><small>EMA samples per second on cameras that run continuous analysis.</small></label>
                  <label>Simultaneous EMA cameras<input type="number" min="1" max="16" step="1" value={config.motion_qualification?.max_concurrent_analysis ?? 2} onChange={(event) => updateConfig(["motion_qualification", "max_concurrent_analysis"], Number(event.target.value))} /><small>How many cameras may run visual analysis at once. Raise this on a larger NVR if EMA backup coverage is falling behind. Capture and recording are not limited by this.</small></label>
                  <label>ONVIF background upkeep<select value={String(config.motion_qualification?.camera_mode_background_fps ?? 2)} onChange={(event) => updateConfig(["motion_qualification", "camera_mode_background_fps"], Number(event.target.value))}><option value="1">Low CPU (1 frame/sec)</option><option value="2">Balanced (2 frames/sec)</option><option value="3">Faster adaptation (3 frames/sec)</option><option value="5">Maximum adaptation (5 frames/sec)</option></select><small>When camera alerts trigger motion, SurvNG maintains the visual background at this lower rate. Trigger validation still analyzes the full buffered window.</small></label>
                  {config.motion_qualification?.mode === "camera_rescue" ? <>
                    <div className="detection-settings-subhead"><strong>Visual backup safeguards</strong><small>These conservative limits control when SurvNG may compensate for a missing camera notice.</small></div>
                    <label>Scene learning time<input type="number" min="0" max="120" step="1" value={config.motion_qualification?.visual_backup_warmup_seconds ?? 10} onChange={(event) => updateConfig(["motion_qualification", "visual_backup_warmup_seconds"], Number(event.target.value))} /><small>After this unchanged startup period, EMA also waits for a quiet scene baseline. Camera alerts continue normally throughout.</small></label>
                    <label>Wait for camera notice<input type="number" min="0" max="5" step="0.25" value={config.motion_qualification?.visual_backup_grace_seconds ?? 1.5} onChange={(event) => updateConfig(["motion_qualification", "visual_backup_grace_seconds"], Number(event.target.value))} /><small>Seconds strong visual motion must persist while SurvNG waits for ONVIF.</small></label>
                    <label>Minimum visual confidence<input type="number" min="0" max="1" step="0.01" value={config.motion_qualification?.visual_backup_min_score ?? 0.7} onChange={(event) => updateConfig(["motion_qualification", "visual_backup_min_score"], Number(event.target.value))} /><small>Absolute adaptive score required before visual backup is considered.</small></label>
                    <label>Confidence above normal<input type="number" min="0" max="0.5" step="0.01" value={config.motion_qualification?.visual_backup_score_margin ?? 0.15} onChange={(event) => updateConfig(["motion_qualification", "visual_backup_score_margin"], Number(event.target.value))} /><small>Additional margin above the camera&apos;s adaptive threshold.</small></label>
                    <label>Consecutive strong samples<input type="number" min="2" max="10" step="1" value={config.motion_qualification?.visual_backup_min_consecutive ?? 3} onChange={(event) => updateConfig(["motion_qualification", "visual_backup_min_consecutive"], Number(event.target.value))} /><small>Prevents a single noisy frame from invoking object detection.</small></label>
                    <label>Backup cooldown<input type="number" min="5" max="300" step="5" value={config.motion_qualification?.visual_backup_cooldown_seconds ?? 20} onChange={(event) => updateConfig(["motion_qualification", "visual_backup_cooldown_seconds"], Number(event.target.value))} /><small>Minimum seconds between visual backup attempts and after a camera notice.</small></label>
                    <label>Maximum backups per 5 minutes<input type="number" min="1" max="30" step="1" value={config.motion_qualification?.visual_backup_max_triggers_5m ?? 3} onChange={(event) => updateConfig(["motion_qualification", "visual_backup_max_triggers_5m"], Number(event.target.value))} /><small>Hard per-camera safety limit for object-detector work.</small></label>
                  </> : null}
                  <label>Window Seconds<input type="number" min="0.8" max="4" step="0.1" value={config.motion_qualification?.window_seconds ?? 1.6} onChange={(event) => updateConfig(["motion_qualification", "window_seconds"], Number(event.target.value))} /></label>
                  <label>Post-trigger Seconds<input type="number" min="0.5" max="6" step="0.1" value={config.motion_qualification?.post_trigger_seconds ?? 2.5} onChange={(event) => updateConfig(["motion_qualification", "post_trigger_seconds"], Number(event.target.value))} /></label>
                  <label>Burst Quiet Seconds<input type="number" min="0.1" max="2" step="0.1" value={config.motion_qualification?.burst_quiet_seconds ?? 0.5} onChange={(event) => updateConfig(["motion_qualification", "burst_quiet_seconds"], Number(event.target.value))} /></label>
                  <label>Save rejected motion images<select value={String(config.motion_qualification?.rejected_sample_rate ?? 1)} onChange={(event) => updateConfig(["motion_qualification", "rejected_sample_rate"], Number(event.target.value))}><option value="1">Every rejection (Recommended)</option><option value="0.5">About half</option><option value="0.1">About 1 in 10</option><option value="0.05">About 1 in 20</option><option value="0">Never</option></select><small>Used by Motion Audit and the AI Advisor. SurvNG keeps the latest 100 per camera.</small></label>
                  <label>Double-check filtered motion<select value={String(config.motion_qualification?.suppression_verification_rate ?? 0.05)} onChange={(event) => updateConfig(["motion_qualification", "suppression_verification_rate"], Number(event.target.value))}><option value="0">Off</option><option value="0.01">About 1 in 100</option><option value="0.05">About 1 in 20 (Recommended)</option><option value="0.1">About 1 in 10</option></select><small>Runs object detection on a sample of visual rejections. If a configured object is found, SurvNG restores the incident; otherwise only Motion Audit records the check.</small></label>
                  <label className="check-field"><input type="checkbox" checked={config.motion_qualification?.borderline_rescue_enabled ?? true} onChange={(event) => updateConfig(["motion_qualification", "borderline_rescue_enabled"], event.target.checked)} /> Borderline object rescue</label>
                  <label>Rescue Margin<input type="number" min="0" max="0.1" step="0.005" value={config.motion_qualification?.borderline_margin ?? 0.03} onChange={(event) => updateConfig(["motion_qualification", "borderline_margin"], Number(event.target.value))} /></label>
                </div>
              </details>
              <MotionDecisionEditor
                fusion={config.motion_qualification?.pipeline?.fusion}
                mode={config.motion_qualification?.mode || "camera_rescue"}
                onModeChange={(mode) => updateConfig(["motion_qualification", "mode"], mode)}
                onChange={(fusion) => updateConfig(
                  ["motion_qualification", "pipeline"],
                  { ...(config.motion_qualification?.pipeline || {}), fusion },
                )}
              />
            </div>
          </details> : null}

          {detectionSection === "depth" ? <details className="detection-settings-card detection-feature-card wide-card" open>
            <summary><span className="detection-settings-card-icon"><Layers size={18} /></span><span><strong>Monocular depth</strong><small>Estimate per-object distance on representative incident frames.</small></span></summary>
            <div className="detection-feature-body detection-field-grid">
              <label className="compact-toggle"><input type="checkbox" checked={config.detector?.depth?.enabled ?? false} onChange={(event) => updateConfig(["detector", "depth", "enabled"], event.target.checked)} /><span>Depth enrichment enabled</span></label>
              <label>Depth Model<input value={config.detector?.depth?.model_path || ""} onChange={(event) => updateConfig(["detector", "depth", "model_path"], event.target.value)} placeholder="yolo26n-depth_openvino_model/yolo26n-depth.xml" /></label>
              <label>Depth Device<select value={config.detector?.depth?.device || "CPU"} onChange={(event) => updateConfig(["detector", "depth", "device"], event.target.value)}>
                {deviceOptions.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
              </select><small>CPU by default. Live gvadetect owns the Intel GPU; AUTO is treated as CPU.</small></label>
              <label>Input Size<input type="number" min="320" max="1280" step="32" value={config.detector?.depth?.input_size ?? 768} onChange={(event) => updateConfig(["detector", "depth", "input_size"], Number(event.target.value))} /></label>
              <label>Minimum Distance (m)<input type="number" min="0.01" max="500" step="0.01" value={config.detector?.depth?.min_distance_m ?? 0.05} onChange={(event) => updateConfig(["detector", "depth", "min_distance_m"], Number(event.target.value))} /></label>
              <label>Maximum Distance (m)<input type="number" min="1" max="500" step="0.1" value={config.detector?.depth?.max_distance_m ?? 150} onChange={(event) => updateConfig(["detector", "depth", "max_distance_m"], Number(event.target.value))} /></label>
              <label>Ignore Incidents Beyond (m)<input type="number" min="0.5" max="500" step="0.1" value={config.detector?.depth?.max_incident_distance_m ?? ""} onChange={(event) => updateConfig(["detector", "depth", "max_incident_distance_m"], event.target.value === "" ? null : Number(event.target.value))} placeholder="optional" /></label>
              <label className="compact-toggle"><input type="checkbox" checked={config.detector?.depth?.store_heatmap ?? false} onChange={(event) => updateConfig(["detector", "depth", "store_heatmap"], event.target.checked)} /><span>Store representative depth heatmap</span></label>
            </div>
          </details> : null}

          {detectionSection === "faces" ? <details className="detection-settings-card detection-feature-card wide-card" open>
            <summary><span className="detection-settings-card-icon"><ScanFace size={18} /></span><span><strong>Face recognition</strong><small>Identify detected faces using a separate embedding model.</small></span></summary>
            <div className="detection-feature-body detection-field-grid">
              <label className="compact-toggle"><input type="checkbox" checked={config.detector?.face_recognition_enabled ?? false} onChange={(event) => updateConfig(["detector", "face_recognition_enabled"], event.target.checked)} /><span>Recognition enabled</span></label>
              <label>Embedding Model<input value={config.detector?.face_embedding_model_path || ""} onChange={(event) => updateConfig(["detector", "face_embedding_model_path"], event.target.value)} placeholder="face_model/model.xml" /></label>
              <label>Landmark Model<input value={config.detector?.face_landmark_model_path || ""} onChange={(event) => updateConfig(["detector", "face_landmark_model_path"], event.target.value)} placeholder="face_model/landmarks.xml" /></label>
              <label>Face Detector Model<input value={config.detector?.face_detection_model_path || ""} onChange={(event) => updateConfig(["detector", "face_detection_model_path"], event.target.value)} placeholder="face_detector/model.xml" /></label>
              <label>Recognition Device<select value={config.detector?.face_recognition_device || "CPU"} onChange={(event) => updateConfig(["detector", "face_recognition_device"], event.target.value)}>
                {deviceOptions.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
              </select><small>CPU by default. Live gvadetect owns the Intel GPU; AUTO is treated as CPU.</small></label>
              <label>Face Detection Confidence<input type="number" min="0.01" max="0.99" step="0.01" value={config.detector?.face_detection_threshold ?? 0.6} onChange={(event) => updateConfig(["detector", "face_detection_threshold"], Number(event.target.value))} /></label>
              <label>Suggestion Threshold<input type="number" min="0" max="1" step="0.01" value={config.detector?.face_match_threshold ?? 0.4} onChange={(event) => updateConfig(["detector", "face_match_threshold"], Number(event.target.value))} /></label>
              <label>Minimum Face Size<input type="number" min="16" max="1024" step="8" value={config.detector?.face_min_size ?? 48} onChange={(event) => updateConfig(["detector", "face_min_size"], Number(event.target.value))} /></label>
              <label>References Per Person<input type="number" min="1" max="200" step="1" value={config.detector?.face_max_references ?? 20} onChange={(event) => updateConfig(["detector", "face_max_references"], Number(event.target.value))} /><small>SurvNG chooses the clearest, most varied confirmed faces; pinned references are always retained.</small></label>
              <label>Saved face limit<input type="number" min="100" max="100000" step="100" value={config.detector?.face_max_observations ?? 1000} onChange={(event) => updateConfig(["detector", "face_max_observations"], Number(event.target.value))} /><small>Oldest observations are removed first.</small></label>
              <label className="compact-toggle"><input type="checkbox" checked={config.detector?.face_auto_identify_enabled ?? false} onChange={(event) => updateConfig(["detector", "face_auto_identify_enabled"], event.target.checked)} /><span>Automatically identify very strong matches</span></label>
              <label>Automatic Match Threshold<input type="number" min="0" max="1" step="0.01" value={config.detector?.face_auto_identify_threshold ?? 0.55} onChange={(event) => updateConfig(["detector", "face_auto_identify_threshold"], Number(event.target.value))} /></label>
              <label>Minimum Lead Over Next Person<input type="number" min="0" max="1" step="0.01" value={config.detector?.face_auto_identify_margin ?? 0.12} onChange={(event) => updateConfig(["detector", "face_auto_identify_margin"], Number(event.target.value))} /></label>
            </div>
          </details> : null}
          </div>
        </div>
      ) : null}

      {section === "motion-review" ? (
        <MotionAiReviewPanel
          cameras={config.cameras || []}
          runtimeStatus={runtimeStatus}
          advisorEnabled={config.audit_ai?.enabled ?? false}
          cameraId={advisorCameraId}
          onCameraIdChange={onAdvisorCameraIdChange}
          hideScopePicker={Boolean(onAdvisorCameraIdChange)}
        />
      ) : null}
    </div>
  );
}

export function MotionPipelineRuntimeCard({ label, pipeline, origin, motionCatalog }) {
  if (!pipeline) return null;
  const metrics = Object.values(pipeline.stages || {});
  const calls = Math.max(0, ...metrics.map((item) => Number(item.calls) || 0));
  const failures = metrics.reduce((total, item) => total + (Number(item.failures) || 0), 0);
  const averageMs = metrics.reduce((total, item) => total + (Number(item.average_ms) || 0), 0);
  const lastMs = metrics.reduce((total, item) => total + (Number(item.last_ms) || 0), 0);
  const health = failures ? "attention" : calls ? "healthy" : "ready";
  const parallelGroups = (pipeline.execution_groups || []).filter((group) => group.mode === "parallel");
  const parallelStages = parallelGroups.reduce((total, group) => total + (group.stages?.length || 0), 0);
  const stageNames = new Map(
    (motionCatalog?.stages || []).map((stage) => [stage.implementation, stage.name]),
  );
  const originLabel = origin === "camera" ? "Camera override" : origin === "global" ? "Global setting" : "Built-in default";
  return (
    <div className={`motion-pipeline-runtime-card ${health}`}>
      <div className="motion-pipeline-runtime-head">
        <strong>{label}</strong>
        <span>{health === "attention" ? "Needs attention" : health === "healthy" ? "Healthy" : "Ready"}</span>
      </div>
      <small>{originLabel} · {pipeline.configuration?.length || 0} steps · {calls.toLocaleString()} cycles{parallelStages ? ` · ${parallelStages} parallel branches` : ""}</small>
      <div className="motion-pipeline-timing">
        <span>Last <strong>{lastMs.toFixed(2)} ms</strong></span>
        <span>Average <strong>{averageMs.toFixed(2)} ms</strong></span>
        <span>Failures <strong>{failures}</strong></span>
      </div>
      <details>
        <summary>Processing steps</summary>
        <div className="motion-pipeline-stage-list">
          {(pipeline.configuration || []).map((stage) => {
            const stageMetrics = pipeline.stages?.[stage.stage_id] || {};
            return (
              <div key={stage.stage_id}>
                <span><strong>{stageNames.get(stage.implementation) || stage.implementation}</strong><small>{stage.stage_id}</small></span>
                <span>{Number(stageMetrics.average_ms || 0).toFixed(2)} ms avg{stageMetrics.failures ? ` · ${stageMetrics.failures} failed` : ""}</span>
              </div>
            );
          })}
        </div>
      </details>
    </div>
  );
}

export function MotionDebugViewer({ cameraId, timeZone }) {
  const [status, setStatus] = useState(null);
  const [selectedLayer, setSelectedLayer] = useState("overlay");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const ownedRef = useRef(false);
  const debugRequestSequence = useRef(0);

  async function loadStatus(renew = false) {
    const sequence = ++debugRequestSequence.current;
    const response = await fetch(`/api/cameras/${encodeURIComponent(cameraId)}/motion-debug`, renew ? {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: true }),
    } : undefined);
    if (!response.ok) throw new Error("Could not load motion diagnostics");
    const payload = await response.json();
    if (sequence !== debugRequestSequence.current) return null;
    setStatus(payload);
    const layers = payload.snapshot?.layers || [];
    if (layers.length && !layers.some((layer) => layer.id === selectedLayer)) {
      setSelectedLayer(layers[0].id);
    }
    return payload;
  }

  useEffect(() => {
    let active = true;
    setStatus(null);
    setError("");
    fetch(`/api/cameras/${encodeURIComponent(cameraId)}/motion-debug`)
      .then((response) => response.ok ? response.json() : Promise.reject(new Error("Could not load motion diagnostics")))
      .then((payload) => { if (active) setStatus(payload); })
      .catch((loadError) => { if (active) setError(loadError.message); });
    return () => {
      active = false;
      debugRequestSequence.current += 1;
      if (ownedRef.current) {
        fetch(`/api/cameras/${encodeURIComponent(cameraId)}/motion-debug`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ enabled: false }),
          keepalive: true,
        }).catch(() => { });
        ownedRef.current = false;
      }
    };
  }, [cameraId]);

  const motionDebugInFlightRef = useRef(false);
  useVisiblePolling(async () => {
    if (motionDebugInFlightRef.current) return;
    motionDebugInFlightRef.current = true;
    try {
      await loadStatus(ownedRef.current && Number(status?.expires_in_seconds || 0) < 70);
    } catch (loadError) {
      setError(loadError.message);
    } finally {
      motionDebugInFlightRef.current = false;
    }
  }, 2000, Boolean(status?.enabled), { immediate: true, restartKey: cameraId });

  async function setEnabled(enabled) {
    setBusy(true);
    setError("");
    try {
      const response = await fetch(`/api/cameras/${encodeURIComponent(cameraId)}/motion-debug`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled }),
      });
      if (!response.ok) throw new Error("Could not update motion diagnostics");
      ownedRef.current = enabled;
      setStatus(await response.json());
    } catch (updateError) {
      setError(updateError.message);
    } finally {
      setBusy(false);
    }
  }

  const snapshot = status?.snapshot;
  const layers = snapshot?.layers || [];
  const imageUrl = snapshot && selectedLayer
    ? appUrl(`/api/cameras/${encodeURIComponent(cameraId)}/motion-debug/${encodeURIComponent(selectedLayer)}.jpg?t=${snapshot.captured_at}`)
    : "";
  return (
    <div className="sub-panel motion-debug-viewer">
      <div className="motion-debug-heading">
        <div>
          <h3>Motion Diagnostics</h3>
          <span>See what each processing step sees. Runs only for this camera and expires automatically.</span>
        </div>
        <button type="button" className={status?.enabled ? "danger" : ""} disabled={busy} onClick={() => setEnabled(!status?.enabled)}>
          {busy ? <RefreshCcw className="spin" size={15} /> : <Activity size={15} />}
          {status?.enabled ? "Stop Diagnostics" : "Start Diagnostics"}
        </button>
      </div>
      {error ? <div className="motion-analysis-warning">{error}</div> : null}
      {status?.last_error ? <div className="motion-analysis-warning">{status.last_error}</div> : null}
      {status?.enabled && !snapshot ? <div className="motion-debug-waiting"><RefreshCcw className="spin" size={16} /> Collecting the first diagnostic frame...</div> : null}
      {snapshot ? (
        <div className="motion-debug-content">
          <div className="motion-debug-image-panel">
            <label>Diagnostic view<select value={selectedLayer} onChange={(event) => setSelectedLayer(event.target.value)}>
              {layers.map((layer) => <option key={layer.id} value={layer.id}>{layer.label}</option>)}
            </select></label>
            {imageUrl ? <img src={imageUrl} alt={layers.find((layer) => layer.id === selectedLayer)?.label || "Motion diagnostic"} /> : null}
          </div>
          <div className="motion-debug-details">
            <strong>{snapshot.accepted ? "Motion accepted" : "Motion not accepted"}</strong>
            <span>{Math.round(Number(snapshot.score || 0) * 100)}% score · {Math.round(Number(snapshot.threshold || 0) * 100)}% needed</span>
            <span>{snapshot.frame_count || 0} frames · {snapshot.blob_count || 0} regions · {snapshot.track_points || 0} tracked points</span>
            <span>{snapshot.reason || "No reason reported"}</span>
            <span>{formatDateTime(new Date(Number(snapshot.captured_at) * 1000).toISOString(), timeZone)}</span>
            <details>
              <summary>Stage timing</summary>
              {Object.entries(snapshot.timings || {}).map(([stage, milliseconds]) => <span key={stage}>{stage}: {Number(milliseconds).toFixed(2)} ms</span>)}
            </details>
          </div>
        </div>
      ) : null}
    </div>
  );
}

export function MotionEffectiveness({ cameraId, mode }) {
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState("");

  const load = async () => {
    try {
      const response = await fetch("/api/motion-effectiveness?days=7");
      if (!response.ok) throw new Error("Effectiveness history unavailable");
      const payload = await response.json();
      setSummary(payload?.by_camera?.[cameraId]?.[mode] || null);
      setError("");
    } catch (loadError) {
      setError(loadError.message || "Effectiveness history unavailable");
    }
  };
  useVisiblePolling(load, 60000, true, { restartKey: `${cameraId}\u0000${mode}` });

  if (error) return <span className="motion-runtime-warning">{error}</span>;
  if (!summary) return <span>No durable motion decisions for this mode in the last 7 days.</span>;
  return (
    <div className="motion-effectiveness-summary">
      <strong>Last 7 days in this mode</strong>
      <span>{summary.allowed_events || 0} allowed · {summary.visual_filtered || 0} visually filtered · {summary.state_deduplicated || 0} merged with ongoing activity</span>
      <span>{summary.object_events || 0} allowed events found a configured object · {summary.no_object_events || 0} found none</span>
      <span>{Math.round(Number(summary.visual_rejection_rate || 0) * 100)}% visually filtered · {Math.round(Number(summary.object_yield_rate || 0) * 100)}% object yield · {summary.borderline_rescued || 0} borderline rescues</span>
      <span>{summary.suppression_verification_checks || 0} filtered events double-checked · {summary.suppression_verification_rescues || 0} restored after finding an object</span>
      {summary.visual_backup_attempts ? <span>{summary.visual_backup_attempts} visual backup attempts · {summary.visual_backup_objects || 0} found an object · {summary.visual_backup_no_object || 0} found none{summary.visual_backup_incomplete ? ` · ${summary.visual_backup_incomplete} incomplete` : ""}</span> : null}
      {summary.unreviewed_visual_filters ? <span className="motion-runtime-warning">{summary.unreviewed_visual_filters} visual filters were not independently checked by object detection.</span> : null}
    </div>
  );
}

export function DepthShadowPerformance({ cameraId = "", mode = "", label = "Depth shadow · last 24 hours" }) {
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState("");

  const load = async () => {
    try {
      const response = await fetch("/api/motion-effectiveness?days=1");
      if (!response.ok) throw new Error("Depth shadow history unavailable");
      const payload = await response.json();
      const cameras = cameraId ? { [cameraId]: payload?.by_camera?.[cameraId] || {} } : payload?.by_camera || {};
      const totals = { decisions: 0, objects_evaluated: 0, valid_depth: 0, near_depth: 0, would_admit: 0, alignment_reliable: 0, spatial_match: 0, stable_geometry: 0, correlation_accepted: 0, correlation_rejected: 0 };
      Object.entries(cameras).forEach(([, modes]) => Object.entries(modes || {}).forEach(([entryMode, value]) => {
        if (mode && entryMode !== mode) return;
        const depth = value?.depth_shadow || {};
        Object.keys(totals).forEach((key) => { totals[key] += Number(depth[key] || 0); });
      }));
      setSummary(totals);
      setError("");
    } catch (loadError) {
      setError(loadError.message || "Depth shadow history unavailable");
    }
  };
  useVisiblePolling(load, 30000, true, { restartKey: `${cameraId}\u0000${mode}` });

  if (error) return <span className="motion-runtime-warning">{error}</span>;
  if (!summary?.decisions) return <span>Depth shadow: no recorded-frame correlations yet. This does not indicate a depth failure.</span>;
  return <div className="motion-effectiveness-summary">
    <strong>{label}</strong>
    <span>{summary.decisions.toLocaleString()} decisions · {summary.correlation_accepted.toLocaleString()} correlation accepted · {summary.correlation_rejected.toLocaleString()} correlation rejected</span>
    <span>{summary.objects_evaluated.toLocaleString()} objects sampled · {summary.valid_depth.toLocaleString()} valid depth · {summary.near_depth.toLocaleString()} near</span>
    <span>{summary.alignment_reliable.toLocaleString()} aligned · {summary.spatial_match.toLocaleString()} spatially matched · {summary.stable_geometry.toLocaleString()} geometrically stable</span>
    <span>{summary.would_admit.toLocaleString()} would meet the experimental depth rule</span>
    <span>Informational only — depth does not affect incidents, EMA, or fusion.</span>
  </div>;
}

export function RuntimeStatus({ status, timeZone, motionCatalog }) {
  if (!status) {
    return <div className="probe-result"><strong>Runtime</strong><span>Save this camera to start workers.</span></div>;
  }
  const motionMode = status.motion_qualification?.mode;
  const cameraAlertsOnly = !["adaptive", "enforce"].includes(motionMode);
  const visualBackupEnabled = motionMode === "camera_rescue";
  const missingMotionNotices = cameraAlertsOnly
    && status.onvif_enabled
    && Number(status.onvif_motion_events_received || 0) === 0;
  const missingCameraTrigger = cameraAlertsOnly && !status.onvif_enabled;
  return (
    <div className="probe-result runtime-result">
      <strong>Runtime</strong>
      <span>Stream worker: {status.running ? "running" : "not running"}</span>
      <span>Live source: {status.live_pipeline?.source_element || "not reported yet"}{status.live_pipeline?.model_instance_id ? ` · ${status.live_pipeline.model_instance_id}` : ""}</span>
      <span>Recording: {status.recording ? "running" : "stopped"}</span>
      <span>ONVIF: {status.onvif_enabled ? (status.onvif_connected ? "connected" : `not connected${status.onvif_last_error ? `: ${status.onvif_last_error}` : ""}`) : "disabled"}</span>
      {status.onvif_last_event_at ? <span>Last ONVIF notification (any type): {formatDateTime(status.onvif_last_event_at, timeZone)}</span> : null}
      {status.onvif_enabled ? <span>{status.onvif_notifications_received || 0} notifications · {status.onvif_motion_events_received || 0} active motion · {status.onvif_inactive_motion_events || 0} inactive motion · {status.onvif_renewals || 0} subscription renewals</span> : null}
      {status.motion_qualification ? (
        <div className="motion-runtime-status">
          <div className="motion-runtime-summary">
            <strong>Motion processing</strong>
            <span>{motionModeInfo(status.motion_qualification.mode).status} · {status.motion_qualification.sensitivity} sensitivity · {status.motion_qualification.frame_width || 320}px</span>
            <span>{status.motion_qualification.passed || 0} accepted · {status.motion_qualification.audit_rejected || 0} legacy preview rejects · {status.motion_qualification.suppressed || 0} filtered</span>
            <span>{status.motion_qualification.continuous_frames || 0} visual frames analyzed · {status.motion_qualification.continuous_candidates || 0} accepted analysis frames · {status.motion_qualification.triggers || 0} triggers delivered · {status.motion_qualification.analysis_frames_dropped || 0} stale requests replaced</span>
            <span>Capture-to-analysis p95 {formatMilliseconds(status.motion_qualification.analysis_runtime?.capture_to_analysis_p95_ms)} · preprocessing p99 {formatMilliseconds(status.motion_qualification.analysis_runtime?.preprocess_p99_ms)} · {formatBytes(status.motion_qualification.analysis_runtime?.copy_bytes || 0)} copied for motion analysis</span>
            <span>Light and shadow filtering {status.motion_qualification.illumination_filter_enabled ? "enabled" : "measuring only"} · {status.motion_qualification.illumination_evaluations || 0} evaluated · {status.motion_qualification.illumination_candidates || 0} likely illumination changes · {status.motion_qualification.illumination_filtered || 0} filtered</span>
            <span>{status.motion_qualification.validation_failures || 0} validator errors · {status.motion_qualification.validation_fail_opens || 0} allowed through safely</span>
            <span>{status.motion_qualification.active_followup_triggers || 0} active-event follow-ups · {status.motion_qualification.active_followup_objects || 0} found an object · {status.motion_qualification.active_followup_no_object || 0} found none · {status.motion_qualification.active_followup_episode_limited || 0} held by the episode limit</span>
            {missingCameraTrigger ? <span className="motion-runtime-warning">{visualBackupEnabled ? "ONVIF is disabled, so the conservative visual backup is the only automatic trigger. Restore ONVIF for primary coverage." : "ONVIF is disabled. Camera-triggered mode has no automatic trigger source; only manual tests can run object detection."}</span> : null}
            {missingMotionNotices ? <span className="motion-runtime-warning">{visualBackupEnabled ? "No recognized ONVIF motion notices since this worker started. Strong persistent visual motion can still invoke the backup detector path." : "No recognized ONVIF motion notices since this worker started. In this mode, visual analysis alone cannot create an incident."}</span> : null}
            {visualBackupEnabled ? <>
              <span>{status.motion_qualification.visual_backup?.scene_ready ? "EMA background ready" : "EMA learning scene"} · {status.motion_qualification.visual_backup_triggers || 0} visual backups · {status.motion_qualification.visual_backup_onvif_matches || 0} strong candidates matched to camera notices · {status.motion_qualification.visual_backup_rate_limited || 0} limited</span>
              <span>{status.motion_qualification.visual_backup_not_ready || 0} strong candidates held during scene learning · {status.motion_qualification.visual_backup_uncorrelated_objects || 0} detected objects outside motion areas rejected</span>
            </> : null}
            <MotionEffectiveness cameraId={status.id} mode={status.motion_qualification.mode} />
            <DepthShadowPerformance cameraId={status.id} mode={status.motion_qualification.mode} label="Depth shadow · current mode · last 24 hours" />
          </div>
          <div className="motion-pipeline-runtime-grid">
            <MotionPipelineRuntimeCard label="Motion analysis" pipeline={status.motion_qualification.pipeline} origin={status.motion_qualification.pipeline_origins?.qualification} motionCatalog={motionCatalog} />
            <MotionPipelineRuntimeCard label="Extra sources" pipeline={status.motion_qualification.observation_pipeline} origin={status.motion_qualification.pipeline_origins?.observation} motionCatalog={motionCatalog} />
            <MotionPipelineRuntimeCard label="Decision" pipeline={status.motion_qualification.fusion_pipeline} origin={status.motion_qualification.pipeline_origins?.fusion} motionCatalog={motionCatalog} />
          </div>
          <div className="motion-evidence-runtime">
            {Object.entries(status.motion_qualification.evidence_sources || {}).map(([source, evidence]) => (
              <span key={source} className={evidence.enabled ? "enabled" : "disabled"}>
                <strong>{source === "onvif" ? "Camera signal" : source === "depth_object" ? "Depth evidence" : source}</strong>
                {evidence.enabled ? `${evidence.sample_count || 0} samples${evidence.last?.score != null ? ` · ${Math.round(Number(evidence.last.score) * 100)}% last confidence` : ""}${source === "depth_object" && evidence.last?.nearest_m != null ? ` · nearest ${Number(evidence.last.nearest_m).toFixed(1)}m` : ""}` : "Disabled"}
              </span>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}
