import React, { useState } from "react";
import { Activity, ArrowRight, CircleAlert, CircleCheck, CircleDot, CircleHelp, Cpu, Gauge } from "lucide-react";
import { useVisiblePolling } from "../visibilityPolling.mjs";
import { fetch } from "../shared/api.js";
import {
  OCCUPANCY_TONES,
  backupCorrelation,
  buildOccupancyReport,
  cameraEffectiveness,
  cameraOnvifHealthy,
  coverageFromCameraMotion,
  coverageFromRuntimeHistory,
  occupancyToneLabel,
  resolveDetectorHealth,
  siteEffectiveness,
  siteOnvifHealthy,
} from "../detectionOccupancy.mjs";

const TONE_ICONS = {
  [OCCUPANCY_TONES.good]: CircleCheck,
  [OCCUPANCY_TONES.warning]: CircleAlert,
  [OCCUPANCY_TONES.bad]: CircleAlert,
  [OCCUPANCY_TONES.idle]: CircleHelp,
};

const PILLAR_ICONS = {
  admission: CircleDot,
  engine: Cpu,
  tracking: Activity,
  capacity: Gauge,
  waste: CircleAlert,
};

function cameraMode(config, camera) {
  const override = camera?.motion_qualification?.mode;
  if (override && override !== "inherit") return override;
  return config?.motion_qualification?.mode || "camera_rescue";
}

function cameraRequiresZone(config, camera) {
  if (camera?.require_incident_zone != null) return Boolean(camera.require_incident_zone);
  return config?.detector?.require_incident_zone !== false;
}

function backupGraceSeconds(config, camera) {
  const inherited = Number(config?.motion_qualification?.visual_backup_grace_seconds);
  const override = camera?.motion_qualification?.visual_backup_grace_seconds;
  const value = override == null || override === "" ? inherited : Number(override);
  return Number.isFinite(value) && value >= 0 ? value : 1.5;
}

function OccupancyRow({ row, onOpenSetting, compact = false }) {
  const Icon = PILLAR_ICONS[row.id] || TONE_ICONS[row.tone] || CircleDot;
  const showAction = !compact && row.tone !== OCCUPANCY_TONES.good && row.tone !== OCCUPANCY_TONES.idle;
  return (
    <article className={`occupancy-row ${row.tone}`}>
      <div className="occupancy-row-head">
        <Icon size={18} aria-hidden="true" />
        <div>
          <strong>{row.title}</strong>
          <em>{occupancyToneLabel(row.tone)}</em>
        </div>
      </div>
      <p className="occupancy-headline">{row.headline}</p>
      <p className="occupancy-detail">{row.detail}</p>
      {showAction ? <p className="occupancy-suggestion"><span>What to do</span>{row.suggestion}</p> : null}
      {row.setting && onOpenSetting ? (
        <button type="button" className="occupancy-setting-link" onClick={() => onOpenSetting(row.setting)}>
          Open {row.setting.label} <ArrowRight size={14} />
        </button>
      ) : null}
    </article>
  );
}

function OccupancyPanel({ title, subtitle, report, onOpenSetting, primary = false, icon: HeaderIcon = Cpu }) {
  const heading = primary && report.summary?.headline ? report.summary.headline : title;
  const description = report.summary?.detail || subtitle;
  const cards = report.pillars || report.rows || [];
  return (
    <section className={`detection-settings-card occupancy-card ${primary ? "primary" : ""} ${report.tone}`}>
      <header className="detection-settings-card-head">
        <div className="detection-settings-card-icon"><HeaderIcon size={18} /></div>
        <div>
          <h3>{heading}</h3>
          <p>{description}</p>
        </div>
        <span className={`occupancy-tone-chip ${report.tone}`}>{occupancyToneLabel(report.tone)}</span>
      </header>
      {primary && report.context ? <p className="occupancy-context">{report.context}. {subtitle}</p> : null}
      <div className="occupancy-row-grid occupancy-pillar-grid">
        {cards.map((row) => (
          <OccupancyRow
            key={row.id}
            row={row}
            compact={row.tone === OCCUPANCY_TONES.good || row.tone === OCCUPANCY_TONES.idle}
            onOpenSetting={onOpenSetting}
          />
        ))}
      </div>
    </section>
  );
}

export function DetectionOccupancyCard({
  telemetry,
  cameraId,
  config,
  onOpenSetting,
}) {
  const [effectiveness, setEffectiveness] = useState(null);
  const [error, setError] = useState("");
  const cameras = telemetry?.cameras || [];
  const selected = cameraId ? cameras.find((camera) => camera.id === cameraId) : null;
  const configCameras = config?.cameras || [];

  useVisiblePolling(async () => {
    const response = await fetch("/api/motion-effectiveness?days=7");
    if (!response.ok) throw new Error("Motion history unavailable");
    const payload = await response.json();
    setEffectiveness(payload);
    setError("");
  }, 60000, true, { restartKey: String(cameraId || "") });

  const coverageHistory = coverageFromRuntimeHistory(telemetry?.runtime_history?.short || []);
  const liveCoverage = cameras.map((camera) => coverageFromCameraMotion(camera));
  const siteCoverage = {
    ...coverageHistory,
    deferred: liveCoverage.reduce((total, item) => total + item.deferred, 0),
    analysisWaitP95Ms: liveCoverage.reduce((max, item) => Math.max(max, item.analysisWaitP95Ms || 0), 0),
    captureToAnalysisP95Ms: liveCoverage.reduce((max, item) => Math.max(max, item.captureToAnalysisP95Ms || 0), 0),
  };

  const byCamera = effectiveness?.by_camera || {};
  const slotCount = Number(config?.motion_qualification?.max_concurrent_analysis ?? 2);
  const trackingEnabled = config?.detector?.tracking?.enabled !== false;
  const detectorHealth = resolveDetectorHealth({ config, telemetry });
  const { configured: configuredWorkerCount, running: runningWorkerCount } = detectorHealth;
  const workerCount = runningWorkerCount || configuredWorkerCount;
  const backend = detectorHealth.backend || config?.detector?.backend || "openvino";
  const backupEnabled = (config?.motion_qualification?.mode || "camera_rescue") === "camera_rescue"
    || (config?.motion_qualification?.mode || "") === "adaptive";
  const selectedConfigCamera = selected
    ? configCameras.find((camera) => camera.id === selected.id)
    : null;
  const correlation = backupCorrelation(selected ? [selected] : cameras);
  const graceSeconds = backupGraceSeconds(config, selectedConfigCamera);

  const siteReport = buildOccupancyReport({
    coverage: selected ? coverageFromCameraMotion(selected) : siteCoverage,
    effectiveness: selected
      ? cameraEffectiveness(byCamera, selected.id, cameraMode(config, selectedConfigCamera))
      : siteEffectiveness(byCamera, configCameras),
    slotCount,
    trackingEnabled,
    workerCount,
    configuredWorkerCount,
    runningWorkerCount,
    backend,
    requireZone: selected
      ? cameraRequiresZone(config, selectedConfigCamera || selected)
      : config?.detector?.require_incident_zone !== false,
    backupEnabled: selected
      ? ["camera_rescue", "adaptive"].includes(cameraMode(config, selectedConfigCamera))
      : backupEnabled,
    onvifHealthy: selected ? cameraOnvifHealthy(selected) !== false : siteOnvifHealthy(cameras),
    backupMatchedNotices: correlation.matches,
    backupWithoutNotices: correlation.without,
    backupGraceSeconds: graceSeconds,
    detectorHealth,
    includeDetectorHealth: true,
  });

  const attentionCameras = selected ? [] : cameras.map((camera) => {
    const configCamera = configCameras.find((item) => item.id === camera.id);
    const report = buildOccupancyReport({
      coverage: coverageFromCameraMotion(camera),
      effectiveness: cameraEffectiveness(byCamera, camera.id, cameraMode(config, configCamera)),
      slotCount,
      trackingEnabled,
      workerCount,
      configuredWorkerCount,
      runningWorkerCount,
      backend,
      requireZone: cameraRequiresZone(config, configCamera || camera),
      backupEnabled: ["camera_rescue", "adaptive"].includes(cameraMode(config, configCamera)),
      onvifHealthy: cameraOnvifHealthy(camera) !== false,
      backupMatchedNotices: backupCorrelation([camera]).matches,
      backupWithoutNotices: backupCorrelation([camera]).without,
      backupGraceSeconds: backupGraceSeconds(config, configCamera),
      includeDetectorHealth: false,
    });
    return { camera, report };
  }).filter((item) => item.report.tone === OCCUPANCY_TONES.warning || item.report.tone === OCCUPANCY_TONES.bad);

  return (
    <div className="occupancy-workspace wide-card" aria-label="Detection occupancy">
      {error ? <p className="occupancy-error">{error}</p> : null}
      <OccupancyPanel
        primary
        title="Detection at a glance"
        subtitle={selected
          ? `${selected.name || selected.id} · last 7 days of incidents plus live detector health`
          : "Last 7 days of incidents plus live detector health"}
        report={siteReport}
        onOpenSetting={onOpenSetting}
      />
      {attentionCameras.length ? (
        <div className="occupancy-attention">
          <h4>Cameras that need a look</h4>
          {attentionCameras.map(({ camera, report }) => (
            <OccupancyPanel
              key={camera.id}
              title={camera.name || camera.id}
              subtitle="Only the checks that are not green"
              report={{
                ...report,
                summary: null,
                pillars: (report.pillars || []).filter((row) => (
                  row.tone === OCCUPANCY_TONES.warning || row.tone === OCCUPANCY_TONES.bad
                )),
              }}
              icon={CircleAlert}
              onOpenSetting={(setting) => onOpenSetting?.({
                ...setting,
                workspace: setting.workspace === "general" && setting.detectionSection === "motion"
                  ? "cameras"
                  : setting.workspace,
                subsection: setting.workspace === "general" && setting.detectionSection === "motion"
                  ? "motion"
                  : setting.subsection,
                cameraId: camera.id,
              })}
            />
          ))}
        </div>
      ) : null}
    </div>
  );
}
