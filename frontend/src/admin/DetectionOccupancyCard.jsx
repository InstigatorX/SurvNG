import React, { useState } from "react";
import { ArrowRight, CircleAlert, CircleCheck, CircleDot, CircleHelp, Cpu } from "lucide-react";
import { useVisiblePolling } from "../visibilityPolling.mjs";
import { fetch } from "../shared/api.js";
import {
  OCCUPANCY_TONES,
  buildOccupancyReport,
  cameraEffectiveness,
  coverageFromCameraMotion,
  coverageFromRuntimeHistory,
  occupancyToneLabel,
  resolveObjectWorkerCount,
  siteEffectiveness,
} from "../detectionOccupancy.mjs";

const TONE_ICONS = {
  [OCCUPANCY_TONES.good]: CircleCheck,
  [OCCUPANCY_TONES.warning]: CircleAlert,
  [OCCUPANCY_TONES.bad]: CircleAlert,
  [OCCUPANCY_TONES.idle]: CircleHelp,
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

function cameraOnvifHealthy(camera) {
  if (camera?.onvif?.enabled === false) return false;
  if (camera?.onvif && camera.onvif.connected === false) return false;
  return true;
}

function OccupancyRow({ row, onOpenSetting }) {
  const Icon = TONE_ICONS[row.tone] || CircleDot;
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
      <p className="occupancy-suggestion"><span>What to do</span>{row.suggestion}</p>
      {row.setting && onOpenSetting ? (
        <button type="button" className="occupancy-setting-link" onClick={() => onOpenSetting(row.setting)}>
          Open {row.setting.label} <ArrowRight size={14} />
        </button>
      ) : null}
    </article>
  );
}

function OccupancyPanel({ title, subtitle, report, onOpenSetting, primary = false, icon: HeaderIcon = Cpu }) {
  return (
    <section className={`detection-settings-card occupancy-card ${primary ? "primary" : ""} ${report.tone}`}>
      <header className="detection-settings-card-head">
        <div className="detection-settings-card-icon"><HeaderIcon size={18} /></div>
        <div>
          <h3>{title}</h3>
          <p>{subtitle}</p>
        </div>
        <span className={`occupancy-tone-chip ${report.tone}`}>{occupancyToneLabel(report.tone)}</span>
      </header>
      <div className="occupancy-row-grid">
        {report.rows.map((row) => (
          <OccupancyRow key={row.id} row={row} onOpenSetting={onOpenSetting} />
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
  const siteCoverage = {
    ...coverageHistory,
    deferred: cameras.reduce((total, camera) => total + coverageFromCameraMotion(camera).deferred, 0),
  }

  const byCamera = effectiveness?.by_camera || {};
  const slotCount = Number(config?.motion_qualification?.max_concurrent_analysis ?? 2);
  const trackingEnabled = config?.detector?.tracking?.enabled !== false;
  const { configured: configuredWorkerCount, running: runningWorkerCount } = resolveObjectWorkerCount({
    config,
    telemetry,
  });
  const workerCount = runningWorkerCount || configuredWorkerCount;
  const backend = config?.detector?.backend || "openvino";
  const backupEnabled = (config?.motion_qualification?.mode || "camera_rescue") === "camera_rescue"
    || (config?.motion_qualification?.mode || "") === "adaptive";
  const siteOnvifHealthy = cameras.length
    ? cameras.every((camera) => cameraOnvifHealthy(camera))
    : true;

  const siteReport = buildOccupancyReport({
    coverage: selected ? coverageFromCameraMotion(selected) : siteCoverage,
    effectiveness: selected
      ? cameraEffectiveness(byCamera, selected.id, cameraMode(config, configCameras.find((camera) => camera.id === selected.id)))
      : siteEffectiveness(byCamera, configCameras),
    slotCount,
    trackingEnabled,
    workerCount,
    configuredWorkerCount,
    runningWorkerCount,
    backend,
    requireZone: selected
      ? cameraRequiresZone(config, configCameras.find((camera) => camera.id === selected.id) || selected)
      : config?.detector?.require_incident_zone !== false,
    backupEnabled: selected
      ? ["camera_rescue", "adaptive"].includes(cameraMode(config, configCameras.find((camera) => camera.id === selected.id)))
      : backupEnabled,
    onvifHealthy: selected ? cameraOnvifHealthy(selected) : siteOnvifHealthy,
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
      onvifHealthy: cameraOnvifHealthy(camera),
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
          ? `${selected.name || selected.id} · last 7 days of incidents, plus current visual-analysis load`
          : "Last 7 days of incidents, plus current visual-analysis load. Green means leave it. Amber or red says what to change."}
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
                rows: report.rows.filter((row) => row.tone === OCCUPANCY_TONES.warning || row.tone === OCCUPANCY_TONES.bad),
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
