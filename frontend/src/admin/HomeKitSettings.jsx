import React, { useEffect, useRef, useState } from "react";
import { Home, KeyRound, RotateCcw } from "lucide-react";
import { fetch, appUrl } from "../shared/api.js";
import "./homekit.css";

const ERROR_LABELS = {
  state_in_use: "Waiting for the previous HomeKit camera worker to finish shutting down.",
  software_encoding_not_allowed: "This server needs an application restart to enable hardware-only HomeKit startup.",
  encoding_unavailable: "No permitted HEVC encoder passed its startup test. Check QSV device access or allow software fallback.",
  worker_not_built: "Build the HomeKit worker before enabling cameras.",
  node_24_required: "HomeKit requires Node.js 24. Set its executable path below.",
  ffmpeg_pacing_unsupported: "Use a server FFmpeg build that supports -readrate_initial_burst.",
  encoder_unavailable: "FFmpeg needs HEVC encoding plus libopus, AAC, and MJPEG support.",
  encoder_offline: "The assigned encoder is offline. HomeKit stays on this server and is not using a local encoder.",
  encoder_waiting: "Waiting for the encoder to measure this camera.",
  encoder_software_not_allowed: "This encoder has no hardware HEVC. Allow software encoding or choose another encoder.",
  bind_address_unavailable: "The LAN address is not available on this server. Check the address below.",
  runtime_or_state_unavailable: "Check the Node/FFmpeg paths and private HomeKit state permissions.",
  worker_start_failed: "The camera worker could not start. Check the LAN address and port.",
  worker_stopped: "The camera worker stopped. Retrying; check its LAN address, port, and runtime requirements.",
  control_unavailable: "Waiting for the HomeKit camera to become ready…",
  camera_off: "Turn this camera on in SurvNG to resume HomeKit.",
};

const UPDATE_LABELS = {
  current: "Up to date",
  outdated: "Update available",
  unknown: "Running version not reported",
  unavailable: "No release available",
  syncing: "Syncing encoder…",
  failed: "Encoder update failed",
};

function EncoderVersion({ label, version, fallback }) {
  if (!version) return <span>{label} {fallback}</span>;
  return <details>
    <summary>{label} <code title={version}>{version.slice(0, 12)}</code></summary>
    <code className="homekit-node-digest">{version}</code>
  </details>;
}

function HomeKitEncoder({ node }) {
  const [request, setRequest] = useState(null);
  const requestController = useRef(null);
  const state = node.updateState || "unknown";
  const canSync = node.availableVersion && ["outdated", "unknown", "failed"].includes(state);
  const awaitingStatus = request?.phase === "accepted" && node === request.node;
  const syncing = request?.phase === "sending" || awaitingStatus || state === "syncing";
  const statusLabel = request?.phase === "sending" ? "Requesting sync…" : awaitingStatus ? "Sync requested" : UPDATE_LABELS[state] || UPDATE_LABELS.unknown;

  useEffect(() => () => requestController.current?.abort(), []);
  useEffect(() => {
    if (request?.phase !== "accepted") return undefined;
    if (!awaitingStatus) {
      setRequest(null);
      return undefined;
    }
    const timeout = setTimeout(() => setRequest({ phase: "unconfirmed" }), 12000);
    return () => clearTimeout(timeout);
  }, [request, awaitingStatus]);

  async function syncEncoder() {
    if (requestController.current || !canSync || !node.online || syncing) return;
    const controller = new AbortController();
    requestController.current = controller;
    setRequest({ phase: "sending" });
    try {
      const response = await fetch(`/api/homekit/nodes/${encodeURIComponent(node.id)}/resync`, {
        method: "POST", cache: "no-store", signal: AbortSignal.any([controller.signal, AbortSignal.timeout(15000)]),
      });
      if (!response.ok || (await response.json()).requested !== true) throw new Error("Sync request rejected");
      setRequest({ phase: "accepted", node });
    } catch {
      if (!controller.signal.aborted) setRequest({ phase: "error" });
    } finally {
      if (requestController.current === controller) requestController.current = null;
    }
  }

  return <article className="sub-panel homekit-node" aria-label={node.name || "Encoder"}>
    <div className="homekit-camera-head">
      <h4>{node.name || "Encoder"}</h4>
      <span className="homekit-badge">{node.online ? "Online" : "Offline"}</span>
    </div>
    <p className="settings-help">{node.hardware === "qsv" ? "Intel QSV HEVC" : node.hardware === "software" ? "Software HEVC" : "Encoder type not reported yet"}{node.cameras?.length ? ` · ${node.cameras.length} ${node.cameras.length === 1 ? "camera" : "cameras"}` : ""}</p>
    <div className="settings-help homekit-node-versions">
      <EncoderVersion label="Running version" version={node.runningVersion} fallback="Not reported" />
      <EncoderVersion label="Available version" version={node.availableVersion} fallback="Unavailable" />
    </div>
    <div className="homekit-node-action">
      <p className="settings-help" role="status">{statusLabel}</p>
      {canSync || state === "syncing" ? <button type="button" disabled={!node.online || syncing} onClick={syncEncoder}>Sync now</button> : null}
    </div>
    {request?.phase === "error" && canSync && !syncing ? <p className="settings-help homekit-error" role="alert">Could not request an encoder sync. Try again.</p> : null}
    {request?.phase === "unconfirmed" && canSync && !syncing ? <p className="settings-help homekit-error" role="alert">The encoder has not confirmed the sync. Check its status or try again.</p> : null}
  </article>;
}

export function HomeKitSettings({ config, updateConfig, cameraId = "" }) {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState("");
  const [pairing, setPairing] = useState(null);
  const [busy, setBusy] = useState("");
  const [installer, setInstaller] = useState("");
  const pairingRequest = useRef(null);
  const hidePairing = () => {
    pairingRequest.current?.abort();
    pairingRequest.current = null;
    setPairing(null);
  };
  const settings = { enabled: false, bind: "", node_path: "node", ...config.homekit };
  const cameras = config.cameras || [];
  const update = (key, value) => { hidePairing(); updateConfig(["homekit"], { ...settings, [key]: value }); };
  const updateCamera = (index, key, value) => {
    hidePairing();
    updateConfig(["cameras", index, "homekit"], { ...cameras[index].homekit, [key]: value });
  };
  useEffect(() => {
    const controller = new AbortController();
    let timer;
    const refresh = async () => {
      try {
        const response = await fetch("/api/homekit/status", { signal: controller.signal, cache: "no-store" });
        if (!response.ok) throw new Error("Unable to load HomeKit status.");
        const next = await response.json();
        setStatus(next); setError("");
        const eligible = next.cameras.filter((camera) => camera.state === "running" && !camera.paired);
        if (pairingRequest.current?.cameraId && !eligible.some(camera => camera.id === pairingRequest.current.cameraId)) pairingRequest.current.abort();
        setPairing((current) => current && eligible.some((camera) => camera.id === current.id) ? current : null);
      } catch (failure) {
        if (!controller.signal.aborted) { setError(failure.message || "HomeKit status unavailable."); setStatus(null); hidePairing(); }
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(refresh, 3000);
      }
    };
    void refresh();
    return () => { controller.abort(); pairingRequest.current?.abort(); clearTimeout(timer); };
  }, []);
  useEffect(() => {
    if (!pairing) return undefined;
    const timer = setTimeout(() => setPairing(null), 120000);
    return () => clearTimeout(timer);
  }, [pairing]);
  useEffect(() => {
    const hide = () => { if (document.hidden) hidePairing(); };
    document.addEventListener("visibilitychange", hide);
    return () => document.removeEventListener("visibilitychange", hide);
  }, []);

  async function addEncoder() {
    setBusy("encoder"); setError(""); setInstaller("");
    try {
      const response = await fetch("/api/homekit/nodes", { method: "POST", headers: { "Content-Type": "application/json" }, cache: "no-store", body: "{}" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Could not create an encoder installer.");
      setInstaller(payload.command || "");
    } catch (failure) { setError(failure.message || "Could not create an encoder installer."); }
    finally { setBusy(""); }
  }

  async function cameraAction(camera, action) {
    if (action === "reset-pairing" && !window.confirm(`Reset HomeKit pairing for ${camera.name}? Remove its old tile from Apple Home, then pair it again. Other cameras keep their pairing.`)) return;
    setBusy(camera.id); setError(""); setPairing(null);
    const controller = new AbortController();
    if (action === "pairing") {
      controller.cameraId = camera.id;
      pairingRequest.current?.abort();
      pairingRequest.current = controller;
    }
    try {
      const response = await fetch(`/api/homekit/cameras/${encodeURIComponent(camera.id)}/${action}`, {
        method: "POST", headers: { "Content-Type": "application/json" }, cache: "no-store",
        signal: controller.signal,
        body: JSON.stringify(action === "reset-pairing" ? { confirm_camera_id: camera.id } : {}),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "HomeKit request failed.");
      if (action === "pairing" && !controller.signal.aborted && !document.hidden) setPairing({ id: camera.id, name: camera.name, ...payload });
    } catch (failure) { if (!controller.signal.aborted) setError(failure.message || "HomeKit request failed."); }
    finally { if (pairingRequest.current === controller) pairingRequest.current = null; setBusy(""); }
  }

  if (cameraId) {
    const index = cameras.findIndex((camera) => camera.id === cameraId);
    if (index < 0) return null;
    const camera = cameras[index];
    const options = { enabled: false, allow_software_encoding: false, source: "live", audio: false, object_motion: false, port: 0, encoder: "local", ...camera.homekit };
    const live = status?.cameras?.find((item) => item.id === camera.id);
    const available = settings.enabled && options.enabled && live?.state === "running";
    return <section className="sub-panel homekit-settings homekit-camera camera-homekit-panel" aria-labelledby={`camera-homekit-title-${camera.id}`}>
      <div className="homekit-camera-head"><h3 id={`camera-homekit-title-${camera.id}`}><Home size={15} />HomeKit</h3><span className="homekit-badge">{live?.paired ? "Paired" : live?.state === "running" ? "Ready to pair" : live?.state || "Not enabled"}</span></div>
      <p className="settings-help">Save camera settings to apply changes, then pair with Apple Home.</p>
      {!settings.enabled ? <p className="settings-help">Enable the service in <a href={appUrl("/admin?section=general&subsection=homekit")}>HomeKit settings</a> to start this camera.</p> : null}
      {error ? <p role="alert" className="settings-help homekit-error">{error} Status retries automatically.</p> : null}
      {!status && !error ? <p role="status">Loading HomeKit status…</p> : null}
      <label className="check-field"><input type="checkbox" checked={options.enabled} onChange={(event) => updateCamera(index, "enabled", event.target.checked)} />Enable HomeKit for this camera</label>
      {options.enabled ? <>
        <label className="check-field"><input type="checkbox" checked={options.allow_software_encoding} onChange={(event) => updateCamera(index, "allow_software_encoding", event.target.checked)} />Allow software HEVC fallback</label>
        <p className="settings-help">QSV is preferred automatically. This permits software encoding when QSV is unavailable. Viewers and recording share an encoder for the same video profile.</p>
        {live?.runtime?.encoder ? <p className="settings-help" role="status">Encoder: <strong>{live.runtime.encoder.backend === "qsv" ? "Intel QSV HEVC" : "Software HEVC"}</strong>{live.runtime.encoder.sources?.[0]?.fps != null ? ` · ${Number(live.runtime.encoder.sources[0].fps).toFixed(1)} fps` : ""}{live.runtime.encoder.fallbackReason ? " · QSV unavailable; using permitted software fallback" : ""}</p> : null}
        <div className="homekit-fields">
          <label>Encoder<select value={options.encoder || "local"} onChange={(event) => updateCamera(index, "encoder", event.target.value)}><option value="local">This server</option>{(status?.nodes || []).map((node) => <option key={node.id} value={node.id}>{node.name || "Encoder"} ({node.online ? "online" : "offline"})</option>)}{options.encoder !== "local" && !(status?.nodes || []).some((node) => node.id === options.encoder) ? <option value={options.encoder}>Assigned encoder</option> : null}</select></label>
          <label>Video source<select value={options.source} onChange={(event) => updateCamera(index, "source", event.target.value)}><option value="live">Live stream</option><option value="main">Main stream</option></select></label>
          <label className="check-field"><input type="checkbox" checked={options.audio} onChange={(event) => updateCamera(index, "audio", event.target.checked)} />Include camera audio</label>
        </div>
        <label className="check-field"><input type="checkbox" checked={options.object_motion} onChange={(event) => updateCamera(index, "object_motion", event.target.checked)} />Trigger HomeKit motion from object detection</label>
        <p className="settings-help">Recent objects recognized by SurvNG trigger a ten-second motion window. Turn Detection on for this camera. Recordings also require Stream &amp; Allow Recording in Apple Home. Leave Home activity zones off for these motion events.</p>
        {live?.error ? <p className="settings-help homekit-error">{ERROR_LABELS[live.error] || "HomeKit is unavailable."}</p> : null}
        {live?.runtime?.zonesBlockMotion ? <p className="settings-help homekit-error">Home activity zones are suppressing object-triggered motion. Disable those zones in Apple Home to use this trigger.</p> : null}
        {live?.mediaCapacity === "unavailable" ? <p className="settings-help homekit-error">Media capacity is full. Close another live viewer and retry.</p> : null}
        {live?.runtime?.media?.failures ? <p className="settings-help homekit-error">The source encoder has reported failures. Check this camera’s source and audio availability.</p> : null}
        <div className="homekit-actions"><button type="button" disabled={!available || !!busy || live?.paired} onClick={() => cameraAction(camera, "pairing")}><KeyRound size={15} />{busy === camera.id ? "Working…" : "Pair with Apple Home"}</button><button type="button" disabled={!available || !!busy} onClick={() => cameraAction(camera, "reset-pairing")}><RotateCcw size={15} />Reset pairing</button></div>
        <details className="telemetry-technical"><summary>Connection details</summary><label>HomeKit port<input type="number" min={0} max={65535} value={options.port} onChange={(event) => updateCamera(index, "port", Number(event.target.value))} /></label><p className="settings-help">Zero uses this camera’s stable automatic port{live?.port ? ` (${live.port})` : ""}. Each camera needs a unique port. Talkback is unavailable.</p></details>
      </> : null}
      {pairing?.id === camera.id ? <div className="homekit-pairing" role="region" aria-label={`Pair ${camera.name}`}>
        <img src={`data:image/svg+xml;charset=utf-8,${encodeURIComponent(pairing.qrSvg)}`} alt={`HomeKit pairing QR for ${camera.name}`} width="192" height="192" />
        <div><h4>Add to Apple Home</h4><p>Open Home → Add Accessory and scan this QR, or choose More options and enter the setup code.</p><strong className="homekit-code">{pairing.setupCode}</strong><p className="settings-help">Keep this code private. It hides after two minutes.</p><button type="button" onClick={() => setPairing(null)}>Hide pairing code</button></div>
      </div> : null}
    </section>;
  }

  return <div className="homekit-settings subsection-settings-content">
    <div className="settings-heading"><div><h2><Home size={21} /> HomeKit</h2><p className="settings-help">Share selected SurvNG cameras with Apple Home. Manage live viewing and secure recording permissions in the Home app.</p></div><span className="homekit-badge">Experimental HKSV3</span></div>
    <fieldset className="sub-panel">
      <legend>HomeKit service</legend>
      <label className="check-field"><input type="checkbox" checked={settings.enabled} onChange={(event) => { setPairing(null); update("enabled", event.target.checked); }} />Enable HomeKit</label>
      <p className="settings-help">Save changes to apply them. Cameras stay paired when you disable HomeKit. HomeKit prefers verified Intel QSV HEVC encoding within SurvNG’s media limits; available capacity determines concurrent streams.</p>
      <div className="homekit-fields">
        <label>Server LAN IPv4 address<input value={settings.bind} placeholder="192.168.1.10" maxLength={64} required={settings.enabled} onChange={(event) => update("bind", event.target.value)} autoComplete="off" /><small>The address Apple devices use to reach this server.</small></label>
        <label>Node.js 24 executable<input value={settings.node_path} placeholder="/usr/local/bin/node" maxLength={4096} onChange={(event) => update("node_path", event.target.value)} autoComplete="off" /><small>Use an absolute path when the server’s default Node version differs.</small></label>
      </div>
    </fieldset>
    <section className="homekit-nodes" aria-label="HomeKit encoders">
      {status?.nodeError ? <p role="alert" className="settings-help homekit-error">HomeKit encoder state is unavailable. Check its private directory permissions and saved state.</p> : null}
      <div className="telemetry-section-head"><div><h3>Encoders</h3><p>Run the installer once as root in a new container. It checks in here. Apple Home still reaches this server.</p></div><button type="button" disabled={!!busy} onClick={addEncoder}>{busy === "encoder" ? "Working…" : "Add encoder"}</button></div>
      {installer ? <div className="homekit-installer" role="region" aria-label="Encoder installer"><p className="settings-help">This command is shown once and enrolls one encoder.</p><code>{installer}</code><button type="button" onClick={() => setInstaller("")}>Hide installer</button></div> : null}
      {status?.nodes?.length ? status.nodes.map((node) => <HomeKitEncoder key={node.id} node={node} />) : <p className="settings-help">No encoder nodes have checked in. Cameras continue to encode on this server.</p>}
    </section>
    <p className="settings-help">Configure HomeKit for each camera in <a href={appUrl("/admin?section=cameras")}>Camera settings</a>, alongside live view framing.</p>
    {error ? <p role="alert" className="settings-help homekit-error">{error} Status retries automatically.</p> : null}
    {!status && !error ? <p role="status">Loading HomeKit status…</p> : null}
  </div>;
}

export function CameraHomeKitSettings({ camera, enabled, onChange }) {
  return <HomeKitSettings key={camera.id} cameraId={camera.id}
    config={{ homekit: { enabled }, cameras: [camera] }}
    updateConfig={(_path, value) => onChange(["homekit"], value)} />;
}
