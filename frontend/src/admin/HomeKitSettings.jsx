import React, { useEffect, useState } from "react";
import { Camera, Home, KeyRound, RotateCcw } from "lucide-react";
import { fetch, appUrl } from "../shared/api.js";
import "./homekit.css";

const ERROR_LABELS = {
  state_in_use: "Waiting for the previous HomeKit camera worker to finish shutting down.",
  software_encoding_not_allowed: "This server needs an application restart to enable hardware-only HomeKit startup.",
  encoding_unavailable: "No permitted HEVC encoder passed its startup test. Check QSV device access or allow software fallback.",
  worker_not_built: "Build the HomeKit worker before enabling cameras.",
  node_24_required: "HomeKit requires Node.js 24. Set its executable path below.",
  encoder_unavailable: "FFmpeg needs HEVC encoding plus libopus, AAC, and MJPEG support.",
  bind_address_unavailable: "The LAN address is not available on this server. Check the address below.",
  runtime_or_state_unavailable: "Check the Node/FFmpeg paths and private HomeKit state permissions.",
  worker_start_failed: "The camera worker could not start. Check the LAN address and port.",
  worker_stopped: "The camera worker stopped. Retrying; check its LAN address, port, and runtime requirements.",
  control_unavailable: "Waiting for the HomeKit camera to become ready…",
  camera_off: "Turn this camera on in SurvNG to resume HomeKit.",
};

export function HomeKitSettings({ config, updateConfig }) {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState("");
  const [pairing, setPairing] = useState(null);
  const [busy, setBusy] = useState("");
  const settings = { enabled: false, bind: "", node_path: "node", ...config.homekit };
  const cameras = config.cameras || [];
  const update = (key, value) => updateConfig(["homekit"], { ...settings, [key]: value });
  const updateCamera = (index, key, value) => {
    setPairing(null);
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
        setPairing((current) => current && next.cameras.some((camera) => camera.id === current.id && camera.state === "running" && !camera.paired) ? current : null);
      } catch (failure) {
        if (!controller.signal.aborted) { setError(failure.message || "HomeKit status unavailable."); setStatus(null); setPairing(null); }
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(refresh, 3000);
      }
    };
    void refresh();
    return () => { controller.abort(); clearTimeout(timer); };
  }, []);
  useEffect(() => {
    if (!pairing) return undefined;
    const timer = setTimeout(() => setPairing(null), 120000);
    return () => clearTimeout(timer);
  }, [pairing]);
  useEffect(() => {
    const hide = () => { if (document.hidden) setPairing(null); };
    document.addEventListener("visibilitychange", hide);
    return () => document.removeEventListener("visibilitychange", hide);
  }, []);

  async function cameraAction(camera, action) {
    if (action === "reset-pairing" && !window.confirm(`Reset HomeKit pairing for ${camera.name}? Remove its old tile from Apple Home, then pair it again. Other cameras keep their pairing.`)) return;
    setBusy(camera.id); setError(""); setPairing(null);
    try {
      const response = await fetch(`/api/homekit/cameras/${encodeURIComponent(camera.id)}/${action}`, {
        method: "POST", headers: { "Content-Type": "application/json" }, cache: "no-store",
        body: JSON.stringify(action === "reset-pairing" ? { confirm_camera_id: camera.id } : {}),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "HomeKit request failed.");
      if (action === "pairing") setPairing({ id: camera.id, name: camera.name, ...payload });
    } catch (failure) { setError(failure.message || "HomeKit request failed."); }
    finally { setBusy(""); }
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
    <section className="homekit-cameras" aria-label="HomeKit cameras">
      <div className="telemetry-section-head"><div><h3>Cameras</h3><p>Enable a camera, save, then use Pair with Apple Home to reveal its code and QR.</p></div><a className="button" href={appUrl("/admin?section=cameras")}><Camera size={15} />Camera setup</a></div>
      {error ? <p role="alert" className="settings-help homekit-error">{error} Status retries automatically.</p> : null}
      {!status && !error ? <p role="status">Loading HomeKit status…</p> : null}
      {!cameras.length ? <div className="empty-state">Add a camera in Camera setup, then enable it here.</div> : cameras.map((camera, index) => {
        const options = { enabled: false, allow_software_encoding: false, source: "live", audio: false, object_motion: false, port: 0, ...camera.homekit };
        const live = status?.cameras?.find((item) => item.id === camera.id);
        const available = settings.enabled && options.enabled && live?.state === "running";
        return <article className="sub-panel homekit-camera" key={camera.id}>
          <div className="homekit-camera-head"><h4><Camera size={17} />{camera.name}</h4><span className="homekit-badge">{live?.paired ? "Paired" : live?.state === "running" ? "Ready to pair" : live?.state || "Not enabled"}</span></div>
          <label className="check-field"><input type="checkbox" checked={options.enabled} onChange={(event) => updateCamera(index, "enabled", event.target.checked)} />Enable HomeKit for this camera</label>
          {options.enabled ? <>
            <label className="check-field"><input type="checkbox" checked={options.allow_software_encoding} onChange={(event) => updateCamera(index, "allow_software_encoding", event.target.checked)} />Allow software HEVC fallback</label>
            <p className="settings-help">QSV is preferred automatically. This permits software encoding when QSV is unavailable. Viewers and recording share an encoder for the same video profile.</p>
            {live?.runtime?.encoder ? <p className="settings-help" role="status">Encoder: <strong>{live.runtime.encoder.backend === "qsv" ? "Intel QSV HEVC" : "Software HEVC"}</strong>{live.runtime.encoder.sources?.[0]?.fps != null ? ` · ${Number(live.runtime.encoder.sources[0].fps).toFixed(1)} fps` : ""}{live.runtime.encoder.fallbackReason ? " · QSV unavailable; using permitted software fallback" : ""}</p> : null}
            <div className="homekit-fields">
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
        </article>;
      })}
    </section>
  </div>;
}
