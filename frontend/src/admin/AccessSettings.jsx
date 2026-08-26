import React, { useEffect, useState } from "react";
import { KeyRound, Plus, RefreshCcw, ShieldCheck, Trash2, Upload } from "lucide-react";

import { fetch } from "../shared/api.js";

function apiDetail(payload, fallback) {
  const detail = payload?.detail;
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    const messages = detail.map((item) => item?.msg || item?.message).filter(Boolean);
    if (messages.length) return messages.join("; ");
  }
  if (detail && typeof detail === "object" && typeof detail.message === "string") return detail.message;
  return fallback;
}

export function AccessSettings({ config, commitImmediateConfig }) {
  const [users, setUsers] = useState(config.web_auth?.users || []);
  const [enabled, setEnabled] = useState(Boolean(config.web_auth?.enabled));
  const [tls, setTls] = useState(null);
  const [draft, setDraft] = useState({ username: "", display_name: "", password: "", role: "viewer" });
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [tlsHostname, setTlsHostname] = useState(config.tls?.hostname || "");
  const [tlsPort, setTlsPort] = useState(config.tls?.port || 0);

  async function load() {
    const [usersResponse, tlsResponse] = await Promise.all([
      fetch("/api/auth/users"),
      fetch("/api/tls"),
    ]);
    const userPayload = await usersResponse.json().catch(() => ({}));
    const tlsPayload = await tlsResponse.json().catch(() => ({}));
    if (usersResponse.ok) {
      setUsers(userPayload.users || []);
      setEnabled(Boolean(userPayload.enabled));
    }
    if (tlsResponse.ok) {
      setTls(tlsPayload);
      setTlsHostname(tlsPayload.hostname || "");
      setTlsPort(tlsPayload.port || 0);
    }
  }

  useEffect(() => {
    void load();
  }, []);

  function syncConfig(nextEnabled, nextUsers) {
    commitImmediateConfig(["web_auth", "enabled"], nextEnabled);
    commitImmediateConfig(["web_auth", "users"], nextUsers.map((user) => ({
      ...user,
      password_hash: user.password_hash || "__SURVNG_SECRET_SET__",
    })));
  }

  async function createUser(event) {
    event.preventDefault();
    setBusy("user");
    setError("");
    try {
      const response = await fetch("/api/auth/users", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(draft),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(apiDetail(payload, "Could not create user"));
      const next = [...users, payload.user];
      setUsers(next);
      syncConfig(enabled, next);
      setDraft({ username: "", display_name: "", password: "", role: "viewer" });
    } catch (caught) {
      setError(caught.message);
    } finally {
      setBusy("");
    }
  }

  async function setRole(userId, role) {
    setBusy(userId);
    setError("");
    try {
      const response = await fetch(`/api/auth/users/${encodeURIComponent(userId)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(apiDetail(payload, "Could not update user"));
      const next = users.map((user) => (user.id === userId ? payload.user : user));
      setUsers(next);
      syncConfig(enabled, next);
    } catch (caught) {
      setError(caught.message);
    } finally {
      setBusy("");
    }
  }

  async function removeUser(user) {
    const lastAdmin = user.role === "admin" && users.filter((item) => item.role === "admin").length === 1;
    const confirmed = window.confirm(
      lastAdmin && enabled
        ? "This is the last administrator. Deleting it will turn off sign-in. Continue?"
        : lastAdmin
          ? "Delete the last SurvNG administrator?"
          : "Delete this SurvNG user?",
    );
    if (!confirmed) return;
    setBusy(user.id);
    setError("");
    try {
      const response = await fetch(`/api/auth/users/${encodeURIComponent(user.id)}`, { method: "DELETE" });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(apiDetail(payload, "Could not delete user"));
      setUsers(payload.users || []);
      setEnabled(Boolean(payload.enabled));
      syncConfig(Boolean(payload.enabled), payload.users || []);
    } catch (caught) {
      setError(caught.message);
    } finally {
      setBusy("");
    }
  }

  async function toggleSignIn(nextEnabled) {
    setBusy("signin");
    setError("");
    try {
      const response = await fetch("/api/auth/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: nextEnabled }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(apiDetail(payload, "Could not update sign-in"));
      setEnabled(Boolean(payload.enabled));
      setUsers(payload.users || users);
      syncConfig(Boolean(payload.enabled), payload.users || users);
    } catch (caught) {
      setError(caught.message);
    } finally {
      setBusy("");
    }
  }

  async function saveTls(enabledValue) {
    setBusy("tls");
    setError("");
    try {
      const response = await fetch("/api/tls", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          enabled: enabledValue,
          hostname: tlsHostname,
          port: Number(tlsPort) || 0,
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(apiDetail(payload, "Could not update HTTPS"));
      setTls(payload);
      commitImmediateConfig(["tls"], {
        enabled: payload.enabled,
        hostname: payload.hostname || "",
        port: payload.port || 0,
      });
    } catch (caught) {
      setError(caught.message);
    } finally {
      setBusy("");
    }
  }

  async function generateCert() {
    setBusy("cert");
    setError("");
    try {
      const response = await fetch("/api/tls/self-signed", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ hostname: tlsHostname }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(apiDetail(payload, "Could not generate certificate"));
      setTls(payload);
    } catch (caught) {
      setError(caught.message);
    } finally {
      setBusy("");
    }
  }

  async function uploadCert(event) {
    const form = event.currentTarget;
    event.preventDefault();
    const files = new FormData(form);
    const certFile = form.elements.namedItem("certificate");
    const keyFile = form.elements.namedItem("private_key");
    const hasFiles = certFile instanceof HTMLInputElement && keyFile instanceof HTMLInputElement
      && certFile.files?.[0] && keyFile.files?.[0];
    const certPem = String(files.get("certificate_pem") || "").trim();
    const keyPem = String(files.get("private_key_pem") || "").trim();
    setBusy("upload");
    setError("");
    try {
      let response;
      if (hasFiles) {
        const payload = new FormData();
        payload.set("certificate", certFile.files[0]);
        payload.set("private_key", keyFile.files[0]);
        response = await fetch("/api/tls/upload", { method: "POST", body: payload });
      } else if (certPem && keyPem) {
        response = await fetch("/api/tls/certificate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ certificate_pem: certPem, private_key_pem: keyPem }),
        });
      } else {
        throw new Error("Choose a certificate and private key file, or paste both PEM blocks.");
      }
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(apiDetail(payload, "Could not store certificate"));
      setTls(payload);
      form.reset();
    } catch (caught) {
      setError(caught.message);
    } finally {
      setBusy("");
    }
  }

  async function restartForTls() {
    if (!window.confirm("Restart SurvNG to apply HTTPS? Live view and recording playback will pause briefly.")) return;
    setBusy("restart");
    setError("");
    try {
      const response = await fetch("/api/tls/apply", { method: "POST" });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(apiDetail(payload, "Could not restart SurvNG"));
    } catch (caught) {
      setError(caught.message);
      setBusy("");
    }
  }

  return (
    <div className="sub-panel access-settings">
      <h3>Users &amp; HTTPS</h3>
      <section className="api-access-settings">
        <div className="detection-settings-subhead">
          <div>
            <strong>Sign-in</strong>
            <small>Local SurvNG accounts. Administrators can change settings; viewers can watch cameras and review incidents.</small>
          </div>
          <div className="admin-action-status">
            <span className="admin-action-kind">Applies immediately</span>
            <span className={`retention-state ${enabled ? "running" : "idle"}`}>{enabled ? "Required" : "Open LAN"}</span>
          </div>
        </div>
        <label className="check-field">
          <input
            type="checkbox"
            checked={enabled}
            disabled={Boolean(busy) || ( !enabled && !users.some((user) => user.role === "admin"))}
            onChange={(event) => void toggleSignIn(event.target.checked)}
          />
          Require sign-in for the browser console
        </label>
        <p className="settings-help">Turn sign-in off to demote or delete the last administrator. Deleting that account also turns sign-in off.</p>
        <div className="api-token-list">
          {users.map((user) => (
            <article key={user.id}>
              <div>
                <strong>{user.display_name || user.username}</strong>
                <code>{user.username}</code>
                <small>{user.role}</small>
              </div>
              <select value={user.role} disabled={Boolean(busy)} onChange={(event) => void setRole(user.id, event.target.value)}>
                <option value="admin">Admin</option>
                <option value="viewer">Viewer</option>
              </select>
              <button type="button" className="danger" onClick={() => void removeUser(user)} disabled={Boolean(busy)}>
                <Trash2 size={14} /> Delete
              </button>
            </article>
          ))}
          {!users.length ? <p className="settings-help">No users yet. Create an administrator before requiring sign-in.</p> : null}
        </div>
        <form className="api-token-create" onSubmit={createUser}>
          <label>Username<input value={draft.username} onChange={(event) => setDraft((current) => ({ ...current, username: event.target.value }))} placeholder="alex" required minLength={3} /></label>
          <label>Display name<input value={draft.display_name} onChange={(event) => setDraft((current) => ({ ...current, display_name: event.target.value }))} placeholder="Alex" /></label>
          <label>Password<input type="password" value={draft.password} onChange={(event) => setDraft((current) => ({ ...current, password: event.target.value }))} required minLength={8} autoComplete="new-password" /></label>
          <label>Role<select value={draft.role} onChange={(event) => setDraft((current) => ({ ...current, role: event.target.value }))}><option value="admin">Admin</option><option value="viewer">Viewer</option></select></label>
          <button type="submit" className="primary" disabled={busy === "user"}>{busy === "user" ? <RefreshCcw className="spin" size={15} /> : <Plus size={15} />} Add user</button>
        </form>
      </section>

      <section className="api-access-settings">
        <div className="detection-settings-subhead">
          <div>
            <strong>HTTPS</strong>
            <small>Serve the console with TLS. Generate a self-signed certificate for LAN use, or upload a certificate and private key from your CA.</small>
          </div>
          <div className="admin-action-status">
            <span className="admin-action-kind">Restart to apply</span>
            <span className={`retention-state ${tls?.enabled ? "running" : "idle"}`}>{tls?.enabled ? "Enabled" : "HTTP"}</span>
          </div>
        </div>
        <div className="admin-field-grid">
          <label className="check-field">
            <input type="checkbox" checked={Boolean(tls?.enabled)} disabled={Boolean(busy)} onChange={(event) => void saveTls(event.target.checked)} />
            Serve SurvNG over HTTPS
          </label>
          <label>Hostname<input value={tlsHostname} onChange={(event) => setTlsHostname(event.target.value)} placeholder="survng.local" /></label>
          <label>TLS port<input type="number" min="0" max="65535" value={tlsPort} onChange={(event) => setTlsPort(Number(event.target.value))} /><small>0 keeps the current listen port.</small></label>
        </div>
        {tls?.certificate_present ? (
          <div className="probe-result ok">
            <strong>{tls.self_signed ? "Self-signed certificate" : "Installed certificate"}</strong>
            <span>{tls.subject || "Certificate on disk"}</span>
            {tls.not_after ? <span>Expires {tls.not_after}</span> : null}
            {tls.fingerprint_sha256 ? <span>SHA-256 {tls.fingerprint_sha256}</span> : null}
            {tls.error ? <span>{tls.error}</span> : null}
          </div>
        ) : <p className="settings-help">No certificate stored yet. Generate a self-signed certificate or upload one from your CA.</p>}
        <div className="preference-action-buttons">
          <button type="button" onClick={() => void generateCert()} disabled={Boolean(busy)}><KeyRound size={15} /> Generate self-signed</button>
          <button type="button" className="primary" onClick={() => void restartForTls()} disabled={busy === "restart"}><ShieldCheck size={15} /> Restart with HTTPS</button>
        </div>
        <form className="api-token-create tls-upload-form" onSubmit={uploadCert}>
          <div className="detection-settings-subhead">
            <div>
              <strong>Upload certificate</strong>
              <small>Choose PEM files, or paste the certificate and private key. A full chain is accepted.</small>
            </div>
          </div>
          <label>Certificate file<input name="certificate" type="file" accept=".pem,.crt,.cer,.cert,.txt" /></label>
          <label>Private key file<input name="private_key" type="file" accept=".pem,.key,.txt" /></label>
          <label>Certificate PEM<textarea name="certificate_pem" rows={6} spellCheck="false" placeholder="-----BEGIN CERTIFICATE-----" /></label>
          <label>Private key PEM<textarea name="private_key_pem" rows={6} spellCheck="false" placeholder="-----BEGIN PRIVATE KEY-----" /></label>
          <button type="submit" className="primary" disabled={busy === "upload"}><Upload size={15} /> {busy === "upload" ? "Storing…" : "Store uploaded certificate"}</button>
        </form>
      </section>
      {error ? <div className="error-banner">{error}</div> : null}
    </div>
  );
}
