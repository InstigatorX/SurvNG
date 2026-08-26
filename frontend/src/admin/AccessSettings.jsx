import React, { useEffect, useState } from "react";
import { KeyRound, Network, Plus, RefreshCcw, ShieldCheck, Trash2, Upload } from "lucide-react";

import { fetch } from "../shared/api.js";

function userLabel(user) {
  const display = String(user.display_name || "").trim();
  const username = String(user.username || "").trim();
  if (display && display.toLowerCase() !== username.toLowerCase()) return display;
  return username;
}

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

function formatTrustedProxies(value) {
  const items = Array.isArray(value) ? value.map((item) => String(item).trim()).filter(Boolean) : [];
  return items.length ? items.join("\n") : "127.0.0.1\n::1";
}

function parseTrustedProxies(text) {
  return [...new Set(
    String(text || "")
      .split(/[\n,]+/)
      .map((item) => item.trim())
      .filter(Boolean),
  )];
}

export function AccessSettings({ config, updateConfig, commitImmediateConfig, onOpenApiTokens = null }) {
  const [users, setUsers] = useState(config.web_auth?.users || []);
  const enabled = Boolean(config.web_auth?.enabled);
  const [passwordEdits, setPasswordEdits] = useState({});
  const [tls, setTls] = useState(null);
  const [draft, setDraft] = useState({ username: "", display_name: "", password: "", role: "viewer" });
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [tlsHostname, setTlsHostname] = useState(config.tls?.hostname || "");
  const [tlsPort, setTlsPort] = useState(config.tls?.port || 0);
  const [proxyText, setProxyText] = useState(() => formatTrustedProxies(config.proxy?.trusted_proxies));
  const [accessSection, setAccessSection] = useState("users");

  async function load() {
    const [usersResponse, tlsResponse] = await Promise.all([
      fetch("/api/auth/users"),
      fetch("/api/tls"),
    ]);
    const userPayload = await usersResponse.json().catch(() => ({}));
    const tlsPayload = await tlsResponse.json().catch(() => ({}));
    if (usersResponse.ok) {
      setUsers(userPayload.users || []);
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

  function syncUsers(nextUsers) {
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
      syncUsers(next);
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
      syncUsers(next);
    } catch (caught) {
      setError(caught.message);
    } finally {
      setBusy("");
    }
  }

  async function changePassword(user) {
    const password = String(passwordEdits[user.id] || "");
    if (password.length < 8) {
      setError("New password must be at least 8 characters.");
      return;
    }
    setBusy(`${user.id}-password`);
    setError("");
    try {
      const response = await fetch(`/api/auth/users/${encodeURIComponent(user.id)}/password`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(apiDetail(payload, "Could not update password"));
      setPasswordEdits((current) => ({ ...current, [user.id]: "" }));
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
      syncUsers(payload.users || []);
      if (Boolean(payload.enabled) !== enabled) {
        commitImmediateConfig(["web_auth", "enabled"], Boolean(payload.enabled));
      }
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
    <div className="sub-panel access-settings subsection-workspace">
      <nav className="admin-section-tabs camera-section-tabs detection-subsection-tabs security-subsection-tabs" aria-label="Security settings">
        {[['users', 'Users & sign-in', KeyRound], ['https', 'HTTPS certificates', ShieldCheck], ['proxies', 'Trusted proxies', Network]].map(([value, label, Icon]) => (
          <button type="button" key={value} className={accessSection === value ? "active" : ""} aria-pressed={accessSection === value} onClick={() => setAccessSection(value)}><Icon size={15} />{label}</button>
        ))}
        {onOpenApiTokens ? <button type="button" className="security-api-token-link" onClick={onOpenApiTokens}><KeyRound size={15} />API tokens <span aria-hidden="true">↗</span></button> : null}
      </nav>
      <div className="subsection-workspace-content">
      <section className="api-access-settings" hidden={accessSection !== "users"}>
        <div className="detection-settings-subhead">
          <div>
            <strong className="section-heading-with-icon"><span className="section-heading-icon"><KeyRound size={16} /></span>Sign-in</strong>
            <small>Local SurvNG accounts. Administrators can change settings; viewers can watch cameras and review incidents.</small>
          </div>
        </div>
        <label className="check-field">
          <input
            type="checkbox"
            checked={enabled}
            disabled={Boolean(busy)}
            onChange={(event) => updateConfig(["web_auth", "enabled"], event.target.checked)}
          />
          Require sign-in for the browser console
        </label>
        <p className="settings-help">Check this box, then Save settings. If no administrator exists yet, SurvNG will ask you to create one. You can also add users here first, then enable sign-in and save.</p>
        <label>Session length (days)
          <input
            type="number"
            min="1"
            max="365"
            step="1"
            value={config.web_auth?.session_days ?? 14}
            onChange={(event) => updateConfig(["web_auth", "session_days"], Number(event.target.value))}
          />
          <small>How long a signed-in browser stays signed in. Save settings to apply. New sign-ins use this length; current sessions keep their existing expiry unless you change that user’s password.</small>
        </label>
        <div className="api-token-list">
          {users.map((user) => (
            <article className="access-user-row" key={user.id}>
              <div>
                <strong>{userLabel(user)}</strong>
                {userLabel(user).toLowerCase() === String(user.username || "").toLowerCase()
                  ? null
                  : <code>{user.username}</code>}
              </div>
              <select
                className="access-role-select"
                value={user.role}
                aria-label={`Role for ${userLabel(user)}`}
                disabled={Boolean(busy)}
                onChange={(event) => void setRole(user.id, event.target.value)}
              >
                <option value="admin">Admin</option>
                <option value="viewer">Viewer</option>
              </select>
              <label className="access-password-field">
                <span className="visually-hidden">New password for {userLabel(user)}</span>
                <input
                  type="password"
                  value={passwordEdits[user.id] || ""}
                  placeholder="New password"
                  minLength={8}
                  autoComplete="new-password"
                  disabled={Boolean(busy)}
                  onChange={(event) => setPasswordEdits((current) => ({ ...current, [user.id]: event.target.value }))}
                />
              </label>
              <button
                type="button"
                onClick={() => void changePassword(user)}
                disabled={Boolean(busy) || String(passwordEdits[user.id] || "").length < 8}
              >
                <KeyRound size={14} /> Set password
              </button>
              <button type="button" className="danger" onClick={() => void removeUser(user)} disabled={Boolean(busy)}>
                <Trash2 size={14} /> Delete
              </button>
            </article>
          ))}
          {!users.length ? <p className="settings-help">No users yet. Add an administrator here, or enable sign-in and save to create one.</p> : null}
        </div>
        <form className="api-token-create access-user-create" onSubmit={createUser}>
          <label>Username<input value={draft.username} onChange={(event) => setDraft((current) => ({ ...current, username: event.target.value }))} placeholder="alex" required minLength={3} /></label>
          <label>Display name<input value={draft.display_name} onChange={(event) => setDraft((current) => ({ ...current, display_name: event.target.value }))} placeholder="Alex" /></label>
          <label>Password<input type="password" value={draft.password} onChange={(event) => setDraft((current) => ({ ...current, password: event.target.value }))} required minLength={8} autoComplete="new-password" /></label>
          <label className="access-role-field">Role<select className="access-role-select" value={draft.role} onChange={(event) => setDraft((current) => ({ ...current, role: event.target.value }))}><option value="admin">Admin</option><option value="viewer">Viewer</option></select></label>
          <button type="submit" className="primary" disabled={busy === "user"}>{busy === "user" ? <RefreshCcw className="spin" size={15} /> : <Plus size={15} />} Add user</button>
        </form>
      </section>

      <section className="api-access-settings" hidden={accessSection !== "proxies"}>
        <div className="detection-settings-subhead">
          <div><strong className="section-heading-with-icon"><span className="section-heading-icon"><Network size={16} /></span>Trusted reverse proxies</strong><small>Only listed proxy addresses may set HTTPS and client-IP headers.</small></div>
        </div>
        <label className="trusted-proxies-field">Proxy addresses
          <textarea className="trusted-proxies-input" rows={6} value={proxyText} onChange={(event) => { const text = event.target.value; setProxyText(text); updateConfig(["proxy", "trusted_proxies"], parseTrustedProxies(text)); }} spellCheck="false" wrap="off" autoComplete="off" placeholder={"127.0.0.1\n::1"} />
          <small>One IP or CIDR per line (commas are also fine). Use 127.0.0.1 when nginx is on this computer; Docker networks often need 172.16.0.0/12. Save settings to apply. See Help → Reverse proxy.</small>
        </label>
      </section>

      <section className="api-access-settings" hidden={accessSection !== "https"}>
        <div className="detection-settings-subhead">
          <div>
            <strong className="section-heading-with-icon"><span className="section-heading-icon"><ShieldCheck size={16} /></span>HTTPS</strong>
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
    </div>
  );
}
