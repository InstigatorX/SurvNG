import React, { useEffect, useState } from "react";
import { Eye, EyeOff, Lock, ShieldCheck } from "lucide-react";

import { appUrl, fetch } from "../shared/api.js";

export function LoginScreen({ session, onSignedIn }) {
  const bootstrap = Boolean(session?.bootstrap_required && !session?.user);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    document.documentElement.dataset.theme = "dark";
    document.title = bootstrap ? "SurvNG · Create administrator" : "SurvNG · Sign in";
  }, [bootstrap]);

  async function submit(event) {
    event.preventDefault();
    if (busy) return;
    setBusy(true);
    setError("");
    try {
      const path = bootstrap ? "/api/auth/bootstrap" : "/api/auth/login";
      const response = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: username.trim(),
          password,
          display_name: displayName.trim(),
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "Could not sign in");
      onSignedIn(payload);
    } catch (caught) {
      setError(caught.message || "Could not sign in");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="login-stage">
      <div className="login-grid" aria-hidden="true" />
      <div className="login-scan" aria-hidden="true" />
      <div className="login-vignette" aria-hidden="true" />
      <main className="login-card">
        <a className="login-brand" href={appUrl("/")}>
          <span className="login-mark"><img src={appUrl("/static/favicon.svg")} alt="" /></span>
          <span>
            <strong>SurvNG</strong>
            <small>Site surveillance</small>
          </span>
        </a>
        <p className="login-kicker">{bootstrap ? "First operator" : "Secure access"}</p>
        <h1>{bootstrap ? "Create the administrator" : "Sign in"}</h1>
        <p className="login-copy">
          {bootstrap
            ? "This SurvNG host has no users yet. Create an administrator to lock the console to people you trust."
            : "Enter your SurvNG credentials to watch live cameras, review incidents, and manage the site."}
        </p>
        <form className="login-form" onSubmit={submit}>
          {bootstrap ? (
            <label>
              Display name
              <input
                value={displayName}
                onChange={(event) => setDisplayName(event.target.value)}
                autoComplete="name"
                placeholder="Alex"
              />
            </label>
          ) : null}
          <label>
            Username
            <input
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              autoComplete="username"
              autoCapitalize="none"
              spellCheck="false"
              required
              minLength={3}
              placeholder="operator"
            />
          </label>
          <label className="login-password">
            Password
            <span>
              <input
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                autoComplete={bootstrap ? "new-password" : "current-password"}
                required
                minLength={8}
                placeholder="••••••••"
              />
              <button type="button" onClick={() => setShowPassword((current) => !current)} aria-label={showPassword ? "Hide password" : "Show password"}>
                {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </span>
          </label>
          {error ? <p className="login-error" role="alert">{error}</p> : null}
          <button className="login-submit" type="submit" disabled={busy || username.trim().length < 3 || password.length < 8}>
            <Lock size={16} />
            {busy ? "Signing in…" : bootstrap ? "Create administrator" : "Continue"}
          </button>
        </form>
        <p className="login-foot">
          <ShieldCheck size={14} />
          {bootstrap ? "Passwords are hashed on this server. SurvNG never stores them in readable form." : "Sessions stay on this device. Ask an administrator for an account."}
        </p>
      </main>
    </div>
  );
}
