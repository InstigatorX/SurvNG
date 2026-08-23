import React from "react";

/** Workspace-level command bar for Admin destinations (scope → meta → actions). */
export function AdminCommandBar({ scope = null, meta = null, actions = null, className = "" }) {
  if (!scope && !meta && !actions) return null;
  return (
    <header className={`admin-command-bar${className ? ` ${className}` : ""}`} aria-label="Admin workspace controls">
      {scope ? <div className="admin-command-scope">{scope}</div> : null}
      {meta ? <div className="admin-command-meta">{meta}</div> : null}
      {actions ? <div className="admin-command-actions">{actions}</div> : null}
    </header>
  );
}

export function AdminCommandLabel({ children, icon: Icon = null }) {
  return (
    <span className="admin-command-label">
      {Icon ? <Icon size={15} aria-hidden="true" /> : null}
      <strong>{children}</strong>
    </span>
  );
}
