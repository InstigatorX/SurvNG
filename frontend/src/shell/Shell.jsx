import React, { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  CircleHelp,
  Clock3,
  Cog,
  Download,
  Gauge,
  Search,
  LogOut,
  PanelLeftClose,
  PanelLeftOpen,
  Siren,
  Sun,
  Users,
  Rows3,
  Video,
  X,
} from "lucide-react";
import { DESKTOP_PRIMARY_WORKSPACES, MOBILE_PRIMARY_WORKSPACES, workspaceDefinition, workspaceHref } from "../workspaceNavigation.mjs";
import { appUrl, recordingsHref } from "../shared/api.js";
import { useStoredState, useModalFocus } from "../shared/hooks.js";
import { RecordingHealthBar } from "./RecordingHealthBar.jsx";

export const WORKSPACE_ICONS = Object.freeze({
  live: Video,
  incidents: Siren,
  timeline: Clock3,
  exports: Download,
  search: Search,
  people: Users,
  admin: Cog,
});

export function MobileMoreSheet({ links, page, session = null, onClose }) {
  const modalRef = useModalFocus(onClose);
  return createPortal((
    <div ref={modalRef} className="mobile-more-sheet" role="dialog" aria-modal="true" aria-labelledby="mobile-more-title">
      <button type="button" className="mobile-more-backdrop" onClick={onClose} aria-label="Close more menu" />
      <div id="mobile-more-panel" className="mobile-more-panel" tabIndex={-1}>
        <header><h2 id="mobile-more-title">More</h2><button type="button" data-modal-initial onClick={onClose} aria-label="Close more menu"><X size={20} /></button></header>
        {links.map(([id, label, href, Icon]) => <a className={page === id ? "active" : ""} aria-current={page === id ? "page" : undefined} href={href} key={id}><Icon size={20} /><span>{label}</span></a>)}
        {session?.user?.role === "viewer" ? null : (
          <>
            <a href={appUrl("/admin?section=telemetry")}><Gauge size={20} /><span>System status</span></a>
            <a href={appUrl("/admin?section=general")}><Sun size={20} /><span>Appearance</span></a>
          </>
        )}
        <a href={appUrl("/help")}><CircleHelp size={20} /><span>Help</span></a>
      </div>
    </div>
  ), document.body);
}
export function Shell({ page, theme, recordingContext, session = null, onSignOut = null, children }) {
  const shellRef = useRef(null);
  const topbarRef = useRef(null);
  const workspaceHeadingRef = useRef(null);
  const mobileMoreButtonRef = useRef(null);
  const headerSearchRef = useRef(null);
  const [mobileMoreOpen, setMobileMoreOpen] = useState(false);
  const [headerSearchQuery, setHeaderSearchQuery] = useState("");
  const [railCollapsedValue, setRailCollapsedValue] = useStoredState("survng.workspaceRailCollapsed.v1", "false");
  const railCollapsed = railCollapsedValue === "true";
  const workspaceLink = (id) => {
    const definition = workspaceDefinition(id);
    return [
      id,
      definition.label,
      id === "timeline" ? recordingsHref(recordingContext) : appUrl(workspaceHref(id)),
      WORKSPACE_ICONS[id],
    ];
  };
  const workspaceLinks = [...DESKTOP_PRIMARY_WORKSPACES, ...(session?.user?.role === "viewer" ? [] : ["admin"])].map(workspaceLink);
  const mobileLinks = MOBILE_PRIMARY_WORKSPACES.filter((id) => id !== "more").map(workspaceLink);
  const mobilePrimaryIds = new Set(MOBILE_PRIMARY_WORKSPACES.filter((id) => id !== "more"));
  const moreLinks = workspaceLinks.filter(([id]) => !mobilePrimaryIds.has(id));

  useEffect(() => {
    const label = workspaceDefinition(page)?.label || "SurvNG";
    document.title = label === "Live" ? "SurvNG" : `SurvNG · ${label}`;
    window.requestAnimationFrame(() => workspaceHeadingRef.current?.focus({ preventScroll: true }));
  }, [page]);

  useEffect(() => {
    function focusHeaderSearch(event) {
      if (event.key !== "/" || event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey) return;
      const target = event.target;
      if (target instanceof HTMLElement && target.closest("input, textarea, select, [contenteditable=true]")) return;
      event.preventDefault();
      headerSearchRef.current?.focus();
    }
    window.addEventListener("keydown", focusHeaderSearch);
    return () => window.removeEventListener("keydown", focusHeaderSearch);
  }, []);

  function submitHeaderSearch(event) {
    event.preventDefault();
    const query = headerSearchQuery.trim();
    window.location.assign(appUrl(query ? `/search?q=${encodeURIComponent(query)}` : "/search"));
  }

  useLayoutEffect(() => {
    const shell = shellRef.current;
    const topbar = topbarRef.current;
    if (!shell || !topbar) return undefined;
    const updateTopbarHeight = () => {
      shell.style.setProperty(
        "--topbar-height",
        `${Math.ceil(topbar.getBoundingClientRect().height)}px`,
      );
    };
    updateTopbarHeight();
    const observer = typeof ResizeObserver === "function"
      ? new ResizeObserver(updateTopbarHeight)
      : null;
    observer?.observe(topbar);
    window.addEventListener("resize", updateTopbarHeight);
    return () => {
      observer?.disconnect();
      window.removeEventListener("resize", updateTopbarHeight);
    };
  }, []);
  return (
    <div ref={shellRef} className={`app-shell page-${page}${railCollapsed ? " workspace-rail-collapsed" : ""}`}>
      <aside className="workspace-sidebar" aria-label="SurvNG navigation">
        <a className="workspace-brand" href={appUrl("/")} aria-label="SurvNG Live">
          <span className="brand-mark"><img src={appUrl("/static/favicon.svg")} alt="" aria-hidden="true" /></span>
          <strong>SurvNG</strong>
        </a>
        <nav className="workspace-navigation" aria-label="Primary">
          {workspaceLinks.map(([id, label, href, Icon]) => <a className={page === id ? "active" : ""} aria-current={page === id ? "page" : undefined} aria-label={label} title={label} href={href} key={id}><Icon size={19} /><span>{label}</span></a>)}
        </nav>
        <a className="workspace-help-link" href={appUrl("/help")} aria-label="Help" title="Help"><CircleHelp size={19} /><span>Help</span></a>
        <button type="button" className="workspace-rail-toggle" onClick={() => setRailCollapsedValue(railCollapsed ? "false" : "true")} aria-label={railCollapsed ? "Expand navigation" : "Collapse navigation"} title={railCollapsed ? "Expand navigation" : "Collapse navigation"}>
          {railCollapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}<span>{railCollapsed ? "Expand" : "Collapse"}</span>
        </button>
        {session?.user ? (
          <div className="workspace-account">
            <span><strong>{session.user.display_name || session.user.username}</strong><small>{session.user.role}</small></span>
            {onSignOut ? <button type="button" onClick={onSignOut} aria-label="Sign out" title="Sign out"><LogOut size={16} /></button> : null}
          </div>
        ) : null}
      </aside>
      <header ref={topbarRef} className="topbar">
        <a className="brand-block mobile-brand-block" href={appUrl("/")} aria-label="SurvNG Live">
          <div className="brand-mark">
            <img src={appUrl("/static/favicon.svg")} alt="" aria-hidden="true" />
          </div>
          <div className="brand-title">
            <strong>SurvNG</strong>
          </div>
        </a>
        <form className="workspace-search-entry" onSubmit={submitHeaderSearch} role="search">
          <Search size={16} aria-hidden="true" />
          <input ref={headerSearchRef} value={headerSearchQuery} onChange={(event) => setHeaderSearchQuery(event.target.value)} placeholder="Search incidents..." aria-label="Search incidents semantically" />
          <kbd>/</kbd>
        </form>
        <div className="workspace-system-bar" aria-label="System status"><RecordingHealthBar /></div>
      </header>
      <div className="workspace-content"><h1 ref={workspaceHeadingRef} className="sr-only" tabIndex={-1}>SurvNG — {workspaceDefinition(page)?.label || "Workspace"}</h1>{children}</div>
      <nav className="mobile-workspace-nav" aria-label="Primary">
        {mobileLinks.map(([id, label, href, Icon]) => <a className={page === id ? "active" : ""} aria-current={page === id ? "page" : undefined} aria-label={label} href={href} key={id}><Icon size={21} /><span>{label}</span></a>)}
        <button ref={mobileMoreButtonRef} type="button" className={!mobilePrimaryIds.has(page) || mobileMoreOpen ? "active" : ""} onClick={() => setMobileMoreOpen((current) => !current)} aria-expanded={mobileMoreOpen} aria-controls="mobile-more-panel"><Rows3 size={21} /><span>More</span></button>
      </nav>
      {mobileMoreOpen ? <MobileMoreSheet links={moreLinks} page={page} session={session} onClose={() => setMobileMoreOpen(false)} /> : null}
    </div>
  );
}
