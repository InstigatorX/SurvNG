import React, { useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  CircleHelp,
  Clock3,
  Cog,
  Download,
  Map,
  Gauge,
  Search,
  LogOut,
  ChevronDown,
  UserRound,
  Siren,
  Sun,
  Users,
  Rows3,
  Video,
  X,
} from "lucide-react";
import { compactDesktopNavigation, DESKTOP_PRIMARY_WORKSPACES, MOBILE_PRIMARY_WORKSPACES, workspaceDefinition, workspaceHref } from "../workspaceNavigation.mjs";
import { appUrl, recordingsHref } from "../shared/api.js";
import { useModalFocus } from "../shared/hooks.js";
import { RecordingHealthBar } from "./RecordingHealthBar.jsx";

export const WORKSPACE_ICONS = Object.freeze({
  live: Video,
  property: Map,
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
const TOPBAR_FIT_LEVELS = 4;

export function accountInitialsFor(name = "") {
  const words = String(name || "").trim().split(/[\s._-]+/).filter(Boolean);
  if (!words.length) return "";
  const first = Array.from(words[0]);
  const letters = words.length > 1 ? first[0] + Array.from(words[1])[0] : first.slice(0, 2).join("");
  return letters.toUpperCase();
}

/** Small disclosure menu for the top bar (More workspaces, account). Closes on outside click, Escape and selection. */
function TopbarMenu({ className, label, buttonLabel = undefined, menuLabel, children }) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef(null);
  const buttonRef = useRef(null);
  const menuId = useId();
  useEffect(() => {
    if (!open) return undefined;
    function closeForOutside(event) {
      if (!rootRef.current?.contains(event.target)) setOpen(false);
    }
    function closeForEscape(event) {
      if (event.key !== "Escape") return;
      event.preventDefault();
      setOpen(false);
      buttonRef.current?.focus();
    }
    document.addEventListener("pointerdown", closeForOutside);
    document.addEventListener("keydown", closeForEscape);
    return () => {
      document.removeEventListener("pointerdown", closeForOutside);
      document.removeEventListener("keydown", closeForEscape);
    };
  }, [open]);
  useEffect(() => {
    if (open) rootRef.current?.querySelector("[role=menuitem]")?.focus();
  }, [open]);
  function moveFocus(event) {
    if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) return;
    const items = [...(rootRef.current?.querySelectorAll("[role=menuitem]") || [])];
    if (!items.length) return;
    event.preventDefault();
    const current = items.indexOf(document.activeElement);
    const next = event.key === "Home" ? 0
      : event.key === "End" ? items.length - 1
        : (current + (event.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
    items[next].focus();
  }
  return (
    <div ref={rootRef} className={`topbar-menu ${className}`} onBlur={(event) => { if (open && event.relatedTarget && !rootRef.current?.contains(event.relatedTarget)) setOpen(false); }}>
      <button ref={buttonRef} type="button" className="topbar-menu-button" aria-haspopup="menu" aria-expanded={open} aria-controls={menuId} aria-label={buttonLabel} onClick={() => setOpen((value) => !value)}>
        {label ? <span>{label}</span> : <UserRound size={17} aria-hidden="true" />}
        {className.includes("topbar-more") ? <ChevronDown size={14} aria-hidden="true" /> : null}
      </button>
      {open ? <div id={menuId} className="topbar-menu-panel" role="menu" aria-label={menuLabel} onKeyDown={moveFocus} onClick={(event) => { if (event.target.closest("[role=menuitem]")) setOpen(false); }}>{children}</div> : null}
    </div>
  );
}

export function Shell({ page, theme, textSize = "default", recordingContext, session = null, onSignOut = null, children }) {
  const shellRef = useRef(null);
  const topbarRef = useRef(null);
  const workspaceHeadingRef = useRef(null);
  const mobileMoreButtonRef = useRef(null);
  const headerSearchRef = useRef(null);
  const [mobileMoreOpen, setMobileMoreOpen] = useState(false);
  const [headerSearchQuery, setHeaderSearchQuery] = useState("");
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
  const compactNavigation = compactDesktopNavigation(workspaceLinks.map(([id]) => id), page);
  const compactInlineIds = new Set(compactNavigation.inline);
  const accountName = session?.user?.display_name || session?.user?.username || "";
  const accountInitials = accountInitialsFor(accountName);

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

  // The top bar's breakpoints assume the default text size. When its contents
  // still do not fit (larger text, long names), step through more compact
  // levels until they do: 1 hides the brand name, 2 moves workspaces under
  // More, 3 shrinks search to its icon and status pills to icons, 4 lets the
  // remaining workspace links scroll sideways inside the bar.
  useLayoutEffect(() => {
    const topbar = topbarRef.current;
    if (!topbar) return undefined;
    const fit = () => {
      let level = 0;
      for (; level <= TOPBAR_FIT_LEVELS; level += 1) {
        topbar.dataset.fit = String(level);
        if (topbar.scrollWidth <= topbar.clientWidth + 1) break;
      }
      topbar.dataset.fit = String(Math.min(level, TOPBAR_FIT_LEVELS));
    };
    fit();
    const observer = typeof ResizeObserver === "function" ? new ResizeObserver(fit) : null;
    observer?.observe(topbar);
    const status = topbar.querySelector(".workspace-system-bar");
    if (status) observer?.observe(status);
    window.addEventListener("resize", fit);
    return () => {
      observer?.disconnect();
      window.removeEventListener("resize", fit);
    };
  }, [textSize, page, session?.user?.role]);

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
    <div ref={shellRef} className={`app-shell page-${page}`}>
      <a className="skip-link" href="#main-content">Skip to main content</a>
      <header ref={topbarRef} className="topbar">
        <a className="brand-block mobile-brand-block topbar-brand" href={appUrl("/")} aria-label="SurvNG Live">
          <div className="brand-mark">
            <img src={appUrl("/static/favicon.svg")} alt="" aria-hidden="true" />
          </div>
          <div className="brand-title">
            <strong>SurvNG</strong>
          </div>
        </a>
        <nav className="topbar-navigation" aria-label="Primary">
          {workspaceLinks.map(([id, label, href]) => <a className={`${page === id ? "active" : ""}${compactInlineIds.has(id) ? "" : " topbar-overflow-link"}`.trim() || undefined} aria-current={page === id ? "page" : undefined} href={href} key={id}>{label}</a>)}
        </nav>
        {compactNavigation.overflow.length ? (
          <TopbarMenu className="topbar-more" label="More" menuLabel="More workspaces">
            {compactNavigation.overflow.map((id) => {
              const [, label, href] = workspaceLink(id);
              return <a role="menuitem" href={href} key={id}>{label}</a>;
            })}
          </TopbarMenu>
        ) : null}
        <form className="workspace-search-entry" onSubmit={submitHeaderSearch} role="search">
          <Search size={16} aria-hidden="true" />
          <input ref={headerSearchRef} value={headerSearchQuery} onChange={(event) => setHeaderSearchQuery(event.target.value)} placeholder="Search incidents..." aria-label="Search incidents semantically" />
          <kbd>/</kbd>
        </form>
        <div className="workspace-system-bar" aria-label="System status"><RecordingHealthBar /></div>
        <TopbarMenu className="topbar-account" label={accountInitials} buttonLabel={session?.user ? `Account: ${accountName}` : "Account and help"} menuLabel="Account">
          {session?.user ? <div className="topbar-account-identity" role="none"><strong>{accountName}</strong><small>{session.user.role}</small></div> : null}
          <a role="menuitem" href={appUrl("/help")}><CircleHelp size={16} aria-hidden="true" /><span>Help</span></a>
          {session?.user && onSignOut ? <button type="button" role="menuitem" onClick={onSignOut}><LogOut size={16} aria-hidden="true" /><span>Sign out</span></button> : null}
        </TopbarMenu>
      </header>
      <div id="main-content" className="workspace-content" tabIndex={-1}>
        <h1 ref={workspaceHeadingRef} className="sr-only" tabIndex={-1}>SurvNG — {workspaceDefinition(page)?.label || "Workspace"}</h1>
        {children}
      </div>
      <nav className="mobile-workspace-nav" aria-label="Primary">
        {mobileLinks.map(([id, label, href, Icon]) => <a className={page === id ? "active" : ""} aria-current={page === id ? "page" : undefined} aria-label={label} href={href} key={id}><Icon size={21} /><span>{label}</span></a>)}
        <button ref={mobileMoreButtonRef} type="button" className={!mobilePrimaryIds.has(page) || mobileMoreOpen ? "active" : ""} onClick={() => setMobileMoreOpen((current) => !current)} aria-expanded={mobileMoreOpen} aria-controls="mobile-more-panel"><Rows3 size={21} /><span>More</span></button>
      </nav>
      {mobileMoreOpen ? <MobileMoreSheet links={moreLinks} page={page} session={session} onClose={() => setMobileMoreOpen(false)} /> : null}
    </div>
  );
}
