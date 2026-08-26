import React, { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  Activity,
  Camera,
  CircleHelp,
  Clock3,
  Cog,
  Cpu,
  Download,
  Gauge,
  HardDrive,
  Search,
  LogOut,
  Monitor,
  PanelLeftClose,
  PanelLeftOpen,
  ShieldCheck,
  Siren,
  Sun,
  Users,
  Rows3,
  Video,
  X,
} from "lucide-react";
import { useVisiblePolling } from "../visibilityPolling.mjs";
import { DESKTOP_PRIMARY_WORKSPACES, MOBILE_PRIMARY_WORKSPACES, systemHealthState, workspaceDefinition, workspaceHref } from "../workspaceNavigation.mjs";
import { appUrl, recordingsHref, fetch } from "../shared/api.js";
import { formatBytes, formatMilliseconds, formatRate } from "../shared/format.js";
import { useStoredState, useModalFocus } from "../shared/hooks.js";
import { useAppEvents } from "../shared/events.js";
import { useRuntimeState } from "../shared/runtimeState.jsx";

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
        <div className="workspace-system-bar" aria-label="System status"><LiveHeaderStats /></div>
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
export function LiveHeaderStats() {
  const runtimeState = useRuntimeState();
  const [stats, setStats] = useState({
    lifecycle: "",
    resources: null,
    storage: null,
    detector: null,
    cameras: null,
  });

  async function loadSystem() {
    try {
      const systemResponse = await fetch("/api/system/status");
      if (!systemResponse.ok) return;
      const system = await systemResponse.json();
      setStats((current) => ({
        ...current,
        lifecycle: system.lifecycle || "",
        resources: system.resources || null,
        storage: system.storage || null,
        detector: system.detector || null,
        cameras: system.cameras || null,
      }));
    } catch {
      // Keep the last known status; the next event or interval retries.
    }
  }

  useAppEvents(({ type, data }) => {
    if (type === "system_state") {
      setStats((current) => ({
        ...current,
        lifecycle: data.lifecycle || current.lifecycle,
        resources: data.resources || null,
        storage: data.storage || null,
        detector: data.detector || null,
        cameras: data.cameras || null,
      }));
    }
  });

  useVisiblePolling(loadSystem, 60_000, !runtimeState);

  useEffect(() => {
    if (runtimeState?.system) setStats((current) => ({ ...current, ...runtimeState.system }));
  }, [runtimeState?.system]);

  const detector = stats.detector || {};
  const runtime = detector.runtime || {};
  const inferenceStages = runtime.stages || {};
  const isolation = detector.isolation || {};
  const inferenceWorkers = detector.workers || {};
  const objectWorker = inferenceWorkers.object || isolation;
  const faceWorker = inferenceWorkers.face || {};
  const lastStages = inferenceStages.last_ms || {};
  const averageStages = inferenceStages.average_ms || {};
  const storageLabel = stats.storage ? `${formatBytes(stats.storage.free_bytes)} free` : "--";
  const memoryLabel = stats.resources ? formatBytes(stats.resources.application_memory_bytes) : "--";
  const cpuLabel = Number.isFinite(stats.resources?.cpu_load_percent) ? `${stats.resources.cpu_load_percent.toFixed(1)}%` : "--";
  const cameraLabel = stats.cameras ? `${stats.cameras.recording}/${stats.cameras.total} rec` : "--";
  const { severity: healthSeverity, label: healthLabel } = systemHealthState({
    lifecycle: stats.lifecycle,
    storage: stats.storage,
    detector: stats.detector,
    cameras: stats.cameras,
  });

  return (
    <div className="header-stats" aria-label="System summary">
      <span className={`header-stat header-health ${healthSeverity}`}><ShieldCheck size={15} /><small>System</small><strong>{healthLabel}</strong></span>
      <span className="header-stat"><HardDrive size={15} /><small>Storage</small><strong>{storageLabel}</strong></span>
      <span className="header-stat"><Monitor size={15} /><small>Memory</small><strong>{memoryLabel}</strong></span>
      <span className="header-stat"><Activity size={15} /><small>CPU</small><strong>{cpuLabel}</strong></span>
      <span className="header-stat infer-stat" tabIndex={0}>
        <Cpu size={15} /><small>Infer</small><strong>{formatMilliseconds(runtime.last_inference_ms)}</strong>
        <span className="infer-tooltip" role="tooltip">
          <span className="infer-tooltip-head"><strong>OpenVINO latency</strong><small>{detector.loaded_device || detector.configured_device || "device"} · {detector.performance_hint || "default"}</small></span>
          <span className="infer-tooltip-summary">
            <span><small>Average</small><strong>{formatMilliseconds(runtime.average_inference_ms)}</strong></span>
            <span><small>Detection rate</small><strong>{formatRate(runtime.detection_fps)} det/s</strong></span>
          </span>
          <span className="infer-tooltip-row labels"><b>Stage</b><b>Last</b><b>Average</b></span>
          {[["Queue", "queue"], ["Preprocess", "preprocess"], ["Accelerator", "inference"], ["Postprocess", "postprocess"], ["Total", "total"]].map(([label, key]) => (
            <span className="infer-tooltip-row" key={key}><span>{label}</span><strong>{formatMilliseconds(lastStages[key])}</strong><strong>{formatMilliseconds(averageStages[key])}</strong></span>
          ))}
          <span className="infer-tooltip-foot">{objectWorker.configured_workers || 1} detector process{(objectWorker.configured_workers || 1) === 1 ? "" : "es"} · mmap {detector.mmap_enabled ? "on" : "off"} · cache {detector.cache_enabled ? "on" : "off"} · warm-up {formatMilliseconds(detector.warmup_ms)}</span>
          <span className="infer-tooltip-foot">object {objectWorker.configured_workers > 1 ? `${objectWorker.alive_workers || 0}/${objectWorker.configured_workers} online` : objectWorker.worker_alive ? `#${objectWorker.worker_pid}` : "offline"} · {objectWorker.configured_device || detector.configured_device || "device"} · {objectWorker.pending_requests || 0} queued · restarts {objectWorker.restart_count ?? 0}{objectWorker.fallback_active ? " · CPU fallback" : ""}</span>
          <span className="infer-tooltip-foot">face {faceWorker.enabled ? (faceWorker.worker_alive ? `#${faceWorker.worker_pid}` : "offline") : "disabled"} · {faceWorker.configured_device || "AUTO"} · gen {faceWorker.generation ?? "--"} · restarts {faceWorker.restart_count ?? 0}{faceWorker.fallback_active ? " · CPU fallback" : ""}</span>
        </span>
      </span>
      <span className="header-stat"><Camera size={15} /><small>Cameras</small><strong>{cameraLabel}</strong></span>
    </div>
  );
}
