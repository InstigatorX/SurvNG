import React, { lazy, Suspense, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { CircleAlert } from "lucide-react";

import "./typeScale.css";
import "./styles.css";
import "./shell/shell.css";
import "./timeline/timeline.css";
import "./timeline/rail.css";
import "./live/live.css";
import "./timeline/investigation.css";
import "./incidents/incidents.css";
import "./search/search.css";
import "./people/people.css";
import "./property/property.css";
import "./admin/admin.css";
import "./shell/responsive.css";
import "./admin/workspace.css";
import "./shell/mobile.css";

import "./auth/login.css";

import { appPathname, appUrl, fetch } from "./shared/api.js";
import { DEFAULT_TIME_ZONE, THEMES } from "./shared/constants.js";
import { useStoredState } from "./shared/hooks.js";
import { LoginScreen } from "./auth/LoginScreen.jsx";
import { Shell } from "./shell/Shell.jsx";
import { AssistantPanel } from "./assistant/AssistantPanel.jsx";
import { RuntimeStateProvider } from "./shared/runtimeState.jsx";
import { canonicalWorkspaceUrl, resolveWorkspace } from "./workspaceNavigation.mjs";
import { registerSurvngServiceWorker } from "./registerServiceWorker.mjs";
import { applyBrowserAppearance } from "./browserAppearance.mjs";
import { applyTextSize, TEXT_SIZE_STORAGE_KEY } from "./textSize.mjs";
import { applyInterfaceFont, INTERFACE_FONT_STORAGE_KEY } from "./interfaceFont.mjs";

function lazyExport(importer, exportName) {
  return lazy(() => importer().then((module) => ({ default: module[exportName] })));
}

// One importer per page, so the page for the current address can start
// loading before the app has rendered anything.
const pageImporters = {
  live: () => import("./live/LivePage.jsx"),
  incident: () => import("./incidents/IncidentDetailPage.jsx"),
  observations: () => import("./incidents/ObservationsPage.jsx"),
  incidents: () => import("./incidents/IncidentsPage.jsx"),
  timelinePages: () => import("./timeline/TimelinePages.jsx"),
  admin: () => import("./admin/ConfigPage.jsx"),
  people: () => import("./people/PeoplePage.jsx"),
  property: () => import("./property/PropertyPage.jsx"),
};
const LivePage = lazyExport(pageImporters.live, "LivePage");
const IncidentDetailPage = lazyExport(pageImporters.incident, "IncidentDetailPage");
const ObservationsPage = lazyExport(pageImporters.observations, "ObservationsPage");
const IncidentsPage = lazyExport(pageImporters.incidents, "IncidentsPage");
const ExportCenterPage = lazyExport(pageImporters.timelinePages, "ExportCenterPage");
const RecordingsPage = lazyExport(pageImporters.timelinePages, "RecordingsPage");
const SemanticSearchPage = lazyExport(pageImporters.timelinePages, "SemanticSearchPage");
const ConfigPage = lazyExport(pageImporters.admin, "ConfigPage");
const PeoplePage = lazyExport(pageImporters.people, "PeoplePage");
const PropertyPage = lazyExport(pageImporters.property, "PropertyPage");

// The page's code used to be requested only after the session check came
// back. Ask for it at once; the import is shared with the lazy component, and
// a failure here is reported when that component loads.
function preloadCurrentPage() {
  try {
    const pathname = appPathname();
    const page = /^\/incidents\/incident-[^/]+\/?$/.test(pathname) ? "incident" : resolveWorkspace(pathname)?.id;
    const importer = pageImporters[{ exports: "timelinePages", timeline: "timelinePages", search: "timelinePages" }[page] || page];
    importer?.().catch(() => {});
  } catch { /* the app renders its own error for an unusable address */ }
}

function WorkspaceFallback() {
  return <main className="workspace-not-found" aria-busy="true"><p>Loading workspace…</p></main>;
}

function App() {
  const [timeZone, setTimeZone] = useStoredState("survng.timeZone", DEFAULT_TIME_ZONE);
  const [theme, setTheme] = useStoredState("survng.theme", "dark");
  const [textSize, setTextSize] = useStoredState(TEXT_SIZE_STORAGE_KEY, "default");
  const [interfaceFont, setInterfaceFont] = useStoredState(INTERFACE_FONT_STORAGE_KEY, "inter");
  const [recordingContext, setRecordingContext] = useState(null);
  const [assistantAsk, setAssistantAsk] = useState(null);
  const [session, setSession] = useState(null);
  const pathname = appPathname();
  const workspace = resolveWorkspace(pathname);
  const page = workspace?.id || "not-found";
  const canonicalPath = canonicalWorkspaceUrl(pathname, window.location.search, window.location.hash);
  const [assistantContext, setAssistantContext] = useState({ page });
  const viewer = session?.user?.role === "viewer";
  function askAssistant(prompt) {
    setAssistantAsk({ id: Date.now(), prompt: String(prompt || "").trim() || "Analyze this incident" });
  }
  async function loadSession() {
    try {
      const response = await fetch("/api/auth/session");
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "Could not read session");
      setSession(payload);
    } catch {
      setSession({ enabled: false, bootstrap_required: false, user: null });
    }
  }
  async function signOut() {
    await fetch("/api/auth/logout", { method: "POST" });
    setSession((current) => ({ ...(current || {}), user: null, enabled: true, bootstrap_required: false }));
  }
  useEffect(() => {
    void loadSession();
    function onAuthRequired() {
      void loadSession();
    }
    window.addEventListener("survng:auth-required", onAuthRequired);
    return () => window.removeEventListener("survng:auth-required", onAuthRequired);
  }, []);
  useEffect(() => {
    const nextUrl = appUrl(canonicalPath);
    const currentUrl = `${window.location.pathname}${window.location.search}${window.location.hash}`;
    if (nextUrl !== currentUrl) window.history.replaceState(window.history.state, "", nextUrl);
  }, [canonicalPath]);
  useEffect(() => {
    setAssistantContext({ page });
  }, [page]);
  useEffect(() => {
    applyBrowserAppearance(THEMES.includes(theme) ? theme : "auto");
    if (theme !== "auto" || typeof window.matchMedia !== "function") return undefined;
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const sync = () => applyBrowserAppearance("auto");
    media.addEventListener?.("change", sync);
    media.addListener?.(sync);
    return () => {
      media.removeEventListener?.("change", sync);
      media.removeListener?.(sync);
    };
  }, [theme]);
  useEffect(() => {
    applyTextSize(textSize);
  }, [textSize]);

  useEffect(() => {
    applyInterfaceFont(interfaceFont);
  }, [interfaceFont]);
  if (!session) {
    return <main className="workspace-not-found" aria-busy="true"><p>Loading SurvNG…</p></main>;
  }
  if (session.enabled && !session.user) {
    return <LoginScreen session={session} onSignedIn={setSession} />;
  }
  const detailMatch = pathname.match(/^\/incidents\/(incident-[^/]+)\/?$/);
  if (detailMatch) {
    // The incident detail lives inside the workspace shell (top navigation,
    // Incidents active) but without the assistant panel.
    return <RuntimeStateProvider><Shell page="incidents" theme={theme} textSize={textSize} interfaceFont={interfaceFont} recordingContext={recordingContext} session={session} onSignOut={() => void signOut()}>
      <Suspense fallback={<WorkspaceFallback />}><IncidentDetailPage incidentId={(() => { try { return decodeURIComponent(detailMatch[1]); } catch { return detailMatch[1]; } })()} timeZone={timeZone} canCorrectIncident={!viewer} /></Suspense>
    </Shell></RuntimeStateProvider>;
  }
  const workspacePage = viewer && page === "admin" ? "live" : page;
  return (
    <RuntimeStateProvider><Shell page={workspacePage} theme={theme} textSize={textSize} interfaceFont={interfaceFont} recordingContext={recordingContext} session={session} onSignOut={() => void signOut()}>
      <Suspense fallback={<WorkspaceFallback />}>
        {workspacePage === "admin"
          ? <ConfigPage timeZone={timeZone} setTimeZone={setTimeZone} theme={theme} setTheme={setTheme} textSize={textSize} setTextSize={setTextSize} interfaceFont={interfaceFont} setInterfaceFont={setInterfaceFont} onAssistantContextChange={setAssistantContext} />
          : workspacePage === "exports"
            ? <ExportCenterPage timeZone={timeZone} onAssistantContextChange={setAssistantContext} />
            : workspacePage === "timeline"
              ? <RecordingsPage timeZone={timeZone} onAssistantContextChange={setAssistantContext} />
              : workspacePage === "search"
                ? <SemanticSearchPage timeZone={timeZone} onAssistantContextChange={setAssistantContext} />
                : workspacePage === "observations"
                  ? <ObservationsPage timeZone={timeZone} />
                  : workspacePage === "incidents"
                  ? <IncidentsPage timeZone={timeZone} canCorrectIncident={!viewer} onRecordingContextChange={setRecordingContext} onAssistantContextChange={setAssistantContext} onAskAssistant={askAssistant} />
                  : workspacePage === "people"
                    ? <PeoplePage timeZone={timeZone} />
                    : workspacePage === "property"
                      ? <PropertyPage canEdit={!viewer} />
                      : workspacePage === "live"
                      ? <LivePage timeZone={timeZone} onRecordingContextChange={setRecordingContext} onAssistantContextChange={setAssistantContext} />
                      : <main className="workspace-not-found"><CircleAlert size={30} /><h2>Page not found</h2><p>This SurvNG workspace does not exist.</p><a className="nav-button" href={appUrl("/")}>Return to Live</a></main>}
      </Suspense>
      {viewer ? null : <AssistantPanel pageContext={{ ...assistantContext, page: workspacePage }} timeZone={timeZone} askRequest={assistantAsk} onAskRequestHandled={() => setAssistantAsk(null)} />}
    </Shell></RuntimeStateProvider>
  );
}

preloadCurrentPage();
createRoot(document.getElementById("root")).render(<App />);
registerSurvngServiceWorker();
