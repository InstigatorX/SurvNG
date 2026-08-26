import React, { lazy, Suspense, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { CircleAlert } from "lucide-react";

import "./styles.css";
import "./shell/shell.css";
import "./timeline/timeline.css";
import "./live/live.css";
import "./timeline/investigation.css";
import "./incidents/incidents.css";
import "./search/search.css";
import "./people/people.css";
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

function lazyExport(importer, exportName) {
  return lazy(() => importer().then((module) => ({ default: module[exportName] })));
}

const LivePage = lazyExport(() => import("./live/LivePage.jsx"), "LivePage");
const IncidentsPage = lazyExport(() => import("./incidents/IncidentsPage.jsx"), "IncidentsPage");
const ExportCenterPage = lazyExport(() => import("./timeline/TimelinePages.jsx"), "ExportCenterPage");
const RecordingsPage = lazyExport(() => import("./timeline/TimelinePages.jsx"), "RecordingsPage");
const SemanticSearchPage = lazyExport(() => import("./timeline/TimelinePages.jsx"), "SemanticSearchPage");
const ConfigPage = lazyExport(() => import("./admin/ConfigPage.jsx"), "ConfigPage");
const FacesPage = lazyExport(() => import("./people/FacesPage.jsx"), "FacesPage");

function WorkspaceFallback() {
  return <main className="workspace-not-found" aria-busy="true"><p>Loading workspace…</p></main>;
}

function App() {
  const [timeZone, setTimeZone] = useStoredState("survng.timeZone", DEFAULT_TIME_ZONE);
  const [theme, setTheme] = useStoredState("survng.theme", "dark");
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
  if (!session) {
    return <main className="workspace-not-found" aria-busy="true"><p>Loading SurvNG…</p></main>;
  }
  if (session.enabled && !session.user) {
    return <LoginScreen session={session} onSignedIn={setSession} />;
  }
  const workspacePage = viewer && page === "admin" ? "live" : page;
  return (
    <RuntimeStateProvider><Shell page={workspacePage} theme={theme} recordingContext={recordingContext} session={session} onSignOut={() => void signOut()}>
      <Suspense fallback={<WorkspaceFallback />}>
        {workspacePage === "admin"
          ? <ConfigPage timeZone={timeZone} setTimeZone={setTimeZone} theme={theme} setTheme={setTheme} onAssistantContextChange={setAssistantContext} />
          : workspacePage === "exports"
            ? <ExportCenterPage timeZone={timeZone} onAssistantContextChange={setAssistantContext} />
            : workspacePage === "timeline"
              ? <RecordingsPage timeZone={timeZone} onAssistantContextChange={setAssistantContext} onAskAssistant={askAssistant} />
              : workspacePage === "search"
                ? <SemanticSearchPage timeZone={timeZone} onAssistantContextChange={setAssistantContext} />
                : workspacePage === "incidents"
                  ? <IncidentsPage timeZone={timeZone} onRecordingContextChange={setRecordingContext} onAssistantContextChange={setAssistantContext} onAskAssistant={askAssistant} />
                  : workspacePage === "people"
                    ? <FacesPage timeZone={timeZone} onAssistantContextChange={setAssistantContext} />
                    : workspacePage === "live"
                      ? <LivePage timeZone={timeZone} onRecordingContextChange={setRecordingContext} onAssistantContextChange={setAssistantContext} />
                      : <main className="workspace-not-found"><CircleAlert size={30} /><h2>Page not found</h2><p>This SurvNG workspace does not exist.</p><a className="nav-button" href={appUrl("/")}>Return to Live</a></main>}
      </Suspense>
      {viewer ? null : <AssistantPanel pageContext={{ ...assistantContext, page: workspacePage }} timeZone={timeZone} askRequest={assistantAsk} onAskRequestHandled={() => setAssistantAsk(null)} />}
    </Shell></RuntimeStateProvider>
  );
}

createRoot(document.getElementById("root")).render(<App />);
registerSurvngServiceWorker();
