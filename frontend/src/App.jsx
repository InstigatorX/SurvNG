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

import { appPathname, appUrl } from "./shared/api.js";
import { DEFAULT_TIME_ZONE, THEMES } from "./shared/constants.js";
import { useStoredState } from "./shared/hooks.js";
import { Shell } from "./shell/Shell.jsx";
import { AssistantPanel } from "./assistant/AssistantPanel.jsx";
import { RuntimeStateProvider } from "./shared/runtimeState.jsx";
import { canonicalWorkspaceUrl, resolveWorkspace } from "./workspaceNavigation.mjs";

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
  const pathname = appPathname();
  const workspace = resolveWorkspace(pathname);
  const page = workspace?.id || "not-found";
  const canonicalPath = canonicalWorkspaceUrl(pathname, window.location.search, window.location.hash);
  const [assistantContext, setAssistantContext] = useState({ page });
  useEffect(() => {
    const nextUrl = appUrl(canonicalPath);
    const currentUrl = `${window.location.pathname}${window.location.search}${window.location.hash}`;
    if (nextUrl !== currentUrl) window.history.replaceState(window.history.state, "", nextUrl);
  }, [canonicalPath]);
  useEffect(() => {
    setAssistantContext({ page });
  }, [page]);
  useEffect(() => {
    document.documentElement.dataset.theme = THEMES.includes(theme) ? theme : "auto";
  }, [theme]);
  return (
    <RuntimeStateProvider><Shell page={page} theme={theme} recordingContext={recordingContext}>
      <Suspense fallback={<WorkspaceFallback />}>
        {page === "admin"
          ? <ConfigPage timeZone={timeZone} setTimeZone={setTimeZone} theme={theme} setTheme={setTheme} onAssistantContextChange={setAssistantContext} />
          : page === "exports"
            ? <ExportCenterPage timeZone={timeZone} onAssistantContextChange={setAssistantContext} />
            : page === "timeline"
              ? <RecordingsPage timeZone={timeZone} onAssistantContextChange={setAssistantContext} />
              : page === "search"
                ? <SemanticSearchPage timeZone={timeZone} onAssistantContextChange={setAssistantContext} />
                : page === "incidents"
                  ? <IncidentsPage timeZone={timeZone} onRecordingContextChange={setRecordingContext} onAssistantContextChange={setAssistantContext} />
                  : page === "people"
                    ? <FacesPage timeZone={timeZone} onAssistantContextChange={setAssistantContext} />
                    : page === "live"
                      ? <LivePage timeZone={timeZone} onRecordingContextChange={setRecordingContext} onAssistantContextChange={setAssistantContext} />
                      : <main className="workspace-not-found"><CircleAlert size={30} /><h2>Page not found</h2><p>This SurvNG workspace does not exist.</p><a className="nav-button" href={appUrl("/")}>Return to Live</a></main>}
      </Suspense>
      <AssistantPanel pageContext={{ ...assistantContext, page }} timeZone={timeZone} />
    </Shell></RuntimeStateProvider>
  );
}

createRoot(document.getElementById("root")).render(<App />);
