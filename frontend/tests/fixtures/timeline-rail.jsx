import React from "react";
import { createRoot } from "react-dom/client";
import { RecordingsPage } from "../../src/timeline/TimelinePages.jsx";
import { Shell } from "../../src/shell/Shell.jsx";
import "../../src/styles.css";
import "../../src/shell/shell.css";
import "../../src/timeline/timeline.css";
import "../../src/timeline/rail.css";
import "../../src/live/live.css";
import "../../src/timeline/investigation.css";
import "../../src/incidents/incidents.css";
import "../../src/search/search.css";
import "../../src/people/people.css";
import "../../src/admin/admin.css";
import "../../src/shell/responsive.css";
import "../../src/admin/workspace.css";
import "../../src/shell/mobile.css";
import "../../src/auth/login.css";

const theme = new URLSearchParams(location.search).get("theme") || "dark";
document.documentElement.dataset.theme = theme;
createRoot(document.getElementById("root")).render(
  <Shell page="timeline" theme={theme}><RecordingsPage timeZone="UTC" onAssistantContextChange={() => {}} /></Shell>,
);
