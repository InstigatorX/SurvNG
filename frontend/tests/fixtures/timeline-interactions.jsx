import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { RecordingsPage } from "../../src/timeline/TimelinePages.jsx";
import { Shell } from "../../src/shell/Shell.jsx";
import "../../src/styles.css";
import "../../src/shell/shell.css";
import "../../src/timeline/timeline.css";
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

const ids = new WeakMap();
let nextId = 1;
const events = [];
let context = null;
const onContext = (value) => { context = value; };
const attached = new WeakSet();
const video = () => document.querySelector(".recording-hero-media video");
function snapshot() {
  const element = video();
  if (element && !ids.has(element)) ids.set(element, nextId++);
  const track = document.querySelector(".recordings-v2-track");
  const slider = track?.querySelector('input[type="range"]');
  return {
    url: location.href, videoId: element ? ids.get(element) : null,
    src: element?.currentSrc || element?.src || "", currentTime: element?.currentTime,
    duration: element?.duration, paused: element?.paused, seeking: element?.seeking,
    readyState: element?.readyState, error: element?.error?.code || null,
    context, playhead: context?.recording_epoch,
    rangeOffset: Number(slider?.value), rangeSeconds: Number(slider?.max),
    ticks: [...document.querySelectorAll(".recordings-v2-ticks small")].map(node => node.textContent),
    notices: [...document.querySelectorAll(".recordings-v2-notice,.recordings-v2-message")].map(node => node.textContent),
    events: events.slice(-15),
  };
}
window.timelineFixture = { snapshot, video, events, requests: async () => (await fetch("/timeline-interactions-requests.json")).json() };
function Fixture() {
  const [status, setStatus] = useState({});
  useEffect(() => {
    const timer = setInterval(() => {
      const element = video();
      if (element && !attached.has(element)) {
        attached.add(element);
        for (const name of ["loadedmetadata", "seeking", "seeked", "playing", "pause", "error"]) {
          element.addEventListener(name, () => events.push({ name, time: element.currentTime, at: performance.now() }));
        }
      }
      setStatus(snapshot());
    }, 250);
    return () => clearInterval(timer);
  }, []);
  return <>
    <Shell page="timeline" theme="dark"><RecordingsPage timeZone="UTC" onAssistantContextChange={onContext} /></Shell>
    <details style={{ position: "fixed", bottom: 0, right: 0, zIndex: 500, maxWidth: "95vw", background: "#111", color: "white", fontSize: 10 }}>
      <summary>Fixture diagnostics · video {status.videoId || "—"} · {Number(status.currentTime || 0).toFixed(1)}s · ready {status.readyState || 0}</summary>
      <pre style={{ maxHeight: "45vh", overflow: "auto", maxWidth: "95vw", whiteSpace: "pre-wrap" }}>{JSON.stringify(status, null, 2)}</pre>
    </details>
  </>;
}
createRoot(document.getElementById("root")).render(<Fixture />);
