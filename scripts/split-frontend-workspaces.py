#!/usr/bin/env python3
"""Split frontend/src/App.jsx into workspace folders. Safe to re-run on the original App.jsx."""

from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "frontend" / "src"
APP_PATH = SRC / "App.jsx"
APP = APP_PATH.read_text()
if "function LivePage" not in APP or APP.count("\n") < 10_000:
    raise SystemExit("App.jsx is already split or unexpected; aborting")
LINES = APP.splitlines(keepends=True)


def slice_lines(start: int, end: int) -> str:
    return "".join(LINES[start - 1 : end])


def export_toplevel(body: str) -> str:
    body = re.sub(r"^async function ", "export async function ", body, flags=re.M)
    body = re.sub(r"^function ", "export function ", body, flags=re.M)
    body = re.sub(r"^const ", "export const ", body, flags=re.M)
    body = re.sub(r"^let ", "export let ", body, flags=re.M)
    return body


def used(name: str, body: str) -> bool:
    return re.search(rf"\b{re.escape(name)}\b", body) is not None


LUCIDE = [
    "Activity", "ArrowLeft", "ArrowRight", "ArrowUpDown", "Bike", "Bot", "BusFront",
    "Camera", "CarFront", "Cat", "Check", "CircleAlert", "ChevronLeft", "ChevronRight",
    "Copy", "CircleDot", "Clock3", "Crop", "Cog", "Download", "Dog", "Cpu", "Film",
    "Gauge", "Grid2X2", "Images", "GripVertical", "HardDrive", "Search", "ListTree",
    "Maximize2", "Monitor", "Moon", "Pause", "PanelLeftClose", "PanelLeftOpen", "Play",
    "Plus", "Power", "Radar", "Radio", "RefreshCcw", "RotateCcw", "Save", "ScanFace",
    "ShieldCheck", "SlidersHorizontal", "Sparkles", "Siren", "SkipBack", "SkipForward",
    "Sun", "Trash2", "Truck", "Undo2", "UserRound", "UserPlus", "Users", "Rows3",
    "Video", "Wrench", "X",
]

REACT_NAMES = [
    "forwardRef", "useContext", "useEffect", "useImperativeHandle", "useLayoutEffect",
    "useMemo", "useRef", "useState",
]

EXTERNAL = {
    "buildMotionDecisionFusion": "../motionDecisionConfig.mjs",
    "MOTION_BEHAVIOR_OPTIONS": "../motionDecisionConfig.mjs",
    "motionBehaviorOption": "../motionDecisionConfig.mjs",
    "motionBehaviorSettings": "../motionDecisionConfig.mjs",
    "motionBehaviorValue": "../motionDecisionConfig.mjs",
    "motionModeInfo": "../motionDecisionConfig.mjs",
    "readMotionDecisionFusion": "../motionDecisionConfig.mjs",
    "availableQualificationPresets": "../motionAnalysisConfig.mjs",
    "motionAnalysisPresetSelectionUseful": "../motionAnalysisConfig.mjs",
    "presetQualificationGraph": "../motionAnalysisConfig.mjs",
    "readMotionAnalysisPreset": "../motionAnalysisConfig.mjs",
    "TUNEUP_PERIODS": "../detectionTuneup.mjs",
    "TUNEUP_SETTING_NAMES": "../detectionTuneup.mjs",
    "tuneupHistoryTitle": "../detectionTuneup.mjs",
    "tuneupOutcome": "../detectionTuneup.mjs",
    "tuneupRecommendationGroup": "../detectionTuneup.mjs",
    "tuneupValue": "../detectionTuneup.mjs",
    "formatServerUptime": "../duration.mjs",
    "clearWebRtcFailure": "../liveTransport.mjs",
    "initialLiveTransport": "../liveTransport.mjs",
    "nextNativeFallbackSource": "../liveTransport.mjs",
    "rememberWebRtcFailure": "../liveTransport.mjs",
    "webRtcRetryDelay": "../liveTransport.mjs",
    "aspectFromDimensions": "../liveAspect.mjs",
    "cameraSourceAspect": "../liveAspect.mjs",
    "initialCameraAspect": "../liveAspect.mjs",
    "liveAspectStorageKey": "../liveAspect.mjs",
    "normalizedLiveSource": "../liveAspect.mjs",
    "validLiveAspect": "../liveAspect.mjs",
    "resetLiveDefaultsForServer": "../liveDefaults.mjs",
    "browserStorage": "../storage.mjs",
    "readStoredValue": "../storage.mjs",
    "removeStoredValue": "../storage.mjs",
    "writeStoredValue": "../storage.mjs",
    "readAssistantHistory": "../assistantStorage.mjs",
    "writeAssistantHistory": "../assistantStorage.mjs",
    "assistantContextLabel": "../assistantContext.mjs",
    "assistantContextPrompts": "../assistantContext.mjs",
    "snapshotAssistantContext": "../assistantContext.mjs",
    "safeMediaUrl": "../mediaUrl.mjs",
    "liveMediaShouldRun": "../pollingPolicy.mjs",
    "liveSnapshotRefreshMs": "../pollingPolicy.mjs",
    "logPayloadSignature": "../pollingPolicy.mjs",
    "useVisiblePolling": "../visibilityPolling.mjs",
    "assistantEvidenceHref": "../assistantNavigation.mjs",
    "assistantIncidentHref": "../assistantNavigation.mjs",
    "containedFrameTransform": "../objectTrackReplay.mjs",
    "hlsPlaybackOffset": "../objectTrackReplay.mjs",
    "hlsProgramStartEpoch": "../objectTrackReplay.mjs",
    "incidentTrackingSource": "../objectTrackReplay.mjs",
    "playbackEpochAt": "../objectTrackReplay.mjs",
    "storedObjectTracks": "../objectTrackReplay.mjs",
    "trackFrameAt": "../objectTrackReplay.mjs",
    "adjustRecordingExportRange": "../recordingPlayback.mjs",
    "describePlaybackError": "../recordingPlayback.mjs",
    "gridPlaybackNeedsSeek": "../recordingPlayback.mjs",
    "isUnsupportedPlaybackError": "../recordingPlayback.mjs",
    "mergeRecordingAvailability": "../recordingPlayback.mjs",
    "playbackMediaTimeForEpoch": "../recordingPlayback.mjs",
    "playbackRowsCoverEpoch": "../recordingPlayback.mjs",
    "recordingCameraAspect": "../recordingGrid.mjs",
    "recordingGridBestEpoch": "../recordingGrid.mjs",
    "liveCustomDropTarget": "../liveCustomLayout.mjs",
    "liveCustomGridMetrics": "../liveCustomLayout.mjs",
    "liveCustomTilePlacement": "../liveCustomLayout.mjs",
    "moveLiveCamera": "../liveCustomLayout.mjs",
    "readLiveCustomLayout": "../liveCustomLayout.mjs",
    "resizeLiveCamera": "../liveCustomLayout.mjs",
    "resizeLiveCameraToAspect": "../liveCustomLayout.mjs",
    "focusedLiveCameraId": "../liveWorkspace.mjs",
    "LIVE_DENSITY_OPTIONS": "../liveWorkspace.mjs",
    "liveActivityEventId": "../liveWorkspace.mjs",
    "liveActivityIncidentHref": "../liveWorkspace.mjs",
    "liveActivityQuickFilter": "../liveWorkspace.mjs",
    "liveActivityQuickSelection": "../liveWorkspace.mjs",
    "liveDensityPage": "../liveWorkspace.mjs",
    "normalizedLiveDensity": "../liveWorkspace.mjs",
    "orderedLiveCamerasForFocus": "../liveWorkspace.mjs",
    "uniformLiveGridLayout": "../liveWorkspace.mjs",
    "camerasWithLiveFraming": "../liveFraming.mjs",
    "liveFramingStyle": "../liveFraming.mjs",
    "normalizedLiveFraming": "../liveFraming.mjs",
    "expectedTimelineCameras": "../timelineWorkspace.mjs",
    "filteredTimelineCameras": "../timelineWorkspace.mjs",
    "normalizedTimelinePlaybackRate": "../timelineWorkspace.mjs",
    "parseTimelineView": "../timelineWorkspace.mjs",
    "timelineCompanionGrid": "../timelineWorkspace.mjs",
    "timelineEventMatchesFilter": "../timelineWorkspace.mjs",
    "timelineEvidenceWindow": "../timelineWorkspace.mjs",
    "timelineStageCameras": "../timelineWorkspace.mjs",
    "timelineStagePage": "../timelineWorkspace.mjs",
    "timelineViewport": "../timelineWorkspace.mjs",
    "timelineViewportPage": "../timelineWorkspace.mjs",
    "TIMELINE_PLAYBACK_RATES": "../timelineWorkspace.mjs",
    "adjacentIncident": "../incidentNavigation.mjs",
    "createIncidentPageCache": "../incidentNavigation.mjs",
    "incidentArrowNavigationAllowed": "../incidentNavigation.mjs",
    "incidentDetectionFrameSize": "../incidentNavigation.mjs",
    "incidentDetailQuery": "../incidentNavigation.mjs",
    "incidentEvidenceFrames": "../incidentNavigation.mjs",
    "incidentMosaicEvents": "../incidentNavigation.mjs",
    "incidentMosaicPage": "../incidentNavigation.mjs",
    "incidentObjectIconName": "../incidentNavigation.mjs",
    "incidentProgressiveImageWidth": "../incidentNavigation.mjs",
    "incidentSelectionHref": "../incidentNavigation.mjs",
    "incidentThumbnailPageSize": "../incidentNavigation.mjs",
    "incidentTrackingFrameSize": "../incidentNavigation.mjs",
    "incidentZoomLayout": "../incidentNavigation.mjs",
    "incidentsNewestFirst": "../incidentNavigation.mjs",
    "incidentTriggerLabel": "../incidentNavigation.mjs",
    "linkedIncidentEventFilter": "../incidentNavigation.mjs",
    "retainFocusedIncident": "../incidentNavigation.mjs",
    "showIncidentCardAnnotations": "../incidentNavigation.mjs",
    "motionAuditRegions": "../motionAudit.mjs",
    "addSemanticSearchHistory": "../semanticSearchState.mjs",
    "clearSemanticSearchSession": "../semanticSearchState.mjs",
    "readSemanticSearchHistory": "../semanticSearchState.mjs",
    "readSemanticSearchSession": "../semanticSearchState.mjs",
    "semanticSearchResultsForCamera": "../semanticSearchState.mjs",
    "writeSemanticSearchHistory": "../semanticSearchState.mjs",
    "writeSemanticSearchSession": "../semanticSearchState.mjs",
    "mapWithConcurrency": "../incidentSemanticSearch.mjs",
    "rankSemanticIncidentDetails": "../incidentSemanticSearch.mjs",
    "semanticIncidentRequest": "../incidentSemanticSearch.mjs",
    "insertZonePointWithIndex": "../zoneGeometry.mjs",
    "relatedEvidenceLabel": "../relatedIncidents.mjs",
    "relatedIncidentThumbnailPath": "../relatedIncidents.mjs",
    "relatedIncidentsPath": "../relatedIncidents.mjs",
    "visibleRelatedAppearances": "../relatedIncidents.mjs",
    "nextFaceReviewObservation": "../faceReview.mjs",
    "PEOPLE_REVIEW_FILTERS": "../peopleWorkspace.mjs",
    "peopleWorkspaceSearch": "../peopleWorkspace.mjs",
    "readPeopleWorkspaceQuery": "../peopleWorkspace.mjs",
    "ADMIN_RESPONSIBILITY_GROUPS": "../adminWorkspace.mjs",
    "GENERAL_SECTION_LABELS": "../adminWorkspace.mjs",
    "adminDestination": "../adminWorkspace.mjs",
    "adminWorkspaceSearch": "../adminWorkspace.mjs",
    "cameraConfigDirtyState": "../adminWorkspace.mjs",
    "comparableCameraSettings": "../adminWorkspace.mjs",
    "comparableSystemConfig": "../adminWorkspace.mjs",
    "configValuesEqual": "../adminWorkspace.mjs",
    "nextTabId": "../adminWorkspace.mjs",
    "preferredStoredValue": "../adminWorkspace.mjs",
    "readAdminSubsection": "../adminWorkspace.mjs",
    "readAdminWorkspace": "../adminWorkspace.mjs",
    "canonicalWorkspaceUrl": "../workspaceNavigation.mjs",
    "DESKTOP_PRIMARY_WORKSPACES": "../workspaceNavigation.mjs",
    "MOBILE_PRIMARY_WORKSPACES": "../workspaceNavigation.mjs",
    "resolveWorkspace": "../workspaceNavigation.mjs",
    "systemHealthState": "../workspaceNavigation.mjs",
    "timelineWorkspaceHref": "../workspaceNavigation.mjs",
    "workspaceDefinition": "../workspaceNavigation.mjs",
    "workspaceHref": "../workspaceNavigation.mjs",
}

EXPORT_RE = re.compile(
    r"^export (?:async )?function (\w+)|^export const (\w+)|^export let (\w+)|^export \{ ([^}]+) \}",
    re.M,
)

REGISTRY: dict[str, str] = {}  # symbol -> import path relative to src/


def register_file(path: Path) -> None:
    text = path.read_text()
    rel = "./" + path.relative_to(SRC).as_posix()
    for match in EXPORT_RE.finditer(text):
        if match.group(4):
            for part in match.group(4).split(","):
                name = part.strip().split(" as ")[0].strip()
                if name:
                    REGISTRY[name] = rel
        else:
            name = match.group(1) or match.group(2) or match.group(3)
            REGISTRY[name] = rel


def import_path_for(target: str, from_file: Path) -> str:
    dest = (SRC / target[2:]).resolve()
    rel = Path(os.path.relpath(dest, from_file.parent)).as_posix()
    return rel if rel.startswith(".") else f"./{rel}"


def build_header(body: str, dest: Path, skip: set[str] | None = None) -> str:
    skip = skip or set()
    lines: list[str] = []
    react = [n for n in REACT_NAMES if used(n, body)]
    needs_react = dest.suffix == ".jsx" or bool(react) or used("createPortal", body)
    if needs_react:
        if react:
            lines.append(f'import React, {{ {", ".join(react)} }} from "react";')
        else:
            lines.append('import React from "react";')
    if used("createPortal", body):
        lines.append('import { createPortal } from "react-dom";')
    icons = [name for name in LUCIDE if used(name, body)]
    if icons:
        lines.append("import {\n  " + ",\n  ".join(icons) + ',\n} from "lucide-react";')

    ext_groups: dict[str, list[str]] = defaultdict(list)
    for name, path in EXTERNAL.items():
        if name in skip or not used(name, body):
            continue
        spec = f"timelineHref as timelineWorkspaceHref" if name == "timelineWorkspaceHref" else name
        ext_groups[path].append(spec)
    for path, names in ext_groups.items():
        lines.append(f'import {{ {", ".join(names)} }} from "{path}";')

    shared_groups: dict[str, list[str]] = defaultdict(list)
    for name, target in REGISTRY.items():
        if name in skip or not used(name, body):
            continue
        if Path(SRC / target[2:]) == dest:
            continue
        shared_groups[import_path_for(target, dest)].append(name)
    for path, names in shared_groups.items():
        unique = list(dict.fromkeys(names))
        lines.append(f'import {{ {", ".join(unique)} }} from "{path}";')
    return "\n".join(lines) + "\n\n"


def write_module(dest: Path, body: str, *, jsx: bool | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    exported = export_toplevel(body)
    defined = set()
    for match in EXPORT_RE.finditer(exported):
        if match.group(4):
            continue
        defined.add(match.group(1) or match.group(2) or match.group(3))
    header = build_header(exported, dest, skip=defined)
    dest.write_text(header + exported)
    register_file(dest)
    print(f"wrote {dest.relative_to(ROOT)} ({len(dest.read_text().splitlines())} lines)")


def write_raw(dest: Path, text: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text)
    register_file(dest)
    print(f"wrote {dest.relative_to(ROOT)} ({len(text.splitlines())} lines)")


# --- curated shared modules ---

write_raw(
    SRC / "shared" / "api.js",
    '''import { safeMediaUrl } from "../mediaUrl.mjs";
import { timelineHref as timelineWorkspaceHref, workspaceHref } from "../workspaceNavigation.mjs";

export const APP_BASE_PATH = String(window.__SURVNG_BASE_PATH__ || "").replace(/\\/+$/, "");
document.documentElement.dataset.embedded = window.self !== window.top ? "true" : "false";

export function appUrl(path = "/") {
  if (typeof path !== "string" || !path.startsWith("/") || path.startsWith("//")) return path;
  if (APP_BASE_PATH && (path === APP_BASE_PATH || path.startsWith(`${APP_BASE_PATH}/`))) return path;
  return `${APP_BASE_PATH}${path}`;
}

export function mediaUrl(value) {
  return safeMediaUrl(value, APP_BASE_PATH, window.location.origin);
}

export function appPathname() {
  const pathname = window.location.pathname;
  if (!APP_BASE_PATH || (!pathname.startsWith(`${APP_BASE_PATH}/`) && pathname !== APP_BASE_PATH)) return pathname;
  return pathname.slice(APP_BASE_PATH.length) || "/";
}

export function incidentRecordingContext(item) {
  if (!item?.camera_id || !item?.created_at) return null;
  const epoch = new Date(item.created_at).getTime() / 1000;
  if (!Number.isFinite(epoch)) return null;
  return { cameraId: item.camera_id, epoch };
}

export function recordingsHref(context) {
  if (!context?.cameraId || !Number.isFinite(context?.epoch)) return appUrl(workspaceHref("timeline"));
  return appUrl(timelineWorkspaceHref({
    cameraId: context.cameraId,
    epoch: Math.round(context.epoch * 1000) / 1000,
    source: context.source,
  }));
}

export const fetch = (resource, options) => window.fetch(
  typeof resource === "string" ? appUrl(resource) : resource,
  options,
);
''',
)

write_raw(
    SRC / "shared" / "constants.js",
    '''export const DEFAULT_TIME_ZONE = "America/New_York";
export const MEDIA_STORAGE_ROLES = [
  ["recordings", "Recordings"],
  ["snapshots", "Snapshots"],
  ["motion_audits", "Motion audits"],
  ["clips", "Clips"],
  ["exports", "Exports"],
];
export const CAMERA_ADMIN_SECTIONS = ["settings", "motion", "zones", "info"];
export const TELEMETRY_ADMIN_SECTIONS = ["overview", "cameras", "diagnostics"];
export const GENERAL_ADMIN_SECTIONS = ["general", "storage", "mqtt", "access", "detection", "motion-review"];
export const LEGACY_INCIDENT_FILTER_KEYS = [
  "survng.liveEventFilter.v2",
  "survng.incidentDay.v1",
  "survng.incidentCameraFilter.v1",
  "survng.incidentObjectFilter.v1",
  "survng.incidentZoneFilter.v1",
];
export const US_TIME_ZONES = [
  ["America/New_York", "Eastern"],
  ["America/Chicago", "Central"],
  ["America/Denver", "Mountain"],
  ["America/Phoenix", "Arizona"],
  ["America/Los_Angeles", "Pacific"],
  ["America/Anchorage", "Alaska"],
  ["Pacific/Honolulu", "Hawaii"],
];
export const THEMES = ["auto", "light", "dark"];
export const SECRET_PLACEHOLDER = "__SURVNG_SECRET_SET__";
export const PREFER_NATIVE_HLS = /iPad|iPhone|iPod/.test(navigator.userAgent)
  || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
export const APP_EVENT_TYPES = ["camera_state", "cameras_state", "motion", "object", "incident", "system_state"];
export const INCIDENT_REFRESH_FALLBACK_MS = 15_000;
export const STREAM_MODES = ["motion", "mjpeg", "webrtc"];
export const STREAM_LABELS = {
  motion: "Auto",
  mjpeg: "MJPEG",
  webrtc: "WebRTC",
};
export const MOTION_WEBRTC_HOLD_MS = 30_000;
export const LIVE_TRANSPORT_LABELS = {
  webrtc: "WebRTC",
  mse: "MSE",
  mjpeg: "MJPEG",
  recording: "Recording",
  snapshot: "Snapshot",
};
export const ALL_RECORDING_CAMERAS_ID = "all";
''',
)

write_raw(
    SRC / "shared" / "secrets.js",
    '''import { SECRET_PLACEHOLDER } from "./constants.js";

'''
    + export_toplevel(slice_lines(286, 301)),
)
register_file(SRC / "shared" / "secrets.js")

write_module(SRC / "shared" / "format.js", slice_lines(911, 996))
write_module(SRC / "shared" / "datetime.js", slice_lines(1082, 1123))
write_module(SRC / "shared" / "hooks.js", slice_lines(870, 888) + slice_lines(998, 1060))
write_module(
    SRC / "shared" / "cameras.js",
    slice_lines(890, 909)
    + slice_lines(1062, 1068)
    + slice_lines(1078, 1080)
    + slice_lines(1125, 1213),
)
write_module(SRC / "shared" / "identity.jsx", slice_lines(230, 278))
write_module(
    SRC / "shared" / "mediaUrls.js",
    slice_lines(853, 868) + slice_lines(6255, 6263) + slice_lines(6315, 6377),
)
write_raw(
    SRC / "shared" / "events.js",
    '''import { useEffect, useRef } from "react";
import { appUrl } from "./api.js";
import { APP_EVENT_TYPES } from "./constants.js";

const appEventListeners = new Set();
let appEventSource = null;
let appEventCloseTimer = null;

'''
    + export_toplevel(slice_lines(1936, 1973)),
)
register_file(SRC / "shared" / "events.js")

write_module(SRC / "shared" / "media.jsx", slice_lines(303, 851))
write_module(SRC / "shared" / "polling.js", slice_lines(2065, 2166))
write_module(
    SRC / "shared" / "evidence.jsx",
    slice_lines(2674, 4726) + slice_lines(6265, 6313),
)

write_module(SRC / "shell" / "Shell.jsx", slice_lines(1799, 1928) + slice_lines(1975, 2063))
write_module(SRC / "assistant" / "AssistantPanel.jsx", slice_lines(1406, 1797))
write_module(
    SRC / "live" / "LivePage.jsx",
    slice_lines(2176, 2673) + slice_lines(5440, 6253),
)
write_module(SRC / "incidents" / "IncidentsPage.jsx", slice_lines(4727, 5439))
write_module(
    SRC / "timeline" / "TimelinePages.jsx",
    slice_lines(6378, 6624) + slice_lines(6627, 8844),
)
write_module(
    SRC / "admin" / "cameraEditors.jsx",
    slice_lines(1214, 1404),
)
write_module(
    SRC / "admin" / "ConfigPage.jsx",
    slice_lines(151, 165)
    + slice_lines(187, 191)
    + slice_lines(8846, 13404),
)
write_module(SRC / "people" / "FacesPage.jsx", slice_lines(13406, 13854))

# CSS: extract documented campaign sections while preserving cascade via App.jsx import order.
STYLES = SRC / "styles.css"
style_lines = STYLES.read_text().splitlines(keepends=True)


def write_css(path: Path, start: int, end: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(style_lines[start - 1 : end]))
    print(f"wrote {path.relative_to(ROOT)} ({end - start + 1} lines)")


write_css(SRC / "shell" / "shell.css", 14265, 15083)
write_css(SRC / "timeline" / "timeline.css", 15084, 16203)
write_css(SRC / "live" / "live.css", 16204, 16619)
write_css(SRC / "timeline" / "investigation.css", 16620, 16648)
write_css(SRC / "incidents" / "incidents.css", 16649, 16875)
write_css(SRC / "search" / "search.css", 16876, 17243)
write_css(SRC / "admin" / "admin.css", 17244, 18113)
write_css(SRC / "shell" / "responsive.css", 18114, 18391)
write_css(SRC / "admin" / "workspace.css", 18392, 19017)
write_css(SRC / "shell" / "mobile.css", 19018, len(style_lines))
(SRC / "people" / "people.css").parent.mkdir(parents=True, exist_ok=True)
(SRC / "people" / "people.css").write_text(
    "/* People workspace styles live with Search in search/search.css (shared campaign). */\n"
)
(SRC / "assistant" / "assistant.css").write_text(
    "/* Assistant chrome remains in styles.css until a later CSS ownership pass. */\n"
)
print("wrote people/people.css and assistant/assistant.css stubs")

STYLES.write_text("".join(style_lines[:14264]))
print("truncated styles.css to shared/base rules")

app_jsx = '''import React, { useEffect, useState } from "react";
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
import { LivePage } from "./live/LivePage.jsx";
import { IncidentsPage } from "./incidents/IncidentsPage.jsx";
import { ExportCenterPage, RecordingsPage, SemanticSearchPage } from "./timeline/TimelinePages.jsx";
import { ConfigPage } from "./admin/ConfigPage.jsx";
import { FacesPage } from "./people/FacesPage.jsx";
import { canonicalWorkspaceUrl, resolveWorkspace } from "./workspaceNavigation.mjs";

function App() {
  const [timeZone, setTimeZone] = useStoredState("survng.timeZone", DEFAULT_TIME_ZONE);
  const [theme, setTheme] = useStoredState("survng.theme", "dark");
  const [recordingContext, setRecordingContext] = useState(null);
  const pathname = appPathname();
  const workspace = resolveWorkspace(pathname);
  const page = workspace?.id || "not-found";
  const canonicalPath = canonicalWorkspaceUrl(pathname, window.location.search, window.location.hash);
  const isExportCenter = pathname.startsWith("/recordings/exports") || pathname.startsWith("/timeline/exports");
  const isSemanticSearch = page === "search";
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
    <Shell page={page} theme={theme} recordingContext={recordingContext}>
      {page === "admin"
        ? <ConfigPage timeZone={timeZone} setTimeZone={setTimeZone} theme={theme} setTheme={setTheme} onAssistantContextChange={setAssistantContext} />
        : page === "timeline"
          ? isExportCenter
            ? <ExportCenterPage timeZone={timeZone} onAssistantContextChange={setAssistantContext} />
            : <RecordingsPage timeZone={timeZone} onAssistantContextChange={setAssistantContext} />
          : isSemanticSearch
            ? <SemanticSearchPage timeZone={timeZone} onAssistantContextChange={setAssistantContext} />
            : page === "incidents"
              ? <IncidentsPage timeZone={timeZone} onRecordingContextChange={setRecordingContext} onAssistantContextChange={setAssistantContext} />
              : page === "people"
                ? <FacesPage timeZone={timeZone} onAssistantContextChange={setAssistantContext} />
                : page === "live"
                  ? <LivePage timeZone={timeZone} onRecordingContextChange={setRecordingContext} onAssistantContextChange={setAssistantContext} />
                  : <main className="workspace-not-found"><CircleAlert size={30} /><h2>Page not found</h2><p>This SurvNG workspace does not exist.</p><a className="nav-button" href={appUrl("/")}>Return to Live</a></main>}
      <AssistantPanel pageContext={{ ...assistantContext, page: isExportCenter ? "exports" : page }} timeZone={timeZone} />
    </Shell>
  );
}

createRoot(document.getElementById("root")).render(<App />);
'''
APP_PATH.write_text(app_jsx)
print("wrote slim App.jsx")
