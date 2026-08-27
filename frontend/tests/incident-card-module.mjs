import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const directory = dirname(fileURLToPath(import.meta.url));
const evidence = readFileSync(join(directory, "../src/shared/evidence.jsx"), "utf8");
const incidentCard = readFileSync(join(directory, "../src/incidents/IncidentCard.jsx"), "utf8");
const incidentsPage = readFileSync(join(directory, "../src/incidents/IncidentsPage.jsx"), "utf8");
const livePage = readFileSync(join(directory, "../src/live/LivePage.jsx"), "utf8");

assert.match(incidentCard, /export function IncidentCard\(/);
assert.match(incidentCard, /export function IncidentInspector\(/);
assert.match(incidentCard, /const clipWindow = incidentClipWindow\(/);
assert.doesNotMatch(incidentCard, /const window = incidentClipWindow\(/);
assert.match(incidentCard, /window\.requestAnimationFrame\(/);
assert.match(incidentCard, /incident-workspace-chrome/);
assert.match(incidentCard, /active=\{findSimilarActive\}/);
assert.match(incidentCard, /RelatedAppearanceIncidents active=\{open\}/);
assert.match(incidentCard, /export function IncidentClipLayer\(/);
assert.match(incidentCard, /export function RelatedAppearanceIncidents\(/);
assert.match(incidentCard, /export function CrossCameraTracePanel\(/);
assert.doesNotMatch(incidentCard, /export function IncidentListItem\(/);
assert.match(incidentCard, /prefersNativeMobilePlayback\(\)/);
assert.match(incidentCard, /url: info\.downloadUrl, mimeType: "video\/mp4"/);
assert.match(incidentCard, /playback\.mimeType === "video\/mp4" \? <video/);

assert.match(evidence, /export function IncidentListItem\(/);
assert.match(evidence, /export function SnapshotImage\(/);
assert.match(evidence, /export function EventOverlay\(/);
assert.doesNotMatch(evidence, /export function IncidentCard\(/);
assert.doesNotMatch(evidence, /export function IncidentInspector\(/);
assert.match(evidence, /loadIncidentClipInfo\(viewerEvent, \(\) => cancelled, prefersNativeMobilePlayback\(\)\)/);
assert.match(evidence, /playback\.mimeType === "video\/mp4" \? <video/);

assert.match(incidentsPage, /import \{ IncidentCard, IncidentInspector \} from "\.\/IncidentCard\.jsx"/);
assert.match(livePage, /import \{ IncidentListItem, EventOverlay \} from "\.\.\/shared\/evidence\.jsx"/);

console.log("incident card module tests passed");
