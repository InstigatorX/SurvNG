// Event Rail, edge rail, camera lanes and catch-up views on the real
// RecordingsPage, with mocked recording APIs and generated media.
//   node tests/timeline-rail.mjs                 assertions
//   TIMELINE_RAIL_SHOTS=/dir node tests/timeline-rail.mjs   also save screenshots
//   TIMELINE_RAIL_SERVE=1 node tests/timeline-rail.mjs      keep the fixture running
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createReadStream, existsSync, mkdirSync, mkdtempSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";
import { chromium } from "playwright";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const temporary = mkdtempSync(join(tmpdir(), "timeline-rail-"));
const media = join(temporary, "media");
const shots = process.env.TIMELINE_RAIL_SHOTS || "";
const date = "2026-09-09";
const dayStart = Date.parse(`${date}T00:00:00Z`) / 1000;
const at = (hours) => dayStart + Math.round(hours * 3600);
const cameras = [["gate", "Gate"], ["front-door", "Front Door"], ["upper-garage", "Upper Garage"]]
  .map(([id, name]) => ({ id, name, enabled: true, recording: true, sub_recording: true, online: true }));
let nextIncident = 1;
const incident = (camera_id, hours, seconds, labels, identity = null) => {
  const id = nextIncident++;
  return {
    ...(identity ? { primary_identity: { id: identity } } : {}),
    id, representative_event_id: id * 10, camera_id,
    start_epoch: at(hours), last_epoch: at(hours) + seconds,
    has_objects: labels.length > 0, labels,
    snapshot_url: "/timeline-rail-media/preview.jpg",
  };
};
const incidents = [
  incident("gate", 6.4, 25, ["person"]), incident("gate", 8.1, 12, []), incident("gate", 9.7, 41, ["car"]),
  incident("gate", 11.2, 30, ["person"]), incident("gate", 12.68, 12, []), incident("gate", 13.82, 35, ["person"]),
  incident("gate", 14.13, 41, ["car"]), incident("gate", 14.53, 18, ["person"], 17), incident("gate", 14.71, 9, []),
  incident("gate", 14.95, 64, ["truck"]), incident("gate", 15.1, 22, ["person"]),
  incident("front-door", 10, 10, []), incident("front-door", 14.54, 14, ["person"], 17), incident("front-door", 15.1, 20, ["person"]),
  incident("upper-garage", 9, 8, []), incident("upper-garage", 14.95, 50, ["car"]),
];
const availability = (camera_id = undefined) => [
  { start_epoch: at(0), end_epoch: at(13.97), ...(camera_id ? { camera_id, source: "main" } : {}) },
  { start_epoch: at(14.03), end_epoch: at(15.4), ...(camera_id ? { camera_id, source: "main" } : {}) },
];
let server;
let browser;
function json(res, payload) { res.setHeader("Content-Type", "application/json"); res.setHeader("Cache-Control", "no-store"); res.end(JSON.stringify(payload)); }
function serveFile(req, res, file) {
  const size = statSync(file).size;
  let start = 0; let end = size - 1;
  res.setHeader("Content-Type", ({ ".ts": "video/mp2t", ".mp4": "video/mp4", ".jpg": "image/jpeg" })[extname(file)] || "application/octet-stream");
  res.setHeader("Accept-Ranges", "bytes");
  if (req.headers.range) {
    const match = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range);
    if (match && (match[1] || match[2])) {
      start = match[1] ? Number(match[1]) : Math.max(0, size - Number(match[2]));
      if (match[1] && match[2]) end = Math.min(end, Number(match[2]));
      res.statusCode = 206; res.setHeader("Content-Range", `bytes ${start}-${end}/${size}`);
    }
  }
  res.setHeader("Content-Length", end - start + 1);
  if (req.method === "HEAD") res.end();
  else createReadStream(file, { start, end }).on("error", () => res.destroy()).pipe(res);
}
function rows(start, end) {
  return Array.from({ length: Math.ceil((end - start) / 10) }, (_, i) => ({
    start_epoch: start + i * 10, end_epoch: Math.min(end, start + (i + 1) * 10),
    duration_seconds: 10, media_start: i * 10, media_end: (i + 1) * 10,
    path: `/fixture/${start + i * 10}.mp4`, name: `${start + i * 10}.mp4`, size_bytes: 10000,
  }));
}
function fixturePlugin() {
  return { name: "timeline-rail-fixture", configureServer(vite) {
    vite.middlewares.use((req, res, next) => {
      const url = new URL(req.url, "http://localhost");
      const path = url.pathname;
      if (path === "/" || path === "/timeline") {
        res.setHeader("Content-Type", "text/html");
        res.end('<!doctype html><html data-theme="dark"><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Timeline rail fixture</title></head><body><div id="root"></div><script type="module" src="/tests/fixtures/timeline-rail.jsx"></script></body></html>');
        return;
      }
      if (path === "/api/events/stream") { res.writeHead(200, { "Content-Type": "text/event-stream" }); res.write(": ok\n\n"); return; }
      if (path === "/api/cameras") { json(res, cameras); return; }
      if (path === "/api/config") { json(res, { cameras, detector: { tracking: { camera_transition_routes: [{ from_camera: "gate", to_camera: "front-door", bidirectional: true }] } } }); return; }
      if (path === "/api/semantic-search/status" || path === "/api/speech/status") { json(res, { enabled: false, ready: false }); return; }
      if (path === "/api/exports") { json(res, { exports: [] }); return; }
      if (path === "/api/system/status") { json(res, {}); return; }
      if (path === "/api/recordings/grid/day") {
        // Sub coverage has its own gap across 2:31 PM, so lanes showing the wrong source are visible.
        const sub = cameras.flatMap((camera) => [[0, 13.97], [14.03, 14.5], [14.56, 15.4]].map(([from, to]) => ({ camera_id: camera.id, source: "live", start_epoch: at(from), end_epoch: at(to) })));
        json(res, { available_sources: ["main", "live"], availability: [...cameras.flatMap((camera) => availability(camera.id)), ...sub], incidents });
        return;
      }
      if (path === "/api/recordings/grid/updates") { json(res, { availability: [], incidents: [] }); return; }
      if (/^\/api\/events\/\d+\/thumbnail\.jpg$/.test(path)) { serveFile(req, res, join(media, "preview.jpg")); return; }
      const recording = /^\/api\/cameras\/([^/]+)\/recordings\/(.+)$/.exec(path);
      if (recording) {
        const [, cameraId, kind] = recording;
        const start = Number(url.searchParams.get("start_epoch"));
        const end = Number(url.searchParams.get("end_epoch"));
        // Headless Chromium cannot play the Sub fixture, so a Sub page advertises Sub only and stays on it.
        if (kind === "day") { json(res, { available_sources: url.searchParams.get("source") === "live" ? ["live"] : ["main", "live"], availability: availability(), incidents: incidents.filter((item) => item.camera_id === cameraId) }); return; }
        if (kind === "window") { json(res, { start_epoch: start, end_epoch: end, recordings: rows(start, end) }); return; }
        if (kind === "updates") { json(res, { availability: [], incidents: [] }); return; }
        if (kind === "day.m3u8") {
          const lines = ["#EXTM3U", "#EXT-X-VERSION:3", "#EXT-X-TARGETDURATION:10", "#EXT-X-MEDIA-SEQUENCE:0", "#EXT-X-PLAYLIST-TYPE:VOD"];
          // Every part is the same 10 s file, so each restarts its timestamps.
          for (let i = 0; i < Math.ceil((end - start) / 10); i++) { if (i) lines.push("#EXT-X-DISCONTINUITY"); lines.push("#EXTINF:10.000,", `/timeline-rail-media/0.ts?part=${i}`); }
          lines.push("#EXT-X-ENDLIST"); res.setHeader("Content-Type", "application/vnd.apple.mpegurl"); res.end(lines.join("\n") + "\n"); return;
        }
        if (kind === "segment.mp4") { serveFile(req, res, join(media, "segment.mp4")); return; }
        if (kind === "preview.jpg") { serveFile(req, res, join(media, "preview.jpg")); return; }
      }
      if (path === "/timeline-rail-media/0.ts") { serveFile(req, res, join(media, "0.ts")); return; }
      if (path === "/timeline-rail-media/preview.jpg") { serveFile(req, res, join(media, "preview.jpg")); return; }
      if (path.startsWith("/api/")) { res.writeHead(404); res.end(`Unmocked fixture API: ${path}`); return; }
      next();
    });
  } };
}
const ffmpeg = process.env.FFMPEG_PATH || "ffmpeg";
async function open(viewport, query, mobile = false, options = {}) {
  const page = await browser.newPage({ ...(mobile ? { viewport, hasTouch: true, isMobile: true, deviceScaleFactor: 2 } : { viewport, deviceScaleFactor: 1 }), ...options });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(`${base}timeline?${query}`);
  page.errors = errors;
  return page;
}
async function shot(page, name) {
  if (!shots) return;
  await page.waitForTimeout(600);
  await page.screenshot({ path: join(shots, `${name}.png`) });
}
let base = "";
try {
  mkdirSync(media);
  execFileSync(ffmpeg, ["-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=12:d=10", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-g", "12", "-f", "mpegts", join(media, "0.ts")]);
  execFileSync(ffmpeg, ["-hide_banner", "-loglevel", "error", "-i", join(media, "0.ts"), "-c", "copy", "-movflags", "+faststart", join(media, "segment.mp4")]);
  execFileSync(ffmpeg, ["-hide_banner", "-loglevel", "error", "-i", join(media, "0.ts"), "-frames:v", "1", join(media, "preview.jpg")]);
  if (shots) mkdirSync(shots, { recursive: true });
  server = await createServer({ root, configFile: false, logLevel: "error", cacheDir: join(temporary, "vite"), server: { host: "127.0.0.1", port: Number(process.env.TIMELINE_RAIL_PORT) || 0 }, plugins: [fixturePlugin()] });
  await server.listen();
  base = server.resolvedUrls.local[0];
  const query = `camera=gate&date=${date}&at=${at(14.53)}&source=main`;
  console.log(`Timeline rail fixture: ${base}timeline?${query}`);
  if (process.env.TIMELINE_RAIL_SERVE === "1") {
    await new Promise((done) => { process.once("SIGINT", done); process.once("SIGTERM", done); });
  } else {
    const macChrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
    browser = await chromium.launch({ executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH || (existsSync(macChrome) ? macChrome : undefined), headless: true });
    const { checks } = await import("./fixtures/timeline-rail-checks.mjs");
    await checks({ open, shot, query, at, date, assert });
    console.log("Timeline rail checks passed");
  }
} finally {
  await browser?.close();
  await server?.close();
  rmSync(temporary, { recursive: true, force: true });
}
