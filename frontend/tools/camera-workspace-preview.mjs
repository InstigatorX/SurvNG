// Isolated UI preview: synthetic cameras/media and in-memory export jobs only.
// Run: node tools/camera-workspace-preview.mjs (requires FFmpeg on PATH).
import { createServer } from "vite";
import react from "@vitejs/plugin-react";
import { execFileSync } from "node:child_process";
import { readFileSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, basename } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const temporary = mkdtempSync(join(tmpdir(), "survng-workspace-preview-"));
const start = Math.floor(Date.now() / 1000) - 300;
const duration = 600;
const baseNames = ["Driveway", "Front door", "Side gate", "Garden", "Garage", "Workshop"];
const cameraCount = Math.max(1, Math.min(64, Math.floor(Number(process.env.PREVIEW_CAMERA_COUNT) || 6)));
const primaryAspect = Math.max(0.25, Math.min(4, Number(process.env.PREVIEW_PRIMARY_ASPECT) || 16 / 9));
const names = Array.from({ length: cameraCount }, (_, index) => baseNames[index] || `Camera ${index + 1}`);
const ids = names.map((name) => name.toLowerCase().replaceAll(" ", "-"));
const cameraWidth = (index) => index === 0 ? Math.round(720 * primaryAspect) : 1280;
const jobs = new Map();
const clients = new Set();
let scenario = "healthy";
let server;

execFileSync(process.env.FFMPEG_PATH || "ffmpeg", [
  "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
  `testsrc2=s=640x360:r=12:d=${duration}`, "-c:v", "libx264", "-preset", "ultrafast",
  "-crf", "36", "-pix_fmt", "yuv420p", "-g", "120", "-an", "-movflags", "+faststart",
  join(temporary, "sample.mp4"),
]);
execFileSync(process.env.FFMPEG_PATH || "ffmpeg", [
  "-hide_banner", "-loglevel", "error", "-i", join(temporary, "sample.mp4"),
  "-c", "copy", "-hls_time", "10", "-hls_playlist_type", "vod",
  "-hls_segment_filename", join(temporary, "part%03d.ts"), join(temporary, "sample.m3u8"),
]);

function cameras() {
  return ids.map((id, index) => ({
    id, name: names[index], running: true, connected: true, capture_connectivity: "healthy",
    expected_enabled: true, recording_configured: true, recording_enabled: true,
    recording: !(scenario === "recording-failure" && index === 2), sub_recording: true,
    record_sub_enabled: true, detection_enabled: false, last_motion_at: null,
    width: cameraWidth(index), height: 720,
  }));
}

function system() {
  const freePercent = scenario === "low-storage" ? 3 : 38;
  const total = 2 * 1024 ** 4;
  return {
    instance_id: "workspace-preview-v1", lifecycle: "running", uptime_seconds: 86400,
    resources: { application_memory_bytes: 420 * 1024 ** 2, cpu_load_percent: 8.2 },
    storage: { available: true, total_bytes: total, free_bytes: total * freePercent / 100,
      used_bytes: total * (100 - freePercent) / 100, used_percent: 100 - freePercent,
      sampled_at: new Date().toISOString() },
    detector: { enabled: false, loaded_backend: "preview" },
    cameras: { total: cameraCount, enabled: cameraCount, online: cameraCount, recording_expected: cameraCount,
      recording: scenario === "recording-failure" && cameraCount > 2 ? cameraCount - 1 : cameraCount },
  };
}

function emit() {
  if (scenario === "unavailable") return;
  for (const response of clients) {
    response.write(`event: cameras_state\ndata: ${JSON.stringify(cameras())}\n\n`);
    response.write(`event: system_state\ndata: ${JSON.stringify(system())}\n\n`);
  }
}

function snapshot(id) {
  const index = Math.max(0, ids.indexOf(id));
  const skies = ["#789aa9", "#a9b7bc", "#859a91", "#6f9986", "#8f9299", "#7f8b9a"];
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${cameraWidth(index)}" height="720" viewBox="0 0 1280 720" preserveAspectRatio="xMidYMid slice"><rect width="1280" height="720" fill="${skies[index % skies.length]}"/><path d="M0 310 170 160 420 330 800 120 1280 340V720H0Z" fill="#53695d"/><path d="M0 460H1280V720H0Z" fill="#4a5054"/><path d="M500 400H780L1100 720H160Z" fill="#929391"/><path d="M30 350 260 210 480 350V570H30Z" fill="#d9d4c7"/><rect x="80" y="380" width="300" height="170" fill="#777f80"/><path d="M890 360 1080 230 1250 360V540H890Z" fill="#b1b7ab"/><rect x="940" y="390" width="210" height="140" fill="#596663"/><path d="M400 600h420" stroke="#c6cbcc" stroke-width="6"/><text x="32" y="670" fill="white" font-family="sans-serif" font-size="32">${names[index]} · synthetic preview</text></svg>`;
}

function json(res, value, status = 200) {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify(value));
}

function media(req, res, name) {
  // Only generated filenames can be served; never accept an arbitrary path.
  if (!/^(sample\.mp4|part\d+\.ts)$/.test(name)) return json(res, {}, 404);
  const bytes = readFileSync(join(temporary, basename(name)));
  res.setHeader("Content-Type", name.endsWith(".ts") ? "video/mp2t" : "video/mp4");
  res.setHeader("Accept-Ranges", "bytes");
  const range = /^bytes=(\d+)-(\d*)$/.exec(req.headers.range || "");
  if (range) {
    const from = Number(range[1]);
    const to = Math.min(bytes.length - 1, range[2] ? Number(range[2]) : bytes.length - 1);
    if (from > to) { res.statusCode = 416; res.end(); return; }
    res.statusCode = 206;
    res.setHeader("Content-Range", `bytes ${from}-${to}/${bytes.length}`);
    res.setHeader("Content-Length", to - from + 1);
    res.end(bytes.subarray(from, to + 1));
  } else { res.setHeader("Content-Length", bytes.length); res.end(bytes); }
}

server = await createServer({
  root, configFile: false, base: "/", cacheDir: join(temporary, "vite-cache"),
  server: { host: "127.0.0.1", port: Number(process.env.PREVIEW_PORT) || 5182, strictPort: true,
    watch: { usePolling: true, interval: 500 } },
  plugins: [react(), { name: "camera-workspace-preview", configureServer(vite) {
    vite.middlewares.use(async (req, res, next) => {
      const url = new URL(req.url, "http://127.0.0.1");
      const path = url.pathname;
      if (path === "/static/favicon.svg") { res.setHeader("Content-Type", "image/svg+xml"); res.end(readFileSync(join(root, "public/favicon.svg"))); return; }
      if (path.startsWith("/__preview-media/")) return media(req, res, path.split("/").at(-1));
      if (path === "/__preview/phone") {
        const target = url.searchParams.get("view") === "timeline" ? `/timeline?camera=driveway&at=${start + 300}` : "/";
        res.setHeader("Content-Type", "text/html; charset=utf-8");
        res.end(`<!doctype html><html><head><title>SurvNG phone preview</title></head><body style="margin:20px;background:#172026;color:#eef3f5;font-family:system-ui"><p>Phone layout · 390 × 844 <a style="color:#8de3d5" href="/__preview">Preview controls</a></p><iframe title="Phone workspace" src="${target}" width="390" height="844" style="border:1px solid #71838b;border-radius:12px"></iframe></body></html>`);
        return;
      }
      if (path === "/__preview") {
        res.setHeader("Content-Type", "text/html; charset=utf-8");
        res.end(`<!doctype html><html><head><title>SurvNG workspace experiment</title><style>body{max-width:740px;margin:64px auto;padding:0 24px;background:#10171b;color:#edf3f5;font:16px/1.6 system-ui}a{color:#8de3d5}button{background:#25343d;border:1px solid #657980;border-radius:6px;color:inherit;padding:10px 16px;cursor:pointer}form{display:inline-block;margin:0 8px 8px 0}</style></head><body><h1>Camera workspace preview</h1><p>Synthetic cameras and footage. No connection to a running SurvNG service.</p><p><a href="/">Live</a> · <a href="/timeline?camera=driveway&at=${start + 300}">Timeline</a></p><p><a href="/__preview/phone">Phone Live</a> · <a href="/__preview/phone?view=timeline">Phone Timeline</a></p><p>Health scenarios (apply to all preview tabs):</p>${["healthy", "recording-failure", "low-storage", "unavailable"].map((name) => `<form method="post" action="/__preview/scenario?name=${name}"><button>${name}</button></form>`).join("")}<p>Active scenario: <strong>${scenario}</strong>. Reload the app after choosing unavailable to check its initial error state.</p></body></html>`);
        return;
      }
      if (path === "/__preview/scenario" && req.method === "POST") {
        const requested = url.searchParams.get("name");
        if (!["healthy", "recording-failure", "low-storage", "unavailable"].includes(requested)) return json(res, {}, 400);
        scenario = requested; emit();
        res.writeHead(303, { Location: "/__preview" }); res.end(); return;
      }
      if (!path.startsWith("/api/")) return next();
      if (path === "/api/events/stream") {
        res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" });
        clients.add(res); res.write("event: connected\ndata: {}\n\n"); emit();
        req.on("close", () => clients.delete(res)); return;
      }
      if (path === "/api/auth/session") return json(res, { enabled: false, user: null });
      if (scenario === "unavailable" && ["/api/cameras", "/api/system/status"].includes(path)) return json(res, { detail: "Preview status unavailable" }, 503);
      if (path === "/api/cameras") return json(res, cameras());
      if (path === "/api/config") return json(res, { cameras: ids.map((id, index) => ({ id, name: names[index], record: true, record_sub: true })), retention: { minimum_free_percent: 15, emergency_free_percent: 5 }, detector: { tracking: {} } });
      if (path === "/api/system/status") return json(res, system());
      if (path === "/api/incidents/feed") return json(res, { items: [], facets: { camera_ids: [], labels: [], zones: [] }, has_more: false });
      if (path === "/api/semantic-search/status") return json(res, { enabled: false, state: "disabled" });
      if (/\/(snapshot|preview)\.jpg$/.test(path)) {
        res.setHeader("Content-Type", "image/svg+xml"); res.end(snapshot(path.split("/")[3])); return;
      }
      if (path.endsWith("/day.m3u8")) {
        res.setHeader("Content-Type", "application/vnd.apple.mpegurl");
        res.end(readFileSync(join(temporary, "sample.m3u8"), "utf8").replace(/^(part\d+\.ts)$/gm, "/__preview-media/$1")); return;
      }
      if (path.endsWith(".mp4")) return media(req, res, "sample.mp4");
      const rows = [{ start_epoch: start, end_epoch: start + duration, duration, source: url.searchParams.get("source") || "main", camera_id: path.split("/")[3] }];
      if (path.endsWith("/recordings/day")) return json(res, { availability: rows, events: [], available_sources: ["main", "live"] });
      if (path.endsWith("/recordings/window")) return json(res, { start_epoch: start, end_epoch: start + duration, recordings: rows });
      if (path.endsWith("/recordings/updates")) return json(res, { availability: [], events: [] });
      if (path === "/api/exports" && req.method === "POST") {
        let body = ""; for await (const chunk of req) body += chunk;
        const payload = JSON.parse(body);
        const job = { ...payload, id: `preview-${jobs.size + 1}`, status: "queued", created_at: new Date().toISOString() };
        jobs.set(job.id, job);
        setTimeout(() => jobs.set(job.id, { ...job, status: "completed", download_url: "/__preview-media/sample.mp4" }), 1500);
        return json(res, job);
      }
      if (path === "/api/exports") return json(res, [...jobs.values()]);
      if (path.startsWith("/api/exports/")) return json(res, jobs.get(path.split("/").at(-1)) || {});
      // Test mutations cannot reach real camera controls.
      return json(res, { detail: "This operation is not part of the isolated preview" }, 404);
    });
  } }],
});
await server.listen();
const timer = setInterval(emit, 15_000);
console.log(`Synthetic workspace preview: ${server.resolvedUrls.local[0]}__preview`);
console.log(`Timeline: ${server.resolvedUrls.local[0]}timeline?camera=driveway&at=${start + 300}`);
async function stop() { clearInterval(timer); for (const res of clients) res.end(); await server.close(); rmSync(temporary, { recursive: true, force: true }); process.exit(0); }
process.on("SIGINT", stop); process.on("SIGTERM", stop);
