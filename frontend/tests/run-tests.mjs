import { readFileSync, readdirSync } from "node:fs";
import { spawn, spawnSync } from "node:child_process";
import { createServer } from "node:net";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

// node tests/run-tests.mjs unit|browser [--offline]
// A file that imports Playwright is a browser test. A browser test may say
// what it needs on a line of its own:
//   // browser-test: preview-server [NAME=value ...]
//       started here for that file alone: tools/camera-workspace-preview.mjs
//       with those settings, its URL passed as LIVE_PREVIEW_URL
//   // browser-test: live-survng
//       a running SurvNG at 127.0.0.1:8088; skipped with --offline (CI)
const directory = dirname(fileURLToPath(import.meta.url));
const mode = process.argv[2] || "unit";
const offline = process.argv.includes("--offline");
if (!new Set(["unit", "browser"]).has(mode)) throw new Error(`Unknown test mode: ${mode}`);

const files = readdirSync(directory)
  .filter((name) => name.endsWith(".mjs") && name !== "run-tests.mjs")
  .map((name) => {
    const source = readFileSync(join(directory, name), "utf8");
    const need = /^\/\/ browser-test: (\S+)(.*)$/m.exec(source);
    return { name, browser: source.includes('from "playwright"'), need: need?.[1] || "", settings: need?.[2].trim().split(/\s+/).filter(Boolean) || [] };
  })
  .filter((file) => (mode === "browser" ? file.browser : !file.browser))
  .sort((a, b) => a.name.localeCompare(b.name));

function freePort() {
  return new Promise((resolve, reject) => {
    const probe = createServer().once("error", reject).listen(0, "127.0.0.1", () => {
      const { port } = probe.address();
      probe.close(() => resolve(port));
    });
  });
}

async function startPreview(settings) {
  const port = await freePort();
  const env = { ...process.env, PREVIEW_PORT: String(port) };
  for (const setting of settings) {
    const [key, ...value] = setting.split("=");
    env[key] = value.join("=");
  }
  const child = spawn(process.execPath, [join(directory, "../tools/camera-workspace-preview.mjs")], { env, stdio: ["ignore", "ignore", "inherit"] });
  const url = `http://127.0.0.1:${port}/`;
  const deadline = Date.now() + 120_000;
  while (Date.now() < deadline) {
    if (child.exitCode !== null) throw new Error(`Preview server exited (${child.exitCode})`);
    try {
      if ((await fetch(url)).status < 500) return { url, child };
    } catch { /* not listening yet */ }
    await new Promise((done) => setTimeout(done, 250));
  }
  child.kill("SIGTERM");
  throw new Error("Preview server did not start within 120 s");
}

async function stopPreview(child) {
  if (child.exitCode !== null) return;
  const exited = new Promise((done) => child.once("exit", done));
  child.kill("SIGTERM");
  await Promise.race([exited, new Promise((done) => setTimeout(done, 5000))]);
  if (child.exitCode === null) child.kill("SIGKILL");
}

let ran = 0;
const skipped = [];
for (const file of files) {
  if (file.need === "live-survng" && offline) {
    skipped.push(file.name);
    continue;
  }
  if (file.need && !new Set(["preview-server", "live-survng"]).has(file.need)) throw new Error(`${file.name}: unknown browser-test need "${file.need}"`);
  process.stdout.write(`\n[${mode}] ${file.name}\n`);
  const preview = file.need === "preview-server" ? await startPreview(file.settings) : null;
  let result;
  try {
    const env = preview ? { ...process.env, LIVE_PREVIEW_URL: preview.url } : process.env;
    result = spawnSync(process.execPath, [join(directory, file.name)], { stdio: "inherit", env });
  } finally {
    if (preview) await stopPreview(preview.child);
  }
  if (result.status !== 0) process.exit(result.status ?? 1);
  ran += 1;
}

console.log(`\n${ran} ${mode} frontend test files passed`);
if (skipped.length) console.log(`Skipped (need a running SurvNG): ${skipped.join(", ")}`);
