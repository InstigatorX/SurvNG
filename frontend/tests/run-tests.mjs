import { readFileSync, readdirSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const directory = dirname(fileURLToPath(import.meta.url));
const mode = process.argv[2] || "unit";
if (!new Set(["unit", "browser"]).has(mode)) throw new Error(`Unknown test mode: ${mode}`);

const files = readdirSync(directory)
  .filter((name) => name.endsWith(".mjs") && name !== "run-tests.mjs")
  .filter((name) => {
    const browser = readFileSync(join(directory, name), "utf8").includes('from "playwright"');
    return mode === "browser" ? browser : !browser;
  })
  .sort();

for (const file of files) {
  process.stdout.write(`\n[${mode}] ${file}\n`);
  const result = spawnSync(process.execPath, [join(directory, file)], { stdio: "inherit" });
  if (result.status !== 0) process.exit(result.status ?? 1);
}

console.log(`\n${files.length} ${mode} frontend test files passed`);
