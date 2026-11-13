import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const dockerfile = readFileSync(join(root, "Dockerfile"), "utf8");
const entrypoint = readFileSync(join(root, "docker/entrypoint.sh"), "utf8");
const example = readFileSync(join(root, "docker/go2rtc.example.yaml"), "utf8");

assert.match(dockerfile, /GO2RTC_VERSION=1\.9\.15/);
assert.match(dockerfile, /go2rtc_linux_amd64/);
assert.match(dockerfile, /GO2RTC_SHA256=20b46c6910734cb05ecefb234066f36868ff19b42ba1418a7a32e13a3e035712/);
assert.match(dockerfile, /go2rtc\.example\.yaml/);

assert.match(entrypoint, /SURVNG_GO2RTC/);
assert.match(entrypoint, /go2rtc -config/);
assert.match(entrypoint, /Started go2rtc/);
assert.match(example, /listen: "127\.0\.0\.1:1984"/);
assert.match(example, /listen: "127\.0\.0\.1:8554"/);

console.log("docker go2rtc packaging tests passed");
