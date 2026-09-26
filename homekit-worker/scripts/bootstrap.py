#!/usr/bin/env python3
"""Build the audited HAP revision before installing the locked worker packages."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "d81fba565ee26e82170d5f4f8cd358c2fd773f6c"
SHA256 = "102b6e61900d4e7dcf85e6be3ba5970f9d214ca6829855c5f8981f95373f2efe"


def run(*args: str, cwd: Path = ROOT) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def main() -> None:
    major = subprocess.check_output(["node", "-p", "process.versions.node.split('.')[0]"], text=True).strip()
    if major != "24":
        raise SystemExit("Node.js 24 is required; use npm exec --package=node@24 -- python3 scripts/bootstrap.py")
    cache = ROOT / ".cache"
    cache.mkdir(exist_ok=True)
    with urllib.request.urlopen(f"https://codeload.github.com/seydx/HAP-NodeJS/tar.gz/{COMMIT}", timeout=60) as response:
        archive = response.read()
    if hashlib.sha256(archive).hexdigest() != SHA256:
        raise SystemExit("HAP source checksum mismatch")
    source = cache / f"HAP-NodeJS-{COMMIT}"
    if source.exists():
        shutil.rmtree(source)
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        bundle.extractall(cache, filter="data")
    # Upstream's own lock pins build tooling; no lifecycle scripts execute on install.
    run("npm", "ci", "--ignore-scripts", "--no-audit", "--no-fund", cwd=source)
    run("npm", "run", "build", cwd=source)
    packed = json.loads(subprocess.check_output(
        ["npm", "pack", "--ignore-scripts", "--json", "--pack-destination", str(cache)], cwd=source, text=True,
    ))[0]["filename"]
    (cache / packed).replace(cache / "hap-nodejs.tgz")
    run("npm", "ci" if (ROOT / "package-lock.json").exists() else "install", "--ignore-scripts", "--no-audit", "--no-fund")
    run("npm", "run", "build")


if __name__ == "__main__":
    main()
