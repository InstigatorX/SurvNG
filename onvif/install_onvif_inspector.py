#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OVERLAY = ROOT / "overlay"

PATCHES = {
    "survng/app/onvif_events.py": [
        (
            "from .security import redact_secret_text\n",
            "from .security import redact_secret_text\n"
            "from .onvif_inspector import ONVIF_INSPECTOR\n",
        ),
        (
            "        self._subscription_expires_monotonic: float | None = None\n",
            "        self._subscription_expires_monotonic: float | None = None\n"
            "        ONVIF_INSPECTOR.register_listener(self.camera.id, self)\n",
        ),
        (
            "                        if motion_state is True:\n",
            "                        try:\n"
            "                            ONVIF_INSPECTOR.record(\n"
            "                                camera_id=self.camera.id,\n"
            "                                topic=topic,\n"
            "                                normalized_topic=(\n"
            "                                    raw_notification.normalized_topic\n"
            "                                    if raw_notification is not None\n"
            "                                    else self._normalized_topic(topic)\n"
            "                                ),\n"
            "                                active=motion_state,\n"
            "                                simple_items=(\n"
            "                                    raw_notification.simple_items\n"
            "                                    if raw_notification is not None\n"
            "                                    else ()\n"
            "                                ),\n"
            "                                message_xml=(\n"
            "                                    raw_notification.message_xml\n"
            "                                    if raw_notification is not None\n"
            "                                    else message\n"
            "                                ),\n"
            "                                received_at=received_at,\n"
            "                                event_at=event_at,\n"
            "                            )\n"
            "                        except Exception:\n"
            "                            LOGGER.debug(\n"
            "                                \"ONVIF inspector observer failed for %s\",\n"
            "                                self.camera.id,\n"
            "                                exc_info=True,\n"
            "                            )\n"
            "                        if motion_state is True:\n",
        ),
    ],
    "survng/app/frontend_routes.py": [
        (
            '        "live_page": ("/live", "index.html"),\n',
            '        "live_page": ("/live", "index.html"),\n'
            '        "onvif_page": ("/onvif", "onvif.html"),\n',
        ),
    ],
    "survng/app/main.py": [
        (
            "from .frontend_routes import FrontendRouteDependencies, create_frontend_router\n",
            "from .frontend_routes import FrontendRouteDependencies, create_frontend_router\n"
            "from .onvif_inspector_routes import create_onvif_inspector_router\n",
        ),
        (
            "app.include_router(_frontend_route_bundle.router)\n",
            "app.include_router(_frontend_route_bundle.router)\n"
            "app.include_router(create_onvif_inspector_router())\n",
        ),
        (
            'live_page = _frontend_route_bundle.handlers["live_page"]\n',
            'live_page = _frontend_route_bundle.handlers["live_page"]\n'
            'onvif_page = _frontend_route_bundle.handlers["onvif_page"]\n',
        ),
    ],
    "frontend/vite.config.js": [
        (
            '        config: resolve(__dirname, "config.html"),\n',
            '        config: resolve(__dirname, "config.html"),\n'
            '        onvif: resolve(__dirname, "onvif.html"),\n',
        ),
    ],
}


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def patch_file(repo: Path, relative: str, dry_run: bool) -> None:
    path = repo / relative
    if not path.exists():
        fail(f"missing expected file: {relative}")
    text = path.read_text()

    changed = False
    for old, new in PATCHES[relative]:
        if new in text:
            continue
        if old not in text:
            fail(
                f"expected v1.0 anchor not found in {relative}. "
                "Refusing to guess; integrate manually or regenerate against your branch."
            )
        text = text.replace(old, new, 1)
        changed = True

    if not changed:
        print(f"already patched: {relative}")
        return

    print(f"{'would patch' if dry_run else 'patching'}: {relative}")
    if not dry_run:
        backup = path.with_suffix(path.suffix + ".onvif-inspector.bak")
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(text)


def copy_overlay(repo: Path, dry_run: bool) -> None:
    for source in sorted(p for p in OVERLAY.rglob("*") if p.is_file()):
        relative = source.relative_to(OVERLAY)
        target = repo / relative
        print(f"{'would install' if dry_run else 'installing'}: {relative}")
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install the standalone SurvNG ONVIF Inspector overlay."
    )
    parser.add_argument(
        "--repo",
        default=".",
        help="SurvNG repository root (default: current directory)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate and show changes without writing",
    )
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    required = [
        repo / "survng/app/onvif_events.py",
        repo / "frontend/vite.config.js",
    ]
    if not all(path.exists() for path in required):
        fail(f"{repo} does not look like the SurvNG repository root")

    copy_overlay(repo, args.check)
    for relative in PATCHES:
        patch_file(repo, relative, args.check)

    print()
    if args.check:
        print("Check passed. No files were changed.")
    else:
        print("ONVIF Inspector installed.")
        print("Next:")
        print("  cd frontend && npm run build")
        print("  cd .. && pytest -q tests/test_onvif_inspector.py")
        print("  restart SurvNG")
        print("  open /onvif")


if __name__ == "__main__":
    main()
