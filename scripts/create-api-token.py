#!/usr/bin/env -S .venv/bin/python
"""Create a scoped SurvNG API token while persisting only its digest."""

from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from survng.app.config import ApiTokenConfig, config_path, load_config, save_config  # noqa: E402
from survng.app.security import hash_api_token  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a long-lived SurvNG bearer token and print it once."
    )
    parser.add_argument("--id", required=True, help="Stable token identifier")
    parser.add_argument("--name", required=True, help="Human-readable token name")
    parser.add_argument(
        "--scope",
        action="append",
        choices=("read", "camera:control", "admin"),
        default=[],
        help="Granted scope; repeat as needed (default: read)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Configuration file (defaults to SURVNG_CONFIG_PATH or config.json)",
    )
    parser.add_argument(
        "--enable",
        action="store_true",
        help="Enable API authentication immediately after adding the token",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = args.config or config_path()
    config = load_config(path)
    if any(item.id == args.id for item in config.api_auth.tokens):
        print(f"API token id already exists: {args.id}", file=sys.stderr)
        return 2
    token = f"survng_{secrets.token_urlsafe(32)}"
    config.api_auth.tokens.append(
        ApiTokenConfig(
            id=args.id,
            name=args.name,
            token_hash=hash_api_token(token),
            scopes=args.scope or ["read"],
        )
    )
    if args.enable:
        config.api_auth.enabled = True
    save_config(config, path, assign_ids=False)
    print("SurvNG API token created. Save it now; only its digest was persisted.")
    print(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
