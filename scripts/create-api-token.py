#!/usr/bin/env -S .venv/bin/python
"""Create, list, or delete scoped SurvNG API tokens."""

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
        description="Create, list, or delete long-lived SurvNG bearer tokens."
    )
    parser.add_argument(
        "action",
        nargs="?",
        choices=("create", "list", "delete"),
        default="create",
        help="Operation to perform (default: create)",
    )
    parser.add_argument("--id", help="Stable token identifier")
    parser.add_argument("--name", help="Human-readable token name")
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
        help=(
            "Enable API authentication immediately after adding the token. "
            "WARNING: the browser UI also needs an injected Authorization header."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = args.config or config_path()
    config = load_config(path)
    if args.action == "list":
        print(f"API authentication: {'enabled' if config.api_auth.enabled else 'disabled'}")
        if not config.api_auth.tokens:
            print("No API tokens configured.")
            return 0
        for item in config.api_auth.tokens:
            print(f"{item.id}\t{item.name}\t{','.join(item.scopes)}")
        return 0
    if args.action == "delete":
        if not args.id:
            print("delete requires --id", file=sys.stderr)
            return 2
        retained = [item for item in config.api_auth.tokens if item.id != args.id]
        if len(retained) == len(config.api_auth.tokens):
            print(f"API token id not found: {args.id}", file=sys.stderr)
            return 2
        config.api_auth.tokens = retained
        if not retained:
            config.api_auth.enabled = False
        save_config(config, path, assign_ids=False)
        print(f"Deleted API token: {args.id}")
        if not retained:
            print("API authentication disabled because no tokens remain.")
        return 0
    if not args.id or not args.name:
        print("create requires --id and --name", file=sys.stderr)
        return 2
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
