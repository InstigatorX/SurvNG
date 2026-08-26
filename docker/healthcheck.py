#!/usr/bin/env python3
"""Container liveness check that accepts HTTP or HTTPS on the listen port."""

from __future__ import annotations

import ssl
import sys
import urllib.request

URLS = (
    "http://127.0.0.1:8088/api/health",
    "https://127.0.0.1:8088/api/health",
)


def main() -> int:
    context = ssl._create_unverified_context()
    for url in URLS:
        try:
            urllib.request.urlopen(
                url,
                timeout=3,
                context=context if url.startswith("https") else None,
            )
            return 0
        except Exception:
            continue
    return 1


if __name__ == "__main__":
    sys.exit(main())
