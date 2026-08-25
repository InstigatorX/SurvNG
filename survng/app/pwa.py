"""Progressive Web App manifest and service-worker helpers."""

from __future__ import annotations

import json
from typing import Any


def web_app_manifest(base_path: str) -> dict[str, Any]:
    """Build an installable web app manifest for the configured base path."""
    prefix = str(base_path or "").rstrip("/")
    start_url = f"{prefix}/" if prefix else "/"
    icons_base = f"{prefix}/static" if prefix else "/static"
    return {
        "id": start_url,
        "name": "SurvNG",
        "short_name": "SurvNG",
        "description": "Live camera monitoring for SurvNG. Requires a network connection.",
        "start_url": start_url,
        "scope": start_url,
        "display": "standalone",
        "orientation": "any",
        "background_color": "#071015",
        "theme_color": "#071015",
        "lang": "en",
        "icons": [
            {
                "src": f"{icons_base}/pwa-icon-192.png",
                "sizes": "192x192",
                "type": "image/png",
                "purpose": "any",
            },
            {
                "src": f"{icons_base}/pwa-icon-512.png",
                "sizes": "512x512",
                "type": "image/png",
                "purpose": "any",
            },
            {
                "src": f"{icons_base}/pwa-icon-maskable-192.png",
                "sizes": "192x192",
                "type": "image/png",
                "purpose": "maskable",
            },
            {
                "src": f"{icons_base}/pwa-icon-maskable-512.png",
                "sizes": "512x512",
                "type": "image/png",
                "purpose": "maskable",
            },
        ],
    }


def normalize_service_worker_cache_version(cache_version: str | None = None) -> str:
    """Return a filesystem/HTTP-safe cache token for the installable shell."""
    raw = str(cache_version or "").strip().lower()
    if not raw:
        # Fallback when no baked git SHA is available (native/dev without SURVNG_GIT_SHA).
        # Keep in sync with the last fixed bucket bump on v1.1 before SHA versioning.
        return "v2"
    cleaned_chars: list[str] = []
    for ch in raw:
        cleaned_chars.append(ch if ch.isalnum() else "-")
    cleaned = "".join(cleaned_chars).strip("-")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned[:48] if cleaned else "v2"


def service_worker_script(base_path: str, cache_version: str | None = None) -> str:
    """Return a service worker that only caches hashed static assets.

    The cache bucket is versioned (preferably with the baked git SHA) so a new
    Docker/native image drops prior asset caches on activate. Hashed filenames
    stay cache-first; HTML/API/live media are never claimed.
    """
    prefix = str(base_path or "").rstrip("/")
    version = normalize_service_worker_cache_version(cache_version)
    cache_name = f"survng-static-{version}"
    static_prefix = f"{prefix}/static/assets/" if prefix else "/static/assets/"
    return f"""/* SurvNG installable shell — online only; never cache API or live media. */
const STATIC_CACHE = {json.dumps(cache_name)};
const STATIC_ASSET_PREFIX = {json.dumps(static_prefix)};

self.addEventListener("install", (event) => {{
  event.waitUntil(self.skipWaiting());
}});

self.addEventListener("activate", (event) => {{
  event.waitUntil((async () => {{
    const keys = await caches.keys();
    await Promise.all(
      keys
        .filter((key) => key.startsWith("survng-static-") && key !== STATIC_CACHE)
        .map((key) => caches.delete(key)),
    );
    await self.clients.claim();
  }})());
}});

function isHashedStaticAsset(url) {{
  return url.origin === self.location.origin && url.pathname.startsWith(STATIC_ASSET_PREFIX);
}}

function mustBypassCache(request, url) {{
  if (request.method !== "GET") return true;
  if (request.headers.get("upgrade") === "websocket") return true;
  const path = url.pathname;
  if (path.includes("/api/")) return true;
  if (path.endsWith("/sw.js") || path.endsWith("/manifest.webmanifest")) return true;
  return false;
}}

self.addEventListener("fetch", (event) => {{
  const request = event.request;
  let url;
  try {{
    url = new URL(request.url);
  }} catch {{
    return;
  }}
  if (mustBypassCache(request, url) || !isHashedStaticAsset(url)) return;
  event.respondWith((async () => {{
    const cache = await caches.open(STATIC_CACHE);
    const cached = await cache.match(request);
    if (cached) return cached;
    const response = await fetch(request);
    if (response.ok) {{
      cache.put(request, response.clone());
    }}
    return response;
  }})());
}});
"""


def service_worker_allowed_scope(base_path: str) -> str:
    prefix = str(base_path or "").rstrip("/")
    return f"{prefix}/" if prefix else "/"
