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


def service_worker_script(base_path: str, cache_version: str = "v2") -> str:
    """Return a network-first service worker that only caches hashed static assets."""
    prefix = str(base_path or "").rstrip("/")
    cache_name = f"survng-static-{cache_version}"
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
