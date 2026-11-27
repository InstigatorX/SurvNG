# HTTP API architecture

SurvNG treats `survng.app.main` as the application composition root. It owns
process startup and shutdown, creates long-lived managers, installs middleware,
and assembles focused API routers. New feature routes should not be added to
`main.py` unless they are part of process lifecycle itself.

## Focused boundaries

- `config_routes.py` owns configuration and camera-administration HTTP routes.
- `recording_routes.py` owns recording history, synchronized grid history,
  previews, exports, HLS playback, and event-clip HTTP routes.
- `incident_presenter.py` owns pure event-to-incident presentation transforms
  shared by incident and recording views.

The recording router receives explicit dependencies rather than importing the
application globals. Dependency callables resolve the current manager generation
at request time, which keeps configuration reloads visible without reconstructing
the FastAPI application. A handler snapshots that manager once and uses the same
generation throughout the request so a concurrent reload cannot combine an old
event store with a new recorder.

`main.py` does not re-export route handlers. Tests that call a handler directly
take it from the router bundle, for example
`main._recording_route_bundle.handlers["create_media_export"]`.

## Adding an API area

1. Put validation models and route handlers in a module named for the feature.
2. Define a small immutable dependency bundle containing only required services.
3. Snapshot reloadable services once at the start of each request.
4. Keep reusable business logic in application services or pure modules, not in
   route handlers.
5. Assemble the router in `main.py` and add focused route and lifecycle tests.

This boundary is intentionally incremental. Recording remux caching, prewarming,
and event-clip construction still live behind injected callbacks; they can move
into a dedicated media service without changing the public routes.
