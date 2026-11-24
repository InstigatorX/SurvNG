# Current-only compatibility

This build intentionally stops supporting prior-generation configuration,
databases, recording layouts, and semantic index identities. It does not migrate
or reset existing data, and drops only tables and indexes that no build reads.
Do not use it as an automatic upgrade path for older persisted generations. Keep application and data backups together.

Only databases that SurvNG v1.2.0 or later has opened are supported: later
builds no longer carry the column additions v1.2.0 applied to older databases,
and there is no startup check for older ones. Column additions that fresh
databases still need, because the column is not in its table's CREATE
statement, remain. `secrets.json` must be the encrypted envelope.

Fresh installations initialize their configuration explicitly:

```sh
python -m survng.app.config_init --template config.example.json --config config.json
```

Use `--storage-dir PATH` to set both storage and the primary media location, and
`--ffmpeg-path PATH` to override FFmpeg. Initialization refuses existing config,
sidecar, or key files. Docker and native installation scripts invoke this only
for new installs. Normal loading requires the secrets sidecar and rejects inline
credentials. Back up `config.json`, `secrets.json`, and `secrets.key` together.

Supported motion modes are `camera`, `camera_rescue`, and `adaptive`, with
camera-level `inherit`. Retired motion settings, pipeline implementations,
tracking aliases, and rectangle property overlays are unsupported. Property
zones use polygon `points`; custom media locations must be explicit.

Recordings use `{camera}/{main|live}/date/hour`. Current recording-index mount
rebasing remains supported. Semantic indexes use the declared embedding contract.
Incident notifications use schema version 3 exclusively.

The face embedding profile is now named `openvino_raw`; `legacy_openvino` is
rejected. The rename preserves raw input behavior and fingerprints, so it alone
does not require regenerating embeddings. AdaFace and ArcFace profiles remain.

Tracking comparison UI, APIs, replay CLI, and optional Ultralytics backends are
removed. Sparse Identity and Hybrid remain production choices; Hybrid retains
its internal ByteTrack implementation. Vendor ONVIF handling and current runtime
repairs remain supported.
