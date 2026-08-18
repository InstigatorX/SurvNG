# SurvNG ONVIF Inspector overlay

Target: `InstigatorX/SurvNG`, branch `v1.0`, as inspected August 18, 2026.

This adds a lightweight standalone ONVIF PullPoint sniffer/state visualizer at:

`/onvif`

It deliberately does **not** create a second ONVIF subscription. It observes the existing
`OnvifEventListener` after the notification has been received and parsed.

## What it shows

- live ONVIF notifications
- camera connection state
- granted subscription time remaining
- normalized topic
- classification: motion/person/vehicle/animal/face
- active/inactive/unknown state
- state transitions
- extracted `SimpleItem` values
- bounded/redacted message XML
- unknown notification counts
- renew/renew-error/resubscription/poll counters
- filters for recognized-only and changes-only

The inspector is in-memory only. Restarting SurvNG clears its event history.

## Install

From the ZIP directory, validate the target repo first:

```bash
python3 install_onvif_inspector.py --repo /root/SurvNG --check
```

If the check passes:

```bash
python3 install_onvif_inspector.py --repo /root/SurvNG
cd /root/SurvNG/frontend
npm run build
cd ..
pytest -q tests/test_onvif_inspector.py
```

Then restart SurvNG and open:

```text
http://<survng-host>/onvif
```

If SurvNG uses a configured base path, `/onvif` sits underneath that base path.

## Files installed

New files:

- `survng/app/onvif_inspector.py`
- `survng/app/onvif_inspector_routes.py`
- `frontend/onvif.html`
- `frontend/src/onvif/main.jsx`
- `frontend/src/onvif/OnvifInspector.jsx`
- `frontend/src/onvif/onvif.css`
- `tests/test_onvif_inspector.py`

Small guarded edits:

- `survng/app/onvif_events.py`
- `survng/app/frontend_routes.py`
- `survng/app/main.py`
- `frontend/vite.config.js`

The installer creates `*.onvif-inspector.bak` backups for patched existing files.

## Design notes

The hook is observational. Inspector exceptions are caught and logged at debug level so the
diagnostic UI cannot interrupt normal ONVIF motion processing.

Raw XML is capped at 16,384 characters per event and passed through SurvNG's existing
`redact_secret_text()` before being retained in the in-memory ring buffer.

The event buffer is capped at 1,000 events.

The frontend polls once per second rather than opening a WebSocket, keeping the feature isolated
from SurvNG's async/websocket lifecycle.
