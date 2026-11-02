# Integrations

SurvNG can notify and be controlled by other systems on your network. The native **Home Assistant integration** uses the authenticated HTTP API and event stream. **MQTT** is an optional, independent transport. **Grok Bot** connects through the MCP endpoint below.

## Native Home Assistant incident notifications

Update SurvNG and the custom Home Assistant integration together. The native
integration does not require MQTT for incident notifications. It subscribes to:

```http
GET /api/events/stream?incidents_only=1
Authorization: Bearer YOUR_TOKEN
```

Use a token with `read` scope. The first connection receives an
`incident_notifications_state` snapshot, followed by `incident_lifecycle` events.
Schema 2 includes `new` / `updated` / `complete` state, increasing per-incident
`revision`, detected objects, recognized identities, factual summaries, image
availability, and timestamps. `Last-Event-ID` or `last_event_id` resumes replay.
The integration metadata endpoint advertises `incident_notifications.schema_version`.

SurvNG owns grouping and settlement even when MQTT is disabled or disconnected.
Late cover and identity refinements update the same incident, including after
completion. Active groups and the latest 256 completed groups are journaled in
`incident_notifications.json` in the configured database directory. Shutdown
suspends active groups; restart resumes their inactivity deadlines. A recovery
snapshot repairs missed revisions when the stream cursor is too old or belongs
to a previous process. This is bounded recovery, not an unlimited event archive.

The native integration supplies authenticated HA media attachments and a
notification blueprint. See the integration README for image retention and
mobile-device setup. If MQTT incident publishing is enabled, the same enriched
lifecycle payloads are also published to `survng/events/incidents` (using the
configured topic prefix). MQTT discovery remains independent of native entities.

## MQTT and Home Assistant

Enable MQTT under **Admin → Integrations**.

SurvNG publishes topics such as:

```text
survng/status
survng/camera/CAMERA_ID/state
survng/camera/CAMERA_ID/motion
survng/camera/CAMERA_ID/object
survng/server/state
survng/server/metrics
```

When Home Assistant discovery is on, cameras and zones appear as devices/entities automatically.

### Example: turn a camera off from MQTT

Publish `OFF` to:

```text
survng/camera/front-door/power/set
```

Use `ON` to turn it back on. Replace `survng` with your configured topic prefix and `front-door` with the camera id.

## API tokens

Optional bearer tokens let automations call SurvNG safely when authentication is enabled.

Deleting the last token while API authentication is enabled is rejected. Create a
replacement token first, or explicitly disable API authentication if anonymous
access is intended. Open event streams recheck credentials and read permissions
before delivery and while idle, so revocation also closes existing streams.

Scopes:

| Scope | Allows |
| --- | --- |
| `read` | GET-style reads |
| `camera:control` | Camera power, recording, and detection toggles |
| `admin` | Configuration changes and other writes (includes the other scopes) |

Create a token in **Admin → Server → API**, or with:

```bash
.venv/bin/python scripts/create-api-token.py \
  --id home-assistant \
  --name "Home Assistant" \
  --scope read \
  --scope camera:control \
  --enable
```

Send it as:

```http
Authorization: Bearer YOUR_TOKEN
```

Important: browser users (Admin → Access) and API tokens can both be enabled. The UI uses a session cookie; automations keep using `Authorization: Bearer`.

## Grok Bot

Grok Bot reaches SurvNG through a stateless MCP endpoint on the same process:

```text
POST /api/mcp
Authorization: Bearer YOUR_TOKEN
```

With the usual proxy prefix that is `https://<your-host>/survng/api/mcp`. The endpoint speaks JSON-RPC (`initialize`, `ping`, `tools/list`, `tools/call`) and returns one JSON result per call. It does not keep a session and it does not replace the incident event stream.

Enable API authentication first. MCP answers 401 while API authentication is off, and it ignores browser session cookies. Create a token with the `read` scope for a standing bot. Add `camera:control` only on a separate token whose job is to start or stop a camera, recording, or detection. Clip export is available to an `admin` token.

A read token can check health, explain configuration, search and inspect incidents, summarize recent activity, run Smart Search, trace an incident across cameras, fetch one snapshot, and poll the latest incident journal. `tools/list` hides camera control and export unless the token's scope includes them. The snapshot tool returns one JPEG with the long edge capped at 1280 pixels. Responses omit credentials, stream URLs, and filesystem paths.

Point the bot's MCP configuration at that URL and send the bearer token in the `Authorization` header:

```json
{
  "mcpServers": {
    "survng": {
      "url": "https://nvr.example/survng/api/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_TOKEN"
      }
    }
  }
}
```

The token lives on the shared Grok Bot computer, so every bot on that account can use it. Keep the standing token at `read`.

Leave port `8088` unpublished. Publish SurvNG only through the existing HTTPS reverse proxy, the same way browser sign-in is published.

Useful standing jobs:

- Morning health digest: `get_system_health` and `recent_incidents`.
- Overnight summary: `summarize_recent_activity` or `search_incidents` for people and vehicles.
- Follow-up on one event: `get_incident_image`, then `trace_across_cameras`.

A starter handoff:

```text
Check SurvNG health, then summarize person and vehicle incidents since yesterday evening. For anything unusual, fetch the incident image and trace it across cameras. Leave a short list with camera, time, and what you saw. Leave camera settings unchanged.
```

Camera restart stays on the `camera:control` token, and only after you explicitly ask for that camera to start or stop.

## Stream sources for Home Assistant

Integrations can ask SurvNG for a go2rtc stream descriptor:

```http
GET /api/cameras/front-door/stream-source?source=live
```

Treat stream URLs as operational secrets even when passwords are stripped.

## Network placement

Keep SurvNG on a trusted LAN or VPN, or on the internet only with browser sign-in and an HTTPS reverse proxy. Do not publish port `8088`. API bearer tokens stay separate from browser users; give them the smallest scope they need. See [Reverse proxy](reverse-proxy.md).

## Related

- [HTTP API](api.md)
- [Access](access.md)
- [Admin](admin.md)
- [Cameras](cameras.md)

### Zone notification controls

In Zone config, **HA/MQTT Notifications** is directly below **Exclude from EMA**.
Checked means on (the default). Save the zone settings to apply it. The native HA
zone notification switch controls the same persisted setting; update the HA
integration to use shared controls. HA refreshes external changes during metadata
polling. Existing HA-local mutes migrate when the updated integration loads.

If every zone matched by an incident is off, HA notification bus deliveries and
MQTT incident notifications are suppressed. Any enabled matched zone permits the
whole incident; incidents without zones still pass. MQTT zone object topics also
respect the setting. Camera activity telemetry, detection, recording, and HA
incident entity updates continue. Turning notifications on does not replay old
notifications.

### Shared HA and MQTT notification settings

Under **API & MQTT/HA → MQTT/HA → General**, enable **Exclude motion-only
incidents** to suppress motion-only lifecycle notifications before they reach HA
or MQTT. It applies to live delivery, reconnect replay, and HA recovery snapshots.
The default is off. If an incident later detects an object or person, subsequent
updates and completion are delivered. Detection, recording, and camera activity
telemetry continue normally.

**Notification URL** sets the public SurvNG base URL for incident links, for example
`https://ha.loebees.com/survng`. Include the proxy path prefix; do not include
credentials or a query string. Leave it blank to preserve existing URLs.
The HA integration must be updated to honor this URL. HA downloads images using
its configured API connection and serves notification attachments from its own
authenticated media directory; the public SurvNG URL does not relocate HA media.
Both settings apply when saved without restarting camera workers.

### Opening notification incidents

New notification links use `/incidents/incident-{camera_id}-{first_event_id}`.
They open a focused, responsive page with a large evidence image, summary, camera,
time, zones, lifecycle status, and a **Play incident** action using the existing
recording window and pre-roll. Ongoing incidents also offer **Live view**.
The page refreshes while visible and preserves the last details during a temporary
connection failure. **Open full investigation** leads to the existing workspace.

Evidence shows the currently saved first detection and representative image
(labeled final after completion), followed by detection times and recognized
people. Stored images can be refined; this is not an archive of every image
revision sent to a phone. If the lifecycle journal has expired, retained events
still resolve the link and the status reads **Recorded**. Deleted evidence shows
an unavailable state. Existing browser authentication and proxy prefixes apply.

### Per-camera incident notifications

In camera settings under **Motion/Object → HA/MQTT Options**, **Send incident
notifications** controls native HA and MQTT incident lifecycle messages for that
camera. It defaults to enabled. Disabling it also excludes the camera from native
incident replay and recovery snapshots; local detection, recording, and evidence
remain available. Shared motion filtering and zone notification settings still apply.

The camera-control API exposes `PUT /api/cameras/{camera_id}/incident-notifications`
with `{"enabled": false}` or `{"enabled": true}`. This changes the saved setting
without restarting camera workers. Camera status includes
`incident_notifications_enabled` for the native HA switch.

The global **API & MQTT/HA → MQTT/HA → General → Enable incident notifications**
switch defaults to on. Turning it off suppresses native HA and MQTT incident
lifecycle delivery for every camera, including replay and recovery snapshots.
It preserves the individual camera/zone preferences, detection, recording, and
broker connection settings. Save settings to apply; no camera restart is needed.
