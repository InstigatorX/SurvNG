# Integrations

SurvNG can notify and be controlled by other systems on your network. The native **Home Assistant integration** uses the authenticated HTTP API and event stream. **MQTT** is an optional, independent transport.

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
