# Integrations

SurvNG can notify and be controlled by other systems on your network. The two common paths are **MQTT / Home Assistant** and the **HTTP API**.

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

Keep SurvNG on a trusted LAN or VPN, or behind an authenticated reverse proxy. The HTTP API is an administrative surface, not a public internet service.

## Related

- [HTTP API](api.md)
- [Access](access.md)
- [Admin](admin.md)
- [Cameras](cameras.md)
