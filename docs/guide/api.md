# HTTP API

SurvNG exposes an HTTP API used by the browser UI and by automations. This page is the operator-friendly map of those endpoints. Interactive OpenAPI docs are also available at `/docs` on the SurvNG host when the service is running (with your configured `base_path` prefix, for example `/survng/docs`).

## Conventions

- Base URL example: `http://127.0.0.1:8088`
- If `base_path` is `/survng`, browser URLs look like `http://host:8088/survng/...`
- Unprefixed `/api/...` routes remain available to local clients
- JSON request and response bodies unless a path ends with `.jpg`, `.mp4`, or `.m3u8`
- Timestamps are generally ISO 8601; some recording endpoints use Unix epoch seconds

### Authentication

Browser users and API tokens are optional and off by default.

#### Browser sessions

Enable sign-in under **Admin → Users & Access**. The UI then uses an HTTP-only session cookie after `POST /api/auth/login`.

| Role | Typical use |
| --- | --- |
| `admin` | Full console, including configuration |
| `viewer` | Watch and review; cannot change configuration |

#### API tokens

When API authentication is enabled, send:

```http
Authorization: Bearer YOUR_TOKEN
```

| Scope | Typical use |
| --- | --- |
| `read` | List cameras, incidents, recordings, status |
| `camera:control` | Start/stop camera, recording, detection |
| `admin` | Change configuration, run maintenance, most POST/PUT/PATCH/DELETE |

`GET /api/health` and the sign-in routes stay reachable without a session.

See [Integrations](integrations.md) for creating tokens and [Access](access.md) for browser users.

### Quick health check

```bash
curl -s http://127.0.0.1:8088/api/health
```

```json
{"status":"ok"}
```

---

## System and status

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Liveness |
| GET | `/api/system/status` | Lifecycle, cameras, storage, detector summary |
| GET | `/api/telemetry` | Detailed telemetry snapshot |
| POST | `/api/telemetry/diagnostics` | Start a diagnostics capture |
| GET | `/api/telemetry/diagnostics/{session_id}` | Read diagnostics session |
| DELETE | `/api/telemetry/diagnostics/{session_id}` | End diagnostics session |
| GET | `/api/accelerator` | Accelerator / device info |
| GET | `/api/detector/status` | Object detector status |
| GET | `/api/detector/models` | Discoverable models |
| GET | `/api/object-tracking/catalog` | Tracking engine catalog |
| GET | `/api/motion/pipeline/catalog` | Motion pipeline catalog |
| GET | `/api/integrations/home-assistant` | Home Assistant integration summary |
| GET | `/api/logs` | Recent logs (`admin`) |
| GET | `/api/events/stream` | Server-sent events for live UI updates |
| GET | `/api/auth/session` | Current browser session |
| POST | `/api/auth/login` | Sign in |
| POST | `/api/auth/logout` | Sign out |
| GET | `/api/auth/users` | List local users (`admin`) |
| GET | `/api/tls` | HTTPS certificate status (`admin`) |
| POST | `/api/tls/upload` | Store a PEM certificate and key from files (`admin`) |
| POST | `/api/tls/certificate` | Store pasted PEM certificate and key (`admin`) |

### Example: system status

```bash
curl -s http://127.0.0.1:8088/api/system/status | python -m json.tool
```

---

## Cameras

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/cameras` | List cameras and runtime state |
| GET | `/api/cameras/{camera_id}/snapshot.jpg` | Fresh snapshot |
| GET | `/api/cameras/{camera_id}/zone-snapshot.jpg` | Snapshot for zone editing |
| GET | `/api/cameras/{camera_id}/live-info` | Live transport details |
| GET | `/api/cameras/{camera_id}/stream-source` | go2rtc stream descriptor (`source=live\|main`) |
| GET | `/api/cameras/{camera_id}/stream.mjpg` | MJPEG preview stream |
| POST | `/api/cameras/{camera_id}/camera/start` | Start camera (`camera:control`) |
| POST | `/api/cameras/{camera_id}/camera/stop` | Stop camera (`camera:control`) |
| POST | `/api/cameras/{camera_id}/recording/start` | Start recording (`camera:control`) |
| POST | `/api/cameras/{camera_id}/recording/stop` | Stop recording (`camera:control`) |
| PUT | `/api/cameras/{camera_id}/recording` | Set recording enabled flag (`camera:control`) |
| PUT | `/api/cameras/{camera_id}/detection` | Set detection enabled flag (`camera:control`) |
| POST | `/api/cameras/{camera_id}/motion-test` | Inject a manual motion test |
| GET/PUT | `/api/cameras/{camera_id}/motion-debug` | Motion diagnostics controls |
| GET | `/api/cameras/{camera_id}/motion-debug/{layer}.jpg` | Diagnostics layer image |

### Example: list cameras

```bash
curl -s http://127.0.0.1:8088/api/cameras | python -m json.tool
```

### Example: Home Assistant stream descriptor

```bash
curl -s 'http://127.0.0.1:8088/api/cameras/front-door/stream-source?source=live'
```

---

## Configuration

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/config` | Current configuration (secrets masked) |
| PUT | `/api/config` | Replace configuration (`admin`) |
| PUT | `/api/config/cameras/{camera_id}` | Update one camera (`admin`) |
| DELETE | `/api/config/cameras/{camera_id}` | Remove camera (`admin`) |
| PUT | `/api/config/cameras/{camera_id}/zones` | Replace zones (`admin`) |
| PUT | `/api/config/cameras/order` | Reorder cameras (`admin`) |
| POST | `/api/config/probe` | Probe camera/ONVIF capabilities (`admin`) |
| GET | `/api/config/api-tokens` | List token metadata |
| POST | `/api/config/api-tokens` | Create token (`admin`) |
| DELETE | `/api/config/api-tokens/{token_id}` | Delete token (`admin`) |

### Example: read masked config

```bash
curl -s http://127.0.0.1:8088/api/config | python -m json.tool
```

---

## Incidents and events

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/events` | Recent raw events |
| GET | `/api/incidents` | Incident list |
| GET | `/api/incidents/feed` | Compact live feed |
| GET | `/api/incidents/detail` | Detailed incident payload |
| GET | `/api/incidents/search` | Filtered incident search |
| GET | `/api/incidents/by-event/{event_id}` | Incident containing an event |
| GET | `/api/events/{event_id}/snapshot.jpg` | Full evidence image |
| GET | `/api/events/{event_id}/thumbnail.jpg` | Resized thumbnail (`object_focus=true` crops to detected objects from the full snapshot before resize; optional `zoom`, `aspect_w`, `aspect_h`) |
| GET | `/api/events/{event_id}/clip.mp4` | Generated clip |
| GET | `/api/events/{event_id}/stream.m3u8` | HLS around the event |
| GET | `/api/events/{event_id}/related-incidents` | Nearby related incidents |
| GET | `/api/events/{event_id}/appearance-matches` | Visual appearance matches |
| POST | `/api/events/{event_id}/detect` | Re-run detection helpers |
| POST | `/api/incidents/{event_id}/ai-apply` | Apply confirmed incident AI recommendation |

### Example: search person incidents on one camera

```bash
curl -s -G 'http://127.0.0.1:8088/api/incidents/search' \
  --data-urlencode 'camera_ids=front-door' \
  --data-urlencode 'object_labels=person' \
  --data-urlencode 'limit=20' | python -m json.tool
```

Exact query parameter names follow the live OpenAPI schema at `/docs` if your build adds filters.

---

## Recordings, timeline, and exports

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/cameras/{camera_id}/recordings` | Recording index summary |
| GET | `/api/cameras/{camera_id}/recordings/day` | One day of segments |
| GET | `/api/cameras/{camera_id}/recordings/day.m3u8` | Day HLS playlist |
| GET | `/api/cameras/{camera_id}/recordings/window` | Segments covering a window |
| GET | `/api/cameras/{camera_id}/recordings/preview.jpg` | Frame near an epoch |
| GET | `/api/cameras/{camera_id}/recordings/events` | Events overlapping recordings |
| GET | `/api/cameras/{camera_id}/recordings/updates` | Incremental day updates |
| GET | `/api/recordings/grid/day` | Multi-camera day grid |
| GET | `/api/recordings/grid/updates` | Grid incremental updates |
| POST | `/api/exports` | Start export job |
| GET | `/api/exports` | List exports |
| GET | `/api/exports/summary` | Export totals |
| POST | `/api/exports/batch` | Batch export operations |
| GET | `/api/exports/{job_id}` | Job status |
| GET | `/api/exports/{job_id}/download` | Download finished file |
| GET | `/api/exports/{job_id}/media` | Media URL helpers |
| PATCH | `/api/exports/{job_id}/protection` | Protect / unprotect |
| PATCH | `/api/exports/{job_id}/metadata` | Edit export metadata |
| DELETE | `/api/exports/{job_id}` | Delete export |

### Example: exact frame at a timestamp

```bash
curl -s -o frame.jpg \
  'http://127.0.0.1:8088/api/cameras/front-door/recordings/preview.jpg?epoch=1724500000&source=main&width=1280&exact=true' \
  -D -
```

Response headers can include `X-SurvNG-Requested-Timestamp` and `X-SurvNG-Actual-Timestamp`. Details: [Recording frame API](../recording-frame-api.md).

### Example: start an export

```bash
curl -s -X POST http://127.0.0.1:8088/api/exports \
  -H 'Content-Type: application/json' \
  -d '{
    "camera_id": "front-door",
    "source": "main",
    "start_epoch": 1724500000,
    "end_epoch": 1724500045
  }'
```

Field names must match your running build’s OpenAPI schema — use `/docs` if this example needs adjusting for a newer export payload.

---

## Smart Search

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/semantic-search/status` | Index and model status |
| POST | `/api/semantic-search` | Search by text description |

### Example

```bash
curl -s -X POST http://127.0.0.1:8088/api/semantic-search \
  -H 'Content-Type: application/json' \
  -d '{"query":"person in a red jacket","limit":20}' | python -m json.tool
```

Setup: [Smart Search model packages](../semantic-search.md).

---

## People and faces

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/faces/status` | Face subsystem status |
| GET | `/api/faces/people` | Known people |
| POST | `/api/faces/people` | Create person |
| DELETE | `/api/faces/people/{person_id}` | Delete person |
| GET | `/api/faces/people/{person_id}/history` | Sighting history |
| GET | `/api/faces/people/{person_id}/representation` | Gallery representation |
| GET | `/api/faces/observations` | Face observations |
| GET | `/api/faces/observations/{observation_id}` | One observation |
| PUT | `/api/faces/observations/{observation_id}` | Update / assign observation |
| GET | `/api/faces/observations/{observation_id}/crop.jpg` | Face crop |
| PUT | `/api/faces/observations/{observation_id}/reference` | Pin reference |
| GET | `/api/faces/review/queue` | Review queue |
| GET | `/api/faces/review/confirmed` | Confirmed reviews |
| POST | `/api/faces/review/bulk` | Bulk review actions |
| GET | `/api/faces/unknown-clusters` | Unknown clusters |
| GET | `/api/faces/unknown-clusters/{cluster_id}/members` | Cluster members |
| POST | `/api/faces/unknown-clusters/rebuild` | Rebuild clusters |
| GET | `/api/faces/benchmark` | Recognition benchmark summaries |
| GET | `/api/faces/calibration` | Calibration status helpers |
| POST | `/api/faces/people/{person_id}/gallery/optimize` | Optimize gallery |
| POST | `/api/faces/people/{person_id}/gallery/enrich` | Enrich gallery |

---

## Motion intelligence and assistant

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/motion-audit` | Motion audit list |
| GET | `/api/motion-audit/{audit_id}` | Audit detail |
| GET | `/api/motion-audit/{audit_id}/snapshot.jpg` | Audit image |
| GET | `/api/motion-effectiveness` | Effectiveness summary |
| POST | `/api/motion-audit/{audit_id}/ai-analyze` | Analyze one audit |
| POST | `/api/motion-audit/{audit_id}/ai-apply` | Apply confirmed audit advice |
| POST | `/api/motion-ai-reviews` | Start camera intelligence review |
| GET | `/api/motion-ai-reviews/latest` | Latest review |
| GET | `/api/motion-ai-reviews/{review_id}` | Review detail |
| POST | `/api/motion-ai-reviews/{review_id}/apply` | Apply confirmed review advice |
| GET | `/api/camera-intelligence/evaluations/latest` | Latest evaluation |
| POST | `/api/camera-intelligence/evaluations/{evaluation_id}/follow-up` | Follow-up review |
| GET | `/api/assistant/status` | Assistant readiness |
| POST | `/api/assistant/chat` | Chat with tools |
| POST | `/api/calibration/runs` | Start detection tune-up |
| GET | `/api/calibration/runs` | List tune-up runs |
| GET | `/api/calibration/runs/{run_id}` | Run detail |
| POST | `/api/calibration/runs/{run_id}/cancel` | Cancel run |
| POST | `/api/calibration/runs/{run_id}/retry` | Retry run |
| POST | `/api/calibration/runs/{run_id}/preview` | Preview recommendations |
| POST | `/api/calibration/runs/{run_id}/apply` | Apply selected recommendations |
| GET | `/api/calibration/change-sets` | Applied change sets |
| POST | `/api/calibration/change-sets/{change_set_id}/evaluate` | Evaluate after wait |
| POST | `/api/calibration/change-sets/{change_set_id}/keep` | Keep changes |
| POST | `/api/calibration/change-sets/{change_set_id}/rollback` | Roll back changes |

### Example: assistant chat

```bash
curl -s -X POST http://127.0.0.1:8088/api/assistant/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Which cameras need attention?"}' | python -m json.tool
```

---

## Detection utilities

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/api/detector/frame` | Detect on an uploaded/selected frame |
| GET | `/api/detector/model-evaluation` | Evaluation job status |
| POST | `/api/detector/model-evaluation` | Start evaluation |
| DELETE | `/api/detector/model-evaluation` | Cancel evaluation |
| GET | `/api/tracking-comparisons` | Tracking comparison list |
| PUT | `/api/tracking-comparisons/{comparison_id}/verdict` | Record verdict |
| POST | `/api/events/{event_id}/tracking-comparison` | Compare trackers on an event |
| GET | `/api/appearance-index/status` | Appearance index status |
| POST | `/api/appearance-index/backfill` | Backfill appearance signatures |
| GET | `/api/event-clip/settings` | Clip window settings |
| GET | `/api/recordings/cache/status` | Playback cache status |

---

## Training samples

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/training/samples` | Paginated incident samples for external labeling |

### Example

```bash
curl -s -G 'http://127.0.0.1:8088/api/training/samples' \
  --data-urlencode 'start_at=2026-08-01T00:00:00-04:00' \
  --data-urlencode 'end_at=2026-08-08T00:00:00-04:00' \
  --data-urlencode 'camera_ids=gate,front-door' \
  --data-urlencode 'object_labels=person,car' \
  --data-urlencode 'limit=100' | python -m json.tool
```

Full contract: [Training samples API](../training-api.md).

---

## Operations and maintenance

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/retention/status` | Retention plan / status |
| POST | `/api/retention/run` | Run retention cleanup |
| GET | `/api/maintenance/storage` | Storage maintenance status |
| POST | `/api/maintenance/storage` | Start storage maintenance |
| DELETE | `/api/maintenance/storage` | Cancel storage maintenance |
| POST | `/api/system/restart` | Request process restart |
| GET | `/api/system/update` | Product update status |
| POST | `/api/system/update` | Start product update |

---

## ONVIF inspector

Prefix: `/api/onvif-inspector`

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/onvif-inspector/events` | Recent inspected ONVIF events |
| GET | `/api/onvif-inspector/state` | Inspector state |
| POST | `/api/onvif-inspector/clear` | Clear buffer |

Useful when learning which motion topics a camera actually emits.

---

## Errors you may see

| Code | Meaning |
| --- | --- |
| 401 | API auth enabled and token missing/invalid |
| 403 | Token lacks required scope |
| 404 | Unknown camera, event, or missing recording coverage |
| 409 | Conflict such as active storage or AI work blocking restart |
| 503 | Dependency unavailable (stream, detector, go2rtc) |

---

## Related reading

- [Integrations](integrations.md)
- [API architecture](../api-architecture.md) (developer-oriented module boundaries)
- [Recording frame API](../recording-frame-api.md)
- [Training samples API](../training-api.md)
