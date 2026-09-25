# Proxmox remote inference worker

Install this only on an accelerator guest. The primary SurvNG server continues
to own cameras, recordings, the database, and the web UI. Keep
`detector.inference_mode` set to `local` until at least one worker is connected
and ready.

The worker service runs only `python -m survng.inference_worker`. It does not
start the web API, camera capture, recording, database, ONVIF, MQTT, or media
services. Do not install or enable `survng.service` in the guest.

Commands are meant to be pasted in order. Set the variables in section 1, then
paste each later block as-is.

## 1. Set these values, then paste the rest

Use the same Git revision the primary server is running. A worker and primary
on different protocol revisions cannot connect.

```bash
SURVNG_UID=1600
SURVNG_GID=1600
SURVNG_ROOT=/opt/survng
SURVNG_GIT_URL=https://github.com/InstigatorX/SurvNG.git
SURVNG_GIT_REF=v1.2
SURVNG_SERVER_URL=https://survng.example.internal/survng
SURVNG_WORKER_NAME="$(hostname)"
SURVNG_WORKER_ROLES=object,face,reid,depth

getent passwd "$SURVNG_UID" || true
getent group "$SURVNG_GID" || true
```

`SURVNG_SERVER_URL` is the primary base URL, including SurvNG's configured base
path. `SURVNG_WORKER_ROLES` may be any comma-separated subset of
`object,face,reid,depth`. The guest loads and warms only those roles.

## 2. Prepare the LXC guest

Use an unprivileged Debian or Ubuntu guest with a bridged private-network
address and outbound access to the primary server. No inbound worker port is
required.

On an Intel accelerator node, pass `/dev/dri/renderD128` into the guest. The
Proxmox UID/GID mapping depends on that host's `render` group and existing
subuid/subgid allocation; verify it on the host rather than copying IDs from
another node. Restrict the Proxmox HA group to nodes with a compatible device.
Do not depend on live migration preserving active GPU work.

Inside the guest:

```bash
ls -l /dev/dri/renderD128
```

## 3. Install host packages

```bash
sudo apt-get update
sudo apt-get install -y \
  ca-certificates curl git python3 python3-venv python3-pip \
  pkg-config build-essential \
  libgl1 libglib2.0-0 libgomp1

python3 --version
```

Python must be 3.12 or newer. Node.js and FFmpeg are not required on a worker.

## 4. Create the worker user and directories

```bash
getent group survng-inference >/dev/null \
  || sudo groupadd --system --gid "$SURVNG_GID" survng-inference
getent passwd survng-inference >/dev/null \
  || sudo useradd --system --uid "$SURVNG_UID" --gid survng-inference \
       --home-dir /var/lib/survng-inference --create-home \
       --shell /usr/sbin/nologin survng-inference

getent group video >/dev/null && sudo usermod -aG video survng-inference || true
getent group render >/dev/null && sudo usermod -aG render survng-inference || true

sudo install -d -o survng-inference -g survng-inference -m 0750 \
  /var/lib/survng-inference /var/lib/survng-inference/models
sudo install -d -o root -g root -m 0755 /etc/survng-inference
```

Do not copy or mount model files. On the first authenticated connection, and
again whenever a configured model changes, the primary streams the required
files. The worker verifies each SHA-256 digest and installs the bundle under
`/var/lib/survng-inference/models` before starting inference. Files already
cached are not transferred again. `/opt/survng/.cache` remains local to the
guest and stores only the OpenVINO compilation cache.

## 5. Install the same SurvNG revision

`/opt` is not writable by the worker user, so create the checkout directory as root first. It must be empty and owned by `survng-inference`.

```bash
sudo install -d -o survng-inference -g survng-inference -m 0750 "$SURVNG_ROOT"
sudo -u survng-inference git clone --branch "$SURVNG_GIT_REF" --single-branch \
  "$SURVNG_GIT_URL" "$SURVNG_ROOT"
sudo install -d -o survng-inference -g survng-inference -m 0750 "$SURVNG_ROOT/.cache"
cd "$SURVNG_ROOT"

sudo -u survng-inference python3 -m venv "$SURVNG_ROOT/.venv"
sudo -u survng-inference "$SURVNG_ROOT/.venv/bin/pip" install --upgrade pip
sudo -u survng-inference "$SURVNG_ROOT/.venv/bin/pip" install -r "$SURVNG_ROOT/requirements.txt"
```

If `SURVNG_GIT_REF` is a commit rather than a branch, clone without
`--branch` and then run `sudo -u survng-inference git checkout "$SURVNG_GIT_REF"`.
Do not build the frontend and do not create a primary `config.json`.

## 6. Create the worker token on the primary

On the primary server, open **Admin → Integrations → API & MQTT**. The **API
Tokens** tab contains a separate **Inference worker token** control. Create or
rotate it and copy the displayed `survng_worker_...` value immediately. It is
shown only once and is not an integration API token.

Alternatively, while authenticated as an administrator:

```bash
curl -fsS -X POST \
  -H "Authorization: Bearer $SURVNG_ADMIN_TOKEN" \
  "$SURVNG_SERVER_URL/api/config/inference-worker-token"
```

Rotating the token changes credentials for new connections. Existing worker
connections remain active until they disconnect. Update the worker environment
and restart the service to apply a rotation immediately.

## 7. Configure and start systemd

```bash
cd "$SURVNG_ROOT"
sudo cp deploy/survng-inference.service /etc/systemd/system/survng-inference.service
sudo cp deploy/survng-inference.env.example /etc/survng-inference/worker.env
sudo python3 - <<PY
from pathlib import Path
path = Path("/etc/survng-inference/worker.env")
replacements = {
    "SURVNG_INFERENCE_SERVER": "$SURVNG_SERVER_URL",
    "SURVNG_INFERENCE_WORKER_NAME": "$SURVNG_WORKER_NAME",
    "SURVNG_INFERENCE_WORKER_ROLES": "$SURVNG_WORKER_ROLES",
}
lines = []
for line in path.read_text().splitlines():
    key, separator, _value = line.partition("=")
    if separator and key in replacements:
        line = f"{key}={replacements[key]}"
    lines.append(line)
path.write_text("\n".join(lines) + "\n")
PY
sudoedit /etc/survng-inference/worker.env
sudo chown root:root /etc/survng-inference/worker.env
sudo chmod 600 /etc/survng-inference/worker.env

sudo systemctl daemon-reload
sudo systemctl enable --now survng-inference
sudo systemctl status survng-inference --no-pager
sudo journalctl -u survng-inference -f
```

Set `SURVNG_INFERENCE_TOKEN` in `/etc/survng-inference/worker.env` to the
one-time worker token before starting the service. The remaining environment
values are:

| Variable | Purpose |
| --- | --- |
| `SURVNG_INFERENCE_SERVER` | Primary base URL, including any configured base path |
| `SURVNG_INFERENCE_WORKER_NAME` | Name shown in detector status |
| `SURVNG_INFERENCE_WORKER_ROLES` | Inference roles this guest should load |
| `SURVNG_INFERENCE_WORKER_ID_FILE` | Stable private worker identity |
| `SURVNG_INFERENCE_MODEL_CACHE_DIR` | Verified model bundle cache |
| `SURVNG_LOG_LEVEL` | Worker log verbosity |

The worker connects outbound to `/api/inference/workers/connect`.

## 8. Verify before cutover

On the primary:

```bash
curl -fsS \
  -H "Authorization: Bearer $SURVNG_ADMIN_TOKEN" \
  "$SURVNG_SERVER_URL/api/detector/status"
```

Confirm `remote_registry.ready` is at least `1`, the expected roles are ready,
and `config_generation` is populated. The first connection can take longer
while model files transfer and engines warm up. Then change the primary
configuration:

```json
{
  "detector": {
    "inference_mode": "hybrid",
    "remote_incident_fallback": true
  }
}
```

Save the complete existing configuration with those fields changed. Do not
replace it with this abbreviated example. Hybrid mode sends work to the worker
and retains local fallback for initial incident detection. After validating
queue delay, failure behavior, and model output, `remote` mode stops loading
the corresponding local model workers.

## Failure behavior

- Expired heartbeats remove a worker from routing.
- Reconnecting with the same worker ID fences the old connection.
- Configuration or model-content changes cause the worker to reconnect, fetch
  only changed model blobs, and reload after its next heartbeat.
- In hybrid mode, only initial incident detection may fall back locally.
  Tracking and enrichment wait until remote capacity is available.
- In remote mode, unavailable workers produce the existing
  detector-unavailable behavior. Camera capture and recording continue on the
  primary.
