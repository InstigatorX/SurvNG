# Proxmox remote inference worker

Remote inference is opt-in. Keep `detector.inference_mode` set to `local` until
at least one worker is connected and ready.

## Prepare an LXC guest

Use an unprivileged Debian or Ubuntu guest with a bridged private-network
address. On an Intel accelerator node, expose the render device to the guest.
The exact Proxmox ID mapping depends on the host's `render` group and existing
subuid/subgid allocation; verify it rather than copying IDs from another node.

The resulting guest must be able to read `/dev/dri/renderD128`:

```bash
ls -l /dev/dri/renderD128
```

Restrict the Proxmox HA group to nodes with a compatible device. Do not depend
on live migration preserving active GPU work.

## Install the worker

Install the same SurvNG revision and Python dependencies as the main server:

```bash
sudo useradd --system --home /var/lib/survng-inference \
  --create-home --shell /usr/sbin/nologin survng-inference
sudo install -d -o survng-inference -g survng-inference \
  /opt/survng /opt/survng/.cache /etc/survng-inference \
  /var/lib/survng-inference

# Place or clone the repository at /opt/survng, then:
cd /opt/survng
sudo -u survng-inference python3 -m venv .venv
sudo -u survng-inference .venv/bin/pip install --upgrade pip
sudo -u survng-inference .venv/bin/pip install -r requirements.txt
```

Mount or copy model files at the same absolute paths used by the main SurvNG
configuration. The server sends its detector configuration to the worker and
rejects workers running a different configuration generation.

## Create a worker credential

Create the dedicated credential while authenticated as an administrator:

```bash
curl -fsS -X POST \
  -H "Authorization: Bearer $SURVNG_ADMIN_TOKEN" \
  https://survng.example.internal/survng/api/config/inference-worker-token
```

The response reveals the worker token once. Rotating it changes the credential
for new connections; restart existing workers after rotation.

## Configure systemd

```bash
sudo cp deploy/survng-inference.service \
  /etc/systemd/system/survng-inference.service
sudo cp deploy/survng-inference.env.example \
  /etc/survng-inference/worker.env
sudo chmod 600 /etc/survng-inference/worker.env
sudo chown root:root /etc/survng-inference/worker.env
sudoedit /etc/survng-inference/worker.env

sudo systemctl daemon-reload
sudo systemctl enable --now survng-inference
sudo journalctl -u survng-inference -f
```

`SURVNG_INFERENCE_SERVER` may include SurvNG's configured base path. Workers
connect outbound to `/api/inference/workers/connect`; no inbound worker port is
required.

## Verify before cutover

On the main server:

```bash
curl -fsS \
  -H "Authorization: Bearer $SURVNG_ADMIN_TOKEN" \
  https://survng.example.internal/survng/api/detector/status
```

Confirm `remote_registry.ready` is at least `1`, the expected roles and device
are listed, and `config_generation` is populated. Then change:

```json
{
  "detector": {
    "inference_mode": "hybrid",
    "remote_incident_fallback": true
  }
}
```

Save the complete existing configuration with those fields changed; do not
replace it with the abbreviated example. Hybrid mode sends work remotely but
retains local fallback for initial incident detection. After validating queue
delay, failure behavior, and model output, `remote` mode can avoid loading local
model workers.

## Failure behavior

- Expired heartbeats remove a worker from routing.
- Reconnecting with the same worker ID fences the old connection.
- Configuration changes cause the worker to reconnect and reload models after
  its next heartbeat.
- In hybrid mode only `INCIDENT_INITIAL` may fall back locally. Tracking and
  enrichment are deferred when remote capacity is unavailable.
- In remote mode unavailable workers produce the existing detector-unavailable
  behavior; camera capture and recording continue.
