# HomeKit review: 2026-11-14

Reviewed the integration through `80bda7f` (remote encoders/current-keyframe
replay), including `6dca5b2` (camera aspect ratios), and the changes below.

## Execution boundaries

`AppManager` owns `HomeKitService`, the media budget, camera power/detection
callbacks, and delivery of persisted object events. Targeted configuration
application reconfigures HomeKit without reloading unrelated cameras. Service
operations serialize worker replacement, shutdown, pairing reset, and encoder
removal; worker shutdown happens outside the service state lock.

Apple pairing, HAP, RTP/WebRTC and HDS remain on the primary server. An assigned
node probes the camera, qualifies HEVC, and uploads demanded MPEG-TS tiers. Local
Unix feeds pass that media to the primary worker. Node authentication is separate
from browser sessions; enrollment is single use. The primary's relay capacity and
the node's bounded qualification/snapshot work have separate ownership.

## Findings fixed

| Area | Failure | Correction / regression |
| --- | --- | --- |
| Encoder removal | Compared a node ID against assigned camera IDs, allowing deletion of an assigned encoder. | Reject nonempty assignments and serialize removal with reconfiguration. |
| Upload authorization | An upload retained authority after camera reassignment, revocation, power-off, or shutdown. | Revalidate each write against live assignments, registry identity and demand. |
| Upload ownership | Concurrent uploads could interleave MPEG-TS bytes. | One claim per feed; replacement invalidates the old claim, and old cleanup cannot release its successor. |
| Remote qualification | Old source geometry/backend could be reused after source, audio, transport or software-permission changes. | Invalidate agent caches and require the report's assignment revision to match the current source settings. |
| Encoder permission | A profile without a qualified backend was treated as software-ready. | Remain waiting until an explicit permitted backend is reported. |
| Remote lifecycle | Ended/rejected uploads leaked FFmpeg; a late old failure could kill a replacement. | Bind completion to its child, await termination, escalate SIGTERM to SIGKILL and handle spawn failures. |
| Heartbeat | Serial probing, qualification and snapshots could block sync and privacy reconciliation. | Bounded cancellable background jobs, fair qualification scheduling, sync deadlines and teardown after sync failure. |
| Snapshot IPC | A 12-second snapshot wait blocked media admission, release and motion heartbeats. | Defer the reply and poll completion from the supervisor loop. |
| Snapshot ownership | Late or duplicate JPEGs could satisfy another request or race file replacement. | Unique request IDs, one pending request per camera, locked acceptance and cancellation. |
| Input bounds | Snapshot bodies were buffered before checking size; invalid tier metadata could reach HAP. | Enforce the JPEG limit while reading and validate unique tiers, dimensions, cadence, bitrate and advertised resolutions. |
| Registry lifetime | Stale requests could use an old registry after manager shutdown; sync used assignments captured before its wait/report. | Reject closed registries and evaluate assignments at response time. Feed accept/close also drains safely. |
| Optional state | Corrupt encoder state could prevent the entire application from starting. | Preserve the file, disable node operations, and expose an admin error while leaving SurvNG startup available. |
| Viewer replay | Joining between TS packet fragments omitted the already-broadcast packet prefix. | Include the partial packet at the replay/live boundary. |
| Recording completion | Repeated false motion heartbeats continually extended post-roll. | Record only the true-to-false transition. |
| Recording privacy | A pending fragment read could deliver data after privacy changed. | Recheck cancellation and privacy after the read, before yielding. |
| Pairing supervision | A failed reset left the camera without supervision; repeated control failures never restarted a stuck worker. | Restore supervision after rejected resets, reject lock symlinks, and restart after three consecutive control failures. |
| Pairing UI | A delayed response restored material hidden by navigation/configuration/visibility changes. | Abort pending disclosure requests and discard hidden/aborted responses. |
| Installation/update | Predictable root-owned temporary paths; PATH-based FFmpeg failed distribution; partial updates mixed components. | Private temporary directories, architecture check, resolve FFmpeg, validate archives, stage all components and roll back failed installation. |

## Validation and scope

Regressions exercise invalidated uploads, real private socket delivery, snapshot
request generations and nonblocking admission, corrupt state preservation,
permission/profile checks, TS boundaries, recording privacy/post-roll, pairing
reset failure and delayed browser responses. Agent tests spawn real children,
including a SIGTERM-resistant encoder, and run the actual agent against an HTTP
update server for both success and failure. Update tests verify multi-component
rollback and reject archive links.

Run Python coverage with:

```sh
.venv/bin/python -m pytest -q tests/test_homekit.py tests/test_homekit_nodes_review.py tests/test_media_sessions.py tests/test_config_reload.py tests/test_api_security.py
```

Build the worker and run its suite. The optional media/QSV checks use
`HKSV_MEDIA_TESTS=1 HKSV_QSV_TESTS=1` and `HKSV_FFMPEG` pointing to the qualified
FFmpeg. Real supervisor lifecycle checks use `HKSV_CAMERA_RUNTIME_TESTS=1` and
`HKSV_NODE_PATH` pointing to Node 24. The browser check is
`node tests/homekit-settings.mjs` in `frontend`.

The revised remote agent and primary must be deployed together: assignment
revisions and snapshot request IDs deliberately reject stale reports/results.
The normal node release mechanism supplies the updated agent. Existing Apple
pairing state is retained.

Automated media peers establish protocol and lifecycle behavior, not complete
Apple hardware acceptance. Real-camera cellular/Apple TV, Home recording and
exported pre-roll/audio-off checks, remote-container installation, reconnect
acceptance and sustained live operation still require their hardware checks.
No finite code review proves an absence of all defects.

Recorded results for this campaign:

- Python integration/configuration/security/media budget selection: 198 passed,
  26 subtests passed; the opt-in supervisor test is skipped in that selection.
- Isolated real Node supervisor run: 43 passed, including preserved identity,
  independent replacement/reset and recovery from unavailable control responses.
- Full worker media/QSV suite: 49 passed, no skips.
- Frontend production build and HomeKit browser checks: passed. Vite reports its
  existing large-chunk advisory.
- `git diff --check`: passed.

The initial opt-in supervisor run hit its old startup deadline after the newer
geometry probe tried the dummy camera address. The fixture now uses a closed
loopback port, honors `HKSV_FFMPEG`, and allows bounded probe/startup time; the
corrected run passed. Test fixtures and runtime state were kept outside Git.

The tested worker build was installed atomically and `survng.service` restarted.
The owner-only observability socket reports the new server process running with
13 configured cameras. Private worker status confirms Gate, Upper Garage and
Front Door paired, HomeKit active, QSV selected and zero media failures. All three
use the remote encoder; its latest report contains qualification revisions for
all three cameras. SHA-256 comparisons confirm their existing identity files are
unchanged. This verifies restart/recovery, not a fresh visual playback acceptance.

## Follow-up outage

The remote encoder stopped reporting at approximately 19:09 Eastern on
2026-11-14. Its service log showed a successful process exit (`status=0`) and
`Restart=on-failure`, leaving all three assigned accessories offline. Earlier
entries also showed two OOM kills in the 2 GiB container; those are a separate
resource issue, not proof of what caused the successful exit.

Restarting the encoder restored all three paired QSV workers. An exit guard now
reports unexpected event-loop exhaustion as failure, while explicit SIGTERM/SIGINT
shutdown remains successful. Process tests reproduce a pending promise with no
remaining handles and verify both unexpected-exit and requested-stop behavior.
New installer units use `Restart=always`; existing units require a systemd
drop-in/reload to receive that policy. The tested agent is distributed through
the existing release mechanism without restarting SurvNG.

Follow-up checks: 46 Python tests passed (one optional test skipped), 41 worker
tests passed (four optional media/hardware tests skipped). Memory pressure still
requires measurements from the encoder; do not classify the OOMs as resolved.

Recovery confirmation: the operator applied the existing-unit drop-in and
confirmed `Restart=always`, service active, current memory 459,886,592 bytes
(about 439 MiB) and peak 460,632,064 bytes. The primary observed the agent archive
download followed by fresh qualification reports. All three private worker
endpoints again report paired, HomeKit active, QSV and zero media failures;
Front Door has an active local session. This confirms recovery and current
healthy memory use, not resolution of the earlier OOM cause.

## Upper Garage wide-frame correction

The selected live source probes as H.264, 1920×832. The previous closest-family
selection advertised 1920×1080 and the increase/crop filter removed approximately
23% of the horizontal field of view. Tier geometry now fits the source aspect
inside each family's resource budget, producing 1920×832, 1280×554 and 640×278 for
this camera. Conventional 16:9, 4:3 and portrait family sizes are retained when
the source aspect matches them.

The shared encoder now fits the complete display aspect into the advertised
canvas and pads the small rounding mismatch instead of cropping. This also
preserves the field of view if a source changes geometry after its initial probe.
A real-frame regression encodes a wide image with contrasting left/right edge
markers, decodes it, and checks both edges at every tier using software and QSV.
It also tests a stale 16:9 tier to ensure the filter itself preserves the edges.

Wide-frame validation passed: 52 worker tests with media and QSV enabled, no
skips; 46 Python HomeKit tests passed, one optional lifecycle test skipped.
The supported recording configuration hash in HAP-NodeJS invalidates cached
selected recording settings when these advertised resolutions change. The
worker build was installed for the remote agent's normal update mechanism.

After automatic distribution, the live node report confirms Upper Garage tiers
1920×832, 1280×554 and 640×278. Its private worker status reports HomeKit active,
QSV, paired and zero media failures; all three camera identity hashes still
match their pre-review values. Visual confirmation in Apple Home was requested.

## Intermittent connectivity after wide-frame acceptance

The user confirmed both sides of Upper Garage are visible, but reported dropouts
across all cameras. New node logs showed repeated `encoder exited unexpectedly`
events after 70–170 seconds, with current memory about 451 MiB and peak about
709 MiB, without corresponding new OOM events. The restart guard recovered these
exits but did not prevent them.

The daemon supervisor now deliberately retains a referenced event-loop timer
through the main task and cleanup. Pending promises and AbortSignal timeout
handles alone do not keep Node running. A monotonic progress watchdog identifies
the current operation (no URLs or credentials), aborts stalled operations, and
bounds shutdown before requesting a failed restart. Explicit service shutdown
still drains and exits successfully. Tests verify a real unreferenced
AbortSignal timeout completes, stuck work has bounded recovery, intentional stop
is clean, and HTTP update success/failure still behave correctly. Latest ordinary
worker selection: 42 passed, five optional media/hardware tests skipped.

The first three-minute continuity check failed at 175 seconds: all three workers
became unavailable after a 22.5-second heartbeat gap. The node watchdog then
reported a stall specifically during feed cleanup. Thus the lifetime/watchdog
change was recovery and diagnosis, not resolution of the underlying stall.

A local reproducer established the blocked cleanup: a producer's stdout was
piped to a backpressured consumer, then the producer was terminated. Its process
had exited with SIGTERM, but the child `close` event remained pending until the
paused pipe was manually drained. The previous signal-only stop could therefore
wait forever and block all subsequent node heartbeats.

Stopping a child now closes its owned stdio pipes, as well as signaling the
process, before awaiting closure. Every agent feed also owns an AbortController
for its HTTP upload; stopping or natural child closure cancels that upload.
Regressions cover a deliberately blocked Writable, a real HTTP server that stops
reading an upload, natural producer exit, stale upload callbacks and SIGTERM
escalation. Blocked cleanup now completes in tens of milliseconds. Full
media/QSV selection passed 55 tests; the subsequently added real HTTP regression
passed with all nine targeted agent-state tests. The build was distributed
through the normal agent release mechanism.

Feed-cleanup runtime acceptance: 210 seconds, 42 samples, no worker restarts,
unavailable endpoints or media failures; maximum observed encoder report age
7.2 seconds. A separate bounded diagnostic connected to Upper Garage's private
tier-2 feed three times, read/discarded 138–161 KiB each time and closed each
consumer. All three start/stop cycles completed (about 6.3 seconds to first
sample per cycle), with no shared-agent stall. No media was persisted. This
crossed the 175-second failure point in the preceding failed check. A fresh
Apple Home close/reopen check was requested separately.
