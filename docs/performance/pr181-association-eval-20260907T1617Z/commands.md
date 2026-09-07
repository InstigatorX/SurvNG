# Reproduction and execution record

All work used the existing service interpreter read-only. No package installation,
production checkout change, or application startup import was used. Original
blocked-run artifacts in the parent directory remain unchanged.

The scratch checkout is an archive of commit
`67e67ad73b7dc950dd55f15cabd4cb6d8f35531d`, not a checkout over production.
The only source-tree addition is `tests/__init__.py`, a scratch-only docstring
package marker preventing the installed `tests` package from shadowing source.

## Regression gate

From the isolated checkout, using private writable scratch:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
TMPDIR=SCRATCH/private XDG_CACHE_HOME=SCRATCH/private \
timeout 180 nice -n 10 ionice -c 3 SERVICE_VENV/bin/python \
  -m pytest -p no:cacheprovider \
  tests/test_tracking_multicue.py tests/test_tracking_candidate.py -q
```

Result: exit 0, 64 passed and 64 subtests passed, 0.47s.
Both the original failed log and rerun log are in the archive.

## Capture record, not permission to recapture

`api_session.py` was run in a private terminal. It disabled terminal echo,
accepted the operator token on stdin, and held it in memory only. It admitted
only allowlisted GETs and corpus-listed POSTs after read-only health/history
checks. Session ended with `quit` after all six captures. No token is included.

Each selected event was captured once, sequentially:

```text
POST /survng/api/events/EVENT/tracking-comparison?duration_seconds=30&sampling_profile=fixed_2fps
EVENT order: 63156, 63346, 64897, 65548, 64911, 64006
New comparison IDs: 77, 78, 79, 80, 81, 82
```

Do not rerun these POSTs: they would now overwrite comparisons and clear verdicts.
`api_session.py` refuses existing event comparison rows. Capture/cache writes
were performed solely through the authorized service endpoint. Full responses
and checksummed replay objects are in private local files, excluded from sharing.

`media_check.py` and `selection_extra.py` perform bounded read-only recording
index queries and sparse local frame extraction; `lock_corpus.py` froze the
six actual child IDs and profile plan before any result was inspected. Their
host-specific paths are sanitized to placeholders in the ZIP; they are execution
records, not portable replacements for deployment discovery.

## Offline reproduction without service access

Use a new isolated checkout of the exact SHA, with imports resolving there and
the same read-only dependencies. Do not import application startup, instantiate
a detector, recapture inputs, or change OpenCV threads. Raw replays must be
provided privately. Run from that checkout (replace placeholders):

```sh
PYTHONDONTWRITEBYTECODE=1 timeout 240 nice -n 10 ionice -c 3 \
  SERVICE_VENV/bin/python /PATH/TO/offline_reproduce.py PRIVATE_INPUT_DIR NEW_OUTPUT_DIR
```

The wrapper executes the existing module, once per planned profile:

```sh
SERVICE_VENV/bin/python -m survng.app.tracking_evaluation \
  capture-EVENT.replay.json --association-cues --profile fixed_2fps --output OUTPUT.json
```

Profiles: fixed_2fps and fixed_075fps for all six; sparse_gaps only for 64897.
The immutable raw captures have 90 frames each; the profile implementations
select identical timestamped observations for both engines. Each output is
checked for exactly the two requested engines and individual engine errors.

`run_pairs.py` records validation, command return codes, selected timestamps,
replay/source digests and result hashes. `analyze_pairs.py` produces the tables
and whole-track agreement records; it creates no identity labels. The analyzer
uses the repository's existing assignment algorithm, not a new evaluation
framework. There were no disagreement episodes to overlay.

`timing_repeat.py` handles only 63156/fixed_2fps: one warmup, then three alternating
engine-order repetitions on the same saved input. No inference is involved.
The original results remain untouched; repetitions have separate filenames.

`finalize.py` is a host-specific packaging/evidence script; it intentionally
requires original private files and verifies the original blocked archive hash.
It is not a general standalone collector. The shareable ZIP excludes private
inputs, embeddings, full configurations/databases/recordings and credentials.
