# Bounded CPU profiling evidence

Read [the full report](REPORT.md) for results, attribution limits and the single
recommended follow-up. This is a historical evidence bundle, not a runtime fix.

## Provenance

Captured on 2026-09-05 from SurvNG PID 651962, with deployed revision inferred as
`8177935464ed3c398cad01082c7dfab47b2d2e05`. Originals are preserved under
`/root/survng-measurements/cpu-profile-20260905T203530Z/` on the measurement host.
The report and captured artifacts are copied without changing their contents.
The report's statement that no GitHub write occurred describes the investigation;
this documentation PR was separately requested afterward.

## Contents

- `REPORT.md`: complete Astra-reviewed findings and limitations.
- `pyspy-native.speedscope.json`: raw native-stack profile; open with a compatible
  local Speedscope viewer. No local-variable or subprocess capture was enabled.
- `cpu.jsonl`: independent timestamped process, thread and cgroup samples.
- `metadata.jsonl`, `completion.jsonl`, `status-before.jsonl`,
  `status-after.jsonl`: deployment/configuration identity and allowlisted runtime
  snapshots, with capture-time redaction. Persisted settings are explicitly not
  a substitute for effective runtime settings.
- `cpu-summary.json`, `profile-analysis.json`: derived CPU and stack attribution.
- `profile-window.json`, `pyspy-record.log`, `py-spy-record-help.txt`,
  `profiler-packages.txt`: exact invocation, timing bounds, tool errors and version.
- `collect_cpu.py`, `profile_run.py`, `summarize_cpu.py`, `analyze_profile.py`:
  unchanged scripts used for this investigation.
- `support/collector.py`: unchanged snapshot of the shared helper originally at
  `/root/survng-measurements/pre-deployment/collector.py`.
- `SHA256SUMS`: integrity manifest for the archived evidence and script dependency.

The isolated profiler virtual environment/binary, stale collector PID file and
empty collector log are intentionally excluded. No recordings, images, model
artifacts, full environment dumps or production configuration files are included.
For attribution and auditability, the evidence retains operational metadata:
camera IDs/labels (including person-like labels), thread names, local paths,
deployment/capture timestamps, effective settings and captured activity/state.
These are not anonymized; credentials, tokens and stream URLs are excluded.

## Reproduction limits and safety

Scripts retain their original host-specific assumptions for audit fidelity; do
not run the collection/attachment scripts against a live service without a new
authorization. In particular, `profile_run.py` targets the original PID, expects
the omitted local profiler environment and overwrites output files.
`collect_cpu.py` imports the original sibling `pre-deployment/collector.py`, not
the archived `support/collector.py` path. `analyze_profile.py` reads source from
`/root/SurvNG`; reproducing its stage classification requires the recorded source
revision, not an arbitrary later checkout. Analyses write derived files beside
the scripts, so preserve this evidence and work on a scratch copy if rerunning.

Verify archived bytes from this directory with `sha256sum -c SHA256SUMS`.
Packaging validation checks JSON/JSONL structure, profile sample/frame references,
script syntax and file integrity only. No new profiling, replay, inference or unit
test workload was run to produce this PR.
