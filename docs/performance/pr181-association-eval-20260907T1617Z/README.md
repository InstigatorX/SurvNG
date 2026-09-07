# PR #181 association evaluation results

Start with [the full report](report.md), [summary CSV](summary.csv), or
[machine-readable summary](summary.json). This evidence-only publication does
not merge or enable PR #181 and contains no production code changes.

## Outcome

- 13 of 14 requested paired evaluations completed across six incidents and four
  cameras. The second suitable sparse-gap case was unavailable.
- Confirmed frame observations and final track summaries matched exactly in all
  13 pairs. Cue scores changed in seven profiles across three incidents without
  changing confirmed associations.
- One bounded timing repeat found medians of 0.167 ms/frame (baseline) and
  0.185 ms/frame (candidate), a small selected-case overhead—not fleet CPU impact.
- Identity accuracy, true identity switches and true false merges were not measured.
- The candidate should remain offline; the evidence does not justify promotion.

The candidate and baseline classes were both evaluated from exact revision
`67e67ad73b7dc950dd55f15cabd4cb6d8f35531d`. The required isolated regression tests
passed after an approved scratch-only test-package import correction: 64 tests
and 64 subtests. No tracker logic or assertions were changed.

## Evidence and reproduction

- [Manifest](manifest.json): revisions, selection rationale, sampling provenance,
  digests, effective-setting allowlist, coverage limitations and side effects.
- [Individual results](results/): paired outputs, whole-track agreement records,
  and the bounded timing repetitions.
- [Timing summary](timing-summary.json), [resource summary](resource-summary.json),
  [preservation evidence](preservation.json), and [execution record](executions.json).
- [Commands and scripts](commands.md): reproduction instructions and the
  distinction between portable offline replay and host-specific capture records.
- [Review notes](review/README.md): no disagreement overlays were manufactured.
- [Original sanitized results archive](shareable-results.zip): all 65 originally
  exported evidence files, preserved byte-for-byte. This README is a publication
  index added outside the archive.

Archive SHA-256:

```text
9bb9a6c9f8cb3097cade55271e7bb69c331ed4b2d1e76c64d2e8d85522a612e1
```

The archive and extracted files exclude raw replay embeddings, full configs,
credentials, databases, models, recordings and selection images. Full raw replay
inputs remain private on the server; they must be supplied privately for replay.
Publishing the report does not authorize recapturing incidents or overwriting
existing Compare history. Existing comparison rows/verdicts were preserved;
the measurement added only six authorized comparison rows and normal clip caches.

Statements in the report about no GitHub publication describe the measurement
run itself. This documentation PR publishes its sanitized evidence afterward
at the user's explicit request. The deployed checkout and service are not used
as the publication worktree.
