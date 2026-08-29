# Smart Search model packages

SurvNG Smart Search finds object incidents from a visual description such as
`person in a red jacket` or `white delivery truck`. Images and embeddings stay
on the SurvNG host. Smart Search does not upload camera images to the configured
AI assistant provider.

## Model choices

Use **MobileCLIP2-B exported to OpenVINO FP16**. It offers a practical accuracy,
memory, and throughput balance for an Intel GPU. SurvNG deliberately keeps the
encoder interface and every stored generation versioned; a future compatible
dual-encoder package can build a new index without deleting or mixing the
previous generation.

Smart Search is disabled by default. Put a self-contained model package on a
local filesystem, then enable it under **Admin → Detection → Smart Search**.
Docker installations should mount the package read-only beneath
`/config/models`.

## Build the official Apple model

SurvNG includes a one-command exporter for Apple's official MobileCLIP2-B
checkpoint. The export dependencies are needed only while building the model
package; SurvNG's server does not load PyTorch or OpenCLIP.

```bash
.venv/bin/pip install -r requirements-semantic-export.txt
.venv/bin/python scripts/export-mobileclip2-openvino.py \
  --output models/mobileclip2-b-openvino-fp16 \
  --cache-dir models/.mobileclip2-download-cache
```

The exporter downloads `mobileclip2_b.pt` from `apple/MobileCLIP2-B`, applies
Apple's required inference reparameterization, creates dynamic-batch FP16
OpenVINO image and text encoders, bundles the matching OpenCLIP BPE vocabulary,
and compares both OpenVINO encoders with the PyTorch source. It refuses to keep
a package if either normalized embedding falls below the required cosine
parity. Use `--force` to intentionally replace an existing package, or
`--checkpoint /path/to/mobileclip2_b.pt` to use an already downloaded file.

The resulting package is about 288 MB. A systemd installation in
`/root/SurvNG` should configure the absolute path as
`/root/SurvNG/models/mobileclip2-b-openvino-fp16`. Docker should mount that
host directory read-only at `/config/models/mobileclip2-b-openvino-fp16` and
use the container path. Select `GPU` on an Intel GPU host, then enable Smart
Search. Keep the included `LICENSE` with the package.

## Repeatable ranking benchmark

Create a reviewed benchmark from real incident IDs using
[`semantic-search-benchmark.example.json`](semantic-search-benchmark.example.json),
then run:

```bash
.venv/bin/python scripts/evaluate-semantic-search.py benchmark.json \
  --base-url http://127.0.0.1:8088/survng \
  --output mobileclip2-baseline.json
```

The report records result IDs and scores, request latency, precision and recall
at five and ten, and reciprocal rank. Queries marked `judged: false` still
produce candidates for review but are excluded from quality claims. Re-run the
identical benchmark after a candidate generation is indexed.

## Build and evaluate a comparison generation

The offline builder writes only the candidate package's fingerprinted
generation. It does not change `config.json`, restart SurvNG, or alter which
generation serves live searches. The command is safe to interrupt and resume;
already-complete full-frame and crop evidence is skipped.

```bash
.venv/bin/python scripts/build-semantic-index.py \
  --database runtime/database/survng.sqlite3 \
  --storage-dir /path/to/survng/storage \
  --model-dir /path/to/candidate-openvino-package \
  --device GPU \
  --pause-seconds 0.03 \
  --report candidate-index-report.json
```

Only one offline comparison build may use a database at a time. Progress and
the final report distinguish events scanned from events actually encoded and
record parent, inference-worker, and combined memory high-water marks. Live
events continue to be indexed by the active model generation independently.

Evaluate the candidate directly from its local generation, without activating
it in the application:

```bash
.venv/bin/python scripts/evaluate-semantic-search.py benchmark.json \
  --model-dir /path/to/candidate-openvino-package \
  --database runtime/database/survng.sqlite3 \
  --device GPU \
  --output candidate-comparison.json
```

The evaluator de-duplicates crop and whole-frame matches to the best result per
incident. Quality metrics are emitted only for queries whose relevant incident
IDs have been reviewed; unjudged queries remain useful candidate lists but are
not treated as accuracy evidence.

## Package layout

```text
mobileclip2-b-openvino/
├── semantic_model.json
├── VALIDATION.json
├── LICENSE
├── image_encoder.xml
├── image_encoder.bin
├── text_encoder.xml
├── text_encoder.bin
└── tokenizer/
    └── bpe_simple_vocab_16e6.txt.gz
```

The runtime tokenizer is a small NumPy implementation of OpenCLIP's BPE format.
It never downloads missing model or tokenizer files and does not import
PyTorch.

Example `semantic_model.json`:

```json
{
  "implementation": "mobileclip2_openvino",
  "dimensions": 512,
  "image_model": "image_encoder.xml",
  "text_model": "text_encoder.xml",
  "image": {
    "input": "pixel_values",
    "output": "image_features",
    "size": 224,
    "interpolation": "bicubic",
    "resize_mode": "shortest_center_crop",
    "mean": [0.0, 0.0, 0.0],
    "std": [1.0, 1.0, 1.0]
  },
  "text": {
    "input": "input_ids",
    "output": "text_features",
    "tokenizer_kind": "openclip_bpe",
    "tokenizer_path": "tokenizer/bpe_simple_vocab_16e6.txt.gz",
    "max_length": 77
  }
}
```

Input and output names must match the exported OpenVINO models. All paths are
resolved inside the package; SurvNG rejects manifest paths that escape it.

## Index behavior

SurvNG indexes object incidents rather than every continuous-recording frame.
It stores one whole-scene embedding and, when available, embeddings for each
detected object crop. This bounds GPU, CPU, and database growth while preserving
scene context and detailed appearance searches.

Indexing runs asynchronously after camera startup. Historical incidents are
processed in bounded batches, already-complete evidence is skipped, and live
camera/event handling never waits for an embedding. New incidents outrank
historical work, reserved queue capacity prevents backfill from crowding them
out, and configurable pacing limits sustained accelerator load. The status endpoint is
`GET /api/semantic-search/status`; search is available through
`POST /api/semantic-search`, the Recordings **Smart Search** page, and the
SurvNG Assistant.

Model files and preprocessing settings produce separate fingerprints. Changing
the model starts a new generation in the same SQLite index, so incompatible
embedding spaces are never compared. Old generations can remain available
during migration.

Semantic similarity is a ranking signal, not a probability or identity proof.
Confirm important results from the linked incident or recording.
