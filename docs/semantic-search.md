# Smart Search model packages

SurvNG Smart Search finds object incidents from a visual description such as
`person in a red jacket` or `white delivery truck`. Images and embeddings stay
on the SurvNG host. Smart Search does not upload camera images to the configured
AI assistant provider.

## Model choices

Use **MobileCLIP2-B exported to OpenVINO FP16**. It offers a practical accuracy,
memory, and throughput balance for an Intel GPU. SurvNG deliberately keeps the
encoder interface and every stored generation versioned; a future MobileCLIP2,
SigLIP2, or other dual-encoder package can build a new index without deleting
or mixing the previous generation.

For stronger natural-language and visual-attribute matching, SurvNG also ships
an exporter for Google's official **SigLIP2 Base Patch16 224** checkpoint. Its
OpenVINO package is about 726 MB and produces 768-dimensional embeddings, so it
uses more disk, accelerator memory, and indexing time than MobileCLIP2-B. Build
and evaluate it as a separate generation before switching production search.

Smart Search is disabled by default. Put a self-contained model package on a
local filesystem, then enable it under **Admin → Object Detection → Smart
Search**. Docker installations should mount the package read-only beneath
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

## Build SigLIP2 Base

```bash
.venv/bin/pip install -r requirements-semantic-export.txt
.venv/bin/python scripts/export-siglip2-openvino.py \
  --output models/siglip2-base-patch16-224-openvino-fp16 \
  --cache-dir models/.siglip2-download-cache
```

The exporter downloads `google/siglip2-base-patch16-224`, creates isolated
image and text OpenVINO FP16 encoders, bundles its tokenizer and Apache-2.0
license, and validates source/OpenVINO embedding parity plus exact runtime
preprocessing and tokenization before retaining the package.

The model's official fixed-resolution processor is represented explicitly in
the manifest. Other SigLIP2 checkpoints that use aspect-aware packed patches
are also supported by the runtime contract. Production uses only OpenVINO,
Pillow, and the lightweight `tokenizers` package; PyTorch and Transformers are
export-only dependencies.

Do not point the active configuration at this package until comparison is
complete. When selected later, use the generic `openvino_manifest`
implementation and the package's absolute model path. Its fingerprint and
768-dimensional space automatically create a separate index generation.

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
