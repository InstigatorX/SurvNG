# Smart Search model packages

SurvNG Smart Search finds object incidents from a visual description such as
`person in a red jacket` or `white delivery truck`. Images and embeddings stay
on the SurvNG host. Smart Search does not upload camera images to the configured
AI assistant provider.

## Recommended starting model

Use **MobileCLIP2-B exported to OpenVINO FP16**. It offers a practical accuracy,
memory, and throughput balance for an Intel GPU. SurvNG deliberately keeps the
encoder interface and every stored generation versioned; a future MobileCLIP2,
SigLIP2, or other dual-encoder package can build a new index without deleting
or mixing the previous generation.

Smart Search is disabled by default. Put a self-contained model package on a
local filesystem, then enable it under **Admin → Object Detection → Smart
Search**. Docker installations should mount the package read-only beneath
`/config/models`.

## Package layout

```text
mobileclip2-b-openvino/
├── semantic_model.json
├── image_encoder.xml
├── image_encoder.bin
├── text_encoder.xml
├── text_encoder.bin
└── tokenizer/
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── special_tokens_map.json
```

Tokenizer files must be complete enough for
`transformers.AutoTokenizer.from_pretrained(..., local_files_only=True)`. The
runtime never downloads missing model or tokenizer files.

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
    "size": 256,
    "mean": [0.48145466, 0.4578275, 0.40821073],
    "std": [0.26862954, 0.26130258, 0.27577711]
  },
  "text": {
    "tokenizer_path": "tokenizer",
    "max_length": 77,
    "inputs": {
      "input_ids": "input_ids",
      "attention_mask": "attention_mask"
    },
    "output": "text_features"
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
camera/event handling never waits for an embedding. The status endpoint is
`GET /api/semantic-search/status`; search is available through
`POST /api/semantic-search`, the Recordings **Smart Search** page, and the
SurvNG Assistant.

Model files and preprocessing settings produce separate fingerprints. Changing
the model starts a new generation in the same SQLite index, so incompatible
embedding spaces are never compared. Old generations can remain available
during migration.

Semantic similarity is a ranking signal, not a probability or identity proof.
Confirm important results from the linked incident or recording.
