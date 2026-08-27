# Forensic visual search

Click an object in one incident frame and find historically similar
appearances across indexed evidence, then open the matching timeline or clip.

This is **ranked visual similarity**, not forensic proof of identity. The UI
and APIs must keep that distinction explicit (`visual_similarity`,
`visually_similar`, match strength labels).

## Product promise

| Entry | Retrieval | Best for |
| --- | --- | --- |
| Text description | MobileCLIP text → image/crop index | Open-world cues (`red backpack`) |
| Click object crop | MobileCLIP image → image/crop index | Arbitrary detected classes |
| Click person/vehicle | Appearance/ReID index (optional CLIP broaden) | Instance continuity |

Indexes today cover **incident snapshots and object crops**, not every frame of
continuous recordings. “Every historical clip” means every matching indexed
incident that can resolve to a recording window.

## Phases

### Phase 0 — Contracts

- Query modes: `text`, `visual`, later `appearance` / `hybrid`
- Result shape: event + matched evidence (`source_kind`, `bbox`, `object_label`,
  score, strength) + timeline deep link with `camera`, `at`, and `event`
- Feature gates: Smart Search enabled/ready; ReID for appearance path
- Search backend interface stays swappable (SQLite+NumPy now; ANN later)

### Phase 1 — Selectable objects

- Object boxes expose index / label / optional track id
- Incident snapshot and inspector support selection
- **Find similar** action when visual search or ReID is available

### Phase 2 — Visual (image-query) Smart Search

- `POST /api/semantic-search/visual` with `event_id` + `object_index`
- Crop-from-event only for MVP (no arbitrary upload)
- Reuse `encode_images` + `SemanticIndex.search`
- Optional `source_kinds` filter; exclude anchor event by default

### Phase 3 — Find-similar UI

- Results panel on Incidents (and Search when useful)
- Match strength, camera/time, incident + timeline links with `event=`

### Phase 4 — Object-anchored appearance ✅

- Appearance matches accept `track_id`
- Person/vehicle selection prefers ReID, then broadens with CLIP visual search
- Track id resolved from the object or a unique matching stored track

### Phase 5 — Hybrid funnel ✅ (Find similar)

- One Find similar panel tags each hit with `query_mode`: `appearance` or `visual`
- Appearance hits rank first; visual hits fill remaining slots without score fusion
- Text Smart Search remains the separate open-world entry point

### Phase 6 — Timeline forensic walk ✅

- Find similar Timeline links carry `trail=` ordered event IDs
- Session stores hit metadata for camera/time/mode
- Selected-incident rail shows Find similar prev/next and seeks across cameras/days

### Phase 7 — Recording-frame anchor

- Select/draw on a timeline frame; encode crop through the same visual path

### Phase 8 — Hardening

- Crop thumbnails, ranking weights, backfill status, eval harness for image queries
- Measure SQLite scan limits before introducing ANN

### Phase 9 — Stretch

- Dense/sparse embeddings of continuous recordings
- FAISS / Qdrant (or similar) as a derived ANN index
- External one-shot detectors only if CLIP crops prove insufficient

## API sketch (MVP)

### Visual search

`POST /api/semantic-search/visual`

```json
{
  "event_id": 123,
  "object_index": 0,
  "camera_ids": [],
  "object_labels": [],
  "start_at": "",
  "end_at": "",
  "limit": 50,
  "minimum_score": -1.0,
  "source_kinds": [],
  "exclude_anchor": true
}
```

`object_index` is the index into snapshot-visible labeled detections (same
ordering as semantic indexing / `semantic_event_objects`).

### Appearance (track-aware)

`GET /api/events/{event_id}/appearance-matches?track_id=2&hours=24&limit=12`

## Vector databases (later)

Current semantic and appearance indexes store embeddings in SQLite and score a
bounded NumPy candidate set. That fits incident-scale search.

A vector database enhances **scale, broad-query recall, and dense-archive
ambition** — not detector quality or timeline UX:

| Option | Fit for SurvNG |
| --- | --- |
| Keep SQLite+NumPy | MVP through Phase 6; simplest ops |
| FAISS (in-process) | First ANN step; rebuildable sidecar |
| Qdrant | Filtered ANN + payloads if Compose ops are acceptable |
| Milvus | Usually overkill for single-host NVR |

Keep SQLite as system of record for events and media paths. Treat ANN as a
derived index keyed by model generation fingerprints, cut over on backfill.

Introduce ANN when candidate caps truncate recall, p95 search latency rises, or
Phase 9 dense frame indexing is committed.

## Ship cuts

| Cut | Phases | Value |
| --- | --- | --- |
| MVP | 0–3 | Click any detected object → similar incidents |
| Forensic | 4–5 | Person/vehicle appearance trail + visual broaden |
| Timeline | 6 | Multi-hit timeline walk |
| Full | +7–8 | Scrub-to-search + production hardening |
| Research | 9 | Archive-wide dense search |

## Related

- [Smart Search model packages](semantic-search.md)
- [Search guide](guide/search.md)
- [Timeline guide](guide/timeline.md)
