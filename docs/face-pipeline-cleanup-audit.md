# Old face pipeline: cleanup audit (report only)

2026-11-25, on branch `claude/face-binding`. Nothing is removed yet; this lists
what the previous face pipeline left, who still uses it, and a proposed order.

The previous pipeline is off (`detector.face_recognition_enabled` is false), its
data was reset on 2026-11-24, and face capture (`survng/app/face_capture/`)
replaces it: YuNet detection, AdaFace embeddings, grouping, the new People page
and faces on incidents. About 9,000 lines remain from the old pipeline:
`face_store/` (4,260 lines) plus about 4,700 in the modules below, and 16 test
files.

## What remains and who uses it

| Part | Lines | Used by today | Notes |
|---|---|---|---|
| `face_store/` (old store, recognition queue, unknown clusters, benchmarks) | 4,260 | `faces.py`, `face_evaluation.py`, `face_routes.py`, `inference_lifecycle` (face queue) | Empty tables in `survng.sqlite3` (`face_people`, `face_observations`, …) |
| `faces.py` | 31 | `inference_lifecycle` | Facade for the old store |
| `face_routes.py` (`/api/faces/*`, `/api/people/visits`) | 529 | `main.py`; frontend: `FacesPage`, `VisitsPanel`, `peopleWorkspace.mjs`, the incident list's face review | |
| `face_detection_routes.py`, `face_detection_dataset.py`, `face_detection_evaluation.py` | 1,702 | `main.py` only | The old People page's detector-labelling; no screen mounts it now |
| `face_evaluation.py` | 219 | nothing | Offline tool for the old recognizer |
| `face_recognition.py` (`OpenVinoFaceRecognizer`) | 268 | face worker (`process.py`), `face_store`, `face_frame_search` | Face capture does not use it |
| `face_detection.py` (retail-0004 detector) | 153 | face worker, `face_detection_routes`, `face_frame_search` | Face capture uses YuNet |
| `face_candidates.py` | 292 | refinement (`object_detection`, `decision_handler`), `face_capture/pipeline` (`_padded_crop` only), `face_frame_search` | Runs only with `face_recognition_enabled` |
| Refinement face evidence (`_refine_face_evidence`, `_detect_faces_in_people`, `face_search_*` counters) in `motion_pipeline/object_detection.py`; `_persist_face_candidates` in `decision_handler.py` | about 250 | incident refinement, when the old pipeline is on | The `face_search_*` counts land in the timing averages (review note) |
| Face worker: recognizer and detector in `inference_runtime/process.py`; `embed`, `detect_faces` in the supervisor; `IsolatedFaceRecognizer` | about 120 | `inference_lifecycle` | The worker also hosts the capture engine, which stays |
| `identity_projection.py`, `person_visits.py`, `unknown_identity.py` | 644 | `incident_queries`, `semantic_routes`, `manager` | Feed `incident.identities`, visits and search names from the old store; now empty |
| Frontend: `FacesPage.jsx`, `FaceFindingPanel.jsx`, `VisitsPanel.jsx`, `peopleWorkspace.mjs` | 862 | `FacesPage` only for `FaceReviewDialog`, imported by `IncidentsPage`; the rest only by tests | The People tab mounts `PeoplePage` now |
| Incident list: "Faces" box in the inspector (`IncidentCard`), face review dialog (`IncidentsPage`) | about 60 | always shown | Always says "No recognized faces" now |
| Config fields `face_recognition_enabled`, `face_embedding_*`, `face_landmark_model_path`, `face_detection_*`, `face_match_threshold`, `face_unknown_cluster_threshold`, `face_auto_identify_*`, `face_min_size`, `face_max_references`, `face_max_observations`, `face_enrich_max_people`, `face_evidence_*` | | the old pipeline; Settings › Faces › "Previous face detection" | Follow the existing rule for retired fields: stop reading them, keep loading files that contain them |
| Data: empty face tables in `survng.sqlite3`, `face-detection-dataset.sqlite3`, `face-detection-crops/`, `image-cache/faces/` | | nothing | Empty since the reset |
| Tools: `face_funnel` (reads the old counters), `face_frame_search` (old detector, comparison runs) | | nobody at runtime | Keep `face_frame_search` as the evidence tool, or retire both |

## Proposed order

**1. Screens and routes nobody reaches (low risk).**
- Remove the old People components (`FacesPage`, `FaceFindingPanel`, `VisitsPanel`, `peopleWorkspace.mjs`) with their tests.
- Remove the incident inspector's "Faces" box and the face review dialog.
- Remove the detector-labelling backend (`face_detection_routes`, `_dataset`, `_evaluation`) and `face_evaluation.py`.
- Remove `face_routes.py`.

**2. The old pipeline's runtime path (medium risk, touches refinement and the face worker).**
- Remove refinement face evidence and `_persist_face_candidates`.
- Remove the old store (`face_store/`, `faces.py`) and the face queue in `inference_lifecycle`.
- Remove the recognizer and retail detector from the face worker, the supervisor's `embed`/`detect_faces`, and `IsolatedFaceRecognizer`.
- Move `_padded_crop` into `face_capture`.
- Retire the config fields and the "Previous face detection" card.
- Drop the empty tables and files.
- Check this with the `lifecycle` and `recording` suites of `scripts/verify-architecture.sh`.

**3. Decide, rather than delete: identities on incidents, visits and search.**
`identity_projection`, `person_visits` and `unknown_identity` are where names reach `incident.identities`, notifications and semantic search. Two choices:
- **Recommended:** feed them from face capture, so names flow to everything that already reads identities.
- **Alternatively:** remove them, and have those consumers read face capture directly.

Step 1 can happen at any time. Step 2 is best after the 14-day seed has run, in case anything there turns out to be needed. Step 3 needs a decision first.
