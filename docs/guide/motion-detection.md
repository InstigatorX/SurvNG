# Motion & detection

SurvNG separates two jobs:

1. **Notice that something changed** (motion)
2. **Decide what it was** (object detection)

You can run recording without detection. Detection without sensible motion settings usually wastes effort on empty scenes.

## Motion behaviors (guided choices)

Under **Admin → Cameras → Motion/Object** you pick one complete behavior:

| Choice | Meaning | Cost | Good when |
| --- | --- | --- | --- |
| **Camera only** | Trust every ordinary camera motion notice | Lowest CPU | Camera notices are excellent |
| **Camera + EMA validation** | Camera notices still start work; SurvNG double-checks the picture | Continuous EMA plus a short wait | Notices are frequent but sometimes noisy |
| **Camera + EMA backup** (default) | Camera notices are primary; SurvNG can still rescue a miss | Continuous EMA for recall, not for fewer false positives | Typical home/business cameras |
| **EMA only** | SurvNG watches the video itself for motion | Continuous EMA is the only automatic trigger | ONVIF notices are unavailable or unreliable |

**EMA** means SurvNG’s enhanced motion analysis on the live picture. It learns a quiet baseline for the scene and looks for lasting, credible change — not every leaf flicker.

Full technical flow: [Motion triggers and validation](../adaptive-motion.md).

## Object detection

When motion qualifies, SurvNG can run an object detector on recent frames.

### What you configure

- Enable/disable detection
- Model files and labels
- How confident a label must be
- How many frames must agree before an incident is kept
- Optional per-label overrides (for example, require a lawn robot to be clearer than a person)

### Example

You care about people at a side gate but not distant sidewalk traffic:

1. Enable detection with a model that includes `person`.
2. Draw an **Incident** zone on the gate area.
3. Keep global confidence near the default.
4. Review **Incidents** after walking the gate path.
5. If windy bushes create junk, tighten zones or raise sensitivity carefully — do not jump straight to extreme thresholds.

## Camera reports and model detections

Some cameras report only motion. Others report a person, vehicle, animal or face in an ONVIF topic or a supported object-type field. SurvNG normalizes these reports across camera brands and keeps them separate from its own object detections.

| Camera report | Meaning | Possible configured model classes |
| --- | --- | --- |
| People detection | Camera reported a person | `person` |
| Vehicle detection | Camera reported a vehicle | `car`, `truck`, and other recognized vehicle labels present in the model |
| Dog/cat detection | Camera reported a dog or cat without distinguishing them | `dog`, `cat` |
| Explicit dog object type | Camera reported a dog | `dog` |

The candidate list contains compatible model labels, not detected objects. An empty list means no compatible model label was resolved; it does not invalidate the camera's report. Unrecognized custom labels are not guessed.

SurvNG still checks every enabled model class and applies its usual confidence, confirmation and incident eligibility rules. For example, a camera animal report can lead to a SurvNG detection of both a dog and a person. Camera reports never populate object badges, object filters, training labels or model-confirmed MQTT classes by themselves.

Incident details show camera reports separately from model detections. The reports travel with stored event qualification, durable detection work and incident API/MQTT data. Existing events without this metadata remain readable and show no inferred camera report.

For API and MQTT consumers, `camera_semantics.reports` contains the original `topic`, normalized `category`, optional `reported_class`, and advisory `candidate_model_classes`. Grouped incident reports also identify their `source_event_id` and `source_created_at`. Continue to use the existing object/class fields for SurvNG detections. No database migration or camera-specific setting is required.

This is support for recognized message formats, not automatic understanding of every vendor's analytics. Unsupported payload fields and unknown topics do not gain semantic priority. Line-crossing and intrusion describe behavior and are not automatically treated as object classes.

## Object tracking

When tracking is enabled, SurvNG can follow an object across frames **after** recorded confirmation. That improves cover selection and review overlays. It does not decide whether the incident is kept, and it spends extra detector time, so it starts only once refinement has finished (or could not run). Tracking settings live with detection configuration.

## Motion Audit

**Admin → Audit** shows samples where SurvNG looked and decided not to create an incident. Use it to learn whether you are too strict, too loose, or missing a zone.

## Detection Tune-Up

**Admin → Tune-Up** runs a guided review of historical evidence and suggests bounded setting changes. SurvNG can monitor the effect afterward. You confirm before anything is applied.

## Camera Advisor

**Admin → Camera Advisor** reviews a balanced sample of one camera’s recent outcomes and may recommend camera-scoped motion adjustments. Applying still requires your confirmation.

## Related

- [Zones](zones.md)
- [Incidents](incidents.md)
- [Admin](admin.md)
- [AI assistant](assistant.md)
