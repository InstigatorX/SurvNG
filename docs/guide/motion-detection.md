# Motion & detection

SurvNG separates two jobs:

1. **Notice that something changed** (motion)
2. **Decide what it was** (object detection)

You can run recording without detection. Detection without sensible motion settings usually wastes effort on empty scenes.

## Motion behaviors (guided choices)

Under **Admin → Cameras → Motion/Object** you pick one complete behavior:

| Choice | Plain meaning | Good when |
| --- | --- | --- |
| **Camera only** | Trust every ordinary camera motion notice | Camera notices are excellent and you want minimum extra CPU |
| **Camera + EMA validation** | Camera notices still start work; SurvNG double-checks the picture | Notices are frequent but sometimes noisy |
| **Camera + EMA backup** (default) | Camera notices are primary; SurvNG can still rescue a miss | Typical home/business cameras |
| **EMA only** | SurvNG watches the video itself for motion | ONVIF notices are unavailable or unreliable |

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

## Object tracking

When tracking is enabled, SurvNG can follow an object across frames inside an incident. That improves cover selection and review overlays. Tracking settings live with detection configuration.

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
