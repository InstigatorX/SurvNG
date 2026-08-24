# Zones

Zones tell SurvNG which parts of a camera picture matter.

Without zones, every part of the frame is treated the same. With zones, you can watch a doorway while ignoring a busy street edge or a tree that constantly moves.

## Draw a zone

1. Open **Admin → Cameras → Zones** for the camera.
2. Use a snapshot of the scene as your canvas.
3. Draw a polygon around the area of interest.
4. Choose what the zone should do for objects and for motion analysis.
5. Save.

## Common zone intents

| Intent | Typical use |
| --- | --- |
| **Incident** | Create incidents when matching objects appear here |
| **Ignore** | Suppress object incidents in this region |
| **No object effect** | Shape exists for motion exclusion only |
| **Exclude from EMA** | Do not let motion inside this shape drive visual triggers |

You can combine ideas. Example: an Incident zone on a porch that also excludes EMA for a fluttering flag in the corner of that same polygon — depending on the controls you select for that zone.

## Practical examples

### Package porch

Draw a polygon on the porch floor and railing opening. Prefer `person` (and maybe `package` if your model has it). Ignore the sidewalk beyond the steps if strangers walking by are not your concern.

### Driveway gate

Watch the gate apron, not the entire street. Headlights and opposite-lane traffic create motion that is rarely useful.

### Tree problem

If a tree fills half the frame and constantly triggers camera notices, either:

- exclude that region from EMA, or
- shrink the incident zone away from the foliage

## Tips

- Start with one simple zone per camera.
- Revisit zones after seasonal foliage changes.
- Zone geometry is independent from Live view framing/cropping.

## Related

- [Cameras](cameras.md)
- [Motion & detection](motion-detection.md)
- [Motion triggers and validation](../adaptive-motion.md)
