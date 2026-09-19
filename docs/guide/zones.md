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
| **Exclude from motion wake-up** | Prevent movement inside this shape from waking or keeping adaptive inference active |

Motion exclusion is independent of object behavior. For a clock overlay, draw a zone around it, select **No object effect** (or **Ignore** if object incidents should also be suppressed), and enable **Exclude from motion wake-up**. Exclusions override incident-zone approach margins. Movement crossing a boundary can still wake inference where it extends into an eligible, unexcluded area. Recording and AI object detection are unchanged.

Existing saved motion exclusions retain their values. The configuration key remains `exclude_from_ema` for compatibility, but the control now applies to native motion wake-up. Motion rectangles can extend beyond the pixels that moved, so allow some room around a clock or other nuisance source.

## Practical examples

### Package porch

Draw a polygon on the porch floor and railing opening. Prefer `person` (and maybe `package` if your model has it). Ignore the sidewalk beyond the steps if strangers walking by are not your concern.

### Driveway gate

Watch the gate apron, not the entire street. Headlights and opposite-lane traffic create motion that is rarely useful.

### Tree problem

If a tree fills half the frame and constantly wakes adaptive inference, either:

- exclude that region from motion wake-up, or
- shrink the incident zone away from the foliage

## Tips

- Start with one simple zone per camera.
- Revisit zones after seasonal foliage changes.
- Zone geometry is independent from Live view framing/cropping.

## Related

- [Cameras](cameras.md)
- [Motion & detection](motion-detection.md)
- [Motion triggers and validation](../adaptive-motion.md)
