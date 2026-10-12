# People

**People** labels the upper-body crop this installation's face detector already sees.

A label is a face box, a crop with no face, a region that is not a face, or a drop. Labels stay out of the face gallery. They do not name anyone.

Naming a detected face still happens from an incident. Those confirmed crops are faces the current detector already found, so they cannot teach it to find faces it missed.

## Label crops

1. Open **People**.
2. Choose **Materialize** to cut upper-body crops from recorded person observations.
3. Draw a face, click an orange baseline box to mark it as not a face, choose **No face**, or **Drop** with a reason.
4. **Freeze splits**, then **Export**. Train the detector outside SurvNG.
5. **Score IR** with the candidate and baseline XML paths before changing **Admin → Detection → Face Detector Model**.

The export format, day splits, and promotion gate are in [Camera-specific face detector](../face-detection-training.md).

## Privacy note

Face crops stay on your SurvNG installation. Treat them as sensitive household or workplace information.

## Related

- [Incidents](incidents.md)
- [AI assistant](assistant.md)
- [Admin](admin.md)
