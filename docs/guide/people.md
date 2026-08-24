# People

**People** answers: who was seen?

When face recognition is enabled, SurvNG stores face sightings separately from ordinary object detections. You review suggestions, name people, and later search their history.

## First-time face setup

1. Install the face model package on the server (see the project README face section).
2. Enable face recognition under **Admin → Detection**.
3. Confirm the embedding, landmark, and face-detector model paths.
4. Wait for new incidents that include clear faces.

Face recognition needs reasonably clear, front-ish faces. Distant blobs and heavy motion blur will not identify well.

## Review queue

New matches usually appear as suggestions. Confirm good matches so SurvNG can build a trusted gallery for that person. Automatic identification, if you enable it, still keeps a high bar and does not silently teach the gallery from unverified guesses.

## Naming a person

1. Open **People**.
2. Review an unknown cluster or suggestion.
3. Create or select a person record.
4. Confirm the best reference images.

Pin an especially clear reference when you have one. SurvNG prefers a diverse, quality-weighted gallery over simply keeping the newest crop.

## Example

A neighbor’s regular walker keeps appearing at the sidewalk:

1. Open the latest porch incident and note the face suggestion.
2. In **People**, confirm several clear crops of the same person.
3. Name the person `Alex Walker`.
4. Later, ask the assistant to trace that person across cameras, or filter history for that identity.

## Privacy note

Face data stays on your SurvNG installation. Treat People records as sensitive household or workplace information.

## Related

- [Incidents](incidents.md)
- [AI assistant](assistant.md)
- [Admin](admin.md)
