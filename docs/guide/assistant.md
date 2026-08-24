# AI assistant

The sparkle control opens SurvNG’s assistant. It is a read-only helper for status questions, incident investigation, and guided reviews. It is not a remote shell and cannot freely change your system.

## What it can do

- Explain health and configuration
- Search incidents with structured tools
- Attach the actual incident picture when one exists
- Visually review a selected incident
- Build a bounded cross-camera timeline (“trace this incident”)
- Propose only narrow, camera-scoped motion adjustments that SurvNG itself calculates

## What it cannot do by default

- Run host commands
- Delete media
- Restart services on its own
- Apply setting changes without your explicit confirmation
- See credentials, raw stream URLs, or filesystem paths in the evidence it sends

## Set up AI analysis

Under **Admin → Detection → AI analysis & assistant**:

1. Choose a provider and API key.
2. Set the everyday model (fast status and routing).
3. Optionally set a detailed model for harder reviews.
4. Leave “allow confirmed changes” off until you are comfortable with recommendation apply.

The same provider settings serve Motion Audit image analysis and the assistant.

## Example questions

- “Which cameras are offline?”
- “Show person incidents at the gate after 8pm yesterday.”
- “Visually analyze this incident.”
- “Trace this incident across cameras.”
- “Why might EMA backup be firing so often on Driveway?”

## Applying a recommendation

When a visual review suggests a motion setting change:

1. Read the before/after values SurvNG computed.
2. Enable allow-confirmed-changes if you want apply available at all.
3. Confirm in the drawer.
4. SurvNG rejects stale or edited proposals.

Camera Advisor and Tune-Up follow the same confirm-before-apply idea.

## Privacy and retention

Assistant chat history in the browser expires after 24 hours of inactivity. Evidence sent to the provider is bounded and redacted. Prefer providers and keys you trust with operational summaries and selected snapshots.

## Related

- [Incidents](incidents.md)
- [Motion & detection](motion-detection.md)
- [Admin](admin.md)
- [HTTP API](api.md) (`/api/assistant/chat`)
