# Motion Pipeline Roadmap

## Optional optical-flow evidence

Status: **Saved for later — not implemented**

Add optical flow as an optional, independently registered observation stage. It should run as a parallel evidence source and publish an `optical_flow` confidence score for the existing fusion pipeline; it should not replace Enhanced Motion Analysis initially.

Recommended first release:

- Disabled globally by default.
- Configurable per camera.
- Plain-language GUI label: **Directional motion analysis (optical flow, higher CPU)**.
- Low-resolution processing at 2–3 FPS.
- Audit/monitor-only rollout before allowing it to influence object-detection decisions.
- Useful for cameras where shadows, rain, vegetation, or lighting changes confuse frame-difference analysis.
- Expose availability, timing, failures, and debug visualization through the existing stage catalog and Motion Diagnostics UI.

Implementation path:

1. Implement and register an `optical_flow_evidence` `MotionStage`.
2. Add it to the observation graph as its own parallel branch.
3. Add `optical_flow` thresholds and weights to evidence fusion.
4. Add global and per-camera GUI controls with an explicit CPU-cost warning.
5. Benchmark multiple simultaneous cameras before enabling optical-flow validation.

Reminder: revisit this item when adding the next motion algorithm or tuning cameras that remain unreliable with ONVIF plus Enhanced Motion Analysis.
