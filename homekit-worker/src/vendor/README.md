# Retained upstream media helpers

`sframe.ts` and `sframeRtp.ts` originate from camera.ui's HomeKit plugin:

https://github.com/cameraui/plugins/tree/91ce370391c0f18e63faeb26a876967315a317bc/camera-ui-homekit/src/camera

Copyright (c) 2023–2026 seydx. MIT; see `LICENSE.camera-ui.md`.

SurvNG changes the SFrame type-only import to the pinned HAP package. It adds
input/counter bounds and authentication comparison hardening. No camera.ui
runtime, plugin framework, or camera configuration is required.

These helpers encode upstream interoperability observations. Automated crypto
and packet tests do not establish Apple-device compatibility.

The SDP decoration in `../transports.ts` is also adapted from the same revision's
`webrtcSessions.ts` under this MIT license. SurvNG adds negotiated on-wire RTP
stream identifiers and explicit BUNDLE selection; these are covered by the
local-peer media test.
