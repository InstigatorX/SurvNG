# Third-party software

This document summarizes the main third-party components SurvNG uses or ships.
It is not a substitute for the full license texts of those projects. Exact
versions are pinned in `requirements.txt`, `frontend/package-lock.json`, and
the `Dockerfile`.

## Bundled in Docker images

| Component | Role | Typical license | Notes |
|-----------|------|-----------------|-------|
| [go2rtc](https://github.com/AlexxIT/go2rtc) | Restream / WebRTC / MSE helper | MIT | Binary downloaded in the image build; keep copyright notice when redistributing |
| [FFmpeg](https://ffmpeg.org/) | Recording, remux, decode helpers | LGPL/GPL (build-dependent) | Installed from `ppa:ubuntuhandbook1/ffmpeg8` (see `Dockerfile`). Redistributing the binary requires GPL/LGPL compliance (source offer for corresponding FFmpeg) |
| Ubuntu / Debian packages | Base OS libraries | Various | Base image `ubuntu:24.04` |
| Intel GPU userspace (`runtime-intel`) | OpenCL / VA-API / QSV | Vendor / package licenses | From `ppa:kobuk-team/intel-graphics`; confirm redistributable terms for your use |

## Core Python runtime dependencies

| Package | License (as declared) |
|---------|------------------------|
| FastAPI | MIT |
| Uvicorn | BSD-3-Clause |
| websockets | BSD-3-Clause |
| opencv-python | Apache-2.0 |
| openvino | Apache-2.0 |
| pydantic | MIT |
| onvif-zeep | MIT |
| paho-mqtt | EPL-2.0 / EDL-1.0 (see package metadata) |
| ftfy / regex | MIT / Apache-2.0 (see package metadata) |

Optional Darwin-only packages (`coremltools`, Pillow) are not installed in the
Linux container image.

## Frontend dependencies

Most frontend packages are MIT, ISC, Apache-2.0, or BSD-3-Clause. See
`frontend/package-lock.json` for the locked set. Notable exceptions may include
font licenses (for example OFL-1.1) and documentation assets (for example
CC-BY-4.0); retain those notices when redistributing the corresponding files.

## Optional / external tools

| Component | Notes |
|-----------|--------|
| Ultralytics (optional tracking comparison) | AGPL-3.0 — only if you install and use it; not part of the default SurvNG MIT distribution. See `VIDEO_PIPELINE.md` |
| OpenAI / other AI providers | Used only when configured; subject to their API terms |

## Source offers

For GPL-covered binaries shipped in SurvNG images (notably FFmpeg), corresponding
source is obtainable from the upstream project and from the package versions
recorded in the `Dockerfile`. SurvNG application source for a tagged release is
the matching Git tag in this repository.
