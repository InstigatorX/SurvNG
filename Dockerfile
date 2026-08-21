FROM node:22-bookworm-slim AS frontend
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM ubuntu:24.04 AS runtime-base

ARG SURVNG_UID=1000
ARG SURVNG_GID=1000
# Prototype-qualified FFmpeg 8.1.2 from ubuntuhandbook1/ffmpeg8 (Noble).
ARG FFMPEG_VERSION=10:8.1.2-0build1~ubuntu24.04
ARG GO2RTC_VERSION=1.9.14
ARG GO2RTC_SHA256=32d616af226bd731678ffde328b94cfb94e30339bfefc469cfb76323144615a6

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/survng-venv/bin:$PATH \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    SURVNG_CONFIG_PATH=/config/config.json

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gosu \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libusb-1.0-0 \
        procps \
        python3 \
        python3-venv \
        software-properties-common \
        tini \
    && add-apt-repository -y ppa:ubuntuhandbook1/ffmpeg8 \
    && apt-get update \
    && apt-get install -y --no-install-recommends "ffmpeg=${FFMPEG_VERSION}" \
    && apt-mark hold ffmpeg \
    && ffmpeg -version | head -1 | grep -E 'ffmpeg version 8\.1\.2' \
    && curl -fsSL -o /usr/local/bin/go2rtc \
        "https://github.com/AlexxIT/go2rtc/releases/download/v${GO2RTC_VERSION}/go2rtc_linux_amd64" \
    && printf '%s  %s\n' "$GO2RTC_SHA256" /usr/local/bin/go2rtc | sha256sum -c - \
    && chmod 755 /usr/local/bin/go2rtc \
    && go2rtc -version | grep -F "$GO2RTC_VERSION" \
    && apt-get purge -y --auto-remove software-properties-common curl \
    && rm -rf /var/lib/apt/lists/* \
    && existing_group="$(getent group "$SURVNG_GID" | cut -d: -f1)" \
    && if [ -n "$existing_group" ]; then groupmod --new-name survng "$existing_group"; else groupadd --gid "$SURVNG_GID" survng; fi \
    && existing_user="$(getent passwd "$SURVNG_UID" | cut -d: -f1)" \
    && if [ -n "$existing_user" ]; then usermod --login survng --home /home/survng --move-home --gid survng "$existing_user"; else useradd --uid "$SURVNG_UID" --gid survng --create-home survng; fi \
    && python3 -m venv /opt/survng-venv

WORKDIR /app
ARG SURVNG_GIT_SHA=
ENV SURVNG_GIT_SHA=$SURVNG_GIT_SHA
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY survng/ ./survng/
COPY config.example.json /usr/share/survng/config.example.json
COPY docker/config.example.json /usr/share/survng/config.docker.example.json
COPY docker/go2rtc.example.yaml /usr/share/survng/go2rtc.example.yaml
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/survng-entrypoint
COPY --from=frontend /build/survng/static/ ./survng/static/
RUN if [ -n "$SURVNG_GIT_SHA" ]; then printf '%s\n' "$SURVNG_GIT_SHA" > /app/SURVNG_GIT_SHA; fi

RUN mkdir -p /config /data /models \
    && chown survng:survng /config /data /models

EXPOSE 8088
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8088/api/health', timeout=3).read()"

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/survng-entrypoint"]
CMD ["uvicorn", "survng.app.main:app", "--host", "0.0.0.0", "--port", "8088", "--loop", "asyncio", "--timeout-graceful-shutdown", "45"]

# Optional Intel OpenVINO GPU and VA-API/QSV userspace. Select this target with
# docker compose -f compose.yaml -f compose.intel-gpu.yaml up -d --build.
# Pins match the prototype Noble + kobuk-team/intel-graphics stack.
FROM runtime-base AS runtime-intel
USER root
ARG INTEL_COMPUTE_VERSION=26.27.39122.14-1~24.04~ppa1
ARG INTEL_IGC_VERSION=2.38.5-1~24.04
ARG INTEL_GMMLIB_VERSION=22.10.0-1~24.04~ppa1
ARG INTEL_LEVEL_ZERO_VERSION=1.32.0-1~24.04~ppa1
ARG INTEL_MEDIA_VERSION=26.2.2-1~24.04~ppa1
ARG INTEL_VPL_VERSION=1:2.16.0-1~24.04~ppa1
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:kobuk-team/intel-graphics \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        "intel-media-va-driver-non-free=${INTEL_MEDIA_VERSION}" \
        "intel-opencl-icd=${INTEL_COMPUTE_VERSION}" \
        "libigc2=${INTEL_IGC_VERSION}" \
        "libigdfcl2=${INTEL_IGC_VERSION}" \
        "libigdgmm12=${INTEL_GMMLIB_VERSION}" \
        "libmfx-gen1.2=${INTEL_MEDIA_VERSION}" \
        "libvpl2=${INTEL_VPL_VERSION}" \
        "libze-intel-gpu1=${INTEL_COMPUTE_VERSION}" \
        "libze1=${INTEL_LEVEL_ZERO_VERSION}" \
        ocl-icd-libopencl1 \
    && apt-mark hold \
        intel-media-va-driver-non-free \
        intel-opencl-icd \
        libigc2 \
        libigdfcl2 \
        libigdgmm12 \
        libmfx-gen1.2 \
        libvpl2 \
        libze-intel-gpu1 \
        libze1 \
    && apt-get purge -y --auto-remove software-properties-common \
    && rm -rf /var/lib/apt/lists/*

FROM runtime-base AS runtime
