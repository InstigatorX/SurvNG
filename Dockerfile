FROM node:22-bookworm-slim AS frontend
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm AS runtime-base

ARG SURVNG_UID=1000
ARG SURVNG_GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/survng-venv/bin:$PATH \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    SURVNG_CONFIG_PATH=/config/config.json

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        gosu \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libusb-1.0-0 \
        procps \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "$SURVNG_GID" survng \
    && useradd --uid "$SURVNG_UID" --gid "$SURVNG_GID" --create-home survng \
    && python -m venv /opt/survng-venv

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY survng/ ./survng/
COPY config.example.json /usr/share/survng/config.example.json
COPY docker/config.example.json /usr/share/survng/config.docker.example.json
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/survng-entrypoint
COPY --from=frontend /build/survng/static/ ./survng/static/

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
FROM runtime-base AS runtime-intel
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        intel-media-va-driver \
        intel-opencl-icd \
        ocl-icd-libopencl1 \
    && rm -rf /var/lib/apt/lists/*

FROM runtime-base AS runtime
