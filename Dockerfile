# syntax=docker/dockerfile:1.7
# Production image for the patch_sim web UI (core library + Reflex app).
#
# Reflex prod mode serves the compiled frontend and the backend on one port
# (3000). The tag pins the uv release; its Python minor version must match
# .python-version, because UV_PYTHON_DOWNLOADS=never forbids a fallback.
FROM ghcr.io/astral-sh/uv:0.13.8-python3.13-trixie-slim

# Reflex's bun installer needs curl and unzip.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl unzip \
    && rm -rf /var/lib/apt/lists/*

# Non-root user. Reflex writes .web/, .states/ and ~/.local/share/reflex at
# build time and at startup, so the app directory and home belong to it.
RUN groupadd --system --gid 1000 app \
    && useradd --system --uid 1000 --gid app --create-home --home-dir /home/app app \
    && mkdir /app \
    && chown app:app /app

ENV HOME=/home/app \
    UV_CACHE_DIR=/home/app/.cache/uv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    REFLEX_TELEMETRY_ENABLED=false \
    REFLEX_CHECK_LATEST_VERSION=false

USER app
WORKDIR /app

# Dependency layer: cached until pyproject.toml, uv.lock or .python-version change.
RUN --mount=type=cache,target=/home/app/.cache/uv,uid=1000,gid=1000 \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv sync --frozen --no-dev --group=ui --no-install-project

COPY --chown=app:app . .
RUN --mount=type=cache,target=/home/app/.cache/uv,uid=1000,gid=1000 \
    uv sync --frozen --no-dev --group=ui

# Install bun and the frontend packages and compile the frontend once, so the
# container needs no network at startup and compile errors fail the build.
RUN reflex export --frontend-only --no-zip

EXPOSE 3000

# Reflex prod mode rebuilds the frontend bundle on every start (a few seconds),
# so that REFLEX_API_URL set at `docker run` reaches the browser.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:3000/ping', timeout=4)"]

CMD ["reflex", "run", "--env", "prod", "--single-port", "--frontend-port", "3000", "--backend-host", "0.0.0.0"]
