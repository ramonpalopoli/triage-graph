# Approval API image: non-root, no build tools at runtime, SQLite checkpoints on a volume.
#   docker build -t triage-graph .
#   docker run --rm -p 8000:8000 -e TRIAGE_API_TOKEN=$(openssl rand -hex 24) \
#     -v triage-data:/data triage-graph
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install ".[api,sqlite,postgres]"

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    TRIAGE_CHECKPOINT_URL=sqlite:////data/triage.db \
    TRIAGE_AUDIT_LOG=/data/api-audit.jsonl \
    TRIAGE_LLM=rule
RUN groupadd --system --gid 10001 triage \
 && useradd --system --uid 10001 --gid triage --home-dir /nonexistent --shell /usr/sbin/nologin triage \
 && mkdir /data && chown triage:triage /data
COPY --from=build /opt/venv /opt/venv
USER 10001:10001
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]
# TRIAGE_API_TOKEN must be provided at runtime (>= 24 chars); without it the API answers 503.
CMD ["uvicorn", "--factory", "triage_graph.api:create_app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
