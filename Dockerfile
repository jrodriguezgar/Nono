# ── Build stage ───────────────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /build

COPY pyproject.toml ./
COPY nono/ ./nono/

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir . \
    && pip install --no-cache-dir uvicorn fastapi

# ── Runtime stage ─────────────────────────────────────────────────
FROM python:3.13-slim

LABEL maintainer="DatamanEdge" \
      description="Nono GenAI Framework — Agentic API Server"

# Non-root user for security
RUN groupadd -r nono && useradd -r -g nono -d /app -s /sbin/nologin nono

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project source
COPY nono/ ./nono/
COPY pyproject.toml ./

# Create directories for output and config overrides
RUN mkdir -p /app/output /app/config && chown -R nono:nono /app

USER nono

# Default port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the API server
CMD ["uvicorn", "nono.server:app", "--host", "0.0.0.0", "--port", "8000"]
