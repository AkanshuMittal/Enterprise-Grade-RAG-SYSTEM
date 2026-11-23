# API image. Build from the PROJECT ROOT:
#   docker build -f deploy/Dockerfile -t rag-api .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed

WORKDIR /app

COPY deploy/requirements-prod.txt .
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && pip install --no-cache-dir -r requirements-prod.txt \
 && apt-get purge -y build-essential \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

# Bake NeMo's embedding model into the image so cold starts do not download it.
# If your NeMo config uses a different model, change the name here.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')" \
 || echo "WARNING: fastembed prebake failed - model will download at first start"

COPY app ./app

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# Cloud Run injects $PORT (8080).
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
