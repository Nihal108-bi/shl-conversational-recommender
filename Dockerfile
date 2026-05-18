# SHL Conversational Recommender — HF Spaces Dockerfile
# HF Spaces requires:
#   - Container runs as a non-root user (uid 1000)
#   - App listens on port 7860 (matches app_port in README frontmatter)
#   - Image stays under the 50 GB disk quota (slim base + CPU-only torch)

FROM python:3.11-slim

# System deps that some Python wheels need at install time
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps as root for clean system-wide install
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# HF Spaces convention: non-root user with uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/user/.cache/huggingface

WORKDIR /home/user/app

# Copy app code with correct ownership
COPY --chown=user:user . .

# Pre-download embedding model + build BM25 index at BUILD time.
# This means /health responds instantly on first request — no cold-start
# model download. Adds ~80 MB to the image, worth it.
RUN python -m app.indexer

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]