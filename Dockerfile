# Use python:3.11 (the full image you already have locally from your venv work)
# Switch to python:3.11-slim only when Docker Hub is reachable
FROM mirror.gcr.io/library/python:3.11

WORKDIR /app

# System deps for PyMuPDF image rendering and OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        poppler-utils \
        tesseract-ocr \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python deps first (cached layer) ──────────────────────────────────
COPY pyproject.toml ./
RUN mkdir -p src && touch src/__init__.py
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir ".[docling]" \
 && pip install --no-cache-dir httpx

# ── Copy source ───────────────────────────────────────────────────────────────
COPY . .

# ── Create artifact dirs ──────────────────────────────────────────────────────
RUN mkdir -p \
    .refinery/profiles \
    .refinery/extracted \
    .refinery/ldu \
    .refinery/pageindex \
    .refinery/chroma \
    .refinery/qa \
    .refinery/uploads \
    data/raw

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8502

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8502/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8502", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]