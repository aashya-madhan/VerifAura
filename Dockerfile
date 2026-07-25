# ── Base image ────────────────────────────────────────────────
FROM python:3.11-slim

# ── System deps needed by OpenCV, PyMuPDF, Pillow ─────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies first (cached layer) ──────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project files ─────────────────────────────────────────
COPY . .

# ── Make sure the uploads directory exists ─────────────────────
RUN mkdir -p static/uploads

# ── HuggingFace Spaces requires port 7860 ─────────────────────
EXPOSE 7860

# ── Run gunicorn ───────────────────────────────────────────────
CMD ["gunicorn", "--chdir", "src", "app:app", \
     "--bind", "0.0.0.0:7860", \
     "--workers", "1", \
     "--timeout", "120"]
