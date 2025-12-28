# ===============================
# Stage 1: Build Frontend
# ===============================
# Cache bust: 2025-12-28 fix
FROM node:20-alpine AS frontend-builder

WORKDIR /frontend

COPY FRRONTEEEND/package*.json ./
RUN npm install

COPY FRRONTEEEND/ ./
RUN npm run build


# ===============================
# Stage 2: Build Python environment
# ===============================
FROM python:3.12-slim AS builder

# Install build dependencies (needed for ML wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip tooling
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ===============================
# Stage 3: Runtime environment
# ===============================
FROM python:3.12-slim

# Install runtime shared libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# App working directory
WORKDIR /app

# Copy backend code
COPY src/ /app/src/
COPY examples/ /app/examples/

# Copy frontend build
COPY --from=frontend-builder /frontend/dist /app/FRRONTEEEND/dist

# Cloud Run ephemeral directories
RUN mkdir -p \
    /tmp/data_science_agent \
    /tmp/outputs/models \
    /tmp/outputs/plots \
    /tmp/outputs/reports \
    /tmp/outputs/data \
    /tmp/cache_db

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV OUTPUT_DIR=/tmp/outputs
ENV CACHE_DB_PATH=/tmp/cache_db/cache.db
ENV ARTIFACT_BACKEND=local

EXPOSE 8080

# Start FastAPI
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
