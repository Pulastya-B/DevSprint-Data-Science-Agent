# Multi-stage build for Google Cloud Run
# Stage 1: Build Frontend
FROM node:20-alpine as frontend-builder

WORKDIR /frontend

# Copy frontend files
COPY FRRONTEEEND/package*.json ./
RUN npm install

COPY FRRONTEEEND/ ./
RUN npm run build

# Stage 2: Build Python environment
FROM python:3.13-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 3: Runtime environment
FROM python:3.13-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ /app/src/
COPY examples/ /app/examples/

# Copy built frontend from frontend-builder
COPY --from=frontend-builder /frontend/dist /app/FRRONTEEEND/dist

# Create necessary directories for Cloud Run ephemeral storage
RUN mkdir -p /tmp/data_science_agent \
    /tmp/outputs/models \
    /tmp/outputs/plots \
    /tmp/outputs/reports \
    /tmp/outputs/data \
    /tmp/cache_db

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV OUTPUT_DIR=/tmp/outputs
ENV CACHE_DB_PATH=/tmp/cache_db/cache.db
ENV ARTIFACT_BACKEND=local

# Cloud Run expects the service to listen on the PORT env variable
EXPOSE 8080

# Health check (optional, Cloud Run handles this)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# Run the FastAPI application
CMD ["python", "src/api/app.py"]
