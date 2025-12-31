# --- Stage 1: Build React Frontend ---
FROM node:18-alpine AS frontend-build

WORKDIR /app/frontend

# Copy only package files first for better caching
COPY frontend/package*.json ./

# Install dependencies (fallback to install if ci fails)
RUN if [ -f package-lock.json ]; then \
      npm ci --prefer-offline --no-audit; \
    else \
      npm install --prefer-offline --no-audit; \
    fi

# Copy the rest of the frontend source
COPY frontend/ .

# Build the frontend
RUN npm run build

# --- Stage 2: Python Backend with ML and Frontend ---
FROM python:3.11-slim

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install all Python dependencies in one layer
COPY ml_server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir \
    torch \
    transformers[torch] \
    datasets \
    accelerate \
    sacrebleu \
    bert-score \
    nltk \
    pandas \
    numpy \
    tqdm \
    flask \
    flask-cors

# Copy backend code
COPY ml_server/ ./ml_server/

# Copy built frontend from previous stage
COPY --from=frontend-build /app/frontend/dist ./static

# Create necessary runtime directories
RUN mkdir -p training_data model_outputs results

# Download NLTK data (only valid ones)
RUN python -c "import nltk; nltk.download('punkt')"

# Expose the backend port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=ml_server/app.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

# Healthcheck for container health monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:5000/api/health || exit 1

# Start the app with exec form (better signal handling)
CMD exec python ml_server/app.py

