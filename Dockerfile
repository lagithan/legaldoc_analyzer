# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Create __init__.py files to ensure proper Python packages
RUN find /app -type d -name "__pycache__" -delete && \
    touch /app/__init__.py && \
    find /app -type d -exec touch {}/__init__.py \; && \
    ls -la /app/

# Verify the app structure
RUN echo "=== App structure ===" && \
    ls -la /app/ && \
    echo "=== App/app structure ===" && \
    ls -la /app/app/ 2>/dev/null || echo "No app/app directory" && \
    echo "=== Models structure ===" && \
    ls -la /app/app/models/ 2>/dev/null || ls -la /app/models/ 2>/dev/null || echo "No models directory found"

# Make sure scripts have execute permissions
RUN chmod +x run.py

# Railway automatically provides PORT environment variable
EXPOSE 8000

# Create startup script with better error handling and debugging
RUN echo '#!/bin/bash\n\
set -e\n\
echo "=== Starting Application ==="\n\
echo "Current directory: $(pwd)"\n\
echo "Python path: $PYTHONPATH"\n\
echo "Contents of /app:"\n\
ls -la /app/\n\
echo "Python version: $(python --version)"\n\
echo "Installing packages check:"\n\
python -c "import sys; print(sys.path)" || true\n\
echo "=== Running Application ==="\n\
python run.py' > /app/start.sh

RUN chmod +x /app/start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use the startup script as the command
CMD ["/app/start.sh"]