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



# Make sure scripts have execute permissions
RUN chmod +x  run.py

# Railway automatically provides PORT environment variable
EXPOSE 8000

# Create startup script that runs setup.py then run.py
RUN echo '#!/bin/bash\nset -e\necho "Running setup..."\necho "Starting application..."\npython run.py' > /app/start.sh

RUN chmod +x /app/start.sh

# Use the startup script as the command
CMD ["/app/start.sh"]