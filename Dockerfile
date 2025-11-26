# Lightweight production image for FastAPI inference
FROM python:3.9-slim

# Prevent Python from buffering stdout/stderr and writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port for Vertex AI
EXPOSE 8080

# Start FastAPI with uvicorn
CMD ["uvicorn", "src.service.api:app", "--host", "0.0.0.0", "--port", "8080"]
