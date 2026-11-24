FROM python:3.11-slim

# Prevent Python from writing bytecode and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies if required for sqlite/curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . /app

# Expose FastAPI (8000) and Streamlit (8501) ports
EXPOSE 8000 8501

# Default entrypoint runs FastAPI; docker-compose overrides for UI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
