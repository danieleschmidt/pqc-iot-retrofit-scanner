# Production Dockerfile for PQC IoT Retrofit Scanner
# Multi-stage build for optimized production image

# Build stage
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create app user for security
RUN groupadd -r pqcuser && useradd -r -g pqcuser pqcuser

# Create application directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY setup.py .
COPY README.md .

# Install application
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/cache /app/logs /app/data && \
    chown -R pqcuser:pqcuser /app

# Switch to non-root user
USER pqcuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# Default command
CMD ["python", "-m", "src.pqc_iot_retrofit.cli", "--serve", "--host", "0.0.0.0", "--port", "8080"]

# Labels for metadata
LABEL maintainer="Terragon Labs" \
      version="1.0.0" \
      description="PQC IoT Retrofit Scanner - Production Ready" \
      license="MIT"