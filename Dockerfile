# Multi-stage Dockerfile for PQC IoT Retrofit Scanner
# Optimized for security, size, and development workflow

# Build stage - includes build tools and dependencies
FROM python:3.11-slim-bookworm as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Add metadata labels
LABEL org.opencontainers.image.title="PQC IoT Retrofit Scanner"
LABEL org.opencontainers.image.description="CLI + GitHub Action for post-quantum cryptography retrofitting of IoT firmware"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL org.opencontainers.image.vendor="Terragon Labs"
LABEL org.opencontainers.image.licenses="MIT"

# Install system dependencies required for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    cmake \
    pkg-config \
    libffi-dev \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for build process
RUN groupadd -r pqcbuild && useradd -r -g pqcbuild pqcbuild

# Set up Python environment
ENV PYTHONPATH=/app/src
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Create application directory
WORKDIR /app

# Copy dependency files first (for better Docker layer caching)
COPY pyproject.toml ./
COPY README.md ./
COPY LICENSE ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e ".[analysis]"

# Production stage - minimal runtime environment
FROM python:3.11-slim-bookworm as production

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libffi8 \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for runtime
RUN groupadd -r pqciot && useradd -r -g pqciot -d /app -s /bin/bash pqciot

# Set up application directory
WORKDIR /app

# Copy Python dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source code
COPY --chown=pqciot:pqciot src/ ./src/
COPY --chown=pqciot:pqciot pyproject.toml ./
COPY --chown=pqciot:pqciot README.md ./
COPY --chown=pqciot:pqciot LICENSE ./

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PQC_CONTAINER=1

# Create directories for data and output
RUN mkdir -p /app/data /app/output /app/logs && \
    chown -R pqciot:pqciot /app

# Install the package in development mode
RUN pip install -e .

# Switch to non-root user
USER pqciot

# Expose ports (if needed for future web interface)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import pqc_iot_retrofit; print('OK')" || exit 1

# Default command
CMD ["pqc-iot", "--help"]

# Development stage - includes development tools and test dependencies
FROM production as development

# Switch back to root to install dev dependencies
USER root

# Install development system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    vim \
    less \
    && rm -rf /var/lib/apt/lists/*

# Install development Python dependencies
RUN pip install -e ".[dev,analysis]"

# Install additional development tools
RUN pip install \
    jupyter \
    ipython \
    pre-commit \
    tox \
    nox

# Set up pre-commit hooks
COPY .pre-commit-config.yaml ./
RUN git init . && pre-commit install-hooks || true

# Create development directories
RUN mkdir -p /app/notebooks /app/experiments && \
    chown -R pqciot:pqciot /app

# Switch back to non-root user
USER pqciot

# Development command
CMD ["bash"]

# Testing stage - includes test dependencies and test data
FROM development as testing

USER root

# Copy test files
COPY --chown=pqciot:pqciot tests/ ./tests/
COPY --chown=pqciot:pqciot pytest.ini ./
COPY --chown=pqciot:pqciot .bandit ./

# Install test-specific dependencies
RUN pip install pytest-xdist pytest-benchmark pytest-timeout

USER pqciot

# Run tests by default
CMD ["pytest", "-v"]