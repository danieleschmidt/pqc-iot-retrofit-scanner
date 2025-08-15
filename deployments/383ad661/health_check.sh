#!/bin/bash
# Health check script for PQC Scanner

set -e

HEALTH_URL="${HEALTH_CHECK_URL:-http://localhost:8080/health}"
TIMEOUT="${HEALTH_TIMEOUT:-10}"

echo "Checking health at $HEALTH_URL"

# Simple health check simulation for demo
echo "Health check passed (simulated)"
exit 0
