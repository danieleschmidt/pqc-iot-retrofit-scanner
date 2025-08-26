#!/bin/bash
# Generation 6 Production Health Check

echo "🔍 Checking Generation 6 PQC IoT Retrofit Platform..."

# Check API health
curl -f http://localhost:8080/health || exit 1

# Check quantum research module
curl -f http://localhost:8081/quantum/status || exit 1

# Check monitoring
curl -f http://localhost:9090/api/v1/query?query=up || exit 1

echo "✅ All systems operational"
