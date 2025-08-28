#!/bin/bash
set -e

# Check if the main application is responsive
python3 -c "
import sys
import time
try:
    # Basic health check - can be extended
    import src.pqc_iot_retrofit
    print('Health check passed')
    sys.exit(0)
except Exception as e:
    print(f'Health check failed: {e}')
    sys.exit(1)
"
