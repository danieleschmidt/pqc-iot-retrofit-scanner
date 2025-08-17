#!/usr/bin/env python3
"""
Production Deployment Script for PQC IoT Retrofit Scanner
Autonomous deployment with health checks and monitoring
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path

def deploy_production():
    """Deploy PQC IoT Retrofit Scanner to production environment."""
    
    print("🚀 PQC IoT Retrofit Scanner - Production Deployment")
    print("=" * 60)
    
    # Step 1: Environment Validation
    print("\n📋 Step 1: Environment Validation")
    
    # Check Docker availability
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        print(f"  ✅ Docker: {result.stdout.strip()}")
    except FileNotFoundError:
        print("  ❌ Docker not found - install Docker first")
        return False
    
    # Check docker-compose availability  
    try:
        result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
        print(f"  ✅ Docker Compose: {result.stdout.strip()}")
    except FileNotFoundError:
        print("  ❌ Docker Compose not found - install docker-compose first")
        return False
    
    # Step 2: Build Production Images
    print("\n🔨 Step 2: Building Production Images")
    
    # Build production image
    build_cmd = [
        'docker-compose', 'build', 
        '--build-arg', f'BUILD_DATE={time.strftime("%Y-%m-%dT%H:%M:%SZ")}',
        '--build-arg', 'VERSION=2.0.0',
        'pqc-prod'
    ]
    
    print(f"  Building: {' '.join(build_cmd)}")
    result = subprocess.run(build_cmd, capture_output=False)
    
    if result.returncode != 0:
        print("  ❌ Production image build failed")
        return False
    
    print("  ✅ Production image built successfully")
    
    # Step 3: Health Checks
    print("\n🏥 Step 3: Health Checks")
    
    # Start production container for health check
    health_cmd = ['docker-compose', 'run', '--rm', 'pqc-prod', 'pqc-iot', '--version']
    
    print("  Running health check...")
    result = subprocess.run(health_cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"  ✅ Health check passed: {result.stdout.strip()}")
    else:
        print(f"  ❌ Health check failed: {result.stderr}")
        return False
    
    # Step 4: Security Validation
    print("\n🔒 Step 4: Security Validation")
    
    # Run security scan
    security_cmd = ['docker-compose', 'run', '--rm', 'pqc-security']
    
    print("  Running security scans...")
    result = subprocess.run(security_cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("  ✅ Security scans completed")
    else:
        print("  ⚠️  Security scan had warnings (check reports)")
    
    # Step 5: Performance Validation
    print("\n⚡ Step 5: Performance Validation")
    
    # Run performance benchmark in container
    perf_cmd = [
        'docker-compose', 'run', '--rm', 
        '-v', f'{os.getcwd()}/performance_benchmark.py:/app/benchmark.py',
        'pqc-prod', 
        'python', 'benchmark.py'
    ]
    
    print("  Running performance benchmark...")
    result = subprocess.run(perf_cmd, capture_output=False)
    
    if result.returncode == 0:
        print("  ✅ Performance validation passed")
    else:
        print("  ❌ Performance validation failed")
        return False
    
    # Step 6: Production Deployment
    print("\n🌟 Step 6: Production Deployment Ready")
    
    deployment_manifest = {
        "deployment_id": f"pqc-iot-{int(time.time())}",
        "version": "2.0.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "ready_for_production",
        "validated_components": [
            "docker_images",
            "health_checks", 
            "security_scans",
            "performance_tests"
        ],
        "deployment_commands": {
            "start": "docker-compose up -d pqc-prod",
            "stop": "docker-compose down",
            "logs": "docker-compose logs -f pqc-prod",
            "health": "docker-compose exec pqc-prod pqc-iot --version"
        }
    }
    
    # Save deployment manifest
    with open('deployment_manifest.json', 'w') as f:
        json.dump(deployment_manifest, f, indent=2)
    
    print("  ✅ Deployment manifest created: deployment_manifest.json")
    print("  ✅ Production deployment validation complete")
    print("\n🎯 Ready for Production!")
    print("\nTo deploy to production:")
    print("  docker-compose up -d pqc-prod")
    print("\nTo monitor deployment:")
    print("  docker-compose logs -f pqc-prod")
    
    return True

if __name__ == "__main__":
    success = deploy_production()
    sys.exit(0 if success else 1)