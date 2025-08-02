# Deployment Guide

This directory contains deployment configurations and documentation for the PQC IoT Retrofit Scanner.

## Quick Start

### Docker Deployment

```bash
# Build and run development environment
docker-compose up pqc-dev

# Run tests
docker-compose up pqc-test

# Production deployment
docker-compose up pqc-prod
```

### Local Installation

```bash
# Install from PyPI (when published)
pip install pqc-iot-retrofit-scanner

# Install from source
git clone https://github.com/terragon-ai/pqc-iot-retrofit-scanner.git
cd pqc-iot-retrofit-scanner
pip install -e ".[dev,analysis]"
```

## Deployment Options

### 1. Container Deployment (Recommended)

The project provides multi-stage Docker images optimized for different use cases:

- **Production**: Minimal runtime environment
- **Development**: Includes development tools and debugging capabilities
- **Testing**: Pre-configured for CI/CD testing
- **Analysis**: Enhanced with additional analysis tools and Jupyter

#### Container Images

```bash
# Pull from registry (when published)
docker pull terragon/pqc-iot-retrofit-scanner:latest
docker pull terragon/pqc-iot-retrofit-scanner:dev

# Or build locally
./scripts/build.sh --docker
```

#### Environment Variables

Key environment variables for container deployment:

```bash
# Logging configuration
PQC_LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR, CRITICAL
PQC_LOG_FORMAT=structured             # structured, simple, json

# Analysis configuration
PQC_MAX_ANALYSIS_THREADS=4            # Parallel analysis threads
PQC_ANALYSIS_TIMEOUT=300              # Analysis timeout in seconds
PQC_MEMORY_LIMIT=2048                 # Memory limit in MB

# Security settings
PQC_ENABLE_SIDE_CHANNEL_PROTECTION=1  # Enable timing attack protection
PQC_SECURE_MEMORY_CLEAR=1             # Clear sensitive memory after operations

# Storage paths
PQC_WORK_DIR=/app/data                # Working directory
PQC_OUTPUT_DIR=/app/output            # Output directory
PQC_CACHE_DIR=/app/.cache             # Cache directory
```

### 2. Kubernetes Deployment

For production Kubernetes deployments, see the [kubernetes/](../kubernetes/) directory for:

- Deployment manifests
- Service configurations  
- ConfigMaps and Secrets
- Horizontal Pod Autoscaling
- Network policies

### 3. Cloud Provider Deployments

#### AWS ECS/Fargate

```yaml
# ecs-task-definition.json
{
  "family": "pqc-iot-retrofit-scanner",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "pqc-scanner",
      "image": "terragon/pqc-iot-retrofit-scanner:latest",
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "PQC_LOG_LEVEL",
          "value": "INFO"
        }
      ]
    }
  ]
}
```

#### Google Cloud Run

```yaml
# cloud-run-service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: pqc-iot-retrofit-scanner
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/memory: "2Gi"
        run.googleapis.com/cpu: "1"
    spec:
      containers:
      - image: terragon/pqc-iot-retrofit-scanner:latest
        ports:
        - containerPort: 8080
        env:
        - name: PQC_LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
```

#### Azure Container Instances

```yaml
# azure-container-group.yaml
apiVersion: 2021-03-01
location: eastus
name: pqc-iot-retrofit-scanner
properties:
  containers:
  - name: pqc-scanner
    properties:
      image: terragon/pqc-iot-retrofit-scanner:latest
      resources:
        requests:
          cpu: 1
          memoryInGb: 2
      ports:
      - port: 8080
        protocol: TCP
      environmentVariables:
      - name: PQC_LOG_LEVEL
        value: INFO
  osType: Linux
  restartPolicy: Always
type: Microsoft.ContainerInstance/containerGroups
```

## Performance Considerations

### Resource Requirements

#### Minimum Requirements
- **CPU**: 1 core
- **Memory**: 1GB RAM
- **Storage**: 5GB available space
- **Network**: Internet access for dependency downloads

#### Recommended for Production
- **CPU**: 4+ cores
- **Memory**: 4GB+ RAM  
- **Storage**: 20GB+ available space
- **Network**: High-bandwidth for large firmware analysis

#### Large-Scale Deployment
- **CPU**: 8+ cores per instance
- **Memory**: 8GB+ RAM per instance
- **Storage**: 100GB+ shared storage
- **Load Balancer**: For distributing analysis workload

### Scaling Strategies

#### Horizontal Scaling
```yaml
# kubernetes/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: pqc-scanner-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: pqc-iot-retrofit-scanner
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

#### Vertical Scaling
- Scale up CPU/memory for single large firmware analysis
- Use memory-optimized instances for complex analysis workloads
- Consider GPU acceleration for cryptographic operations (future)

## Security Hardening

### Container Security

```dockerfile
# Security-hardened container example
FROM python:3.11-slim-bookworm

# Create non-root user
RUN groupadd -r pqciot && useradd -r -g pqciot pqciot

# Install only essential packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set security-focused environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PQC_SECURE_MODE=1

# Copy application with correct ownership
COPY --chown=pqciot:pqciot . /app
WORKDIR /app

# Switch to non-root user
USER pqciot

# Use read-only root filesystem
VOLUME ["/tmp", "/app/data", "/app/output"]
```

### Network Security

```yaml
# kubernetes/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: pqc-scanner-netpol
spec:
  podSelector:
    matchLabels:
      app: pqc-iot-retrofit-scanner
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS only
    - protocol: TCP
      port: 53   # DNS
```

### Secrets Management

#### Kubernetes Secrets
```yaml
# kubernetes/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: pqc-scanner-secrets
type: Opaque
data:
  signing-key: <base64-encoded-key>
  api-token: <base64-encoded-token>
```

#### External Secret Management
- **HashiCorp Vault**: For enterprise secret management
- **AWS Secrets Manager**: For AWS deployments
- **Azure Key Vault**: For Azure deployments  
- **Google Secret Manager**: for GCP deployments

## Monitoring and Observability

### Health Checks

```yaml
# kubernetes/deployment.yaml (excerpt)
containers:
- name: pqc-scanner
  image: terragon/pqc-iot-retrofit-scanner:latest
  livenessProbe:
    exec:
      command:
      - python
      - -c
      - "import pqc_iot_retrofit; print('OK')"
    initialDelaySeconds: 30
    periodSeconds: 30
  readinessProbe:
    httpGet:
      path: /health
      port: 8080
    initialDelaySeconds: 5
    periodSeconds: 5
```

### Logging Configuration

```yaml
# Structured logging configuration
logging:
  version: 1
  formatters:
    structured:
      format: '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "component": "%(name)s", "message": "%(message)s"}'
  handlers:
    console:
      class: logging.StreamHandler
      formatter: structured
  root:
    level: INFO
    handlers: [console]
```

### Metrics Collection

```yaml
# Prometheus metrics configuration
metrics:
  enabled: true
  port: 9090
  path: /metrics
  labels:
    service: pqc-iot-retrofit-scanner
    version: "0.1.0"
```

## Backup and Recovery

### Data Backup Strategy

```bash
#!/bin/bash
# backup-script.sh

# Backup analysis results
kubectl exec deployment/pqc-scanner -- tar czf - /app/data | \
  aws s3 cp - s3://pqc-backups/data-$(date +%Y%m%d).tar.gz

# Backup configuration
kubectl get configmap pqc-config -o yaml > \
  backup/pqc-config-$(date +%Y%m%d).yaml
```

### Disaster Recovery

1. **Automated Backups**: Daily backups to cloud storage
2. **Multi-Region Deployment**: Active-passive failover
3. **Data Replication**: Real-time replication of critical data
4. **Recovery Testing**: Monthly disaster recovery drills

## Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Check memory usage
kubectl top pods -l app=pqc-scanner

# Adjust memory limits
kubectl patch deployment pqc-scanner -p '{"spec":{"template":{"spec":{"containers":[{"name":"pqc-scanner","resources":{"limits":{"memory":"4Gi"}}}]}}}}'
```

#### Slow Analysis Performance
```bash
# Check CPU utilization
kubectl top pods -l app=pqc-scanner

# Scale horizontally
kubectl scale deployment pqc-scanner --replicas=5

# Check for resource contention
kubectl describe node <node-name>
```

#### Network Connectivity Issues
```bash
# Test network connectivity
kubectl exec -it deployment/pqc-scanner -- ping 8.8.8.8

# Check DNS resolution
kubectl exec -it deployment/pqc-scanner -- nslookup google.com

# Verify network policies
kubectl get networkpolicy
```

### Debug Mode

```bash
# Enable debug logging
kubectl set env deployment/pqc-scanner PQC_LOG_LEVEL=DEBUG

# Access debug information
kubectl logs -f deployment/pqc-scanner

# Interactive debugging
kubectl exec -it deployment/pqc-scanner -- bash
```

## Migration and Upgrades

### Rolling Updates

```bash
# Update to new version
kubectl set image deployment/pqc-scanner \
  pqc-scanner=terragon/pqc-iot-retrofit-scanner:v0.2.0

# Monitor rollout
kubectl rollout status deployment/pqc-scanner

# Rollback if needed
kubectl rollout undo deployment/pqc-scanner
```

### Blue-Green Deployment

```bash
# Deploy new version to staging
kubectl apply -f kubernetes/deployment-green.yaml

# Test new version
curl -f http://pqc-scanner-green:8080/health

# Switch traffic
kubectl patch service pqc-scanner -p '{"spec":{"selector":{"version":"green"}}}'

# Clean up old version
kubectl delete -f kubernetes/deployment-blue.yaml
```

## Cost Optimization

### Resource Optimization

```yaml
# Optimized resource requests/limits
resources:
  requests:
    cpu: 100m      # Minimum required
    memory: 256Mi  # Minimum required
  limits:
    cpu: 1000m     # Burst capacity
    memory: 2Gi    # Maximum allowed
```

### Spot/Preemptible Instances

```yaml
# Use spot instances for cost savings
nodeSelector:
  node-type: spot
tolerations:
- key: "spot"
  operator: "Equal"
  value: "true"
  effect: "NoSchedule"
```

### Auto-scaling Configuration

```yaml
# Cost-aware auto-scaling
minReplicas: 1          # Minimum for cost
maxReplicas: 5          # Maximum for cost control
targetCPUUtilization: 80  # Higher threshold for cost savings
scaleDownStabilization: 300s  # Prevent thrashing
```

---

For more detailed deployment scenarios and platform-specific guides, see the individual deployment directories:

- [kubernetes/](../kubernetes/) - Kubernetes deployment manifests
- [docker/](../docker/) - Docker-specific configurations  
- [cloud/](../cloud/) - Cloud provider deployment templates
- [monitoring/](../monitoring/) - Monitoring and observability setup