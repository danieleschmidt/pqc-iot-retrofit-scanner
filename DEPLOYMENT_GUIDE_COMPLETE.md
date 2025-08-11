# PQC IoT Retrofit Scanner - Complete Deployment Guide

## 🚀 Production-Ready Deployment Guide

This guide provides comprehensive instructions for deploying the PQC IoT Retrofit Scanner in production environments with full Generation 4 capabilities including AI-powered analysis, autonomous research, and global multi-region support.

## 📋 Table of Contents

- [System Requirements](#system-requirements)
- [Pre-Deployment Checklist](#pre-deployment-checklist)
- [Installation Methods](#installation-methods)
- [Configuration](#configuration)
- [Multi-Region Deployment](#multi-region-deployment)
- [Monitoring and Observability](#monitoring-and-observability)
- [Security Configuration](#security-configuration)
- [Performance Tuning](#performance-tuning)
- [Disaster Recovery](#disaster-recovery)
- [Maintenance and Updates](#maintenance-and-updates)

## 🔧 System Requirements

### Minimum Requirements
- **CPU**: 4 cores, 2.4 GHz
- **Memory**: 8 GB RAM
- **Storage**: 100 GB SSD
- **Network**: 100 Mbps bandwidth
- **OS**: Ubuntu 20.04+ / RHEL 8+ / Docker

### Recommended Production Requirements
- **CPU**: 16+ cores, 3.2 GHz
- **Memory**: 32+ GB RAM
- **Storage**: 500 GB NVMe SSD
- **Network**: 1 Gbps bandwidth
- **Load Balancer**: HAProxy/NGINX
- **Database**: PostgreSQL 13+ cluster

### Cloud Provider Requirements

#### AWS Deployment
- **Instance Type**: c5.4xlarge or larger
- **Storage**: EBS gp3 with 3000 IOPS
- **Network**: Enhanced networking enabled
- **Security Groups**: Configured for HTTPS, SSH

#### Azure Deployment
- **VM Size**: Standard_D16s_v3 or larger
- **Storage**: Premium SSD with caching
- **Network**: Accelerated networking enabled
- **NSG**: Configured for production traffic

#### Google Cloud Platform
- **Machine Type**: c2-standard-16 or larger
- **Storage**: SSD persistent disk
- **Network**: Premium tier networking
- **Firewall**: Configured for secure access

## ✅ Pre-Deployment Checklist

### Security Prerequisites
- [ ] SSL/TLS certificates obtained and validated
- [ ] Security scanning completed (OWASP ZAP, Nessus)
- [ ] Penetration testing performed
- [ ] Secrets management configured (HashiCorp Vault, AWS Secrets Manager)
- [ ] Network security groups/firewalls configured
- [ ] VPN/bastion host access configured
- [ ] Backup and recovery procedures tested

### Infrastructure Prerequisites
- [ ] Load balancer configured with health checks
- [ ] Database cluster deployed and tested
- [ ] Monitoring stack deployed (Prometheus, Grafana)
- [ ] Logging aggregation configured (ELK Stack)
- [ ] CDN configured for static assets
- [ ] DNS records configured with failover

### Compliance Prerequisites
- [ ] GDPR compliance validated (EU deployments)
- [ ] SOC 2 Type II audit completed
- [ ] Data retention policies implemented
- [ ] Incident response procedures documented
- [ ] Privacy policy and terms of service updated

## 🐳 Installation Methods

### Method 1: Docker Container (Recommended)

```bash
# Pull the latest production image
docker pull terragon/pqc-iot-retrofit-scanner:latest

# Run with production configuration
docker run -d \
  --name pqc-scanner-prod \
  --restart unless-stopped \
  -p 8080:8080 \
  -p 8443:8443 \
  -v /opt/pqc-scanner/config:/app/config \
  -v /opt/pqc-scanner/data:/app/data \
  -v /opt/pqc-scanner/logs:/app/logs \
  -e ENVIRONMENT=production \
  -e DATABASE_URL=postgresql://user:pass@db:5432/pqc_scanner \
  -e REDIS_URL=redis://redis:6379/0 \
  terragon/pqc-iot-retrofit-scanner:latest

# Verify deployment
docker logs pqc-scanner-prod
curl -k https://localhost:8443/health
```

### Method 2: Kubernetes Deployment

```yaml
# kubernetes/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: pqc-scanner
  labels:
    name: pqc-scanner
    tier: production
---
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pqc-scanner
  namespace: pqc-scanner
  labels:
    app: pqc-scanner
    version: v1.0.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pqc-scanner
  template:
    metadata:
      labels:
        app: pqc-scanner
        version: v1.0.0
    spec:
      containers:
      - name: pqc-scanner
        image: terragon/pqc-iot-retrofit-scanner:1.0.0
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 8443
          name: https
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: pqc-scanner-secrets
              key: database-url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: data
          mountPath: /app/data
      volumes:
      - name: config
        configMap:
          name: pqc-scanner-config
      - name: data
        persistentVolumeClaim:
          claimName: pqc-scanner-data
```

Deploy to Kubernetes:
```bash
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/secrets.yaml
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/ingress.yaml

# Verify deployment
kubectl get pods -n pqc-scanner
kubectl logs -f deployment/pqc-scanner -n pqc-scanner
```

### Method 3: Native Installation

```bash
# Install system dependencies
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev \
  build-essential libssl-dev libffi-dev pkg-config \
  postgresql-client redis-tools

# Create application user
sudo useradd -r -s /bin/false -m -d /opt/pqc-scanner pqc-scanner

# Install application
sudo -u pqc-scanner -H bash << 'EOF'
cd /opt/pqc-scanner
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install pqc-iot-retrofit-scanner[production]
EOF

# Configure systemd service
sudo tee /etc/systemd/system/pqc-scanner.service << 'EOF'
[Unit]
Description=PQC IoT Retrofit Scanner
After=network.target postgresql.service redis.service
Requires=postgresql.service redis.service

[Service]
Type=notify
User=pqc-scanner
Group=pqc-scanner
WorkingDirectory=/opt/pqc-scanner
Environment=PATH=/opt/pqc-scanner/venv/bin
Environment=ENVIRONMENT=production
ExecStart=/opt/pqc-scanner/venv/bin/pqc-iot server --config /opt/pqc-scanner/config/production.yaml
ExecReload=/bin/kill -HUP $MAINPID
KillMode=process
Restart=on-failure
RestartSec=5s
LimitNOFILE=65536
LimitNPROC=32768

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable pqc-scanner
sudo systemctl start pqc-scanner
sudo systemctl status pqc-scanner
```

## ⚙️ Configuration

### Production Configuration File

Create `/opt/pqc-scanner/config/production.yaml`:

```yaml
# Production Configuration for PQC IoT Retrofit Scanner
environment: production
debug: false

# Server Configuration
server:
  host: "0.0.0.0"
  port: 8080
  ssl_port: 8443
  ssl_cert: "/opt/pqc-scanner/ssl/server.crt"
  ssl_key: "/opt/pqc-scanner/ssl/server.key"
  worker_processes: 16
  max_connections: 10000
  timeout: 300

# Database Configuration
database:
  url: "postgresql://pqc_user:secure_password@postgres-cluster:5432/pqc_scanner"
  pool_size: 20
  max_overflow: 30
  pool_timeout: 30
  pool_recycle: 3600
  ssl_mode: "require"

# Cache Configuration
cache:
  redis_url: "redis://redis-cluster:6379/0"
  default_timeout: 3600
  max_memory_policy: "allkeys-lru"
  cluster_enabled: true

# AI and Research Configuration
ai:
  adaptive_learning: true
  ensemble_detection: true
  anomaly_detection: true
  autonomous_research: true
  model_update_interval: 86400  # 24 hours
  research_data_retention: 2592000  # 30 days

# Performance Configuration
performance:
  auto_scaling: true
  min_workers: 8
  max_workers: 64
  concurrent_scans: 32
  memory_limit_mb: 8192
  cpu_limit_percent: 80
  cache_size_mb: 2048

# Security Configuration
security:
  secret_key: "${SECRET_KEY}"  # From environment/secrets manager
  jwt_secret: "${JWT_SECRET}"
  encryption_key: "${ENCRYPTION_KEY}"
  password_policy:
    min_length: 12
    require_uppercase: true
    require_lowercase: true
    require_numbers: true
    require_symbols: true
  session_timeout: 3600
  max_login_attempts: 5
  account_lockout_duration: 1800

# Monitoring Configuration
monitoring:
  enabled: true
  metrics_port: 9090
  health_check_interval: 30
  prometheus_enabled: true
  jaeger_enabled: true
  log_level: "INFO"
  structured_logging: true
  correlation_ids: true

# Global Deployment Configuration
global:
  region: "us-east-1"
  backup_regions: ["us-west-2", "eu-west-1", "ap-northeast-1"]
  cdn_enabled: true
  edge_computing: true
  multi_region_sync: true
  data_sovereignty: true

# Internationalization
i18n:
  default_language: "en"
  supported_languages: ["en", "es", "fr", "de", "ja", "zh", "pt", "it", "ru", "ko"]
  default_region: "na"
  auto_detect_locale: true

# Compliance Configuration
compliance:
  gdpr_enabled: true
  ccpa_enabled: true
  hipaa_enabled: false
  sox_enabled: false
  data_retention_days: 30
  audit_logging: true
  privacy_controls: true

# Backup Configuration
backup:
  enabled: true
  schedule: "0 2 * * *"  # Daily at 2 AM
  retention_days: 30
  encryption_enabled: true
  backup_location: "s3://pqc-scanner-backups/prod/"
  verify_backups: true
```

### Environment Variables

Create `/opt/pqc-scanner/.env`:
```bash
# Environment Variables for Production Deployment
ENVIRONMENT=production
SECRET_KEY=your-256-bit-secret-key-here
JWT_SECRET=your-jwt-signing-key-here  
ENCRYPTION_KEY=your-encryption-key-here
DATABASE_URL=postgresql://user:pass@localhost:5432/pqc_scanner
REDIS_URL=redis://localhost:6379/0

# Cloud Provider Configuration
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=us-east-1

# Monitoring and Logging
PROMETHEUS_GATEWAY=http://prometheus:9091
JAEGER_AGENT_HOST=jaeger
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project

# External Services
VIRUS_TOTAL_API_KEY=your-virustotal-api-key
SHODAN_API_KEY=your-shodan-api-key
```

## 🌍 Multi-Region Deployment

### Global Deployment Architecture

```yaml
# Global deployment configuration
regions:
  us-east-1:
    primary: true
    provider: aws
    instance_type: c5.4xlarge
    availability_zones: 3
    auto_scaling: true
    backup_regions: [us-west-2, eu-west-1]
    
  us-west-2:
    primary: false
    provider: aws
    instance_type: c5.2xlarge
    availability_zones: 2
    auto_scaling: true
    
  eu-west-1:
    primary: false
    provider: aws
    instance_type: c5.2xlarge
    availability_zones: 2
    auto_scaling: true
    compliance: [GDPR, ISO27001]
    
  ap-northeast-1:
    primary: false
    provider: aws
    instance_type: c5.2xlarge
    availability_zones: 2
    auto_scaling: true

# Cross-region configuration
cross_region:
  data_replication: true
  config_sync: true
  load_balancing: true
  failover_automatic: true
  rto_target: 300  # 5 minutes
  rpo_target: 60   # 1 minute
```

### DNS Configuration

```yaml
# Route53 configuration for global load balancing
dns:
  primary_domain: pqc-scanner.terragon.ai
  health_check_enabled: true
  failover_enabled: true
  geo_routing: true
  
  records:
    - name: api.pqc-scanner.terragon.ai
      type: A
      routing_policy: geolocation
      regions:
        us: us-east-1.pqc-scanner.terragon.ai
        eu: eu-west-1.pqc-scanner.terragon.ai
        ap: ap-northeast-1.pqc-scanner.terragon.ai
    
    - name: cdn.pqc-scanner.terragon.ai
      type: CNAME
      value: d1234567890.cloudfront.net
```

## 📊 Monitoring and Observability

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "pqc_scanner_rules.yml"

scrape_configs:
  - job_name: 'pqc-scanner'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 10s
    metrics_path: /metrics
    
  - job_name: 'pqc-scanner-health'
    static_configs:
      - targets: ['localhost:8080']
    scrape_interval: 30s
    metrics_path: /health/metrics

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']
```

### Grafana Dashboards

Key metrics to monitor:

1. **System Health Dashboard**
   - CPU utilization per region
   - Memory usage and garbage collection
   - Disk I/O and storage utilization
   - Network throughput and latency

2. **Application Metrics Dashboard**
   - Firmware scans per minute
   - Vulnerability detection rate  
   - PQC patch generation time
   - API response times and errors

3. **Security Dashboard**
   - Failed authentication attempts
   - Suspicious scanning patterns
   - Compliance violations
   - Security alerts and incidents

4. **Global Performance Dashboard**
   - Regional response times
   - Cross-region replication lag
   - CDN cache hit rates
   - Load balancer health

### Alerting Rules

```yaml
# pqc_scanner_rules.yml
groups:
  - name: pqc_scanner_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(pqc_scanner_errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: High error rate detected
          description: "Error rate is {{ $value }} errors per second"

      - alert: HighMemoryUsage
        expr: pqc_scanner_memory_usage_percent > 90
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: High memory usage
          description: "Memory usage is {{ $value }}%"

      - alert: DatabaseConnectionFailure
        expr: pqc_scanner_db_connection_errors > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: Database connection failure
          description: "Cannot connect to database"

      - alert: SlowResponseTime
        expr: histogram_quantile(0.95, rate(pqc_scanner_request_duration_seconds_bucket[5m])) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: Slow response time
          description: "95th percentile response time is {{ $value }}s"
```

## 🔒 Security Configuration

### SSL/TLS Configuration

Generate production certificates:
```bash
# Generate private key and certificate signing request
openssl genrsa -out server.key 4096
openssl req -new -key server.key -out server.csr \
  -subj "/C=US/ST=CA/L=San Francisco/O=Terragon Labs/CN=pqc-scanner.terragon.ai"

# Submit CSR to Certificate Authority or use Let's Encrypt
certbot certonly --webroot -w /var/www/html \
  -d pqc-scanner.terragon.ai \
  -d api.pqc-scanner.terragon.ai

# Install certificates
sudo cp /etc/letsencrypt/live/pqc-scanner.terragon.ai/fullchain.pem /opt/pqc-scanner/ssl/server.crt
sudo cp /etc/letsencrypt/live/pqc-scanner.terragon.ai/privkey.pem /opt/pqc-scanner/ssl/server.key
sudo chown pqc-scanner:pqc-scanner /opt/pqc-scanner/ssl/*
sudo chmod 600 /opt/pqc-scanner/ssl/server.key
```

### Firewall Configuration

```bash
# Configure iptables for production security
# Allow SSH (port 22)
iptables -A INPUT -p tcp --dport 22 -j ACCEPT

# Allow HTTP (port 80) for redirects
iptables -A INPUT -p tcp --dport 80 -j ACCEPT

# Allow HTTPS (port 443)
iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Allow monitoring (port 9090) from monitoring subnet only
iptables -A INPUT -p tcp --dport 9090 -s 10.0.1.0/24 -j ACCEPT

# Block all other incoming traffic
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A INPUT -j DROP

# Save configuration
iptables-save > /etc/iptables/rules.v4
```

### Secrets Management

Using HashiCorp Vault:
```bash
# Store secrets in Vault
vault kv put secret/pqc-scanner/prod \
  database_password=secure_db_password \
  jwt_secret=secure_jwt_secret \
  encryption_key=secure_encryption_key

# Configure application to use Vault
export VAULT_ADDR=https://vault.internal.com
export VAULT_TOKEN=your-vault-token
```

## ⚡ Performance Tuning

### Database Optimization

PostgreSQL configuration for high performance:

```sql
-- postgresql.conf settings for production
shared_buffers = 8GB
effective_cache_size = 24GB  
work_mem = 256MB
maintenance_work_mem = 2GB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
max_connections = 200

-- Create optimized indexes
CREATE INDEX CONCURRENTLY idx_vulnerabilities_algorithm_risk 
ON vulnerabilities(algorithm, risk_level);

CREATE INDEX CONCURRENTLY idx_scan_results_created_at 
ON scan_results(created_at) WHERE created_at >= CURRENT_DATE - INTERVAL '30 days';

-- Analyze tables for query optimization
ANALYZE vulnerabilities;
ANALYZE scan_results;
ANALYZE firmware_metadata;
```

### Redis Configuration

```conf
# redis.conf for production caching
maxmemory 16gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
timeout 0
tcp-keepalive 300
```

### Application Performance Tuning

```yaml
# Performance optimization settings
performance:
  # Worker process configuration
  worker_processes: 16
  worker_connections: 1024
  worker_rlimit_nofile: 65536
  
  # Memory management
  max_memory_per_worker: "2GB"
  garbage_collection_threshold: 700
  
  # Connection pooling
  db_pool_size: 20
  db_pool_max_overflow: 30
  db_pool_timeout: 30
  
  # Caching strategy
  cache_default_timeout: 3600
  cache_max_entries: 100000
  cache_eviction_policy: "lru"
  
  # Concurrency limits
  max_concurrent_scans: 32
  max_batch_size: 1000
  scan_timeout: 300
  
  # Resource limits
  max_upload_size: "100MB"
  max_firmware_size: "1GB"
  temp_file_cleanup_interval: 3600
```

## 🔄 Disaster Recovery

### Backup Strategy

```bash
#!/bin/bash
# backup.sh - Production backup script

set -euo pipefail

BACKUP_DIR="/backups/pqc-scanner"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

# Database backup
pg_dump -h postgres-cluster -U pqc_user pqc_scanner | \
  gzip > "${BACKUP_DIR}/db_backup_${DATE}.sql.gz"

# Application data backup
tar -czf "${BACKUP_DIR}/app_data_${DATE}.tar.gz" \
  /opt/pqc-scanner/data \
  /opt/pqc-scanner/config

# Configuration backup
tar -czf "${BACKUP_DIR}/config_${DATE}.tar.gz" \
  /opt/pqc-scanner/config \
  /etc/systemd/system/pqc-scanner.service

# Upload to cloud storage
aws s3 cp "${BACKUP_DIR}/db_backup_${DATE}.sql.gz" \
  s3://pqc-scanner-backups/prod/database/
aws s3 cp "${BACKUP_DIR}/app_data_${DATE}.tar.gz" \
  s3://pqc-scanner-backups/prod/application/
aws s3 cp "${BACKUP_DIR}/config_${DATE}.tar.gz" \
  s3://pqc-scanner-backups/prod/configuration/

# Cleanup old backups
find "${BACKUP_DIR}" -name "*.gz" -mtime +${RETENTION_DAYS} -delete

# Verify backup integrity
gzip -t "${BACKUP_DIR}/db_backup_${DATE}.sql.gz"
tar -tzf "${BACKUP_DIR}/app_data_${DATE}.tar.gz" >/dev/null

echo "Backup completed successfully: ${DATE}"
```

### Recovery Procedures

```bash
#!/bin/bash  
# restore.sh - Production recovery script

set -euo pipefail

BACKUP_DATE="${1:-latest}"
BACKUP_DIR="/backups/pqc-scanner"
S3_BUCKET="s3://pqc-scanner-backups/prod"

# Stop application
sudo systemctl stop pqc-scanner

# Download backup from S3
if [[ "$BACKUP_DATE" == "latest" ]]; then
  aws s3 cp "${S3_BUCKET}/database/" "${BACKUP_DIR}/" --recursive
  aws s3 cp "${S3_BUCKET}/application/" "${BACKUP_DIR}/" --recursive
  DB_BACKUP=$(ls -t "${BACKUP_DIR}"/db_backup_*.sql.gz | head -1)
  APP_BACKUP=$(ls -t "${BACKUP_DIR}"/app_data_*.tar.gz | head -1)
else
  aws s3 cp "${S3_BUCKET}/database/db_backup_${BACKUP_DATE}.sql.gz" "${BACKUP_DIR}/"
  aws s3 cp "${S3_BUCKET}/application/app_data_${BACKUP_DATE}.tar.gz" "${BACKUP_DIR}/"
  DB_BACKUP="${BACKUP_DIR}/db_backup_${BACKUP_DATE}.sql.gz"
  APP_BACKUP="${BACKUP_DIR}/app_data_${BACKUP_DATE}.tar.gz"
fi

# Restore database
dropdb -h postgres-cluster -U pqc_user pqc_scanner
createdb -h postgres-cluster -U pqc_user pqc_scanner
gunzip -c "$DB_BACKUP" | psql -h postgres-cluster -U pqc_user pqc_scanner

# Restore application data
tar -xzf "$APP_BACKUP" -C /

# Start application
sudo systemctl start pqc-scanner
sudo systemctl status pqc-scanner

echo "Recovery completed successfully from backup: $(basename $DB_BACKUP)"
```

### Failover Configuration

```yaml
# Load balancer failover configuration
failover:
  primary_region: us-east-1
  backup_regions:
    - us-west-2
    - eu-west-1
    
  health_checks:
    interval: 30
    timeout: 10
    retries: 3
    
  automatic_failover:
    enabled: true
    threshold_failures: 3
    recovery_time: 300
    
  notifications:
    - type: email
      recipients: ["ops@terragon.ai", "oncall@terragon.ai"]
    - type: slack
      webhook: "https://hooks.slack.com/services/..."
    - type: pagerduty
      service_key: "your-pagerduty-service-key"
```

## 🔄 Maintenance and Updates

### Update Procedures

```bash
#!/bin/bash
# update.sh - Production update script

set -euo pipefail

NEW_VERSION="${1:-latest}"
BACKUP_PREFIX="pre_update_$(date +%Y%m%d_%H%M%S)"

# Create pre-update backup
./backup.sh
cp /opt/pqc-scanner/config/production.yaml "/tmp/config_backup_${BACKUP_PREFIX}.yaml"

# Health check before update
curl -f http://localhost:8080/health || {
  echo "Health check failed before update"
  exit 1
}

# Rolling update for zero downtime
if command -v kubectl >/dev/null; then
  # Kubernetes rolling update
  kubectl set image deployment/pqc-scanner \
    pqc-scanner=terragon/pqc-iot-retrofit-scanner:${NEW_VERSION} \
    -n pqc-scanner
    
  kubectl rollout status deployment/pqc-scanner -n pqc-scanner --timeout=300s
else
  # Docker update
  docker pull terragon/pqc-iot-retrofit-scanner:${NEW_VERSION}
  docker stop pqc-scanner-prod
  docker rm pqc-scanner-prod
  
  # Start new container with updated image
  docker run -d \
    --name pqc-scanner-prod \
    --restart unless-stopped \
    -p 8080:8080 -p 8443:8443 \
    -v /opt/pqc-scanner/config:/app/config \
    -v /opt/pqc-scanner/data:/app/data \
    -v /opt/pqc-scanner/logs:/app/logs \
    terragon/pqc-iot-retrofit-scanner:${NEW_VERSION}
fi

# Wait for application to be ready
sleep 30

# Post-update health checks
curl -f http://localhost:8080/health || {
  echo "Health check failed after update - rolling back"
  # Rollback procedure would go here
  exit 1
}

# Run post-update tests
python3 /opt/pqc-scanner/tests/integration/test_post_update.py

echo "Update to version ${NEW_VERSION} completed successfully"
```

### Maintenance Windows

```yaml
# Scheduled maintenance configuration
maintenance:
  windows:
    - name: "Weekly Maintenance"
      schedule: "0 2 * * 0"  # Sunday 2 AM UTC
      duration: 2  # hours
      type: "minor_updates"
      
    - name: "Monthly Maintenance"
      schedule: "0 2 1 * *"  # First day of month, 2 AM UTC
      duration: 4  # hours
      type: "major_updates"
      
  notifications:
    advance_notice: 72  # hours
    reminder_notice: 2  # hours
    channels: ["email", "slack", "status_page"]
    
  procedures:
    - backup_verification
    - security_updates
    - dependency_updates
    - certificate_renewal
    - log_rotation
    - performance_optimization
```

## 📈 Scaling Guidelines

### Horizontal Scaling

```yaml
# Auto-scaling configuration
auto_scaling:
  min_replicas: 3
  max_replicas: 20
  target_cpu: 70
  target_memory: 80
  scale_up_cooldown: 300
  scale_down_cooldown: 600
  
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
    - type: Pods
      pods:
        metric:
          name: requests_per_second
        target:
          type: AverageValue
          averageValue: "100"
```

### Vertical Scaling

Monitor these metrics to determine when to scale vertically:
- Memory usage consistently above 85%
- CPU usage consistently above 80%
- Database connection pool exhaustion
- Cache hit rate below 90%
- Increased GC frequency

## 🚨 Troubleshooting

### Common Issues and Solutions

#### High Memory Usage
```bash
# Investigate memory usage
ps aux --sort=-%mem | head -20
pmap -x $(pgrep pqc-scanner)

# Check for memory leaks
valgrind --leak-check=full pqc-iot

# Tune garbage collection
export PYTHONHASHSEED=random
export MALLOC_ARENA_MAX=2
```

#### Database Performance Issues
```sql
-- Identify slow queries
SELECT query, calls, mean_time, total_time
FROM pg_stat_statements 
ORDER BY total_time DESC 
LIMIT 10;

-- Check for blocking queries
SELECT blocked_locks.pid AS blocked_pid,
       blocked_activity.usename AS blocked_user,
       blocking_locks.pid AS blocking_pid,
       blocking_activity.usename AS blocking_user,
       blocked_activity.query AS blocked_statement
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity 
  ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks 
  ON blocking_locks.locktype = blocked_locks.locktype
  AND blocking_locks.database IS NOT DISTINCT FROM blocked_locks.database
JOIN pg_catalog.pg_stat_activity blocking_activity 
  ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;
```

#### Network Connectivity Issues
```bash
# Test connectivity to services
nc -zv database-server 5432
nc -zv redis-server 6379
nc -zv prometheus-server 9090

# Check DNS resolution
dig pqc-scanner.terragon.ai
nslookup api.pqc-scanner.terragon.ai

# Monitor network traffic
netstat -tuln | grep LISTEN
ss -tulpn | grep :8080
```

### Log Analysis

```bash
# Search for errors in logs
grep -i "error\|exception\|failed" /opt/pqc-scanner/logs/application.log

# Monitor real-time logs
tail -f /opt/pqc-scanner/logs/application.log | grep -E "(ERROR|WARN|FATAL)"

# Analyze performance logs
awk '/scan_duration/ {sum+=$NF; count++} END {print "Average scan time:", sum/count "ms"}' \
  /opt/pqc-scanner/logs/performance.log
```

## 📞 Support and Contact

### Production Support Contacts

- **Technical Support**: support@terragon.ai
- **Security Issues**: security@terragon.ai  
- **Operations Team**: ops@terragon.ai
- **Emergency On-Call**: +1-555-PQC-SCAN

### Documentation and Resources

- **API Documentation**: https://docs.pqc-scanner.terragon.ai
- **Status Page**: https://status.pqc-scanner.terragon.ai
- **Security Advisories**: https://security.pqc-scanner.terragon.ai
- **Community Forum**: https://community.terragon.ai

---

**Document Version**: 1.0.0  
**Last Updated**: August 11, 2025  
**Next Review**: September 11, 2025

*This deployment guide is part of the PQC IoT Retrofit Scanner Generation 4 release with autonomous SDLC capabilities.*