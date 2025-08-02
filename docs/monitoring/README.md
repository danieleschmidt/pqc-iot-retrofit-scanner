# Monitoring and Observability

Comprehensive monitoring and observability setup for the PQC IoT Retrofit Scanner, including health checks, performance monitoring, structured logging, and metrics collection.

## Overview

The monitoring stack provides:

- **Health Checks**: System and application health monitoring
- **Performance Monitoring**: Detailed performance metrics and profiling
- **Structured Logging**: JSON-formatted logs with correlation IDs and security context
- **Prometheus Metrics**: Comprehensive metrics for monitoring and alerting
- **Security Audit Logging**: Detailed audit trails for compliance and security analysis

## Components

### 1. Health Checks (`health_check.py`)

Comprehensive health checking system that monitors:

- Python version compatibility
- Critical dependencies availability
- System resources (memory, disk, CPU)
- File system permissions
- Cryptographic library functionality
- Container environment health

#### Usage

```python
from monitoring.health_check import health_checker, check_health

# Run all health checks
health_status = check_health()
print(f"Overall status: {health_status['overall_status']}")

# Run specific check
result = health_checker.run_check("memory")
print(f"Memory check: {result.status.value} - {result.message}")
```

#### Health Check Endpoints

```bash
# CLI health check
pqc-iot health

# HTTP endpoint (if web interface is enabled)
curl http://localhost:8080/health
```

### 2. Performance Monitoring (`performance_config.py`)

Performance monitoring and profiling utilities:

- Context manager for measuring operations
- Function decorators for automatic measurement
- System resource monitoring
- Performance profiling and statistics

#### Usage

```python
from monitoring.performance_config import measure_operation, measure_performance

# Context manager
with measure_operation("firmware_analysis", firmware_size=1024000) as metrics:
    # Perform analysis
    result = analyze_firmware(firmware_data)

# Decorator
@measure_performance("crypto_operation")
def sign_data(data, key):
    return crypto_sign(data, key)
```

#### Performance Metrics

- Operation duration (milliseconds)  
- CPU usage during operation
- Memory usage (current and peak)
- I/O operations (read/write bytes)
- Error tracking and context

### 3. Prometheus Metrics (`prometheus_metrics.py`)

Comprehensive metrics collection for Prometheus monitoring:

#### Application Metrics

```python
# Firmware analysis metrics
pqc_firmware_analyses_total{architecture="arm", status="success"}
pqc_firmware_analysis_duration_seconds{architecture="arm"}
pqc_firmware_size_bytes{architecture="arm"}

# Vulnerability detection metrics
pqc_vulnerabilities_detected_total{algorithm="RSA-2048", severity="high", architecture="arm"}
pqc_vulnerability_confidence{algorithm="RSA-2048", architecture="arm"}

# Patch generation metrics  
pqc_patches_generated_total{algorithm="dilithium2", target_device="STM32L4", status="success"}
pqc_patch_generation_duration_seconds{algorithm="dilithium2", target_device="STM32L4"}

# Cryptographic operation metrics
pqc_crypto_operations_total{operation="sign", algorithm="dilithium2", status="success"}
pqc_crypto_operation_duration_seconds{operation="sign", algorithm="dilithium2"}
```

#### System Metrics

```python
# Resource usage
pqc_system_cpu_usage_percent
pqc_system_memory_usage_bytes{type="available|used|total"}
pqc_process_memory_usage_bytes{type="rss|vms"}
pqc_open_file_descriptors

# Health and errors
pqc_health_check_status{check_name="memory"}
pqc_health_check_duration_seconds{check_name="memory"}
pqc_errors_total{component="scanner", error_type="ValueError"}
```

#### Usage

```python
from monitoring.prometheus_metrics import (
    record_firmware_analysis, 
    record_vulnerability,
    start_metrics_server
)

# Start metrics server
start_metrics_server()  # Serves on http://localhost:9090/metrics

# Record metrics
record_firmware_analysis("arm", 2.5, 1048576, "success")
record_vulnerability("RSA-2048", "high", "arm", 0.95)
```

### 4. Structured Logging (`structured_logging.py`)

JSON-structured logging with correlation IDs, security context, and audit trails:

#### Features

- **Correlation Tracking**: Request and correlation IDs for distributed tracing
- **Security Context**: User, session, and permission information
- **Security Audit Logging**: Specialized logging for security events
- **Performance Logging**: Performance metrics in log format
- **Structured JSON Output**: Machine-parseable log format

#### Usage

```python
from monitoring.structured_logging import (
    set_correlation_id, 
    set_security_context,
    log_vulnerability_detection,
    log_patch_generation
)

# Set context for request tracing
set_correlation_id("req-12345")
set_security_context(SecurityContext(
    user_id="user-456",
    session_id="sess-789"
))

# Security audit logging
log_vulnerability_detection(
    algorithm="RSA-2048",
    confidence=0.95,
    firmware_path="/path/to/firmware.bin",
    details={"key_size": 2048, "usage": "signature"}
)

log_patch_generation(
    algorithm="dilithium2",
    target_device="STM32L4", 
    patch_id="patch-123",
    success=True
)
```

#### Log Format

```json
{
  "timestamp": 1704067200.123,
  "datetime": "2024-01-01T00:00:00.123Z",
  "level": "INFO",
  "logger": "pqc_iot_retrofit.scanner",
  "message": "Vulnerability detected: RSA-2048 (confidence: 0.95)",
  "service": "pqc-iot-retrofit-scanner",
  "hostname": "scanner-01",
  "pid": 1234,
  "correlation_id": "req-12345",
  "request_id": "req-67890",
  "security": {
    "user_id": "user-456",
    "session_id": "sess-789"
  },
  "event_type": "vulnerability_detected",
  "security_event": true,
  "audit": true,
  "details": {
    "algorithm": "RSA-2048",
    "confidence": 0.95,
    "firmware_path": "/path/to/firmware.bin",
    "key_size": 2048,
    "usage": "signature"
  }
}
```

## Configuration

### Environment Variables

```bash
# Health checks
PQC_HEALTH_CHECK_ENABLED=true

# Performance monitoring
PQC_PERFORMANCE_MONITORING=true
PQC_PROFILING=true

# Metrics
PQC_METRICS_ENABLED=true
PQC_METRICS_SERVER=true
PQC_METRICS_PORT=9090
PQC_METRICS_PUSH=false
PQC_PUSHGATEWAY_URL=http://pushgateway:9091

# Logging
PQC_LOG_LEVEL=INFO
PQC_LOG_FORMAT=structured  # structured | simple
PQC_LOG_FILE=logs/pqc-scanner.log
PQC_AUDIT_LOGGING=true
PQC_AUDIT_LOG_FILE=logs/pqc-audit.log
PQC_PERFORMANCE_LOGGING=true
PQC_PERFORMANCE_LOG_LEVEL=DEBUG
```

### Logging Configuration

Custom logging configuration via JSON or YAML:

```yaml
# logging-config.yaml
version: 1
disable_existing_loggers: false

formatters:
  structured:
    (): monitoring.structured_logging.StructuredFormatter
    service_name: pqc-iot-retrofit-scanner
  
handlers:
  console:
    class: logging.StreamHandler
    formatter: structured
    stream: ext://sys.stdout
  
  file:
    class: logging.handlers.RotatingFileHandler
    filename: logs/pqc-scanner.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
    formatter: structured

loggers:
  pqc_iot_retrofit:
    level: INFO
    handlers: [console, file]
    propagate: false
    
  pqc.security.audit:
    level: INFO
    handlers: [console, audit_file]
    propagate: false
```

## Monitoring Stack Deployment

### Docker Compose

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

  loki:
    image: grafana/loki:latest
    ports:
      - "3100:3100"
    volumes:
      - ./monitoring/loki-config.yml:/etc/loki/local-config.yaml
    command: -config.file=/etc/loki/local-config.yaml

  promtail:
    image: grafana/promtail:latest
    volumes:
      - ./logs:/var/log/pqc
      - ./monitoring/promtail-config.yml:/etc/promtail/config.yml
    command: -config.file=/etc/promtail/config.yml

volumes:
  prometheus_data:
  grafana_data:
```

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'pqc-scanner'
    static_configs:
      - targets: ['pqc-dev:9090']
    scrape_interval: 5s
    metrics_path: /metrics

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Alert Rules

```yaml
# monitoring/alert_rules.yml
groups:
  - name: pqc_scanner_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(pqc_errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate in PQC Scanner"
          description: "Error rate is {{ $value }} errors per second"

      - alert: SlowFirmwareAnalysis
        expr: histogram_quantile(0.95, rate(pqc_firmware_analysis_duration_seconds_bucket[5m])) > 60
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Slow firmware analysis detected"
          description: "95th percentile analysis time is {{ $value }} seconds"

      - alert: HighMemoryUsage
        expr: pqc_system_memory_usage_bytes{type="used"} / pqc_system_memory_usage_bytes{type="total"} > 0.9
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value | humanizePercentage }}"

      - alert: ServiceDown
        expr: up{job="pqc-scanner"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "PQC Scanner service is down"
          description: "PQC Scanner has been down for more than 1 minute"
```

### Grafana Dashboards

The monitoring setup includes pre-configured Grafana dashboards for:

1. **Application Overview**: High-level application metrics and health
2. **Firmware Analysis**: Analysis performance and vulnerability detection
3. **Security Audit**: Security events and compliance monitoring  
4. **System Resources**: CPU, memory, disk, and network usage
5. **Error Tracking**: Error rates, types, and troubleshooting

## Integration Examples

### CLI Integration

```bash
# Check system health
pqc-iot health --json

# Monitor performance during analysis
PQC_PERFORMANCE_MONITORING=true pqc-iot scan firmware.bin --verbose

# Enable metrics server
PQC_METRICS_ENABLED=true PQC_METRICS_SERVER=true pqc-iot scan firmware.bin
```

### API Integration

```python
from monitoring.health_check import health_checker
from monitoring.prometheus_metrics import metrics_collector
from monitoring.structured_logging import set_correlation_id

@app.route('/health')
def health_check():
    results = health_checker.run_all_checks()
    summary = health_checker.get_health_summary(results)
    return jsonify(summary)

@app.route('/metrics')
def metrics():
    return metrics_collector.get_metrics(), 200, {'Content-Type': 'text/plain'}

@app.before_request
def before_request():
    correlation_id = request.headers.get('X-Correlation-ID', str(uuid.uuid4()))
    set_correlation_id(correlation_id)
```

## Troubleshooting

### Common Issues

1. **Metrics Server Won't Start**
   ```bash
   # Check if prometheus_client is installed
   pip install prometheus_client
   
   # Verify port is available
   netstat -tulpn | grep 9090
   ```

2. **Logs Not Structured**
   ```bash
   # Set log format
   export PQC_LOG_FORMAT=structured
   
   # Check log directory permissions
   mkdir -p logs && chmod 755 logs
   ```

3. **Health Checks Failing**
   ```bash
   # Run individual health check
   python -c "from monitoring.health_check import health_checker; print(health_checker.run_check('memory'))"
   
   # Check system resources
   python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"
   ```

4. **Performance Monitoring Disabled**
   ```bash
   # Enable performance monitoring
   export PQC_PERFORMANCE_MONITORING=true
   export PQC_PROFILING=true
   ```

### Debug Mode

Enable debug logging for troubleshooting:

```bash
export PQC_LOG_LEVEL=DEBUG
export PQC_PERFORMANCE_LOG_LEVEL=DEBUG
pqc-iot scan firmware.bin --verbose
```

---

For more details on specific monitoring components, see:
- [Health Check Configuration](health-checks.md)
- [Metrics Collection](metrics.md)
- [Log Analysis](logging.md)
- [Alert Configuration](alerts.md)