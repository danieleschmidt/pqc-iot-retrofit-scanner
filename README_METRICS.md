# Metrics & Automation Guide

This document describes the comprehensive metrics collection, health monitoring, and automation systems implemented for the PQC IoT Retrofit Scanner project.

## Overview

The project includes a sophisticated metrics and automation infrastructure designed to:

- **Track Development Health**: Monitor code quality, security, performance, and maintainability
- **Automate Quality Gates**: Ensure consistent quality standards across releases  
- **Provide Insights**: Generate actionable reports and recommendations
- **Enable Continuous Improvement**: Track trends and identify areas for enhancement

## Architecture

### Core Components

```
├── .github/
│   ├── project-metrics.json          # Metrics configuration and thresholds
│   ├── automation-config.json        # Automation settings and schedules
│   ├── metrics-results.json          # Latest metrics data
│   └── health-report.json            # Current health status
├── scripts/
│   ├── collect-metrics.py            # Comprehensive metrics collector
│   ├── health-monitor.py             # Health monitoring and alerting
│   ├── benchmark-runner.py           # Performance benchmarking
│   ├── technical-debt-tracker.py     # Technical debt analysis
│   └── setup-automation.sh           # Automation setup script
└── monitoring/
    ├── prometheus_metrics.py         # Prometheus integration
    └── structured_logging.py         # Centralized logging
```

## Metrics Categories

### 1. Code Quality Metrics

**Test Coverage**
- Target: 90%
- Thresholds: Excellent (95%), Good (85%), Warning (70%), Critical (50%)
- Sources: codecov, pytest-cov

**Code Complexity**
- Target: ≤10 cyclomatic complexity
- Analysis: Function and class-level complexity monitoring
- Tools: radon, mccabe

**Technical Debt Ratio**
- Target: ≤5%
- Tracking: TODO/FIXME comments, code duplication, maintainability issues
- Sources: Static analysis, custom debt tracker

### 2. Security Metrics

**Vulnerability Count**
- Target: 0 vulnerabilities
- Breakdown: Critical, High, Medium, Low severity
- Tools: bandit, safety, semgrep, trivy

**Dependency Vulnerabilities**
- Target: 0 vulnerable dependencies
- Sources: pip-audit, osv-scanner, safety
- Auto-updates: Dependabot integration

### 3. Performance Metrics

**Build Time**
- Target: ≤120 seconds
- Monitoring: CI/CD pipeline performance
- Trends: Track performance regression

**Memory Usage**
- Target: ≤256 MB for typical operations
- Profiling: Memory usage patterns and leaks
- Tools: memory-profiler, pytest-benchmark

**Docker Image Size**
- Target: ≤500 MB
- Optimization: Multi-stage builds, layer caching
- Security: Regular base image updates

### 4. Business Metrics

**Scan Accuracy**
- Target: ≥95% accuracy
- Validation: False positive/negative rates
- User Feedback: Community-driven validation

**User Adoption**
- Tracking: Active users, feature usage
- Growth: Monthly active users trends

## Automation Workflows

### Daily Operations

**06:00 UTC - Metrics Collection**
```bash
# Automated daily metrics gathering
python scripts/collect-metrics.py --config .github/project-metrics.json
```

**Every 6 Hours - Health Monitoring**
```bash
# Continuous health monitoring with alerting
python scripts/health-monitor.py --send-alerts
```

**03:00 UTC - Security Scanning**
```bash
# Daily security vulnerability scanning
python scripts/collect-metrics.py --focus security
```

### Weekly Operations

**Sunday 02:00 UTC - Performance Benchmarks**
```bash
# Comprehensive performance benchmarking
python scripts/benchmark-runner.py --category all
```

**Monday 04:00 UTC - Technical Debt Analysis**
```bash
# Weekly technical debt assessment
python scripts/technical-debt-tracker.py
```

## Setup and Configuration

### Quick Start

1. **Run Automation Setup**
   ```bash
   ./scripts/setup-automation.sh --environment production
   ```

2. **Configure Environment Variables**
   ```bash
   cp .env.automation .env.automation.local
   # Edit .env.automation.local with your actual values
   ```

3. **Test Manual Execution**
   ```bash
   python scripts/collect-metrics.py --help
   python scripts/health-monitor.py --help
   ```

### Environment-Specific Configuration

**Development**
```bash
./scripts/setup-automation.sh --environment development --dry-run
```

**Production**  
```bash
./scripts/setup-automation.sh --environment production
```

### GitHub Integration

The automation integrates with GitHub through:

- **Actions Workflows**: `.github/workflows/automation.yml`
- **Repository Secrets**: Configure tokens and webhook URLs
- **Issue Creation**: Automatic issue creation for critical alerts
- **Status Checks**: Quality gates for pull requests

### Required Secrets

Set these in GitHub Repository Settings > Secrets:

```bash
GITHUB_TOKEN=<personal_access_token>
SLACK_WEBHOOK_URL=<slack_webhook_for_alerts>
CODECOV_TOKEN=<codecov_integration_token>
TECH_LEAD_EMAIL=<technical_lead_email>
```

## Monitoring and Alerting

### Alert Severity Levels

**Critical**
- 0 tolerance for security vulnerabilities
- Immediate Slack + Email + GitHub issue
- Escalation: 15 minutes

**Warning**
- Quality threshold violations
- Slack notification
- Escalation: 2 hours  

**Info**
- Metrics collection status
- Email digest

### Notification Channels

**Slack Integration**
```json
{
  "webhook_url": "${SLACK_WEBHOOK_URL}",
  "channels": {
    "critical": "#pqc-scanner-critical",
    "warning": "#pqc-scanner-warnings",
    "info": "#pqc-scanner-info"
  }
}
```

**Email Notifications**
- HTML formatted reports
- Attachment support for detailed metrics
- Environment-specific recipient lists

## Advanced Features

### Custom Metrics

Extend metrics collection by modifying `project-metrics.json`:

```json
{
  "custom_metrics": {
    "firmware_analysis_speed": {
      "target": 100,
      "unit": "files_per_hour",
      "thresholds": {
        "excellent": 200,
        "good": 100,
        "warning": 50,
        "critical": 25
      }
    }
  }
}
```

### Prometheus Integration

Optional Prometheus metrics export:

```python
from monitoring.prometheus_metrics import MetricsCollector

collector = MetricsCollector()
collector.record_firmware_scan(architecture="arm", duration=1.5, vulnerabilities=3)
```

### Technical Debt Tracking

Comprehensive debt analysis including:

- **Comment-based Debt**: TODO, FIXME, HACK comments
- **Complexity Debt**: High cyclomatic complexity functions
- **Duplication Debt**: Code duplication detection
- **Pattern Debt**: Outdated coding patterns
- **Maintainability Debt**: Large files, poor structure

## Reporting

### Automated Reports

**Weekly Summary**
- Monday 09:00 UTC
- Recipients: Tech leads, Product managers
- Format: Slack + Email

**Monthly Executive Report**
- First day of month, 10:00 UTC
- Recipients: Stakeholders, Executives
- Format: Executive dashboard

**Quarterly Retrospective**
- Quarterly, first day 11:00 UTC
- Recipients: All team
- Format: Comprehensive analysis with trends

### Manual Report Generation

```bash
# Generate comprehensive metrics report
python scripts/collect-metrics.py --output custom-report.json

# Generate health report
python scripts/health-monitor.py --output health-status.json

# Generate technical debt analysis
python scripts/technical-debt-tracker.py --output debt-analysis.json

# Run performance benchmarks
python scripts/benchmark-runner.py --output benchmark-results.json
```

## Troubleshooting

### Common Issues

**Metrics Collection Fails**
```bash
# Check dependencies
pip install -r requirements-automation.txt

# Validate configuration
python -c "import json; json.load(open('.github/project-metrics.json'))"

# Test GitHub API access
python scripts/collect-metrics.py --help
```

**Health Monitoring Alerts Not Sending**
```bash
# Check Slack webhook
curl -X POST -H 'Content-type: application/json' \
    --data '{"text":"Test message"}' \
    $SLACK_WEBHOOK_URL

# Verify environment variables
python -c "import os; print(os.getenv('SLACK_WEBHOOK_URL'))"
```

**Performance Benchmarks Running Slowly**
```bash
# Run with reduced iterations
python scripts/benchmark-runner.py --iterations 2 --category analysis

# Check system resources
python -c "import psutil; print(f'CPU: {psutil.cpu_count()}, RAM: {psutil.virtual_memory().total//1024//1024//1024}GB')"
```

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
python scripts/collect-metrics.py --verbose
python scripts/health-monitor.py --verbose  
python scripts/benchmark-runner.py --quiet
```

## Best Practices

### Configuration Management

1. **Version Control**: Keep configurations in git
2. **Environment Separation**: Use environment-specific configs
3. **Secret Management**: Never commit secrets, use environment variables
4. **Validation**: Always validate configuration changes

### Monitoring Strategy

1. **Threshold Tuning**: Regularly review and adjust thresholds
2. **Alert Fatigue**: Minimize false positives
3. **Trend Analysis**: Focus on trends over absolute values
4. **Actionable Alerts**: Ensure every alert has a clear action

### Performance Optimization

1. **Incremental Collection**: Only collect changed metrics when possible
2. **Parallel Processing**: Use concurrency for independent operations
3. **Caching**: Cache expensive operations
4. **Resource Limits**: Set appropriate timeouts and resource limits

## Future Enhancements

### Planned Features

- **ML-based Anomaly Detection**: Automatic threshold adjustment
- **Cost Tracking**: Infrastructure and development cost monitoring
- **Predictive Analytics**: Trend prediction and capacity planning
- **Integration Expansion**: Additional tool integrations (SonarQube, etc.)

### Community Contributions

Contributions welcome for:

- Additional metrics collectors
- New alerting channels
- Performance optimizations
- Documentation improvements

## Support

For questions or issues:

1. Check this documentation
2. Review automation logs in `logs/automation/`
3. Test individual scripts manually
4. Create GitHub issue with automation label

---

*This metrics and automation system is designed to scale with the project and provide actionable insights for continuous improvement. Regular review and tuning of thresholds and configurations is recommended.*