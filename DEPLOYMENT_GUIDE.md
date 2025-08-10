# 🚀 PQC IoT Retrofit Scanner - Generation 4 Deployment Guide

## Overview

This guide covers deployment of the **PQC IoT Retrofit Scanner Generation 4** - an advanced AI-powered CLI tool for auditing embedded firmware and generating post-quantum cryptography solutions.

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 2GB minimum, 8GB recommended
- **Storage**: 1GB free space
- **OS**: Linux, macOS, Windows

### Recommended Requirements
- **Python**: 3.10+
- **RAM**: 16GB for AI features
- **Storage**: 5GB for full research datasets
- **CPU**: Multi-core processor (4+ cores recommended)

## 🔧 Installation Options

### Option 1: Standard Installation (Core Features)
```bash
# Install core dependencies
pip install -e .

# Verify installation
python -m pqc_iot_retrofit.cli --version
```

### Option 2: Full Installation (All Features)
```bash
# Install with all optional dependencies
pip install -e ".[analysis,dev]"

# Install additional AI dependencies
pip install numpy scipy scikit-learn

# Verify Generation 4 features
python -c "from pqc_iot_retrofit.adaptive_ai import adaptive_ai; print('✅ AI features ready')"
```

### Option 3: Docker Deployment
```bash
# Build Docker image
docker build -t pqc-iot-scanner .

# Run with Docker
docker run -v $(pwd)/firmware:/firmware pqc-iot-scanner scan /firmware/sample.bin --arch cortex-m4
```

### Option 4: Development Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## 🌟 Feature Tiers

### Tier 1: Core Scanner (Always Available)
- ✅ Firmware pattern detection
- ✅ Multi-architecture support
- ✅ Risk assessment
- ✅ Basic reporting
- ✅ CLI interface

**Dependencies**: Python standard library only

### Tier 2: Enhanced Analysis
- ✅ Advanced disassembly (requires `capstone`)
- ✅ Binary format parsing (requires `lief`)
- ✅ Rich CLI interface (requires `rich`)
- ✅ Performance optimization

**Dependencies**: `pip install capstone lief rich`

### Tier 3: Generation 4 AI Features
- 🧠 Adaptive AI ensemble detection
- 🔮 Quantum resilience analysis
- 🧪 Autonomous research capabilities
- ⚡ Real-time learning and optimization

**Dependencies**: `pip install numpy scipy scikit-learn`

## 🚀 Quick Start

### 1. Basic Firmware Scan
```bash
# Scan firmware with core features
python -m pqc_iot_retrofit.cli scan firmware.bin --arch cortex-m4 --output report.json

# View results
cat report.json | jq '.scan_summary'
```

### 2. Enhanced Scan (with dependencies)
```bash
# Install enhanced dependencies
pip install capstone lief rich

# Run enhanced scan
python -m pqc_iot_retrofit.cli_enhanced scan firmware.bin --arch cortex-m4 --verbose
```

### 3. Generation 4 AI-Powered Analysis
```bash
# Install AI dependencies
pip install numpy scipy

# Run AI-powered scan
python -m pqc_iot_retrofit.cli_gen4 scan firmware.bin --arch cortex-m4 --quantum-timeline --adaptive-patches

# Train AI models
python -m pqc_iot_retrofit.cli_gen4 train-ai baseline1.bin baseline2.bin baseline3.bin

# Run autonomous research
python -m pqc_iot_retrofit.cli_gen4 research --objectives performance_benchmarking --duration 30
```

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  Generation 4 Architecture              │
├─────────────────────────────────────────────────────────┤
│  CLI Layer                                              │
│  ├── cli.py (Core)                                      │
│  ├── cli_enhanced.py (Generation 3)                     │
│  └── cli_gen4.py (Generation 4 AI)                      │
├─────────────────────────────────────────────────────────┤
│  AI & Research Layer (Generation 4)                     │
│  ├── adaptive_ai.py (Ensemble ML)                       │
│  ├── quantum_resilience.py (Quantum Analysis)           │
│  └── autonomous_research.py (Scientific Discovery)      │
├─────────────────────────────────────────────────────────┤
│  Analysis Layer (Generation 3)                          │
│  ├── performance.py (Optimization)                      │
│  ├── concurrency.py (Parallel Processing)               │
│  └── monitoring.py (Observability)                      │
├─────────────────────────────────────────────────────────┤
│  Core Layer (Always Available)                          │
│  ├── scanner.py (Pattern Detection)                     │
│  ├── patcher.py (PQC Generation)                        │
│  └── error_handling.py (Reliability)                    │
└─────────────────────────────────────────────────────────┘
```

## 🐳 Docker Deployment

### Dockerfile (Production)
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install -e ".[analysis]"

# Optional: Install AI dependencies
RUN pip install numpy scipy scikit-learn

ENTRYPOINT ["python", "-m", "pqc_iot_retrofit.cli"]
```

### Docker Compose (Development)
```yaml
version: '3.8'
services:
  scanner:
    build: .
    volumes:
      - ./firmware:/firmware:ro
      - ./reports:/reports
    environment:
      - PYTHONUNBUFFERED=1
    command: ["scan", "/firmware/sample.bin", "--arch", "cortex-m4"]
  
  research:
    build: .
    volumes:
      - ./models:/app/models
      - ./research:/app/research
    command: ["research", "--objectives", "performance_benchmarking"]
```

## 🔒 Security Configuration

### Production Security Checklist
- [ ] Use virtual environments or containers
- [ ] Validate all firmware inputs
- [ ] Limit memory and CPU usage
- [ ] Enable logging and monitoring
- [ ] Restrict file system access
- [ ] Use least-privilege principles

### Security Configuration
```python
# config/security.py
SECURITY_CONFIG = {
    'max_firmware_size': 100 * 1024 * 1024,  # 100MB
    'max_analysis_time': 300,  # 5 minutes
    'max_memory_usage': 2 * 1024 * 1024 * 1024,  # 2GB
    'allowed_file_types': ['.bin', '.hex', '.elf'],
    'enable_sandboxing': True,
    'log_security_events': True
}
```

## 📊 Monitoring & Observability

### Metrics Collection
```bash
# Enable Prometheus metrics
export ENABLE_METRICS=true
export METRICS_PORT=8080

# Run with monitoring
python -m pqc_iot_retrofit.cli scan firmware.bin --arch cortex-m4
```

### Health Checks
```bash
# Check system health
python -m pqc_iot_retrofit.monitoring.health_check

# Monitor performance
python -m pqc_iot_retrofit.monitoring.performance_config
```

### Logging Configuration
```python
# config/logging.py
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'pqc_scanner.log',
            'formatter': 'detailed',
        },
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'detailed',
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['file', 'console']
    }
}
```

## 🧪 Testing Strategy

### Test Levels
1. **Unit Tests**: Individual component testing
2. **Integration Tests**: Cross-component functionality
3. **Performance Tests**: Load and stress testing
4. **Security Tests**: Vulnerability and penetration testing
5. **End-to-End Tests**: Complete workflow validation

### Running Tests
```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests  
pytest tests/integration/ -v

# Generation 4 tests
pytest tests/test_generation4_integration.py -v

# Security validation
pytest tests/test_security_validation.py -v

# Performance tests
pytest tests/benchmarks/ -v

# Full test suite
pytest tests/ -v --cov=src --cov-report=html
```

## 🚀 Production Deployment Scenarios

### Scenario 1: Enterprise Security Team
```bash
# Install with security focus
pip install -e ".[analysis]"

# Configure security policies
export PQC_MAX_FIRMWARE_SIZE=50MB
export PQC_ENABLE_SANDBOXING=true

# Run with monitoring
python -m pqc_iot_retrofit.cli scan \
  --arch cortex-m4 \
  --output-format json \
  --security-level high \
  firmware_samples/*.bin
```

### Scenario 2: IoT Device Manufacturer
```bash
# Install with AI features for optimization
pip install -e ".[analysis]" numpy scipy

# Train models on known-good firmware
python -m pqc_iot_retrofit.cli_gen4 train-ai \
  baseline_firmware/*.bin

# Automated CI/CD integration
python -m pqc_iot_retrofit.cli scan \
  --arch esp32 \
  --output ci_report.json \
  --fail-on critical \
  build/firmware.bin
```

### Scenario 3: Research Institution
```bash
# Full installation for research
pip install -e ".[analysis,dev]" numpy scipy scikit-learn

# Autonomous research mode
python -m pqc_iot_retrofit.cli_gen4 research \
  --objectives performance_benchmarking algorithm_optimization \
  --duration 480 \
  --auto-publish

# Generate research publications
python -m pqc_iot_retrofit.autonomous_research \
  generate-publication \
  --experiment-data research_results/ \
  --output research_paper.md
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Error: No module named 'numpy'
pip install numpy scipy

# Error: No module named 'capstone' 
pip install capstone

# Error: No module named 'lief'
pip install lief
```

#### 2. Memory Issues
```bash
# Reduce memory usage
export PQC_MAX_MEMORY=1GB
export PQC_DISABLE_CACHING=true

# Use streaming mode for large firmware
python -m pqc_iot_retrofit.cli scan --streaming firmware.bin
```

#### 3. Performance Issues
```bash
# Enable performance mode
export PQC_PERFORMANCE_MODE=true
export PQC_MAX_WORKERS=4

# Use faster algorithms
python -m pqc_iot_retrofit.cli scan --fast-mode firmware.bin
```

#### 4. AI Model Issues
```bash
# Reset AI models
rm -rf models/
python -m pqc_iot_retrofit.cli_gen4 train-ai baseline/*.bin

# Check AI system status
python -m pqc_iot_retrofit.cli_gen4 ai-status
```

### Debug Mode
```bash
# Enable debug logging
export PQC_LOG_LEVEL=DEBUG

# Verbose output
python -m pqc_iot_retrofit.cli scan firmware.bin --verbose --debug

# Profile performance
python -m cProfile -o profile.stats \
  -m pqc_iot_retrofit.cli scan firmware.bin
```

## 📈 Performance Tuning

### Optimization Settings
```bash
# CPU optimization
export PQC_MAX_WORKERS=$(nproc)
export PQC_ENABLE_CONCURRENCY=true

# Memory optimization  
export PQC_CACHE_SIZE=512MB
export PQC_ENABLE_STREAMING=true

# AI optimization
export PQC_AI_BATCH_SIZE=32
export PQC_AI_CONFIDENCE_THRESHOLD=0.8
```

### Hardware Recommendations
- **CPU**: Intel/AMD 64-bit, 4+ cores
- **RAM**: 8GB minimum, 16GB for AI features
- **Storage**: SSD recommended for better I/O
- **GPU**: Optional, CUDA support for AI acceleration

## 🔄 Updates and Maintenance

### Update Strategy
```bash
# Check for updates
git fetch origin
git status

# Update dependencies
pip install -U -r requirements.txt

# Update AI models
python -m pqc_iot_retrofit.cli_gen4 update-models

# Backup configuration
cp -r config/ config_backup_$(date +%Y%m%d)/
```

### Maintenance Tasks
- **Daily**: Check logs for errors
- **Weekly**: Update threat intelligence
- **Monthly**: Retrain AI models
- **Quarterly**: Security audit and updates

## 📞 Support and Resources

### Documentation
- **User Guide**: [docs/USER_GUIDE.md](docs/USER_GUIDE.md)
- **API Reference**: [docs/API_REFERENCE.md](docs/API_REFERENCE.md)
- **Architecture**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

### Community
- **Issues**: [GitHub Issues](https://github.com/terragon-ai/pqc-iot-retrofit-scanner/issues)
- **Discussions**: [GitHub Discussions](https://github.com/terragon-ai/pqc-iot-retrofit-scanner/discussions)
- **Security**: [SECURITY.md](SECURITY.md)

### Commercial Support
- **Enterprise Support**: contact@terragon.ai
- **Custom Development**: consulting@terragon.ai
- **Training Services**: training@terragon.ai

## 📜 License and Attribution

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

### Citation
```bibtex
@software{pqc_iot_retrofit_scanner_2024,
  title={PQC IoT Retrofit Scanner - Generation 4},
  author={Daniel Schmidt and Terragon Labs},
  year={2024},
  url={https://github.com/terragon-ai/pqc-iot-retrofit-scanner},
  version={1.0.0}
}
```

---

**🚀 Ready to deploy Generation 4 PQC IoT security analysis!**