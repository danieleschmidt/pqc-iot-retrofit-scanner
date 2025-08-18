# Quick Start Guide

Get started with the PQC IoT Retrofit Scanner in 5 minutes.

## Prerequisites

- Python 3.8 or higher
- Docker (optional, for containerized analysis)
- 1GB free disk space
- Internet connection for initial setup

## Installation

### Option 1: pip install (Recommended)

```bash
# Install the CLI tool
pip install pqc-iot-retrofit-scanner

# Verify installation
pqc-iot scan --version
```

### Option 2: Docker

```bash
# Pull the latest image
docker pull terragon/pqc-iot-retrofit:latest

# Run with current directory mounted
docker run -v $(pwd):/workspace terragon/pqc-iot-retrofit:latest --help
```

### Option 3: From Source

```bash
# Clone the repository
git clone https://github.com/terragon-ai/pqc-iot-retrofit-scanner.git
cd pqc-iot-retrofit-scanner

# Install in development mode
pip install -e .

# Run tests to verify setup
pytest tests/
```

## Basic Usage

### 1. Analyze Firmware

```bash
# Scan a single firmware file
pqc-iot scan firmware.bin --arch cortex-m4

# Scan with detailed output
pqc-iot scan firmware.bin --arch cortex-m4 --verbose --output report.json
```

### 2. Generate PQC Patches

```bash
# Generate patches for vulnerabilities
pqc-iot scan firmware.bin \
  --arch cortex-m4 \
  --generate-patches \
  --patch-dir ./patches/

# View generated patches
ls -la patches/
```

### 3. Quick Analysis Report

```bash
# Get a summary report
pqc-iot report patches/analysis.json --format summary

# Generate detailed HTML report
pqc-iot report patches/analysis.json --format html --output vulnerabilities.html
```

## Example Workflow

Here's a complete example analyzing a smart meter firmware:

```bash
# Step 1: Download sample firmware
curl -O https://example.com/smart-meter-v2.3.bin

# Step 2: Analyze for vulnerabilities
pqc-iot scan smart-meter-v2.3.bin \
  --arch cortex-m4 \
  --memory-flash 512KB \
  --memory-ram 128KB \
  --output analysis.json \
  --verbose

# Step 3: Generate PQC patches
pqc-iot patch analysis.json \
  --target-device STM32L4 \
  --optimization size \
  --output-dir pqc-patches/

# Step 4: Create deployment package
pqc-iot package pqc-patches/ \
  --format ota \
  --compression lzma \
  --signing-key deploy.pem \
  --output smart-meter-v2.4-pqc.bin
```

## Understanding Results

### Vulnerability Report

```json
{
  "firmware_info": {
    "file": "smart-meter-v2.3.bin",
    "size": 524288,
    "architecture": "cortex-m4",
    "detected_crypto": 3
  },
  "vulnerabilities": [
    {
      "id": "RSA-2048-001",
      "algorithm": "RSA-2048",
      "location": "0x08001234",
      "risk_level": "high",
      "pqc_replacement": "Dilithium2",
      "estimated_patch_size": "87KB"
    }
  ]
}
```

### Risk Levels

- **Critical**: Immediate attention required (e.g., RSA-1024)
- **High**: Should be addressed soon (e.g., RSA-2048, ECDSA-P256)
- **Medium**: Address in next update cycle (e.g., AES-128)
- **Low**: Monitor for future updates (e.g., secure implementations)

## Common Architectures

| Architecture | Command | Notes |
|-------------|---------|--------|
| ARM Cortex-M4 | `--arch cortex-m4` | Most common IoT MCU |
| ESP32 | `--arch esp32` | Wi-Fi enabled MCU |
| RISC-V | `--arch riscv32` | Open source architecture |
| AVR | `--arch avr` | Arduino and similar |

## Next Steps

- **Fleet Analysis**: See [Fleet Management Guide](fleet-management.md)
- **Custom Integration**: Check [Python API Reference](../api/python-api.md)
- **Production Deployment**: Read [Migration Guide](migration-guide.md)
- **Contributing**: Review [Development Setup](dev/development-setup.md)

## Getting Help

- 📖 **Documentation**: [Full Documentation](../README.md)
- 🐛 **Issues**: [GitHub Issues](https://github.com/terragon-ai/pqc-iot-retrofit-scanner/issues)
- 💬 **Discussion**: [GitHub Discussions](https://github.com/terragon-ai/pqc-iot-retrofit-scanner/discussions)
- 🔒 **Security**: [Security Policy](../SECURITY.md)

## Troubleshooting

### Common Issues

**Error: "Architecture not detected"**
```bash
# Manually specify architecture
pqc-iot scan firmware.bin --arch cortex-m4 --force
```

**Error: "Insufficient memory for PQC"**
```bash
# Use smaller algorithm variant
pqc-iot scan firmware.bin --pqc-variant small --memory-constrained
```

**Error: "No crypto patterns found"**
```bash
# Increase detection sensitivity
pqc-iot scan firmware.bin --sensitivity high --include-weak-crypto
```

---

*Need more detailed information? Check our [comprehensive documentation](../README.md) or open an [issue](https://github.com/terragon-ai/pqc-iot-retrofit-scanner/issues).*