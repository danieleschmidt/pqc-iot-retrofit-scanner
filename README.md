# pqc-iot-retrofit-scanner

> CLI + GitHub Action that audits embedded firmware and suggests post-quantum cryptography drop-ins (Kyber, Dilithium)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Firmware](https://img.shields.io/badge/Firmware-IoT-orange.svg)](https://www.arm.com/)
[![NIST PQC](https://img.shields.io/badge/NIST-PQC-blue.svg)](https://csrc.nist.gov/projects/post-quantum-cryptography)

## 🔐 Overview

**pqc-iot-retrofit-scanner** addresses the critical security gap in IoT devices facing quantum threats. Motivated by TechRadar's "smart-meter PQC challenge" and ISACA's call for 2035 compliance, this tool automatically scans embedded firmware, identifies quantum-vulnerable cryptography, and generates drop-in post-quantum replacements.

## ✨ Key Features

- **Firmware Analysis**: Deep scanning of binary firmware images
- **Multi-Architecture Support**: ARM Cortex-M, ESP32, RISC-V, AVR
- **OTA Patch Generation**: Ready-to-deploy over-the-air updates
- **Risk Heat Maps**: Visual vulnerability assessment with SBOM integration
- **Minimal Overhead**: PQC implementations optimized for constrained devices

## 📊 IoT Threat Landscape

| Device Type | Vulnerable Devices | Average Lifespan | PQC Ready |
|-------------|-------------------|------------------|-----------|
| Smart Meters | 1.2B | 15-20 years | <1% |
| Industrial Sensors | 800M | 10-15 years | <2% |
| Connected Cars | 400M | 8-12 years | <5% |
| Medical Devices | 300M | 10-15 years | <1% |

## 🚀 Quick Start

### Installation

```bash
# Install CLI tool
pip install pqc-iot-retrofit-scanner

# Install with firmware analysis tools
pip install pqc-iot-retrofit-scanner[analysis]

# Verify installation
pqc-iot scan --version
```

### Basic Firmware Scan

```bash
# Scan single firmware image
pqc-iot scan firmware.bin --arch cortex-m4 --output report.json

# Scan with patch generation
pqc-iot scan firmware.bin \
  --arch cortex-m4 \
  --generate-patches \
  --patch-dir patches/

# Batch scan IoT fleet
pqc-iot scan-fleet \
  --manifest fleet_manifest.json \
  --parallel 8 \
  --risk-threshold high
```

### Python API Usage

```python
from pqc_iot_retrofit import FirmwareScanner, PQCPatcher

# Initialize scanner
scanner = FirmwareScanner(
    architecture="cortex-m4",
    memory_constraints={"flash": 512*1024, "ram": 64*1024}
)

# Scan firmware
vulnerabilities = scanner.scan_firmware(
    firmware_path="smart_meter_v2.3.bin",
    base_address=0x08000000
)

print(f"Found {len(vulnerabilities)} quantum-vulnerable crypto implementations")

# Generate PQC patches
patcher = PQCPatcher(
    target_device="STM32L4",
    optimization_level="size"  # Optimize for code size
)

for vuln in vulnerabilities:
    if vuln.algorithm == "RSA-2048":
        patch = patcher.create_dilithium_patch(
            vuln,
            security_level=2,  # NIST Level 2
            stack_size=vuln.available_stack
        )
    elif vuln.algorithm in ["ECDH-P256", "ECDSA-P256"]:
        patch = patcher.create_kyber_patch(
            vuln,
            security_level=1,
            shared_memory=True  # Share memory between Kyber operations
        )
    
    patch.save(f"patches/{vuln.function_name}.patch")
```

### Risk Assessment Dashboard

```python
from pqc_iot_retrofit.visualization import RiskDashboard

# Generate fleet-wide risk assessment
dashboard = RiskDashboard()

# Analyze device fleet
fleet_analysis = dashboard.analyze_fleet(
    device_manifest="fleet_devices.json",
    firmware_database="firmware_versions.db"
)

# Generate interactive heat map
dashboard.create_risk_heatmap(
    fleet_analysis,
    dimensions=["device_type", "firmware_version", "deployment_region"],
    risk_factors={
        "crypto_strength": 0.4,
        "device_lifespan": 0.3,
        "update_capability": 0.2,
        "exposure_level": 0.1
    }
)

# Export reports
dashboard.export_executive_summary("pqc_risk_summary.pdf")
dashboard.export_sbom_diff("sbom_crypto_changes.json")
```

## 📊 Device-Specific Implementations

### Cortex-M Series

```python
from pqc_iot_retrofit.targets import CortexMTarget

# Cortex-M4 with DSP extensions
target = CortexMTarget(
    variant="M4F",
    has_dsp=True,
    has_fpu=True,
    flash_size=512*1024,
    ram_size=128*1024
)

# Generate optimized assembly
kyber_asm = target.generate_kyber_asm(
    variant="kyber512",
    use_dsp_instructions=True,
    unroll_loops=True
)

# Benchmark on target
benchmark = target.benchmark_implementation(
    kyber_asm,
    metrics=["cycles", "stack", "power"]
)

print(f"Key generation: {benchmark.keygen_cycles} cycles")
print(f"Encapsulation: {benchmark.encaps_cycles} cycles")
print(f"Decapsulation: {benchmark.decaps_cycles} cycles")
```

### ESP32 Integration

```python
from pqc_iot_retrofit.targets import ESP32Target

# ESP32 with hardware crypto acceleration
esp32 = ESP32Target(
    variant="ESP32-S3",
    use_hw_aes=True,
    use_hw_sha=True,
    psram_size=8*1024*1024  # 8MB PSRAM
)

# Port Dilithium with ESP-IDF integration
dilithium_component = esp32.create_component(
    algorithm="dilithium2",
    component_name="esp_dilithium",
    use_psram=True  # Offload tables to PSRAM
)

# Generate ESP-IDF component files
dilithium_component.generate_files("components/esp_dilithium/")

# Example integration code
example_code = """
#include "esp_dilithium.h"

void app_main() {
    // Initialize PQC
    esp_dilithium_init();
    
    // Generate keypair
    uint8_t pk[DILITHIUM2_PUBLICKEYBYTES];
    uint8_t sk[DILITHIUM2_SECRETKEYBYTES];
    esp_dilithium_keygen(pk, sk);
    
    // Sign message
    uint8_t msg[] = "Firmware v2.4.0";
    uint8_t sig[DILITHIUM2_BYTES];
    size_t siglen;
    
    esp_dilithium_sign(sig, &siglen, msg, sizeof(msg), sk);
}
"""
```

## 🛡️ Security Analysis

### Side-Channel Protection

```python
from pqc_iot_retrofit.security import SideChannelAnalyzer

analyzer = SideChannelAnalyzer()

# Analyze implementation for timing leaks
timing_report = analyzer.analyze_timing(
    implementation="kyber_cortexm4.bin",
    test_vectors=1000,
    statistical_tests=["t-test", "chi-squared"]
)

if timing_report.has_timing_leaks:
    print("⚠️ Timing vulnerabilities detected!")
    for leak in timing_report.leaks:
        print(f"  Function: {leak.function}")
        print(f"  Confidence: {leak.confidence:.1%}")
        print(f"  Suggested fix: {leak.mitigation}")

# Power analysis resistance
power_report = analyzer.simulate_power_analysis(
    implementation="dilithium_esp32.elf",
    traces=10000,
    noise_level=0.1
)
```

### Hybrid Mode Support

```python
from pqc_iot_retrofit.hybrid import HybridCrypto

# Implement hybrid classical + PQC
hybrid = HybridCrypto(
    classical="ECDSA-P256",
    pqc="Dilithium2",
    combination="concatenate"  # or "nested", "xor"
)

# Generate hybrid patches
hybrid_patch = hybrid.create_transition_patch(
    target_function=vuln.function,
    compatibility_mode=True,
    rollback_timeout=30*24*3600  # 30 days
)

# Compatibility testing
compat_report = hybrid.test_compatibility(
    old_devices=["device_v1.0", "device_v1.5"],
    new_devices=["device_v2.0"],
    test_scenarios=["key_exchange", "signatures", "firmware_update"]
)
```

## 📈 Fleet Management

### Batch Deployment

```python
from pqc_iot_retrofit.fleet import FleetManager

fleet = FleetManager(
    backend="aws_iot",
    region="us-east-1"
)

# Create deployment campaign
campaign = fleet.create_campaign(
    name="PQC_Retrofit_Phase1",
    target_devices=fleet.query_devices(
        "firmware_version < 2.4 AND device_type = 'smart_meter'"
    ),
    rollout_strategy={
        "type": "canary",
        "initial_percentage": 1,
        "increment": 10,
        "wait_between": 24*3600,  # 24 hours
        "success_criteria": {
            "error_rate": 0.01,
            "rollback_threshold": 0.05
        }
    }
)

# Monitor deployment
deployment_status = fleet.monitor_campaign(campaign.id)
print(f"Devices updated: {deployment_status.successful_count}")
print(f"Success rate: {deployment_status.success_rate:.1%}")
```

### Telemetry Collection

```python
from pqc_iot_retrofit.telemetry import CryptoTelemetry

telemetry = CryptoTelemetry()

# Collect crypto operation metrics
@telemetry.instrument
def pqc_sign_operation(message, key):
    return dilithium_sign(message, key)

# Aggregate fleet-wide statistics
stats = telemetry.get_fleet_statistics(
    time_range="last_7_days",
    metrics=[
        "crypto_operations_per_second",
        "average_signing_time",
        "memory_usage_p99",
        "failed_operations"
    ]
)

# Alert on anomalies
telemetry.set_alert(
    condition="signing_time > 100ms",
    action="notify",
    channels=["ops-team@company.com"]
)
```

## 🔬 Testing Framework

### Hardware-in-Loop Testing

```python
from pqc_iot_retrofit.testing import HILTestBench

# Setup hardware test bench
testbench = HILTestBench(
    devices=[
        {"type": "STM32L4", "port": "/dev/ttyUSB0"},
        {"type": "ESP32", "port": "/dev/ttyUSB1"},
        {"type": "nRF52840", "port": "/dev/ttyUSB2"}
    ]
)

# Run comprehensive tests
test_suite = testbench.create_test_suite([
    "functional_correctness",
    "performance_benchmarks",
    "power_consumption",
    "interoperability",
    "stress_testing"
])

results = testbench.run_tests(
    test_suite,
    firmware_variants=["original", "pqc_patched"],
    duration_hours=24
)

# Generate certification report
testbench.generate_certification_report(
    results,
    standards=["NIST", "Common Criteria"],
    output="pqc_certification_report.pdf"
)
```

### Fuzzing PQC Implementations

```python
from pqc_iot_retrofit.fuzzing import PQCFuzzer

fuzzer = PQCFuzzer(
    target="kyber_implementation.bin",
    sanitizers=["address", "undefined", "memory"]
)

# Fuzz with crypto-specific mutations
fuzzing_campaign = fuzzer.run(
    duration_hours=48,
    corpus="crypto_test_vectors/",
    mutations=[
        "bit_flip",
        "boundary_values",
        "invalid_lengths",
        "malformed_keys"
    ]
)

# Analyze crashes
for crash in fuzzing_campaign.unique_crashes:
    print(f"Crash type: {crash.type}")
    print(f"Input: {crash.input_hex}")
    print(f"Stack trace: {crash.stack_trace}")
```

## 📊 Compliance Reporting

### Regulatory Compliance

```python
from pqc_iot_retrofit.compliance import ComplianceReporter

reporter = ComplianceReporter()

# Generate compliance documentation
compliance_package = reporter.generate_package(
    device_family="industrial_sensors",
    standards=[
        "NIST_SP_800_208",  # PQC recommendations
        "ETSI_TR_103_619",  # IoT baseline security
        "IEC_62443"         # Industrial security
    ],
    evidence={
        "test_results": test_results,
        "vulnerability_scans": scan_results,
        "crypto_inventory": sbom_crypto
    }
)

# Validation against requirements
validation = reporter.validate_compliance(
    compliance_package,
    deadline="2035-01-01"
)

print(f"Compliance score: {validation.score:.1%}")
print(f"Critical gaps: {validation.critical_gaps}")
```

## 🚀 Deployment Strategies

### Gradual Migration Path

```yaml
# pqc_migration_plan.yaml
migration_phases:
  phase_1:
    name: "Assessment & Planning"
    duration: "3 months"
    activities:
      - Firmware inventory
      - Vulnerability scanning
      - Risk prioritization
      
  phase_2:
    name: "Pilot Deployment"
    duration: "6 months"
    targets:
      - device_types: ["smart_meters"]
      - percentage: 1%
      - regions: ["test_lab", "field_trial_site"]
      
  phase_3:
    name: "Staged Rollout"
    duration: "12 months"
    strategy:
      - week_1_4: 5%
      - week_5_8: 15%
      - week_9_16: 40%
      - week_17_24: 100%
      
  phase_4:
    name: "Legacy Support"
    duration: "24 months"
    approach: "hybrid_mode"
    sunset_date: "2027-01-01"
```

## 🛠️ Development Infrastructure

This project implements a comprehensive Software Development Lifecycle (SDLC) with enterprise-grade tooling:

### Development Environment
- **Containerized Development**: Full devcontainer support with Docker and docker-compose
- **Automated Testing**: Comprehensive test suite with unit, integration, e2e, and security tests
- **Code Quality**: ESLint, Prettier, and pre-commit hooks for consistent code quality
- **Performance Benchmarking**: Hardware-in-loop testing and benchmarking framework

### CI/CD & Automation
- **GitHub Actions Templates**: Ready-to-deploy CI/CD workflows in `docs/workflows/examples/`
- **Security Scanning**: Automated dependency scanning and vulnerability assessment
- **Metrics Collection**: Comprehensive project health and performance metrics
- **Automated Deployments**: OTA update generation and deployment automation

### Monitoring & Observability
- **Health Checks**: Automated system health monitoring with `monitoring/health_check.py`
- **Performance Metrics**: Prometheus-compatible metrics collection
- **Structured Logging**: Centralized logging with correlation IDs
- **Alerting**: Configurable alerting for operational anomalies

### Documentation & Governance
- **Architecture Decision Records**: Documented decisions in `docs/adr/`
- **Security Documentation**: Threat modeling and security checklists
- **Compliance**: NIST PQC and IoT security standard compliance tracking
- **Project Charter**: Clear scope, success criteria, and stakeholder alignment

### Repository Management
- **Branch Protection**: Automated review requirements and merge policies
- **Code Owners**: Automatic reviewer assignment via `CODEOWNERS`
- **Issue Templates**: Standardized bug reports and feature requests
- **Security Policy**: Clear vulnerability reporting and response procedures

For detailed setup instructions, see [docs/SETUP_REQUIRED.md](docs/SETUP_REQUIRED.md).

## 📚 Documentation

Full documentation: [https://pqc-iot-retrofit.readthedocs.io](https://pqc-iot-retrofit.readthedocs.io)

### Guides
- [IoT PQC Migration Guide](docs/guides/migration_guide.md)
- [Embedded Crypto Best Practices](docs/guides/embedded_crypto.md)
- [OTA Update Security](docs/guides/ota_security.md)
- [Compliance Roadmap](docs/guides/compliance.md)
- [Development Setup](docs/DEVELOPMENT.md)
- [Architecture Overview](docs/ARCHITECTURE.md)

## 🤝 Contributing

We welcome contributions! Priority areas:
- Additional MCU architectures
- Lightweight PQC variants
- Power analysis tools
- Formal verification

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 Citation

```bibtex
@article{pqc_iot_retrofit_scanner,
  title={Automated Post-Quantum Cryptography Retrofitting for IoT Devices},
  author={Daniel Schmidt},
  journal={IEEE Internet of Things Journal},
  year={2025},
  doi={10.1109/JIOT.2025.XXXXXX}
}
```

## 🏆 Acknowledgments

- NIST PQC standardization team
- ISACA for IoT security guidelines
- Open Quantum Safe project
- MCU vendor communities

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

## ⚠️ Security Notice

This tool identifies and patches cryptographic vulnerabilities. Always test patches thoroughly in isolated environments before production deployment. The authors are not responsible for any system failures or security breaches resulting from improper use.

## 🏗️ Architecture

### Firmware Analysis Pipeline

```python
from pqc_iot_retrofit.analysis import BinaryAnalyzer, CryptoDetector

class IoTFirmwareAnalyzer:
    def __init__(self, architecture):
        self.disassembler = BinaryAnalyzer(arch=architecture)
        self.crypto_detector = CryptoDetector()
        
    def analyze(self, firmware_bytes):
        # Disassemble firmware
        instructions = self.disassembler.disassemble(firmware_bytes)
        
        # Detect crypto functions
        crypto_functions = []
        for func in self.disassembler.extract_functions(instructions):
            # Pattern matching for crypto operations
            if self.crypto_detector.is_rsa_operation(func):
                crypto_functions.append({
                    "type": "RSA",
                    "address": func.address,
                    "key_size": self.detect_rsa_key_size(func),
                    "stack_usage": func.stack_frame_size
                })
            elif self.crypto_detector.is_ecc_operation(func):
                crypto_functions.append({
                    "type": "ECC",
                    "curve": self.detect_ecc_curve(func),
                    "address": func.address
                })
                
        return crypto_functions
```

### PQC Implementation Library

```c
// Optimized Dilithium for Cortex-M4
// pqc_patches/dilithium2_cortexm4.c

#include "dilithium2.h"
#include "symmetric-shake.h"

// Stack-optimized implementation
int crypto_sign_dilithium2_aes_m4(
    uint8_t *sm, size_t *smlen,
    const uint8_t *m, size_t mlen,
    const uint8_t *sk)
{
    // Use in-place operations to minimize stack
    poly mat[K][L];
    poly s1[L], s2[K], t[K];
    
    // Unpack secret key with minimal memory
    unpack_sk_inplace(sk, mat, s1, s2, t);
    
    // Sign with NTT optimizations for Cortex-M4
    return sign_constrained(sm, smlen, m, mlen, 
                           mat, s1, s2, t);
}

// Drop-in replacement for RSA signing
int rsa_sign_compat(uint8_t *sig, size_t *siglen,
                    const uint8_t *msg, size_t msglen,
                    const void *key)
{
    // Transparent wrapper
    return crypto_sign_dilithium2_aes_m4(
        sig, siglen, msg, msglen, 
        (const uint8_t *)key
    );
}
```

## 🔧 Advanced Features

### Memory-Constrained Optimization

```python
from pqc_iot_retrofit.optimization import MemoryOptimizer

optimizer = MemoryOptimizer(
    target_device="ESP32",
    available_flash=4*1024*1024,  # 4MB
    available_ram=520*1024,        # 520KB
    reserved_ram=100*1024          # Reserve 100KB for app
)

# Optimize PQC implementation
optimized_impl = optimizer.optimize_pqc(
    algorithm="kyber512",
    constraints={
        "max_stack": 8*1024,      # 8KB stack limit
        "max_heap": 16*1024,      # 16KB heap limit
        "shared_memory": True,     # Reuse memory buffers
        "in_place_ops": True      # Minimize copies
    }
)

print(f"Code size: {optimized_impl.code_size} bytes")
print(f"RAM usage: {optimized_impl.ram_usage} bytes")
print(f"Stack depth: {optimized_impl.max_stack} bytes")
```

### OTA Update Generation

```python
from pqc_iot_retrofit.ota import OTAUpdateBuilder

# Create OTA update package
ota_builder = OTAUpdateBuilder(
    device_family="smart_meter_v2",
    bootloader="mcuboot",
    signing_key="ota_signing_key.pem"
)

# Add PQC patches
for patch in patches:
    ota_builder.add_patch(
        patch,
        target_address=patch.target_address,
        verification="crc32"
    )

# Add rollback protection
ota_builder.set_rollback_protection(
    min_version="2.3.0",
    anti_rollback_counter=True
)

# Generate differential update
ota_package = ota_builder.build(
    compression="lzma",
    encryption="aes-256-gcm",
    chunk_size=4096  # 4KB chunks for reliable transmission
)

# Save OTA package
ota_package.save("firmware_v2.3_to_v2.4_pqc.bin")
print(f"OTA package size: {ota_package.size / 1024:.1f} KB")
print(f"Estimated transfer time (LoRa): {ota_package.estimate_transfer_time('lora'):.1f} minutes")
