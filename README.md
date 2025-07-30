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
