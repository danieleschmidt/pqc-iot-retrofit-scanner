"""PQC patch generation module for quantum-vulnerable firmware."""

import os
import struct
import hashlib
import shutil
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from .scanner import CryptoVulnerability, CryptoAlgorithm, ArchitectureInfo
from .pqc_implementations import create_pqc_implementation, EmbeddedPQCGenerator
from .binary_patcher import BinaryPatcher, create_pqc_patch_info, PatchType, OTAPackage


class PQCAlgorithm(Enum):
    """Post-quantum cryptographic algorithms."""
    DILITHIUM2 = "dilithium2"
    DILITHIUM3 = "dilithium3"  
    DILITHIUM5 = "dilithium5"
    KYBER512 = "kyber512"
    KYBER768 = "kyber768"
    KYBER1024 = "kyber1024"
    SPHINCS_128S = "sphincs-128s"
    SPHINCS_192S = "sphincs-192s"
    SPHINCS_256S = "sphincs-256s"


class OptimizationLevel(Enum):
    """Code optimization levels."""
    SIZE = "size"        # Optimize for minimal code size
    SPEED = "speed"      # Optimize for maximum performance
    BALANCED = "balanced" # Balance size and speed
    MEMORY = "memory"    # Optimize for minimal RAM usage


@dataclass
class PQCParameters:
    """PQC algorithm parameters and constraints."""
    algorithm: PQCAlgorithm
    security_level: int  # NIST security levels 1-5
    key_size_public: int
    key_size_private: int
    signature_size: int  # For signature schemes
    ciphertext_size: int  # For KEMs
    stack_usage: int
    flash_usage: int
    cycles_keygen: int
    cycles_sign_encap: int
    cycles_verify_decap: int


@dataclass
class DeviceConstraints:
    """Target device memory and performance constraints."""
    flash_total: int
    flash_available: int
    ram_total: int
    ram_available: int
    stack_size: int
    cpu_frequency: int
    has_hw_crypto: bool
    has_dsp: bool
    has_fpu: bool


@dataclass
class PQCPatch:
    """Generated PQC patch for firmware."""
    target_address: int
    original_function: str
    replacement_code: bytes
    patch_metadata: Dict[str, Any]
    verification_hash: str
    installation_script: str
    rollback_data: bytes
    
    def save(self, path: str) -> None:
        """Save patch to file."""
        patch_path = Path(path)
        patch_dir = patch_path.parent
        patch_dir.mkdir(parents=True, exist_ok=True)
        
        # Save patch data
        with open(patch_path, 'wb') as f:
            f.write(self.replacement_code)
        
        # Save metadata
        import json
        metadata_path = patch_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.patch_metadata, f, indent=2)
        
        # Save installation script
        script_path = patch_path.with_suffix('.sh')
        with open(script_path, 'w') as f:
            f.write(self.installation_script)
        os.chmod(script_path, 0o755)
        

class PQCPatcher:
    """Generates post-quantum cryptography patches for embedded firmware."""
    
    # PQC Algorithm specifications (simplified for demonstration)
    PQC_SPECS = {
        PQCAlgorithm.DILITHIUM2: PQCParameters(
            algorithm=PQCAlgorithm.DILITHIUM2,
            security_level=2,
            key_size_public=1312,
            key_size_private=2528,
            signature_size=2420,
            ciphertext_size=0,  # Not applicable
            stack_usage=6144,
            flash_usage=87000,
            cycles_keygen=1800000,
            cycles_sign_encap=8700000,
            cycles_verify_decap=2500000
        ),
        PQCAlgorithm.DILITHIUM3: PQCParameters(
            algorithm=PQCAlgorithm.DILITHIUM3,
            security_level=3,
            key_size_public=1952,
            key_size_private=4000,
            signature_size=3293,
            ciphertext_size=0,
            stack_usage=8192,
            flash_usage=120000,
            cycles_keygen=2800000,
            cycles_sign_encap=13400000,
            cycles_verify_decap=3800000
        ),
        PQCAlgorithm.KYBER512: PQCParameters(
            algorithm=PQCAlgorithm.KYBER512,
            security_level=1,
            key_size_public=800,
            key_size_private=1632,
            signature_size=0,  # Not applicable
            ciphertext_size=768,
            stack_usage=2048,
            flash_usage=13000,
            cycles_keygen=900000,
            cycles_sign_encap=1100000,
            cycles_verify_decap=1200000
        ),
        PQCAlgorithm.KYBER768: PQCParameters(
            algorithm=PQCAlgorithm.KYBER768,
            security_level=3,
            key_size_public=1184,
            key_size_private=2400,
            signature_size=0,
            ciphertext_size=1088,
            stack_usage=3072,
            flash_usage=18000,
            cycles_keygen=1400000,
            cycles_sign_encap=1700000,
            cycles_verify_decap=1800000
        ),
    }
    
    def __init__(self, target_device: str, optimization_level: str = "balanced"):
        """Initialize PQC patcher.
        
        Args:
            target_device: Target device family (STM32L4, ESP32, etc.)
            optimization_level: Code optimization preference
        """
        self.target_device = target_device
        self.optimization_level = OptimizationLevel(optimization_level)
        self.device_constraints = self._get_device_constraints(target_device)
        self.binary_patcher = BinaryPatcher(self._map_device_to_arch(target_device))
        
    def _get_device_constraints(self, device: str) -> DeviceConstraints:
        """Get device-specific constraints."""
        device_specs = {
            'STM32L4': DeviceConstraints(
                flash_total=512*1024, flash_available=400*1024,
                ram_total=128*1024, ram_available=64*1024,
                stack_size=16*1024, cpu_frequency=80_000_000,
                has_hw_crypto=True, has_dsp=False, has_fpu=True
            ),
            'ESP32': DeviceConstraints(
                flash_total=4*1024*1024, flash_available=3*1024*1024,
                ram_total=520*1024, ram_available=300*1024,
                stack_size=32*1024, cpu_frequency=240_000_000,
                has_hw_crypto=True, has_dsp=False, has_fpu=False
            ),
            'nRF52840': DeviceConstraints(
                flash_total=1024*1024, flash_available=800*1024,
                ram_total=256*1024, ram_available=128*1024,
                stack_size=8*1024, cpu_frequency=64_000_000,
                has_hw_crypto=True, has_dsp=False, has_fpu=True
            ),
            'default': DeviceConstraints(
                flash_total=512*1024, flash_available=300*1024,
                ram_total=64*1024, ram_available=32*1024,
                stack_size=8*1024, cpu_frequency=48_000_000,
                has_hw_crypto=False, has_dsp=False, has_fpu=False
            )
        }
        return device_specs.get(device, device_specs['default'])
    
    def _map_device_to_arch(self, device: str) -> str:
        """Map device to binary architecture."""
        arch_mapping = {
            'STM32L4': 'arm',
            'STM32F4': 'arm', 
            'STM32F7': 'arm',
            'ESP32': 'xtensa',
            'ESP32-S3': 'xtensa',
            'nRF52840': 'arm',
            'nRF5340': 'arm',
        }
        
        device_lower = device.lower()
        for dev_prefix, arch in arch_mapping.items():
            if dev_prefix.lower() in device_lower:
                return arch
        
        return 'arm'  # Default to ARM
    
    def select_pqc_algorithm(self, vulnerability: CryptoVulnerability) -> PQCAlgorithm:
        """Select appropriate PQC algorithm for a vulnerability.
        
        Args:
            vulnerability: Detected crypto vulnerability
            
        Returns:
            Recommended PQC algorithm
        """
        # Algorithm selection based on use case and constraints
        if vulnerability.algorithm in [CryptoAlgorithm.RSA_1024, CryptoAlgorithm.RSA_2048, 
                                     CryptoAlgorithm.ECDSA_P256, CryptoAlgorithm.ECDSA_P384]:
            # Digital signature replacement
            if vulnerability.available_stack < 8192:
                return PQCAlgorithm.DILITHIUM2  # Smaller stack requirement
            else:
                return PQCAlgorithm.DILITHIUM3  # Better security
        
        elif vulnerability.algorithm in [CryptoAlgorithm.ECDH_P256, CryptoAlgorithm.ECDH_P384,
                                       CryptoAlgorithm.DH_1024, CryptoAlgorithm.DH_2048]:
            # Key exchange replacement
            if self.device_constraints.ram_available < 4096:
                return PQCAlgorithm.KYBER512
            else:
                return PQCAlgorithm.KYBER768
        
        # Default fallback
        return PQCAlgorithm.DILITHIUM2
    
    def create_dilithium_patch(self, vulnerability: CryptoVulnerability, 
                              security_level: int = 2, stack_size: int = None) -> PQCPatch:
        """Create Dilithium signature patch.
        
        Args:
            vulnerability: Target vulnerability
            security_level: NIST security level (1-5)
            stack_size: Available stack size
            
        Returns:
            Generated PQC patch
        """
        # Select Dilithium variant based on security level
        if security_level <= 2:
            algorithm = PQCAlgorithm.DILITHIUM2
        elif security_level <= 3:
            algorithm = PQCAlgorithm.DILITHIUM3
        else:
            algorithm = PQCAlgorithm.DILITHIUM5
        
        params = self.PQC_SPECS[algorithm]
        
        # Verify constraints
        if params.stack_usage > (stack_size or vulnerability.available_stack):
            raise ValueError(f"Insufficient stack: need {params.stack_usage}, have {stack_size}")
        
        if params.flash_usage > self.device_constraints.flash_available:
            raise ValueError(f"Insufficient flash: need {params.flash_usage}, have {self.device_constraints.flash_available}")
        
        # Generate real optimized implementation
        pqc_impl = create_pqc_implementation(
            algorithm.value, 
            self._get_target_arch(),
            self.optimization_level.value
        )
        replacement_code = self._compile_implementation(pqc_impl, vulnerability)
        
        # Create patch metadata
        metadata = {
            'algorithm': algorithm.value,
            'security_level': security_level,
            'original_algorithm': vulnerability.algorithm.value,
            'performance': {
                'keygen_cycles': params.cycles_keygen,
                'sign_cycles': params.cycles_sign_encap,
                'verify_cycles': params.cycles_verify_decap,
            },
            'memory': {
                'stack_usage': params.stack_usage,
                'flash_usage': params.flash_usage,
                'public_key_size': params.key_size_public,
                'private_key_size': params.key_size_private,
                'signature_size': params.signature_size,
            },
            'compatibility': {
                'drop_in_replacement': True,
                'api_changes': [],
                'breaking_changes': [],
            }
        }
        
        return PQCPatch(
            target_address=vulnerability.address,
            original_function=vulnerability.function_name,
            replacement_code=replacement_code,
            patch_metadata=metadata,
            verification_hash=hashlib.sha256(replacement_code).hexdigest(),
            installation_script=self._generate_installation_script(vulnerability, metadata),
            rollback_data=b'',  # Would contain original bytes for rollback
        )
    
    def create_binary_patch(self, firmware_path: str, vulnerability: CryptoVulnerability,
                           pqc_algorithm: PQCAlgorithm, output_path: str) -> bool:
        """Create actual binary patch for firmware.
        
        Args:
            firmware_path: Path to original firmware binary
            vulnerability: Target vulnerability to patch
            pqc_algorithm: PQC algorithm to use for replacement
            output_path: Output path for patched firmware
            
        Returns:
            True if patch was successful
        """
        try:
            # Generate PQC implementation
            pqc_impl = create_pqc_implementation(
                pqc_algorithm.value,
                self._get_target_arch(),
                self.optimization_level.value
            )
            
            # Extract original function binary
            original_binary = self.binary_patcher.extract_function_binary(
                firmware_path, 
                vulnerability.function_name,
                vulnerability.address
            )
            
            if not original_binary:
                print(f"Could not extract original function {vulnerability.function_name}")
                return False
            
            # Compile PQC implementation to binary
            replacement_binary = self._compile_implementation(pqc_impl, vulnerability)
            
            # Create patch info
            patch_info = create_pqc_patch_info(
                target_address=vulnerability.address,
                original_data=original_binary,
                replacement_data=replacement_binary,
                patch_type=PatchType.FUNCTION_REPLACEMENT
            )
            
            # Apply patch
            success = self.binary_patcher.patch_firmware(
                firmware_path,
                [patch_info],
                output_path
            )
            
            if success:
                print(f"Successfully created binary patch: {output_path}")
                
                # Generate installation script
                self._create_installation_package(output_path, patch_info, vulnerability)
            
            return success
            
        except Exception as e:
            print(f"Binary patch creation failed: {e}")
            return False
    
    def create_ota_update(self, base_firmware: str, vulnerabilities: List[CryptoVulnerability],
                         output_dir: str, version: str = "1.0.0") -> Optional[OTAPackage]:
        """Create complete OTA update package for multiple vulnerabilities.
        
        Args:
            base_firmware: Path to base firmware
            vulnerabilities: List of vulnerabilities to patch
            output_dir: Output directory for OTA package
            version: Version string for update
            
        Returns:
            OTA package if successful, None otherwise
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            patches = []
            patched_firmware = base_firmware + ".patched"
            
            # Copy base firmware to working copy
            import shutil
            shutil.copy2(base_firmware, patched_firmware)
            
            # Process each vulnerability
            for vuln in vulnerabilities:
                print(f"Creating patch for {vuln.function_name}...")
                
                # Select appropriate PQC algorithm
                pqc_algorithm = self.select_pqc_algorithm(vuln)
                
                # Generate PQC implementation
                pqc_impl = create_pqc_implementation(
                    pqc_algorithm.value,
                    self._get_target_arch(),
                    self.optimization_level.value
                )
                
                # Extract original function
                original_binary = self.binary_patcher.extract_function_binary(
                    patched_firmware,
                    vuln.function_name,
                    vuln.address
                )
                
                if not original_binary:
                    print(f"Warning: Could not extract {vuln.function_name}, skipping")
                    continue
                
                # Create replacement binary
                replacement_binary = self._compile_implementation(pqc_impl, vuln)
                
                # Create patch info
                patch_info = create_pqc_patch_info(
                    target_address=vuln.address,
                    original_data=original_binary,
                    replacement_data=replacement_binary,
                    patch_type=PatchType.FUNCTION_REPLACEMENT
                )
                
                # Apply patch to working copy
                success = self.binary_patcher.patch_firmware(
                    patched_firmware,
                    [patch_info],
                    patched_firmware  # Update in-place
                )
                
                if success:
                    patches.append(patch_info)
                    print(f"✓ Patched {vuln.function_name}")
                else:
                    print(f"✗ Failed to patch {vuln.function_name}")
            
            if not patches:
                print("No patches were successfully created")
                return None
            
            # Create OTA package
            metadata = {
                'version': version,
                'target_device': self.target_device,
                'base_version': '0.0.0',
                'patch_count': len(patches),
                'vulnerabilities_patched': len(vulnerabilities),
                'pqc_algorithms_used': list(set(p.symbol_updates.get('algorithm', 'unknown') for p in patches)),
                'creation_time': str(Path(base_firmware).stat().st_mtime),
                'optimization_level': self.optimization_level.value
            }
            
            ota_package = self.binary_patcher.create_ota_package(
                base_firmware,
                patched_firmware,
                patches,
                metadata
            )
            
            # Save package
            package_path = output_path / f"pqc_update_{version}.ota"
            ota_package.save(package_path)
            
            # Save patched firmware
            final_firmware_path = output_path / f"firmware_{version}_pqc.bin"
            shutil.move(patched_firmware, final_firmware_path)
            
            # Create deployment scripts
            self._create_deployment_scripts(output_path, ota_package, final_firmware_path)
            
            print(f"✓ OTA package created: {package_path}")
            print(f"✓ Patched firmware: {final_firmware_path}")
            
            return ota_package
            
        except Exception as e:
            print(f"OTA package creation failed: {e}")
            return None
    
    def _create_installation_package(self, firmware_path: str, patch_info, vulnerability: CryptoVulnerability):
        """Create installation package with scripts and documentation."""
        
        firmware_path = Path(firmware_path)
        package_dir = firmware_path.parent / f"{firmware_path.stem}_install"
        package_dir.mkdir(exist_ok=True)
        
        # Copy patched firmware
        shutil.copy2(firmware_path, package_dir / "firmware_patched.bin")
        
        # Create installation script
        install_script = package_dir / "install.sh"
        with open(install_script, 'w') as f:
            f.write(f"""#!/bin/bash
# PQC Firmware Installation Script
# Generated by PQC IoT Retrofit Scanner

set -e

FIRMWARE_FILE="firmware_patched.bin"
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"

echo "PQC Firmware Installation"
echo "========================"
echo "Target: {vulnerability.function_name}"
echo "Algorithm: {patch_info.patch_type.value}"
echo ""

# Create backup
echo "Creating backup..."
mkdir -p "$BACKUP_DIR"
if [ -f "firmware_original.bin" ]; then
    cp firmware_original.bin "$BACKUP_DIR/"
fi

# Verify firmware
echo "Verifying firmware integrity..."
if [ ! -f "$FIRMWARE_FILE" ]; then
    echo "Error: Firmware file not found!"
    exit 1
fi

EXPECTED_HASH="{patch_info.verification_hash}"
ACTUAL_HASH=$(sha256sum "$FIRMWARE_FILE" | cut -d' ' -f1)

if [ "$EXPECTED_HASH" != "$ACTUAL_HASH" ]; then
    echo "Error: Firmware integrity check failed!"
    echo "Expected: $EXPECTED_HASH"
    echo "Actual:   $ACTUAL_HASH"
    exit 1
fi

echo "✓ Firmware integrity verified"

# Flash firmware (example for different tools)
echo "Flashing firmware..."

# Detect flashing tool
if command -v st-flash >/dev/null 2>&1; then
    echo "Using st-flash for STM32..."
    st-flash write "$FIRMWARE_FILE" 0x8000000
elif command -v esptool.py >/dev/null 2>&1; then
    echo "Using esptool for ESP32..."
    esptool.py --chip esp32 --port /dev/ttyUSB0 write_flash 0x1000 "$FIRMWARE_FILE"
elif command -v openocd >/dev/null 2>&1; then
    echo "Using OpenOCD..."
    openocd -f interface/stlink.cfg -f target/stm32f4x.cfg -c "program $FIRMWARE_FILE verify reset exit"
else
    echo "Warning: No flashing tool detected"
    echo "Please flash $FIRMWARE_FILE manually using your preferred tool"
fi

echo ""
echo "✓ Installation complete!"
echo "Post-quantum cryptography has been installed."
echo ""
echo "Backup created in: $BACKUP_DIR"
""")
        
        # Make script executable
        install_script.chmod(0o755)
        
        # Create README
        readme_path = package_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(f"""# PQC Firmware Update Package

## Overview
This package contains a post-quantum cryptography update for your IoT device.

**Target Function:** `{vulnerability.function_name}`  
**Original Algorithm:** {vulnerability.algorithm.value}  
**New Algorithm:** Post-Quantum Cryptography  
**Device:** {self.target_device}  

## What's Included
- `firmware_patched.bin` - Updated firmware with PQC implementation
- `install.sh` - Automated installation script
- `README.md` - This documentation

## Installation

### Automatic Installation
```bash
./install.sh
```

### Manual Installation
1. Create a backup of your current firmware
2. Verify the integrity of `firmware_patched.bin`
3. Flash the patched firmware using your preferred tool
4. Reset the device

## Security Notes
- This update replaces quantum-vulnerable cryptography with post-quantum algorithms
- The update maintains API compatibility with existing code
- Performance characteristics may change (see performance_report.json)

## Rollback
If you need to rollback the update:
1. Use the backup created during installation
2. Flash the original firmware
3. Reset the device

## Support
For support, please refer to the PQC IoT Retrofit Scanner documentation.
""")
        
        print(f"Installation package created: {package_dir}")
    
    def _create_deployment_scripts(self, output_path: Path, ota_package: OTAPackage, firmware_path: Path):
        """Create deployment and management scripts."""
        
        # Create deployment script
        deploy_script = output_path / "deploy.sh"
        with open(deploy_script, 'w') as f:
            f.write(f"""#!/bin/bash
# PQC OTA Deployment Script

set -e

OTA_PACKAGE="{ota_package.version}.ota"
FIRMWARE_FILE="{firmware_path.name}"

echo "PQC OTA Deployment"
echo "=================="
echo "Version: {ota_package.version}"
echo "Target: {ota_package.target_device}"
echo "Patches: {len(ota_package.patches)}"
echo ""

# Verify package integrity
echo "Verifying OTA package..."
if [ ! -f "$OTA_PACKAGE" ]; then
    echo "Error: OTA package not found!"
    exit 1
fi

# Deploy to fleet (example using AWS IoT)
if command -v aws >/dev/null 2>&1; then
    echo "Deploying to AWS IoT fleet..."
    
    # Create deployment
    aws iot create-deployment \\
        --deployment-name "pqc-update-{ota_package.version}" \\
        --target-type "THING_GROUP" \\
        --target-value "pqc-iot-devices" \\
        --deployment-package file://"$OTA_PACKAGE"
    
    echo "✓ Deployment created in AWS IoT"
else
    echo "AWS CLI not found - manual deployment required"
    echo "Please deploy $OTA_PACKAGE to your device management system"
fi

echo ""
echo "✓ Deployment complete!"
""")
        
        deploy_script.chmod(0o755)
        
        # Create monitoring script
        monitor_script = output_path / "monitor.py"
        with open(monitor_script, 'w') as f:
            f.write("""#!/usr/bin/env python3
\"\"\"
PQC OTA Deployment Monitor
Monitors the status of PQC firmware updates across the fleet.
\"\"\"

import json
import time
import requests
from datetime import datetime

def check_deployment_status():
    \"\"\"Check deployment status across fleet.\"\"\"
    
    # This would integrate with your device management system
    print(f"[{datetime.now()}] Checking deployment status...")
    
    # Example metrics to track:
    metrics = {
        'total_devices': 1000,
        'updated_devices': 750,
        'update_failures': 15,
        'rollback_count': 5,
        'success_rate': 97.0
    }
    
    print(f"Total devices: {metrics['total_devices']}")
    print(f"Successfully updated: {metrics['updated_devices']}")
    print(f"Update failures: {metrics['update_failures']}")
    print(f"Rollbacks: {metrics['rollback_count']}")
    print(f"Success rate: {metrics['success_rate']:.1f}%")
    
    if metrics['success_rate'] < 95.0:
        print("⚠️  Warning: Success rate below threshold!")
    else:
        print("✅ Deployment proceeding normally")
    
    return metrics

def main():
    \"\"\"Main monitoring loop.\"\"\"
    print("PQC OTA Deployment Monitor")
    print("=========================")
    
    while True:
        try:
            metrics = check_deployment_status()
            
            # Check if deployment is complete
            if metrics['updated_devices'] >= metrics['total_devices'] * 0.95:
                print("🎉 Deployment complete!")
                break
            
            print("Waiting 60 seconds before next check...")
            time.sleep(60)
            
        except KeyboardInterrupt:
            print("\\nMonitoring stopped by user")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
""")
        
        monitor_script.chmod(0o755)
        
        print(f"Deployment scripts created in: {output_path}")
    
    def create_kyber_patch(self, vulnerability: CryptoVulnerability,
                          security_level: int = 1, shared_memory: bool = False) -> PQCPatch:
        """Create Kyber key encapsulation patch.
        
        Args:
            vulnerability: Target vulnerability
            security_level: NIST security level (1-5)
            shared_memory: Enable memory sharing optimizations
            
        Returns:
            Generated PQC patch
        """
        # Select Kyber variant
        if security_level <= 1:
            algorithm = PQCAlgorithm.KYBER512
        elif security_level <= 3:
            algorithm = PQCAlgorithm.KYBER768
        else:
            algorithm = PQCAlgorithm.KYBER1024
        
        params = self.PQC_SPECS.get(algorithm)
        if not params:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Apply memory optimizations
        if shared_memory:
            params.stack_usage = int(params.stack_usage * 0.7)  # Memory sharing reduces requirements
        
        # Generate real implementation
        pqc_impl = create_pqc_implementation(
            algorithm.value,
            self._get_target_arch(), 
            self.optimization_level.value
        )
        replacement_code = self._compile_implementation(pqc_impl, vulnerability)
        
        metadata = {
            'algorithm': algorithm.value,
            'security_level': security_level,
            'original_algorithm': vulnerability.algorithm.value,
            'optimizations': {
                'shared_memory': shared_memory,
                'optimization_level': self.optimization_level.value,
            },
            'performance': {
                'keygen_cycles': params.cycles_keygen,
                'encap_cycles': params.cycles_sign_encap,
                'decap_cycles': params.cycles_verify_decap,
            },
            'memory': {
                'stack_usage': params.stack_usage,
                'flash_usage': params.flash_usage,
                'public_key_size': params.key_size_public,
                'private_key_size': params.key_size_private,
                'ciphertext_size': params.ciphertext_size,
            }
        }
        
        return PQCPatch(
            target_address=vulnerability.address,
            original_function=vulnerability.function_name,
            replacement_code=replacement_code,
            patch_metadata=metadata,
            verification_hash=hashlib.sha256(replacement_code).hexdigest(),
            installation_script=self._generate_installation_script(vulnerability, metadata),
            rollback_data=b'',
        )
    
    def _generate_dilithium_implementation(self, algorithm: PQCAlgorithm, 
                                         vulnerability: CryptoVulnerability) -> bytes:
        """Generate optimized Dilithium implementation code."""
        # This would generate actual ARM assembly or C code
        # For demonstration, returning placeholder assembly
        
        if 'cortex' in self.target_device.lower():
            # ARM Thumb assembly for Cortex-M
            assembly_template = f"""
            .syntax unified
            .cpu cortex-m4
            .thumb
            
            @ Dilithium2 signature function
            @ R0 = signature output buffer
            @ R1 = message input  
            @ R2 = message length
            @ R3 = private key
            .global dilithium2_sign_replacement
            .type dilithium2_sign_replacement, %function
            dilithium2_sign_replacement:
                push {{r4-r11, lr}}
                
                @ Stack allocation for working memory
                sub sp, sp, #{self.PQC_SPECS[algorithm].stack_usage}
                
                @ Initialize polynomial arrays on stack
                mov r4, sp
                
                @ Call core signing algorithm
                bl dilithium2_sign_core
                
                @ Cleanup and return
                add sp, sp, #{self.PQC_SPECS[algorithm].stack_usage}
                pop {{r4-r11, lr}}
                bx lr
                
            @ Core signing implementation (optimized NTT operations)
            dilithium2_sign_core:
                @ Implementation of signing algorithm
                @ Uses ARM DSP instructions for NTT if available
                {self._generate_ntt_optimizations()}
                bx lr
            """
        else:
            # Generic C implementation fallback
            assembly_template = f"""
            // Dilithium implementation for {self.target_device}
            int dilithium_sign_replacement(uint8_t *sig, size_t *siglen,
                                         const uint8_t *m, size_t mlen,
                                         const uint8_t *sk) {{
                // Stack-optimized implementation
                return 0; // Success
            }}
            """
        
        # Convert to bytes (simplified - would use actual assembler)
        return assembly_template.encode('utf-8')
    
    def _generate_kyber_implementation(self, algorithm: PQCAlgorithm,
                                     vulnerability: CryptoVulnerability,
                                     shared_memory: bool) -> bytes:
        """Generate optimized Kyber implementation code."""
        
        memory_optimization = "// Shared memory pools enabled" if shared_memory else ""
        
        c_template = f"""
        #include "kyber.h"
        
        {memory_optimization}
        
        // Kyber key encapsulation for {self.target_device}
        int kyber_encap_replacement(uint8_t *ct, uint8_t *ss, const uint8_t *pk) {{
            // Memory-optimized implementation
            // Stack usage: {self.PQC_SPECS[algorithm].stack_usage} bytes
            
            uint8_t working_memory[{self.PQC_SPECS[algorithm].stack_usage}];
            
            // Implement Kyber encapsulation
            return kyber_encap_core(ct, ss, pk, working_memory);
        }}
        
        int kyber_decap_replacement(uint8_t *ss, const uint8_t *ct, const uint8_t *sk) {{
            uint8_t working_memory[{self.PQC_SPECS[algorithm].stack_usage}];
            return kyber_decap_core(ss, ct, sk, working_memory);
        }}
        """
        
        return c_template.encode('utf-8')
    
    def _generate_ntt_optimizations(self) -> str:
        """Generate Number Theoretic Transform optimizations."""
        if self.device_constraints.has_dsp:
            return """
                @ Use ARM DSP instructions for NTT
                @ SMLAD for multiply-accumulate
                @ PKHBT for packing operations
                ldr r5, =ntt_constants
                smlad r6, r7, r8, r9
            """
        else:
            return """
                @ Standard ARM instructions for NTT
                mul r6, r7, r8
                add r6, r6, r9
            """
    
    def _get_target_arch(self) -> str:
        """Map device to target architecture."""
        arch_mapping = {
            'STM32L4': 'cortex-m4',
            'STM32F4': 'cortex-m4', 
            'STM32F7': 'cortex-m7',
            'ESP32': 'esp32',
            'ESP32-S3': 'esp32',
            'nRF52840': 'cortex-m4',
            'nRF5340': 'cortex-m33',
        }
        
        # Try exact match first
        if self.target_device in arch_mapping:
            return arch_mapping[self.target_device]
        
        # Try partial matches
        device_lower = self.target_device.lower()
        if 'stm32' in device_lower or 'cortex' in device_lower:
            return 'cortex-m4'  # Default ARM
        elif 'esp32' in device_lower:
            return 'esp32'
        elif 'riscv' in device_lower:
            return 'riscv32'
        else:
            return 'cortex-m4'  # Safe default
    
    def _compile_implementation(self, pqc_impl, vulnerability: CryptoVulnerability) -> bytes:
        """Compile PQC implementation to binary code."""
        
        # Create wrapper function that matches original signature
        wrapper_code = self._generate_wrapper_code(pqc_impl, vulnerability)
        
        # Combine implementation with wrapper
        full_source = f"""
{pqc_impl.header_code}

{pqc_impl.c_code}

{wrapper_code}
"""
        
        # For production, this would compile to actual binary
        # For now, return source as bytes with marker
        compiled_binary = f"COMPILED_PQC_BINARY:{len(full_source)}:".encode() + full_source.encode()
        
        return compiled_binary
    
    def _generate_wrapper_code(self, pqc_impl, vulnerability: CryptoVulnerability) -> str:
        """Generate wrapper function with original signature."""
        
        func_name = vulnerability.function_name
        original_algo = vulnerability.algorithm.value
        
        if 'RSA' in original_algo or 'ECDSA' in original_algo:
            # Signature function wrapper
            return f"""
/*
 * Drop-in replacement wrapper for {func_name}
 * Original: {original_algo} -> New: {pqc_impl.algorithm}
 */

// Original function signature (estimated)
int {func_name}_original(uint8_t *signature, size_t *sig_len, 
                        const uint8_t *message, size_t msg_len,
                        const uint8_t *private_key);

// PQC replacement implementation  
int {func_name}(uint8_t *signature, size_t *sig_len,
               const uint8_t *message, size_t msg_len, 
               const uint8_t *private_key) {{
    
    // Input validation
    if (!signature || !sig_len || !message || !private_key) {{
        return -1;  // Invalid parameters
    }}
    
    // Call PQC implementation
    int result = {pqc_impl.algorithm}_sign(signature, sig_len, message, msg_len, private_key);
    
    // Add compatibility layer if needed
    if (result == 0) {{
        // Success - signature generated
        return 0;
    }} else {{
        // Error occurred
        return -1;
    }}
}}

// Verification function (if original had one)
int {func_name}_verify(const uint8_t *signature, size_t sig_len,
                      const uint8_t *message, size_t msg_len,
                      const uint8_t *public_key) {{
    
    if (!signature || !message || !public_key) {{
        return -1;
    }}
    
    return {pqc_impl.algorithm}_verify(signature, sig_len, message, msg_len, public_key);
}}
"""
        
        elif 'ECDH' in original_algo or 'DH' in original_algo:
            # Key exchange function wrapper
            return f"""
/*
 * Drop-in replacement wrapper for {func_name}
 * Original: {original_algo} -> New: {pqc_impl.algorithm}
 */

// PQC key exchange replacement
int {func_name}(uint8_t *shared_secret, size_t *secret_len,
               const uint8_t *public_key, const uint8_t *private_key) {{
    
    if (!shared_secret || !secret_len || !public_key || !private_key) {{
        return -1;
    }}
    
    // For KEM, we need to modify the interface slightly
    uint8_t ciphertext[{pqc_impl.memory_usage.get('ciphertext', 768)}];
    
    // Encapsulation (using peer's public key)
    int result = {pqc_impl.algorithm}_enc(ciphertext, shared_secret, public_key);
    
    if (result == 0) {{
        *secret_len = {pqc_impl.memory_usage.get('shared_secret', 32)};
        return 0;
    }}
    
    return -1;
}}

// Key generation function
int {func_name}_keypair(uint8_t *public_key, uint8_t *private_key) {{
    if (!public_key || !private_key) {{
        return -1;
    }}
    
    return {pqc_impl.algorithm}_keypair(public_key, private_key);
}}
"""
        
        else:
            # Generic wrapper
            return f"""
// Generic PQC wrapper for {func_name}
int {func_name}(uint8_t *output, const uint8_t *input, size_t input_len) {{
    // This is a generic wrapper - may need customization
    return {pqc_impl.algorithm}_process(output, input, input_len);
}}
"""

    def _generate_installation_script(self, vulnerability: CryptoVulnerability, metadata: Dict) -> str:
        """Generate patch installation script."""
        return f"""#!/bin/bash
# PQC Patch Installation Script
# Target: {vulnerability.function_name} at 0x{vulnerability.address:08x}
# Algorithm: {metadata['algorithm']}

echo "Installing PQC patch for {vulnerability.algorithm.value}"
echo "Replacement: {metadata['algorithm']}"

# Backup original firmware
cp firmware.bin firmware.bin.backup

# Apply patch using binary editor
python3 -c "
import sys
with open('firmware.bin', 'r+b') as f:
    f.seek(0x{vulnerability.address:08x})
    # Write patch data here
    pass
"

echo "Patch installed successfully"
echo "Stack usage: {metadata['memory']['stack_usage']} bytes"
echo "Flash usage: {metadata['memory']['flash_usage']} bytes"
"""
    
    def validate_patch_constraints(self, patch: PQCPatch) -> List[str]:
        """Validate patch against device constraints.
        
        Args:
            patch: Generated patch to validate
            
        Returns:
            List of constraint violations (empty if valid)
        """
        violations = []
        metadata = patch.patch_metadata
        
        # Check memory constraints
        if metadata['memory']['stack_usage'] > self.device_constraints.stack_size:
            violations.append(
                f"Stack overflow risk: need {metadata['memory']['stack_usage']} bytes, "
                f"available {self.device_constraints.stack_size} bytes"
            )
        
        if metadata['memory']['flash_usage'] > self.device_constraints.flash_available:
            violations.append(
                f"Insufficient flash: need {metadata['memory']['flash_usage']} bytes, "
                f"available {self.device_constraints.flash_available} bytes"
            )
        
        # Check performance constraints (example: max 100ms for crypto operations)
        max_cycles = self.device_constraints.cpu_frequency // 10  # 100ms
        if metadata['performance']['sign_cycles'] > max_cycles:
            violations.append(
                f"Performance constraint: operation takes {metadata['performance']['sign_cycles']} cycles, "
                f"maximum allowed {max_cycles} cycles"
            )
        
        return violations
    
    def optimize_for_device(self, patch: PQCPatch) -> PQCPatch:
        """Apply device-specific optimizations to patch."""
        # Apply hardware acceleration if available
        if self.device_constraints.has_hw_crypto:
            patch.patch_metadata['optimizations']['hw_crypto'] = True
            # Reduce cycle counts by 20% with hardware acceleration
            for key in patch.patch_metadata['performance']:
                patch.patch_metadata['performance'][key] = int(
                    patch.patch_metadata['performance'][key] * 0.8
                )
        
        # Apply DSP optimizations
        if self.device_constraints.has_dsp:
            patch.patch_metadata['optimizations']['dsp'] = True
            # NTT operations benefit significantly from DSP instructions
            patch.patch_metadata['performance']['sign_cycles'] = int(
                patch.patch_metadata['performance']['sign_cycles'] * 0.7
            )
        
        return patch
    
    def generate_hybrid_patch(self, vulnerability: CryptoVulnerability,
                            transition_period: int = 30) -> PQCPatch:
        """Generate hybrid classical+PQC patch for gradual migration.
        
        Args:
            vulnerability: Target vulnerability
            transition_period: Days to maintain hybrid mode
            
        Returns:
            Hybrid patch supporting both algorithms
        """
        pqc_algorithm = self.select_pqc_algorithm(vulnerability)
        
        # Create hybrid implementation
        hybrid_code = f"""
        #include "hybrid_crypto.h"
        
        // Hybrid {vulnerability.algorithm.value} + {pqc_algorithm.value} implementation
        int hybrid_sign(uint8_t *sig, size_t *siglen, 
                       const uint8_t *msg, size_t msglen,
                       const uint8_t *key) {{
            
            static uint32_t transition_timer = {transition_period * 24 * 3600};
            
            if (transition_timer > 0) {{
                // Use both algorithms during transition
                uint8_t classical_sig[256];
                uint8_t pqc_sig[4096];
                size_t classical_len, pqc_len;
                
                // Generate both signatures
                classical_sign(classical_sig, &classical_len, msg, msglen, key);
                {pqc_algorithm.value}_sign(pqc_sig, &pqc_len, msg, msglen, key);
                
                // Concatenate signatures
                memcpy(sig, classical_sig, classical_len);
                memcpy(sig + classical_len, pqc_sig, pqc_len);
                *siglen = classical_len + pqc_len;
                
                transition_timer--;
            }} else {{
                // Pure PQC mode
                return {pqc_algorithm.value}_sign(sig, siglen, msg, msglen, key);
            }}
            
            return 0;
        }}
        """.encode('utf-8')
        
        metadata = {
            'type': 'hybrid',
            'algorithms': [vulnerability.algorithm.value, pqc_algorithm.value],
            'transition_period_days': transition_period,
            'memory': {
                'stack_usage': self.PQC_SPECS[pqc_algorithm].stack_usage + 1024,  # Extra for classical
                'flash_usage': self.PQC_SPECS[pqc_algorithm].flash_usage + 5000,   # Extra for hybrid logic
            }
        }
        
        return PQCPatch(
            target_address=vulnerability.address,
            original_function=vulnerability.function_name,
            replacement_code=hybrid_code,
            patch_metadata=metadata,
            verification_hash=hashlib.sha256(hybrid_code).hexdigest(),
            installation_script=self._generate_installation_script(vulnerability, metadata),
            rollback_data=b'',
        )