"""PQC patch generation module for quantum-vulnerable firmware."""

import os
import struct
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from .scanner import CryptoVulnerability, CryptoAlgorithm, ArchitectureInfo


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
        
        # Generate optimized implementation
        replacement_code = self._generate_dilithium_implementation(algorithm, vulnerability)
        
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
        
        # Generate implementation
        replacement_code = self._generate_kyber_implementation(algorithm, vulnerability, shared_memory)
        
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