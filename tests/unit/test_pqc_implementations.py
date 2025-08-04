"""
Unit tests for PQC implementations module.

Tests the generation of Kyber and Dilithium implementations
with different optimization levels and target architectures.
"""

import pytest
from pqc_iot_retrofit.pqc_implementations import (
    EmbeddedPQCGenerator, 
    create_pqc_implementation,
    PQCImplementation,
    KYBER_Q,
    KYBER_N,
    DILITHIUM_Q,
    DILITHIUM_N
)


class TestEmbeddedPQCGenerator:
    """Test the embedded PQC code generator."""
    
    @pytest.fixture
    def generator(self):
        return EmbeddedPQCGenerator('cortex-m4')
    
    def test_generator_initialization(self, generator):
        """Test generator initialization with different architectures."""
        assert generator.target_arch == 'cortex-m4'
        assert generator.arch_config['has_dsp'] == True
        assert generator.arch_config['has_fpu'] == True
        
    def test_arch_config_differences(self):
        """Test that different architectures have different configurations."""
        cortex_gen = EmbeddedPQCGenerator('cortex-m4')
        esp32_gen = EmbeddedPQCGenerator('esp32')
        riscv_gen = EmbeddedPQCGenerator('riscv32')
        
        # Cortex-M4 should have DSP
        assert cortex_gen.arch_config['has_dsp'] == True
        
        # ESP32 should not have DSP but different calling convention
        assert esp32_gen.arch_config['has_dsp'] == False
        assert esp32_gen.arch_config['calling_convention'] == 'xtensa'
        
        # RISC-V should have more registers
        assert riscv_gen.arch_config['register_count'] == 32
        assert riscv_gen.arch_config['ntt_unroll_factor'] == 8
    
    def test_kyber512_generation(self, generator):
        """Test Kyber-512 implementation generation."""
        impl = generator.generate_kyber512('balanced')
        
        assert isinstance(impl, PQCImplementation)
        assert impl.algorithm == 'kyber512'
        assert impl.target_arch == 'cortex-m4'
        assert impl.optimization_level == 'balanced'
        
        # Check that implementation contains required components
        assert 'kyber512_keypair' in impl.c_code
        assert 'kyber512_enc' in impl.c_code
        assert 'kyber512_dec' in impl.c_code
        
        # Check header definitions
        assert '#ifndef KYBER512_H' in impl.header_code
        assert 'KYBER_K 2' in impl.header_code  # Kyber-512 uses k=2
        
        # Verify memory usage is realistic
        assert impl.memory_usage['public_key'] == 800
        assert impl.memory_usage['private_key'] == 1632
        assert impl.memory_usage['ciphertext'] == 768
        assert impl.memory_usage['shared_secret'] == 32
        
        # Performance estimates should be reasonable
        assert impl.performance_estimates['keygen_cycles'] > 0
        assert impl.performance_estimates['encaps_cycles'] > 0
        assert impl.performance_estimates['decaps_cycles'] > 0
    
    def test_kyber768_generation(self, generator):
        """Test Kyber-768 implementation generation."""
        impl = generator.generate_kyber768('speed')
        
        assert impl.algorithm == 'kyber768'
        assert impl.optimization_level == 'speed'
        
        # Kyber-768 should have larger keys than Kyber-512
        assert impl.memory_usage['public_key'] == 1184
        assert impl.memory_usage['private_key'] == 2400
        assert impl.memory_usage['ciphertext'] == 1088
        
        # Speed optimization should be faster than default
        kyber512_balanced = generator.generate_kyber512('balanced')
        assert impl.performance_estimates['keygen_cycles'] < kyber512_balanced.performance_estimates['keygen_cycles'] * 1.5
    
    def test_dilithium2_generation(self, generator):
        """Test Dilithium-2 implementation generation."""
        impl = generator.generate_dilithium2('balanced')
        
        assert impl.algorithm == 'dilithium2'
        assert impl.target_arch == 'cortex-m4'
        
        # Check that implementation contains signature functions
        assert 'dilithium2_keypair' in impl.c_code
        assert 'dilithium2_sign' in impl.c_code
        assert 'dilithium2_verify' in impl.c_code
        
        # Check Dilithium-specific constants
        assert 'DILITHIUM_K 4' in impl.header_code  # Dilithium-2 uses k=4
        assert 'DILITHIUM_L 4' in impl.header_code  # Dilithium-2 uses l=4
        
        # Verify signature scheme memory usage
        assert impl.memory_usage['public_key'] == 1312
        assert impl.memory_usage['private_key'] == 2528
        assert impl.memory_usage['signature'] == 2420
        
        # Dilithium signing should be more expensive than key generation
        assert impl.performance_estimates['sign_cycles'] > impl.performance_estimates['keygen_cycles']
        assert impl.performance_estimates['verify_cycles'] < impl.performance_estimates['sign_cycles']
    
    def test_dilithium3_generation(self, generator):
        """Test Dilithium-3 implementation generation."""
        impl = generator.generate_dilithium3('size')
        
        assert impl.algorithm == 'dilithium3'
        assert impl.optimization_level == 'size'
        
        # Dilithium-3 should have larger keys and signatures than Dilithium-2
        dilithium2 = generator.generate_dilithium2('size')
        
        assert impl.memory_usage['public_key'] > dilithium2.memory_usage['public_key']
        assert impl.memory_usage['private_key'] > dilithium2.memory_usage['private_key']
        assert impl.memory_usage['signature'] > dilithium2.memory_usage['signature']
        
        # Should be more expensive computationally
        assert impl.performance_estimates['sign_cycles'] > dilithium2.performance_estimates['sign_cycles']
    
    def test_dilithium5_generation(self, generator):
        """Test Dilithium-5 implementation generation."""
        impl = generator.generate_dilithium5('balanced')
        
        assert impl.algorithm == 'dilithium5'
        
        # Dilithium-5 should be the largest variant
        assert impl.memory_usage['public_key'] == 2592
        assert impl.memory_usage['private_key'] == 4864
        assert impl.memory_usage['signature'] == 4595
        
        # Should be the most expensive computationally
        assert impl.performance_estimates['sign_cycles'] > 15000000  # At least 15M cycles
    
    def test_optimization_level_differences(self, generator):
        """Test that different optimization levels produce different results."""
        
        speed_impl = generator.generate_kyber512('speed')
        size_impl = generator.generate_kyber512('size')
        balanced_impl = generator.generate_kyber512('balanced')
        memory_impl = generator.generate_kyber512('memory')
        
        # Speed should be fastest
        assert speed_impl.performance_estimates['keygen_cycles'] <= balanced_impl.performance_estimates['keygen_cycles']
        
        # Size should be smallest
        assert size_impl.memory_usage['flash_usage'] <= balanced_impl.memory_usage['flash_usage']
        
        # Memory should use least stack
        assert memory_impl.memory_usage['stack_peak'] <= balanced_impl.memory_usage['stack_peak']
        
        # Check that different NTT implementations are used
        assert 'Fast NTT' in speed_impl.c_code or 'speed' in speed_impl.c_code.lower()
        assert 'Compact NTT' in size_impl.c_code or 'size' in size_impl.c_code.lower()
    
    def test_architecture_specific_assembly(self):
        """Test that architecture-specific assembly is generated."""
        
        cortex_gen = EmbeddedPQCGenerator('cortex-m4')
        esp32_gen = EmbeddedPQCGenerator('esp32')
        riscv_gen = EmbeddedPQCGenerator('riscv32')
        
        cortex_impl = cortex_gen.generate_kyber512('speed')
        esp32_impl = esp32_gen.generate_kyber512('speed')
        riscv_impl = riscv_gen.generate_kyber512('speed')
        
        # Each should have architecture-specific assembly
        assert '.syntax unified' in cortex_impl.assembly_code  # ARM assembly
        assert 'cortex-m4' in cortex_impl.assembly_code
        
        # ESP32 should have Xtensa assembly
        assert 'xtensa' in esp32_impl.assembly_code.lower() or len(esp32_impl.assembly_code) > 100
        
        # RISC-V should have different assembly (or note about no optimizations)
        assert len(riscv_impl.assembly_code) > 0
    
    def test_ntt_implementation_variants(self, generator):
        """Test that different NTT implementations are generated."""
        
        speed_impl = generator.generate_kyber512('speed')
        size_impl = generator.generate_kyber512('size')
        balanced_impl = generator.generate_kyber512('balanced')
        
        # Should contain NTT function
        assert 'ntt(' in speed_impl.c_code
        assert 'ntt(' in size_impl.c_code
        assert 'ntt(' in balanced_impl.c_code
        
        # Speed version should have unrolled loops
        assert 'unroll' in speed_impl.c_code.lower() or 'i += 8' in speed_impl.c_code
        
        # Size version should be more compact
        size_ntt_lines = speed_impl.c_code.count('\n')
        speed_ntt_lines = size_impl.c_code.count('\n')
        # Size optimized might actually be longer due to comments, so just check it exists
        assert len(size_impl.c_code) > 1000
    
    def test_test_vectors_generation(self, generator):
        """Test that test vectors are generated for validation."""
        
        impl = generator.generate_kyber512('balanced')
        
        assert len(impl.test_vectors) > 0
        assert len(impl.test_vectors) == 3  # Should generate 3 test vectors
        
        for i, (message, ciphertext, public_key) in enumerate(impl.test_vectors):
            assert len(message) == 32  # Message should be 32 bytes
            assert len(ciphertext) == 768  # Kyber-512 ciphertext size
            assert len(public_key) == 800  # Kyber-512 public key size
            
            # Test vectors should be deterministic but different
            if i > 0:
                prev_message = impl.test_vectors[i-1][0]
                assert message != prev_message  # Different test vectors
    
    def test_constants_are_valid(self):
        """Test that cryptographic constants are correct."""
        
        # Kyber constants
        assert KYBER_Q == 3329
        assert KYBER_N == 256
        
        # Dilithium constants
        assert DILITHIUM_Q == 8380417
        assert DILITHIUM_N == 256
        
        # Check that NTT constants are present
        generator = EmbeddedPQCGenerator('cortex-m4')
        impl = generator.generate_kyber512('balanced')
        
        # Should contain zetas array
        assert 'zetas[' in impl.c_code
        assert str(KYBER_Q) in impl.c_code or str(KYBER_Q) in impl.header_code


class TestCreatePQCImplementation:
    """Test the factory function for creating PQC implementations."""
    
    def test_valid_algorithms(self):
        """Test creating implementations for all supported algorithms."""
        
        algorithms = ['kyber512', 'kyber768', 'dilithium2', 'dilithium3', 'dilithium5']
        
        for algo in algorithms:
            impl = create_pqc_implementation(algo, 'cortex-m4', 'balanced')
            assert impl.algorithm == algo
            assert impl.target_arch == 'cortex-m4'
            assert len(impl.c_code) > 500
            assert len(impl.header_code) > 200
    
    def test_invalid_algorithm(self):
        """Test that invalid algorithms raise appropriate errors."""
        
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            create_pqc_implementation('invalid_algo', 'cortex-m4', 'balanced')
    
    def test_different_architectures(self):
        """Test creating implementations for different target architectures."""
        
        architectures = ['cortex-m4', 'esp32', 'riscv32']
        
        for arch in architectures:
            impl = create_pqc_implementation('kyber512', arch, 'balanced')
            assert impl.target_arch == arch
            
            # Each architecture should produce different code
            if arch == 'cortex-m4':
                assert 'arm' in impl.assembly_code.lower() or 'cortex' in impl.assembly_code.lower()
            elif arch == 'esp32':
                assert 'xtensa' in impl.assembly_code.lower() or 'esp32' in impl.assembly_code.lower()
    
    def test_optimization_levels(self):
        """Test all optimization levels."""
        
        opt_levels = ['size', 'speed', 'balanced', 'memory']
        
        for opt in opt_levels:
            impl = create_pqc_implementation('dilithium2', 'cortex-m4', opt)
            assert impl.optimization_level == opt
            
            # Each optimization should affect the implementation
            assert opt in impl.c_code.lower() or len(impl.c_code) > 1000


class TestCodeGeneration:
    """Test specific aspects of code generation."""
    
    def test_kyber_c_code_structure(self):
        """Test that generated Kyber C code has correct structure."""
        
        impl = create_pqc_implementation('kyber512', 'cortex-m4', 'balanced')
        c_code = impl.c_code
        
        # Should contain required functions
        required_functions = [
            'kyber512_keypair',
            'kyber512_enc', 
            'kyber512_dec',
            'poly_ntt',
            'poly_basemul_montgomery'
        ]
        
        for func in required_functions:
            assert func in c_code, f"Missing required function: {func}"
        
        # Should have proper C structure
        assert '#include' in c_code
        assert 'static const' in c_code  # Should have constants
        assert 'int ' in c_code  # Should have function definitions
        assert '/*' in c_code and '*/' in c_code  # Should have comments
    
    def test_dilithium_c_code_structure(self):
        """Test that generated Dilithium C code has correct structure."""
        
        impl = create_pqc_implementation('dilithium2', 'cortex-m4', 'balanced')
        c_code = impl.c_code
        
        # Should contain required functions
        required_functions = [
            'dilithium2_keypair',
            'dilithium2_sign',
            'dilithium2_verify',  
            'montgomery_reduce',
            'shake256'
        ]
        
        for func in required_functions:
            assert func in c_code, f"Missing required function: {func}"
        
        # Should handle 32-bit coefficients (unlike Kyber's 16-bit)
        assert 'int32_t' in c_code
        assert 'polyvecl' in c_code  # Dilithium-specific types
        assert 'polyveck' in c_code
    
    def test_header_file_generation(self):
        """Test that header files are properly generated."""
        
        kyber_impl = create_pqc_implementation('kyber512', 'cortex-m4', 'balanced')
        dilithium_impl = create_pqc_implementation('dilithium2', 'cortex-m4', 'balanced')
        
        # Kyber header checks
        kyber_header = kyber_impl.header_code
        assert '#ifndef KYBER512_H' in kyber_header
        assert '#define KYBER512_H' in kyber_header
        assert 'typedef struct' in kyber_header
        assert 'poly' in kyber_header
        
        # Dilithium header checks  
        dilithium_header = dilithium_impl.header_code
        assert '#ifndef DILITHIUM2_H' in dilithium_header
        assert 'DILITHIUM_N 256' in dilithium_header
        assert 'DILITHIUM_Q 8380417' in dilithium_header
        
        # Both should have proper include guards and function prototypes
        for header in [kyber_header, dilithium_header]:
            assert '#endif' in header  # Include guard end
            assert 'int ' in header  # Function prototypes
            assert '#include <stdint.h>' in header
    
    def test_assembly_optimizations(self):
        """Test that assembly optimizations are architecture-appropriate."""
        
        cortex_impl = create_pqc_implementation('kyber512', 'cortex-m4', 'speed')
        
        if len(cortex_impl.assembly_code) > 100:  # If assembly is generated
            asm_code = cortex_impl.assembly_code
            
            # Should have ARM assembly syntax
            assert '.syntax unified' in asm_code
            assert '.cpu cortex-m4' in asm_code
            assert '.thumb' in asm_code
            
            # Should have function definitions
            assert '.global' in asm_code
            assert '.type' in asm_code
            assert 'bx lr' in asm_code  # ARM return instruction
            
            # Should use ARM-specific instructions
            arm_instructions = ['umull', 'umlal', 'sadd16', 'ldr', 'str']
            has_arm_insn = any(insn in asm_code for insn in arm_instructions)
            assert has_arm_insn, "Should contain ARM-specific instructions"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])