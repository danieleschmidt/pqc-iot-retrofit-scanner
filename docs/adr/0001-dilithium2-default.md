# ADR-0001: Use Dilithium2 as Default Signature Algorithm

## Status

Accepted

## Context

Post-quantum signature algorithms standardized by NIST include Dilithium (Levels 1-5) and SPHINCS+ (various parameter sets). For IoT device retrofitting, we need to choose a default signature algorithm that balances:

- Security level appropriate for IoT threat models
- Memory constraints of embedded devices (32KB-512KB flash, 8KB-128KB RAM)
- Performance requirements for real-time operations
- Implementation maturity and side-channel resistance

Key considerations:
- Dilithium offers faster verification than SPHINCS+
- Dilithium has smaller signature sizes than SPHINCS+
- Dilithium2 provides NIST Security Level 2 (equivalent to AES-128)
- Most IoT devices being retrofitted currently use RSA-2048 or ECDSA-P256

## Decision

We will use **Dilithium2** as the default post-quantum signature algorithm for IoT device retrofitting, with the following implementation strategy:

1. **Primary Algorithm**: Dilithium2 (NIST Level 2)
2. **Memory-Optimized Variant**: Stack-based implementation with <12KB RAM usage
3. **Fallback Option**: SPHINCS+-128s for devices requiring stateless signatures
4. **Hybrid Mode**: Optional classical+PQC hybrid signatures during transition

Implementation specifics:
- Use optimized NTT implementations for ARM Cortex-M
- Implement constant-time operations for side-channel resistance
- Provide drop-in compatibility layer for existing RSA/ECDSA APIs

## Consequences

**Positive:**
- Dilithium2 signatures are ~2.4KB, manageable for IoT communication
- Verification is fast (~2.5ms on Cortex-M4), suitable for real-time systems
- Well-studied algorithm with available optimized implementations
- Memory footprint fits in typical IoT device constraints

**Negative:**
- Larger key and signature sizes than classical algorithms
- Requires careful implementation to avoid side-channel vulnerabilities
- May need fallback for extremely constrained devices (<32KB flash)

**Neutral:**
- Requires retraining of developers on PQC algorithm specifics
- May impact OTA update mechanisms due to larger signature sizes

## References

- [NIST PQC Standardization](https://csrc.nist.gov/projects/post-quantum-cryptography)
- [Dilithium Specification](https://pq-crystals.org/dilithium/)
- [ARM Cortex-M Dilithium Implementation](https://github.com/mupq/pqm4)