# Architecture Overview

## System Design

PQC IoT Retrofit Scanner is designed as a modular system for analyzing embedded firmware and generating post-quantum cryptography patches.

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLI Interface │    │  Python API     │    │  GitHub Action  │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │      Core Engine          │
                    │  ┌─────────────────────┐  │
                    │  │  Firmware Scanner   │  │
                    │  └─────────────────────┘  │
                    │  ┌─────────────────────┐  │
                    │  │   PQC Patcher      │  │
                    │  └─────────────────────┘  │
                    │  ┌─────────────────────┐  │
                    │  │  Target Adapters   │  │
                    │  └─────────────────────┘  │
                    └───────────┬───────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌───────▼────────┐    ┌─────────▼─────────┐    ┌───────▼────────┐
│ Binary Analysis│    │   Crypto Library  │    │  Output Engine │
│   - Capstone   │    │   - Kyber/Dilith. │    │   - Reports    │
│   - LIEF       │    │   - Optimizations │    │   - Patches    │
│   - Angr       │    │   - Side-channel  │    │   - SBOM       │
└────────────────┘    └───────────────────┘    └────────────────┘
```

## Core Components

### 1. Firmware Scanner (`scanner.py`)

**Purpose**: Analyze firmware binaries to detect quantum-vulnerable cryptography

**Key Responsibilities**:
- Disassemble firmware for target architecture
- Pattern match cryptographic function signatures
- Extract cryptographic parameters (key sizes, algorithms)
- Assess memory constraints and performance characteristics

**Architecture Detection**:
- ARM Cortex-M series (M0, M3, M4, M7)
- ESP32/ESP8266 (Xtensa)
- RISC-V variants
- AVR microcontrollers

### 2. PQC Patcher (`patcher.py`)

**Purpose**: Generate post-quantum cryptography replacement patches

**Supported Algorithms**:
- **Dilithium** (NIST Level 1-5) - Digital signatures
- **Kyber** (NIST Level 1-5) - Key encapsulation
- **SPHINCS+** - Stateless signatures (optional)

**Optimization Strategies**:
- Memory-constrained implementations
- Speed-optimized variants
- Hybrid classical+PQC modes
- Progressive deployment support

### 3. Target Adapters (`targets/`)

**Purpose**: Device-specific optimizations and constraints

```python
class TargetBase:
    def get_memory_layout(self) -> MemoryLayout
    def get_instruction_set(self) -> InstructionSet
    def optimize_pqc_params(self, algorithm: str) -> PQCParams
    def generate_device_specific_code(self) -> str
```

**Supported Targets**:
- STM32 family (L4, F4, H7 series)
- ESP32 variants (ESP32, ESP32-S2, ESP32-S3)
- Nordic nRF52 series
- TI MSP430 series

## Data Flow

### 1. Firmware Analysis Pipeline

```
Firmware Binary
     │
     ▼
┌─────────────────┐
│   File Parser   │ ← LIEF, custom parsers
├─────────────────┤
│  Architecture   │ ← Auto-detection via headers/signatures
│   Detection     │
├─────────────────┤
│ Disassembly     │ ← Capstone engine
├─────────────────┤
│ Function        │ ← Control flow analysis
│ Extraction      │
├─────────────────┤
│ Crypto Pattern  │ ← Signature matching, constant detection
│ Matching        │
├─────────────────┤
│ Vulnerability   │ ← Risk assessment, algorithm classification
│ Assessment      │
└─────────────────┘
     │
     ▼
Vulnerability Report
```

### 2. Patch Generation Pipeline

```
Vulnerability Report
     │
     ▼
┌─────────────────┐
│ Target Analysis │ ← Memory constraints, performance requirements
├─────────────────┤
│ Algorithm       │ ← Select PQC algorithm and parameters
│ Selection       │
├─────────────────┤
│ Code Generation │ ← Generate optimized implementation
├─────────────────┤
│ Integration     │ ← Create drop-in replacement interface
├─────────────────┤
│ Validation      │ ← Test compatibility and correctness
├─────────────────┤
│ Packaging       │ ← Create deployable patch
└─────────────────┘
     │
     ▼
Deployment-Ready Patch
```

## Memory Management

### Constrained Device Considerations

**Flash Memory Usage**:
- Kyber-512: ~13KB (including key generation)
- Dilithium2: ~87KB (optimized implementation)
- Shared code: ~15KB (NTT, polynomial operations)

**RAM Usage Optimization**:
- In-place operations to minimize memory copies
- Stack-based computation where possible
- Shared memory pools for temporary variables
- Progressive key generation for large keys

**Example Memory Layout**:
```
Flash (512KB total):
├── Bootloader (32KB)
├── Application (300KB)
├── PQC Library (100KB)
├── Configuration (16KB)
└── Reserved (64KB)

RAM (128KB total):
├── Application Stack (32KB)
├── PQC Working Memory (48KB)
├── Network Buffers (32KB)
└── System Reserved (16KB)
```

## Security Architecture

### Threat Model

**In Scope**:
- Quantum computer attacks on classical cryptography
- Side-channel attacks (timing, power analysis)
- Implementation vulnerabilities in PQC algorithms
- Rollback attacks during migration

**Out of Scope**:
- Physical hardware attacks
- Social engineering
- Supply chain attacks on hardware

### Side-Channel Protection

**Timing Attack Prevention**:
- Constant-time implementations for all cryptographic operations
- Dummy operations to normalize execution time
- Secret-independent memory access patterns

**Power Analysis Protection**:
- Randomized computation order where possible
- Masking techniques for sensitive operations
- Hardware countermeasures when available

## Performance Characteristics

### Benchmarks (Cortex-M4 @ 80MHz)

| Operation | Classical | Kyber-512 | Dilithium2 |
|-----------|-----------|-----------|------------|
| Key Generation | 0.1ms | 0.9ms | 1.8ms |
| Sign/Encap | 0.1ms | 1.1ms | 8.7ms |
| Verify/Decap | 0.1ms | 1.2ms | 2.5ms |
| Memory Peak | 2KB | 6KB | 11KB |

### Optimization Techniques

**Algorithmic Optimizations**:
- Number Theoretic Transform (NTT) optimizations
- Precomputed constants and tables
- Loop unrolling for critical paths
- Assembly optimizations for target architecture

**System Optimizations**:
- DMA usage for large data transfers
- Hardware cryptographic accelerators
- Interrupt-aware implementations
- Power management integration

## Extensibility

### Adding New Architectures

1. **Create target adapter** in `targets/new_arch.py`
2. **Implement instruction patterns** for crypto detection
3. **Add optimized PQC implementations** for the architecture
4. **Create test suite** with sample firmware
5. **Update documentation** and architecture matrix

### Adding New PQC Algorithms

1. **Implement algorithm interface** in `crypto/new_algorithm.py`
2. **Add optimization variants** for different targets
3. **Create security evaluation** and side-channel analysis
4. **Add integration tests** and benchmarks
5. **Update patcher** to support new algorithm

## Quality Assurance

### Testing Strategy

**Unit Tests**: Individual component functionality
**Integration Tests**: End-to-end firmware analysis
**Hardware-in-Loop**: Real device validation
**Security Tests**: Side-channel and fuzzing tests
**Performance Tests**: Benchmark regression testing

### Continuous Integration

```yaml
Test Matrix:
  - Python versions: 3.8, 3.9, 3.10, 3.11
  - Target architectures: ARM, RISC-V, AVR
  - Test firmware samples: 50+ real-world binaries
  - Security validation: Side-channel test suite
```

This architecture provides a solid foundation for secure, efficient post-quantum cryptography retrofitting of IoT devices while maintaining extensibility for future algorithms and target platforms.