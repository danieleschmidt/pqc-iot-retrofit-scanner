# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- SDLC implementation with checkpointed approach
- Architecture Decision Records (ADRs) structure
- Comprehensive project roadmap and charter
- Enhanced documentation structure

### Changed
- Improved README with detailed usage examples
- Enhanced project structure documentation

### Fixed
- N/A

## [0.1.0] - 2025-01-15

### Added
- Initial firmware analysis engine for ARM Cortex-M
- Basic Dilithium2 signature algorithm support
- Kyber512 key encapsulation mechanism support
- CLI interface with essential commands
- Python API for programmatic access
- Basic vulnerability detection and reporting
- Memory-constrained optimization for embedded devices
- Side-channel protection implementations
- Docker containerization support
- Comprehensive testing framework
- Documentation and architecture guides

### Security
- Constant-time implementations for all cryptographic operations
- Side-channel attack resistance measures
- Secure memory handling for sensitive operations

## [0.0.1] - 2024-12-01

### Added
- Project structure and initial codebase
- Basic ARM Cortex-M disassembly capability
- Proof-of-concept cryptographic pattern detection
- Initial CLI framework
- Testing infrastructure
- Documentation foundation

---

## Release Types

This project uses [Semantic Versioning](https://semver.org/):

- **MAJOR** version when you make incompatible API changes
- **MINOR** version when you add functionality in a backwards compatible manner  
- **PATCH** version when you make backwards compatible bug fixes

## Security Releases

Security vulnerabilities are treated with highest priority:

- **Critical**: Immediate patch release (0.x.y+1)
- **High**: Patch within 7 days
- **Medium**: Patch within 30 days
- **Low**: Patch in next minor release

## Deprecation Policy

- Features marked as deprecated will be supported for at least 2 minor releases
- Breaking changes will be communicated 60 days in advance
- Migration guides will be provided for all breaking changes