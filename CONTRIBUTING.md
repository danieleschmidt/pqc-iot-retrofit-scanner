# Contributing to PQC IoT Retrofit Scanner

We welcome contributions to enhance post-quantum cryptography adoption in IoT devices!

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/terragon-ai/pqc-iot-retrofit-scanner.git
   cd pqc-iot-retrofit-scanner
   ```

2. **Install dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## Testing

Run the test suite:
```bash
pytest
```

With coverage:
```bash
pytest --cov
```

## Code Quality

We use several tools to maintain code quality:

- **Black**: Code formatting
- **Ruff**: Linting and import sorting  
- **MyPy**: Type checking

Run all checks:
```bash
black src/ tests/
ruff check src/ tests/
mypy src/
```

## Security Focus

This project focuses on **defensive security only**:

✅ **Allowed contributions:**
- Vulnerability detection algorithms
- PQC implementation optimizations
- Firmware analysis improvements
- Security documentation
- Compliance tooling

❌ **Not accepted:**
- Exploit development
- Attack tools
- Malicious code analysis

## Contribution Process

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes
4. **Add** tests for new functionality
5. **Ensure** all tests pass
6. **Commit** with clear messages
7. **Push** to your fork
8. **Create** a Pull Request

## Pull Request Guidelines

- **Title**: Clear, descriptive summary
- **Description**: Explain the problem and solution
- **Tests**: Include appropriate test coverage
- **Documentation**: Update docs if needed
- **Breaking Changes**: Clearly document any breaking changes

## Priority Areas

We especially welcome contributions in:

- **Additional MCU architectures** (RISC-V, AVR, etc.)
- **Lightweight PQC variants** for constrained devices
- **Firmware analysis techniques**
- **Power analysis resistance**
- **Formal verification methods**

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- **Security vulnerabilities**: Email security@terragon.ai
- Provide detailed reproduction steps
- Include system information and firmware samples if applicable

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/).
Be respectful, inclusive, and professional in all interactions.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.