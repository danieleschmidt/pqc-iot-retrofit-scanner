# Development Guide

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Optional: Docker for containerized development

### Development Environment Setup

1. **Clone and setup**
   ```bash
   git clone https://github.com/terragon-ai/pqc-iot-retrofit-scanner.git
   cd pqc-iot-retrofit-scanner
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

2. **Install in development mode**
   ```bash
   pip install -e ".[dev,analysis]"
   ```

3. **Setup pre-commit hooks**
   ```bash
   pre-commit install
   ```

### Project Structure

```
src/pqc_iot_retrofit/          # Main package
├── __init__.py                # Package initialization
├── cli.py                     # Command-line interface
├── scanner.py                 # Firmware scanning logic
├── patcher.py                 # PQC patch generation
├── analysis/                  # Binary analysis modules
├── crypto/                    # Cryptographic implementations
├── targets/                   # Target device support
└── utils/                     # Utility functions

tests/                         # Test suite
├── unit/                      # Unit tests
├── integration/              # Integration tests
└── fixtures/                 # Test data

docs/                         # Documentation
├── guides/                   # User guides
├── api/                      # API documentation
└── examples/                 # Usage examples
```

### Testing

Run the complete test suite:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=src --cov-report=html
```

Run specific test categories:
```bash
pytest tests/unit/            # Unit tests only
pytest tests/integration/     # Integration tests only
pytest -m "not slow"          # Skip slow tests
```

### Code Quality

We maintain high code quality standards:

```bash
# Format code
black src/ tests/

# Lint and check imports
ruff check src/ tests/

# Type checking
mypy src/

# Run all quality checks
make lint  # If Makefile is available
```

### Debugging

For debugging firmware analysis:

1. **Enable verbose logging**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Use test fixtures**
   ```bash
   pytest tests/test_scanner.py::test_arm_firmware -v -s
   ```

3. **Interactive debugging**
   ```python
   from pqc_iot_retrofit import FirmwareScanner
   scanner = FirmwareScanner("cortex-m4")
   # Use debugger or print statements
   ```

### Adding New Target Architectures

1. Create target module in `src/pqc_iot_retrofit/targets/`
2. Implement `TargetBase` interface
3. Add architecture detection logic
4. Create test cases with sample firmware
5. Update documentation

### Performance Testing

For performance-critical changes:

```bash
# Run benchmarks
python -m pytest tests/benchmarks/ --benchmark-only

# Profile specific functions
python -m cProfile -o profile.out script.py
```

### Documentation

Update documentation when making changes:

```bash
# Build docs locally (if using Sphinx)
cd docs/
make html
```

### Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Create release PR
5. Tag release after merge

### Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Create a GitHub Issue
- **Security**: Email security@terragon.ai
- **Chat**: Join our development Slack (link in README)