# GitHub Workflows Documentation and Templates

This directory contains comprehensive CI/CD workflow documentation and templates for the PQC IoT Retrofit Scanner project.

⚠️ **Important**: Due to GitHub App permission limitations, workflow files cannot be created automatically. Repository maintainers must manually create these workflows in the `.github/workflows/` directory using the templates provided below.

## Core Workflows

### 1. Continuous Integration (`ci.yml`)

**Purpose**: Run tests, linting, and security checks on every PR and push

**Recommended triggers**:
- Pull requests to `main` branch
- Pushes to `main` branch
- Manual workflow dispatch

**Key steps**:
```yaml
jobs:
  test:
    strategy:
      matrix:
        python-version: ["3.8", "3.9", "3.10", "3.11"]
        os: [ubuntu-latest, windows-latest, macos-latest]
    
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
      - run: pip install -e ".[dev]"
      - run: pytest --cov --cov-report=xml
      - uses: codecov/codecov-action@v3

  lint:
    steps:
      - run: black --check src/ tests/
      - run: ruff check src/ tests/
      - run: mypy src/

  security:
    steps:
      - run: bandit -r src/
      - run: safety check
```

### 2. Security Scanning (`security.yml`)

**Purpose**: Regular security scans and dependency updates

**Recommended schedule**: Weekly

**Key components**:
- Dependency vulnerability scanning with Safety
- SAST scanning with Bandit
- License compliance checking
- Container security scanning (if using Docker)

### 3. Release (`release.yml`)

**Purpose**: Automated package building and publishing

**Triggers**: 
- Git tags matching `v*.*.*`
- Manual workflow dispatch

**Steps**:
- Build wheel and source distributions
- Run full test suite
- Publish to PyPI (with approval)
- Create GitHub release with changelog

### 4. Firmware Testing (`firmware-testing.yml`)

**Purpose**: Hardware-in-loop testing with real firmware samples

**Recommended schedule**: Nightly

**Requirements**:
- Self-hosted runners with test hardware
- Secure firmware sample storage
- Test result aggregation and reporting

## Implementation Guide

### Step 1: Create Workflow Files

Create these files in `.github/workflows/`:

1. `ci.yml` - Core CI/CD pipeline  
2. `security.yml` - Security scanning
3. `release.yml` - Release automation
4. `firmware-testing.yml` - Hardware testing

### Step 2: Configure Secrets

Add these secrets in GitHub repository settings:

- `PYPI_API_TOKEN` - PyPI publishing token
- `CODECOV_TOKEN` - Code coverage reporting
- `SECURITY_EMAIL` - Security notification email

### Step 3: Setup Branch Protection

Configure branch protection rules:

- Require status checks to pass
- Require up-to-date branches
- Require review from code owners
- Restrict pushes to `main` branch

## Workflow Templates

### Basic CI Template

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.8", "3.11"]
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
      
      - name: Run tests
        run: pytest --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Security Scanning Template

```yaml
name: Security

on:
  schedule:
    - cron: '0 3 * * 1'  # Weekly Monday 3AM
  workflow_dispatch:

jobs:
  security:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      
      - name: Install dependencies
        run: pip install -e ".[dev]" bandit safety
      
      - name: Run Bandit
        run: bandit -r src/ -f json -o bandit-report.json
      
      - name: Run Safety
        run: safety check --json --output safety-report.json
      
      - name: Upload security reports
        uses: actions/upload-artifact@v3
        with:
          name: security-reports
          path: "*-report.json"
```

## Best Practices

### Security Considerations

1. **Secrets Management**:
   - Use GitHub secrets for sensitive data
   - Rotate tokens regularly
   - Limit secret access to necessary workflows

2. **Dependency Security**:
   - Pin action versions to specific commits
   - Use Dependabot for dependency updates
   - Regular security scanning of dependencies

3. **Firmware Sample Handling**:
   - Store test firmware in private repositories
   - Use encrypted storage for sensitive samples
   - Implement access logging and audit trails

### Performance Optimization

1. **Matrix Builds**:
   - Use fail-fast: false for comprehensive testing
   - Cache dependencies between runs
   - Parallelize independent test suites

2. **Resource Usage**:
   - Use appropriate runner sizes
   - Consider self-hosted runners for specialized hardware
   - Implement workflow timeouts

### Monitoring and Alerting

1. **Workflow Health**:
   - Monitor workflow success rates
   - Set up alerts for critical failures
   - Track build time trends

2. **Security Notifications**:
   ```yaml
   - name: Notify security team
     if: failure()
     uses: actions/github-script@v6
     with:
       script: |
         github.rest.issues.create({
           owner: context.repo.owner,
           repo: context.repo.repo,
           title: 'Security scan failed',
           body: 'Security workflow failed. Please investigate.',
           labels: ['security', 'urgent']
         })
   ```

## Integration with External Services

### Code Quality Services

- **Codecov**: Coverage reporting and analysis
- **SonarCloud**: Code quality and security analysis  
- **Snyk**: Vulnerability scanning and monitoring

### Deployment Services

- **PyPI**: Package distribution
- **GitHub Packages**: Alternative package registry
- **Docker Hub**: Container image distribution

## Troubleshooting

### Common Issues

1. **Test Failures in Matrix Builds**:
   - Check Python version compatibility
   - Verify dependency availability across platforms
   - Review platform-specific test requirements

2. **Security Scan False Positives**:
   - Use `.bandit` configuration file for exclusions
   - Document justified security exceptions
   - Regular review of security scan results

3. **Resource Limits**:
   - Monitor workflow resource usage
   - Optimize large test suites
   - Consider workflow splitting for complex pipelines

For detailed workflow implementation, see the individual template files in this directory.