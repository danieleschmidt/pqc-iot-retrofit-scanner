# Manual Setup Required

This document outlines the manual setup steps required to complete the SDLC implementation due to GitHub App permission limitations.

## GitHub Actions Workflows

The following GitHub Actions workflows need to be manually created in the `.github/workflows/` directory using the templates provided in `docs/workflows/examples/`:

### Required Workflows

#### 1. Continuous Integration (ci.yml)
**Template:** `docs/workflows/examples/ci.yml`
**Purpose:** PR validation, testing, and security scanning
**Triggers:** Pull requests, push to main
**Required Permissions:** 
- Contents: read
- Actions: read
- Security-events: write

#### 2. Security Scanning (security-scan.yml)
**Template:** `docs/workflows/examples/security-scan.yml`
**Purpose:** Comprehensive security scanning and SBOM generation
**Triggers:** Daily schedule, manual dispatch
**Required Permissions:**
- Contents: read
- Security-events: write
- Actions: read

### Setup Instructions

1. **Copy Template Files:**
   ```bash
   mkdir -p .github/workflows/
   cp docs/workflows/examples/ci.yml .github/workflows/
   cp docs/workflows/examples/security-scan.yml .github/workflows/
   ```

2. **Configure Repository Secrets:**
   Navigate to GitHub repository settings → Secrets and variables → Actions
   
   **Required Secrets:**
   - `CODECOV_TOKEN`: For code coverage reporting (if using Codecov)
   - `SONAR_TOKEN`: For SonarCloud integration (if using SonarQube)
   - `SNYK_TOKEN`: For Snyk vulnerability scanning (if using Snyk)

3. **Enable GitHub Security Features:**
   - Navigate to Settings → Security & analysis
   - Enable "Dependency graph"
   - Enable "Dependabot alerts"
   - Enable "Dependabot security updates"
   - Enable "Secret scanning"
   - Enable "Code scanning"

## Repository Settings Configuration

### Branch Protection Rules

Configure branch protection for the `main` branch:

1. Navigate to Settings → Branches
2. Add branch protection rule for `main`:
   - ✅ Require pull request reviews before merging
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date before merging
   - ✅ Require conversation resolution before merging
   - ✅ Restrict pushes that create files
   - ✅ Do not allow bypassing the above settings

**Required Status Checks:**
- `CI / test-python`
- `CI / security-scan`
- `CI / lint-and-format`
- `CodeQL`

### Repository Configuration

Update repository settings:

1. **General Settings:**
   - Description: "CLI + GitHub Action that audits embedded firmware and suggests post-quantum cryptography drop-ins (Kyber, Dilithium)"
   - Website: `https://pqc-iot-retrofit.readthedocs.io`
   - Topics: `pqc`, `iot`, `firmware`, `cryptography`, `security`, `quantum`, `embedded`

2. **Features:**
   - ✅ Issues
   - ✅ Projects  
   - ✅ Wiki (optional)
   - ✅ Discussions (recommended)

3. **Pull Requests:**
   - ✅ Allow squash merging
   - ✅ Allow auto-merge
   - ✅ Automatically delete head branches
   - ✅ Always suggest updating pull request branches

## Issue and PR Templates

Create the following template files:

### Issue Templates
Copy from `docs/workflows/examples/`:
```bash
mkdir -p .github/ISSUE_TEMPLATE/
cp docs/workflows/examples/bug_report.yml .github/ISSUE_TEMPLATE/
cp docs/workflows/examples/feature_request.yml .github/ISSUE_TEMPLATE/
cp docs/workflows/examples/security_report.yml .github/ISSUE_TEMPLATE/
```

### Pull Request Template
```bash
cp docs/workflows/examples/pull_request_template.md .github/PULL_REQUEST_TEMPLATE.md
```

## External Integrations

### Code Quality Tools

#### CodeCov
1. Sign up at https://codecov.io/
2. Connect your GitHub repository
3. Add `CODECOV_TOKEN` to repository secrets

#### SonarCloud (Optional)
1. Sign up at https://sonarcloud.io/
2. Import your repository
3. Add `SONAR_TOKEN` to repository secrets
4. Create `sonar-project.properties` file

#### Snyk (Optional)
1. Sign up at https://snyk.io/
2. Connect your GitHub repository
3. Add `SNYK_TOKEN` to repository secrets

### Documentation Hosting

#### Read the Docs
1. Sign up at https://readthedocs.org/
2. Connect your GitHub repository
3. Configure build settings for Sphinx documentation

### Package Distribution

#### PyPI
1. Create account at https://pypi.org/
2. Generate API token
3. Add `PYPI_API_TOKEN` to repository secrets for automated publishing

## Monitoring and Observability

### GitHub Insights
Enable and configure:
- Dependency insights
- Code frequency
- Contributor activity
- Traffic analytics

### Third-party Monitoring
Consider integrating:
- **Sentry**: Error tracking and performance monitoring
- **DataDog**: Application and infrastructure monitoring  
- **New Relic**: Performance monitoring

## Automation Setup

### Dependabot Configuration
The `.github/dependabot.yml` file is already configured. Ensure Dependabot is enabled in repository settings.

### GitHub Apps (Recommended)
Consider installing:
- **Renovate**: Advanced dependency management
- **CodeClimate**: Code quality analysis
- **Semantic Pull Requests**: Enforce conventional commits

## Security Configuration

### Security Policy
The `SECURITY.md` file is already created. Update it with:
- Current contact information
- Supported versions
- Reporting procedures

### GPG Signing (Recommended)
Configure commit signing:
1. Generate GPG key
2. Add to GitHub account
3. Configure local git:
   ```bash
   git config --global user.signingkey <key-id>
   git config --global commit.gpgsign true
   ```

## Verification Checklist

After completing manual setup, verify:

- [ ] All GitHub Actions workflows are running successfully
- [ ] Branch protection rules are enforced
- [ ] Code coverage reporting is working
- [ ] Security scanning is active
- [ ] Issue and PR templates are available
- [ ] External integrations are connected
- [ ] Repository settings are properly configured
- [ ] Documentation is building correctly
- [ ] Automated dependency updates are working

## Support

If you encounter issues during setup:

1. Check the workflow logs in GitHub Actions
2. Verify repository permissions and secrets
3. Review the template files in `docs/workflows/examples/`
4. Consult the project documentation
5. Open an issue using the provided templates

## Next Steps

Once manual setup is complete:

1. Create a test pull request to verify CI/CD pipeline
2. Run security scans to establish baseline
3. Configure monitoring dashboards
4. Set up automated releases using semantic-release
5. Establish development workflow documentation

---

**Last Updated:** January 2025  
**Maintained By:** Repository administrators