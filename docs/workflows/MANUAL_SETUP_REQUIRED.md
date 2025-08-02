# Manual Workflow Setup Required

⚠️ **IMPORTANT**: Due to GitHub App permission limitations, the workflow files in this directory cannot be automatically created in your repository. You must manually copy these templates to implement the CI/CD pipeline.

## Required Actions

### 1. Create Workflow Directory
```bash
mkdir -p .github/workflows
```

### 2. Copy Template Files

Copy the following files from `docs/workflows/examples/` to `.github/workflows/`:

#### Core Workflows (Required)
- [ ] `ci.yml` → `.github/workflows/ci.yml`
- [ ] `security-scan.yml` → `.github/workflows/security-scan.yml`

#### Additional Workflows (Recommended)
- [ ] `cd.yml` → `.github/workflows/cd.yml`
- [ ] `dependency-update.yml` → `.github/workflows/dependency-update.yml`

### 3. Configure Repository Secrets

Add these secrets in **Repository Settings > Secrets and variables > Actions**:

#### Required Secrets
```bash
# Automatically provided by GitHub
GITHUB_TOKEN  # For package registry and API access

# Code coverage (optional but recommended)
CODECOV_TOKEN  # From codecov.io

# Security scanning (optional)
SEMGREP_APP_TOKEN  # From semgrep.dev for enhanced SAST
```

#### Deployment Secrets (if using CD workflow)
```bash
# Package publishing
PYPI_API_TOKEN  # For PyPI releases

# Container registry (if using external registry)
DOCKER_HUB_USERNAME
DOCKER_HUB_TOKEN

# Cloud deployment (if applicable)
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
KUBE_CONFIG_DATA
```

### 4. Configure Branch Protection Rules

**Repository Settings > Branches > Add rule** for `main`:

#### Protection Settings
- [x] **Require pull request reviews before merging**
  - Required reviewers: 1
  - Dismiss stale reviews when new commits are pushed
  - Require review from code owners

- [x] **Require status checks to pass before merging**
  - Require branches to be up to date before merging
  - Status checks to require:
    - `Code Quality`
    - `Test Suite (3.8)` (minimum Python version)
    - `Test Suite (3.11)` (latest Python version)
    - `Security Scan`
    - `Docker Build`

- [x] **Require conversation resolution before merging**

- [x] **Restrict pushes that create files exceeding 100MB**

#### Advanced Settings
- [x] **Do not allow bypassing the above settings**
- [x] **Restrict pushes that create files exceeding 100MB**
- [x] **Allow force pushes** (disabled)
- [x] **Allow deletions** (disabled)

### 5. Create Environment Configurations

**Repository Settings > Environments**:

#### Staging Environment
- **Name**: `staging`
- **Protection rules**: None (automatic deployment)
- **Environment secrets**: 
  - `DEPLOY_URL`: staging deployment URL
  - `HEALTH_CHECK_URL`: staging health check endpoint

#### Production Environment  
- **Name**: `production`
- **Protection rules**: 
  - [x] **Required reviewers**: Add production deployment reviewers
  - [x] **Wait timer**: 5 minutes (for rollback window)
- **Environment secrets**:
  - `DEPLOY_URL`: production deployment URL
  - `HEALTH_CHECK_URL`: production health check endpoint

### 6. Configure Issue and PR Templates

Create `.github/ISSUE_TEMPLATE/` and `.github/PULL_REQUEST_TEMPLATE.md`:

#### Bug Report Template
```yaml
# .github/ISSUE_TEMPLATE/bug_report.yml
name: Bug Report
description: Report a bug or unexpected behavior
title: "[BUG] "
labels: ["bug", "needs-triage"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for reporting a bug! Please fill out the sections below.
  
  - type: textarea
    id: description
    attributes:
      label: Bug Description
      description: A clear description of what the bug is
    validations:
      required: true
  
  - type: textarea
    id: reproduction
    attributes:
      label: Steps to Reproduce
      description: Steps to reproduce the behavior
      placeholder: |
        1. Run command '...'
        2. With firmware file '...'
        3. See error
    validations:
      required: true
  
  - type: textarea
    id: expected
    attributes:
      label: Expected Behavior
      description: What you expected to happen
    validations:
      required: true
      
  - type: textarea
    id: environment
    attributes:
      label: Environment
      description: Environment details
      value: |
        - OS: [e.g. Ubuntu 22.04]
        - Python version: [e.g. 3.11.0]
        - Package version: [e.g. 0.1.0]
    validations:
      required: true
```

#### Pull Request Template
```markdown
<!-- .github/PULL_REQUEST_TEMPLATE.md -->
## Description
Brief description of the changes in this PR.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring
- [ ] Security fix

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Manual testing completed
- [ ] Security implications considered

## Security Checklist
- [ ] No hardcoded secrets or credentials
- [ ] Input validation implemented where needed
- [ ] Error handling doesn't leak sensitive information
- [ ] Security tests updated if applicable

## Documentation
- [ ] Code is self-documenting with clear variable/function names
- [ ] Complex logic is commented
- [ ] Public API changes documented
- [ ] README updated if needed

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
```

### 7. Enable Dependency Graph and Security Features

**Repository Settings > Security & analysis**:

- [x] **Dependency graph**: Enabled
- [x] **Dependabot alerts**: Enabled  
- [x] **Dependabot security updates**: Enabled
- [x] **Dependabot version updates**: Enabled (create `.github/dependabot.yml`)
- [x] **Code scanning alerts**: Enabled
- [x] **Secret scanning alerts**: Enabled
- [x] **Secret scanning push protection**: Enabled

#### Dependabot Configuration
```yaml
# .github/dependabot.yml
version: 2
updates:
  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "06:00"
    open-pull-requests-limit: 5
    reviewers:
      - "@core-team"
    labels:
      - "dependencies"
      - "python"
    
  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "06:00"
    open-pull-requests-limit: 5
    reviewers:
      - "@core-team"
    labels:
      - "dependencies"
      - "github-actions"

  # Docker
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "06:00"
    open-pull-requests-limit: 5
    reviewers:
      - "@core-team"
    labels:
      - "dependencies"
      - "docker"
```

### 8. Verify Workflow Configuration

After copying the workflow files:

1. **Check workflow syntax**:
   - GitHub will automatically validate YAML syntax
   - Check the **Actions** tab for any syntax errors

2. **Test with a small PR**:
   - Create a test branch with a minor change
   - Open a pull request to trigger CI
   - Verify all checks pass

3. **Monitor workflow performance**:
   - Check job duration and resource usage
   - Optimize if any jobs take too long
   - Ensure proper artifact cleanup

### 9. Team Training and Documentation

#### For Developers
- Document the CI/CD process in team wiki
- Train team on workflow requirements
- Establish PR review guidelines
- Set up local development environment to match CI

#### For Maintainers  
- Document emergency procedures for workflow failures
- Establish escalation paths for security alerts
- Set up monitoring for workflow health
- Plan regular review of security scan results

## Troubleshooting Common Issues

### Workflow Not Triggering
- Check branch protection rules
- Verify workflow file is in correct location
- Check workflow syntax with GitHub Actions validator

### Test Failures
- Ensure local environment matches CI environment
- Check for race conditions in tests
- Verify test dependencies are properly specified

### Security Scan Failures
- Review security scan results in GitHub Security tab
- Update `.bandit` configuration for false positives
- Document security exceptions with justification

### Permission Issues
- Verify GITHUB_TOKEN has required permissions
- Check if custom secrets are properly configured
- Ensure workflows have correct permission declarations

## Support and Maintenance

### Regular Maintenance Tasks
- [ ] **Weekly**: Review security scan results
- [ ] **Monthly**: Update workflow dependencies (Actions versions)
- [ ] **Quarterly**: Review and optimize workflow performance
- [ ] **Annually**: Audit security configuration and access controls

### Getting Help
- **GitHub Actions Documentation**: https://docs.github.com/en/actions
- **Security Best Practices**: https://docs.github.com/en/actions/security-guides
- **Workflow Examples**: https://github.com/actions/starter-workflows

---

**Remember**: After completing the setup, delete this file or move it to a documentation directory to avoid confusion.