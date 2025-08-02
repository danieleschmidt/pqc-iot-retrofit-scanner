#!/bin/bash
set -euo pipefail

# Automation setup script for PQC IoT Retrofit Scanner
# Sets up automated metrics collection, health monitoring, and reporting

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
AUTOMATION_CONFIG="$PROJECT_ROOT/.github/automation-config.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Setup automation for PQC IoT Retrofit Scanner project.

OPTIONS:
    -h, --help              Show this help message
    -d, --dry-run          Show what would be done without making changes
    -e, --environment      Environment to setup (development|staging|production)
    -c, --config-file      Path to automation config file
    --skip-dependencies    Skip dependency checks and installations
    --skip-validation      Skip configuration validation
    --force                Force setup even if already configured

EXAMPLES:
    $0                                          # Setup with defaults
    $0 --environment production                 # Setup for production
    $0 --dry-run                               # See what would be done
    $0 --config-file custom-config.json        # Use custom config

EOF
}

# Parse command line arguments
DRY_RUN=false
ENVIRONMENT="development"
SKIP_DEPENDENCIES=false
SKIP_VALIDATION=false
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -c|--config-file)
            AUTOMATION_CONFIG="$2"
            shift 2
            ;;
        --skip-dependencies)
            SKIP_DEPENDENCIES=true
            shift
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

log_info "Starting automation setup for environment: $ENVIRONMENT"

# Check if running in supported environment
check_environment() {
    log_info "Checking environment..."
    
    if [[ ! "$ENVIRONMENT" =~ ^(development|staging|production)$ ]]; then
        log_error "Invalid environment: $ENVIRONMENT. Must be development, staging, or production."
        exit 1
    fi
    
    # Check if we're in a git repository
    if [ ! -d "$PROJECT_ROOT/.git" ]; then
        log_error "Not a git repository. Please run from within the project directory."
        exit 1
    fi
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        log_info "Python version: $PYTHON_VERSION"
        
        if [[ $(echo "$PYTHON_VERSION >= 3.8" | bc -l) -eq 0 ]] 2>/dev/null || ! command -v bc &> /dev/null; then
            # Fallback version check without bc
            MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
            MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
            if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 8 ]); then
                log_error "Python 3.8 or higher is required. Found: $PYTHON_VERSION"
                exit 1
            fi
        fi
    else
        log_error "Python 3 is not installed or not in PATH"
        exit 1
    fi
    
    log_success "Environment check passed"
}

# Install required dependencies
install_dependencies() {
    if [ "$SKIP_DEPENDENCIES" = true ]; then
        log_info "Skipping dependency installation"
        return
    fi
    
    log_info "Installing required dependencies..."
    
    # Check if pip is available
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 is not installed or not in PATH"
        exit 1
    fi
    
    # Install Python dependencies for automation scripts
    local requirements_file="$PROJECT_ROOT/requirements-automation.txt"
    
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would install dependencies from $requirements_file"
        return
    fi
    
    # Create requirements file if it doesn't exist
    if [ ! -f "$requirements_file" ]; then
        log_info "Creating automation requirements file..."
        cat > "$requirements_file" << EOF
# Dependencies for automation scripts
requests>=2.28.0
psutil>=5.9.0
GitPython>=3.1.0
pytest>=7.0.0
pytest-benchmark>=4.0.0
bandit>=1.7.0
safety>=2.0.0
coverage>=6.0.0
radon>=5.1.0
EOF
    fi
    
    log_info "Installing Python dependencies..."
    pip3 install -r "$requirements_file" --user --quiet
    
    # Check for optional system dependencies
    check_optional_dependencies
    
    log_success "Dependencies installed successfully"
}

# Check for optional system dependencies
check_optional_dependencies() {
    log_info "Checking optional system dependencies..."
    
    # Docker (optional for container scanning)
    if command -v docker &> /dev/null; then
        log_success "Docker found: $(docker --version)"
    else
        log_warning "Docker not found. Container scanning will be disabled."
    fi
    
    # Git (should be available)
    if command -v git &> /dev/null; then
        log_success "Git found: $(git --version)"
    else
        log_error "Git is required but not found"
        exit 1
    fi
    
    # Optional tools
    local optional_tools=("jq" "curl" "bc")
    for tool in "${optional_tools[@]}"; do
        if command -v "$tool" &> /dev/null; then
            log_success "$tool found"
        else
            log_warning "$tool not found (optional)"
        fi
    done
}

# Validate configuration file
validate_configuration() {
    if [ "$SKIP_VALIDATION" = true ]; then
        log_info "Skipping configuration validation"
        return
    fi
    
    log_info "Validating automation configuration..."
    
    if [ ! -f "$AUTOMATION_CONFIG" ]; then
        log_error "Automation config file not found: $AUTOMATION_CONFIG"
        exit 1
    fi
    
    # Validate JSON syntax
    if command -v jq &> /dev/null; then
        if ! jq empty "$AUTOMATION_CONFIG" 2>/dev/null; then
            log_error "Invalid JSON in configuration file: $AUTOMATION_CONFIG"
            exit 1
        fi
        log_success "Configuration JSON is valid"
    else
        log_warning "jq not available, skipping JSON validation"
    fi
    
    # Basic configuration checks
    python3 -c "
import json
import sys

try:
    with open('$AUTOMATION_CONFIG', 'r') as f:
        config = json.load(f)
    
    # Check required sections
    required_sections = ['automation', 'integrations', 'feature_flags']
    missing_sections = [s for s in required_sections if s not in config]
    
    if missing_sections:
        print(f'Missing required sections: {missing_sections}')
        sys.exit(1)
    
    # Check environment-specific config
    if 'environment_specific' in config and '$ENVIRONMENT' in config['environment_specific']:
        print('Environment-specific configuration found')
    
    print('Configuration validation passed')
    
except Exception as e:
    print(f'Configuration validation failed: {e}')
    sys.exit(1)
"
    
    if [ $? -ne 0 ]; then
        log_error "Configuration validation failed"
        exit 1
    fi
    
    log_success "Configuration validation passed"
}

# Setup directory structure
setup_directories() {
    log_info "Setting up directory structure..."
    
    local directories=(
        ".github/automation"
        ".github/metrics"
        ".github/reports"
        "logs/automation"
        "data/metrics"
        "data/benchmarks"
        "data/debt-reports"
    )
    
    for dir in "${directories[@]}"; do
        local full_path="$PROJECT_ROOT/$dir"
        if [ "$DRY_RUN" = true ]; then
            log_info "[DRY RUN] Would create directory: $full_path"
        else
            mkdir -p "$full_path"
            log_info "Created directory: $full_path"
        fi
    done
    
    log_success "Directory structure setup complete"
}

# Configure automation scripts
configure_scripts() {
    log_info "Configuring automation scripts..."
    
    local scripts_dir="$PROJECT_ROOT/scripts"
    local automation_scripts=(
        "collect-metrics.py"
        "health-monitor.py"
        "benchmark-runner.py"
        "technical-debt-tracker.py"
    )
    
    for script in "${automation_scripts[@]}"; do
        local script_path="$scripts_dir/$script"
        
        if [ ! -f "$script_path" ]; then
            log_warning "Automation script not found: $script_path"
            continue
        fi
        
        if [ "$DRY_RUN" = true ]; then
            log_info "[DRY RUN] Would make $script executable"
        else
            chmod +x "$script_path"
            log_info "Made $script executable"
        fi
        
        # Validate script syntax
        if python3 -m py_compile "$script_path" 2>/dev/null; then
            log_success "$script syntax is valid"
        else
            log_error "$script has syntax errors"
            exit 1
        fi
    done
    
    log_success "Automation scripts configured"
}

# Setup environment variables
setup_environment_variables() {
    log_info "Setting up environment variables..."
    
    local env_file="$PROJECT_ROOT/.env.automation"
    
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would create environment file: $env_file"
        return
    fi
    
    # Create environment file with placeholders
    cat > "$env_file" << EOF
# Automation Environment Variables
# Copy this file to .env.automation.local and fill in actual values

# Environment
ENVIRONMENT=$ENVIRONMENT

# GitHub Integration
GITHUB_TOKEN=your_github_token_here
GITHUB_REPOSITORY=your_org/pqc-iot-retrofit-scanner

# Notification Channels
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
TECH_LEAD_EMAIL=tech-lead@example.com
PRODUCT_MANAGER_EMAIL=pm@example.com
EXECUTIVE_EMAIL=exec@example.com
TEAM_EMAIL_LIST=team@example.com

# Optional Integrations
CODECOV_TOKEN=your_codecov_token_here
SONARQUBE_URL=https://sonarqube.example.com
SONARQUBE_TOKEN=your_sonarqube_token_here

# SMTP Configuration (optional)
SMTP_SERVER=smtp.example.com
SMTP_PORT=587
SMTP_USERNAME=automation@example.com
SMTP_PASSWORD=your_smtp_password_here

# Storage and Backup (optional)
BACKUP_STORAGE_URL=s3://your-backup-bucket/pqc-scanner

# Monitoring (optional)
PROMETHEUS_ENDPOINT=http://prometheus.example.com:9090
GRAFANA_URL=https://grafana.example.com
GRAFANA_API_KEY=your_grafana_api_key_here
EOF
    
    log_success "Environment variables template created: $env_file"
    log_warning "Please copy $env_file to .env.automation.local and fill in actual values"
}

# Setup cron jobs (if not using GitHub Actions)
setup_cron_jobs() {
    if [ "$ENVIRONMENT" = "development" ]; then
        log_info "Skipping cron setup for development environment"
        return
    fi
    
    log_info "Setting up cron jobs for automation..."
    
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would setup cron jobs for automation"
        return
    fi
    
    local cron_file="$PROJECT_ROOT/.github/automation/crontab"
    
    # Create cron configuration
    cat > "$cron_file" << EOF
# PQC IoT Retrofit Scanner Automation Cron Jobs
# Install with: crontab $cron_file

# Daily metrics collection at 6 AM UTC
0 6 * * * cd $PROJECT_ROOT && python3 scripts/collect-metrics.py --config .github/project-metrics.json

# Health monitoring every 6 hours
0 */6 * * * cd $PROJECT_ROOT && python3 scripts/health-monitor.py --send-alerts

# Weekly performance benchmarks on Sunday at 2 AM UTC  
0 2 * * 0 cd $PROJECT_ROOT && python3 scripts/benchmark-runner.py --category all

# Weekly technical debt analysis on Monday at 4 AM UTC
0 4 * * 1 cd $PROJECT_ROOT && python3 scripts/technical-debt-tracker.py

# Daily security scanning at 3 AM UTC
0 3 * * * cd $PROJECT_ROOT && python3 scripts/collect-metrics.py --focus security
EOF
    
    log_success "Cron configuration created: $cron_file"
    log_info "To install cron jobs, run: crontab $cron_file"
}

# Setup GitHub Actions integration
setup_github_actions() {
    log_info "Setting up GitHub Actions integration..."
    
    local workflows_dir="$PROJECT_ROOT/.github/workflows"
    local automation_workflow="$workflows_dir/automation.yml"
    
    if [ ! -d "$workflows_dir" ]; then
        if [ "$DRY_RUN" = true ]; then
            log_info "[DRY RUN] Would create workflows directory"
        else
            mkdir -p "$workflows_dir"
        fi
    fi
    
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would create automation workflow: $automation_workflow"
        return
    fi
    
    # Create automation workflow
    cat > "$automation_workflow" << 'EOF'
name: Automation and Monitoring

on:
  schedule:
    # Daily metrics collection at 6 AM UTC
    - cron: '0 6 * * *'
    # Health monitoring every 6 hours
    - cron: '0 */6 * * *'
    # Weekly benchmarks on Sunday at 2 AM UTC
    - cron: '0 2 * * 0'
    # Weekly technical debt analysis on Monday at 4 AM UTC
    - cron: '0 4 * * 1'
  workflow_dispatch:
    inputs:
      automation_type:
        description: 'Type of automation to run'
        required: true
        default: 'metrics'
        type: choice
        options:
          - metrics
          - health
          - benchmarks
          - debt
          - all

env:
  PYTHON_VERSION: '3.11'

jobs:
  automation:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      issues: write
      security-events: write
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-automation.txt
    
    - name: Run metrics collection
      if: github.event.schedule == '0 6 * * *' || github.event.inputs.automation_type == 'metrics' || github.event.inputs.automation_type == 'all'
      run: |
        python scripts/collect-metrics.py --config .github/project-metrics.json
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        CODECOV_TOKEN: ${{ secrets.CODECOV_TOKEN }}
    
    - name: Run health monitoring
      if: github.event.schedule == '0 */6 * * *' || github.event.inputs.automation_type == 'health' || github.event.inputs.automation_type == 'all'
      run: |
        python scripts/health-monitor.py --send-alerts
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
    
    - name: Run performance benchmarks
      if: github.event.schedule == '0 2 * * 0' || github.event.inputs.automation_type == 'benchmarks' || github.event.inputs.automation_type == 'all'
      run: |
        python scripts/benchmark-runner.py --category all
    
    - name: Run technical debt analysis
      if: github.event.schedule == '0 4 * * 1' || github.event.inputs.automation_type == 'debt' || github.event.inputs.automation_type == 'all'
      run: |
        python scripts/technical-debt-tracker.py
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Upload automation results
      uses: actions/upload-artifact@v4
      if: always()
      with:
        name: automation-results
        path: |
          .github/metrics-results.json
          .github/health-report.json
          benchmark-results.json
          technical-debt-report.json
        retention-days: 90
    
    - name: Create summary
      if: always()
      run: |
        echo "## Automation Results" >> $GITHUB_STEP_SUMMARY
        echo "" >> $GITHUB_STEP_SUMMARY
        
        if [ -f ".github/health-report.json" ]; then
          HEALTH_STATUS=$(python -c "import json; print(json.load(open('.github/health-report.json'))['health_summary']['overall_status'])")
          echo "**Health Status:** $HEALTH_STATUS" >> $GITHUB_STEP_SUMMARY
        fi
        
        if [ -f ".github/metrics-results.json" ]; then
          METRICS_COUNT=$(python -c "import json; print(json.load(open('.github/metrics-results.json'))['metrics_count'])")
          echo "**Metrics Collected:** $METRICS_COUNT" >> $GITHUB_STEP_SUMMARY
        fi
        
        if [ -f "benchmark-results.json" ]; then
          BENCHMARK_COUNT=$(python -c "import json; print(json.load(open('benchmark-results.json'))['benchmark_count'])")
          echo "**Benchmarks Run:** $BENCHMARK_COUNT" >> $GITHUB_STEP_SUMMARY
        fi
        
        if [ -f "technical-debt-report.json" ]; then
          DEBT_ITEMS=$(python -c "import json; print(json.load(open('technical-debt-report.json'))['total_debt_items'])")
          echo "**Technical Debt Items:** $DEBT_ITEMS" >> $GITHUB_STEP_SUMMARY
        fi
EOF
    
    log_success "GitHub Actions automation workflow created"
}

# Validate setup
validate_setup() {
    log_info "Validating automation setup..."
    
    local validation_errors=0
    
    # Check if required files exist
    local required_files=(
        "$PROJECT_ROOT/scripts/collect-metrics.py"
        "$PROJECT_ROOT/scripts/health-monitor.py" 
        "$PROJECT_ROOT/scripts/benchmark-runner.py"
        "$PROJECT_ROOT/scripts/technical-debt-tracker.py"
        "$PROJECT_ROOT/.github/project-metrics.json"
        "$PROJECT_ROOT/.github/automation-config.json"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            log_error "Required file missing: $file"
            ((validation_errors++))
        fi
    done
    
    # Test script execution (dry run)
    local test_scripts=(
        "python3 scripts/collect-metrics.py --help"
        "python3 scripts/health-monitor.py --help"
        "python3 scripts/benchmark-runner.py --help"
        "python3 scripts/technical-debt-tracker.py --help"
    )
    
    for cmd in "${test_scripts[@]}"; do
        if ! $cmd &>/dev/null; then
            log_error "Script test failed: $cmd"
            ((validation_errors++))
        fi
    done
    
    if [ $validation_errors -gt 0 ]; then
        log_error "Setup validation failed with $validation_errors errors"
        exit 1
    fi
    
    log_success "Setup validation passed"
}

# Generate setup report
generate_setup_report() {
    log_info "Generating setup report..."
    
    local report_file="$PROJECT_ROOT/.github/automation/setup-report.json"
    
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would generate setup report: $report_file"
        return
    fi
    
    python3 -c "
import json
import os
from datetime import datetime

report = {
    'setup_timestamp': datetime.now().isoformat(),
    'environment': '$ENVIRONMENT',
    'project_root': '$PROJECT_ROOT',
    'automation_config': '$AUTOMATION_CONFIG',
    'features_enabled': {
        'metrics_collection': True,
        'health_monitoring': True,
        'performance_benchmarks': True,
        'technical_debt_tracking': True,
        'github_actions': os.path.exists('$PROJECT_ROOT/.github/workflows/automation.yml'),
        'cron_jobs': os.path.exists('$PROJECT_ROOT/.github/automation/crontab')
    },
    'next_steps': [
        'Fill in environment variables in .env.automation.local',
        'Test automation scripts manually',
        'Configure notification channels (Slack, email)',
        'Set up GitHub repository secrets',
        'Review and adjust automation schedules'
    ]
}

with open('$report_file', 'w') as f:
    json.dump(report, f, indent=2)

print('Setup report generated successfully')
"
    
    log_success "Setup report generated: $report_file"
}

# Main setup function
main() {
    log_info "=== PQC IoT Retrofit Scanner Automation Setup ==="
    log_info "Environment: $ENVIRONMENT"
    log_info "Config file: $AUTOMATION_CONFIG"
    if [ "$DRY_RUN" = true ]; then
        log_info "Mode: DRY RUN (no changes will be made)"
    fi
    
    # Run setup steps
    check_environment
    install_dependencies
    validate_configuration
    setup_directories
    configure_scripts
    setup_environment_variables
    
    # Environment-specific setup
    if [ "$ENVIRONMENT" = "production" ] || [ "$ENVIRONMENT" = "staging" ]; then
        setup_github_actions
        setup_cron_jobs
    fi
    
    validate_setup
    generate_setup_report
    
    log_success "=== Automation setup completed successfully ==="
    
    # Print next steps
    echo ""
    log_info "Next steps:"
    echo "  1. Review and fill in environment variables in .env.automation.local"
    echo "  2. Test automation scripts manually:"
    echo "     python3 scripts/collect-metrics.py --help"
    echo "     python3 scripts/health-monitor.py --help"
    echo "  3. Configure notification channels (Slack webhook, email SMTP)"
    echo "  4. Set up GitHub repository secrets for automation workflow"
    echo "  5. Review automation schedules in .github/automation-config.json"
    echo ""
    
    if [ "$ENVIRONMENT" != "development" ]; then
        echo "  6. Install cron jobs (if not using GitHub Actions):"
        echo "     crontab .github/automation/crontab"
        echo ""
    fi
    
    log_info "Setup report available at: .github/automation/setup-report.json"
}

# Run main function
main "$@"