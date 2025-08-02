#!/bin/bash
# Build script for PQC IoT Retrofit Scanner
# Handles package building, SBOM generation, and security scanning

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
DIST_DIR="${PROJECT_ROOT}/dist"
REPORTS_DIR="${PROJECT_ROOT}/reports"

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
PQC IoT Retrofit Scanner Build Script

Usage: $0 [OPTIONS] [TARGETS]

OPTIONS:
    -h, --help          Show this help message
    -v, --verbose       Enable verbose output
    -c, --clean         Clean build artifacts before building
    -s, --security      Run security scans
    --sbom              Generate SBOM (Software Bill of Materials)
    --docker            Build Docker images
    --sign              Sign artifacts (requires signing key)
    --version VERSION   Override version number

TARGETS:
    all                 Build everything (default)
    package             Build Python package
    docker              Build Docker images
    sbom                Generate SBOM
    security            Run security scans
    clean               Clean build artifacts

EXAMPLES:
    $0                  # Build package with default settings
    $0 --clean all      # Clean and build everything
    $0 --docker --sbom  # Build Docker images and generate SBOM
    $0 security         # Run only security scans

EOF
}

# Parse command line arguments
VERBOSE=false
CLEAN=false
SECURITY=false
GENERATE_SBOM=false
BUILD_DOCKER=false
SIGN_ARTIFACTS=false
VERSION=""
TARGETS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -s|--security)
            SECURITY=true
            shift
            ;;
        --sbom)
            GENERATE_SBOM=true
            shift
            ;;
        --docker)
            BUILD_DOCKER=true
            shift
            ;;
        --sign)
            SIGN_ARTIFACTS=true
            shift
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        -*)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            TARGETS+=("$1")
            shift
            ;;
    esac
done

# Set default target if none specified
if [[ ${#TARGETS[@]} -eq 0 ]]; then
    TARGETS=("package")
fi

# Set verbose mode if requested
if [[ "$VERBOSE" == true ]]; then
    set -x
fi

# Get version information
get_version() {
    if [[ -n "$VERSION" ]]; then
        echo "$VERSION"
    else
        # Extract version from pyproject.toml
        python -c "
import tomllib
with open('pyproject.toml', 'rb') as f:
    data = tomllib.load(f)
print(data['project']['version'])
" 2>/dev/null || echo "0.1.0"
    fi
}

# Get git information
get_git_info() {
    local git_commit=""
    local git_branch=""
    local git_dirty=""
    
    if git rev-parse --git-dir > /dev/null 2>&1; then
        git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
        git_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
        if ! git diff --quiet 2>/dev/null; then
            git_dirty="-dirty"
        fi
    else
        git_commit="unknown"
        git_branch="unknown"
    fi
    
    echo "${git_commit}${git_dirty}"
}

# Setup build environment
setup_build_env() {
    log_info "Setting up build environment..."
    
    # Create directories
    mkdir -p "$BUILD_DIR" "$DIST_DIR" "$REPORTS_DIR"
    
    # Set environment variables
    export BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
    export VCS_REF=$(get_git_info)
    export VERSION=$(get_version)
    
    log_info "Build environment:"
    log_info "  Version: $VERSION"
    log_info "  VCS Ref: $VCS_REF"
    log_info "  Build Date: $BUILD_DATE"
}

# Clean build artifacts
clean_build() {
    log_info "Cleaning build artifacts..."
    
    rm -rf "$BUILD_DIR" "$DIST_DIR" "$REPORTS_DIR"
    rm -rf .pytest_cache .mypy_cache .ruff_cache .tox .nox
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true
    
    log_success "Build artifacts cleaned"
}

# Build Python package
build_package() {
    log_info "Building Python package..."
    
    cd "$PROJECT_ROOT"
    
    # Upgrade build tools
    python -m pip install --upgrade pip setuptools wheel build
    
    # Build package
    python -m build --outdir "$DIST_DIR"
    
    # Verify package
    python -m pip install twine
    python -m twine check "$DIST_DIR"/*
    
    # List built artifacts
    log_info "Built packages:"
    ls -la "$DIST_DIR"
    
    log_success "Python package built successfully"
}

# Generate SBOM (Software Bill of Materials)
generate_sbom() {
    log_info "Generating SBOM (Software Bill of Materials)..."
    
    # Install SBOM tools
    if ! command -v syft &> /dev/null; then
        log_warning "syft not found, installing..."
        # Install syft for SBOM generation
        curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin
    fi
    
    # Generate SBOM for Python package
    if [[ -f "$DIST_DIR"/*.whl ]]; then
        local wheel_file=$(ls "$DIST_DIR"/*.whl | head -1)
        syft "$wheel_file" -o spdx-json="$REPORTS_DIR/sbom-package.spdx.json"
        syft "$wheel_file" -o cyclonedx-json="$REPORTS_DIR/sbom-package.cyclonedx.json"
        log_info "SBOM generated for Python package"
    fi
    
    # Generate SBOM for source code
    syft dir:"$PROJECT_ROOT" -o spdx-json="$REPORTS_DIR/sbom-source.spdx.json"
    syft dir:"$PROJECT_ROOT" -o cyclonedx-json="$REPORTS_DIR/sbom-source.cyclonedx.json"
    
    # Generate dependency tree
    python -m pip install pipdeptree
    pipdeptree --json > "$REPORTS_DIR/dependency-tree.json"
    pipdeptree --graph-output png > "$REPORTS_DIR/dependency-graph.png" 2>/dev/null || true
    
    log_success "SBOM generated successfully"
}

# Build Docker images
build_docker() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    local image_name="pqc-iot-retrofit-scanner"
    local registry="${DOCKER_REGISTRY:-terragon}"
    
    # Build production image
    docker build \
        --target production \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg VCS_REF="$VCS_REF" \
        --build-arg VERSION="$VERSION" \
        -t "${registry}/${image_name}:${VERSION}" \
        -t "${registry}/${image_name}:latest" \
        .
    
    # Build development image
    docker build \
        --target development \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg VCS_REF="$VCS_REF" \
        --build-arg VERSION="$VERSION" \
        -t "${registry}/${image_name}:${VERSION}-dev" \
        -t "${registry}/${image_name}:dev" \
        .
    
    # Build testing image
    docker build \
        --target testing \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg VCS_REF="$VCS_REF" \
        --build-arg VERSION="$VERSION" \
        -t "${registry}/${image_name}:${VERSION}-test" \
        -t "${registry}/${image_name}:test" \
        .
    
    # Generate Docker SBOM if requested
    if [[ "$GENERATE_SBOM" == true ]]; then
        log_info "Generating Docker SBOM..."
        syft "${registry}/${image_name}:${VERSION}" -o spdx-json="$REPORTS_DIR/sbom-docker.spdx.json"
        syft "${registry}/${image_name}:${VERSION}" -o cyclonedx-json="$REPORTS_DIR/sbom-docker.cyclonedx.json"
    fi
    
    # List built images
    log_info "Built Docker images:"
    docker images "${registry}/${image_name}"
    
    log_success "Docker images built successfully"
}

# Run security scans
run_security_scans() {
    log_info "Running security scans..."
    
    mkdir -p "$REPORTS_DIR/security"
    
    # Python security scan with bandit
    log_info "Running Bandit security scan..."
    python -m pip install bandit[toml]
    bandit -r src/ -f json -o "$REPORTS_DIR/security/bandit.json" || true
    bandit -r src/ -f txt -o "$REPORTS_DIR/security/bandit.txt" || true
    
    # Dependency vulnerability scan with safety
    log_info "Running Safety dependency scan..."
    python -m pip install safety
    safety check --json --output "$REPORTS_DIR/security/safety.json" || true
    safety check --output "$REPORTS_DIR/security/safety.txt" || true
    
    # License compliance check
    log_info "Running license compliance check..."
    python -m pip install pip-licenses
    pip-licenses --format=json --output-file="$REPORTS_DIR/security/licenses.json"
    pip-licenses --format=csv --output-file="$REPORTS_DIR/security/licenses.csv"
    
    # Secret scanning
    if command -v detect-secrets &> /dev/null; then
        log_info "Running secret detection..."
        detect-secrets scan --all-files --baseline "$REPORTS_DIR/security/secrets-baseline.json" || true
    fi
    
    # Docker security scan (if Docker images were built)
    if [[ "$BUILD_DOCKER" == true ]] && command -v grype &> /dev/null; then
        log_info "Running Docker security scan..."
        local image_name="terragon/pqc-iot-retrofit-scanner:${VERSION}"
        grype "$image_name" -o json > "$REPORTS_DIR/security/docker-vulns.json" || true
        grype "$image_name" -o table > "$REPORTS_DIR/security/docker-vulns.txt" || true
    fi
    
    log_success "Security scans completed"
}

# Sign artifacts
sign_artifacts() {
    log_info "Signing artifacts..."
    
    if [[ -z "${SIGNING_KEY:-}" ]]; then
        log_warning "No signing key specified, skipping artifact signing"
        return 0
    fi
    
    # Sign Python packages
    if command -v gpg &> /dev/null && ls "$DIST_DIR"/*.whl &> /dev/null; then
        for file in "$DIST_DIR"/*; do
            gpg --armor --detach-sign "$file"
            log_info "Signed: $(basename "$file")"
        done
    fi
    
    # Sign Docker images (if cosign is available)
    if command -v cosign &> /dev/null && [[ "$BUILD_DOCKER" == true ]]; then
        local image_name="terragon/pqc-iot-retrofit-scanner:${VERSION}"
        cosign sign "$image_name" || log_warning "Failed to sign Docker image"
    fi
    
    log_success "Artifacts signed"
}

# Generate build report
generate_build_report() {
    log_info "Generating build report..."
    
    local report_file="$REPORTS_DIR/build-report.json"
    
    cat > "$report_file" << EOF
{
  "build": {
    "version": "$VERSION",
    "vcs_ref": "$VCS_REF",
    "build_date": "$BUILD_DATE",
    "build_host": "$(hostname)",
    "build_user": "$(whoami)",
    "python_version": "$(python --version)",
    "platform": "$(uname -a)"
  },
  "artifacts": {
    "python_packages": $(ls "$DIST_DIR"/*.whl "$DIST_DIR"/*.tar.gz 2>/dev/null | jq -R . | jq -s . || echo "[]"),
    "docker_images": $(docker images --format "{{.Repository}}:{{.Tag}}" terragon/pqc-iot-retrofit-scanner 2>/dev/null | jq -R . | jq -s . || echo "[]"),
    "reports": $(find "$REPORTS_DIR" -name "*.json" -o -name "*.txt" -o -name "*.csv" | jq -R . | jq -s . || echo "[]")
  },
  "checksums": {}
}
EOF
    
    # Generate checksums for artifacts
    if ls "$DIST_DIR"/* &> /dev/null; then
        cd "$DIST_DIR"
        sha256sum * > "$REPORTS_DIR/checksums.txt"
        cd - > /dev/null
    fi
    
    log_success "Build report generated: $report_file"
}

# Main build orchestration
main() {
    log_info "Starting PQC IoT Retrofit Scanner build process..."
    
    setup_build_env
    
    # Handle clean target
    if [[ "$CLEAN" == true ]] || [[ " ${TARGETS[*]} " =~ " clean " ]]; then
        clean_build
    fi
    
    # Process targets
    for target in "${TARGETS[@]}"; do
        case "$target" in
            all)
                build_package
                if [[ "$BUILD_DOCKER" == true ]]; then
                    build_docker
                fi
                if [[ "$GENERATE_SBOM" == true ]]; then
                    generate_sbom
                fi
                if [[ "$SECURITY" == true ]]; then
                    run_security_scans
                fi
                if [[ "$SIGN_ARTIFACTS" == true ]]; then
                    sign_artifacts
                fi
                ;;
            package)
                build_package
                ;;
            docker)
                BUILD_DOCKER=true
                build_docker
                ;;
            sbom)
                GENERATE_SBOM=true
                generate_sbom
                ;;
            security)
                SECURITY=true
                run_security_scans
                ;;
            clean)
                # Already handled above
                ;;
            *)
                log_error "Unknown target: $target"
                exit 1
                ;;
        esac
    done
    
    # Always generate build report
    generate_build_report
    
    log_success "Build process completed successfully!"
    log_info "Build artifacts available in: $DIST_DIR"
    log_info "Build reports available in: $REPORTS_DIR"
}

# Trap errors and cleanup
trap 'log_error "Build failed on line $LINENO"' ERR

# Run main function
main "$@"