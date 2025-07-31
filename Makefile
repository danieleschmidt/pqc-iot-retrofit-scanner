# Makefile for PQC IoT Retrofit Scanner
# Provides common development tasks and automation

.PHONY: help install install-dev test test-unit test-integration test-e2e
.PHONY: lint format security clean build docs docker run-dev
.PHONY: benchmark profile analyze release check-deps update-deps

# Default target
.DEFAULT_GOAL := help

# Configuration
PYTHON := python3
PIP := pip
PROJECT_NAME := pqc-iot-retrofit-scanner
DOCKER_IMAGE := pqc-iot-retrofit-scanner
DOCKER_TAG := latest

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)PQC IoT Retrofit Scanner - Development Makefile$(NC)"
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""

# Installation targets
install: ## Install package in production mode
	$(PIP) install -e .

install-dev: ## Install package with development dependencies
	$(PIP) install -e ".[dev,analysis]"
	pre-commit install

install-deps: ## Install/update dependencies from lock files
	$(PIP) install -r requirements-dev.lock

# Testing targets
test: ## Run all tests
	pytest -v

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	pytest tests/integration/ -v

test-e2e: ## Run end-to-end tests only
	pytest tests/e2e/ -v

test-security: ## Run security-focused tests
	pytest -m security -v

test-coverage: ## Run tests with coverage report
	pytest --cov=src --cov-report=html --cov-report=term-missing

test-parallel: ## Run tests in parallel
	pytest -n auto

benchmark: ## Run performance benchmarks
	pytest tests/benchmarks/ --benchmark-only --benchmark-sort=mean

# Code quality targets
lint: ## Run all linting checks
	@echo "$(BLUE)Running code quality checks...$(NC)"
	black --check src/ tests/
	ruff check src/ tests/
	mypy src/

format: ## Format code
	@echo "$(BLUE)Formatting code...$(NC)"
	black src/ tests/
	ruff check --fix src/ tests/
	isort src/ tests/

security: ## Run security scans
	@echo "$(BLUE)Running security scans...$(NC)"
	bandit -r src/ -f json -o reports/bandit.json || true
	safety check --json --output reports/safety.json || true
	detect-secrets scan --baseline .secrets.baseline

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

# Build and package targets
clean: ## Clean build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

build: clean ## Build package distributions
	@echo "$(BLUE)Building package...$(NC)"
	$(PYTHON) -m build

release-check: ## Check if package is ready for release
	@echo "$(BLUE)Checking release readiness...$(NC)"
	$(PYTHON) -m twine check dist/*
	
# Documentation targets
docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	cd docs && make html

docs-serve: ## Serve documentation locally
	cd docs && make livehtml

docs-clean: ## Clean documentation build
	cd docs && make clean

# Docker targets
docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-build-dev: ## Build development Docker image
	docker build --target development -t $(DOCKER_IMAGE):dev .

docker-build-test: ## Build testing Docker image
	docker build --target testing -t $(DOCKER_IMAGE):test .

docker-run: ## Run Docker container
	docker run -it --rm $(DOCKER_IMAGE):$(DOCKER_TAG)

docker-run-dev: ## Run development Docker container
	docker run -it --rm -v $(PWD):/app $(DOCKER_IMAGE):dev

docker-compose-up: ## Start all services with docker-compose
	docker-compose up -d

docker-compose-down: ## Stop all services
	docker-compose down

docker-compose-logs: ## View logs from all services
	docker-compose logs -f

# Development environment targets
dev-setup: install-dev ## Setup development environment
	@echo "$(GREEN)Development environment setup complete!$(NC)"
	@echo "Run '$(YELLOW)make run-dev$(NC)' to start development server"

run-dev: ## Run development server
	$(PYTHON) -c "from pqc_iot_retrofit.cli import main; main()" --help

shell: ## Open Python shell with project context
	$(PYTHON) -c "import pqc_iot_retrofit; print('PQC IoT Retrofit Scanner shell ready'); import IPython; IPython.start_ipython()"

# Analysis and profiling targets
profile: ## Profile the application
	$(PYTHON) -m cProfile -o profile.stats -m pqc_iot_retrofit.cli --version
	$(PYTHON) -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

analyze: ## Run static analysis
	@echo "$(BLUE)Running static analysis...$(NC)"
	vulture src/ --min-confidence 80
	radon cc src/ -a
	radon mi src/ -s

# Dependency management targets
check-deps: ## Check for dependency issues
	@echo "$(BLUE)Checking dependencies...$(NC)"
	pip check
	safety check
	pip-audit

update-deps: ## Update dependencies (use with caution)
	@echo "$(YELLOW)Updating dependencies...$(NC)"
	pip-compile --upgrade requirements-dev.in

freeze-deps: ## Generate requirements lock file
	pip-compile requirements-dev.in

# Utility targets
version: ## Show current version
	@$(PYTHON) -c "import pqc_iot_retrofit; print(pqc_iot_retrofit.__version__)"

info: ## Show project information
	@echo "$(BLUE)Project Information:$(NC)"
	@echo "Name: $(PROJECT_NAME)"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Pip: $(shell $(PIP) --version)"
	@echo "Git: $(shell git --version)"
	@echo "Docker: $(shell docker --version 2>/dev/null || echo 'Not installed')"

env: ## Show environment variables
	@echo "$(BLUE)Environment Variables:$(NC)"
	@env | grep -E "(PQC_|PYTHON|PATH)" | sort

# CI/CD targets
ci-install: ## Install dependencies for CI
	$(PIP) install -e ".[dev,analysis]"

ci-test: ## Run tests in CI mode
	pytest --junitxml=test-results/junit.xml --cov=src --cov-report=xml --cov-report=term

ci-security: ## Run security checks for CI
	bandit -r src/ -f json -o security-reports/bandit.json
	safety check --json --output security-reports/safety.json
	detect-secrets scan --baseline .secrets.baseline

ci-build: ## Build for CI
	$(PYTHON) -m build
	$(PYTHON) -m twine check dist/*

# Cleanup targets
reset: clean ## Reset environment (clean + remove virtual env)
	@echo "$(RED)Resetting development environment...$(NC)"
	deactivate 2>/dev/null || true
	rm -rf venv/ .venv/

# Help for specific tasks
test-help: ## Show testing help
	@echo "$(BLUE)Testing Commands:$(NC)"
	@echo "  make test              - Run all tests"
	@echo "  make test-unit         - Run unit tests only"
	@echo "  make test-integration  - Run integration tests"
	@echo "  make test-coverage     - Run tests with coverage"
	@echo "  make benchmark         - Run performance benchmarks"

docker-help: ## Show Docker help
	@echo "$(BLUE)Docker Commands:$(NC)"
	@echo "  make docker-build      - Build production image"
	@echo "  make docker-build-dev  - Build development image"
	@echo "  make docker-run-dev    - Run development container"
	@echo "  make docker-compose-up - Start all services"

# Validation target for CI
validate: lint test security ## Run all validation checks (CI-friendly)
	@echo "$(GREEN)All validation checks passed!$(NC)"