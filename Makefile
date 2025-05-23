.PHONY: help clean install install-dev test lint format build deploy-dev deploy-staging deploy-prod

PYTHON := python
PIP := pip
PROJECT_NAME := cris-e2e-ml-pipeline

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

clean: ## Clean build artifacts and cache
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

install: ## Install production dependencies
	$(PIP) install -r requirements.txt

install-dev: ## Install development dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"

test: ## Run unit tests
	pytest tests/unit/ -v --cov=src --cov-report=html --cov-report=term

test-integration: ## Run integration tests
	pytest tests/integration/ -v

test-all: ## Run all tests
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint: ## Run code linting
	flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503
	isort --check-only src/ tests/
	black --check src/ tests/

format: ## Format code
	isort src/ tests/
	black src/ tests/

build: clean ## Build wheel package
	$(PYTHON) -m build

# Databricks deployment commands
deploy-dev: build ## Deploy to development environment
	dbx deploy --environment=dev

deploy-staging: build ## Deploy to staging environment
	dbx deploy --environment=staging

deploy-prod: build ## Deploy to production environment
	dbx deploy --environment=prod

launch-dev: ## Launch development job
	dbx launch --environment=dev --job=$(PROJECT_NAME)-data-ingestion-dev --as-run-submit --trace

launch-staging: ## Launch staging job
	dbx launch --environment=staging --job=$(PROJECT_NAME)-data-ingestion-staging --as-run-submit --trace

launch-prod: ## Launch production job
	dbx launch --environment=prod --job=$(PROJECT_NAME)-data-ingestion-prod

# Setup commands
setup-git: ## Setup git hooks and configuration
	git config --local core.autocrlf false
	git config --local core.eol lf

setup-databricks: ## Setup Databricks CLI profiles
	@echo "Setting up Databricks CLI profiles..."
	@echo "Please run 'databricks configure --token' for each environment (dev, staging, prod)"

# Development workflow
dev-setup: install-dev setup-git ## Setup development environment
	@echo "Development environment setup complete!"

dev-test: lint test ## Run development tests (lint + unit tests)

dev-deploy: dev-test deploy-dev launch-dev ## Full development deployment workflow

# CI/CD simulation
ci: lint test build ## Simulate CI pipeline locally

cd-dev: ci deploy-dev ## Simulate CD pipeline for dev

cd-staging: ci deploy-staging ## Simulate CD pipeline for staging

cd-prod: ci deploy-prod ## Simulate CD pipeline for prod 