# CRIS End-to-End ML Pipeline

A comprehensive machine learning pipeline for data ingestion, feature engineering, model training, and deployment on Databricks, following CI/CD best practices.

## 🚀 Features

- **Multi-environment deployment** (dev, staging, prod)
- **Automated CI/CD pipelines** with GitHub Actions
- **Code quality enforcement** (black, isort, flake8)
- **Comprehensive testing** (unit and integration tests)
- **Databricks integration** using `dbx` for deployment
- **MLflow integration** for experiment tracking
- **Azure cloud integration** for storage and compute

## 📁 Project Structure

```
├── .github/workflows/          # GitHub Actions CI/CD workflows
│   ├── ci.yml                 # Continuous Integration pipeline
│   └── cd.yml                 # Continuous Deployment pipeline
├── .dbx/                      # DBX configuration
│   └── project.json          # Environment and workspace settings
├── conf/                      # Deployment configurations
│   └── deployment.json       # Job definitions for all environments
├── config/                    # Application configurations
│   ├── model_config.yaml     # Model training configuration
│   └── pipeline_config.yaml  # Pipeline configuration
├── src/cris_e2e_ml_pipeline/ # Main source code
│   ├── data_ingestion/       # Data ingestion modules
│   ├── data_preprocessing/   # Data preprocessing modules
│   ├── feature_store/        # Feature engineering modules
│   ├── model/                # Model training and inference
│   └── utils/                # Utility functions
├── tests/                     # Test suite
│   ├── unit/                 # Unit tests
│   └── integration/          # Integration tests
├── notebooks/                 # Databricks notebooks
├── workflows/                 # Databricks workflows
├── pipelines/                 # Delta Live Tables pipelines
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
├── pyproject.toml            # Modern Python project configuration
├── Makefile                   # Development commands
└── README.md                  # This file
```

## 🛠 Setup Instructions

### Prerequisites

- Python 3.8+ (currently using 3.13.2)
- Git
- Access to a Databricks workspace
- Azure account (for cloud resources)

### 1. Clone and Setup

```bash
git clone https://github.com/tnshq24/cris-e2e-test-pipeline.git
cd cris-e2e-test-pipeline

# Setup development environment
make dev-setup

# Or manual setup:
pip install -r requirements.txt
pip install -e ".[dev]"
```

### 2. Configure Databricks

```bash
# Install and configure Databricks CLI for each environment
pip install databricks-cli

# Configure profiles for each environment
databricks configure --token --profile dev
databricks configure --token --profile staging  
databricks configure --token --profile prod
```

### 3. GitHub Secrets Configuration

Add the following secrets to your GitHub repository:

#### Development Environment
- `DATABRICKS_HOST_DEV`: Your Databricks workspace URL for dev
- `DATABRICKS_TOKEN_DEV`: Personal access token for dev environment

#### Staging Environment  
- `DATABRICKS_HOST_STAGING`: Your Databricks workspace URL for staging
- `DATABRICKS_TOKEN_STAGING`: Personal access token for staging environment

#### Production Environment
- `DATABRICKS_HOST_PROD`: Your Databricks workspace URL for prod  
- `DATABRICKS_TOKEN_PROD`: Personal access token for prod environment

## 🚀 Usage

### Development Workflow

```bash
# Format code
make format

# Run tests and linting
make dev-test

# Build package
make build

# Deploy to dev environment
make deploy-dev

# Launch dev job
make launch-dev

# Full development workflow
make dev-deploy
```

### Deployment

The CI/CD pipeline automatically:

1. **On PR to main**: Runs CI pipeline (tests, linting, build)
2. **On push to main**: Deploys to dev → staging
3. **On tag (v*)**: Deploys to production

#### Manual Deployment

```bash
# Deploy to specific environment
dbx deploy --environment=dev
dbx deploy --environment=staging  
dbx deploy --environment=prod

# Launch jobs
dbx launch --environment=dev --job=cris-e2e-ml-pipeline-data-ingestion-dev
```

### Available Make Commands

```bash
make help                 # Show all available commands
make clean               # Clean build artifacts
make install             # Install production dependencies
make install-dev         # Install development dependencies
make test                # Run unit tests
make test-integration    # Run integration tests
make test-all            # Run all tests
make lint                # Run code linting
make format              # Format code
make build               # Build wheel package
make deploy-dev          # Deploy to development
make deploy-staging      # Deploy to staging
make deploy-prod         # Deploy to production
```

## 🧪 Testing

### Running Tests

```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests only  
pytest tests/integration/ -v

# All tests with coverage
pytest tests/ -v --cov=src --cov-report=html
```

### Test Structure

- **Unit tests**: Test individual functions and classes in isolation
- **Integration tests**: Test end-to-end workflows and Databricks integration

## 🔄 CI/CD Pipeline

### Continuous Integration (CI)

Triggered on: Push to main/develop, Pull requests to main

**Jobs:**
1. **Code Quality**: black, isort, flake8 checks
2. **Unit Tests**: pytest with coverage reporting
3. **Build Package**: Create wheel distribution
4. **Integration Tests**: End-to-end testing (main branch only)

### Continuous Deployment (CD)

**Development**: Automatic on push to main
**Staging**: Automatic after successful dev deployment
**Production**: Manual approval or on version tags

## 📊 Monitoring and Logging

- **MLflow**: Experiment tracking and model registry
- **Databricks Jobs**: Job monitoring and alerting
- **GitHub Actions**: Build and deployment monitoring

## 🔧 Configuration

### Environment-Specific Settings

Each environment (dev/staging/prod) has:
- Separate Databricks workspaces
- Different cluster configurations  
- Environment-specific secrets
- Isolated artifact storage

### Customization

1. **Update cluster configs** in `conf/deployment.json`
2. **Modify job parameters** in deployment configuration
3. **Add new environments** by extending the configuration files
4. **Configure notifications** for production jobs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`make dev-test`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

- Create an issue in GitHub for bugs or feature requests
- Check the [Databricks documentation](https://docs.databricks.com/) for platform-specific questions
- Review the [dbx documentation](https://dbx.readthedocs.io/) for deployment tool usage

## 🔗 References

- [Databricks CI/CD Templates](https://github.com/databrickslabs/cicd-templates)
- [DBX Documentation](https://dbx.readthedocs.io/)
- [Databricks Jobs API](https://docs.databricks.com/api/workspace/jobs)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
