from setuptools import setup, find_packages

import cris_e2e_ml_pipeline

setup(
    name="cris_e2e_ml_pipeline",
    version=cris_e2e_ml_pipeline.__version__,
    author="Tanishq Singh",
    description="CRIS End-to-End ML Pipeline",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pyspark>=3.5.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0,<2.0.0",
        "scikit-learn>=1.0.0",
        "mlflow>=2.0.0",
        "databricks-feature-store>=0.4.0",
        "azure-storage-blob>=12.0.0",
        "pyyaml>=6.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.0.0",
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
) 