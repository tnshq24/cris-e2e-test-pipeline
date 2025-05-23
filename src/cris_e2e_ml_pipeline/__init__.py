"""
CRIS End-to-End ML Pipeline

A comprehensive machine learning pipeline for data ingestion, feature engineering, 
model training, and deployment on Databricks.
"""

__version__ = "0.1.0"
__author__ = "Tanishq Singh"

from .data_ingestion import *
from .data_preprocessing import *
from .feature_store import *
from .model import *
from .utils import * 