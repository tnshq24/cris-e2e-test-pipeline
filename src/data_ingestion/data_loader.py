import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from typing import Dict, Any
import yaml

class DataLoader:
    def __init__(self, config_path: str):
        self.spark = SparkSession.getActiveSession()
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
    
    def load_raw_data(self, data_path: str = None):
        """Load raw data from specified path or default location"""
        path = data_path or self.config['data']['source_path']
        
        # Support multiple file formats
        if path.endswith('.csv'):
            df = self.spark.read.option("header", "true").option("inferSchema", "true").csv(path)
        elif path.endswith('.parquet'):
            df = self.spark.read.parquet(path)
        elif path.endswith('.json'):
            df = self.spark.read.json(path)
        else:
            # Default to delta format for lakehouse
            df = self.spark.read.format("delta").load(path)
        
        return df
    
    def detect_new_data(self):
        """Check if new data is available for processing"""
        trigger_path = self.config['automation']['trigger_path']
        try:
            files = dbutils.fs.ls(trigger_path)
            return len(files) > 0
        except:
            return False
    
    def validate_data_schema(self, df, expected_columns: list):
        """Validate that data has expected schema"""
        actual_columns = set(df.columns)
        expected_columns = set(expected_columns)
        
        missing_columns = expected_columns - actual_columns
        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")
        
        return True