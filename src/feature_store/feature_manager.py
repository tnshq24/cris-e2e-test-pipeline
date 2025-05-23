from databricks.feature_store import FeatureStoreClient
from pyspark.sql import DataFrame
import yaml

class FeatureManager:
    def __init__(self, config_path: str):
        self.fs = FeatureStoreClient()
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
    
    def create_feature_table(self, df: DataFrame, table_name: str = None):
        """Create feature table in Databricks Feature Store"""
        table_name = table_name or self.config['feature_store']['table_name']
        primary_keys = self.config['feature_store']['primary_keys']
        
        # Create feature table
        self.fs.create_table(
            name=table_name,
            primary_keys=primary_keys,
            df=df,
            description="Regression model features"
        )
        
        return table_name
    
    def write_features(self, df: DataFrame, table_name: str = None, mode: str = "merge"):
        """Write features to feature store"""
        table_name = table_name or self.config['feature_store']['table_name']
        
        self.fs.write_table(
            name=table_name,
            df=df,
            mode=mode
        )
    
    def read_features(self, table_name: str = None, feature_names: list = None):
        """Read features from feature store"""
        table_name = table_name or self.config['feature_store']['table_name']
        
        if feature_names:
            return self.fs.read_table(name=table_name, feature_names=feature_names)
        else:
            return self.fs.read_table(name=table_name)
    
    def get_latest_features(self, primary_key_values: list, table_name: str = None):
        """Get latest features for given primary keys"""
        table_name = table_name or self.config['feature_store']['table_name']
        
        return self.fs.get_latest_features(
            primary_key_values=primary_key_values,
            feature_table_name=table_name
        )