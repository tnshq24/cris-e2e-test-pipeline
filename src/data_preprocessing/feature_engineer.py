from pyspark.sql import DataFrame
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml import Pipeline

class FeatureEngineer:
    def __init__(self):
        pass
    
    def create_time_features(self, df: DataFrame, timestamp_col: str) -> DataFrame:
        """Create time-based features"""
        df = df.withColumn('year', year(col(timestamp_col)))
        df = df.withColumn('month', month(col(timestamp_col)))
        df = df.withColumn('day', dayofmonth(col(timestamp_col)))
        df = df.withColumn('hour', hour(col(timestamp_col)))
        df = df.withColumn('dayofweek', dayofweek(col(timestamp_col)))
        df = df.withColumn('quarter', quarter(col(timestamp_col)))
        return df
    
    def create_polynomial_features(self, df: DataFrame, columns: list, degree: int = 2) -> DataFrame:
        """Create polynomial features"""
        for column in columns:
            for d in range(2, degree + 1):
                df = df.withColumn(f"{column}_poly_{d}", pow(col(column), d))
        return df
    
    def create_interaction_features(self, df: DataFrame, column_pairs: list) -> DataFrame:
        """Create interaction features between column pairs"""
        for col1, col2 in column_pairs:
            df = df.withColumn(f"{col1}_{col2}_interaction", col(col1) * col(col2))
        return df
    
    def create_aggregation_features(self, df: DataFrame, group_cols: list, agg_cols: list) -> DataFrame:
        """Create aggregation features"""
        # Window functions for rolling statistics
        from pyspark.sql.window import Window
        
        for group_col in group_cols:
            window = Window.partitionBy(group_col)
            for agg_col in agg_cols:
                df = df.withColumn(f"{agg_col}_{group_col}_mean", 
                                 avg(col(agg_col)).over(window))
                df = df.withColumn(f"{agg_col}_{group_col}_std", 
                                 stddev(col(agg_col)).over(window))
                df = df.withColumn(f"{agg_col}_{group_col}_min", 
                                 min(col(agg_col)).over(window))
                df = df.withColumn(f"{agg_col}_{group_col}_max", 
                                 max(col(agg_col)).over(window))
        
        return df
    
    def select_features(self, df: DataFrame, target_col: str, method: str = 'correlation') -> list:
        """Feature selection based on specified method"""
        if method == 'correlation':
            # Calculate correlation with target
            feature_cols = [col for col in df.columns if col != target_col]
            correlations = []
            
            for feature in feature_cols:
                try:
                    corr = df.select(corr(col(feature), col(target_col))).collect()[0][0]
                    if corr is not None:
                        correlations.append((feature, abs(corr)))
                except:
                    continue
            
            # Sort by correlation and select top features
            correlations.sort(key=lambda x: x[1], reverse=True)
            selected_features = [feat[0] for feat in correlations[:20]]  # Top 20 features
            
            return selected_features
        
        return df.columns