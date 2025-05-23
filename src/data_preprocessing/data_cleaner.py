from pyspark.sql import DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *

class DataCleaner:
    def __init__(self):
        pass
    
    def remove_duplicates(self, df: DataFrame, subset_cols: list = None) -> DataFrame:
        """Remove duplicate rows"""
        if subset_cols:
            return df.dropDuplicates(subset_cols)
        return df.dropDuplicates()
    
    def handle_missing_values(self, df: DataFrame, strategy: dict) -> DataFrame:
        """Handle missing values based on strategy"""
        for column, method in strategy.items():
            if method == 'drop':
                df = df.filter(col(column).isNotNull())
            elif method == 'mean':
                mean_val = df.select(mean(col(column))).collect()[0][0]
                df = df.fillna({column: mean_val})
            elif method == 'median':
                median_val = df.select(expr(f"percentile_approx({column}, 0.5)")).collect()[0][0]
                df = df.fillna({column: median_val})
            elif method == 'mode':
                mode_val = df.groupBy(column).count().orderBy(desc("count")).first()[0]
                df = df.fillna({column: mode_val})
            elif isinstance(method, (int, float, str)):
                df = df.fillna({column: method})
        
        return df
    
    def remove_outliers(self, df: DataFrame, columns: list, method: str = 'iqr') -> DataFrame:
        """Remove outliers using specified method"""
        if method == 'iqr':
            for column in columns:
                Q1 = df.select(expr(f"percentile_approx({column}, 0.25)")).collect()[0][0]
                Q3 = df.select(expr(f"percentile_approx({column}, 0.75)")).collect()[0][0]
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df.filter((col(column) >= lower_bound) & (col(column) <= upper_bound))
        
        return df
    
    def validate_data_quality(self, df: DataFrame) -> dict:
        """Generate data quality report"""
        total_rows = df.count()
        
        quality_report = {
            'total_rows': total_rows,
            'columns': len(df.columns),
            'missing_values': {},
            'data_types': dict(df.dtypes)
        }
        
        for column in df.columns:
            null_count = df.filter(col(column).isNull()).count()
            quality_report['missing_values'][column] = {
                'count': null_count,
                'percentage': (null_count / total_rows) * 100
            }
        
        return quality_report