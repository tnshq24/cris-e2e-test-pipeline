import sys
import os
sys.path.append('/Workspace/Repos/your-repo/mlops-project/src')

from data_ingestion.data_loader import DataLoader
from data_preprocessing.data_cleaner import DataCleaner
from data_preprocessing.feature_engineer import FeatureEngineer
from feature_store.feature_manager import FeatureManager
from model.trainer import ModelTrainer
from model.evaluator import ModelEvaluator
import mlflow
from sklearn.model_selection import train_test_split
import pandas as pd

class TrainingPipeline:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.data_loader = DataLoader(config_path)
        self.data_cleaner = DataCleaner()
        self.feature_engineer = FeatureEngineer()
        self.feature_manager = FeatureManager(config_path)
        self.model_trainer = ModelTrainer(config_path)
        self.model_evaluator = ModelEvaluator(config_path)
    
    def run_pipeline(self):
        """Execute the complete training pipeline"""
        
        print("🚀 Starting MLOps Training Pipeline...")
        
        # 1. Data Ingestion
        print("📥 Loading data...")
        raw_data = self.data_loader.load_raw_data()
        print(f"✅ Loaded {raw_data.count()} rows of data")
        
        # 2. Data Cleaning
        print("🧹 Cleaning data...")
        
        # Define cleaning strategy
        missing_value_strategy = {
            'age': 'mean',
            'income': 'median',
            'category': 'mode'
        }
        
        cleaned_data = self.data_cleaner.remove_duplicates(raw_data)
        cleaned_data = self.data_cleaner.handle_missing_values(cleaned_data, missing_value_strategy)
        cleaned_data = self.data_cleaner.remove_outliers(cleaned_data, ['income', 'age'])
        
        # Generate quality report
        quality_report = self.data_cleaner.validate_data_quality(cleaned_data)
        print(f"✅ Data cleaning completed. Quality score: {quality_report}")
        
        # 3. Feature Engineering
        print("⚙️ Engineering features...")
        
        # Create time features if timestamp column exists
        if 'timestamp' in cleaned_data.columns:
            cleaned_data = self.feature_engineer.create_time_features(cleaned_data, 'timestamp')
        
        # Create polynomial features for numerical columns
        numerical_cols = [col for col, dtype in cleaned_data.dtypes if dtype in ['int', 'double', 'float']]
        cleaned_data = self.feature_engineer.create_polynomial_features(cleaned_data, numerical_cols[:3], degree=2)
        
        # Feature selection
        target_col = 'target'  # Replace with your target column
        selected_features = self.feature_engineer.select_features(cleaned_data, target_col)
        
        print(f"✅ Selected {len(selected_features)} features")
        
        # 4. Feature Store Operations
        print("🏪 Managing features in Feature Store...")
        
        # Prepare feature DataFrame
        feature_df = cleaned_data.select(selected_features + [target_col, 'id'])
        
        # Write to feature store
        try:
            self.feature_manager.write_features(feature_df)
            print("✅ Features written to Feature Store")
        except Exception as e:
            print(f"⚠️ Creating new feature table: {e}")
            table_name = self.feature_manager.create_feature_table(feature_df)
            print(f"✅ Created feature table: {table_name}")
        
        # 5. Model Training
        print("🤖 Training models...")
        
        # Convert to Pandas for sklearn
        feature_pandas = feature_df.toPandas()
        X = feature_pandas[selected_features]
        y = feature_pandas[target_col]
        
        # Train-validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Set MLflow experiment
        mlflow.set_experiment("/Shared/mlops-regression-experiment")
        
        # Train multiple models
        models = ['random_forest', 'gradient_boosting', 'linear_regression']
        best_model = None
        best_score = float('inf')
        best_model_type = None
        
        for model_type in models:
            print(f"🔄 Training {model_type}...")
            model, params, metrics = self.model_trainer.train_model(
                X_train, X_val, y_train, y_val, model_type
            )
            
            if metrics['mse'] < best_score:
                best_score = metrics['mse']
                best_model = model
                best_model_type = model_type
        
        print(f"✅ Best model: {best_model_type} with MSE: {best_score:.4f}")
        
        # 6. Model Evaluation
        print("📊 Evaluating model...")
        evaluation_results = self.model_evaluator.evaluate_model(
            best_model, X_val, y_val, feature_importance=True
        )
        
        # 7. Model Registration
        print("📝 Registering model...")
        model_name = "regression_model"
        
        with mlflow.start_run():
            mlflow.sklearn.log_model(
                best_model,
                "model",
                registered_model_name=model_name
            )
        
        print("🎉 Training pipeline completed successfully!")
        
        return {
            'best_model': best_model,
            'best_model_type': best_model_type,
            'metrics': evaluation_results,
            'features': selected_features
        }

# Execute pipeline
if __name__ == "__main__":
    config_path = "/Workspace/Repos/your-repo/mlops-project/config/pipeline_config.yaml"
    pipeline = TrainingPipeline(config_path)
    results = pipeline.run_pipeline()