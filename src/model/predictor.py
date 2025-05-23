import mlflow
import mlflow.sklearn
import pandas as pd
from databricks.feature_store import FeatureStoreClient
import yaml

class ModelPredictor:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.fs = FeatureStoreClient()
        self.model = None
        self.load_model()
    
    def load_model(self, model_name=None, stage="Production"):
        """Load model from MLflow Model Registry"""
        model_name = model_name or self.config['model']['model_name']
        
        model_uri = f"models:/{model_name}/{stage}"
        self.model = mlflow.sklearn.load_model(model_uri)
        print(f"Loaded model: {model_name} (stage: {stage})")
    
    def predict_batch(self, input_data):
        """Make batch predictions"""
        if isinstance(input_data, pd.DataFrame):
            predictions = self.model.predict(input_data)
        else:
            # Assume it's a Spark DataFrame
            pandas_df = input_data.toPandas()
            predictions = self.model.predict(pandas_df)
        
        return predictions
    
    def predict_with_features(self, primary_keys):
        """Make predictions using features from Feature Store"""
        
        # Get features from Feature Store
        feature_df = self.fs.get_latest_features(
            primary_key_values=primary_keys,
            feature_table_name=self.config['feature_store']['table_name']
        )
        
        # Make predictions
        predictions = self.predict_batch(feature_df)
        
        return predictions
    
    def predict_single(self, features_dict):
        """Make single prediction"""
        df = pd.DataFrame([features_dict])
        prediction = self.model.predict(df)[0]
        return prediction
    
    def predict_with_confidence(self, input_data, n_estimators=None):
        """Make predictions with confidence intervals (for ensemble models)"""
        if hasattr(self.model, 'estimators_'):
            # For Random Forest or Gradient Boosting
            predictions = []
            for estimator in self.model.estimators_:
                pred = estimator.predict(input_data)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)
            
            # 95% confidence interval
            ci_lower = mean_pred - 1.96 * std_pred
            ci_upper = mean_pred + 1.96 * std_pred
            
            return {
                'prediction': mean_pred,
                'confidence_interval': {
                    'lower': ci_lower,
                    'upper': ci_upper
                },
                'std': std_pred
            }
        else:
            # For models without confidence estimation
            prediction = self.predict_batch(input_data)
            return {'prediction': prediction}