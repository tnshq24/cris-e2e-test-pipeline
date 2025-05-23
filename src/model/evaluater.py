import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import yaml

class ModelEvaluator:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
    
    def evaluate_model(self, model, X_test, y_test, feature_importance=True):
        """Comprehensive model evaluation"""
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        # Log metrics to MLflow
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(f"test_{metric_name}", metric_value)
        
        # Feature importance analysis
        if feature_importance and hasattr(model, 'feature_importances_'):
            feature_names = X_test.columns if hasattr(X_test, 'columns') else [f'feature_{i}' for i in range(X_test.shape[1])]
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            metrics['feature_importance'] = importance_df
        
        # Residual analysis
        residuals = y_test - y_pred
        metrics['residuals'] = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'skewness': pd.Series(residuals).skew(),
            'kurtosis': pd.Series(residuals).kurtosis()
        }
        
        return metrics
    
    def generate_evaluation_report(self, metrics, save_path=None):
        """Generate comprehensive evaluation report"""
        
        report = f"""
        # Model Evaluation Report
        
        ## Performance Metrics
        - **MSE**: {metrics['mse']:.4f}
        - **RMSE**: {metrics['rmse']:.4f}
        - **MAE**: {metrics['mae']:.4f}
        - **R²**: {metrics['r2']:.4f}
        - **MAPE**: {metrics['mape']:.2f}%
        
        ## Model Quality Assessment
        - **Goodness of Fit**: {'Excellent' if metrics['r2'] > 0.9 else 'Good' if metrics['r2'] > 0.8 else 'Fair' if metrics['r2'] > 0.6 else 'Poor'}
        - **Prediction Accuracy**: {'High' if metrics['mape'] < 10 else 'Medium' if metrics['mape'] < 20 else 'Low'}
        
        ## Residual Analysis
        - **Mean Residual**: {metrics['residuals']['mean']:.4f}
        - **Residual Std**: {metrics['residuals']['std']:.4f}
        - **Skewness**: {metrics['residuals']['skewness']:.4f}
        - **Kurtosis**: {metrics['residuals']['kurtosis']:.4f}
        """
        
        if 'feature_importance' in metrics:
            report += "\n## Top 10 Important Features\n"
            for idx, row in metrics['feature_importance'].head(10).iterrows():
                report += f"- **{row['feature']}**: {row['importance']:.4f}\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report

    def detect_model_drift(self, current_metrics, baseline_metrics):
        """Detect model performance drift"""
        
        drift_threshold = self.config['monitoring']['drift_threshold']
        
        drift_detected = False
        drift_report = {}
        
        for metric in ['mse', 'r2', 'mae']:
            if metric in current_metrics and metric in baseline_metrics:
                change = abs(current_metrics[metric] - baseline_metrics[metric])
                relative_change = change / baseline_metrics[metric]
                
                if relative_change > drift_threshold:
                    drift_detected = True
                    drift_report[metric] = {
                        'baseline': baseline_metrics[metric],
                        'current': current_metrics[metric],
                        'change': change,
                        'relative_change': relative_change
                    }
        
        return drift_detected, drift_report