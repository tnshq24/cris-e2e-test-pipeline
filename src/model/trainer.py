import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np
import yaml

class ModelTrainer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.models = {
            'random_forest': RandomForestRegressor(random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42),
            'linear_regression': LinearRegression()
        }
    
    def objective_function(self, params, X_train, X_val, y_train, y_val, model_type):
        """Objective function for hyperparameter optimization"""
        model = self.models[model_type]
        model.set_params(**params)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict and calculate loss
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        
        return {'loss': mse, 'status': STATUS_OK}
    
    def hyperparameter_tuning(self, X_train, X_val, y_train, y_val, model_type: str):
        """Perform hyperparameter tuning using Hyperopt"""
        
        # Define search space based on model type
        if model_type == 'random_forest':
            space = {
                'n_estimators': hp.choice('n_estimators', [100, 200, 300]),
                'max_depth': hp.choice('max_depth', [10, 20, 30, None]),
                'min_samples_split': hp.choice('min_samples_split', [2, 5, 10]),
                'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 4])
            }
        elif model_type == 'gradient_boosting':
            space = {
                'n_estimators': hp.choice('n_estimators', [100, 200, 300]),
                'learning_rate': hp.choice('learning_rate', [0.01, 0.1, 0.2]),
                'max_depth': hp.choice('max_depth', [3, 5, 7]),
                'subsample': hp.choice('subsample', [0.8, 0.9, 1.0])
            }
        else:  # linear_regression
            space = {
                'fit_intercept': hp.choice('fit_intercept', [True, False])
            }
        
        # Run optimization
        trials = Trials()
        best = fmin(
            fn=lambda params: self.objective_function(params, X_train, X_val, y_train, y_val, model_type),
            space=space,
            algo=tpe.suggest,
            max_evals=self.config['hyperparameters']['max_evals'],
            trials=trials
        )
        
        return best, trials
    
    def train_model(self, X_train, X_val, y_train, y_val, model_type: str = 'random_forest'):
        """Train model with hyperparameter tuning"""
        
        with mlflow.start_run(run_name=f"{model_type}_training"):
            # Log parameters
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("val_size", len(X_val))
            
            # Hyperparameter tuning
            best_params, trials = self.hyperparameter_tuning(X_train, X_val, y_train, y_val, model_type)
            
            # Train final model with best parameters
            model = self.models[model_type]
            model.set_params(**best_params)
            model.fit(X_train, y_train)
            
            # Validation predictions
            y_val_pred = model.predict(X_val)
            
            # Calculate metrics
            val_mse = mean_squared_error(y_val, y_val_pred)
            val_rmse = np.sqrt(val_mse)
            val_mae = mean_absolute_error(y_val, y_val_pred)
            val_r2 = r2_score(y_val, y_val_pred)
            
            # Log metrics
            mlflow.log_metric("val_mse", val_mse)
            mlflow.log_metric("val_rmse", val_rmse)
            mlflow.log_metric("val_mae", val_mae)
            mlflow.log_metric("val_r2", val_r2)
            
            # Log best parameters
            for param, value in best_params.items():
                mlflow.log_param(f"best_{param}", value)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            return model, best_params, {
                'mse': val_mse,
                'rmse': val_rmse,
                'mae': val_mae,
                'r2': val_r2
            }
    
    def cross_validate_model(self, X, y, model, cv_folds: int = 5):
        """Perform cross-validation"""
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_squared_error')
        return {
            'cv_mse_mean': -cv_scores.mean(),
            'cv_mse_std': cv_scores.std(),
            'cv_scores': cv_scores
        }