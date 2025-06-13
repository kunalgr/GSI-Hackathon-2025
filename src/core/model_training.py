"""
Model training module for Gold Prospectivity Mapping.
Implements various ML algorithms with hyperparameter tuning.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import joblib
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from utils.config import RANDOM_STATE, CV_FOLDS, HYPERPARAMETERS, SAVED_MODELS_DIR


class ModelTrainer:
    """
    Class for training multiple ML models with hyperparameter tuning.
    """
    
    def __init__(self):
        self.models = {}
        self.best_models = {}
        self.cv_scores = {}
        self.predictions = {}
        
    def get_models(self) -> Dict[str, Any]:
        """
        Initialize all models to be trained.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary of model instances
        """
        models = {
            'logistic_regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            'extra_trees': ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            'gradient_boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
            'xgboost': xgb.XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss'),
        }
        
        return models
    
    def get_hyperparameters(self) -> Dict[str, Dict]:
        """
        Get hyperparameter grids for each model.
        
        Returns:
        --------
        Dict[str, Dict]
            Hyperparameter grids
        """
        hyperparams = {
            'logistic_regression': HYPERPARAMETERS.get('logistic_regression', {}),
            'random_forest': HYPERPARAMETERS.get('random_forest', {}),
            'extra_trees': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5, 7],
                'subsample': [0.7, 0.8, 0.9]
            },
            'xgboost': HYPERPARAMETERS.get('xgboost', {}),
        }
        
        return hyperparams
    
    def train_model_with_cv(self, model, param_grid: Dict, X_train: pd.DataFrame, 
                           y_train: pd.Series, model_name: str) -> Tuple[Any, Dict]:
        """
        Train a single model with cross-validation and hyperparameter tuning.
        
        Parameters:
        -----------
        model : sklearn estimator
            Model to train
        param_grid : Dict
            Hyperparameter grid
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        model_name : str
            Name of the model
            
        Returns:
        --------
        Tuple[Any, Dict]
            Best model and CV scores
        """
        print(f"\nTraining {model_name}...")
        
        # Use StratifiedKFold for cross-validation
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        use_cpu_cores = int(os.cpu_count() * .8)  # 80% of the cpu cores to be used.
        
        
        # Use GridSearchCV for smaller param grids, RandomizedSearchCV for larger ones
        n_combinations = np.prod([len(v) for v in param_grid.values()]) if param_grid else 1
        
        if n_combinations > 50 and param_grid:
            # Use RandomizedSearchCV for large parameter spaces
            search = RandomizedSearchCV(
                model, param_grid, n_iter=50, cv=cv, 
                scoring='roc_auc', n_jobs=use_cpu_cores, random_state=RANDOM_STATE
            )
        elif param_grid:
            # Use GridSearchCV for small parameter spaces
            search = GridSearchCV(
                model, param_grid, cv=cv, 
                scoring='roc_auc', n_jobs=use_cpu_cores
            )
        else:
            # No hyperparameters to tune
            search = model
            search.fit(X_train, y_train)
            return search, {'roc_auc': 0.0}  # Placeholder score
        
        # Fit the model
        search.fit(X_train, y_train)
        
        # Get best model and scores
        best_model = search.best_estimator_ if hasattr(search, 'best_estimator_') else search
        
        cv_scores = {
            'best_params': search.best_params_ if hasattr(search, 'best_params_') else {},
            'best_score': search.best_score_ if hasattr(search, 'best_score_') else 0.0,
            'cv_results': search.cv_results_ if hasattr(search, 'cv_results_') else {}
        }
        
        print(f"{model_name} - Best ROC-AUC: {cv_scores['best_score']:.4f}")
        
        return best_model, cv_scores
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                        X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """
        Train all models and evaluate on validation set.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        X_val : pd.DataFrame
            Validation features
        y_val : pd.Series
            Validation target
            
        Returns:
        --------
        Dict[str, Any]
            Training results
        """
        models = self.get_models()
        hyperparams = self.get_hyperparameters()
        
        results = {}
        
        for model_name, model in models.items():
            # Get hyperparameters for this model
            param_grid = hyperparams.get(model_name, {})
            
            # Train model with cross-validation
            best_model, cv_scores = self.train_model_with_cv(
                model, param_grid, X_train, y_train, model_name
            )
            
            # Store best model
            self.best_models[model_name] = best_model
            self.cv_scores[model_name] = cv_scores
            
            # Evaluate on validation set
            y_pred = best_model.predict(X_val)
            y_pred_proba = best_model.predict_proba(X_val)[:, 1] if hasattr(best_model, 'predict_proba') else y_pred
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_val, y_pred, y_pred_proba)
            
            # Store results
            results[model_name] = {
                'cv_score': cv_scores['best_score'],
                'best_params': cv_scores['best_params'],
                'validation_metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            # Save model
            self.save_model(best_model, model_name)
            
            print(f"{model_name} - Validation ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return results
    
    def calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                         y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Parameters:
        -----------
        y_true : pd.Series
            True labels
        y_pred : np.ndarray
            Predicted labels
        y_pred_proba : np.ndarray
            Predicted probabilities
            
        Returns:
        --------
        Dict[str, float]
            Evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba) if y_pred_proba is not None else 0.0
        }
        
        # Calculate specificity
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['sensitivity'] = metrics['recall']  # Same as recall
        
        # Additional metrics for geological interpretation
        metrics['true_positives'] = tp
        metrics['false_positives'] = fp
        metrics['true_negatives'] = tn
        metrics['false_negatives'] = fn
        
        return metrics
    
    def get_feature_importance(self, model_name: str, feature_names: List[str]) -> pd.DataFrame:
        """
        Get feature importance for tree-based models.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        feature_names : List[str]
            List of feature names
            
        Returns:
        --------
        pd.DataFrame
            Feature importance dataframe
        """
        model = self.best_models.get(model_name)
        
        if model is None:
            return pd.DataFrame()
        
        # Models with feature_importances_ attribute
        tree_based_models = ['random_forest', 'extra_trees', 'gradient_boosting', 
                           'xgboost', 'lightgbm', 'catboost']
        
        if model_name in tree_based_models and hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance
        
        # For logistic regression, use coefficients
        elif model_name == 'logistic_regression' and hasattr(model, 'coef_'):
            importance = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(model.coef_[0])
            }).sort_values('importance', ascending=False)
            
            return importance
        
        return pd.DataFrame()
    
    def save_model(self, model: Any, model_name: str):
        """
        Save trained model to disk.
        
        Parameters:
        -----------
        model : Any
            Trained model
        model_name : str
            Name of the model
        """
        filepath = SAVED_MODELS_DIR / f'{model_name}_model.pkl'
        joblib.dump(model, filepath)
        print(f"Model saved: {filepath}")
    
    def load_model(self, model_name: str) -> Any:
        """
        Load saved model from disk.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
            
        Returns:
        --------
        Any
            Loaded model
        """
        filepath = SAVED_MODELS_DIR / f'{model_name}_model.pkl'
        return joblib.load(filepath)
    
    def get_model_summary(self) -> pd.DataFrame:
        """
        Get summary of all trained models.
        
        Returns:
        --------
        pd.DataFrame
            Model summary dataframe
        """
        summary_data = []
        
        for model_name, scores in self.cv_scores.items():
            summary_data.append({
                'Model': model_name,
                'CV_ROC_AUC': scores.get('best_score', 0.0),
                'Best_Params': str(scores.get('best_params', {}))
            })
        
        return pd.DataFrame(summary_data).sort_values('CV_ROC_AUC', ascending=False)