"""
Ensemble methods module for Gold Prospectivity Mapping.
Implements various ensemble techniques to combine model predictions.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from typing import Dict, List, Tuple, Any
import joblib

from utils.config import RANDOM_STATE, CV_FOLDS, SAVED_MODELS_DIR


class EnsembleMethods:
    """
    Class for implementing various ensemble methods.
    """
    
    def __init__(self):
        self.ensemble_models = {}
        self.ensemble_predictions = {}
        self.ensemble_scores = {}
        
    def create_voting_ensemble(self, models: Dict[str, Any], voting: str = 'soft') -> VotingClassifier:
        """
        Create a voting ensemble from multiple models.
        
        Parameters:
        -----------
        models : Dict[str, Any]
            Dictionary of trained models
        voting : str
            Type of voting ('hard' or 'soft')
            
        Returns:
        --------
        VotingClassifier
            Voting ensemble model
        """
        # Use all available models for ensemble
        estimators = [(name, model) for name, model in models.items()]
        
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting=voting,
            n_jobs=-1
        )
        
        return voting_clf
    
    def create_stacking_ensemble(self, models: Dict[str, Any], 
                                meta_learner: Any = None) -> StackingClassifier:
        """
        Create a stacking ensemble with meta-learner.
        
        Parameters:
        -----------
        models : Dict[str, Any]
            Dictionary of trained models
        meta_learner : Any
            Meta-learner model (default: LogisticRegression)
            
        Returns:
        --------
        StackingClassifier
            Stacking ensemble model
        """
        if meta_learner is None:
            meta_learner = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        
        # Use all available models for stacking
        base_estimators = [(name, model) for name, model in models.items()]
        
        stacking_clf = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=CV_FOLDS,  # Use cross-validation to train meta-learner
            n_jobs=-1
        )
        
        return stacking_clf
    
    def weighted_average_ensemble(self, predictions: Dict[str, np.ndarray], 
                                weights: Dict[str, float] = None) -> np.ndarray:
        """
        Create weighted average ensemble predictions.
        
        Parameters:
        -----------
        predictions : Dict[str, np.ndarray]
            Dictionary of model predictions (probabilities)
        weights : Dict[str, float]
            Weights for each model (default: equal weights)
            
        Returns:
        --------
        np.ndarray
            Weighted ensemble predictions
        """
        if weights is None:
            # Equal weights
            weights = {model: 1.0 / len(predictions) for model in predictions}
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {model: w / total_weight for model, w in weights.items()}
        
        # Calculate weighted average
        weighted_preds = np.zeros_like(list(predictions.values())[0])
        
        for model, preds in predictions.items():
            if model in weights:
                weighted_preds += weights[model] * preds
        
        return weighted_preds
    
    def rank_average_ensemble(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Create rank-based average ensemble predictions.
        
        Parameters:
        -----------
        predictions : Dict[str, np.ndarray]
            Dictionary of model predictions (probabilities)
            
        Returns:
        --------
        np.ndarray
            Rank-averaged ensemble predictions
        """
        # Convert predictions to ranks
        ranks = {}
        for model, preds in predictions.items():
            # Higher probability gets higher rank
            ranks[model] = preds.argsort().argsort() + 1
        
        # Average ranks
        avg_ranks = np.mean(list(ranks.values()), axis=0)
        
        # Convert back to probabilities (normalize)
        rank_probs = avg_ranks / avg_ranks.max()
        
        return rank_probs
    
    def blending_ensemble(self, train_predictions: Dict[str, np.ndarray], 
                         val_predictions: Dict[str, np.ndarray],
                         y_train: pd.Series) -> Tuple[Any, np.ndarray]:
        """
        Create blending ensemble using validation set predictions.
        
        Parameters:
        -----------
        train_predictions : Dict[str, np.ndarray]
            Training set predictions from each model
        val_predictions : Dict[str, np.ndarray]
            Validation set predictions from each model
        y_train : pd.Series
            Training target values
            
        Returns:
        --------
        Tuple[Any, np.ndarray]
            Blender model and blended predictions
        """
        # Create blend features from training predictions
        blend_features_train = np.column_stack(list(train_predictions.values()))
        blend_features_val = np.column_stack(list(val_predictions.values()))
        
        # Train blender (meta-learner)
        blender = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        blender.fit(blend_features_train, y_train)
        
        # Make blended predictions
        blended_predictions = blender.predict_proba(blend_features_val)[:, 1]
        
        return blender, blended_predictions
    
    def train_ensemble_models(self, models: Dict[str, Any], X_train: pd.DataFrame, 
                            y_train: pd.Series, X_val: pd.DataFrame, 
                            y_val: pd.Series) -> Dict[str, Any]:
        """
        Train all ensemble models.
        
        Parameters:
        -----------
        models : Dict[str, Any]
            Dictionary of base models
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
            Ensemble results
        """
        print("\n" + "="*60)
        print("ENSEMBLE MODEL TRAINING")
        print("="*60)
        
        results = {}
        
        # 1. Voting Ensemble (Soft)
        print("\n1. Training Soft Voting Ensemble...")
        print("   Combining predictions using probability averaging")
        voting_soft = self.create_voting_ensemble(models, voting='soft')
        voting_soft.fit(X_train, y_train)
        voting_soft_pred = voting_soft.predict_proba(X_val)[:, 1]
        self.ensemble_models['voting_soft'] = voting_soft
        results['voting_soft'] = {
            'predictions': voting_soft_pred,
            'score': self._evaluate_predictions(y_val, voting_soft_pred)
        }
        print(f"   ROC-AUC: {results['voting_soft']['score']['roc_auc']:.4f}")
        
        # 2. Voting Ensemble (Hard)
        print("\n2. Training Hard Voting Ensemble...")
        print("   Combining predictions using majority voting")
        voting_hard = self.create_voting_ensemble(models, voting='hard')
        voting_hard.fit(X_train, y_train)
        voting_hard_pred = voting_hard.predict(X_val)
        self.ensemble_models['voting_hard'] = voting_hard
        results['voting_hard'] = {
            'predictions': voting_hard_pred,
            'score': self._evaluate_predictions(y_val, voting_hard_pred, is_proba=False)
        }
        print(f"   Accuracy: {results['voting_hard']['score']['accuracy']:.4f}")
        
        # 3. Stacking Ensemble
        print("\n3. Training Stacking Ensemble...")
        print("   Base models: " + ", ".join(models.keys()))
        print("   Meta-learner: Logistic Regression")
        stacking = self.create_stacking_ensemble(models)
        stacking.fit(X_train, y_train)
        stacking_pred = stacking.predict_proba(X_val)[:, 1]
        self.ensemble_models['stacking'] = stacking
        results['stacking'] = {
            'predictions': stacking_pred,
            'score': self._evaluate_predictions(y_val, stacking_pred)
        }
        print(f"   ROC-AUC: {results['stacking']['score']['roc_auc']:.4f}")
        
        # 4. Weighted Average Ensemble
        print("\n4. Creating Weighted Average Ensemble...")
        # Get predictions from all models
        model_predictions = {}
        print("   Getting predictions from individual models...")
        for name, model in models.items():
            if hasattr(model, 'predict_proba'):
                model_predictions[name] = model.predict_proba(X_val)[:, 1]
                print(f"     {name}: predictions obtained")
        
        # Calculate weights based on individual model performance
        print("   Calculating optimal weights based on performance...")
        weights = self._calculate_optimal_weights(model_predictions, y_val)
        
        print("   Model weights:")
        for model, weight in weights.items():
            print(f"     {model}: {weight:.3f}")
        
        weighted_pred = self.weighted_average_ensemble(model_predictions, weights)
        results['weighted_average'] = {
            'predictions': weighted_pred,
            'score': self._evaluate_predictions(y_val, weighted_pred),
            'weights': weights
        }
        print(f"   ROC-AUC: {results['weighted_average']['score']['roc_auc']:.4f}")
        
        # 5. Rank Average Ensemble
        print("\n5. Creating Rank Average Ensemble...")
        print("   Converting predictions to ranks and averaging")
        rank_pred = self.rank_average_ensemble(model_predictions)
        results['rank_average'] = {
            'predictions': rank_pred,
            'score': self._evaluate_predictions(y_val, rank_pred)
        }
        print(f"   ROC-AUC: {results['rank_average']['score']['roc_auc']:.4f}")
        
        # Save ensemble models
        print("\n[SAVING ENSEMBLE MODELS]")
        for name, model in self.ensemble_models.items():
            self.save_ensemble_model(model, name)
            print(f"   Saved: ensemble_{name}_model.pkl")
        
        # Summary
        print("\n[ENSEMBLE TRAINING COMPLETE]")
        print(f"Total ensemble models trained: {len(results)}")
        best_ensemble = max(results.items(), key=lambda x: x[1]['score'].get('roc_auc', 0))
        print(f"Best ensemble model: {best_ensemble[0]} (ROC-AUC: {best_ensemble[1]['score']['roc_auc']:.4f})")
        
        return results
    
    def _calculate_optimal_weights(self, predictions: Dict[str, np.ndarray], 
                                 y_true: pd.Series) -> Dict[str, float]:
        """
        Calculate optimal weights for weighted ensemble based on individual performance.
        
        Parameters:
        -----------
        predictions : Dict[str, np.ndarray]
            Model predictions
        y_true : pd.Series
            True labels
            
        Returns:
        --------
        Dict[str, float]
            Optimal weights
        """
        from sklearn.metrics import roc_auc_score
        
        weights = {}
        for model, preds in predictions.items():
            # Use ROC-AUC as weight
            score = roc_auc_score(y_true, preds)
            weights[model] = score ** 2  # Square to give more weight to better models
        
        return weights
    
    def _evaluate_predictions(self, y_true: pd.Series, predictions: np.ndarray, 
                            is_proba: bool = True) -> Dict[str, float]:
        """
        Evaluate predictions and return metrics.
        
        Parameters:
        -----------
        y_true : pd.Series
            True labels
        predictions : np.ndarray
            Predictions
        is_proba : bool
            Whether predictions are probabilities
            
        Returns:
        --------
        Dict[str, float]
            Evaluation metrics
        """
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
        
        if is_proba:
            y_pred = (predictions >= 0.5).astype(int)
            roc_auc = roc_auc_score(y_true, predictions)
        else:
            y_pred = predictions
            roc_auc = 0.0  # Cannot calculate ROC-AUC for hard predictions
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc
        }
    
    def save_ensemble_model(self, model: Any, name: str):
        """
        Save ensemble model.
        
        Parameters:
        -----------
        model : Any
            Ensemble model
        name : str
            Model name
        """
        filepath = SAVED_MODELS_DIR / f'ensemble_{name}_model.pkl'
        joblib.dump(model, filepath)
        print(f"Ensemble model saved: {filepath}")
    
    def get_ensemble_summary(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Create summary of ensemble performance.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Ensemble results
            
        Returns:
        --------
        pd.DataFrame
            Summary dataframe
        """
        summary_data = []
        
        for method, result in results.items():
            summary_data.append({
                'Ensemble_Method': method,
                'ROC_AUC': result['score'].get('roc_auc', 0.0),
                'Accuracy': result['score'].get('accuracy', 0.0),
                'F1_Score': result['score'].get('f1_score', 0.0)
            })
        
        return pd.DataFrame(summary_data).sort_values('ROC_AUC', ascending=False)