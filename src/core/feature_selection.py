"""
Feature selection module for Gold Prospectivity Mapping.
Implements various feature selection techniques.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest, chi2, f_classif, mutual_info_classif,
    RFE, SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
import xgboost as xgb
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns

from utils.config import RANDOM_STATE, VARIANCE_THRESHOLD, CORRELATION_THRESHOLD, FIGURES_DIR


class FeatureSelector:
    """
    Class for comprehensive feature selection.
    """
    
    def __init__(self):
        self.selected_features = {}
        self.feature_scores = {}
        
    def remove_low_variance_features(self, X: pd.DataFrame, threshold: float = VARIANCE_THRESHOLD) -> List[str]:
        """
        Remove features with low variance.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature dataframe
        threshold : float
            Variance threshold
            
        Returns:
        --------
        List[str]
            List of selected features
        """
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
        
        selected_features = X.columns[selector.get_support()].tolist()
        self.selected_features['variance'] = selected_features
        
        print(f"Variance threshold: {len(selected_features)} features selected from {X.shape[1]}")
        
        return selected_features
    
    def remove_correlated_features(self, X: pd.DataFrame, threshold: float = CORRELATION_THRESHOLD) -> List[str]:
        """
        Remove highly correlated features.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature dataframe
        threshold : float
            Correlation threshold
            
        Returns:
        --------
        List[str]
            List of selected features
        """
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        selected_features = [col for col in X.columns if col not in to_drop]
        self.selected_features['correlation'] = selected_features
        
        print(f"Correlation threshold: {len(selected_features)} features selected from {X.shape[1]}")
        
        return selected_features
    
    def univariate_selection(self, X: pd.DataFrame, y: pd.Series, k: int = 30) -> List[str]:
        """
        Select features using univariate statistical tests.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature dataframe
        y : pd.Series
            Target variable
        k : int
            Number of features to select
            
        Returns:
        --------
        List[str]
            List of selected features
        """
        # Ensure k doesn't exceed number of features
        k = min(k, X.shape[1])
        
        # Use mutual information for geological data (handles non-linear relationships)
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        selector.fit(X, y)
        
        # Get scores
        scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        self.feature_scores['mutual_info'] = scores
        
        selected_features = X.columns[selector.get_support()].tolist()
        self.selected_features['univariate'] = selected_features
        
        print(f"Univariate selection: {len(selected_features)} features selected")
        
        return selected_features
    
    def rfe_selection(self, X: pd.DataFrame, y: pd.Series, n_features: int = 30) -> List[str]:
        """
        Recursive Feature Elimination using Random Forest.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature dataframe
        y : pd.Series
            Target variable
        n_features : int
            Number of features to select
            
        Returns:
        --------
        List[str]
            List of selected features
        """
        # Ensure n_features doesn't exceed number of features
        n_features = min(n_features, X.shape[1])
        
        # Use Random Forest as base estimator (good for geological data)
        estimator = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        
        rfe = RFE(estimator, n_features_to_select=n_features, step=1)
        rfe.fit(X, y)
        
        selected_features = X.columns[rfe.support_].tolist()
        self.selected_features['rfe'] = selected_features
        
        print(f"RFE selection: {len(selected_features)} features selected")
        
        return selected_features
    
    def tree_based_selection(self, X: pd.DataFrame, y: pd.Series, threshold: str = 'median') -> List[str]:
        """
        Select features using tree-based feature importance.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature dataframe
        y : pd.Series
            Target variable
        threshold : str
            Threshold for SelectFromModel
            
        Returns:
        --------
        List[str]
            List of selected features
        """
        # Use XGBoost for feature importance
        clf = xgb.XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, use_label_encoder=False)
        clf.fit(X, y)
        
        # Get feature importance
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_scores['xgb_importance'] = importance
        
        # Select features
        selector = SelectFromModel(clf, threshold=threshold, prefit=True)
        selected_features = X.columns[selector.get_support()].tolist()
        self.selected_features['tree_based'] = selected_features
        
        print(f"Tree-based selection: {len(selected_features)} features selected")
        
        return selected_features
    
    def lasso_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Select features using L1 regularization (Lasso).
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature dataframe
        y : pd.Series
            Target variable
            
        Returns:
        --------
        List[str]
            List of selected features
        """
        # Use LassoCV to find optimal alpha
        lasso = LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=10000)
        lasso.fit(X, y)
        
        # Get selected features
        selected_features = X.columns[lasso.coef_ != 0].tolist()
        self.selected_features['lasso'] = selected_features
        
        print(f"Lasso selection: {len(selected_features)} features selected")
        
        return selected_features
    
    def ensemble_feature_selection(self, X: pd.DataFrame, y: pd.Series, 
                                 min_votes: int = 3) -> List[str]:
        """
        Ensemble feature selection combining multiple methods.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature dataframe
        y : pd.Series
            Target variable
        min_votes : int
            Minimum number of methods that must select a feature
            
        Returns:
        --------
        List[str]
            List of selected features
        """
        # Apply all selection methods
        self.remove_low_variance_features(X)
        self.remove_correlated_features(X)
        self.univariate_selection(X, y)
        self.rfe_selection(X, y)
        self.tree_based_selection(X, y)
        self.lasso_selection(X, y)
        
        # Count votes for each feature
        feature_votes = {}
        for method, features in self.selected_features.items():
            for feature in features:
                feature_votes[feature] = feature_votes.get(feature, 0) + 1
        
        # Select features with minimum votes
        ensemble_features = [feature for feature, votes in feature_votes.items() 
                           if votes >= min_votes]
        
        print(f"\nEnsemble selection: {len(ensemble_features)} features selected with at least {min_votes} votes")
        
        # Store voting results
        self.feature_scores['ensemble_votes'] = pd.DataFrame(
            list(feature_votes.items()), 
            columns=['feature', 'votes']
        ).sort_values('votes', ascending=False)
        
        return ensemble_features
    
    def plot_feature_importance(self, top_n: int = 20):
        """
        Plot feature importance from different methods.
        
        Parameters:
        -----------
        top_n : int
            Number of top features to display
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        # Plot different feature importance scores
        score_methods = ['mutual_info', 'xgb_importance', 'ensemble_votes']
        
        for idx, method in enumerate(score_methods):
            if method in self.feature_scores:
                data = self.feature_scores[method].head(top_n)
                
                ax = axes[idx]
                if method == 'ensemble_votes':
                    sns.barplot(data=data, x='votes', y='feature', ax=ax, palette='viridis')
                    ax.set_xlabel('Number of Votes')
                else:
                    score_col = 'score' if method == 'mutual_info' else 'importance'
                    sns.barplot(data=data, x=score_col, y='feature', ax=ax, palette='viridis')
                    ax.set_xlabel('Score')
                
                ax.set_title(f'Top {top_n} Features - {method.replace("_", " ").title()}')
                ax.set_ylabel('Feature')
        
        # Feature selection summary
        ax = axes[3]
        selection_summary = pd.DataFrame([
            {'Method': method, 'Features Selected': len(features)}
            for method, features in self.selected_features.items()
        ])
        
        sns.barplot(data=selection_summary, x='Method', y='Features Selected', ax=ax, palette='coolwarm')
        ax.set_title('Feature Selection Summary')
        ax.set_xlabel('Selection Method')
        ax.set_ylabel('Number of Features')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'feature_selection_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create correlation heatmap for top features
        if 'xgb_importance' in self.feature_scores:
            self._plot_feature_correlation_heatmap(top_n)
    
    def _plot_feature_correlation_heatmap(self, top_n: int = 20):
        """
        Plot correlation heatmap for top features.
        """
        # This method would need access to the actual feature data
        # It's a placeholder for the correlation visualization
        pass
    
    def get_feature_selection_report(self) -> Dict:
        """
        Generate comprehensive feature selection report.
        
        Returns:
        --------
        Dict
            Feature selection report
        """
        report = {
            'method_summary': {
                method: {
                    'n_features': len(features),
                    'features': features[:10]  # Top 10 for brevity
                }
                for method, features in self.selected_features.items()
            },
            'feature_scores': {
                method: scores.head(10).to_dict('records')
                for method, scores in self.feature_scores.items()
            }
        }
        
        return report