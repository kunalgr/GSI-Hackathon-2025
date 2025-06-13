"""
Model evaluation module for Gold Prospectivity Mapping.
Generates comprehensive evaluation reports and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve
from typing import Dict, List, Tuple, Any
import json

from utils.config import FIGURES_DIR, REPORTS_DIR
from core.model_training import ModelTrainer


class ModelEvaluator:
    """
    Class for comprehensive model evaluation and visualization.
    """
    
    def __init__(self):
        self.evaluation_results = {}
        
    def evaluate_all_models(self, model_results: Dict[str, Any], 
                          ensemble_results: Dict[str, Any],
                          y_val: pd.Series) -> Dict[str, Any]:
        """
        Evaluate all models including ensembles.
        
        Parameters:
        -----------
        model_results : Dict[str, Any]
            Results from individual models
        ensemble_results : Dict[str, Any]
            Results from ensemble models
        y_val : pd.Series
            Validation target
            
        Returns:
        --------
        Dict[str, Any]
            Comprehensive evaluation results
        """
        all_results = {}
        
        # Evaluate individual models
        for model_name, results in model_results.items():
            all_results[model_name] = {
                'type': 'individual',
                'metrics': results['validation_metrics'],
                'predictions': results['predictions'],
                'probabilities': results.get('probabilities', results['predictions'])
            }
        
        # Evaluate ensemble models
        for ensemble_name, results in ensemble_results.items():
            predictions = results['predictions']
            
            # Convert probabilities to class predictions if needed
            if predictions.ndim == 1 and predictions.max() <= 1:
                y_pred = (predictions >= 0.5).astype(int)
                y_proba = predictions
            else:
                y_pred = predictions
                y_proba = predictions
            
            # Calculate metrics
            
            trainer = ModelTrainer()
            metrics = trainer.calculate_metrics(y_val, y_pred, y_proba if y_proba.max() <= 1 else None)
            
            all_results[ensemble_name] = {
                'type': 'ensemble',
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_proba,
                'additional_info': {k: v for k, v in results.items() if k not in ['predictions', 'score']}
            }
        
        self.evaluation_results = all_results
        return all_results
    
    def plot_roc_curves(self, results: Dict[str, Any], y_true: pd.Series):
        """
        Plot ROC curves for all models.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Model results
        y_true : pd.Series
            True labels
        """
        plt.figure(figsize=(12, 8))
        
        # Plot ROC curve for each model
        for model_name, result in results.items():
            if 'probabilities' in result and result['probabilities'].max() <= 1:
                fpr, tpr, _ = roc_curve(y_true, result['probabilities'])
                roc_auc = auc(fpr, tpr)
                
                # Different line styles for individual vs ensemble models
                linestyle = '-' if result['type'] == 'individual' else '--'
                plt.plot(fpr, tpr, linestyle=linestyle, linewidth=2,
                        label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Gold Prospectivity Models', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'roc_curves_all_models.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_precision_recall_curves(self, results: Dict[str, Any], y_true: pd.Series):
        """
        Plot Precision-Recall curves for all models.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Model results
        y_true : pd.Series
            True labels
        """
        plt.figure(figsize=(12, 8))
        
        for model_name, result in results.items():
            if 'probabilities' in result and result['probabilities'].max() <= 1:
                precision, recall, _ = precision_recall_curve(y_true, result['probabilities'])
                avg_precision = average_precision_score(y_true, result['probabilities'])
                
                linestyle = '-' if result['type'] == 'individual' else '--'
                plt.plot(recall, precision, linestyle=linestyle, linewidth=2,
                        label=f'{model_name} (AP = {avg_precision:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves - Gold Prospectivity Models', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrices(self, results: Dict[str, Any], y_true: pd.Series, 
                              top_n: int = 6):
        """
        Plot confusion matrices for top models.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Model results
        y_true : pd.Series
            True labels
        top_n : int
            Number of top models to plot
        """
        # Sort models by ROC-AUC
        sorted_models = sorted(results.items(), 
                             key=lambda x: x[1]['metrics'].get('roc_auc', 0), 
                             reverse=True)[:top_n]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, (model_name, result) in enumerate(sorted_models):
            cm = confusion_matrix(y_true, result['predictions'])
            
            ax = axes[idx]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Non-prospective', 'Prospective'],
                       yticklabels=['Non-prospective', 'Prospective'])
            ax.set_title(f'{model_name}\n(AUC: {result["metrics"].get("roc_auc", 0):.3f})')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        # Hide unused subplots
        for idx in range(len(sorted_models), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Confusion Matrices - Top Models', fontsize=16)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'confusion_matrices_top_models.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_model_comparison(self, results: Dict[str, Any]):
        """
        Plot comprehensive model comparison.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Model results
        """
        # Prepare data for plotting
        metrics_data = []
        for model_name, result in results.items():
            metrics = result['metrics']
            metrics_data.append({
                'Model': model_name,
                'Type': result['type'],
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'ROC-AUC': metrics['roc_auc'],
                'Specificity': metrics['specificity']
            })
        
        df_metrics = pd.DataFrame(metrics_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Bar plot of all metrics
        ax1 = axes[0, 0]
        df_sorted = df_metrics.sort_values('ROC-AUC', ascending=False).head(10)
        metrics_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Specificity']
        df_plot = df_sorted.set_index('Model')[metrics_cols]
        df_plot.plot(kind='bar', ax=ax1)
        ax1.set_title('Model Performance Comparison (Top 10 Models)', fontsize=14)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. Scatter plot: Precision vs Recall
        ax2 = axes[0, 1]
        colors = ['blue' if t == 'individual' else 'red' for t in df_metrics['Type']]
        scatter = ax2.scatter(df_metrics['Recall'], df_metrics['Precision'], 
                            c=colors, s=100, alpha=0.6)
        
        # Annotate top models
        top_models = df_metrics.nlargest(5, 'F1-Score')
        for _, row in top_models.iterrows():
            ax2.annotate(row['Model'], (row['Recall'], row['Precision']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax2.set_xlabel('Recall (Sensitivity)', fontsize=12)
        ax2.set_ylabel('Precision', fontsize=12)
        ax2.set_title('Precision vs Recall Trade-off', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='blue', label='Individual'),
                         Patch(facecolor='red', label='Ensemble')]
        ax2.legend(handles=legend_elements)
        
        # 3. ROC-AUC comparison
        ax3 = axes[1, 0]
        df_auc = df_metrics.sort_values('ROC-AUC', ascending=True).tail(15)
        colors = ['lightblue' if t == 'individual' else 'lightcoral' for t in df_auc['Type']]
        ax3.barh(df_auc['Model'], df_auc['ROC-AUC'], color=colors)
        ax3.set_xlabel('ROC-AUC Score', fontsize=12)
        ax3.set_title('ROC-AUC Scores Comparison', fontsize=14)
        ax3.grid(True, alpha=0.3, axis='x')
        
        # 4. Heatmap of metrics
        ax4 = axes[1, 1]
        heatmap_data = df_metrics.set_index('Model')[metrics_cols].head(10)
        sns.heatmap(heatmap_data.T, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax4)
        ax4.set_title('Performance Metrics Heatmap (Top 10)', fontsize=14)
        
        plt.suptitle('Comprehensive Model Evaluation - Gold Prospectivity Mapping', fontsize=16)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'model_comparison_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_calibration_curves(self, results: Dict[str, Any], y_true: pd.Series, 
                              n_bins: int = 10):
        """
        Plot calibration curves to assess prediction reliability.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Model results
        y_true : pd.Series
            True labels
        n_bins : int
            Number of calibration bins
        """
        plt.figure(figsize=(10, 8))
        
        # Plot perfect calibration line
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        
        # Plot calibration curves for top models
        top_models = sorted(results.items(), 
                          key=lambda x: x[1]['metrics'].get('roc_auc', 0), 
                          reverse=True)[:6]
        
        for model_name, result in top_models:
            if 'probabilities' in result and result['probabilities'].max() <= 1:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_true, result['probabilities'], n_bins=n_bins
                )
                
                plt.plot(mean_predicted_value, fraction_of_positives, 
                        marker='o', linewidth=2, label=model_name)
        
        plt.xlabel('Mean Predicted Probability', fontsize=12)
        plt.ylabel('Fraction of Positives', fontsize=12)
        plt.title('Calibration Curves - Model Reliability Assessment', fontsize=14)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'calibration_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_evaluation_report(self, results: Dict[str, Any], 
                                feature_importance: Dict[str, pd.DataFrame]) -> Dict:
        """
        Generate comprehensive evaluation report.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Model results
        feature_importance : Dict[str, pd.DataFrame]
            Feature importance for each model
            
        Returns:
        --------
        Dict
            Evaluation report
        """
        report = {
            'summary': {
                'total_models': len(results),
                'individual_models': sum(1 for r in results.values() if r['type'] == 'individual'),
                'ensemble_models': sum(1 for r in results.values() if r['type'] == 'ensemble'),
                'best_model': max(results.items(), key=lambda x: x[1]['metrics']['roc_auc'])[0],
                # Convert best_roc_auc to a standard Python float
                'best_roc_auc': float(max(r['metrics']['roc_auc'] for r in results.values()))
            },
            'detailed_results': {},
            'feature_importance': {},
            'recommendations': []
        }
        
        # Add detailed results for each model
        for model_name, result in results.items():
            report['detailed_results'][model_name] = {
                'type': result['type'],
                # FIX: Convert all metric values to standard Python floats
                # A dictionary comprehension is a clean way to do this.
                'metrics': {key: float(value) for key, value in result['metrics'].items()},
                'confusion_matrix': {
                    'tn': int(result['metrics']['true_negatives']),
                    'fp': int(result['metrics']['false_positives']),
                    'fn': int(result['metrics']['false_negatives']),
                    'tp': int(result['metrics']['true_positives'])
                }
            }
        
        # Add feature importance
        for model_name, importance in feature_importance.items():
            if not importance.empty:
                report['feature_importance'][model_name] = importance.head(20).to_dict('records')
        
        # Generate recommendations
        best_model = report['summary']['best_model']
        # Use the already cleaned-up metrics from the report dict
        best_metrics = report['detailed_results'][best_model]['metrics']
        
        report['recommendations'] = [
            f"Best performing model: {best_model} with ROC-AUC of {best_metrics['roc_auc']:.4f}",
            f"The model achieves {best_metrics['recall']:.2%} recall (sensitivity) for identifying prospective areas",
            f"Specificity of {best_metrics['specificity']:.2%} helps minimize false positives",
            f"Consider using ensemble methods if individual model performance is not satisfactory",
            "Regular model updates with new geological data are recommended"
        ]

        print("Successfully generated report object. Saving to file...")
        
        # Save report
        # Ensure REPORTS_DIR is a Path object if you are using the / operator
        # For example: REPORTS_DIR = Path('path/to/your/reports')
        report_path = REPORTS_DIR / 'model_evaluation_report.json'
        report_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report

    
    def create_summary_visualization(self, report: Dict):
        """
        Create a summary visualization of the evaluation report.
        
        Parameters:
        -----------
        report : Dict
            Evaluation report
        """
        # Create a summary infographic
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Model types pie chart
        ax1.pie([report['summary']['individual_models'], report['summary']['ensemble_models']], 
               labels=['Individual', 'Ensemble'], autopct='%1.0f%%', startangle=90,
               colors=['lightblue', 'lightcoral'])
        ax1.set_title('Model Types Distribution')
        
        # 2. Top models performance
        top_models = sorted(report['detailed_results'].items(), 
                          key=lambda x: x[1]['metrics']['roc_auc'], 
                          reverse=True)[:5]
        
        model_names = [m[0] for m in top_models]
        roc_scores = [m[1]['metrics']['roc_auc'] for m in top_models]
        
        ax2.bar(model_names, roc_scores, color='skyblue')
        ax2.set_ylabel('ROC-AUC Score')
        ax2.set_title('Top 5 Models by ROC-AUC')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. Best model metrics radar chart
        best_model_name = report['summary']['best_model']
        best_metrics = report['detailed_results'][best_model_name]['metrics']
        
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
        values = [best_metrics['accuracy'], best_metrics['precision'], 
                 best_metrics['recall'], best_metrics['f1_score'], 
                 best_metrics['specificity']]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        ax3.plot(angles, values, 'o-', linewidth=2, color='darkblue')
        ax3.fill(angles, values, alpha=0.25, color='darkblue')
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(categories)
        ax3.set_ylim(0, 1)
        ax3.set_title(f'Best Model ({best_model_name}) Performance')
        ax3.grid(True)
        
        # 4. Summary text
        ax4.axis('off')
        summary_text = f"""
        Gold Prospectivity Mapping - Model Evaluation Summary
        
        Total Models Evaluated: {report['summary']['total_models']}
        Best Model: {report['summary']['best_model']}
        Best ROC-AUC Score: {report['summary']['best_roc_auc']:.4f}
        
        Key Insights:
        • Ensemble methods generally outperform individual models
        • Feature engineering significantly improves performance
        • Geological domain knowledge is crucial for interpretation
        
        Recommendations:
        • Deploy {report['summary']['best_model']} for prospectivity mapping
        • Consider threshold optimization based on exploration budget
        • Validate predictions with field data
        """
        ax4.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center')
        
        plt.suptitle('Gold Prospectivity Mapping - Evaluation Summary', fontsize=16)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'evaluation_summary.png', dpi=300, bbox_inches='tight')
        plt.close()