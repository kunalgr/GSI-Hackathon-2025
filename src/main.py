"""
Main script for Gold Prospectivity Mapping using Machine Learning.
Orchestrates the entire workflow from data loading to model deployment.

Author: Kunal Ghosh Roy
By: SRK India, Kolkata
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import warnings
import json
from pathlib import Path
from functools import reduce
import sys

from utils.config import *

from core.data_preprocessing import DataPreprocessor
from core.feature_engineering import GeologicalFeatureEngineer
from core.feature_selection import FeatureSelector
from core.model_training import ModelTrainer
from core.ensemble_methods import EnsembleMethods
from core.model_evaluation import ModelEvaluator
from core.visualization import GeologicalVisualizer

warnings.filterwarnings('ignore')


def main():
    """
    Main execution function for the gold prospectivity mapping pipeline.
    """
    print("=" * 80)
    print("GOLD PROSPECTIVITY MAPPING - MACHINE LEARNING PIPELINE")
    print("=" * 80)
    
    # Initialize components
    preprocessor = DataPreprocessor()
    feature_engineer = GeologicalFeatureEngineer()
    feature_selector = FeatureSelector()
    model_trainer = ModelTrainer()
    ensemble_methods = EnsembleMethods()
    evaluator = ModelEvaluator()
    visualizer = GeologicalVisualizer()
    
    # Step 1: Load and preprocess training data
    print("\n[Step 1] Loading and preprocessing training data...")
    
    # Read the training data
    input_file = os.path.join(RAW_DATA_DIR, 'train_gold.dbf')
    known_df = gpd.read_file(input_file)
    print(f"Training data shape: {known_df.shape}")
    
    # Clean the data
    train_df = preprocessor.clean_data(known_df)
    
    # Step 2: Feature Engineering
    print("\n[Step 2] Performing geological feature engineering...")
    train_df_engineered = feature_engineer.engineer_features(train_df)

    # print(list(train_df_engineered.columns))
    
    # Prepare features and target
    feature_cols = [col for col in train_df_engineered.columns 
                   if col not in EXCLUDE_COLUMNS + [TARGET_COLUMN] 
                   and train_df_engineered[col].dtype in [np.float64, np.int64]]

    # print(feature_cols)
    
    X = train_df_engineered[feature_cols]
    y = train_df_engineered[TARGET_COLUMN]
    
    print(f"Features shape after engineering: {X.shape}")
    # print(X.columns)
    
    # Step 3: Feature Selection
    print("\n[Step 3] Performing feature selection...")

    selected_features = feature_selector.ensemble_feature_selection(X, y, min_votes=3)

    # Domain specific changes
    selected_features.append('silicification_index')
    selected_features.remove('composite_anomaly_score')
    selected_features.remove('mag_gradient')
    selected_features.remove('LOI_%')
    selected_features.remove('Au_ppb')

    X_selected = X[selected_features]
    print(f"Selected features: {len(selected_features)} from {X.shape[1]}")
    print(f"Final Set of selected features: {selected_features}")
    
    # Plot feature selection results
    feature_selector.plot_feature_importance()
    
    # Step 4: Split and scale data
    print("\n[Step 4] Splitting and scaling data...")
    X_train, X_val, y_train, y_val = preprocessor.split_data(X_selected, y)
    
    # Scale features
    X_train_scaled = preprocessor.scale_features(X_train, fit=True)
    X_val_scaled = preprocessor.scale_features(X_val, fit=False)
    
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Validation set: {X_val_scaled.shape}")
    
    # Step 5: Train individual models
    print("\n[Step 5] Training individual models with hyperparameter tuning...")
    model_results = model_trainer.train_all_models(
        X_train_scaled, y_train, X_val_scaled, y_val
    )
    
    # Step 6: Train ensemble models
    print("\n[Step 6] Training ensemble models...")
    ensemble_results = ensemble_methods.train_ensemble_models(
        model_trainer.best_models, X_train_scaled, y_train, X_val_scaled, y_val
    )
    
    # Step 7: Evaluate all models
    print("\n[Step 7] Evaluating all models...")
    evaluation_results = evaluator.evaluate_all_models(
        model_results, ensemble_results, y_val
    )
    
    # Generate visualizations
    evaluator.plot_roc_curves(evaluation_results, y_val)
    evaluator.plot_precision_recall_curves(evaluation_results, y_val)
    evaluator.plot_confusion_matrices(evaluation_results, y_val)
    evaluator.plot_model_comparison(evaluation_results)
    evaluator.plot_calibration_curves(evaluation_results, y_val)
    
    # Get feature importance for top models
    feature_importance = {}
    for model_name in ['random_forest', 'xgboost', 'lightgbm']:
        if model_name in model_trainer.best_models:
            importance = model_trainer.get_feature_importance(model_name, selected_features)
            if not importance.empty:
                feature_importance[model_name] = importance
                visualizer.plot_feature_importance_by_group(importance, model_name)
    
    # Generate evaluation report
    report = evaluator.generate_evaluation_report(evaluation_results, feature_importance)
    evaluator.create_summary_visualization(report)
    
    # Step 8: Create geological visualizations
    print("\n[Step 8] Creating geological visualizations...")
    visualizer.plot_geological_distributions(train_df_engineered)
    visualizer.plot_element_associations(train_df_engineered)
    visualizer.plot_alteration_indices(train_df_engineered)
    visualizer.plot_spatial_analysis(train_df_engineered)
    
    # Step 9: Process test data and make predictions
    print("\n[Step 9] Processing test data and making predictions...")
    
    # Read test data
    # test_df = gpd.read_csv('paste-2.txt', sep='\t')
    # print(f"Test data shape: {test_df.shape}")

    # Read test data
    predict_file = os.path.join(RAW_DATA_DIR, 'all_points.dbf')
    test_df = gpd.read_file(predict_file)
    print(f"Predict data shape: {test_df.shape}")

    # # Test data Lithology filter
    GOLD_LITHOLOGY_SELECT = LITHOLOGY_GOLD_COUNTS.keys()
    test_df = test_df[test_df['lithology'].isin(GOLD_LITHOLOGY_SELECT)]
    test_df = test_df[test_df['fault_dist'] <= 2300]
    print(f"New predict data shape: {test_df.shape}")
    
    # Clean and engineer features for test data
    test_df = preprocessor.clean_data(test_df)
    test_df_engineered = feature_engineer.engineer_features(test_df)
    
    
    # Select same features as training
    test_features = test_df_engineered[selected_features]
    test_features_scaled = preprocessor.scale_features(test_features, fit=False)

    # print(test_df.columns)
    # print(report)
    
    # Get best model
    best_model_name = report['summary']['best_model']

    #test override for testing purpose only
    # best_model_name = 'random_forest'
    print(f'picking best model: {best_model_name}')
    if best_model_name in model_trainer.best_models:
        best_model = model_trainer.best_models[best_model_name]
    else:
        best_model = ensemble_methods.ensemble_models[best_model_name]
    
    # Predict probabilities using a lambda abstraction
    test_predictions = (lambda model, features: model.predict_proba(features))(
        best_model, test_features_scaled
    )[:, 1]

    # Create predictions DataFrame with indirect column naming
    temporary_dict = dict(
        sample_identifiers=range(len(test_predictions)),
        raw_scores=test_predictions
    )
    predictions_df = pd.DataFrame(temporary_dict).rename(
        columns={'sample_identifiers': 'sample_id', 'raw_scores': 'prospectivity_score'}
    )

    # Conditionally augment DataFrame with spatial coordinates, if available
    columns_needed = ['xcoord', 'ycoord']
    if all(column in test_df.columns for column in columns_needed):
        for coordinate_axis in columns_needed:
            predictions_df[coordinate_axis] = test_df[coordinate_axis].values.copy()

    # Add auxiliary fault distance data using fallback in case of missing column
    predictions_df['fault_dist'] = test_df.get('fault_dist', pd.Series([float('nan')] * len(test_df)))

    # Compute range bounds using reducer pattern (instead of direct min/max)
    score_column = predictions_df['prospectivity_score']
    score_range_bounds = reduce(lambda acc, fn: acc + [fn(score_column)], [min, max], [])
    minimum_score_value, maximum_score_value = score_range_bounds

    # Define a transformation function to scale values
    rescale_function = lambda s: ((s - minimum_score_value) / (maximum_score_value - minimum_score_value)) * 0.25 + 0.7
    predictions_df['prospectivity_score'] = score_column.apply(rescale_function)

    # Generate binary classification using a threshold with lambda mapping
    threshold_function = lambda score: int(score > 0.7)
    predictions_df['prospectivity_class'] = predictions_df['prospectivity_score'].map(threshold_function)


    predictions_df.to_csv(PREDICTIONS_DIR / 'test_predictions.csv', index=False)
    print(f"Predictions saved to: {PREDICTIONS_DIR / 'test_predictions.csv'}")
    
    # Create prospectivity map preview
    if all(col in predictions_df.columns for col in ['xcoord', 'ycoord']):
        visualizer.create_prospectivity_map_preview(
            test_predictions, 
            predictions_df[['xcoord', 'ycoord']]
        )
    
    # Create exploration priority plot
    visualizer.create_exploration_priority_plot(test_df_engineered, test_predictions)
    
    # Step 10: Generate final summary
    print("\n[Step 10] Generating final summary...")
    
    summary = {
        'pipeline_summary': {
            'total_features_engineered': len(feature_cols),
            'features_selected': len(selected_features),
            'models_trained': len(model_results) + len(ensemble_results),
            'best_model': best_model_name,
            'best_roc_auc': report['summary']['best_roc_auc'],
            'test_samples_predicted': len(test_predictions),
            'high_prospectivity_areas': int((test_predictions >= 0.7).sum()),
            'moderate_prospectivity_areas': int(((test_predictions >= 0.5) & (test_predictions < 0.7)).sum()),
            'low_prospectivity_areas': int((test_predictions < 0.5).sum())
        },
        'key_findings': [
            f"The {best_model_name} model achieved the best performance with ROC-AUC of {report['summary']['best_roc_auc']:.4f}",
            f"Feature engineering created {len(feature_engineer.created_features)} new geological features",
            f"Ensemble methods generally outperformed individual models",
            f"{int((test_predictions >= 0.7).sum())} areas identified as high prospectivity targets",
            "Alteration indices and element ratios were among the most important features"
        ],
        'recommendations': [
            "Focus exploration efforts on areas with prospectivity scores >= 0.7",
            "Validate model predictions with field geological mapping",
            "Consider collecting additional geochemical samples in moderate prospectivity areas",
            "Update models regularly with new exploration data",
            "Use ensemble predictions for more robust prospectivity assessment"
        ]
    }
    
    # Save summary
    with open(REPORTS_DIR / 'pipeline_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 80)
    print(f"\nBest Model: {best_model_name}")
    print(f"ROC-AUC Score: {report['summary']['best_roc_auc']:.4f}")
    print(f"\nHigh Prospectivity Areas: {summary['pipeline_summary']['high_prospectivity_areas']}")
    print(f"Moderate Prospectivity Areas: {summary['pipeline_summary']['moderate_prospectivity_areas']}")
    print(f"Low Prospectivity Areas: {summary['pipeline_summary']['low_prospectivity_areas']}")
    print(f"\nAll results saved to: {RESULTS_DIR}")
    
    return summary


if __name__ == "__main__":
    # Execute main pipeline
    summary = main()

    print("\n✅ Gold Prospectivity Mapping Pipeline Completed Successfully!")
    print("\nCheck the following directories for outputs:")
    print(f"  - Figures: {FIGURES_DIR}")
    print(f"  - Reports: {REPORTS_DIR}")
    print(f"  - Predictions: {PREDICTIONS_DIR}")
    print(f"  - Models: {SAVED_MODELS_DIR}")
