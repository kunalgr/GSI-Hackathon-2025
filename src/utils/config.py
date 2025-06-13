"""
Configuration file for Gold Prospectivity Mapping project.
Contains all project-wide settings and parameters.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "input"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model paths
MODEL_DIR = PROJECT_ROOT / "models"
SAVED_MODELS_DIR = MODEL_DIR / "saved_models"

# Results paths
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = RESULTS_DIR / "reports"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, 
                  SAVED_MODELS_DIR, RESULTS_DIR, FIGURES_DIR, REPORTS_DIR, 
                  PREDICTIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature engineering parameters
CORRELATION_THRESHOLD = 0.95
VARIANCE_THRESHOLD = 0.01

# Model hyperparameters for grid search
HYPERPARAMETERS = {
    'random_forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    },
    'xgboost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    },
    'lightgbm': {
        'n_estimators': [100, 200, 300],
        'num_leaves': [31, 50, 100],
        'learning_rate': [0.01, 0.1, 0.3],
        'feature_fraction': [0.7, 0.8, 0.9],
        'bagging_fraction': [0.7, 0.8, 0.9]
    },
    'logistic_regression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
}

# Geological feature groups
GEOLOGICAL_FEATURES = {
    'geochemistry': ['SiO2_%', 'Al2O3_%', 'Fe2O3_%', 'TiO2_%', 'CaO_%', 'MgO_%', 
                     'MnO_%', 'Na2O_%', 'K2O_%', 'P2O5_%', 'LOI_%'],
    'trace_elements': ['Ba_ppm', 'Ga_ppm', 'Sc_ppm', 'V_ppm', 'Th_ppm', 'Pb_ppm', 
                       'Ni_ppm', 'Co_ppm', 'Rb_ppm', 'Sr_ppm', 'Y_ppm', 'Zr_ppm', 
                       'Nb_ppm', 'Cr_ppm', 'Cu_ppm', 'Zn_ppm', 'Au_ppb'],
    'rare_earth': ['La_ppm', 'Ce_ppm', 'Pr_ppm', 'Nd_ppm', 'Sm_ppm', 'Eu_ppm', 
                   'Tb_ppm', 'Gd_ppm', 'Dy_ppm', 'Ho_ppm', 'Er_ppm', 'Tm_ppm', 
                   'Yb_ppm', 'Lu_ppm'],
    'geophysical': ['gravity_ng', 'mag_tail_t', 'mag_depth', 'spec_doser', 
                    'spec_Kperc', 'spec_Uppm', 'spec_Thppm'],
    'structural': ['fault_dist', 'lineament']
}

# Target column
TARGET_COLUMN = 'likely'

# Columns to exclude from features
EXCLUDE_COLUMNS = ['geometry', 'xcoord', 'ycoord']

GOLD_LITHOLOGY_SELECT = ['META-BASALT', 'ARGILLITE', 'PILLOWED METABASALT', 'PINK GRANITE', 'BANDED IRON FORMATION', 'QUARTZ VEIN/REEF', 'GREY HORNBLENDE BIOTITE GNEISS', 'CHLORITE SERICITE SCHIST', 'CARBONACEOUS PHYLLITE', 'CONGLOMERATE', 'GREY GRANITE']