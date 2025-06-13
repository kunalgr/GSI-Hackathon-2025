"""
Data preprocessing module for Gold Prospectivity Mapping.
Handles data loading, cleaning, and initial preprocessing.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

from utils.config import *


class DataPreprocessor:
    """
    Class for handling all data preprocessing tasks.
    """
    
    def __init__(self):
        self.scaler = None
        self.feature_columns = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from file. Supports CSV and other formats.
        
        Parameters:
        -----------
        filepath : str
            Path to the data file
            
        Returns:
        --------
        pd.DataFrame
            Loaded dataframe
        """
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        else:
            # For DBF or other formats, use geopandas
            try:
                gdf = gpd.read_file(filepath)
                return pd.DataFrame(gdf.drop(columns='geometry', errors='ignore'))
            except:
                return pd.read_csv(filepath)
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the geological data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw dataframe
            
        Returns:
        --------
        pd.DataFrame
            Cleaned dataframe
        """
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Remove duplicate rows if any
        df_clean = df_clean.drop_duplicates()
        
        # Handle missing values in numerical columns
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if df_clean[col].isna().any():
                # For geological data, use median imputation
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        # Handle categorical columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in EXCLUDE_COLUMNS + [TARGET_COLUMN]:
                # For geological categories like lithology, formation
                df_clean[col].fillna('Unknown', inplace=True)
        
        if TARGET_COLUMN in df_clean.columns:
            df_clean[TARGET_COLUMN] = df_clean[TARGET_COLUMN].astype(int)

        
        return df_clean
    
    def prepare_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Prepare features for modeling.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Cleaned dataframe
        is_training : bool
            Whether this is training data
            
        Returns:
        --------
        pd.DataFrame
            Feature dataframe
        """
        # Identify feature columns
        if is_training:
            self.feature_columns = [col for col in df.columns 
                                  if col not in EXCLUDE_COLUMNS + [TARGET_COLUMN] 
                                  and df[col].dtype in [np.float64, np.int64]]
        
        # Select features
        features = df[self.feature_columns].copy()
        
        # Handle any remaining NaN values
        features = features.fillna(0)
        
        # Replace infinite values
        features = features.replace([np.inf, -np.inf], 0)
        
        return features
    
    def scale_features(self, features: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale features using RobustScaler (better for geological data with outliers).
        
        Parameters:
        -----------
        features : pd.DataFrame
            Feature dataframe
        fit : bool
            Whether to fit the scaler
            
        Returns:
        --------
        pd.DataFrame
            Scaled features
        """
        if fit:
            self.scaler = RobustScaler()
            scaled_features = self.scaler.fit_transform(features)
        else:
            scaled_features = self.scaler.transform(features)
        
        return pd.DataFrame(scaled_features, columns=features.columns, index=features.index)
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical geological features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with categorical features
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with encoded features
        """
        df_encoded = df.copy()
        
        # One-hot encode geological categorical features
        categorical_cols = ['age', 'lithology', 'formation'] if 'age' in df.columns else []
        
        for col in categorical_cols:
            if col in df.columns:
                # Create dummy variables
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded.drop(col, axis=1, inplace=True)
        
        return df_encoded
    
    def split_data(self, features: pd.DataFrame, target: pd.Series, 
                   test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE) -> Tuple:
        """
        Split data into training and validation sets.
        
        Parameters:
        -----------
        features : pd.DataFrame
            Feature dataframe
        target : pd.Series
            Target variable
        test_size : float
            Test set size
        random_state : int
            Random state for reproducibility
            
        Returns:
        --------
        Tuple
            X_train, X_val, y_train, y_val
        """
        return train_test_split(features, target, test_size=test_size, 
                              random_state=random_state, stratify=target)
    
    def process_training_data(self, filepath: str) -> Tuple:
        """
        Complete processing pipeline for training data.
        
        Parameters:
        -----------
        filepath : str
            Path to training data
            
        Returns:
        --------
        Tuple
            X_train, X_val, y_train, y_val, feature_names
        """
        # Load data
        df = self.load_data(filepath)
        
        # Clean data
        df = self.clean_data(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Prepare features
        features = self.prepare_features(df, is_training=True)
        
        # Get target
        target = df[TARGET_COLUMN]
        
        # Split data
        X_train, X_val, y_train, y_val = self.split_data(features, target)
        
        # Scale features
        X_train_scaled = self.scale_features(X_train, fit=True)
        X_val_scaled = self.scale_features(X_val, fit=False)
        
        return X_train_scaled, X_val_scaled, y_train, y_val, list(features.columns)
    
    def process_test_data(self, filepath: str) -> pd.DataFrame:
        """
        Process test data using fitted preprocessor.
        
        Parameters:
        -----------
        filepath : str
            Path to test data
            
        Returns:
        --------
        pd.DataFrame
            Processed test features
        """
        # Load data
        df = self.load_data(filepath)
        
        # Clean data
        df = self.clean_data(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Prepare features
        features = self.prepare_features(df, is_training=False)
        
        # Scale features
        features_scaled = self.scale_features(features, fit=False)
        
        return features_scaled