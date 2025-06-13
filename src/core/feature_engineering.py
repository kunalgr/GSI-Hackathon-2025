"""
Feature engineering module for Gold Prospectivity Mapping.
Creates domain-specific geological features.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.preprocessing import PolynomialFeatures

from utils.config import GEOLOGICAL_FEATURES


class GeologicalFeatureEngineer:
    """
    Class for creating geological domain-specific features.
    """
    
    def __init__(self):
        self.created_features = []
        
    def create_alteration_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create alteration indices commonly used in mineral exploration.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with geochemical data
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with alteration indices
        """
        df_features = df.copy()
        
        # Alteration Index (AI) - Ishikawa alteration index
        if all(col in df.columns for col in ['K2O_%', 'Na2O_%', 'CaO_%', 'MgO_%']):
            df_features['alteration_index'] = (df['K2O_%'] + df['MgO_%']) / \
                                             (df['K2O_%'] + df['MgO_%'] + df['Na2O_%'] + df['CaO_%']) * 100
            self.created_features.append('alteration_index')
        
        # Chlorite-Carbonate-Pyrite Index (CCPI)
        if all(col in df.columns for col in ['MgO_%', 'Fe2O3_%', 'K2O_%', 'Na2O_%']):
            df_features['ccpi_index'] = ((df['MgO_%'] + df['Fe2O3_%']) / 
                                        (df['MgO_%'] + df['Fe2O3_%'] + df['Na2O_%'] + df['K2O_%'])) * 100
            self.created_features.append('ccpi_index')
        
        # Sericite Index
        if all(col in df.columns for col in ['K2O_%', 'Na2O_%']):
            df_features['sericite_index'] = df['K2O_%'] / (df['K2O_%'] + df['Na2O_%'] + 0.001) * 100
            self.created_features.append('sericite_index')
        
        # Silicification Index
        if 'SiO2_%' in df.columns:
            df_features['silicification_index'] = df['SiO2_%'] / df['SiO2_%'].mean()
            self.created_features.append('silicification_index')
        
        return df_features
    
    def create_element_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create element ratios important for gold mineralization.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with element data
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with element ratios
        """
        df_features = df.copy()
        
        # Au pathfinder element ratios
        if all(col in df.columns for col in ['Au_ppb', 'As_ppm']):
            df_features['Au_As_ratio'] = df['Au_ppb'] / (df['As_ppm'] + 0.001)
            self.created_features.append('Au_As_ratio')
        
        if all(col in df.columns for col in ['Au_ppb', 'Sb_ppm']):
            df_features['Au_Sb_ratio'] = df['Au_ppb'] / (df['Sb_ppm'] + 0.001)
            self.created_features.append('Au_Sb_ratio')
        
        # Base metal ratios
        if all(col in df.columns for col in ['Cu_ppm', 'Pb_ppm']):
            df_features['Cu_Pb_ratio'] = df['Cu_ppm'] / (df['Pb_ppm'] + 0.001)
            self.created_features.append('Cu_Pb_ratio')
        
        if all(col in df.columns for col in ['Zn_ppm', 'Pb_ppm']):
            df_features['Zn_Pb_ratio'] = df['Zn_ppm'] / (df['Pb_ppm'] + 0.001)
            self.created_features.append('Zn_Pb_ratio')
        
        # Mafic/Felsic indicators
        if all(col in df.columns for col in ['MgO_%', 'Fe2O3_%', 'SiO2_%']):
            df_features['mafic_index'] = (df['MgO_%'] + df['Fe2O3_%']) / df['SiO2_%']
            self.created_features.append('mafic_index')
        
        # Alkalinity ratio
        if all(col in df.columns for col in ['Na2O_%', 'K2O_%', 'Al2O3_%']):
            df_features['alkalinity_ratio'] = (df['Na2O_%'] + df['K2O_%']) / df['Al2O3_%']
            self.created_features.append('alkalinity_ratio')
        
        return df_features
    
    def create_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create spatial and structural features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with spatial data
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with spatial features
        """
        df_features = df.copy()
        
        # Distance-based features
        if 'fault_dist' in df.columns:
            # Proximity to faults (inverse distance)
            # df_features['fault_proximity'] = 1 / (df['fault_dist'] + 1)
            # self.created_features.append('fault_proximity')
            
            # Fault distance categories
            df_features['fault_dist_category'] = pd.cut(df['fault_dist'], 
                                                       bins=[0, 500, 1000, 2000, 3000, np.inf],
                                                       labels=['very_close', 'close', 'moderate', 'far', 'very_far'])
            df_features = pd.get_dummies(df_features, columns=['fault_dist_category'], prefix='fault_cat')
            
            self.created_features.extend([col for col in df_features.columns if col.startswith('fault_cat_')])
        
        # Lineament density proxy
        if 'lineament' in df.columns:
            df_features['lineament_density'] = df.groupby('lineament')['lineament'].transform('count')
            self.created_features.append('lineament_density')
        
        # Magnetic features
        if all(col in df.columns for col in ['mag_tail_t', 'mag_depth']):
            df_features['mag_gradient'] = df['mag_tail_t'] / (df['mag_depth'] + 1)
            self.created_features.append('mag_gradient')
        
        return df_features
    
    def create_anomaly_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create anomaly scores for key gold pathfinder elements.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with element data
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with anomaly scores
        """
        df_features = df.copy()
        
        # Gold pathfinder elements
        pathfinder_elements = ['Au_ppb', 'As_ppm', 'Sb_ppm', 'Bi_ppm', 'Cu_ppm', 'Pb_ppm', 'Zn_ppm']
        
        for element in pathfinder_elements:
            if element in df.columns:
                # Calculate percentile-based anomaly score
                percentile_75 = df[element].quantile(0.75)
                percentile_90 = df[element].quantile(0.90)
                percentile_95 = df[element].quantile(0.95)
                
                # Create anomaly score
                anomaly_col = f'{element}_anomaly'
                df_features[anomaly_col] = 0
                df_features.loc[df[element] > percentile_75, anomaly_col] = 1
                df_features.loc[df[element] > percentile_90, anomaly_col] = 2
                df_features.loc[df[element] > percentile_95, anomaly_col] = 3
                
                self.created_features.append(anomaly_col)
        
        # Composite anomaly score
        anomaly_cols = [col for col in df_features.columns if col.endswith('_anomaly')]
        if anomaly_cols:
            df_features['composite_anomaly_score'] = df_features[anomaly_cols].sum(axis=1)
            self.created_features.append('composite_anomaly_score')
        
        return df_features
    
    def create_ree_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create Rare Earth Element (REE) pattern features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with REE data
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with REE pattern features
        """
        df_features = df.copy()
        
        # Light REE (LREE)
        lree_elements = ['La_ppm', 'Ce_ppm', 'Pr_ppm', 'Nd_ppm', 'Sm_ppm']
        if all(elem in df.columns for elem in lree_elements):
            df_features['LREE_sum'] = df[lree_elements].sum(axis=1)
            self.created_features.append('LREE_sum')
        
        # Heavy REE (HREE)
        hree_elements = ['Gd_ppm', 'Dy_ppm', 'Ho_ppm', 'Er_ppm', 'Tm_ppm', 'Yb_ppm', 'Lu_ppm']
        if all(elem in df.columns for elem in hree_elements):
            df_features['HREE_sum'] = df[hree_elements].sum(axis=1)
            self.created_features.append('HREE_sum')
        
        # LREE/HREE ratio
        if 'LREE_sum' in df_features.columns and 'HREE_sum' in df_features.columns:
            df_features['LREE_HREE_ratio'] = df_features['LREE_sum'] / (df_features['HREE_sum'] + 0.001)
            self.created_features.append('LREE_HREE_ratio')
        
        # Eu anomaly
        if all(elem in df.columns for elem in ['Eu_ppm', 'Sm_ppm', 'Gd_ppm']):
            df_features['Eu_anomaly'] = df['Eu_ppm'] / np.sqrt((df['Sm_ppm'] * df['Gd_ppm']) + 0.001)
            self.created_features.append('Eu_anomaly')
        
        return df_features
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering transformations.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with engineered features
        """
        # Reset created features list
        self.created_features = []
        
        # Apply all transformations
        df_engineered = df.copy()
        df_engineered = self.create_alteration_indices(df_engineered)
        df_engineered = self.create_element_ratios(df_engineered)
        df_engineered = self.create_spatial_features(df_engineered)
        df_engineered = self.create_anomaly_scores(df_engineered)
        df_engineered = self.create_ree_patterns(df_engineered)
        
        # Handle any infinite or NaN values created during feature engineering
        df_engineered = df_engineered.replace([np.inf, -np.inf], np.nan)
        df_engineered = df_engineered.fillna(0)
        
        print(f"Created {len(self.created_features)} new geological features")
        
        return df_engineered
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """
        Return feature groups for interpretation.
        
        Returns:
        --------
        Dict[str, List[str]]
            Dictionary of feature groups
        """
        feature_groups = GEOLOGICAL_FEATURES.copy()
        feature_groups['engineered_alteration'] = [f for f in self.created_features 
                                                  if 'index' in f or 'ratio' in f]
        feature_groups['engineered_anomaly'] = [f for f in self.created_features 
                                              if 'anomaly' in f]
        feature_groups['engineered_spatial'] = [f for f in self.created_features 
                                              if 'fault' in f or 'lineament' in f or 'mag' in f]
        feature_groups['engineered_ree'] = [f for f in self.created_features 
                                          if 'REE' in f or 'Eu_anomaly' in f]
        
        return feature_groups