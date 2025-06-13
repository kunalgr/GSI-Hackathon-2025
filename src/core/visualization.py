"""
Visualization module for Gold Prospectivity Mapping.
Creates geological and statistical visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
from shapely.geometry import MultiPoint, Point

from utils.config import FIGURES_DIR, GEOLOGICAL_FEATURES


class GeologicalVisualizer:
    """
    Class for creating geological data visualizations.
    """
    
    def __init__(self):
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        self.geological_colors = self._create_geological_colormap()
        
    def _create_geological_colormap(self):
        """
        Create custom colormap for geological data.
        """
        colors = ['#2b83ba', '#5aae61', '#fdb863', '#e66101', '#b2182b']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('geological', colors, N=n_bins)
        return cmap
    
    def plot_geological_distributions(self, df: pd.DataFrame, target_col: str = 'likely'):
        """
        Plot distributions of geological features by target class.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with geological features
        target_col : str
            Target column name
        """
        # Select key geological features
        key_features = ['Au_ppb', 'As_ppm', 'Sb_ppm', 'Cu_ppm', 'fault_dist', 'gravity_ng']
        available_features = [f for f in key_features if f in df.columns]
        
        n_features = len(available_features)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, feature in enumerate(available_features):
            ax = axes[idx]
            
            # Plot distributions for each class
            for class_val in df[target_col].unique():
                data = df[df[target_col] == class_val][feature]
                label = 'Prospective' if class_val == 1 else 'Non-prospective'
                
                # Use log scale for skewed distributions
                if feature in ['Au_ppb', 'As_ppm', 'fault_dist']:
                    data = np.log1p(data)
                    xlabel = f'Log({feature})'
                else:
                    xlabel = feature
                
                ax.hist(data, bins=30, alpha=0.6, label=label, density=True)
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Density')
            ax.set_title(f'Distribution of {feature}')
            ax.legend()
        
        # Hide unused subplots
        for idx in range(len(available_features), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Geological Feature Distributions by Prospectivity', fontsize=16)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'geological_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_element_associations(self, df: pd.DataFrame):
        """
        Plot element association matrix (correlation).
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with element concentrations
        """
        # Select trace elements
        elements = ['Au_ppb', 'As_ppm', 'Sb_ppm', 'Bi_ppm', 'Cu_ppm', 'Pb_ppm', 
                   'Zn_ppm', 'Ag_ppb', 'Mo_ppm', 'W_ppm']
        available_elements = [e for e in elements if e in df.columns]
        
        # Calculate correlation matrix
        corr_matrix = df[available_elements].corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                    cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                    square=True, linewidths=.5)
        
        plt.title('Pathfinder Element Association Matrix', fontsize=14)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'element_associations.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_alteration_indices(self, df: pd.DataFrame):
        """
        Plot alteration indices if available.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with alteration indices
        """
        # Check for alteration indices
        alteration_cols = [col for col in df.columns if 'index' in col or 'ratio' in col]
        
        if not alteration_cols:
            return
        
        n_indices = len(alteration_cols)
        n_cols = min(3, n_indices)
        n_rows = (n_indices + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, col in enumerate(alteration_cols):
            row = idx // n_cols
            col_idx = idx % n_cols
            ax = axes[row, col_idx]
            
            # Create box plots by lithology if available
            if 'lithology' in df.columns:
                # Get top lithologies
                top_lithologies = df['lithology'].value_counts().head(6).index
                data_to_plot = df[df['lithology'].isin(top_lithologies)]
                
                sns.boxplot(data=data_to_plot, x='lithology', y=col, ax=ax)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            else:
                ax.hist(df[col], bins=30, color='skyblue', edgecolor='black')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
            
            ax.set_title(f'{col.replace("_", " ").title()}')
        
        # Hide unused subplots
        for idx in range(len(alteration_cols), n_rows * n_cols):
            row = idx // n_cols
            col_idx = idx % n_cols
            axes[row, col_idx].set_visible(False)
        
        plt.suptitle('Alteration Indices Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'alteration_indices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_spatial_analysis(self, df: pd.DataFrame):
        """
        Plot spatial analysis if coordinates are available.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with spatial information
        """
        if not all(col in df.columns for col in ['xcoord', 'ycoord']):
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Spatial distribution of samples
        ax1 = axes[0, 0]
        if 'likely' in df.columns:
            scatter = ax1.scatter(df['xcoord'], df['ycoord'], 
                                c=df['likely'], cmap='RdYlGn', 
                                s=50, alpha=0.6)
            plt.colorbar(scatter, ax=ax1, label='Prospectivity')
        else:
            ax1.scatter(df['xcoord'], df['ycoord'], s=50, alpha=0.6)
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.set_title('Spatial Distribution of Samples')
        
        # 2. Au concentration spatial distribution
        if 'Au_ppb' in df.columns:
            ax2 = axes[0, 1]
            scatter = ax2.scatter(df['xcoord'], df['ycoord'], 
                                c=np.log1p(df['Au_ppb']), cmap='hot', 
                                s=50, alpha=0.6)
            plt.colorbar(scatter, ax=ax2, label='Log(Au_ppb)')
            ax2.set_xlabel('X Coordinate')
            ax2.set_ylabel('Y Coordinate')
            ax2.set_title('Gold Concentration Distribution')
        
        # 3. Fault distance analysis
        if 'fault_dist' in df.columns:
            ax3 = axes[1, 0]
            scatter = ax3.scatter(df['xcoord'], df['ycoord'], 
                                c=df['fault_dist'], cmap='viridis_r', 
                                s=50, alpha=0.6)
            plt.colorbar(scatter, ax=ax3, label='Fault Distance')
            ax3.set_xlabel('X Coordinate')
            ax3.set_ylabel('Y Coordinate')
            ax3.set_title('Distance to Nearest Fault')
        
        # 4. Lithology distribution
        if 'lithology' in df.columns:
            ax4 = axes[1, 1]
            # Get unique lithologies and assign colors
            lithologies = df['lithology'].unique()
            colors = plt.cm.Set3(np.linspace(0, 1, len(lithologies)))
            
            for idx, lith in enumerate(lithologies[:10]):  # Limit to 10 for clarity
                mask = df['lithology'] == lith
                ax4.scatter(df.loc[mask, 'xcoord'], df.loc[mask, 'ycoord'], 
                          c=[colors[idx]], label=lith[:20], s=30, alpha=0.6)
            
            ax4.set_xlabel('X Coordinate')
            ax4.set_ylabel('Y Coordinate')
            ax4.set_title('Lithology Distribution')
            ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.suptitle('Spatial Analysis of Geological Data', fontsize=16)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'spatial_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance_by_group(self, importance_df: pd.DataFrame, 
                                       model_name: str):
        """
        Plot feature importance grouped by geological categories.
        
        Parameters:
        -----------
        importance_df : pd.DataFrame
            Feature importance dataframe
        model_name : str
            Name of the model
        """
        if importance_df.empty:
            return
        
        # Categorize features
        feature_categories = {}
        for feature in importance_df['feature']:
            categorized = False
            for category, features in GEOLOGICAL_FEATURES.items():
                if feature in features:
                    feature_categories[feature] = category
                    categorized = True
                    break
            if not categorized:
                if any(keyword in feature for keyword in ['index', 'ratio', 'anomaly']):
                    feature_categories[feature] = 'engineered'
                else:
                    feature_categories[feature] = 'other'
        
        # Add category to dataframe
        importance_df['category'] = importance_df['feature'].map(feature_categories)
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        # Get unique categories and assign colors
        categories = importance_df['category'].unique()
        colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))
        color_map = dict(zip(categories, colors))
        
        # Create grouped bar plot
        top_features = importance_df.head(20)
        bar_colors = [color_map[cat] for cat in top_features['category']]
        
        plt.barh(top_features['feature'], top_features['importance'], 
                color=bar_colors)
        
        # Create legend
        patches = [mpatches.Patch(color=color_map[cat], label=cat.replace('_', ' ').title()) 
                  for cat in categories if cat in top_features['category'].values]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.xlabel('Feature Importance')
        plt.title(f'Feature Importance by Geological Category - {model_name}', fontsize=14)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f'feature_importance_grouped_{model_name}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # def create_prospectivity_map_preview(self, predictions: np.ndarray, 
    #                                    coordinates: pd.DataFrame):
    #     """
    #     Create a preview of prospectivity map.
        
    #     Parameters:
    #     -----------
    #     predictions : np.ndarray
    #         Predicted probabilities
    #     coordinates : pd.DataFrame
    #         Coordinate data
    #     """
    #     if not all(col in coordinates.columns for col in ['xcoord', 'ycoord']):
    #         return
        
    #     plt.figure(figsize=(10, 8))
        
    #     # Create scatter plot with predictions
    #     scatter = plt.scatter(coordinates['xcoord'], coordinates['ycoord'], 
    #                         c=predictions, cmap=self.geological_colors, 
    #                         s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
        
    #     plt.colorbar(scatter, label='Prospectivity Score')
    #     plt.xlabel('X Coordinate')
    #     plt.ylabel('Y Coordinate')
    #     plt.title('Gold Prospectivity Map Preview', fontsize=16)
        
    #     # Add contour lines
    #     try:
    #         from scipy.interpolate import griddata
            
    #         # Create grid
    #         xi = np.linspace(coordinates['xcoord'].min(), coordinates['xcoord'].max(), 100)
    #         yi = np.linspace(coordinates['ycoord'].min(), coordinates['ycoord'].max(), 100)
    #         xi, yi = np.meshgrid(xi, yi)
            
    #         # Interpolate
    #         zi = griddata((coordinates['xcoord'], coordinates['ycoord']), 
    #                      predictions, (xi, yi), method='linear')
            
    #         # Add contours
    #         contours = plt.contour(xi, yi, zi, levels=[0.3, 0.5, 0.7, 0.9], 
    #                              colors='black', alpha=0.4, linewidths=1)
    #         plt.clabel(contours, inline=True, fontsize=8)
    #     except:
    #         pass
        
    #     plt.tight_layout()
    #     plt.savefig(FIGURES_DIR / 'prospectivity_map_preview.png', dpi=300, bbox_inches='tight')
    #     plt.close()


    def create_prospectivity_map_preview(self, predictions: np.ndarray, coordinates: pd.DataFrame):
        """
        Creates, styles, and saves a professional geological prospectivity map.

        This function generates a filled contour plot (heatmap) to visualize
        prospectivity as a continuous surface. It overlays original sample
        locations and clear contour lines to provide context and highlight key zones.

        Parameters:
        -----------
        predictions : np.ndarray
            Predicted probabilities (prospectivity scores), expected to be between 0 and 1.
        coordinates : pd.DataFrame
            DataFrame with 'xcoord' and 'ycoord' columns.

        Returns:
        --------
        Optional[Path]
            The path to the saved image file on success, otherwise None.
        """

        print(f"predictions: {predictions}")
        print(f"coordinates: {coordinates}")
        # --- 1. Input Validation ---
        if not all(col in coordinates.columns for col in ['xcoord', 'ycoord']):
            print("Warning: Coordinate data must include 'xcoord' and 'ycoord' columns.")
            return None
        if predictions.size != len(coordinates):
            print("Warning: Mismatch between number of predictions and coordinates.")
            return None
        if predictions.size == 0:
            print("Warning: Prediction or coordinate data is empty.")
            return None

        # --- 2. Interpolation for Continuous Surface ---
        grid_resolution: complex = 900j
        interpolation_method: str = 'cubic'
        fallback_interpolation_method: str = 'linear'
        
        grid_x, grid_y = np.mgrid[
            coordinates['xcoord'].min():coordinates['xcoord'].max():grid_resolution, 
            coordinates['ycoord'].min():coordinates['ycoord'].max():grid_resolution
        ]
        
        try:
            grid_z = griddata(coordinates[['xcoord', 'ycoord']], predictions, (grid_x, grid_y), method=interpolation_method)
        except Exception as e:
            print(f"Warning: '{interpolation_method}' interpolation failed ({e}). Falling back to '{fallback_interpolation_method}'.")
            try:
                grid_z = griddata(coordinates[['xcoord', 'ycoord']], predictions, (grid_x, grid_y), method=fallback_interpolation_method)
            except Exception as e_fallback:
                print(f"Error: Fallback interpolation also failed ({e_fallback}). Cannot generate map.")
                return None
        
        grid_z = np.clip(grid_z, 0, 1) 
        grid_z = np.nan_to_num(grid_z, nan=0.0)
        # print(list(grid_z))
        print(f"grid_z min: {np.nanmin(grid_z)}, max: {np.nanmax(grid_z)}")

        print("Type:", type(grid_z))

        # --- 3. Plotting ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 10))

        # prepare contour levels from only the valid (non-nan) cells
        valid_z = grid_z[~np.isnan(grid_z)]
        vmin, vmax = valid_z.min(), valid_z.max()
        levels = np.linspace(vmin, vmax, 21)

        try:
            # Plot the main heatmap
            colormap: str = 'YlOrRd'

            # norm = plt.Normalize(vmin=vmin, vmax=vmax)

            contourf_plot = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap=colormap, extend='max')

            # Add and configure the color bar
            cbar = fig.colorbar(contourf_plot, ax=ax, orientation='vertical', pad=0.02)
            cbar.set_label('Gold Prospectivity Score', fontsize=12, weight='bold')

            # Overlay contour lines for key thresholds
            contour_levels: List[float] = [0.5, 0.75, 0.9]
            contour_plot = ax.contour(grid_x, grid_y, grid_z, levels=contour_levels, 
                                    colors='black', linewidths=0.75, linestyles='--')
            ax.clabel(contour_plot, inline=True, fontsize=9, fmt='%1.2f')

            # Plot original sample locations for context
            ax.scatter(coordinates['xcoord'], coordinates['ycoord'], c='black', s=5, 
                    alpha=0.5, label='Sample Locations')

            # Final styling
            ax.set_title('Gold Prospectivity Map', fontsize=18, weight='bold', pad=15)
            ax.set_xlabel('Easting (X Coordinate)', fontsize=12)
            ax.set_ylabel('Northing (Y Coordinate)', fontsize=12)
            ax.set_aspect('equal', adjustable='box')
            ax.legend(loc='upper left')

            # --- 4. Save and Return Path ---
            output_path = FIGURES_DIR / 'gold_prospectivity_map.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
            print(f"Prospectivity map successfully saved to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"An unexpected error occurred during plotting: {e}")
            return None
        finally:
            plt.close(fig) # Ensure figure is always closed to free memory


    
    def create_exploration_priority_plot(self, df: pd.DataFrame, predictions: np.ndarray):
        """
        Create exploration priority visualization.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with features
        predictions : np.ndarray
            Predicted probabilities
        """
        # Create priority categories
        priority_categories = pd.cut(predictions, 
                                   bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0],
                                   labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Priority distribution
        priority_counts = priority_categories.value_counts()
        colors = ['#2b83ba', '#5aae61', '#ffffbf', '#fdb863', '#b2182b']
        ax1.pie(priority_counts, labels=priority_counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax1.set_title('Exploration Priority Distribution')
        
        # 2. Priority vs key indicators
        if 'Au_ppb' in df.columns:
            priority_df = pd.DataFrame({
                'Priority': priority_categories,
                'Au_ppb': df['Au_ppb'],
                'Prospectivity_Score': predictions
            })
            
            # Box plot of Au concentration by priority
            sns.boxplot(data=priority_df, x='Priority', y='Au_ppb', ax=ax2)
            ax2.set_yscale('log')
            ax2.set_ylabel('Au (ppb) - Log Scale')
            ax2.set_title('Gold Concentration by Exploration Priority')
        
        plt.suptitle('Exploration Priority Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'exploration_priority.png', dpi=300, bbox_inches='tight')
        plt.close()