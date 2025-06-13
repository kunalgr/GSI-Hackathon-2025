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
from scipy.interpolate import griddata

from utils.config import FIGURES_DIR, GEOLOGICAL_FEATURES


class GeologicalVisualizer:
    """
    Class for creating geological data visualizations.
    """
    
    def __init__(self):
        # Reset matplotlib to defaults to avoid style conflicts
        plt.rcdefaults()
        # Use a clean style
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            plt.style.use('default')
        # Set white background
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['savefig.facecolor'] = 'white'
        
        self.geological_colors = self._create_geological_colormap()
        
    def _create_geological_colormap(self):
        """
        Create custom colormap for geological data.
        """
        # Use colors from blue (low) to red (high) for prospectivity
        colors = ['#2166ac', '#4393c3', '#92c5de', '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('geological_prospectivity', colors, N=n_bins)
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
    
    def create_prospectivity_map_preview(self, predictions: np.ndarray,
                                        coordinates: pd.DataFrame):
        """
        Creates, styles, and saves a professional geological prospectivity map
        using a filled contour plot for smooth surfaces and subtle sample point overlay.
        """
        # --- Validate inputs ---
        if not all(col in coordinates.columns for col in ['xcoord', 'ycoord']):
            print("Warning: Coordinate data must include 'xcoord' and 'ycoord' columns")
            return

        print("\n[CREATING PROSPECTIVITY MAP]")
        print(f"Points: {len(predictions)}, Range: [{predictions.min():.3f}, {predictions.max():.3f}]")

        # Clip to [0,1]
        predictions = np.clip(predictions, 0, 1)

        # --- Prepare grid for interpolation ---
        x = coordinates['xcoord'].values
        y = coordinates['ycoord'].values
        mask = ~np.isnan(predictions)
        xi = np.linspace(x[mask].min(), x[mask].max(), 200)
        yi = np.linspace(y[mask].min(), y[mask].max(), 200)
        XI, YI = np.meshgrid(xi, yi)

        # Interpolate onto grid
        ZI = griddata(
            points=(x[mask], y[mask]),
            values=predictions[mask],
            xi=(XI, YI),
            method='cubic'   # 'linear' or 'cubic'
        )

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
        ax.set_facecolor('white')

        # Filled contours
        contourf = ax.contourf(
            XI, YI, ZI,
            levels=15,                    # more levels = smoother gradient
            cmap='viridis',               # perceptually uniform
            vmin=0, vmax=1,
            alpha=0.9
        )

        # Contour lines
        contours = ax.contour(
            XI, YI, ZI,
            levels=[0.3, 0.5, 0.7, 0.9],
            colors='k',
            linewidths=1,
            alpha=0.5
        )
        ax.clabel(contours, inline=True, fontsize=8, fmt='%.1f')

        # Overlay sample points lightly
        ax.scatter(
            x, y,
            c='white', edgecolor='black',
            s=20, alpha=0.6,
            linewidth=0.5,
            label='Sample locations'
        )

        # Colorbar
        cbar = plt.colorbar(contourf, ax=ax, pad=0.02)
        cbar.set_label('Prospectivity Score', fontsize=12)

        # Labels/title
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.set_title('Gold Prospectivity Map Preview', fontsize=16, pad=20)

        # Grid and aspect
        ax.grid(True, color='gray', alpha=0.2, linestyle='--')
        ax.set_aspect('equal', adjustable='box')

        # Stats textbox
        stats = {
            'High (>0.7)':  (predictions > 0.7).sum(),
            'Moderate (0.5–0.7)': ((predictions >= 0.5) & (predictions <= 0.7)).sum(),
            'Low (<0.5)':   (predictions < 0.5).sum(),
        }
        stats_str = "\n".join(f"{k}: {v}" for k, v in stats.items())
        ax.text(
            0.98, 0.02, stats_str,
            transform=ax.transAxes,
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)
        )

        # Save outputs
        plt.tight_layout()
        png_path = FIGURES_DIR / 'prospectivity_map_preview.png'
        pdf_path = FIGURES_DIR / 'prospectivity_map_preview.pdf'
        for path in (png_path, pdf_path):
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        print(f"Prospectivity map saved as:\n • {png_path}\n • {pdf_path}")
    
    def _idw_interpolation(self, x, y, z, xi, yi, power=2):
        """
        Inverse Distance Weighting (IDW) interpolation.
        
        Parameters:
        -----------
        x, y : array-like
            Coordinates of data points
        z : array-like
            Values at data points
        xi, yi : 2D arrays
            Grid coordinates for interpolation
        power : float
            Power parameter for IDW (default=2)
            
        Returns:
        --------
        zi : 2D array
            Interpolated values on grid
        """
        zi = np.zeros_like(xi)
        
        for i in range(xi.shape[0]):
            for j in range(xi.shape[1]):
                # Calculate distances from grid point to all data points
                distances = np.sqrt((xi[i, j] - x)**2 + (yi[i, j] - y)**2)
                
                # Avoid division by zero
                distances[distances == 0] = 1e-10
                
                # Calculate weights (inverse distance)
                weights = 1 / distances**power
                
                # Normalize weights
                weights /= weights.sum()
                
                # Calculate interpolated value
                zi[i, j] = np.sum(weights * z)
        
        return zi
    
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
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
        
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
                'Au_ppb': df['Au_ppb'].values,
                'Prospectivity_Score': predictions
            })
            
            # Box plot of Au concentration by priority
            sns.boxplot(data=priority_df, x='Priority', y='Au_ppb', ax=ax2)
            ax2.set_yscale('log')
            ax2.set_ylabel('Au (ppb) - Log Scale')
            ax2.set_title('Gold Concentration by Exploration Priority')
            ax2.set_facecolor('white')
        
        plt.suptitle('Exploration Priority Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'exploration_priority.png', 
                   dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()