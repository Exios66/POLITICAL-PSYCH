import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import traceback

logger = logging.getLogger(__name__)

class ClusterVisualizer:
    """Class for creating interactive cluster visualizations and statistical plots"""
    def __init__(self, save_dir: Union[str, Path] = "results/plots"):
        """
        Initialize the ClusterVisualizer.
        
        Args:
            save_dir: Directory path to save visualization outputs
            
        Raises:
            OSError: If directory creation fails
        """
        try:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initialized ClusterVisualizer with save directory: {self.save_dir}")
        except Exception as e:
            logger.error(f"Failed to create save directory: {str(e)}")
            raise OSError(f"Could not create save directory: {str(e)}")
    
    def plot_cluster_distribution(self, 
                                data: pd.DataFrame,
                                labels: np.ndarray,
                                features: List[str],
                                title: str = "Cluster Distributions",
                                height_per_row: int = 300) -> None:
        """
        Create interactive distribution plots for each feature by cluster.
        
        Args:
            data: DataFrame containing feature data
            labels: Cluster labels for each data point
            features: List of feature names to plot
            title: Main title for the plot
            height_per_row: Height in pixels for each row of plots
            
        Raises:
            ValueError: If input dimensions don't match or features not found
        """
        try:
            # Input validation
            if len(data) != len(labels):
                raise ValueError("Data and labels must have same length")
            if not all(feature in data.columns for feature in features):
                raise ValueError("Not all features found in data")
                
            n_features = len(features)
            n_cols = min(3, n_features)  # Adjust columns based on feature count
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig = make_subplots(rows=n_rows, cols=n_cols,
                               subplot_titles=features)
            
            unique_clusters = np.unique(labels)
            colors = px.colors.qualitative.Set3[:len(unique_clusters)]
            
            for i, feature in enumerate(features):
                row = i // n_cols + 1
                col = i % n_cols + 1
                
                for cluster_idx, cluster in enumerate(unique_clusters):
                    cluster_data = data[labels == cluster][feature]
                    
                    if len(cluster_data) == 0:
                        logger.warning(f"No data for cluster {cluster} in feature {feature}")
                        continue
                        
                    fig.add_trace(
                        go.Violin(x=[f"Cluster {cluster}"] * len(cluster_data),
                                 y=cluster_data,
                                 name=f"Cluster {cluster}",
                                 showlegend=i==0,
                                 line_color=colors[cluster_idx]),
                        row=row, col=col
                    )
            
            fig.update_layout(
                height=height_per_row*n_rows,
                title_text=title,
                showlegend=True,
                hovermode='closest'
            )
            
            output_path = self.save_dir / "cluster_distributions.html"
            fig.write_html(str(output_path))
            logger.info(f"Saved cluster distribution plot to {output_path}")
            
        except Exception as e:
            logger.error(f"Error in plot_cluster_distribution: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def plot_cluster_profiles(self,
                            profiles: pd.DataFrame,
                            title: str = "Cluster Profiles",
                            normalize: bool = True) -> None:
        """
        Create interactive radar plots showing cluster profiles.
        
        Args:
            profiles: DataFrame with cluster profiles
            title: Plot title
            normalize: Whether to normalize values to [0,1] range
            
        Raises:
            ValueError: If profiles DataFrame is invalid
        """
        try:
            if 'cluster' not in profiles.columns:
                raise ValueError("Profiles must have 'cluster' column")
                
            features = [col for col in profiles.columns if col != 'cluster']
            if not features:
                raise ValueError("No feature columns found in profiles")
                
            if normalize:
                for feature in features:
                    profiles[feature] = (profiles[feature] - profiles[feature].min()) / \
                                      (profiles[feature].max() - profiles[feature].min())
            
            fig = go.Figure()
            colors = px.colors.qualitative.Set3[:len(profiles)]
            
            for idx, (_, row) in enumerate(profiles.iterrows()):
                values = row[features].values.tolist()
                values.append(values[0])  # Close the polygon
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=features + [features[0]],
                    name=f"Cluster {int(row['cluster'])}",
                    line_color=colors[idx],
                    fill='toself'
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1] if normalize else None
                    )
                ),
                showlegend=True,
                title=title
            )
            
            output_path = self.save_dir / "cluster_profiles.html"
            fig.write_html(str(output_path))
            logger.info(f"Saved cluster profiles plot to {output_path}")
            
        except Exception as e:
            logger.error(f"Error in plot_cluster_profiles: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def plot_dimensionality_reduction(self,
                                    embeddings: np.ndarray,
                                    labels: np.ndarray,
                                    method: str = "PCA",
                                    title: Optional[str] = None,
                                    point_size: int = 5) -> None:
        """
        Create interactive scatter plot of dimensionality reduction results.
        
        Args:
            embeddings: 2D or 3D embedding coordinates
            labels: Cluster labels
            method: Name of dimensionality reduction method
            title: Plot title
            point_size: Size of scatter plot points
            
        Raises:
            ValueError: If embeddings dimensions invalid
        """
        try:
            if embeddings.shape[0] != len(labels):
                raise ValueError("Number of embeddings must match number of labels")
                
            if embeddings.shape[1] == 2:
                fig = px.scatter(
                    x=embeddings[:, 0],
                    y=embeddings[:, 1],
                    color=labels.astype(str),
                    title=title or f"{method} Visualization of Clusters",
                    labels={'x': f'{method} 1', 'y': f'{method} 2'},
                    color_discrete_sequence=px.colors.qualitative.Set3,
                    size=[point_size] * len(labels)
                )
            elif embeddings.shape[1] == 3:
                fig = px.scatter_3d(
                    x=embeddings[:, 0],
                    y=embeddings[:, 1],
                    z=embeddings[:, 2],
                    color=labels.astype(str),
                    title=title or f"{method} Visualization of Clusters",
                    labels={'x': f'{method} 1', 'y': f'{method} 2', 'z': f'{method} 3'},
                    color_discrete_sequence=px.colors.qualitative.Set3,
                    size=[point_size] * len(labels)
                )
            else:
                raise ValueError("Embeddings must be 2D or 3D")
            
            fig.update_layout(
                legend_title="Clusters",
                hovermode='closest'
            )
            
            output_path = self.save_dir / f"{method.lower()}_visualization.html"
            fig.write_html(str(output_path))
            logger.info(f"Saved {method} visualization to {output_path}")
            
        except Exception as e:
            logger.error(f"Error in plot_dimensionality_reduction: {str(e)}")
            logger.error(traceback.format_exc())
            raise