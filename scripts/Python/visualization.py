import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class ClusterVisualizer:
    """Class for creating cluster visualizations"""
    def __init__(self, save_dir: str = "results/plots"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_cluster_distribution(self, 
                                data: pd.DataFrame,
                                labels: np.ndarray,
                                features: List[str],
                                title: str = "Cluster Distributions") -> None:
        """Create distribution plots for each feature by cluster"""
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig = make_subplots(rows=n_rows, cols=n_cols,
                           subplot_titles=features)
        
        for i, feature in enumerate(features):
            row = i // n_cols + 1
            col = i % n_cols + 1
            
            for cluster in np.unique(labels):
                cluster_data = data[labels == cluster][feature]
                
                fig.add_trace(
                    go.Violin(x=[f"Cluster {cluster}"] * len(cluster_data),
                             y=cluster_data,
                             name=f"Cluster {cluster}",
                             showlegend=i==0),
                    row=row, col=col
                )
        
        fig.update_layout(height=300*n_rows, title_text=title)
        fig.write_html(self.save_dir / "cluster_distributions.html")
    
    def plot_cluster_profiles(self,
                            profiles: pd.DataFrame,
                            title: str = "Cluster Profiles") -> None:
        """Create radar plots for cluster profiles"""
        features = profiles.columns[1:]  # Exclude cluster column
        n_clusters = len(profiles)
        
        fig = go.Figure()
        
        for cluster in range(n_clusters):
            values = profiles.iloc[cluster][features].values.tolist()
            values.append(values[0])  # Close the polygon
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=list(features) + [features[0]],
                name=f"Cluster {cluster}"
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title=title
        )
        
        fig.write_html(self.save_dir / "cluster_profiles.html")
    
    def plot_dimensionality_reduction(self,
                                    embeddings: np.ndarray,
                                    labels: np.ndarray,
                                    method: str = "PCA",
                                    title: Optional[str] = None) -> None:
        """Create scatter plot of dimensionality reduction results"""
        if embeddings.shape[1] == 2:
            fig = px.scatter(
                x=embeddings[:, 0],
                y=embeddings[:, 1],
                color=labels.astype(str),
                title=title or f"{method} Visualization of Clusters"
            )
        elif embeddings.shape[1] == 3:
            fig = px.scatter_3d(
                x=embeddings[:, 0],
                y=embeddings[:, 1],
                z=embeddings[:, 2],
                color=labels.astype(str),
                title=title or f"{method} Visualization of Clusters"
            )
        else:
            raise ValueError("Embeddings must be 2D or 3D")
        
        fig.write_html(self.save_dir / f"{method.lower()}_visualization.html") 