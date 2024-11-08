import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import datetime
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering 
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import KNNImputer, SimpleImputer

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.stats import chi2_contingency, f_oneway, kruskal
from scipy.spatial.distance import pdist, squareform

import umap
from kneed import KneeLocator
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('cluster_analysis.log')
    ]
)

@dataclass
class ClusteringResults:
    """Data class to store clustering results"""
    labels: np.ndarray
    metrics: Dict[str, float]
    model: Any
    timestamp: str
    parameters: Dict[str, Any]
    validation_scores: Dict[str, List[float]]
    cluster_centers: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None

class ClusterAnalysis:
    def __init__(self, file_path: str):
        """Initialize cluster analysis"""
        self.file_path = Path(file_path)
        self.data = None
        self.data_normalized = None
        self.cluster_labels = None
        self.pca_results = None
        self.umap_results = None 
        self.tsne_results = None
        self.clustering_history = []
        self.feature_columns = [
            'News_1', 'News_frequency',
            'Trad_News_print', 'Trad_News_online', 'Trad_News_TV', 'Trad_News_radio',
            'SM_News_1', 'SM_News_2', 'SM_News_3', 'SM_News_4', 'SM_News_5', 'SM_News_6',
            'SM_Sharing'
        ]

    def load_data(self) -> pd.DataFrame:
        """Load and validate survey data"""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File {self.file_path} does not exist.")
        
        try:
            self.data = pd.read_csv(self.file_path)
            logging.info(f"Data loaded successfully with shape {self.data.shape}")
            return self.data
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def clean_data(self) -> pd.DataFrame:
        """Clean and preprocess survey data"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Handle missing values
        imputer = KNNImputer(n_neighbors=5)
        self.data[self.feature_columns] = imputer.fit_transform(self.data[self.feature_columns])

        # Convert categorical columns to numeric
        le = LabelEncoder()
        for col in self.feature_columns:
            if self.data[col].dtype == 'object':
                self.data[col] = le.fit_transform(self.data[col])

        # Remove outliers using Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_mask = iso_forest.fit_predict(self.data[self.feature_columns]) == 1
        self.data = self.data[outlier_mask]

        logging.info(f"Data cleaned. New shape: {self.data.shape}")
        return self.data

    def normalize_features(self, method: str = 'standard') -> np.ndarray:
        """Normalize features using specified method"""
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Invalid normalization method")

        self.data_normalized = scaler.fit_transform(self.data[self.feature_columns])
        return self.data_normalized

    def reduce_dimensions(self, method: str = 'pca', n_components: int = 2) -> np.ndarray:
        """Reduce dimensionality using specified method"""
        if method == 'pca':
            reducer = PCA(n_components=n_components)
            self.pca_results = reducer.fit_transform(self.data_normalized)
            return self.pca_results
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42)
            self.tsne_results = reducer.fit_transform(self.data_normalized)
            return self.tsne_results
        elif method == 'umap':
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            self.umap_results = reducer.fit_transform(self.data_normalized)
            return self.umap_results
        else:
            raise ValueError("Invalid dimension reduction method")

    def find_optimal_clusters(self, max_clusters: int = 15) -> Dict[str, Any]:
        """Find optimal number of clusters using multiple methods"""
        results = {}
        
        # Elbow method using KMeans
        inertias = []
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.data_normalized)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point
        kl = KneeLocator(range(2, max_clusters + 1), inertias, curve='convex', direction='decreasing')
        results['elbow_k'] = kl.elbow

        # Silhouette analysis
        silhouette_scores = []
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(self.data_normalized)
            score = silhouette_score(self.data_normalized, labels)
            silhouette_scores.append(score)
        
        results['silhouette_k'] = silhouette_scores.index(max(silhouette_scores)) + 2

        # Gap statistic
        gap_stats = []
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(self.data_normalized)
            score = calinski_harabasz_score(self.data_normalized, labels)
            gap_stats.append(score)
            
        results['gap_k'] = gap_stats.index(max(gap_stats)) + 2

        # Plot results
        self._plot_cluster_metrics(inertias, silhouette_scores, gap_stats)
        
        return results

    def perform_clustering(self, method: str = 'kmeans', n_clusters: int = None, **kwargs) -> ClusteringResults:
        """Perform clustering using specified method"""
        if n_clusters is None:
            optimal_clusters = self.find_optimal_clusters()
            n_clusters = optimal_clusters['silhouette_k']

        if method == 'kmeans':
            model = KMeans(n_clusters=n_clusters, **kwargs)
        elif method == 'agglomerative':
            model = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
        elif method == 'dbscan':
            model = DBSCAN(**kwargs)
        elif method == 'spectral':
            model = SpectralClustering(n_clusters=n_clusters, **kwargs)
        elif method == 'gaussian_mixture':
            model = GaussianMixture(n_components=n_clusters, **kwargs)
        else:
            raise ValueError("Invalid clustering method")

        # Fit model and get labels
        labels = model.fit_predict(self.data_normalized)
        self.cluster_labels = labels

        # Calculate validation metrics
        metrics = {
            'silhouette': silhouette_score(self.data_normalized, labels),
            'calinski_harabasz': calinski_harabasz_score(self.data_normalized, labels),
            'davies_bouldin': davies_bouldin_score(self.data_normalized, labels)
        }

        # Get cluster centers if available
        centers = getattr(model, 'cluster_centers_', None)

        # Create results object
        results = ClusteringResults(
            labels=labels,
            metrics=metrics,
            model=model,
            timestamp=datetime.datetime.now().isoformat(),
            parameters=kwargs,
            validation_scores={'cross_val': cross_val_score(model, self.data_normalized, labels)},
            cluster_centers=centers
        )

        self.clustering_history.append(results)
        return results

    def analyze_clusters(self, output_dir: str = 'cluster_analysis') -> None:
        """Analyze and visualize clustering results"""
        os.makedirs(output_dir, exist_ok=True)

        # Add cluster labels to original data
        data_with_clusters = self.data.copy()
        data_with_clusters['Cluster'] = self.cluster_labels

        # Generate cluster profiles
        profiles = self._generate_cluster_profiles(data_with_clusters)
        profiles.to_csv(os.path.join(output_dir, 'cluster_profiles.csv'))

        # Create visualizations
        self._plot_cluster_distributions(data_with_clusters, output_dir)
        self._plot_feature_importance(data_with_clusters, output_dir)
        self._plot_cluster_correlations(data_with_clusters, output_dir)
        self._create_interactive_visualizations(data_with_clusters, output_dir)

        # Statistical analysis
        stats = self._perform_statistical_tests(data_with_clusters)
        with open(os.path.join(output_dir, 'statistical_analysis.json'), 'w') as f:
            json.dump(stats, f, indent=4)

    def _generate_cluster_profiles(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate detailed profiles for each cluster"""
        profiles = []
        for cluster in range(len(np.unique(self.cluster_labels))):
            cluster_data = data[data['Cluster'] == cluster]
            profile = {
                'Cluster': cluster,
                'Size': len(cluster_data),
                'Percentage': len(cluster_data) / len(data) * 100
            }
            
            # Add feature statistics
            for col in self.feature_columns:
                profile[f'{col}_mean'] = cluster_data[col].mean()
                profile[f'{col}_std'] = cluster_data[col].std()
                profile[f'{col}_median'] = cluster_data[col].median()
            
            profiles.append(profile)
            
        return pd.DataFrame(profiles)

    def _plot_cluster_metrics(self, inertias: List[float], silhouette_scores: List[float], 
                            gap_stats: List[float]) -> None:
        """Plot cluster evaluation metrics"""
        fig = make_subplots(rows=1, cols=3, 
                           subplot_titles=('Elbow Method', 'Silhouette Score', 'Gap Statistic'))
        
        k_range = list(range(2, len(inertias) + 2))
        
        # Elbow plot
        fig.add_trace(go.Scatter(x=k_range, y=inertias, mode='lines+markers'), row=1, col=1)
        
        # Silhouette plot
        fig.add_trace(go.Scatter(x=k_range, y=silhouette_scores, mode='lines+markers'), row=1, col=2)
        
        # Gap statistic plot
        fig.add_trace(go.Scatter(x=k_range, y=gap_stats, mode='lines+markers'), row=1, col=3)
        
        fig.update_layout(height=400, width=1200, title_text="Cluster Evaluation Metrics")
        fig.write_html('cluster_metrics.html')

    def _plot_cluster_distributions(self, data: pd.DataFrame, output_dir: str) -> None:
        """Plot distribution of features across clusters"""
        for feature in self.feature_columns:
            fig = px.box(data, x='Cluster', y=feature, title=f'{feature} Distribution by Cluster')
            fig.write_html(os.path.join(output_dir, f'{feature}_distribution.html'))

    def _plot_feature_importance(self, data: pd.DataFrame, output_dir: str) -> None:
        """Plot feature importance for clustering"""
        importances = {}
        for feature in self.feature_columns:
            f_stat, p_value = f_oneway(*[group[feature].values 
                                       for name, group in data.groupby('Cluster')])
            importances[feature] = -np.log10(p_value)

        fig = px.bar(x=list(importances.keys()), y=list(importances.values()),
                    title='Feature Importance (-log10(p-value))')
        fig.write_html(os.path.join(output_dir, 'feature_importance.html'))

    def _plot_cluster_correlations(self, data: pd.DataFrame, output_dir: str) -> None:
        """Plot correlation matrix for each cluster"""
        for cluster in data['Cluster'].unique():
            cluster_data = data[data['Cluster'] == cluster]
            corr_matrix = cluster_data[self.feature_columns].corr()
            
            fig = px.imshow(corr_matrix, title=f'Correlation Matrix - Cluster {cluster}')
            fig.write_html(os.path.join(output_dir, f'correlation_matrix_cluster_{cluster}.html'))

    def _create_interactive_visualizations(self, data: pd.DataFrame, output_dir: str) -> None:
        """Create interactive visualizations of clustering results"""
        # 3D scatter plot using PCA
        pca = PCA(n_components=3)
        pca_results = pca.fit_transform(self.data_normalized)
        
        fig = px.scatter_3d(
            data_frame=pd.DataFrame(pca_results, columns=['PC1', 'PC2', 'PC3']),
            x='PC1', y='PC2', z='PC3',
            color=data['Cluster'].astype(str),
            title='3D Cluster Visualization (PCA)'
        )
        fig.write_html(os.path.join(output_dir, 'cluster_3d_visualization.html'))

    def _perform_statistical_tests(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical tests on clusters"""
        stats = {}
        
        # Chi-square tests for categorical variables
        for feature in self.feature_columns:
            contingency_table = pd.crosstab(data['Cluster'], data[feature])
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            stats[feature] = {
                'chi2': chi2,
                'p_value': p_value,
                'dof': dof
            }
            
        # Kruskal-Wallis H-test for continuous variables
        for feature in self.feature_columns:
            h_stat, p_value = kruskal(*[group[feature].values 
                                      for name, group in data.groupby('Cluster')])
            stats[f'{feature}_kruskal'] = {
                'h_statistic': h_stat,
                'p_value': p_value
            }
            
        return stats

def main():
    # Initialize analysis
    analyzer = ClusterAnalysis('survey_data.csv')
    
    # Load and prepare data
    analyzer.load_data()
    analyzer.clean_data()
    analyzer.normalize_features(method='robust')
    
    # Perform dimensionality reduction
    analyzer.reduce_dimensions(method='umap')
    
    # Find optimal clusters
    optimal_clusters = analyzer.find_optimal_clusters(max_clusters=15)
    
    # Perform clustering with multiple methods
    results_kmeans = analyzer.perform_clustering(method='kmeans', 
                                               n_clusters=optimal_clusters['silhouette_k'])
    results_spectral = analyzer.perform_clustering(method='spectral', 
                                                 n_clusters=optimal_clusters['silhouette_k'])
    results_agglomerative = analyzer.perform_clustering(method='agglomerative', 
                                                      n_clusters=optimal_clusters['silhouette_k'])
    
    # Analyze best clustering results (using highest silhouette score)
    best_results = max([results_kmeans, results_spectral, results_agglomerative],
                      key=lambda x: x.metrics['silhouette'])
    analyzer.cluster_labels = best_results.labels
    
    # Generate analysis
    analyzer.analyze_clusters(output_dir='cluster_analysis_results')
    
    logging.info("Cluster analysis completed successfully")

if __name__ == "__main__":
    main()