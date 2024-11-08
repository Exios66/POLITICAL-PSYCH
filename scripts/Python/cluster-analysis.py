import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.stats import chi2_contingency, f_oneway, kruskal
from scipy.spatial.distance import pdist, squareform
import warnings
from sklearn.impute import KNNImputer, SimpleImputer
from textblob import TextBlob
import re
from wordcloud import WordCloud
import logging
import sys
import json
import datetime
import os
from typing import Dict, List, Tuple, Any, Optional, Union
import traceback
from dataclasses import dataclass
import pickle
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import umap
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

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

class ElectionSurveyAnalysis:
    def __init__(self, file_path: str, log_level: int = logging.INFO):
        """
        Initialize the analysis with enhanced logging and error handling
        
        Args:
            file_path (str): Path to the survey data file
            log_level (int): Logging level (default: logging.INFO)
        """
        self.file_path = Path(file_path)
        self.data = None
        self.data_normalized = None
        self.cluster_labels = None
        self.pca_results = None
        self.umap_results = None
        self.tsne_results = None
        self.clustering_history = []
        self.feature_importance = {}
        self.outliers_mask = None
        
        # Analysis parameters
        self.random_state = 42
        self.cv_folds = 5
        
        # Set up logging
        self.setup_logging(log_level)
        
        # Create results directory with timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = Path(f"analysis_results_{timestamp}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.plots_dir = self.results_dir / "plots"
        self.models_dir = self.results_dir / "models"
        self.reports_dir = self.results_dir / "reports"
        
        for dir_path in [self.plots_dir, self.models_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Analysis initialized with file: {file_path}")
        
    def setup_logging(self, log_level: int) -> None:
        """Configure logging with both file and console handlers"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # Prevent duplicate logging
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create formatters and handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler with timestamp
        log_file = f'election_analysis_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        fh = logging.FileHandler(self.results_dir / log_file)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def error_handler(func):
        """Decorator for error handling and logging"""
        def wrapper(self, *args, **kwargs):
            try:
                self.logger.debug(f"Starting {func.__name__} with args: {args}, kwargs: {kwargs}")
                result = func(self, *args, **kwargs)
                self.logger.debug(f"Successfully completed {func.__name__}")
                return result
            except Exception as e:
                self.logger.error(f"Error in {func.__name__}: {str(e)}")
                self.logger.error(traceback.format_exc())
                raise
        return wrapper

    @error_handler
    def load_data(self) -> pd.DataFrame:
        """Load and perform initial data validation and cleaning"""
        self.logger.info("Loading data...")
        
        try:
            self.data = pd.read_csv(self.file_path)
            
            # Validate data structure
            if self.data.empty:
                raise ValueError("Empty dataset loaded")
            
            # Advanced cleaning
            # Clean column names
            self.data.columns = [self._clean_column_name(col) for col in self.data.columns]
            
            # Remove completely empty columns and rows
            self.data.dropna(axis=1, how='all', inplace=True)
            self.data.dropna(axis=0, how='all', inplace=True)
            
            # Check for duplicate entries
            duplicates = self.data.duplicated()
            if duplicates.sum() > 0:
                self.logger.warning(f"Found {duplicates.sum()} duplicate entries")
                duplicate_report = self.data[duplicates].to_dict('records')
                with open(self.reports_dir / "duplicate_entries.json", 'w') as f:
                    json.dump(duplicate_report, f, indent=4)
            
            # Generate comprehensive data quality report
            self.generate_data_quality_report()
            
            # Save raw data snapshot
            self.data.to_csv(self.results_dir / "raw_data_snapshot.csv", index=False)
            
            self.logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise

    def _clean_column_name(self, column: str) -> str:
        """Clean column names to be database friendly"""
        # Convert to lowercase and replace spaces/special chars with underscore
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', column.lower().strip())
        # Remove consecutive underscores
        clean_name = re.sub(r'_+', '_', clean_name)
        # Remove leading/trailing underscores
        return clean_name.strip('_')

    def generate_data_quality_report(self) -> None:
        """Generate comprehensive data quality report"""
        report = {
            'basic_info': {
                'rows': len(self.data),
                'columns': len(self.data.columns),
                'total_cells': self.data.size,
                'memory_usage': self.data.memory_usage(deep=True).sum()
            },
            'missing_values': {
                'total': self.data.isnull().sum().sum(),
                'by_column': self.data.isnull().sum().to_dict(),
                'percentage': (self.data.isnull().sum() / len(self.data) * 100).to_dict()
            },
            'unique_values': {
                col: {
                    'count': self.data[col].nunique(),
                    'top_5': self.data[col].value_counts().nlargest(5).to_dict()
                } for col in self.data.columns
            },
            'data_types': self.data.dtypes.to_dict(),
            'descriptive_stats': self._generate_descriptive_stats(),
            'correlations': self._generate_correlation_report()
        }
        
        # Save report
        with open(self.reports_dir / "data_quality_report.json", 'w') as f:
            json.dump(report, f, indent=4, default=str)
        
        # Generate visualizations
        self._generate_data_quality_visualizations()

    def _generate_descriptive_stats(self) -> Dict[str, Any]:
        """Generate descriptive statistics for numerical columns"""
        numeric_data = self.data.select_dtypes(include=['int64', 'float64'])
        stats = numeric_data.describe().to_dict()
        
        # Add skewness and kurtosis
        for col in numeric_data.columns:
            stats[col]['skewness'] = float(numeric_data[col].skew())
            stats[col]['kurtosis'] = float(numeric_data[col].kurtosis())
        
        return stats

    def _generate_correlation_report(self) -> Dict[str, Any]:
        """Generate correlation analysis"""
        numeric_data = self.data.select_dtypes(include=['int64', 'float64'])
        correlations = numeric_data.corr().to_dict()
        
        # Find highly correlated features
        high_corr = []
        for i in range(len(numeric_data.columns)):
            for j in range(i+1, len(numeric_data.columns)):
                col1, col2 = numeric_data.columns[i], numeric_data.columns[j]
                corr = abs(numeric_data[col1].corr(numeric_data[col2]))
                if corr > 0.7:  # Threshold for high correlation
                    high_corr.append({
                        'feature1': col1,
                        'feature2': col2,
                        'correlation': corr
                    })
        
        return {
            'correlation_matrix': correlations,
            'high_correlations': high_corr
        }

    def _generate_data_quality_visualizations(self) -> None:
        """Generate visualizations for data quality analysis"""
        # Missing values heatmap
        plt.figure(figsize=(15, 8))
        sns.heatmap(self.data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'missing_values_heatmap.png')
        plt.close()
        
        # Distribution plots for numeric columns
        numeric_data = self.data.select_dtypes(include=['int64', 'float64'])
        for col in numeric_data.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=self.data, x=col, kde=True)
            plt.title(f'Distribution of {col}')
            plt.tight_layout()
            plt.savefig(self.plots_dir / f'distribution_{col}.png')
            plt.close()
        
        # Correlation heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'correlation_heatmap.png')
        plt.close()

    @error_handler
    def preprocess_data(self, 
                       method: str = 'robust', 
                       handle_outliers: bool = True,
                       categorical_encoding: str = 'label') -> np.ndarray:
        """
        Enhanced data preprocessing with multiple scaling options and outlier detection
        
        Args:
            method (str): Scaling method ('standard', 'robust', or 'minmax')
            handle_outliers (bool): Whether to detect and handle outliers
            categorical_encoding (str): Method for encoding categorical variables ('label' or 'onehot')
        """
        self.logger.info("Starting data preprocessing...")
        
        # Store original column names and dtypes
        self.original_columns = self.data.columns.tolist()
        self.original_dtypes = self.data.dtypes.to_dict()
        
        # Separate numeric and categorical columns
        numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        
        # Handle missing values
        self._handle_missing_values(numeric_cols, categorical_cols)
        
        # Encode categorical variables
        if categorical_cols.any():
            self.data = self._encode_categorical_variables(categorical_cols, method=categorical_encoding)
        
        # Scale numeric features
        self.data_normalized = self._scale_features(method)
        
        # Handle outliers if requested
        if handle_outliers:
            self._handle_outliers()
        
        # Dimensionality reduction for visualization
        self._perform_dimensionality_reduction()
        
        # Save preprocessing state
        self.save_state('preprocessing')
        
        # Generate preprocessing report
        self._generate_preprocessing_report(method, handle_outliers, categorical_encoding)
        
        self.logger.info("Data preprocessing completed successfully")
        return self.data_normalized

    def _handle_missing_values(self, numeric_cols: pd.Index, categorical_cols: pd.Index) -> None:
        """Handle missing values with advanced imputation techniques"""
        # For numeric columns, use KNN imputation
        if len(numeric_cols) > 0:
            knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
            self.data[numeric_cols] = knn_imputer.fit_transform(self.data[numeric_cols])
        
        # For categorical columns, use mode imputation with frequency analysis
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                mode_value = self.data[col].mode()[0]
                missing_count = self.data[col].isnull().sum()
                if missing_count > 0:
                    self.logger.info(f"Imputing {missing_count} missing values in {col} with mode: {mode_value}")
                    self.data[col].fillna(mode_value, inplace=True)

    def _encode_categorical_variables(self, categorical_cols: pd.Index, method: str = 'label') -> pd.DataFrame:
        """Encode categorical variables with multiple encoding options"""
        encoded_data = self.data.copy()
        
        if method == 'label':
            self.label_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                encoded_data[col] = le.fit_transform(self.data[col].astype(str))
                self.label_encoders[col] = le
                
        elif method == 'onehot':
            encoded_data = pd.get_dummies(self.data, columns=categorical_cols)
            
        return encoded_data

    def _scale_features(self, method: str) -> np.ndarray:
        """Scale features using specified method"""
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {method}")
        
        self.scaler = scaler
        return scaler.fit_transform(self.data)

    def _handle_outliers(self, contamination: float = 0.1) -> None:
        """Detect and handle outliers using multiple methods"""
        # Isolation Forest for outlier detection
        iso_forest = IsolationForest(contamination=contamination, random_state=self.random_state)
        outlier_labels = iso_forest.fit_predict(self.data_normalized)
        
        # Local Outlier Factor for verification
        from sklearn.neighbors import LocalOutlierFactor
        lof = LocalOutlierFactor(contamination=contamination)
        lof_labels = lof.fit_predict(self.data_normalized)
        
        # Combine results from both methods
        self.outliers_mask = (outlier_labels == -1) | (lof_labels == -1)
        outlier_indices = np.where(self.outliers_mask)[0]
        
        if len(outlier_indices) > 0:
            self.logger.warning(f"Detected {len(outlier_indices)} outliers")
            
            # Generate outlier report
            outlier_report = {
                'total_outliers': len(outlier_indices),
                'outlier_indices': outlier_indices.tolist(),
                'outlier_details': self._analyze_outliers(outlier_indices)
            }
            
            # Save outlier report
            with open(self.reports_dir / "outlier_report.json", 'w') as f:
                json.dump(outlier_report, f, indent=4)
            
            # Visualize outliers
            self._visualize_outliers(outlier_indices)

    def _analyze_outliers(self, outlier_indices: np.ndarray) -> Dict[str, Any]:
        """Analyze detected outliers in detail"""
        outlier_data = self.data.iloc[outlier_indices]
        normal_data = self.data.iloc[~self.outliers_mask]
        
        analysis = {
            'feature_statistics': {},
            'outlier_patterns': {}
        }
        
        # Analyze each feature
        for col in self.data.columns:
            if self.data[col].dtype in ['int64', 'float64']:
                analysis['feature_statistics'][col] = {
                    'outlier_mean': float(outlier_data[col].mean()),
                    'normal_mean': float(normal_data[col].mean()),
                    'outlier_std': float(outlier_data[col].std()),
                    'normal_std': float(normal_data[col].std())
                }
        
        return analysis

    def _visualize_outliers(self, outlier_indices: np.ndarray) -> None:
        """Create visualizations for outlier analysis"""
        # PCA visualization with outliers highlighted
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(self.data_normalized)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(pca_results[~self.outliers_mask, 0], 
                   pca_results[~self.outliers_mask, 1], 
                   c='blue', label='Normal')
        plt.scatter(pca_results[self.outliers_mask, 0], 
                   pca_results[self.outliers_mask, 1], 
                   c='red', label='Outliers')
        plt.title('PCA visualization of outliers')
        plt.legend()
        plt.savefig(self.plots_dir / 'outliers_pca.png')
        plt.close()

    def _perform_dimensionality_reduction(self) -> None:
        """Perform multiple dimensionality reduction techniques"""
        # PCA
        pca = PCA(n_components=2)
        self.pca_results = pca.fit_transform(self.data_normalized)
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=self.random_state)
        self.tsne_results = tsne.fit_transform(self.data_normalized)
        
        # UMAP
        reducer = umap.UMAP(random_state=self.random_state)
        self.umap_results = reducer.fit_transform(self.data_normalized)

    def _generate_preprocessing_report(self, 
                                    scaling_method: str, 
                                    handle_outliers: bool, 
                                    categorical_encoding: str) -> None:
        """Generate comprehensive preprocessing report"""
        report = {
            'preprocessing_parameters': {
                'scaling_method': scaling_method,
                'handle_outliers': handle_outliers,
                'categorical_encoding': categorical_encoding
            },
            'data_shape': {
                'original': self.data.shape,
                'processed': self.data_normalized.shape
            },
            'feature_names': self.original_columns.tolist(),
            'scaling_info': {
                'method': scaling_method,
                'parameters': self.scaler.get_params()
            }
        }
        
        if handle_outliers:
            report['outlier_info'] = {
                'total_outliers': int(self.outliers_mask.sum()),
                'outlier_percentage': float(self.outliers_mask.sum() / len(self.outliers_mask) * 100)
            }
        
        # Save report
        with open(self.reports_dir / "preprocessing_report.json", 'w') as f:
            json.dump(report, f, indent=4)

    @error_handler
    def perform_clustering(self, 
                         max_clusters: int = 10, 
                         methods: List[str] = ['kmeans', 'hierarchical', 'dbscan', 'spectral', 'gaussian_mixture'],
                         n_init: int = 10) -> Dict[str, ClusteringResults]:
        """
        Perform multiple clustering analyses with cross-validation and parameter optimization
        
        Args:
            max_clusters (int): Maximum number of clusters to try
            methods (List[str]): List of clustering methods to use
            n_init (int): Number of initialization runs
        """
        self.logger.info(f"Performing clustering analysis with methods: {methods}")
        
        results = {}
        
        for method in methods:
            self.logger.info(f"Running {method} clustering...")
            
            if method == 'kmeans':
                results[method] = self._perform_kmeans(max_clusters, n_init)
            elif method == 'hierarchical':
                results[method] = self._perform_hierarchical(max_clusters)
            elif method == 'dbscan':
                results[method] = self._perform_dbscan()
            elif method == 'spectral':
                results[method] = self._perform_spectral(max_clusters)
            elif method == 'gaussian_mixture':
                results[method] = self._perform_gaussian_mixture(max_clusters)
            
            # Validate clustering results
            self._validate_clustering(results[method])
            
            # Generate cluster profiles
            self._generate_cluster_profiles(results[method])
        
        # Compare clustering results
        self._compare_clustering_results(results)
        
        # Save clustering results
        self.save_clustering_results(results)
        
        return results

    def _perform_kmeans(self, max_clusters: int, n_init: int) -> ClusteringResults:
        """Perform K-means clustering with extensive validation"""
        # Find optimal number of clusters
        optimal_k = self._find_optimal_clusters(max_clusters)
        
        # Grid search for best parameters
        param_grid = {
            'n_clusters': [optimal_k],
            'init': ['k-means++', 'random'],
            'n_init': [n_init],
            'max_iter': [300, 500]
        }
        
        kmeans = KMeans(random_state=self.random_state)
        grid_search = GridSearchCV(kmeans, param_grid, cv=self.cv_folds, n_jobs=-1)
        grid_search.fit(self.data_normalized)
        
        # Get best model
        best_kmeans = grid_search.best_estimator_
        labels = best_kmeans.labels_
        
        # Calculate metrics
        metrics = self._calculate_clustering_metrics(self.data_normalized, labels)
        
        # Calculate feature importance
        feature_importance = self._calculate_feature_importance_kmeans(best_kmeans)
        
        return ClusteringResults(
            labels=labels,
            metrics=metrics,
            model=best_kmeans,
            timestamp=datetime.datetime.now().isoformat(),
            parameters=grid_search.best_params_,
            validation_scores=self._get_validation_scores(grid_search),
            cluster_centers=best_kmeans.cluster_centers_,
            feature_importance=feature_importance
        )

    def _find_optimal_clusters(self, max_clusters: int) -> int:
        """Find optimal number of clusters using multiple methods"""
        scores = {
            'silhouette': [],
            'calinski': [],
            'davies': []
        }
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            labels = kmeans.fit_predict(self.data_normalized)
            
            scores['silhouette'].append(silhouette_score(self.data_normalized, labels))
            scores['calinski'].append(calinski_harabasz_score(self.data_normalized, labels))
            scores['davies'].append(davies_bouldin_score(self.data_normalized, labels))
        
        # Use elbow method
        elbow = KneeLocator(
            range(2, max_clusters + 1),
            scores['silhouette'],
            curve='convex',
            direction='increasing'
        )
        
        optimal_k = elbow.knee if elbow.knee else 2
        
        # Save optimization plots
        self._plot_cluster_optimization(scores, optimal_k)
        
        return optimal_k

    def _plot_cluster_optimization(self, scores: Dict[str, List[float]], optimal_k: int) -> None:
        """Create plots for cluster number optimization"""
        fig = make_subplots(rows=3, cols=1, subplot_titles=('Silhouette Score', 
                                                          'Calinski-Harabasz Score',
                                                          'Davies-Bouldin Score'))
        
        k_range = list(range(2, len(scores['silhouette']) + 2))
        
        # Silhouette score
        fig.add_trace(
            go.Scatter(x=k_range, y=scores['silhouette'], mode='lines+markers'),
            row=1, col=1
        )
        
        # Calinski-Harabasz score
        fig.add_trace(
            go.Scatter(x=k_range, y=scores['calinski'], mode='lines+markers'),
            row=2, col=1
        )
        
        # Davies-Bouldin score
        fig.add_trace(
            go.Scatter(x=k_range, y=scores['davies'], mode='lines+markers'),
            row=3, col=1
        )
        
        # Add vertical line for optimal k
        for i in range(1, 4):
            fig.add_vline(x=optimal_k, line_dash="dash", row=i, col=1)
        
        fig.update_layout(height=900, title_text="Cluster Optimization Metrics")
        fig.write_html(self.plots_dir / 'cluster_optimization.html')

    def _calculate_clustering_metrics(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive clustering metrics"""
        metrics = {
            'silhouette': float(silhouette_score(data, labels)),
            'calinski_harabasz': float(calinski_harabasz_score(data, labels)),
            'davies_bouldin': float(davies_bouldin_score(data, labels))
        }
        
        # Add cluster distribution metrics
        unique_labels, counts = np.unique(labels, return_counts=True)
        metrics['cluster_sizes'] = counts.tolist()
        metrics['cluster_proportions'] = (counts / len(labels)).tolist()
        
        return metrics

    def _calculate_feature_importance_kmeans(self, kmeans_model: KMeans) -> Dict[str, float]:
        """Calculate feature importance for K-means clustering"""
        feature_importance = {}
        cluster_centers = kmeans_model.cluster_centers_
        
        # Calculate the spread of cluster centers for each feature
        for i, feature in enumerate(self.original_columns):
            center_spread = np.std(cluster_centers[:, i])
            feature_importance[feature] = float(center_spread)
        
        # Normalize importance scores
        max_importance = max(feature_importance.values())
        feature_importance = {k: v/max_importance for k, v in feature_importance.items()}
        
        return feature_importance

    def _perform_hierarchical(self, max_clusters: int) -> ClusteringResults:
        """Perform hierarchical clustering with different linkage methods"""
        # Try different linkage methods
        linkage_methods = ['ward', 'complete', 'average']
        best_score = -np.inf
        best_results = None
        
        for method in linkage_methods:
            # Create linkage matrix
            linkage_matrix = linkage(self.data_normalized, method=method)
            
            # Try different distance thresholds
            for n_clusters in range(2, max_clusters + 1):
                labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
                score = silhouette_score(self.data_normalized, labels)
                
                if score > best_score:
                    best_score = score
                    best_results = {
                        'labels': labels,
                        'linkage_matrix': linkage_matrix,
                        'method': method,
                        'n_clusters': n_clusters
                    }
        
        # Calculate metrics for best results
        metrics = self._calculate_clustering_metrics(self.data_normalized, best_results['labels'])
        
        # Create dendrogram
        self._plot_dendrogram(best_results['linkage_matrix'])
        
        return ClusteringResults(
            labels=best_results['labels'],
            metrics=metrics,
            model=best_results['linkage_matrix'],
            timestamp=datetime.datetime.now().isoformat(),
            parameters={'method': best_results['method'], 'n_clusters': best_results['n_clusters']},
            validation_scores={'silhouette': [best_score]},
            cluster_centers=None,
            feature_importance=None
        )

    def _plot_dendrogram(self, linkage_matrix: np.ndarray) -> None:
        """Create and save dendrogram visualization"""
        plt.figure(figsize=(15, 10))
        dendrogram(linkage_matrix)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
    def save_state(self, stage: str) -> None:
        """Save analysis state"""
        state = {
            'data': self.data,
            'data_normalized': self.data_normalized,
            'cluster_labels': self.cluster_labels,
            'pca_results': self.pca_results
        }
        
        with open(f"{self.results_dir}/state_{stage}.pkl", 'wb') as f:
            pickle.dump(state, f)

    def save_clustering_results(self, results: Dict[str, ClusteringResults]) -> None:
        """Save clustering results"""
        for method, result in results.items():
            # Save metrics
            with open(f"{self.results_dir}/{method}_metrics.json", 'w') as f:
                json.dump(result.metrics, f, indent=4)
            
            # Save model
            with open(f"{self.results_dir}/{method}_model.pkl", 'wb') as f:
                pickle.dump(result.model, f)

    @error_handler
    def analyze_clusters(self, results: Dict[str, ClusteringResults]) -> Dict[str, Any]:
        """Comprehensive cluster analysis"""
        analysis_results = {}
        
        for method, result in results.items():
            self.logger.info(f"Analyzing clusters for {method}")
            
            # Basic statistics
            cluster_stats = pd.DataFrame()
            for col in self.data.columns:
                cluster_stats[col] = self.data.groupby(result.labels)[col].mean()
            
            # Statistical tests
            stat_tests = self._perform_statistical_tests(result.labels)
            
            # Save results
            analysis_results[method] = {
                'cluster_stats': cluster_stats,
                'statistical_tests': stat_tests
            }
            
        return analysis_results

    def _perform_statistical_tests(self, labels: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Perform statistical tests between clusters"""
        tests = {}
        
        # ANOVA for numeric variables
        numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        
        for col in numeric_cols:
            f_stat, p_val = f_oneway(*[self.data[col][labels == i] for i in np.unique(labels)])
            tests[col] = {'f_statistic': f_stat, 'p_value': p_val}
        
        return tests

if __name__ == "__main__":
    # Example usage with error handling
    try:
        analyzer = ElectionSurveyAnalysis('PSY_492_Election_Experience_Survey.csv', log_level=logging.DEBUG)
        
        # Load and preprocess data
        data = analyzer.load_data()
        normalized_data = analyzer.preprocess_data(method='robust')
        
        # Perform clustering with multiple methods
        clustering_results = analyzer.perform_clustering(
            max_clusters=10,
            methods=['kmeans', 'hierarchical', 'dbscan']
        )
        
        # Analyze clusters
        analysis_results = analyzer.analyze_clusters(clustering_results)
        
        # Generate visualizations
        analyzer.perform_pca_visualization()
        
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)
