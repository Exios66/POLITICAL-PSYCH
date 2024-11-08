import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import chi2_contingency, f_oneway
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
from typing import Dict, List, Tuple, Any, Optional
import traceback
from dataclasses import dataclass
import pickle

@dataclass
class ClusteringResults:
    """Data class to store clustering results"""
    labels: np.ndarray
    metrics: Dict[str, float]
    model: Any
    timestamp: str

class ElectionSurveyAnalysis:
    def __init__(self, file_path: str, log_level: int = logging.INFO):
        """
        Initialize the analysis with enhanced logging and error handling
        
        Args:
            file_path (str): Path to the survey data file
            log_level (int): Logging level (default: logging.INFO)
        """
        self.file_path = file_path
        self.data = None
        self.data_normalized = None
        self.cluster_labels = None
        self.pca_results = None
        self.clustering_history = []
        
        # Set up logging
        self.setup_logging(log_level)
        
        # Create results directory
        self.results_dir = f"analysis_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.logger.info(f"Analysis initialized with file: {file_path}")
        
    def setup_logging(self, log_level: int) -> None:
        """Configure logging with both file and console handlers"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # Create formatters and handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        fh = logging.FileHandler(f'election_analysis_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
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
        """Decorator for error handling"""
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                self.logger.error(f"Error in {func.__name__}: {str(e)}")
                self.logger.error(traceback.format_exc())
                raise
        return wrapper

    @error_handler
    def load_data(self) -> pd.DataFrame:
        """Load and perform initial data validation"""
        self.logger.info("Loading data...")
        
        try:
            self.data = pd.read_csv(self.file_path)
            
            # Validate data structure
            if self.data.empty:
                raise ValueError("Empty dataset loaded")
            
            # Basic cleaning
            self.data.columns = [col.strip().replace(' ', '_') for col in self.data.columns]
            
            # Check for duplicate entries
            duplicates = self.data.duplicated().sum()
            if duplicates > 0:
                self.logger.warning(f"Found {duplicates} duplicate entries")
            
            # Data quality report
            self.generate_data_quality_report()
            
            self.logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise

    def generate_data_quality_report(self) -> None:
        """Generate comprehensive data quality report"""
        report = {
            'missing_values': self.data.isnull().sum().to_dict(),
            'unique_values': {col: self.data[col].nunique() for col in self.data.columns},
            'data_types': self.data.dtypes.to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).to_dict()
        }
        
        # Save report
        with open(f"{self.results_dir}/data_quality_report.json", 'w') as f:
            json.dump(report, f, indent=4, default=str)

    @error_handler
    def preprocess_data(self, method: str = 'standard') -> np.ndarray:
        """
        Enhanced data preprocessing with multiple scaling options and outlier detection
        
        Args:
            method (str): Scaling method ('standard', 'robust', or 'minmax')
        """
        self.logger.info("Starting data preprocessing...")
        
        # Handle missing values with KNN imputer
        numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        # Advanced imputation
        knn_imputer = KNNImputer(n_neighbors=5)
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        
        # Impute numeric data
        if len(numeric_cols) > 0:
            self.data[numeric_cols] = knn_imputer.fit_transform(self.data[numeric_cols])
        
        # Impute categorical data
        if len(categorical_cols) > 0:
            self.data[categorical_cols] = categorical_imputer.fit_transform(self.data[categorical_cols])
        
        # Encode categorical variables
        le = LabelEncoder()
        for col in categorical_cols:
            self.data[col] = le.fit_transform(self.data[col])
        
        # Outlier detection
        self.detect_outliers()
        
        # Scaling
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {method}")
        
        self.data_normalized = scaler.fit_transform(self.data)
        
        # Save preprocessing state
        self.save_state('preprocessing')
        
        self.logger.info("Data preprocessing completed successfully")
        return self.data_normalized

    def detect_outliers(self, contamination: float = 0.1) -> None:
        """Detect outliers using Isolation Forest"""
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outliers = iso_forest.fit_predict(self.data_normalized)
        outlier_indices = np.where(outliers == -1)[0]
        
        if len(outlier_indices) > 0:
            self.logger.warning(f"Detected {len(outlier_indices)} outliers")
            
            # Save outlier information
            outlier_report = {
                'total_outliers': len(outlier_indices),
                'outlier_indices': outlier_indices.tolist()
            }
            
            with open(f"{self.results_dir}/outlier_report.json", 'w') as f:
                json.dump(outlier_report, f, indent=4)

    @error_handler
    def perform_clustering(self, max_clusters: int = 10, methods: List[str] = ['kmeans', 'hierarchical', 'dbscan']) -> Dict[str, ClusteringResults]:
        """
        Perform multiple clustering analyses with cross-validation
        
        Args:
            max_clusters (int): Maximum number of clusters to try
            methods (List[str]): List of clustering methods to use
        """
        self.logger.info(f"Performing clustering analysis with methods: {methods}")
        
        results = {}
        
        for method in methods:
            self.logger.info(f"Running {method} clustering...")
            
            if method == 'kmeans':
                results[method] = self._perform_kmeans(max_clusters)
            elif method == 'hierarchical':
                results[method] = self._perform_hierarchical(max_clusters)
            elif method == 'dbscan':
                results[method] = self._perform_dbscan()
            
        # Save clustering results
        self.save_clustering_results(results)
        
        return results

    def _perform_kmeans(self, max_clusters: int) -> ClusteringResults:
        """Perform K-means clustering with cross-validation"""
        scores = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            cv_scores = cross_val_score(kmeans, self.data_normalized, cv=5)
            scores.append(np.mean(cv_scores))
        
        # Select optimal k
        optimal_k = np.argmax(scores) + 2
        
        # Final clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        labels = kmeans.fit_predict(self.data_normalized)
        
        metrics = {
            'silhouette': silhouette_score(self.data_normalized, labels),
            'calinski_harabasz': calinski_harabasz_score(self.data_normalized, labels)
        }
        
        return ClusteringResults(
            labels=labels,
            metrics=metrics,
            model=kmeans,
            timestamp=datetime.datetime.now().isoformat()
        )

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
