# Standard library imports
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import sys
import os
import re
import string
import warnings
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
import tqdm
from kneed import KneeLocator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import f_oneway, chi2_contingency, ttest_ind
from factor_analyzer import FactorAnalyzer
from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score, fowlkes_mallows_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pyLDAvis
import pyLDAvis.gensim_models
from pyLDAvis import prepare as prepare_lda
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# SpaCy
import spacy
from spacy.tokens import Doc, Token, Span
from spacy.language import Language
from spacy.pipeline import EntityRuler
from spacy.matcher import Matcher, PhraseMatcher

# Sentiment analysis
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Dimensionality reduction
import umap.umap_ as umap
from statsmodels.stats.multitest import multipletests
from scipy import stats

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cluster_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Get logger for this module
logger = logging.getLogger(__name__)

# Survey metrics categories
SURVEY_METRICS = {
    'news_consumption': [
        'Trad_News_print', 'Trad_News_online', 'Trad_News_TV', 'Trad_News_radio',
        'SM_News_1', 'SM_News_2', 'SM_News_3', 'SM_News_4', 'SM_News_5', 'SM_News_6',
        'News_frequency', 'SM_Sharing'
    ],
    'temporal': ['survey_date'],
    'factor_analysis': ['FA_Components', 'FA_Loadings', 'FA_Variance'],
    'clustering_metrics': ['Silhouette', 'Calinski', 'Davies_Bouldin', 'Fowlkes_Mallows']
}

class ClusteringError(Exception):
    """Custom exception for clustering-related errors"""
    def __init__(self, message: str):
        self.message = message
        logger.error(f"ClusteringError: {message}")
        super().__init__(self.message)

class VisualizationError(Exception):
    """Custom exception for visualization-related errors"""
    def __init__(self, message: str):
        self.message = message
        logger.error(f"VisualizationError: {message}")
        super().__init__(self.message)

class ClusterAnalysisGUI:
    def __init__(self, root):
        """Initialize GUI components with enhanced structure and error handling"""
        logger.info("Initializing ClusterAnalysisGUI")
        
        try:
            self.root = root
            self.root.title("Political Psychology Cluster Analysis")
            self.root.geometry("1200x800")

            # Configure logging
            self._setup_logging()

            # Initialize core variables
            self._initialize_variables()

            # Setup directories
            self._setup_directories()

            # Initialize data variables
            self.cleaned_data = None
            self.normalized_data = None
            self.cluster_labels = None
            self.current_plot = None
            self.random_state = 42

            # Configure window minimum size
            self.root.minsize(800, 600)

            # Configure grid weights for proper scaling
            self.root.grid_rowconfigure(0, weight=1)
            self.root.grid_columnconfigure(0, weight=1)

            # Create main container with tabs
            self.notebook = ttk.Notebook(self.root)
            self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            # Create frames for each tab
            self.data_tab = ttk.Frame(self.notebook)
            self.cluster_tab = ttk.Frame(self.notebook)
            self.viz_tab = ttk.Frame(self.notebook)
            self.analysis_tab = ttk.Frame(self.notebook)
            self.factor_tab = ttk.Frame(self.notebook)

            # Initialize cluster frame
            self.cluster_frame = ttk.LabelFrame(self.cluster_tab, text="Clustering Options")
            self.cluster_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            # Add tabs to notebook
            self.notebook.add(self.data_tab, text="Data Processing")
            self.notebook.add(self.cluster_tab, text="Clustering")
            self.notebook.add(self.viz_tab, text="Visualization")
            self.notebook.add(self.analysis_tab, text="Analysis")
            self.notebook.add(self.factor_tab, text="Factor Analysis")

            # Initialize tab contents
            self.create_data_tab()
            self.create_cluster_tab()
            self.create_viz_tab()
            self.create_analysis_tab()
            self._initialize_factor_analysis_tab()

            # Create status and progress bars
            self._create_status_bars()

            logger.info("ClusterAnalysisGUI initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ClusterAnalysisGUI: {str(e)}\n{traceback.format_exc()}")
            messagebox.showerror("Initialization Error", f"Failed to initialize application: {str(e)}")
            raise

    def _setup_logging(self):
        """Set up logging configuration"""
        try:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)

            # Create log file with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_dir / f"cluster_analysis_{timestamp}.log"

            # Configure logging
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

            self.logger.info("Logging initialized")

        except Exception as e:
            logger.error(f"Failed to setup logging: {str(e)}\n{traceback.format_exc()}")
            raise

    def _initialize_variables(self):
        """Initialize all GUI variables"""
        try:
            logger.info("Initializing GUI variables")

            # File handling variables
            self.file_path = tk.StringVar()

            # Data processing variables
            self.handle_missing = tk.BooleanVar(value=True)
            self.remove_outliers = tk.BooleanVar(value=True)
            self.normalize = tk.BooleanVar(value=True)

            # Method variables
            self.missing_method = tk.StringVar(value="median")
            self.outlier_method = tk.StringVar(value="iqr")
            self.norm_method = tk.StringVar(value="standard")

            # Clustering variables
            self.cluster_method = tk.StringVar(value="kmeans")
            self.n_clusters = tk.IntVar(value=3)
            self.eps = tk.DoubleVar(value=0.5)
            self.min_samples = tk.IntVar(value=5)
            self.max_iter = tk.IntVar(value=300)
            self.n_init = tk.IntVar(value=10)
            self.linkage = tk.StringVar(value="ward")
            self.n_clusters_hierarchical = tk.IntVar(value=3)

            # Analysis variables
            self.analysis_type = tk.StringVar(value="statistical")
            self.stat_test = tk.StringVar(value="anova")
            self.n_iterations = tk.IntVar(value=100)
            self.subsample_size = tk.IntVar(value=80)

            # Plot variables
            self.plot_type = tk.StringVar(value="distribution")
            self.dist_feature = tk.StringVar()
            self.dist_type = tk.StringVar(value="histogram")
            self.profile_type = tk.StringVar(value="heatmap")
            self.reduction_method = tk.StringVar(value="pca")
            self.n_components = tk.IntVar(value=2)
            self.importance_method = tk.StringVar(value="random_forest")
            self.n_top_features = tk.IntVar(value=10)

            # Factor analysis variables
            self.n_factors = tk.IntVar(value=3)
            self.rotation_method = tk.StringVar(value='varimax')
            self.factor_threshold = tk.DoubleVar(value=0.3)

            # Advanced clustering variables
            self.gmm_n_components = tk.IntVar(value=3)
            self.gmm_covariance_type = tk.StringVar(value='full')
            self.umap_n_neighbors = tk.IntVar(value=15)
            self.umap_min_dist = tk.DoubleVar(value=0.1)

            # Status variables
            self.status_var = tk.StringVar(value="Ready")
            self.progress_var = tk.DoubleVar(value=0)

            logger.info("GUI variables initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize variables: {str(e)}\n{traceback.format_exc()}")
            raise

    def _setup_directories(self):
        """Create necessary directories"""
        try:
            logger.info("Setting up directories")

            # Define directory paths
            self.output_dir = Path("output")
            self.models_dir = self.output_dir / "models"
            self.plots_dir = self.output_dir / "plots"
            self.results_dir = self.output_dir / "results"
            self.temp_dir = self.output_dir / "temp"

            # Create directories
            for directory in [self.output_dir, self.models_dir, self.plots_dir,
                            self.results_dir, self.temp_dir]:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {directory}")

            logger.info("Directories created successfully")

        except Exception as e:
            logger.error(f"Failed to setup directories: {str(e)}\n{traceback.format_exc()}")
            raise

    def _initialize_factor_analysis_tab(self):
        """Initialize factor analysis tab with controls and visualization"""
        try:
            self.factor_tab = ttk.Frame(self.notebook)
            self.notebook.add(self.factor_tab, text="Factor Analysis")
            
            # Parameters frame
            params_frame = ttk.LabelFrame(self.factor_tab, text="Factor Analysis Parameters")
            params_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Number of factors
            factors_frame = ttk.Frame(params_frame)
            factors_frame.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(factors_frame, text="Number of factors:").pack(side=tk.LEFT)
            ttk.Entry(
                factors_frame,
                textvariable=self.n_factors,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Rotation method
            rotation_frame = ttk.Frame(params_frame)
            rotation_frame.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(rotation_frame, text="Rotation method:").pack(side=tk.LEFT)
            ttk.OptionMenu(
                rotation_frame,
                self.rotation_method,
                "varimax",
                "varimax", "promax", "oblimin"
            ).pack(side=tk.LEFT, padx=5)
            
            # Loading threshold
            threshold_frame = ttk.Frame(params_frame)
            threshold_frame.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(threshold_frame, text="Loading threshold:").pack(side=tk.LEFT)
            ttk.Entry(
                threshold_frame,
                textvariable=self.factor_threshold,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Control buttons
            button_frame = ttk.Frame(self.factor_tab)
            button_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Button(
                button_frame,
                text="Run Factor Analysis",
                command=self.run_factor_analysis
            ).pack(side=tk.LEFT, padx=5)
            
            ttk.Button(
                button_frame,
                text="Save Results",
                command=self.save_factor_analysis_results
            ).pack(side=tk.LEFT, padx=5)
            
            # Results frame
            self.factor_results_frame = ttk.LabelFrame(self.factor_tab, text="Results")
            self.factor_results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            self.logger.info("Factor analysis tab initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize factor analysis tab: {str(e)}")
            raise

    def run_factor_analysis(self):
        """Perform factor analysis on the data"""
        try:
            if self.normalized_data is None:
                raise ValueError("No processed data available")
                
            # Create factor analyzer instance
            fa = FactorAnalyzer(
                n_factors=self.n_factors.get(),
                rotation=self.rotation_method.get()
            )
            
            # Fit factor analysis
            fa.fit(self.normalized_data)
            
            # Get loadings
            loadings = pd.DataFrame(
                fa.loadings_,
                columns=[f'Factor{i+1}' for i in range(self.n_factors.get())],
                index=self.normalized_data.columns
            )
            
            # Get variance explained
            variance = pd.DataFrame({
                'SS Loadings': fa.get_factor_variance()[0],
                'Proportion Var': fa.get_factor_variance()[1],
                'Cumulative Var': fa.get_factor_variance()[2]
            })
            
            # Create visualization
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Factor Loadings Heatmap', 'Variance Explained')
            )
            
            # Loadings heatmap
            fig.add_trace(
                go.Heatmap(
                    z=loadings.values,
                    x=loadings.columns,
                    y=loadings.index,
                    colorscale='RdBu'
                ),
                row=1, col=1
            )
            
            # Variance explained bar plot
            fig.add_trace(
                go.Bar(
                    x=variance.index,
                    y=variance['Proportion Var'],
                    name='Proportion of Variance'
                ),
                row=2, col=1
            )
            
            fig.update_layout(height=800, showlegend=False)
            
            # Save results
            self.factor_results = {
                'loadings': loadings,
                'variance': variance,
                'model': fa
            }
            
            # Update display
            if hasattr(self, 'factor_plot_widget'):
                self.factor_plot_widget.destroy()
            
            self.factor_plot_widget = go.FigureWidget(fig)
            self.factor_plot_widget.show()
            
            self.logger.info("Factor analysis completed successfully")
            
        except Exception as e:
            self.logger.error(f"Factor analysis failed: {str(e)}")
            messagebox.showerror("Error", f"Factor analysis failed: {str(e)}")

    def perform_advanced_clustering(self):
        """Perform advanced clustering analysis with multiple methods"""
        try:
            if self.normalized_data is None:
                raise ValueError("No processed data available")
                
            results = {}
            
            # Gaussian Mixture Model
            gmm = GaussianMixture(
                n_components=self.gmm_n_components.get(),
                covariance_type=self.gmm_covariance_type.get()
            )
            gmm_labels = gmm.fit_predict(self.normalized_data)
            results['gmm'] = {
                'labels': gmm_labels,
                'bic': gmm.bic(self.normalized_data),
                'aic': gmm.aic(self.normalized_data)
            }
            
            # UMAP + DBSCAN
            umap_reducer = umap.UMAP(
                n_neighbors=self.umap_n_neighbors.get(),
                min_dist=self.umap_min_dist.get()
            )
            umap_embedding = umap_reducer.fit_transform(self.normalized_data)
            
            dbscan = DBSCAN(eps=self.eps.get(), min_samples=self.min_samples.get())
            umap_dbscan_labels = dbscan.fit_predict(umap_embedding)
            
            results['umap_dbscan'] = {
                'labels': umap_dbscan_labels,
                'embedding': umap_embedding
            }
            
            # Calculate clustering metrics
            for method, result in results.items():
                if -1 not in result['labels']:  # Skip if DBSCAN found noise points
                    result['metrics'] = {
                        'silhouette': silhouette_score(self.normalized_data, result['labels']),
                        'calinski': calinski_harabasz_score(self.normalized_data, result['labels']),
                        'davies_bouldin': davies_bouldin_score(self.normalized_data, result['labels']),
                    }
            
            # Store results
            self.advanced_clustering_results = results
            
            # Update display
            self.show_advanced_clustering_results()
            
            self.logger.info("Advanced clustering completed successfully")
            
        except Exception as e:
            self.logger.error(f"Advanced clustering failed: {str(e)}")
            messagebox.showerror("Error", f"Advanced clustering failed: {str(e)}")

    def show_advanced_clustering_results(self):
        """Display results of advanced clustering analysis"""
        try:
            if not hasattr(self, 'advanced_clustering_results'):
                raise ValueError("No advanced clustering results available")
                
            # Create visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'GMM Clustering',
                    'UMAP + DBSCAN',
                    'Clustering Metrics Comparison',
                    'Model Selection Criteria'
                )
            )
            
            results = self.advanced_clustering_results
            
            # GMM plot
            tsne = TSNE(n_components=2, random_state=42)
            tsne_embedding = tsne.fit_transform(self.normalized_data)
            
            fig.add_trace(
                go.Scatter(
                    x=tsne_embedding[:, 0],
                    y=tsne_embedding[:, 1],
                    mode='markers',
                    marker=dict(
                        color=results['gmm']['labels'],
                        colorscale='Viridis'
                    ),
                    name='GMM Clusters'
                ),
                row=1, col=1
            )
            
            # UMAP + DBSCAN plot
            fig.add_trace(
                go.Scatter(
                    x=results['umap_dbscan']['embedding'][:, 0],
                    y=results['umap_dbscan']['embedding'][:, 1],
                    mode='markers',
                    marker=dict(
                        color=results['umap_dbscan']['labels'],
                        colorscale='Viridis'
                    ),
                    name='UMAP + DBSCAN'
                ),
                row=1, col=2
            )
            
            # Metrics comparison
            metrics = ['silhouette', 'calinski', 'davies_bouldin']
            methods = []
            metric_values = []
            metric_names = []
            
            for method, result in results.items():
                if 'metrics' in result:
                    for metric, value in result['metrics'].items():
                        methods.append(method)
                        metric_values.append(value)
                        metric_names.append(metric)
            
            fig.add_trace(
                go.Bar(
                    x=methods,
                    y=metric_values,
                    name='Clustering Metrics',
                    text=metric_names
                ),
                row=2, col=1
            )
            
            # Model selection criteria
            fig.add_trace(
                go.Bar(
                    x=['BIC', 'AIC'],
                    y=[results['gmm']['bic'], results['gmm']['aic']],
                    name='GMM Model Selection'
                ),
                row=2, col=2
            )
            
            fig.update_layout(height=800, showlegend=True)
            
            # Update display
            if hasattr(self, 'advanced_plot_widget'):
                self.advanced_plot_widget.destroy()
            
            self.advanced_plot_widget = go.FigureWidget(fig)
            self.advanced_plot_widget.show()
            
            self.logger.info("Advanced clustering results displayed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to show advanced clustering results: {str(e)}")
            messagebox.showerror("Error", f"Failed to show results: {str(e)}")

    def create_data_tab(self):
        """Create and configure data processing tab"""
        try:
            # File selection frame
            file_frame = ttk.LabelFrame(self.data_tab, text="Data File")
            file_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Entry(
                file_frame,
                textvariable=self.file_path,
                width=60
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            ttk.Button(
                file_frame,
                text="Browse",
                command=self.browse_file
            ).pack(side=tk.LEFT, padx=5)
            
            # Data processing options frame
            options_frame = ttk.LabelFrame(self.data_tab, text="Processing Options")
            options_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Missing values
            missing_frame = ttk.Frame(options_frame)
            missing_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Checkbutton(
                missing_frame,
                text="Handle Missing Values",
                variable=self.handle_missing
            ).pack(side=tk.LEFT)
            
            ttk.OptionMenu(
                missing_frame,
                self.missing_method,
                "median",
                "median", "mean", "mode", "drop"
            ).pack(side=tk.LEFT, padx=5)
            
            # Outliers
            outlier_frame = ttk.Frame(options_frame)
            outlier_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Checkbutton(
                outlier_frame,
                text="Remove Outliers",
                variable=self.remove_outliers
            ).pack(side=tk.LEFT)
            
            ttk.OptionMenu(
                outlier_frame,
                self.outlier_method,
                "iqr",
                "iqr", "zscore", "isolation_forest"
            ).pack(side=tk.LEFT, padx=5)
            
            # Normalization
            norm_frame = ttk.Frame(options_frame)
            norm_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Checkbutton(
                norm_frame,
                text="Normalize Features",
                variable=self.normalize
            ).pack(side=tk.LEFT)
            
            ttk.OptionMenu(
                norm_frame,
                self.norm_method,
                "standard",
                "standard", "minmax", "robust"
            ).pack(side=tk.LEFT, padx=5)
            
            # Process button
            ttk.Button(
                options_frame,
                text="Process Data",
                command=self.process_data
            ).pack(pady=5)
            
            # Preview frame
            preview_frame = ttk.LabelFrame(self.data_tab, text="Data Preview")
            preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Add scrollbars
            preview_scroll_y = ttk.Scrollbar(preview_frame)
            preview_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
            
            preview_scroll_x = ttk.Scrollbar(preview_frame, orient=tk.HORIZONTAL)
            preview_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
            
            # Create text widget
            self.preview_text = tk.Text(
                preview_frame,
                wrap=tk.NONE,
                xscrollcommand=preview_scroll_x.set,
                yscrollcommand=preview_scroll_y.set
            )
            self.preview_text.pack(fill=tk.BOTH, expand=True)
            
            # Configure scrollbars
            preview_scroll_y.config(command=self.preview_text.yview)
            preview_scroll_x.config(command=self.preview_text.xview)
            
            self.logger.info("Data tab created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create data tab: {str(e)}")
            raise

    def create_cluster_tab(self):
        """Create and configure clustering tab"""
        try:
            # Method selection frame
            method_frame = ttk.LabelFrame(self.cluster_frame, text="Clustering Method")
            method_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Label(method_frame, text="Method:").pack(side=tk.LEFT, padx=5)
            ttk.OptionMenu(
                method_frame,
                self.cluster_method,
                "kmeans",
                "kmeans", "dbscan", "hierarchical",
                command=self.update_cluster_options
            ).pack(side=tk.LEFT, padx=5)
            
            # Parameters frame - will be populated by update_cluster_options
            self.params_frame = ttk.LabelFrame(self.cluster_frame, text="Parameters")
            self.params_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Results frame
            results_frame = ttk.LabelFrame(self.cluster_frame, text="Results")
            results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Add scrollbars
            results_scroll_y = ttk.Scrollbar(results_frame)
            results_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
            
            results_scroll_x = ttk.Scrollbar(results_frame, orient=tk.HORIZONTAL)
            results_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
            
            # Results text widget
            self.results_text = tk.Text(
                results_frame,
                wrap=tk.NONE,
                xscrollcommand=results_scroll_x.set,
                yscrollcommand=results_scroll_y.set
            )
            self.results_text.pack(fill=tk.BOTH, expand=True)
            
            # Configure scrollbars
            results_scroll_y.config(command=self.results_text.yview)
            results_scroll_x.config(command=self.results_text.xview)
            
            # Control buttons
            button_frame = ttk.Frame(self.cluster_frame)
            button_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Button(
                button_frame,
                text="Run Clustering",
                command=self.perform_clustering
            ).pack(side=tk.LEFT, padx=5)
            
            ttk.Button(
                button_frame,
                text="Save Results",
                command=self.save_clustering_results
            ).pack(side=tk.LEFT, padx=5)
            
            # Initialize clustering method variable
            self.cluster_method_var = tk.StringVar(value="kmeans")
            
            self.logger.info("Cluster tab created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create cluster tab: {str(e)}")
            raise

    def create_viz_tab(self):
        """Create and configure visualization tab"""
        try:
            # Plot type selection frame
            type_frame = ttk.LabelFrame(self.viz_tab, text="Visualization Type")
            type_frame.pack(fill=tk.X, padx=5, pady=5)
            
            plot_types = [
                ("Distribution", "distribution"),
                ("Cluster Profiles", "profile"),
                ("Dimensionality Reduction", "reduction"),
                ("Feature Importance", "importance")
            ]
            
            for text, value in plot_types:
                ttk.Radiobutton(
                    type_frame,
                    text=text,
                    variable=self.plot_type,
                    value=value,
                    command=self.update_plot_options
                ).pack(side=tk.LEFT, padx=5)
            
            # Options frame - will be populated by update_plot_options
            self.plot_options_frame = ttk.LabelFrame(self.viz_tab, text="Plot Options")
            self.plot_options_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Plot frame
            plot_frame = ttk.LabelFrame(self.viz_tab, text="Plot")
            plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Initialize matplotlib figure
            self.fig = Figure(figsize=(8, 6))
            self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add toolbar
            toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
            toolbar.update()
            
            # Control buttons
            button_frame = ttk.Frame(self.viz_tab)
            button_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Button(
                button_frame,
                text="Generate Plot",
                command=self.generate_plot
            ).pack(side=tk.LEFT, padx=5)
            
            ttk.Button(
                button_frame,
                text="Save Plot",
                command=self.save_plot
            ).pack(side=tk.LEFT, padx=5)
            
            self.logger.info("Visualization tab created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create visualization tab: {str(e)}")
            raise

    def create_analysis_tab(self):
        """Create and configure analysis tab"""
        try:
            # Analysis type selection frame
            type_frame = ttk.LabelFrame(self.analysis_tab, text="Analysis Type")
            type_frame.pack(fill=tk.X, padx=5, pady=5)
            
            analysis_types = [
                ("Statistical Tests", "statistical"),
                ("Feature Importance", "importance"),
                ("Cluster Stability", "stability")
            ]
            
            for text, value in analysis_types:
                ttk.Radiobutton(
                    type_frame,
                    text=text,
                    variable=self.analysis_type,
                    value=value,
                    command=self.update_analysis_options
                ).pack(side=tk.LEFT, padx=5)
            
            # Options frame - will be populated by update_analysis_options
            self.analysis_options_frame = ttk.LabelFrame(self.analysis_tab, text="Analysis Options")
            self.analysis_options_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Results frame
            results_frame = ttk.LabelFrame(self.analysis_tab, text="Analysis Results")
            results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Add scrollbars
            results_scroll_y = ttk.Scrollbar(results_frame)
            results_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
            
            results_scroll_x = ttk.Scrollbar(results_frame, orient=tk.HORIZONTAL)
            results_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
            
            # Results text widget
            self.analysis_text = tk.Text(
                results_frame,
                wrap=tk.NONE,
                xscrollcommand=results_scroll_x.set,
                yscrollcommand=results_scroll_y.set
            )
            self.analysis_text.pack(fill=tk.BOTH, expand=True)
            
                        
            # Configure scrollbars
            results_scroll_y.config(command=self.analysis_text.yview)
            results_scroll_x.config(command=self.analysis_text.xview)
            
            # Control buttons
            button_frame = ttk.Frame(self.analysis_tab)
            button_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Button(
                button_frame,
                text="Run Analysis",
                command=self.run_analysis
            ).pack(side=tk.LEFT, padx=5)
            
            ttk.Button(
                button_frame,
                text="Export Results",
                command=self.export_analysis
            ).pack(side=tk.LEFT, padx=5)
            
            self.logger.info("Analysis tab created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create analysis tab: {str(e)}")
            raise

    def process_data(self):
        """Process the input data according to selected options"""
        try:
            if not self.file_path.get():
                raise ValueError("No input file selected")

            self.status_var.set("Processing data...")
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            raise
            self.progress_var.set(0)

            # Read data
            file_path = Path(self.file_path.get())
            if file_path.suffix.lower() == '.csv':
                data = pd.read_csv(file_path)
            elif file_path.suffix.lower() == '.xlsx':
                data = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")

            # Store original data
            self.original_data = data.copy()
            self.cleaned_data = data.copy()

            # Handle missing values if selected
            if self.handle_missing.get():
                method = self.missing_method.get()
                if method == "mean":
                    self.cleaned_data = self.cleaned_data.fillna(self.cleaned_data.mean())
                elif method == "median":
                    self.cleaned_data = self.cleaned_data.fillna(self.cleaned_data.median())
                elif method == "mode":
                    self.cleaned_data = self.cleaned_data.fillna(self.cleaned_data.mode().iloc[0])
                self.progress_var.set(33)

            # Remove outliers if selected
            if self.remove_outliers.get():
                method = self.outlier_method.get()
                if method == "zscore":
                    z_scores = np.abs(stats.zscore(self.cleaned_data))
                    self.cleaned_data = self.cleaned_data[(z_scores < 3).all(axis=1)]
                elif method == "iqr":
                    Q1 = self.cleaned_data.quantile(0.25)
                    Q3 = self.cleaned_data.quantile(0.75)
                    IQR = Q3 - Q1
                    self.cleaned_data = self.cleaned_data[
                        ~((self.cleaned_data < (Q1 - 1.5 * IQR)) | 
                          (self.cleaned_data > (Q3 + 1.5 * IQR))).any(axis=1)]
                    self.progress_var.set(66)

    def update_plot_options(self, *args):
        """Update visible options based on selected plot type"""
        try:
            plot_type = self.plot_type.get()
            
            # Clear previous options
            for widget in self.plot_options_frame.winfo_children():
                widget.destroy()
                
            # Create new options based on plot type
            if plot_type == "distribution":
                self.create_distribution_options()
            elif plot_type == "profile":
                self.create_profile_options()
            elif plot_type == "reduction":
                self.create_reduction_options()
            elif plot_type == "importance":
                self.create_importance_options()
                
            self.logger.info(f"Updated plot options for {plot_type}")
            
        except Exception as e:
            self.logger.error(f"Failed to update plot options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update options: {str(e)}")

    def update_analysis_options(self, *args):
        """Update analysis options based on selected analysis type"""
        try:
            analysis_type = self.analysis_type.get()
            
            # Clear previous options
            for widget in self.analysis_options_frame.winfo_children():
                widget.destroy()
                
            # Create new options based on analysis type
            if analysis_type == "statistical":
                self.create_statistical_options()
            elif analysis_type == "importance":
                self.create_importance_analysis_options()
            elif analysis_type == "stability":
                self.create_stability_options()
                
            self.logger.info(f"Updated analysis options for {analysis_type}")
            
        except Exception as e:
            self.logger.error(f"Failed to update analysis options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update options: {str(e)}")

    def create_distribution_options(self):
        """Create options for distribution plots"""
        try:
            # Feature selection
            feature_frame = ttk.Frame(self.plot_options_frame)
            feature_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(feature_frame, text="Feature:").pack(side=tk.LEFT)
            if self.normalized_data is not None:
                features = list(self.normalized_data.columns)
                self.dist_feature.set(features[0] if features else "")
                ttk.OptionMenu(
                    feature_frame,
                    self.dist_feature,
                    features[0] if features else "",
                    *features
                ).pack(side=tk.LEFT, padx=5)
                
            # Plot type
            type_frame = ttk.Frame(self.plot_options_frame)
            type_frame.pack(fill=tk.X, padx=5, pady=2)
            
            for text, value in [("Histogram", "histogram"), ("KDE", "kde"), ("Box Plot", "box")]:
                ttk.Radiobutton(
                    type_frame,
                    text=text,
                    variable=self.dist_type,
                    value=value
                ).pack(side=tk.LEFT, padx=5)
                
        except Exception as e:
            self.logger.error(f"Failed to create distribution options: {str(e)}")
            raise

    def create_profile_options(self):
        """Create options for cluster profile plots"""
        try:
            # Profile type selection
            type_frame = ttk.Frame(self.plot_options_frame)
            type_frame.pack(fill=tk.X, padx=5, pady=2)
            
            for text, value in [("Heatmap", "heatmap"), ("Parallel", "parallel"), ("Radar", "radar")]:
                ttk.Radiobutton(
                    type_frame,
                    text=text,
                    variable=self.profile_type,
                    value=value
                ).pack(side=tk.LEFT, padx=5)
                
        except Exception as e:
            self.logger.error(f"Failed to create profile options: {str(e)}")
            raise

    def create_reduction_options(self):
        """Create options for dimensionality reduction plots"""
        try:
            # Method selection
            method_frame = ttk.Frame(self.plot_options_frame)
            method_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(method_frame, text="Method:").pack(side=tk.LEFT)
            ttk.OptionMenu(
                method_frame,
                self.reduction_method,
                "pca",
                "pca", "tsne", "umap"
            ).pack(side=tk.LEFT, padx=5)
            
            # Components
            comp_frame = ttk.Frame(self.plot_options_frame)
            comp_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(comp_frame, text="Components:").pack(side=tk.LEFT)
            ttk.Entry(
                comp_frame,
                textvariable=self.n_components,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            self.logger.error(f"Failed to create reduction options: {str(e)}")
            raise

    def create_importance_options(self):
        """Create options for feature importance plots"""
        try:
            # Method selection
            method_frame = ttk.Frame(self.plot_options_frame)
            method_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(method_frame, text="Method:").pack(side=tk.LEFT)
            ttk.OptionMenu(
                method_frame,
                self.importance_method,
                "random_forest",
                "random_forest", "mutual_info", "chi2"
            ).pack(side=tk.LEFT, padx=5)
            
            # Number of features
            n_features_frame = ttk.Frame(self.plot_options_frame)
            n_features_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(n_features_frame, text="Top features:").pack(side=tk.LEFT)
            ttk.Entry(
                n_features_frame,
                textvariable=self.n_top_features,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            self.logger.error(f"Failed to create importance options: {str(e)}")
            raise

    def create_statistical_options(self):
        """Create options for statistical analysis"""
        try:
            # Test selection
            test_frame = ttk.Frame(self.analysis_options_frame)
            test_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(test_frame, text="Statistical test:").pack(side=tk.LEFT)
            ttk.OptionMenu(
                test_frame,
                self.stat_test,
                "anova",
                "anova", "ttest", "chi2"
            ).pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            self.logger.error(f"Failed to create statistical options: {str(e)}")
            raise

    def create_importance_analysis_options(self):
        """Create options for feature importance analysis"""
        try:
            # Method selection
            method_frame = ttk.Frame(self.analysis_options_frame)
            method_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(method_frame, text="Method:").pack(side=tk.LEFT)
            ttk.OptionMenu(
                method_frame,
                self.importance_method,
                "random_forest",
                "random_forest", "mutual_info", "chi2"
            ).pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            self.logger.error(f"Failed to create importance analysis options: {str(e)}")
            raise

    def create_stability_options(self):
        """Create options for cluster stability analysis"""
        try:
            # Number of iterations
            iter_frame = ttk.Frame(self.analysis_options_frame)
            iter_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(iter_frame, text="Iterations:").pack(side=tk.LEFT)
            ttk.Entry(
                iter_frame,
                textvariable=self.n_iterations,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Sample size
            sample_frame = ttk.Frame(self.analysis_options_frame)
            sample_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(sample_frame, text="Sample size (%):").pack(side=tk.LEFT)
            ttk.Entry(
                sample_frame,
                textvariable=self.subsample_size,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            self.logger.error(f"Failed to create stability options: {str(e)}")
            raise

    def perform_clustering(self):
        """Perform clustering analysis with selected method and parameters"""
        try:
            if self.normalized_data is None:
                raise ValueError("No processed data available")

            self.status_var.set("Performing clustering...")
            self.progress_var.set(0)
            
            method = self.cluster_method.get()
            
            if method == "kmeans":
                clusterer = KMeans(
                    n_clusters=self.n_clusters.get(),
                    n_init=self.n_init.get(),
                    max_iter=self.max_iter.get(),
                    random_state=self.random_state
                )
            elif method == "dbscan":
                clusterer = DBSCAN(
                    eps=self.eps.get(),
                    min_samples=self.min_samples.get()
                )
            elif method == "hierarchical":
                clusterer = AgglomerativeClustering(
                    n_clusters=self.n_clusters_hierarchical.get(),
                    linkage=self.linkage.get()
                )
            else:
                self.normalized_data = self.cleaned_data.copy()

            self.progress_var.set(100)
            self.status_var.set("Data processing complete")
            
            # Update preview
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(tk.END, "Processed Data Preview:\n\n")
            self.preview_text.insert(tk.END, str(self.normalized_data.head()))
            
            self.logger.info("Data processing completed successfully")
            messagebox.showinfo("Success", "Data processing completed successfully")
            
        except Exception as e:
            self.logger.error(f"Data processing error: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Data processing failed: {str(e)}")
            self.progress_var.set(0)

    def browse_file(self):
        """Open file dialog to select input data file"""
        try:
            filetypes = (
                ('CSV files', '*.csv'),
                ('Excel files', '*.xlsx'),
                ('All files', '*.*')
            )
            
            filename = filedialog.askopenfilename(
                title='Select data file',
                initialdir='.',
                filetypes=filetypes
            )
            
            if filename:
                self.file_path.set(filename)
                self.logger.info(f"Selected file: {filename}")
                
                # Clear any previous data
                self.cleaned_data = None
                self.normalized_data = None
                self.cluster_labels = None
                
                # Clear preview
                self.preview_text.delete(1.0, tk.END)
                self.preview_text.insert(tk.END, "Please process the data to see preview.")
                
        except Exception as e:
            self.logger.error(f"Failed to browse file: {str(e)}")
            messagebox.showerror("Error", f"Failed to open file browser: {str(e)}")

    def update_cluster_options(self, *args):
        """Update clustering options based on selected method"""
        try:
            method = self.cluster_method.get()
            
            # Hide all parameter frames
            for frame in [self.kmeans_frame, self.dbscan_frame, self.hierarchical_frame]:
                if hasattr(self, frame.__str__()):
                    frame.pack_forget()
            
            # Show relevant parameter frame
            if method == "kmeans":
                self.create_kmeans_frame()
                self.kmeans_frame.pack(fill=tk.X, padx=5, pady=5)
            elif method == "dbscan":
                self.create_dbscan_frame()
                self.dbscan_frame.pack(fill=tk.X, padx=5, pady=5)
            elif method == "hierarchical":
                self.create_hierarchical_frame()
                self.hierarchical_frame.pack(fill=tk.X, padx=5, pady=5)
                
            self.logger.info(f"Updated clustering options for method: {method}")
            
        except Exception as e:
            self.logger.error(f"Failed to update clustering options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update options: {str(e)}")

    def create_kmeans_frame(self):
        """Create frame for K-means clustering options"""
        try:
            if not hasattr(self, 'kmeans_frame'):
                self.kmeans_frame = ttk.LabelFrame(self.cluster_frame, text="K-means Parameters")
            
            # Clear existing widgets
            for widget in self.kmeans_frame.winfo_children():
                widget.destroy()
            
            # Number of clusters
            clusters_frame = ttk.Frame(self.kmeans_frame)
            clusters_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(clusters_frame, text="Number of clusters (k):").pack(side=tk.LEFT)
            ttk.Entry(
                clusters_frame,
                textvariable=self.n_clusters,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Maximum iterations
            iter_frame = ttk.Frame(self.kmeans_frame)
            iter_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(iter_frame, text="Maximum iterations:").pack(side=tk.LEFT)
            ttk.Entry(
                iter_frame,
                textvariable=self.max_iter,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Number of initializations
            init_frame = ttk.Frame(self.kmeans_frame)
            init_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(init_frame, text="Number of initializations:").pack(side=tk.LEFT)
            ttk.Entry(
                init_frame,
                textvariable=self.n_init,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            self.logger.info("Created K-means parameter frame")
            
        except Exception as e:
            self.logger.error(f"Failed to create K-means frame: {str(e)}")
            raise

    def create_dbscan_frame(self):
        """Create frame for DBSCAN clustering options"""
        try:
            if not hasattr(self, 'dbscan_frame'):
                self.dbscan_frame = ttk.LabelFrame(self.cluster_frame, text="DBSCAN Parameters")
            
            # Clear existing widgets
            for widget in self.dbscan_frame.winfo_children():
                widget.destroy()
            
            # Epsilon parameter
            eps_frame = ttk.Frame(self.dbscan_frame)
            eps_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(eps_frame, text="Epsilon (ε):").pack(side=tk.LEFT)
            ttk.Entry(
                eps_frame,
                textvariable=self.eps,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Minimum samples
            min_samples_frame = ttk.Frame(self.dbscan_frame)
            min_samples_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(min_samples_frame, text="Minimum samples:").pack(side=tk.LEFT)
            ttk.Entry(
                min_samples_frame,
                textvariable=self.min_samples,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Add epsilon estimation button
            ttk.Button(
                self.dbscan_frame,
                text="Estimate Epsilon",
                command=self.estimate_epsilon
            ).pack(padx=5, pady=5)
            
            self.logger.info("Created DBSCAN parameter frame")
            
        except Exception as e:
            self.logger.error(f"Failed to create DBSCAN frame: {str(e)}")
            raise

    def create_hierarchical_frame(self):
        """Create frame for hierarchical clustering options"""
        try:
            if not hasattr(self, 'hierarchical_frame'):
                self.hierarchical_frame = ttk.LabelFrame(self.cluster_frame, text="Hierarchical Parameters")
            
            # Clear existing widgets
            for widget in self.hierarchical_frame.winfo_children():
                widget.destroy()
            
            # Number of clusters
            clusters_frame = ttk.Frame(self.hierarchical_frame)
            clusters_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(clusters_frame, text="Number of clusters:").pack(side=tk.LEFT)
            ttk.Entry(
                clusters_frame,
                textvariable=self.n_clusters_hierarchical,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Linkage method
            linkage_frame = ttk.Frame(self.hierarchical_frame)
            linkage_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(linkage_frame, text="Linkage method:").pack(side=tk.LEFT)
            ttk.OptionMenu(
                linkage_frame,
                self.linkage,
                "ward",
                "ward", "complete", "average", "single"
            ).pack(side=tk.LEFT, padx=5)
            
            # Add dendrogram button
            ttk.Button(
                self.hierarchical_frame,
                text="Show Dendrogram",
                command=self.show_dendrogram
            ).pack(padx=5, pady=5)
            
            self.logger.info("Created hierarchical clustering parameter frame")
            
        except Exception as e:
            self.logger.error(f"Failed to create hierarchical frame: {str(e)}")
            raise

    def save_clustering_results(self):
        """Save clustering results to file"""
        try:
            if self.normalized_data is None or self.cluster_labels is None:
                raise ValueError("No clustering results available")
                
            # Create results directory if it doesn't exist
            results_dir = self.results_dir / datetime.now().strftime('%Y%m%d_%H%M%S')
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save cluster assignments
            results_df = self.normalized_data.copy()
            results_df['Cluster'] = self.cluster_labels
            
            cluster_file = results_dir / 'cluster_assignments.csv'
            results_df.to_csv(cluster_file, index=False)
            
            # Calculate cluster statistics
            cluster_stats = pd.DataFrame({
                'Size': pd.Series(self.cluster_labels).value_counts(),
                'Percentage': pd.Series(self.cluster_labels).value_counts(normalize=True) * 100
            })
            
            stats_file = results_dir / 'cluster_statistics.csv'
            cluster_stats.to_csv(stats_file)
            
            # Calculate feature importance
            if hasattr(self, 'importance_results'):
                importance_file = results_dir / 'feature_importance.csv'
                self.importance_results['importances'].to_csv(importance_file, index=False)
            
            # Save clustering parameters
            params = {
                'method': self.cluster_method.get(),
                'parameters': self._get_clustering_parameters(),
                'metrics': {
                    'silhouette': silhouette_score(
                        self.normalized_data, 
                        self.cluster_labels
                    ),
                    'calinski_harabasz': calinski_harabasz_score(
                        self.normalized_data, 
                        self.cluster_labels
                    ),
                    'davies_bouldin': davies_bouldin_score(
                        self.normalized_data, 
                        self.cluster_labels
                    )
                },
                'timestamp': datetime.now().isoformat()
            }
            
            params_file = results_dir / 'clustering_parameters.json'
            with open(params_file, 'w') as f:
                json.dump(params, f, indent=4)
                
            # Generate detailed report
            report = [
                "# Clustering Analysis Report",
                f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"\n## Method: {self.cluster_method.get()}",
                "\n## Parameters:",
                json.dumps(self._get_clustering_parameters(), indent=2),
                "\n## Results:",
                f"\nTotal samples: {len(self.normalized_data)}",
                f"\nNumber of clusters: {len(np.unique(self.cluster_labels))}",
                "\n### Cluster Sizes:",
                cluster_stats.to_string(),
                "\n### Evaluation Metrics:",
                json.dumps(params['metrics'], indent=2)
            ]
            
            report_file = results_dir / 'detailed_report.md'
            with open(report_file, 'w') as f:
                f.write('\n'.join(report))
                
            # Save visualizations if they exist
            if hasattr(self, 'current_plot'):
                viz_dir = results_dir / 'visualizations'
                viz_dir.mkdir(exist_ok=True)
                
                # Save current plot
                plot_file = viz_dir / f'cluster_visualization.png'
                self.current_plot.savefig(plot_file, dpi=300, bbox_inches='tight')
                
                # Generate additional visualizations
                self._save_additional_visualizations(viz_dir)
            
            self.status_var.set(f"Results saved to {results_dir}")
            self.logger.info(f"Clustering results saved to {results_dir}")
            messagebox.showinfo("Success", f"Results saved to {results_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save clustering results: {str(e)}")
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")

    def _save_additional_visualizations(self, viz_dir: Path):
        """Generate and save additional visualizations"""
        try:
            # Feature importance plot
            if hasattr(self, 'importance_results'):
                plt.figure(figsize=(10, 6))
                sns.barplot(
                    data=self.importance_results['importances'],
                    x='Importance',
                    y='Feature'
                )
                plt.title('Feature Importance')
                plt.tight_layout()
                plt.savefig(viz_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
                plt.close()
                
            # Cluster distribution plot
            plt.figure(figsize=(10, 6))
            sns.countplot(x=self.cluster_labels)
            plt.title('Cluster Size Distribution')
            plt.xlabel('Cluster')
            plt.ylabel('Number of Samples')
            plt.tight_layout()
            plt.savefig(viz_dir / 'cluster_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # If dimensionality reduction was performed
            if hasattr(self, 'reduction_results'):
                reduced_data = self.reduction_results['reduced_data']
                if reduced_data.shape[1] == 2:
                    plt.figure(figsize=(10, 8))
                    plt.scatter(
                        reduced_data[:, 0],
                        reduced_data[:, 1],
                        c=self.cluster_labels,
                        cmap='viridis'
                    )
                    plt.title(f'{self.reduction_results["method"]} Projection')
                    plt.xlabel('Component 1')
                    plt.ylabel('Component 2')
                    plt.colorbar(label='Cluster')
                    plt.tight_layout()
                    plt.savefig(viz_dir / 'cluster_projection.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
            self.logger.info(f"Additional visualizations saved to {viz_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save additional visualizations: {str(e)}")
            # Don't raise - this is a non-critical error

    def generate_plot(self):
        """Generate visualization based on selected plot type"""
        try:
            if self.normalized_data is None:
                raise ValueError("No processed data available")
                
            if self.cluster_labels is None:
                raise ValueError("No clustering results available")
                
            plot_type = self.plot_type.get()
            
            # Clear previous plot
            self.fig.clear()
            
            if plot_type == "distribution":
                self._generate_distribution_plot()
            elif plot_type == "profile":
                self._generate_profile_plot()
            elif plot_type == "reduction":
                self._generate_reduction_plot()
            elif plot_type == "importance":
                self._generate_importance_plot()
                
            # Update canvas
            self.canvas.draw()
            
            # Store current plot for saving
            self.current_plot = self.fig
            
            self.logger.info(f"Generated {plot_type} plot successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to generate plot: {str(e)}")
            messagebox.showerror("Error", f"Failed to generate plot: {str(e)}")

    def _generate_distribution_plot(self):
        """Generate distribution plot for selected feature"""
        try:
            feature = self.dist_feature.get()
            plot_type = self.dist_type.get()
            
            if not feature:
                raise ValueError("No feature selected")
                
            ax = self.fig.add_subplot(111)
            
            if plot_type == "histogram":
                for label in np.unique(self.cluster_labels):
                    mask = self.cluster_labels == label
                    sns.histplot(
                        data=self.normalized_data[mask],
                        x=feature,
                        label=f'Cluster {label}',
                        ax=ax,
                        alpha=0.5
                    )
            elif plot_type == "kde":
                for label in np.unique(self.cluster_labels):
                    mask = self.cluster_labels == label
                    sns.kdeplot(
                        data=self.normalized_data[mask][feature],
                        label=f'Cluster {label}',
                        ax=ax
                    )
            else:  # box plot
                sns.boxplot(
                    data=self.normalized_data,
                    y=feature,
                    x=self.cluster_labels,
                    ax=ax
                )
                
            ax.set_title(f'{feature} Distribution by Cluster')
            ax.legend()
            
        except Exception as e:
            self.logger.error(f"Failed to generate distribution plot: {str(e)}")
            raise

    def _generate_profile_plot(self):
        """Generate cluster profile visualization"""
        try:
            plot_type = self.profile_type.get()
            selected_indices = self.feature_listbox.curselection()
            
            if not selected_indices:
                raise ValueError("No features selected")
                
            features = [self.feature_listbox.get(i) for i in selected_indices]
            data = self.normalized_data[features]
            
            if plot_type == "heatmap":
                # Calculate cluster means
                cluster_means = pd.DataFrame([
                    data[self.cluster_labels == label].mean()
                    for label in np.unique(self.cluster_labels)
                ], index=[f'Cluster {label}' for label in np.unique(self.cluster_labels)])
                
                ax = self.fig.add_subplot(111)
                sns.heatmap(
                    cluster_means,
                    cmap='coolwarm',
                    center=0,
                    annot=True,
                    fmt='.2f',
                    ax=ax
                )
                ax.set_title('Cluster Profiles Heatmap')
                
            elif plot_type == "parallel":
                # Create parallel coordinates plot
                ax = self.fig.add_subplot(111)
                pd.plotting.parallel_coordinates(
                    pd.concat([data, pd.Series(self.cluster_labels, name='Cluster')], axis=1),
                    'Cluster',
                    ax=ax
                )
                ax.set_title('Parallel Coordinates Plot')
                
            else:  # radar plot
                # Prepare data for radar plot
                angles = np.linspace(0, 2*np.pi, len(features), endpoint=False)
                
                # Calculate means for each cluster
                cluster_means = np.array([
                    data[self.cluster_labels == label].mean()
                    for label in np.unique(self.cluster_labels)
                ])
                
                # Create radar plot
                ax = self.fig.add_subplot(111, projection='polar')
                for i, means in enumerate(cluster_means):
                    values = np.concatenate((means, [means[0]]))  # Close the polygon
                    angles_plot = np.concatenate((angles, [angles[0]]))
                    ax.plot(angles_plot, values, label=f'Cluster {i}')
                    
                ax.set_xticks(angles)
                ax.set_xticklabels(features)
                ax.set_title('Radar Chart of Cluster Profiles')
                ax.legend()
                
        except Exception as e:
            self.logger.error(f"Failed to generate profile plot: {str(e)}")
            raise

    def _generate_reduction_plot(self):
        """Generate dimensionality reduction visualization"""
        try:
            method = self.reduction_method.get()
            n_components = self.n_components.get()
            
            if n_components not in [2, 3]:
                raise ValueError("Number of components must be 2 or 3 for visualization")
                
            # Perform dimensionality reduction
            if method == "pca":
                reducer = PCA(n_components=n_components, random_state=self.random_state)
            elif method == "tsne":
                reducer = TSNE(n_components=n_components, random_state=self.random_state)
            else:  # umap
                reducer = umap.UMAP(n_components=n_components, random_state=self.random_state)
                
            reduced_data = reducer.fit_transform(self.normalized_data)
            
            # Store reduction results
            self.reduction_results = {
                'method': method,
                'reduced_data': reduced_data,
                'model': reducer
            }
            
            # Create plot
            if n_components == 2:
                ax = self.fig.add_subplot(111)
                scatter = ax.scatter(
                    reduced_data[:, 0],
                    reduced_data[:, 1],
                    c=self.cluster_labels,
                    cmap='viridis'
                )
                ax.set_xlabel(f'{method.upper()} Component 1')
                ax.set_ylabel(f'{method.upper()} Component 2')
                
            else:  # 3D plot
                ax = self.fig.add_subplot(111, projection='3d')
                scatter = ax.scatter(
                    reduced_data[:, 0],
                    reduced_data[:, 1],
                    reduced_data[:, 2],
                    c=self.cluster_labels,
                    cmap='viridis'
                )
                ax.set_xlabel(f'{method.upper()} Component 1')
                ax.set_ylabel(f'{method.upper()} Component 2')
                ax.set_zlabel(f'{method.upper()} Component 3')
                
            self.fig.colorbar(scatter, label='Cluster')
            ax.set_title(f'{method.upper()} Projection of Clusters')
            
        except Exception as e:
            self.logger.error(f"Failed to generate reduction plot: {str(e)}")
            raise

    def _generate_importance_plot(self):
        """Generate feature importance visualization"""
        try:
            method = self.importance_method.get()
            n_features = self.n_top_features.get()
            
            # Calculate feature importance
            if method == "random_forest":
                clf = RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.random_state
                )
                clf.fit(self.normalized_data, self.cluster_labels)
                importances = clf.feature_importances_
                
            elif method == "mutual_info":
                from sklearn.feature_selection import mutual_info_classif
                importances = mutual_info_classif(
                    self.normalized_data,
                    self.cluster_labels,
                    random_state=self.random_state
                )
                
            else:  # chi2
                from sklearn.feature_selection import chi2
                importances, _ = chi2(
                    self.normalized_data,
                    self.cluster_labels
                )
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'Feature': self.normalized_data.columns,
                'Importance': importances
            })
            
            # Sort and select top features
            importance_df = importance_df.sort_values(
                'Importance',
                ascending=False
            ).head(n_features)
            
            # Store results
            self.analysis_results = {
                'type': 'feature_importance',
                'method': method,
                'results': importance_df
            }
            
            # Create plot
            ax = self.fig.add_subplot(111)
            sns.barplot(
                data=importance_df,
                x='Importance',
                y='Feature',
                ax=ax
            )
            ax.set_title(f'Top {n_features} Important Features ({method})')
            
        except Exception as e:
            self.logger.error(f"Failed to generate importance plot: {str(e)}")
            raise

    def save_plot(self):
        """Save the current plot to a file"""
        try:
            if not hasattr(self, 'current_plot'):
                raise ValueError("No plot available to save")

            # Open file dialog for saving
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("SVG files", "*.svg"),
                    ("All files", "*.*")
                ],
                title="Save Plot As"
            )
            
            if file_path:
                # Save the plot
                self.current_plot.savefig(
                    file_path,
                    dpi=300,
                    bbox_inches='tight'
                )
                
                self.logger.info(f"Plot saved to {file_path}")
                self.status_var.set(f"Plot saved to {file_path}")
                messagebox.showinfo("Success", f"Plot saved to {file_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to save plot: {str(e)}")
            messagebox.showerror("Error", f"Failed to save plot: {str(e)}")

    def run_analysis(self):
        """Run the selected analysis on the clustered data"""
        try:
            if self.normalized_data is None or self.cluster_labels is None:
                raise ValueError("No clustering results available")

            analysis_type = self.analysis_type.get()
            
            if analysis_type == "statistical":
                self._run_statistical_analysis()
            elif analysis_type == "stability":
                self._run_stability_analysis()
            else:  # feature importance
                self._run_feature_importance_analysis()
                
            self.logger.info(f"Analysis ({analysis_type}) completed successfully")
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")

    def _run_statistical_analysis(self):
        """Perform statistical analysis between clusters"""
        try:
            test_type = self.stat_test.get()
            results = []
            
            for column in self.normalized_data.columns:
                if test_type == "anova":
                    # Group data by clusters
                    groups = [
                        self.normalized_data[column][self.cluster_labels == label]
                        for label in np.unique(self.cluster_labels)
                    ]
                    stat, pval = f_oneway(*groups)
                    test_name = "ANOVA"
                    
                elif test_type == "chi2":
                    # Create contingency table
                    contingency = pd.crosstab(
                        self.cluster_labels,
                        self.normalized_data[column].astype('category')
                    )
                    stat, pval, _, _ = chi2_contingency(contingency)
                    test_name = "Chi-square"
                    
                else:  # t-test
                    # Compare each pair of clusters
                    pairs = []
                    for i in range(len(np.unique(self.cluster_labels))):
                        for j in range(i + 1, len(np.unique(self.cluster_labels))):
                            group1 = self.normalized_data[column][self.cluster_labels == i]
                            group2 = self.normalized_data[column][self.cluster_labels == j]
                            stat, pval = ttest_ind(group1, group2)
                            pairs.append((f"Cluster {i} vs {j}", stat, pval))
                    test_name = "T-test"
                
                results.append({
                    'Feature': column,
                    'Test': test_name,
                    'Statistic': stat,
                    'P-value': pval
                })
            
            # Create results DataFrame
            results_df = pd.DataFrame(results)
            
            # Apply multiple testing correction
            results_df['Adjusted P-value'] = multipletests(
                results_df['P-value'],
                method='fdr_bh'
            )[1]
            
            # Store results
            self.analysis_results = {
                'type': 'statistical',
                'test': test_type,
                'results': results_df
            }
            
            # Display results
            self._display_analysis_results()
            
        except Exception as e:
            self.logger.error(f"Statistical analysis failed: {str(e)}")
            raise

    def _run_stability_analysis(self):
        """Assess clustering stability through resampling"""
        try:
            n_iterations = self.n_iterations.get()
            subsample_size = self.subsample_size.get() / 100  # Convert percentage to proportion
            
            stability_scores = []
            
            for _ in tqdm.trange(n_iterations, desc="Assessing stability"):
                # Subsample data
                n_samples = int(len(self.normalized_data) * subsample_size)
                indices = np.random.choice(
                    len(self.normalized_data),
                    size=n_samples,
                    replace=False
                )
                subsample = self.normalized_data.iloc[indices]
                
                # Perform clustering on subsample
                if self.cluster_method.get() == "kmeans":
                    clusterer = KMeans(
                        n_clusters=self.n_clusters.get(),
                        random_state=self.random_state
                    )
                elif self.cluster_method.get() == "dbscan":
                    clusterer = DBSCAN(
                        eps=self.eps.get(),
                        min_samples=self.min_samples.get()
                    )
                else:  # hierarchical
                    clusterer = AgglomerativeClustering(
                        n_clusters=self.n_clusters_hierarchical.get(),
                        linkage=self.linkage.get()
                    )
                
                subsample_labels = clusterer.fit_predict(subsample)
                
                # Calculate stability metrics
                if -1 not in subsample_labels:  # Skip if DBSCAN found noise points
                    stability_scores.append({
                        'silhouette': silhouette_score(subsample, subsample_labels),
                        'calinski': calinski_harabasz_score(subsample, subsample_labels),
                        'davies_bouldin': davies_bouldin_score(subsample, subsample_labels)
                    })
            
            # Calculate stability statistics
            stability_df = pd.DataFrame(stability_scores)
            stability_stats = pd.DataFrame({
                'Metric': stability_df.columns,
                'Mean': stability_df.mean(),
                'Std': stability_df.std(),
                'CV': stability_df.std() / stability_df.mean() * 100  # Coefficient of variation
            })
            
            # Store results
            self.analysis_results = {
                'type': 'stability',
                'iterations': n_iterations,
                'subsample_size': subsample_size,
                'results': stability_stats,
                'raw_scores': stability_df
            }
            
            # Display results
            self._display_analysis_results()
            
        except Exception as e:
            self.logger.error(f"Stability analysis failed: {str(e)}")
            raise

    def _run_feature_importance_analysis(self):
        """Analyze feature importance for clustering results"""
        try:
            method = self.importance_method.get()
            n_features = self.n_top_features.get()
            
            if method == "random_forest":
                clf = RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.random_state
                )
                clf.fit(self.normalized_data, self.cluster_labels)
                importances = clf.feature_importances_
                
            elif method == "mutual_info":
                from sklearn.feature_selection import mutual_info_classif
                importances = mutual_info_classif(
                    self.normalized_data,
                    self.cluster_labels,
                    random_state=self.random_state
                )
                
            else:  # chi2
                from sklearn.feature_selection import chi2
                importances, pvalues = chi2(
                    self.normalized_data,
                    self.cluster_labels
                )
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'Feature': self.normalized_data.columns,
                'Importance': importances
            })
            
            # Sort and select top features
            importance_df = importance_df.sort_values(
                'Importance',
                ascending=False
            ).head(n_features)
            
            # Store results
            self.analysis_results = {
                'type': 'feature_importance',
                'method': method,
                'results': importance_df
            }
            
            # Display results
            self._display_analysis_results()
            
        except Exception as e:
            self.logger.error(f"Feature importance analysis failed: {str(e)}")
            raise

    def _display_analysis_results(self):
        """Display analysis results in the GUI"""
        try:
            if not hasattr(self, 'analysis_results'):
                raise ValueError("No analysis results available")
                
            # Clear previous results
            for widget in self.results_frame.winfo_children():
                widget.destroy()
                
            results = self.analysis_results
            
            if results['type'] == 'statistical':
                self._display_statistical_results()
            elif results['type'] == 'stability':
                self._display_stability_results()
            else:  # feature importance
                self._display_importance_results()
                
        except Exception as e:
            self.logger.error(f"Failed to display analysis results: {str(e)}")
            raise

    def _display_statistical_results(self):
        """Display statistical analysis results in the GUI"""
        try:
            results_df = self.analysis_results['results']
            test_type = self.analysis_results['test']
            
            # Create text widget for results
            results_text = tk.Text(
                self.results_frame,
                wrap=tk.WORD,
                height=20,
                width=60
            )
            results_text.pack(fill=tk.BOTH, expand=True)
            
            # Add scrollbar
            scrollbar = ttk.Scrollbar(
                self.results_frame,
                orient=tk.VERTICAL,
                command=results_text.yview
            )
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            results_text.configure(yscrollcommand=scrollbar.set)
            
            # Format and display results
            results_text.insert(tk.END, f"Statistical Analysis Results ({test_type})\n\n")
            results_text.insert(tk.END, results_df.to_string())
            
            # Make text widget read-only
            results_text.configure(state='disabled')
            
        except Exception as e:
            self.logger.error(f"Failed to display statistical results: {str(e)}")
            raise

    def _display_stability_results(self):
        """Display stability analysis results in the GUI"""
        try:
            stability_stats = self.analysis_results['results']
            raw_scores = self.analysis_results['raw_scores']
            
            # Create text widget for results
            results_text = tk.Text(
                self.results_frame,
                wrap=tk.WORD,
                height=20,
                width=60
            )
            results_text.pack(fill=tk.BOTH, expand=True)
            
            # Add scrollbar
            scrollbar = ttk.Scrollbar(
                self.results_frame,
                orient=tk.VERTICAL,
                command=results_text.yview
            )
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            results_text.configure(yscrollcommand=scrollbar.set)
            
            # Format and display results
            results_text.insert(tk.END, "Stability Analysis Results\n\n")
            results_text.insert(tk.END, f"Number of iterations: {self.analysis_results['iterations']}\n")
            results_text.insert(tk.END, f"Subsample size: {self.analysis_results['subsample_size']*100}%\n\n")
            results_text.insert(tk.END, "Stability Statistics:\n")
            results_text.insert(tk.END, stability_stats.to_string())
            
            # Make text widget read-only
            results_text.configure(state='disabled')
            
            # Create stability plot
            fig = Figure(figsize=(8, 4))
            ax = fig.add_subplot(111)
            raw_scores.boxplot(ax=ax)
            ax.set_title('Stability Metrics Distribution')
            ax.set_ylabel('Score')
            
            # Add plot to GUI
            canvas = FigureCanvasTkAgg(fig, master=self.results_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            self.logger.error(f"Failed to display stability results: {str(e)}")
            raise

    def _display_importance_results(self):
        """Display feature importance analysis results in the GUI"""
        try:
            importance_df = self.analysis_results['results']
            method = self.analysis_results['method']
            
            # Create text widget for results
            results_text = tk.Text(
                self.results_frame,
                wrap=tk.WORD,
                height=20,
                width=60
            )
            results_text.pack(fill=tk.BOTH, expand=True)
            
            # Add scrollbar
            scrollbar = ttk.Scrollbar(
                self.results_frame,
                orient=tk.VERTICAL,
                command=results_text.yview
            )
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            results_text.configure(yscrollcommand=scrollbar.set)
            
            # Format and display results
            results_text.insert(tk.END, f"Feature Importance Analysis Results ({method})\n\n")
            results_text.insert(tk.END, importance_df.to_string())
            
            # Make text widget read-only
            results_text.configure(state='disabled')
            
            # Create importance plot
            fig = Figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            importance_df.plot(
                kind='barh',
                x='Feature',
                y='Importance',
                ax=ax
            )
            ax.set_title('Feature Importance')
            
            # Add plot to GUI
            canvas = FigureCanvasTkAgg(fig, master=self.results_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            self.logger.error(f"Failed to display importance results: {str(e)}")
            raise

    def export_analysis(self):
        """Export analysis results to file"""
        try:
            if not hasattr(self, 'analysis_results'):
                raise ValueError("No analysis results available")
                
            # Open file dialog for saving
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[
                    ("CSV files", "*.csv"),
                    ("Excel files", "*.xlsx"),
                    ("JSON files", "*.json"),
                    ("All files", "*.*")
                ],
                title="Export Analysis Results"
            )
            
            if file_path:
                results = self.analysis_results
                
                # Export based on file type
                if file_path.endswith('.csv'):
                    results['results'].to_csv(file_path, index=False)
                elif file_path.endswith('.xlsx'):
                    with pd.ExcelWriter(file_path) as writer:
                        results['results'].to_excel(writer, sheet_name='Results', index=False)
                        if 'raw_scores' in results:
                            results['raw_scores'].to_excel(writer, sheet_name='Raw Scores', index=False)
                else:  # JSON
                    # Convert DataFrame to dict for JSON serialization
                    export_dict = {
                        'type': results['type'],
                        'results': results['results'].to_dict(orient='records')
                    }
                    if 'raw_scores' in results:
                        export_dict['raw_scores'] = results['raw_scores'].to_dict(orient='records')
                    
                    with open(file_path, 'w') as f:
                        json.dump(export_dict, f, indent=4)
                    
                self.logger.info(f"Analysis results exported to {file_path}")
                self.status_var.set(f"Results exported to {file_path}")
                messagebox.showinfo("Success", f"Results exported to {file_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to export analysis results: {str(e)}")
            messagebox.showerror("Error", f"Failed to export results: {str(e)}")

def main():
    """Main entry point for the clustering GUI application"""
    try:
        logger.info("Starting application")
        
        # Initialize root window
        root = tk.Tk()
        logger.debug("Root window initialized")

        # Create the GUI application instance
        app = ClusterAnalysisGUI(root)
        logger.debug("ClusterAnalysisGUI instance created")

        # Start the main event loop
        logger.info("Starting main event loop")
        root.mainloop()
        
    except Exception as e:
        logger.critical(f"Failed to start application: {str(e)}\n{traceback.format_exc()}")
        print(f"Failed to start application: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
