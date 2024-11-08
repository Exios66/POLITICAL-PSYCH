import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score
from scipy.stats import f_oneway, chi2_contingency, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.neighbors import NearestNeighbors
import tqdm
from kneed import KneeLocator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Survey metrics categories
SURVEY_METRICS = {
    'news_consumption': [
        'Trad_News_print', 'Trad_News_online', 'Trad_News_TV', 'Trad_News_radio',
        'SM_News_1', 'SM_News_2', 'SM_News_3', 'SM_News_4', 'SM_News_5', 'SM_News_6',
        'News_frequency', 'SM_Sharing'
    ],
    'temporal': ['survey_date']
}

class ClusteringError(Exception):
    """Custom exception for clustering-related errors"""
    pass

class VisualizationError(Exception):
    """Custom exception for visualization-related errors"""
    pass

class ClusterAnalysisGUI:
    def __init__(self, master):
        """Initialize GUI components"""
        self.master = master
        self.master.title("Political Psychology Cluster Analysis")
        self.master.geometry("1200x800")
        
        # Create main container with tabs
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.data_tab = ttk.Frame(self.notebook)
        self.cluster_tab = ttk.Frame(self.notebook)
        self.viz_tab = ttk.Frame(self.notebook)
        self.analysis_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.data_tab, text="Data Processing")
        self.notebook.add(self.cluster_tab, text="Clustering")
        self.notebook.add(self.viz_tab, text="Visualization")
        self.notebook.add(self.analysis_tab, text="Analysis")
        
        # Initialize components
        self.create_data_tab()
        self.create_cluster_tab()
        self.create_viz_tab()
        self.create_analysis_tab()
        
        # Initialize analyzer
        self.analyzer = None
        
        # Create status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(
            self.master, 
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.master,
            variable=self.progress_var,
            maximum=100,
            mode='determinate'
        )
        self.progress_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Initialize data variables
        self.current_plot = None
        self.plot_data = None
        self.selected_features = []
        
    def create_data_tab(self):
        """Create data processing tab components"""
        # File selection
        file_frame = ttk.LabelFrame(self.data_tab, text="Data File")
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.file_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path, width=60).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=5
        )
        ttk.Button(
            file_frame, 
            text="Browse",
            command=self.browse_file
        ).pack(side=tk.LEFT, padx=5)
        
        # Data processing options
        process_frame = ttk.LabelFrame(self.data_tab, text="Processing Options")
        process_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Missing values handling
        missing_frame = ttk.Frame(process_frame)
        missing_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.handle_missing = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            missing_frame,
            text="Handle Missing Values",
            variable=self.handle_missing,
            command=self.update_missing_options
        ).pack(side=tk.LEFT)
        
        self.missing_method = tk.StringVar(value="median")
        self.missing_method_menu = ttk.OptionMenu(
            missing_frame,
            self.missing_method,
            "median",
            "median", "mean", "mode", "drop"
        )
        self.missing_method_menu.pack(side=tk.LEFT, padx=5)
        
        # Outlier handling
        outlier_frame = ttk.Frame(process_frame)
        outlier_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.remove_outliers = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            outlier_frame,
            text="Remove Outliers",
            variable=self.remove_outliers,
            command=self.update_outlier_options
        ).pack(side=tk.LEFT)
        
        self.outlier_method = tk.StringVar(value="iqr")
        self.outlier_method_menu = ttk.OptionMenu(
            outlier_frame,
            self.outlier_method,
            "iqr",
            "iqr", "zscore", "isolation_forest"
        )
        self.outlier_method_menu.pack(side=tk.LEFT, padx=5)
        
        # Normalization
        norm_frame = ttk.Frame(process_frame)
        norm_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.normalize = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            norm_frame,
            text="Normalize Features",
            variable=self.normalize,
            command=self.update_norm_options
        ).pack(side=tk.LEFT)
        
        self.norm_method = tk.StringVar(value="standard")
        self.norm_method_menu = ttk.OptionMenu(
            norm_frame,
            self.norm_method,
            "standard",
            "standard", "minmax", "robust"
        )
        self.norm_method_menu.pack(side=tk.LEFT, padx=5)
        
        # Feature selection
        feature_frame = ttk.LabelFrame(self.data_tab, text="Feature Selection")
        feature_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.feature_list = tk.Listbox(
            feature_frame,
            selectmode=tk.MULTIPLE,
            height=6
        )
        self.feature_list.pack(fill=tk.X, padx=5, pady=5)
        
        # Process button
        ttk.Button(
            self.data_tab,
            text="Process Data",
            command=self.process_data
        ).pack(pady=10)
        
        # Data preview
        preview_frame = ttk.LabelFrame(self.data_tab, text="Data Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add scrollbars
        preview_scroll_y = ttk.Scrollbar(preview_frame)
        preview_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        preview_scroll_x = ttk.Scrollbar(preview_frame, orient=tk.HORIZONTAL)
        preview_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.preview_text = tk.Text(
            preview_frame,
            wrap=tk.NONE,
            xscrollcommand=preview_scroll_x.set,
            yscrollcommand=preview_scroll_y.set
        )
        self.preview_text.pack(fill=tk.BOTH, expand=True)
        
        preview_scroll_y.config(command=self.preview_text.yview)
        preview_scroll_x.config(command=self.preview_text.xview)
        
    def create_cluster_tab(self):
        """Create clustering tab components"""
        # Clustering options
        options_frame = ttk.LabelFrame(self.cluster_tab, text="Clustering Options")
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Method selection
        method_frame = ttk.Frame(options_frame)
        method_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(method_frame, text="Method:").pack(side=tk.LEFT)
        self.cluster_method = tk.StringVar(value="kmeans")
        method_combo = ttk.Combobox(
            method_frame,
            textvariable=self.cluster_method,
            values=["kmeans", "dbscan", "hierarchical", "gaussian_mixture"],
            state="readonly",
            width=20
        )
        method_combo.pack(side=tk.LEFT, padx=5)
        method_combo.bind('<<ComboboxSelected>>', self.update_cluster_options)
        
        # Parameters frame
        self.params_frame = ttk.LabelFrame(options_frame, text="Parameters")
        self.params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # K-means parameters
        self.kmeans_frame = ttk.Frame(self.params_frame)
        ttk.Label(self.kmeans_frame, text="Number of Clusters:").grid(
            row=0, column=0, padx=5, pady=2
        )
        self.n_clusters = tk.IntVar(value=3)
        ttk.Entry(
            self.kmeans_frame,
            textvariable=self.n_clusters,
            width=10
        ).grid(row=0, column=1, padx=5)
        
        ttk.Label(self.kmeans_frame, text="Max Iterations:").grid(
            row=1, column=0, padx=5, pady=2
        )
        self.max_iter = tk.IntVar(value=300)
        ttk.Entry(
            self.kmeans_frame,
            textvariable=self.max_iter,
            width=10
        ).grid(row=1, column=1, padx=5)
        
        # DBSCAN parameters
        self.dbscan_frame = ttk.Frame(self.params_frame)
        ttk.Label(self.dbscan_frame, text="Epsilon:").grid(
            row=0, column=0, padx=5, pady=2
        )
        self.eps = tk.DoubleVar(value=0.5)
        ttk.Entry(
            self.dbscan_frame,
            textvariable=self.eps,
            width=10
        ).grid(row=0, column=1, padx=5)
        
        ttk.Label(self.dbscan_frame, text="Min Samples:").grid(
            row=1, column=0, padx=5, pady=2
        )
        self.min_samples = tk.IntVar(value=5)
        ttk.Entry(
            self.dbscan_frame,
            textvariable=self.min_samples,
            width=10
        ).grid(row=1, column=1, padx=5)
        
        # Hierarchical parameters
        self.hierarchical_frame = ttk.Frame(self.params_frame)
        ttk.Label(self.hierarchical_frame, text="Number of Clusters:").grid(
            row=0, column=0, padx=5, pady=2
        )
        self.h_n_clusters = tk.IntVar(value=3)
        ttk.Entry(
            self.hierarchical_frame,
            textvariable=self.h_n_clusters,
            width=10
        ).grid(row=0, column=1, padx=5)
        
        ttk.Label(self.hierarchical_frame, text="Linkage:").grid(
            row=1, column=0, padx=5, pady=2
        )
        self.linkage = tk.StringVar(value="ward")
        ttk.OptionMenu(
            self.hierarchical_frame,
            self.linkage,
            "ward",
            "ward", "complete", "average", "single"
        ).grid(row=1, column=1, padx=5)
        
        # Show initial frame
        self.kmeans_frame.pack(fill=tk.X)
        
        # Optimization frame
        opt_frame = ttk.LabelFrame(options_frame, text="Optimization")
        opt_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.optimize = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            opt_frame,
            text="Find Optimal Parameters",
            variable=self.optimize,
            command=self.update_opt_options
        ).pack(side=tk.LEFT)
        
        self.opt_method = tk.StringVar(value="elbow")
        self.opt_method_menu = ttk.OptionMenu(
            opt_frame,
            self.opt_method,
            "elbow",
            "elbow", "silhouette", "gap"
        )
        self.opt_method_menu.pack(side=tk.LEFT, padx=5)
        
        # Action buttons
        button_frame = ttk.Frame(self.cluster_tab)
        button_frame.pack(pady=10)
        
        ttk.Button(
            button_frame,
            text="Find Optimal Parameters",
            command=self.find_optimal_clusters
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Perform Clustering",
            command=self.perform_clustering
        ).pack(side=tk.LEFT, padx=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(self.cluster_tab, text="Clustering Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add scrollbars
        results_scroll_y = ttk.Scrollbar(results_frame)
        results_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        results_scroll_x = ttk.Scrollbar(results_frame, orient=tk.HORIZONTAL)
        results_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.results_text = tk.Text(
            results_frame,
            wrap=tk.NONE,
            xscrollcommand=results_scroll_x.set,
            yscrollcommand=results_scroll_y.set
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        results_scroll_y.config(command=self.results_text.yview)
        results_scroll_x.config(command=self.results_text.xview)
        
    def create_viz_tab(self):
        """Create visualization tab components"""
        # Visualization options
        options_frame = ttk.LabelFrame(self.viz_tab, text="Visualization Options")
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Plot type selection
        plot_frame = ttk.Frame(options_frame)
        plot_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(plot_frame, text="Plot Type:").pack(side=tk.LEFT)
        self.plot_type = tk.StringVar(value="distribution")
        plot_combo = ttk.Combobox(
            plot_frame,
            textvariable=self.plot_type,
            values=[
                "distribution",
                "profile",
                "reduction",
                "silhouette",
                "elbow",
                "feature_importance"
            ],
            state="readonly",
            width=20
        )
        plot_combo.pack(side=tk.LEFT, padx=5)
        plot_combo.bind('<<ComboboxSelected>>', self.update_plot_options)
        
        # Plot options frame
        self.plot_options_frame = ttk.Frame(options_frame)
        self.plot_options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Distribution plot options
        self.dist_frame = ttk.Frame(self.plot_options_frame)
        ttk.Label(self.dist_frame, text="Feature:").pack(side=tk.LEFT)
        self.dist_feature = tk.StringVar()
        self.dist_feature_menu = ttk.OptionMenu(
            self.dist_frame,
            self.dist_feature,
            ""
        )
        self.dist_feature_menu.pack(side=tk.LEFT, padx=5)
        
        # Profile plot options
        self.profile_frame = ttk.Frame(self.plot_options_frame)
        self.normalize_profiles = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self.profile_frame,
            text="Normalize Profiles",
            variable=self.normalize_profiles
        ).pack(side=tk.LEFT)
        
        # Reduction plot options
        self.reduction_frame = ttk.Frame(self.plot_options_frame)
        ttk.Label(self.reduction_frame, text="Method:").pack(side=tk.LEFT)
        self.reduction_method = tk.StringVar(value="pca")
        ttk.OptionMenu(
            self.reduction_frame,
            self.reduction_method,
            "pca",
            "pca", "tsne", "umap"
        ).pack(side=tk.LEFT, padx=5)
        
        # Show initial frame
        self.dist_frame.pack(fill=tk.X)
        
        # Plot controls
        controls_frame = ttk.Frame(options_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(
            controls_frame,
            text="Generate Plot",
            command=self.generate_plot
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            controls_frame,
            text="Save Plot",
            command=self.save_plot
        ).pack(side=tk.LEFT, padx=5)
        
        # Plot frame
        self.plot_frame = ttk.Frame(self.viz_tab)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def create_analysis_tab(self):
        """Create analysis tab components"""
        # Analysis options
        options_frame = ttk.LabelFrame(self.analysis_tab, text="Analysis Options")
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Analysis type selection
        analysis_frame = ttk.Frame(options_frame)
        analysis_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(analysis_frame, text="Analysis Type:").pack(side=tk.LEFT)
        self.analysis_type = tk.StringVar(value="statistical")
        analysis_combo = ttk.Combobox(
            analysis_frame,
            textvariable=self.analysis_type,
            values=[
                "statistical",
                "feature_importance",
                "cluster_stability",
                "validation"
            ],
            state="readonly",
            width=20
        )
        analysis_combo.pack(side=tk.LEFT, padx=5)
        analysis_combo.bind('<<ComboboxSelected>>', self.update_analysis_options)
        
        # Analysis parameters frame
        self.analysis_params_frame = ttk.Frame(options_frame)
        self.analysis_params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Statistical analysis parameters
        self.stat_frame = ttk.Frame(self.analysis_params_frame)
        ttk.Label(self.stat_frame, text="Test:").pack(side=tk.LEFT)
        self.stat_test = tk.StringVar(value="anova")
        ttk.OptionMenu(
            self.stat_frame,
            self.stat_test,
            "anova",
            "anova", "chi2", "ttest"
        ).pack(side=tk.LEFT, padx=5)
        
        # Feature importance parameters
        self.importance_frame = ttk.Frame(self.analysis_params_frame)
        ttk.Label(self.importance_frame, text="Method:").pack(side=tk.LEFT)
        self.importance_method = tk.StringVar(value="permutation")
        ttk.OptionMenu(
            self.importance_frame,
            self.importance_method,
            "permutation",
            "permutation", "shap", "lime"
        ).pack(side=tk.LEFT, padx=5)
        
        # Stability analysis parameters
        self.stability_frame = ttk.Frame(self.analysis_params_frame)
        ttk.Label(self.stability_frame, text="Iterations:").pack(side=tk.LEFT)
        self.n_iterations = tk.IntVar(value=100)
        ttk.Entry(
            self.stability_frame,
            textvariable=self.n_iterations,
            width=10
        ).pack(side=tk.LEFT, padx=5)
        
        # Show initial frame
        self.stat_frame.pack(fill=tk.X)
        
        # Action buttons
        button_frame = ttk.Frame(self.analysis_tab)
        button_frame.pack(pady=10)
        
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
        
        # Results frame
        results_frame = ttk.LabelFrame(self.analysis_tab, text="Analysis Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add scrollbars
        analysis_scroll_y = ttk.Scrollbar(results_frame)
        analysis_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        analysis_scroll_x = ttk.Scrollbar(results_frame, orient=tk.HORIZONTAL)
        analysis_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.analysis_text = tk.Text(
            results_frame,
            wrap=tk.NONE,
            xscrollcommand=analysis_scroll_x.set,
            yscrollcommand=analysis_scroll_y.set
        )
        self.analysis_text.pack(fill=tk.BOTH, expand=True)
        
        analysis_scroll_y.config(command=self.analysis_text.yview)
        analysis_scroll_x.config(command=self.analysis_text.xview)
        # Get numeric columns first
        numeric_cols = self.cleaned_data.select_dtypes(include=['float64', 'int64']).columns

        # Add back non-numeric columns
        for col in self.cleaned_data.columns:
            if col not in numeric_cols:
                    self.normalized_data[col] = self.cleaned_data[col]
            
            try:
                # Save scaler with error handling
                scaler_path = self.models_dir / 'scaler.joblib'
                try:
                    scaler = StandardScaler()
                    joblib.dump(scaler, scaler_path)
                    self.logger.info(f"Saved scaler to {scaler_path}")
                except Exception as e:
                    self.logger.error(f"Failed to save scaler: {str(e)}")
                    raise RuntimeError(f"Could not save scaler to {scaler_path}")

                # Save normalized data with error handling
                norm_data_path = self.output_dir / 'normalized_data.csv'
                try:
                    self.normalized_data.to_csv(norm_data_path, index=False)
                    self.logger.info(f"Saved normalized data to {norm_data_path}")
                except Exception as e:
                    self.logger.error(f"Failed to save normalized data: {str(e)}")
                    raise RuntimeError(f"Could not save normalized data to {norm_data_path}")

                # Update GUI status
                self.status_var.set("Features normalized successfully")
                self.root.update_idletasks()

                # Log normalization details
                self.logger.info("Features normalized successfully")
                self.logger.info(f"Normalized columns: {list(numeric_cols)}")
                self.logger.info(f"Normalization stats:\n{scaler.get_params()}")

                # Add normalization info to analysis text
                self.analysis_text.insert(tk.END, "\n=== Feature Normalization ===\n")
                self.analysis_text.insert(tk.END, f"Normalized {len(numeric_cols)} columns\n")
                self.analysis_text.insert(tk.END, f"Columns: {', '.join(numeric_cols)}\n")
                self.analysis_text.insert(tk.END, f"Scaler: {type(scaler).__name__}\n")
                self.analysis_text.see(tk.END)

                # Enable analysis buttons now that normalization is complete
                for child in self.button_frame.winfo_children():
                    if isinstance(child, ttk.Button):
                        child.configure(state='normal')

            except Exception as e:
                # Update GUI status
                self.status_var.set("Error normalizing features")
                self.root.update_idletasks()

                # Disable analysis buttons
                for child in self.button_frame.winfo_children():
                    if isinstance(child, ttk.Button):
                        child.configure(state='disabled')

                # Log error details
                self.logger.error(f"Error normalizing features: {str(e)}")
                self.logger.error(f"Stack trace:\n{traceback.format_exc()}")

                # Show error in analysis text
                self.analysis_text.insert(tk.END, "\nERROR: Feature normalization failed\n")
                self.analysis_text.insert(tk.END, f"Details: {str(e)}\n")
                self.analysis_text.see(tk.END)

                raise RuntimeError(f"Feature normalization failed: {str(e)}")

    def find_optimal_clusters(
        self,
        method: str = 'elbow',
        max_k: int = 10,
        min_k: int = 2,
        n_init: int = 10
    ) -> int:
        """
        Find optimal number of clusters using specified method.
        
        Args:
            method: Method to use ('elbow', 'silhouette', or 'gap')
            max_k: Maximum number of clusters to try
            min_k: Minimum number of clusters to try
            n_init: Number of initializations for k-means
            
        Returns:
            Optimal number of clusters
            
        Raises:
            ValueError: If normalized_data is None or empty
            ValueError: If invalid method specified
            RuntimeError: If optimization fails
        """
        if self.normalized_data is None or self.normalized_data.empty:
            raise ValueError("No normalized data available")
            
        try:
            # Update GUI status
            self.status_var.set(f"Finding optimal clusters using {method} method...")
            self.root.update_idletasks()

            # Get numeric data
            numeric_data = self.normalized_data.select_dtypes(
                include=['float64', 'int64']
            )
            
            if len(numeric_data.columns) == 0:
                raise ValueError("No numeric columns available for clustering")
                
            # Validate input parameters
            if not isinstance(min_k, int) or min_k < 2:
                raise ValueError("min_k must be an integer >= 2")
            if not isinstance(max_k, int) or max_k <= min_k:
                raise ValueError("max_k must be an integer > min_k")
            if not isinstance(n_init, int) or n_init < 1:
                raise ValueError("n_init must be a positive integer")
                
            K = range(min_k, max_k + 1)
            
            if method == 'elbow':
                distortions = []
                inertias = []
                
                # Create progress bar
                progress_var = tk.DoubleVar()
                progress_bar = ttk.Progressbar(
                    self.stat_frame,
                    variable=progress_var,
                    maximum=len(K)
                )
                progress_bar.pack(fill=tk.X, padx=5, pady=5)
                
                # Progress bar for long computations
                with tqdm(total=len(K), desc="Computing elbow curve") as pbar:
                    for k in K:
                        kmeans = KMeans(
                            n_clusters=k,
                            n_init=n_init,
                            random_state=self.random_state
                        )
                        kmeans.fit(numeric_data)
                        distortions.append(kmeans.inertia_)
                        inertias.append(kmeans.inertia_)
                        
                        # Update progress
                        progress_var.set(k - min_k + 1)
                        self.root.update_idletasks()
                        pbar.update(1)
                        
                # Calculate first and second derivatives
                d1 = np.diff(distortions)
                d2 = np.diff(d1)
                
                # Find elbow using kneedle algorithm
                kneedle = KneeLocator(
                    list(K),
                    distortions,
                    curve='convex',
                    direction='decreasing',
                    S=1.0  # Sensitivity parameter
                )
                
                if kneedle.elbow is None:
                    # Fallback if no clear elbow found
                    optimal_k = K[len(K)//2]  # Use middle value
                    self.logger.warning("No clear elbow found, using middle value for k")
                else:
                    optimal_k = kneedle.elbow
                
                # Plot elbow curve with derivatives
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
                
                # Main elbow plot
                ax1.plot(K, distortions, 'bo-', label='Distortion')
                ax1.axvline(x=optimal_k, color='r', linestyle='--', 
                          label=f'Optimal k={optimal_k}')
                ax1.set_xlabel('Number of Clusters (k)')
                ax1.set_ylabel('Distortion')
                ax1.set_title('Elbow Method Analysis')
                ax1.legend()
                ax1.grid(True)
                
                # Add normalized curve
                norm_distortions = (distortions - np.min(distortions)) / (np.max(distortions) - np.min(distortions))
                ax1.plot(K, norm_distortions, 'g--', alpha=0.5, label='Normalized')
                
                # Derivatives plot
                ax2.plot(K[1:], d1, 'g.-', label='First Derivative')
                ax2.plot(K[2:], d2, 'm.-', label='Second Derivative') 
                ax2.set_xlabel('Number of Clusters (k)')
                ax2.set_ylabel('Rate of Change')
                ax2.set_title('Derivatives Analysis')
                ax2.legend()
                ax2.grid(True)
                
                # Add percentage change plot
                pct_changes = np.diff(distortions) / distortions[:-1] * 100
                ax3.plot(K[1:], pct_changes, 'r.-', label='% Change')
                ax3.set_xlabel('Number of Clusters (k)')
                ax3.set_ylabel('Percentage Change')
                ax3.set_title('Percentage Change in Distortion')
                ax3.legend()
                ax3.grid(True)
                
                plt.tight_layout()
                
                # Save high quality plot
                plot_path = self.plots_dir / 'elbow_analysis.png'
                try:
                    plt.savefig(
                        plot_path,
                        dpi=300,
                        bbox_inches='tight',
                        metadata={
                            'Title': 'Elbow Analysis',
                            'Author': 'ClusterAnalysis',
                            'Created': datetime.now().isoformat()
                        }
                    )
                    self.logger.info(f"Saved elbow analysis plot to {plot_path}")
                except Exception as e:
                    self.logger.error(f"Failed to save elbow plot: {str(e)}")
                plt.close()
                
                # Save raw data for later analysis
                elbow_data = {
                    'k_values': list(K),
                    'distortions': distortions,
                    'first_derivative': d1.tolist(),
                    'second_derivative': d2.tolist(),
                    'pct_changes': pct_changes.tolist(),
                    'optimal_k': optimal_k,
                    'normalized_distortions': norm_distortions.tolist()
                }
                
                # Save elbow analysis data with error handling
                data_path = self.results_dir / 'elbow_data.json'
                try:
                    with open(data_path, 'w') as f:
                        json.dump(elbow_data, f, indent=4)
                    self.logger.info(f"Saved elbow analysis data to {data_path}")
                except Exception as e:
                    self.logger.error(f"Failed to save elbow data: {str(e)}")
                    raise RuntimeError(f"Could not save elbow data to {data_path}")

            elif method == 'silhouette':
                # Initialize data structures
                silhouette_scores = []
                sample_silhouettes = []
                silhouette_data = {
                    'k_values': list(K),
                    'silhouette_scores': [],
                    'sample_silhouettes': [],
                    'optimal_k': None
                }
                
                # Create and configure progress bar
                progress_var = tk.DoubleVar()
                progress_bar = ttk.Progressbar(
                    self.stat_frame,
                    variable=progress_var,
                    maximum=len(K),
                    mode='determinate',
                    length=200
                )
                progress_bar.pack(fill=tk.X, padx=5, pady=5)
                
                # Add progress label
                progress_label = ttk.Label(
                    self.stat_frame, 
                    text="Computing silhouette scores..."
                )
                progress_label.pack()

                # Compute silhouette scores with error handling
                try:
                    with tqdm(total=len(K), desc="Computing silhouette scores") as pbar:
                        for k in K:
                            # Initialize and fit KMeans
                            kmeans = KMeans(
                                n_clusters=k,
                                n_init=n_init,
                                random_state=self.random_state,
                                max_iter=300
                            )
                            
                            try:
                                labels = kmeans.fit_predict(numeric_data)
                            except Exception as e:
                                self.logger.error(f"KMeans fitting failed for k={k}: {str(e)}")
                                raise RuntimeError(f"KMeans clustering failed for k={k}")
                            
                            # Calculate silhouette scores
                            try:
                                silhouette_avg = silhouette_score(
                                    numeric_data, 
                                    labels,
                                    random_state=self.random_state
                                )
                                sample_silhouette_values = silhouette_samples(
                                    numeric_data,
                                    labels
                                )
                            except Exception as e:
                                self.logger.error(f"Silhouette calculation failed for k={k}: {str(e)}")
                                raise RuntimeError(f"Silhouette score calculation failed for k={k}")
                            
                            # Store results
                            silhouette_scores.append(silhouette_avg)
                            sample_silhouettes.append(sample_silhouette_values)
                            silhouette_data['silhouette_scores'].append(float(silhouette_avg))
                            silhouette_data['sample_silhouettes'].append(sample_silhouette_values.tolist())
                            
                            # Update progress
                            progress_var.set(k - min_k + 1)
                            progress_label.config(text=f"Processing k={k}...")
                            self.root.update_idletasks()
                            pbar.update(1)
                            
                    # Find optimal k
                    optimal_k = K[np.argmax(silhouette_scores)]
                    silhouette_data['optimal_k'] = int(optimal_k)
                    
                    # Save silhouette analysis results
                    silhouette_path = self.results_dir / 'silhouette_analysis.json'
                    try:
                        with open(silhouette_path, 'w') as f:
                            json.dump(silhouette_data, f, indent=4)
                        self.logger.info(f"Saved silhouette analysis to {silhouette_path}")
                    except Exception as e:
                        self.logger.error(f"Failed to save silhouette data: {str(e)}")
                        raise RuntimeError(f"Could not save silhouette data to {silhouette_path}")
                        
                    # Create visualization
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
                    
                    # Silhouette score plot
                    ax1.plot(K, silhouette_scores, 'bo-')
                    ax1.axvline(x=optimal_k, color='r', linestyle='--',
                              label=f'Optimal k={optimal_k}')
                    ax1.set_xlabel('Number of Clusters (k)')
                    ax1.set_ylabel('Silhouette Score')
                    ax1.set_title('Silhouette Analysis')
                    ax1.legend()
                    ax1.grid(True)
                    
                    # Sample silhouette plot for optimal k
                    optimal_idx = K.index(optimal_k)
                    optimal_silhouettes = sample_silhouettes[optimal_idx]
                    y_lower = 10
                    
                    for i in range(optimal_k):
                        cluster_silhouettes = optimal_silhouettes[labels == i]
                        cluster_silhouettes.sort()
                        size = cluster_silhouettes.shape[0]
                        y_upper = y_lower + size
                        
                        color = plt.cm.nipy_spectral(float(i) / optimal_k)
                        ax2.fill_betweenx(np.arange(y_lower, y_upper),
                                        0, cluster_silhouettes,
                                        facecolor=color, alpha=0.7)
                        y_lower = y_upper + 10
                        
                    ax2.set_xlabel('Silhouette Coefficient')
                    ax2.set_ylabel('Cluster')
                    ax2.set_title(f'Silhouette Plot for k={optimal_k}')
                    ax2.axvline(x=silhouette_scores[optimal_idx], color='r',
                              linestyle='--', label='Average Score')
                    ax2.legend()
                    
                    plt.tight_layout()
                    
                    # Save plot
                    plot_path = self.plots_dir / 'silhouette_analysis.png'
                    try:
                        plt.savefig(
                            plot_path,
                            dpi=300,
                            bbox_inches='tight',
                            metadata={
                                'Title': 'Silhouette Analysis',
                                'Author': 'ClusterAnalysis',
                                'Created': datetime.now().isoformat()
                            }
                        )
                        self.logger.info(f"Saved silhouette plot to {plot_path}")
                    except Exception as e:
                        self.logger.error(f"Failed to save silhouette plot: {str(e)}")
                    finally:
                        plt.close()
                        
                finally:
                    # Clean up progress bar
                    progress_bar.destroy()
                    progress_label.destroy()
                    
            else:
                raise ValueError(f"Invalid method: {method}. Must be one of: 'elbow', 'silhouette', 'gap'")
            
            # Save optimization results
            optimization_results = {
                'method': method,
                'optimal_k': int(optimal_k),
                'min_k': min_k,
                'max_k': max_k,
                'n_init': n_init,
                'timestamp': datetime.now().isoformat(),
                'parameters': {
                    'random_state': self.random_state,
                    'n_init': n_init
                }
            }
            
            results_path = self.results_dir / 'optimization_results.json'
            try:
                with open(results_path, 'w') as f:
                    json.dump(optimization_results, f, indent=4)
                self.logger.info(f"Saved optimization results to {results_path}")
            except Exception as e:
                self.logger.error(f"Failed to save optimization results: {str(e)}")
                raise RuntimeError(f"Could not save optimization results to {results_path}")
            
            # Update analysis text
            self.analysis_text.insert(tk.END, "\n=== Cluster Optimization ===\n")
            self.analysis_text.insert(tk.END, f"Method: {method}\n")
            self.analysis_text.insert(tk.END, f"Optimal number of clusters: {optimal_k}\n")
            self.analysis_text.insert(tk.END, f"Search range: {min_k} to {max_k}\n")
            self.analysis_text.see(tk.END)
            
            self.logger.info(f"Optimal number of clusters found: {optimal_k}")
            self.logger.info(f"Method used: {method}")
            self.logger.info(f"Results saved to {results_path}")
            
            return optimal_k
            
        except Exception as e:
            self.logger.error(f"Error finding optimal clusters: {str(e)}")
            self.logger.error(f"Stack trace:\n{traceback.format_exc()}")
            
            # Update GUI with error
            self.status_var.set(f"Error: {str(e)}")
            self.analysis_text.insert(tk.END, f"\nERROR: {str(e)}\n")
            self.analysis_text.see(tk.END)
            
            raise RuntimeError(f"Cluster optimization failed: {str(e)}")
