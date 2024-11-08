import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
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
from scipy import stats

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
        """Initialize GUI components with enhanced structure and error handling"""
        try:
            # Initialize main window
            self.master = master
            self.master.title("Political Psychology Cluster Analysis")
            self.master.geometry("1200x800")
            
            # Configure logging
            self.setup_logging()
            
            # Initialize data variables
            self.cleaned_data = None
            self.normalized_data = None
            self.random_state = 42
            
            # Setup directory structure
            self.setup_directories()
            
            # Create main container with tabs
            self.notebook = ttk.Notebook(self.master)
            self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Initialize tabs
            self.initialize_tabs()
            
            # Create status and progress bars
            self.create_status_bars()
            
            # Initialize additional variables
            self.initialize_variables()
            
            self.logger.info("GUI initialization completed successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize GUI: {str(e)}\n{traceback.format_exc()}")
            messagebox.showerror("Initialization Error", f"Failed to initialize application: {str(e)}")
            raise

    def setup_logging(self):
        """Configure logging with file and console handlers"""
        try:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            
            # Create logs directory
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            # File handler
            log_file = log_dir / f"cluster_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter('%(levelname)s: %(message)s')
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            
        except Exception as e:
            print(f"Failed to setup logging: {str(e)}")
            raise

    def setup_directories(self):
        """Create and validate directory structure"""
        try:
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
                
            # Validate directory creation
            for directory in [self.output_dir, self.models_dir, self.plots_dir, 
                             self.results_dir, self.temp_dir]:
                if not directory.exists():
                    raise RuntimeError(f"Failed to create directory: {directory}")
                    
            self.logger.info("Directory structure created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to setup directories: {str(e)}")
            raise

    def initialize_tabs(self):
        """Initialize and configure all tab components"""
        try:
            # Create tab frames
            self.data_tab = ttk.Frame(self.notebook)
            self.cluster_tab = ttk.Frame(self.notebook)
            self.viz_tab = ttk.Frame(self.notebook)
            self.analysis_tab = ttk.Frame(self.notebook)
            
            # Add tabs to notebook
            self.notebook.add(self.data_tab, text="Data Processing")
            self.notebook.add(self.cluster_tab, text="Clustering")
            self.notebook.add(self.viz_tab, text="Visualization")
            self.notebook.add(self.analysis_tab, text="Analysis")
            
            # Initialize tab contents
            self.create_data_tab()
            self.create_cluster_tab()
            self.create_viz_tab()
            self.create_analysis_tab()
            
            self.logger.info("Tabs initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize tabs: {str(e)}")
            raise

    def create_status_bars(self):
        """Create and configure status and progress bars"""
        try:
            # Status bar
            self.status_var = tk.StringVar()
            self.status_bar = ttk.Label(
                self.master,
                textvariable=self.status_var,
                relief=tk.SUNKEN,
                anchor=tk.W
            )
            self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
            
            # Progress bar
            self.progress_var = tk.DoubleVar()
            self.progress_bar = ttk.Progressbar(
                self.master,
                variable=self.progress_var,
                maximum=100,
                mode='determinate'
            )
            self.progress_bar.pack(side=tk.BOTTOM, fill=tk.X)
            
            # Initial status
            self.status_var.set("Ready")
            self.progress_var.set(0)
            
            self.logger.info("Status bars created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create status bars: {str(e)}")
            raise

    def initialize_variables(self):
        """Initialize all variables and data structures"""
        try:
            # Plot variables
            self.current_plot = None
            self.plot_data = None
            self.selected_features = []
            
            # Data processing variables
            self.file_path = tk.StringVar()
            self.handle_missing = tk.BooleanVar(value=True)
            self.remove_outliers = tk.BooleanVar(value=True)
            self.normalize = tk.BooleanVar(value=True)
            
            # Clustering variables
            self.cluster_method = tk.StringVar(value="kmeans")
            self.n_clusters = tk.IntVar(value=3)
            self.eps = tk.DoubleVar(value=0.5)
            self.min_samples = tk.IntVar(value=5)
            
            # Analysis variables
            self.analysis_type = tk.StringVar(value="statistical")
            self.stat_test = tk.StringVar(value="anova")
            
            # Method variables
            self.missing_method = tk.StringVar(value="median")
            self.outlier_method = tk.StringVar(value="iqr")
            self.norm_method = tk.StringVar(value="standard")
            
            self.logger.info("Variables initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize variables: {str(e)}")
            raise

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
            
            # Data processing options
            self.create_processing_options()
            
            # Preview frame
            self.create_preview_frame()
            
            self.logger.info("Data tab created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create data tab: {str(e)}")
            raise

    def create_processing_options(self):
        """Create data processing options section"""
        try:
            options_frame = ttk.LabelFrame(self.data_tab, text="Processing Options")
            options_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Missing values
            missing_frame = ttk.Frame(options_frame)
            missing_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Checkbutton(
                missing_frame,
                text="Handle Missing Values",
                variable=self.handle_missing,
                command=self.update_missing_options
            ).pack(side=tk.LEFT)
            
            self.missing_method_menu = ttk.OptionMenu(
                missing_frame,
                self.missing_method,
                "median",
                "median", "mean", "mode", "drop"
            )
            self.missing_method_menu.pack(side=tk.LEFT, padx=5)
            
            # Outliers
            outlier_frame = ttk.Frame(options_frame)
            outlier_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Checkbutton(
                outlier_frame,
                text="Remove Outliers",
                variable=self.remove_outliers,
                command=self.update_outlier_options
            ).pack(side=tk.LEFT)
            
            self.outlier_method_menu = ttk.OptionMenu(
                outlier_frame,
                self.outlier_method,
                "iqr",
                "iqr", "zscore", "isolation_forest"
            )
            self.outlier_method_menu.pack(side=tk.LEFT, padx=5)
            
            # Normalization
            norm_frame = ttk.Frame(options_frame)
            norm_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Checkbutton(
                norm_frame,
                text="Normalize Features",
                variable=self.normalize,
                command=self.update_norm_options
            ).pack(side=tk.LEFT)
            
            self.norm_method_menu = ttk.OptionMenu(
                norm_frame,
                self.norm_method,
                "standard",
                "standard", "minmax", "robust"
            )
            self.norm_method_menu.pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            self.logger.error(f"Failed to create processing options: {str(e)}")
            raise

    def create_preview_frame(self):
        """Create data preview frame"""
        try:
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
            
        except Exception as e:
            self.logger.error(f"Failed to create preview frame: {str(e)}")
            raise

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

    def update_missing_options(self):
        """Update missing value handling options based on checkbox state"""
        try:
            if self.handle_missing.get():
                self.missing_method_menu.configure(state='normal')
                # Update preview of affected rows
                if self.cleaned_data is not None:
                    missing_count = self.cleaned_data.isnull().sum()
                    self.preview_text.delete(1.0, tk.END)
                    self.preview_text.insert(tk.END, "Missing Values Summary:\n\n")
                    self.preview_text.insert(tk.END, str(missing_count))
            else:
                self.missing_method_menu.configure(state='disabled')
            
            self.logger.info(f"Missing value handling {'enabled' if self.handle_missing.get() else 'disabled'}")
        except Exception as e:
            self.logger.error(f"Error updating missing options: {str(e)}")
            self.status_var.set("Error updating missing value options")
            messagebox.showerror("Error", f"Failed to update missing value options: {str(e)}")

    def update_outlier_options(self):
        """Update outlier handling options based on checkbox state with enhanced feedback"""
        try:
            if self.remove_outliers.get():
                self.outlier_method_menu.configure(state='normal')
                
                # Show outlier statistics if data is available
                if self.cleaned_data is not None:
                    numeric_cols = self.cleaned_data.select_dtypes(include=['float64', 'int64']).columns
                    outlier_stats = {}
                    
                    for col in numeric_cols:
                        Q1 = self.cleaned_data[col].quantile(0.25)
                        Q3 = self.cleaned_data[col].quantile(0.75)
                        IQR = Q3 - Q1
                        outliers = ((self.cleaned_data[col] < (Q1 - 1.5 * IQR)) | 
                                   (self.cleaned_data[col] > (Q3 + 1.5 * IQR))).sum()
                        outlier_stats[col] = outliers
                    
                    self.preview_text.delete(1.0, tk.END)
                    self.preview_text.insert(tk.END, "Outlier Summary:\n\n")
                    for col, count in outlier_stats.items():
                        self.preview_text.insert(tk.END, f"{col}: {count} outliers\n")
            else:
                self.outlier_method_menu.configure(state='disabled')
            
            self.logger.info(f"Outlier handling {'enabled' if self.remove_outliers.get() else 'disabled'}")
        except Exception as e:
            self.logger.error(f"Error updating outlier options: {str(e)}")
            self.status_var.set("Error updating outlier options")
            messagebox.showerror("Error", f"Failed to update outlier options: {str(e)}")

    def update_norm_options(self):
        """Update normalization options with visualization"""
        try:
            if self.normalize.get():
                self.norm_method_menu.configure(state='normal')
                
                # Show distribution comparison if data is available
                if self.cleaned_data is not None:
                    numeric_cols = self.cleaned_data.select_dtypes(include=['float64', 'int64']).columns
                    
                    # Create distribution plots
                    fig, axes = plt.subplots(len(numeric_cols), 2, figsize=(12, 4*len(numeric_cols)))
                    fig.suptitle('Data Distribution Before/After Normalization')
                    
                    for i, col in enumerate(numeric_cols):
                        # Original distribution
                        sns.histplot(self.cleaned_data[col], ax=axes[i, 0])
                        axes[i, 0].set_title(f'Original {col}')
                        
                        # Normalized distribution (preview)
                        if self.norm_method.get() == 'standard':
                            normalized = StandardScaler().fit_transform(self.cleaned_data[[col]])
                        else:
                            normalized = MinMaxScaler().fit_transform(self.cleaned_data[[col]])
                        
                        sns.histplot(normalized, ax=axes[i, 1])
                        axes[i, 1].set_title(f'Normalized {col}')
                    
                    plt.tight_layout()
                    
                    # Update preview with plot
                    if hasattr(self, 'norm_canvas'):
                        self.norm_canvas.get_tk_widget().destroy()
                    
                    self.norm_canvas = FigureCanvasTkAgg(fig, master=self.data_tab)
                    self.norm_canvas.draw()
                    self.norm_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            else:
                self.norm_method_menu.configure(state='disabled')
                if hasattr(self, 'norm_canvas'):
                    self.norm_canvas.get_tk_widget().destroy()
            
            self.logger.info(f"Normalization {'enabled' if self.normalize.get() else 'disabled'}")
        except Exception as e:
            self.logger.error(f"Error updating normalization options: {str(e)}")
            self.status_var.set("Error updating normalization options")
            messagebox.showerror("Error", f"Failed to update normalization options: {str(e)}")

    def process_data(self):
        """Enhanced data processing with progress tracking and validation"""
        try:
            if not self.file_path.get():
                raise ValueError("No file selected")
            
            self.status_var.set("Processing data...")
            self.progress_var.set(0)
            self.master.update_idletasks()
            
            # Validate file existence
            file_path = Path(self.file_path.get())
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Read data with progress updates
            self.status_var.set("Reading data file...")
            self.progress_var.set(10)
            
            if file_path.suffix.lower() == '.csv':
                self.cleaned_data = pd.read_csv(file_path)
            elif file_path.suffix.lower() == '.xlsx':
                self.cleaned_data = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            # Validate data
            if self.cleaned_data.empty:
                raise ValueError("File contains no data")
            
            # Data type validation and conversion
            self.status_var.set("Validating data types...")
            self.progress_var.set(20)
            
            numeric_cols = self.cleaned_data.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in data")
            
            # Handle missing values
            if self.handle_missing.get():
                self.status_var.set("Handling missing values...")
                self.progress_var.set(40)
                
                method = self.missing_method.get()
                missing_counts_before = self.cleaned_data.isnull().sum()
                
                if method == 'median':
                    self.cleaned_data = self.cleaned_data.fillna(self.cleaned_data.median())
                elif method == 'mean':
                    self.cleaned_data = self.cleaned_data.fillna(self.cleaned_data.mean())
                elif method == 'mode':
                    self.cleaned_data = self.cleaned_data.fillna(self.cleaned_data.mode().iloc[0])
                elif method == 'drop':
                    rows_before = len(self.cleaned_data)
                    self.cleaned_data = self.cleaned_data.dropna()
                    rows_dropped = rows_before - len(self.cleaned_data)
                    self.logger.info(f"Dropped {rows_dropped} rows with missing values")
                
                missing_counts_after = self.cleaned_data.isnull().sum()
                self.logger.info(f"Missing values before/after: {missing_counts_before.sum()}/{missing_counts_after.sum()}")
            
            # Handle outliers
            if self.remove_outliers.get():
                self.status_var.set("Removing outliers...")
                self.progress_var.set(60)
                
                method = self.outlier_method.get()
                rows_before = len(self.cleaned_data)
                
                if method == 'iqr':
                    Q1 = self.cleaned_data.quantile(0.25)
                    Q3 = self.cleaned_data.quantile(0.75)
                    IQR = Q3 - Q1
                    self.cleaned_data = self.cleaned_data[~((self.cleaned_data < (Q1 - 1.5 * IQR)) | 
                                                          (self.cleaned_data > (Q3 + 1.5 * IQR))).any(axis=1)]
                elif method == 'zscore':
                    from scipy import stats
                    z_scores = np.abs(stats.zscore(self.cleaned_data[numeric_cols]))
                    self.cleaned_data = self.cleaned_data[(z_scores < 3).all(axis=1)]
                
                rows_removed = rows_before - len(self.cleaned_data)
                self.logger.info(f"Removed {rows_removed} outlier rows")
            
            # Normalize data
            if self.normalize.get():
                self.status_var.set("Normalizing data...")
                self.progress_var.set(80)
                
                method = self.norm_method.get()
                if method == 'standard':
                    scaler = StandardScaler()
                elif method == 'minmax':
                    scaler = MinMaxScaler()
                elif method == 'robust':
                    from sklearn.preprocessing import RobustScaler
                    scaler = RobustScaler()
                
                self.normalized_data = self.cleaned_data.copy()
                self.normalized_data[numeric_cols] = scaler.fit_transform(self.cleaned_data[numeric_cols])
                
                # Save scaler for later use
                joblib.dump(scaler, self.models_dir / f'{method}_scaler.joblib')
            
            # Generate and save data summary
            self.status_var.set("Generating data summary...")
            self.progress_var.set(90)
            
            summary = {
                'n_rows': len(self.cleaned_data),
                'n_columns': len(self.cleaned_data.columns),
                'numeric_columns': list(numeric_cols),
                'categorical_columns': list(self.cleaned_data.select_dtypes(include=['object']).columns),
                'missing_values': self.cleaned_data.isnull().sum().to_dict(),
                'processing_steps': {
                    'missing_values_handled': self.handle_missing.get(),
                    'outliers_removed': self.remove_outliers.get(),
                    'normalized': self.normalize.get()
                }
            }
            
            with open(self.results_dir / 'data_summary.json', 'w') as f:
                json.dump(summary, f, indent=4)
            
            # Update preview with processed data summary
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(tk.END, "Data Processing Summary:\n\n")
            self.preview_text.insert(tk.END, f"Rows: {summary['n_rows']}\n")
            self.preview_text.insert(tk.END, f"Columns: {summary['n_columns']}\n\n")
            self.preview_text.insert(tk.END, "Sample of processed data:\n\n")
            self.preview_text.insert(tk.END, str(self.cleaned_data.head()))
            
            self.status_var.set("Data processing complete")
            self.progress_var.set(100)
            self.logger.info("Data processing completed successfully")
            
        except Exception as e:
            self.logger.error(f"Data processing error: {str(e)}\n{traceback.format_exc()}")
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Data processing failed: {str(e)}")
            self.progress_var.set(0)

    def perform_clustering(self):
        """Perform clustering analysis with selected parameters"""
        try:
            if self.normalized_data is None:
                raise ValueError("No processed data available")
                
            method = self.cluster_method.get()
            
            if method == 'kmeans':
                n_clusters = self.n_clusters.get()
                max_iter = self.max_iter.get()
                
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    max_iter=max_iter,
                    random_state=self.random_state
                )
                labels = kmeans.fit_predict(self.normalized_data)
                
            elif method == 'dbscan':
                eps = self.eps.get()
                min_samples = self.min_samples.get()
                
                dbscan = DBSCAN(
                    eps=eps,
                    min_samples=min_samples
                )
                labels = dbscan.fit_predict(self.normalized_data)
                
            # Store results
            self.normalized_data['Cluster'] = labels
            
            # Update results text
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Clustering complete\n")
            self.results_text.insert(tk.END, f"Number of clusters: {len(np.unique(labels))}\n")
            self.results_text.insert(tk.END, f"Samples per cluster:\n{pd.Series(labels).value_counts()}\n")
            
            self.status_var.set("Clustering complete")
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))
            self.logger.error(f"Clustering error: {str(e)}")

    def generate_plot(self):
        """Generate visualization based on selected options"""
        try:
            if self.normalized_data is None:
                raise ValueError("No processed data available")
                
            plot_type = self.plot_type.get()
            
            if plot_type == 'distribution':
                feature = self.dist_feature.get()
                fig = plt.figure(figsize=(10, 6))
                sns.histplot(data=self.normalized_data, x=feature, hue='Cluster')
                plt.title(f'Distribution of {feature} by Cluster')
                
            elif plot_type == 'profile':
                fig = plt.figure(figsize=(12, 8))
                cluster_means = self.normalized_data.groupby('Cluster').mean()
                sns.heatmap(cluster_means, cmap='coolwarm', center=0)
                plt.title('Cluster Profiles')
                
            self.current_plot = fig
            
            # Display plot in GUI
            if hasattr(self, 'canvas'):
                self.canvas.get_tk_widget().destroy()
            
            self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))
            self.logger.error(f"Plot generation error: {str(e)}")

    def save_plot(self):
        """Save current plot to file"""
        try:
            if self.current_plot is None:
                raise ValueError("No plot to save")
                
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            
            if file_path:
                self.current_plot.savefig(file_path, dpi=300, bbox_inches='tight')
                self.status_var.set(f"Plot saved to {file_path}")
                
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))
            self.logger.error(f"Plot saving error: {str(e)}")

    def run_analysis(self):
        """Run selected analysis on clustering results"""
        try:
            if self.normalized_data is None:
                raise ValueError("No processed data available")
                
            analysis_type = self.analysis_type.get()
            
            if analysis_type == 'statistical':
                # Perform statistical tests between clusters
                test = self.stat_test.get()
                results = []
                
                for column in self.normalized_data.select_dtypes(include=['float64', 'int64']):
                    if test == 'anova':
                        groups = [group for _, group in self.normalized_data.groupby('Cluster')[column]]
                        f_stat, p_val = f_oneway(*groups)
                        results.append({'Feature': column, 'F-statistic': f_stat, 'p-value': p_val})
                        
                # Display results
                self.analysis_text.delete(1.0, tk.END)
                self.analysis_text.insert(tk.END, "Statistical Analysis Results:\n\n")
                for result in results:
                    self.analysis_text.insert(tk.END, f"{result['Feature']}:\n")
                    self.analysis_text.insert(tk.END, f"F-statistic: {result['F-statistic']:.4f}\n")
                    self.analysis_text.insert(tk.END, f"p-value: {result['p-value']:.4f}\n\n")
                    
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))
            self.logger.error(f"Analysis error: {str(e)}")

    def export_analysis(self):
        """Export analysis results to file"""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(self.analysis_text.get(1.0, tk.END))
                self.status_var.set(f"Analysis results saved to {file_path}")
                
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))
            self.logger.error(f"Export error: {str(e)}")

    def create_cluster_tab(self):
        """Create and configure clustering tab with advanced options"""
        try:
            # Method selection frame
            method_frame = ttk.LabelFrame(self.cluster_tab, text="Clustering Method")
            method_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Clustering method selection
            ttk.Label(method_frame, text="Method:").pack(side=tk.LEFT, padx=5)
            self.cluster_method_menu = ttk.OptionMenu(
                method_frame,
                self.cluster_method,
                "kmeans",
                "kmeans", "dbscan", "hierarchical",
                command=self.update_cluster_options
            )
            self.cluster_method_menu.pack(side=tk.LEFT, padx=5)
            
            # Parameters frames
            self.create_kmeans_frame()
            self.create_dbscan_frame()
            self.create_hierarchical_frame()
            
            # Optimization frame
            opt_frame = ttk.LabelFrame(self.cluster_tab, text="Optimization")
            opt_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Button(
                opt_frame,
                text="Find Optimal Parameters",
                command=self.find_optimal_clusters
            ).pack(side=tk.LEFT, padx=5)
            
            self.opt_method = tk.StringVar(value="elbow")
            ttk.Radiobutton(
                opt_frame,
                text="Elbow Method",
                variable=self.opt_method,
                value="elbow"
            ).pack(side=tk.LEFT, padx=5)
            ttk.Radiobutton(
                opt_frame,
                text="Silhouette Analysis",
                variable=self.opt_method,
                value="silhouette"
            ).pack(side=tk.LEFT, padx=5)
            
            # Results frame
            results_frame = ttk.LabelFrame(self.cluster_tab, text="Clustering Results")
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
            button_frame = ttk.Frame(self.cluster_tab)
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
            
            self.logger.info("Cluster tab created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create cluster tab: {str(e)}")
            raise

    def create_viz_tab(self):
        """Create and configure visualization tab with multiple plot options"""
        try:
            # Plot type selection frame
            type_frame = ttk.LabelFrame(self.viz_tab, text="Visualization Type")
            type_frame.pack(fill=tk.X, padx=5, pady=5)
            
            self.plot_type = tk.StringVar(value="distribution")
            plot_types = [
                ("Distribution Plot", "distribution"),
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
            
            # Options frames
            self.create_distribution_frame()
            self.create_profile_frame()
            self.create_reduction_frame()
            self.create_importance_frame()
            
            # Plot frame
            plot_container = ttk.LabelFrame(self.viz_tab, text="Visualization")
            plot_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Toolbar frame
            self.toolbar_frame = ttk.Frame(plot_container)
            self.toolbar_frame.pack(fill=tk.X)
            
            # Plot frame
            self.plot_frame = ttk.Frame(plot_container)
            self.plot_frame.pack(fill=tk.BOTH, expand=True)
            
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
            
            ttk.Button(
                button_frame,
                text="Export Data",
                command=self.export_plot_data
            ).pack(side=tk.LEFT, padx=5)
            
            self.logger.info("Visualization tab created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create visualization tab: {str(e)}")
            raise

    def create_analysis_tab(self):
        """Create and configure analysis tab with statistical tests and reporting"""
        try:
            # Analysis type selection frame
            type_frame = ttk.LabelFrame(self.analysis_tab, text="Analysis Type")
            type_frame.pack(fill=tk.X, padx=5, pady=5)
            
            analysis_types = [
                ("Statistical Tests", "statistical"),
                ("Feature Importance", "importance"),
                ("Cluster Stability", "stability"),
                ("Cluster Validation", "validation")
            ]
            
            for text, value in analysis_types:
                ttk.Radiobutton(
                    type_frame,
                    text=text,
                    variable=self.analysis_type,
                    value=value,
                    command=self.update_analysis_options
                ).pack(side=tk.LEFT, padx=5)
            
            # Create specific analysis frames
            self.create_statistical_frame()
            self.create_importance_analysis_frame()
            self.create_stability_frame()
            self.create_validation_frame()
            
            # Results frame
            analysis_results = ttk.LabelFrame(self.analysis_tab, text="Analysis Results")
            analysis_results.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Add scrollbars
            analysis_scroll_y = ttk.Scrollbar(analysis_results)
            analysis_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
            
            analysis_scroll_x = ttk.Scrollbar(analysis_results, orient=tk.HORIZONTAL)
            analysis_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
            
            # Results text widget
            self.analysis_text = tk.Text(
                analysis_results,
                wrap=tk.NONE,
                xscrollcommand=analysis_scroll_x.set,
                yscrollcommand=analysis_scroll_y.set
            )
            self.analysis_text.pack(fill=tk.BOTH, expand=True)
            
            # Configure scrollbars
            analysis_scroll_y.config(command=self.analysis_text.yview)
            analysis_scroll_x.config(command=self.analysis_text.xview)
            
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
            
            ttk.Button(
                button_frame,
                text="Generate Report",
                command=self.generate_report
            ).pack(side=tk.LEFT, padx=5)
            
            self.logger.info("Analysis tab created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create analysis tab: {str(e)}")
            raise

    def create_kmeans_frame(self):
        """Create frame for K-means clustering parameters"""
        try:
            self.kmeans_frame = ttk.LabelFrame(self.cluster_tab, text="K-means Parameters")
            
            # Number of clusters
            cluster_frame = ttk.Frame(self.kmeans_frame)
            cluster_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(cluster_frame, text="Number of clusters:").pack(side=tk.LEFT)
            ttk.Entry(
                cluster_frame,
                textvariable=self.n_clusters,
                width=10
            ).pack(side=tk.LEFT, padx=5)
            
            # Maximum iterations
            iter_frame = ttk.Frame(self.kmeans_frame)
            iter_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(iter_frame, text="Max iterations:").pack(side=tk.LEFT)
            self.max_iter = tk.IntVar(value=300)
            ttk.Entry(
                iter_frame,
                textvariable=self.max_iter,
                width=10
            ).pack(side=tk.LEFT, padx=5)
            
            # Initialize method
            init_frame = ttk.Frame(self.kmeans_frame)
            init_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(init_frame, text="Initialization:").pack(side=tk.LEFT)
            self.init_method = tk.StringVar(value="k-means++")
            ttk.OptionMenu(
                init_frame,
                self.init_method,
                "k-means++",
                "k-means++", "random"
            ).pack(side=tk.LEFT, padx=5)
            
            self.logger.info("K-means parameter frame created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create K-means frame: {str(e)}")
            raise

    def create_dbscan_frame(self):
        """Create frame for DBSCAN clustering parameters"""
        try:
            self.dbscan_frame = ttk.LabelFrame(self.cluster_tab, text="DBSCAN Parameters")
            
            # Epsilon
            eps_frame = ttk.Frame(self.dbscan_frame)
            eps_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(eps_frame, text="Epsilon:").pack(side=tk.LEFT)
            ttk.Entry(
                eps_frame,
                textvariable=self.eps,
                width=10
            ).pack(side=tk.LEFT, padx=5)
            
            # Minimum samples
            min_samples_frame = ttk.Frame(self.dbscan_frame)
            min_samples_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(min_samples_frame, text="Min samples:").pack(side=tk.LEFT)
            ttk.Entry(
                min_samples_frame,
                textvariable=self.min_samples,
                width=10
            ).pack(side=tk.LEFT, padx=5)
            
            # Metric
            metric_frame = ttk.Frame(self.dbscan_frame)
            metric_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(metric_frame, text="Distance metric:").pack(side=tk.LEFT)
            self.metric = tk.StringVar(value="euclidean")
            ttk.OptionMenu(
                metric_frame,
                self.metric,
                "euclidean",
                "euclidean", "manhattan", "cosine"
            ).pack(side=tk.LEFT, padx=5)
            
            self.logger.info("DBSCAN parameter frame created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create DBSCAN frame: {str(e)}")
            raise

    def create_hierarchical_frame(self):
        """Create frame for hierarchical clustering parameters"""
        try:
            self.hierarchical_frame = ttk.LabelFrame(
                self.cluster_tab,
                text="Hierarchical Clustering Parameters"
            )
            
            # Number of clusters
            cluster_frame = ttk.Frame(self.hierarchical_frame)
            cluster_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(cluster_frame, text="Number of clusters:").pack(side=tk.LEFT)
            self.h_n_clusters = tk.IntVar(value=3)
            ttk.Entry(
                cluster_frame,
                textvariable=self.h_n_clusters,
                width=10
            ).pack(side=tk.LEFT, padx=5)
            
            # Linkage
            linkage_frame = ttk.Frame(self.hierarchical_frame)
            linkage_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(linkage_frame, text="Linkage:").pack(side=tk.LEFT)
            self.linkage = tk.StringVar(value="ward")
            ttk.OptionMenu(
                linkage_frame,
                self.linkage,
                "ward",
                "ward", "complete", "average", "single"
            ).pack(side=tk.LEFT, padx=5)
            
            # Affinity
            affinity_frame = ttk.Frame(self.hierarchical_frame)
            affinity_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(affinity_frame, text="Affinity:").pack(side=tk.LEFT)
            self.affinity = tk.StringVar(value="euclidean")
            ttk.OptionMenu(
                affinity_frame,
                self.affinity,
                "euclidean",
                "euclidean", "manhattan", "cosine"
            ).pack(side=tk.LEFT, padx=5)
            
            self.logger.info("Hierarchical clustering parameter frame created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create hierarchical clustering frame: {str(e)}")
            raise

    def create_distribution_frame(self):
        """Create frame for distribution plot options"""
        try:
            self.dist_frame = ttk.LabelFrame(self.viz_tab, text="Distribution Plot Options")
            
            # Feature selection
            feature_frame = ttk.Frame(self.dist_frame)
            feature_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(feature_frame, text="Feature:").pack(side=tk.LEFT)
            self.dist_feature = tk.StringVar()
            self.feature_menu = ttk.OptionMenu(
                feature_frame,
                self.dist_feature,
                "",  # default value
                command=self.update_distribution_plot
            )
            self.feature_menu.pack(side=tk.LEFT, padx=5)
            
            # Plot type
            type_frame = ttk.Frame(self.dist_frame)
            type_frame.pack(fill=tk.X, padx=5, pady=2)
            
            self.dist_type = tk.StringVar(value="histogram")
            ttk.Radiobutton(
                type_frame,
                text="Histogram",
                variable=self.dist_type,
                value="histogram"
            ).pack(side=tk.LEFT, padx=5)
            ttk.Radiobutton(
                type_frame,
                text="KDE",
                variable=self.dist_type,
                value="kde"
            ).pack(side=tk.LEFT, padx=5)
            ttk.Radiobutton(
                type_frame,
                text="Box Plot",
                variable=self.dist_type,
                value="box"
            ).pack(side=tk.LEFT, padx=5)
            
            self.logger.info("Distribution frame created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create distribution frame: {str(e)}")
            raise

    def create_profile_frame(self):
        """Create frame for cluster profile visualization options"""
        try:
            self.profile_frame = ttk.LabelFrame(self.viz_tab, text="Cluster Profile Options")
            
            # Visualization type
            viz_frame = ttk.Frame(self.profile_frame)
            viz_frame.pack(fill=tk.X, padx=5, pady=2)
            
            self.profile_type = tk.StringVar(value="heatmap")
            ttk.Radiobutton(
                viz_frame,
                text="Heatmap",
                variable=self.profile_type,
                value="heatmap"
            ).pack(side=tk.LEFT, padx=5)
            ttk.Radiobutton(
                viz_frame,
                text="Parallel Coordinates",
                variable=self.profile_type,
                value="parallel"
            ).pack(side=tk.LEFT, padx=5)
            ttk.Radiobutton(
                viz_frame,
                text="Radar Chart",
                variable=self.profile_type,
                value="radar"
            ).pack(side=tk.LEFT, padx=5)
            
            # Feature selection
            feature_frame = ttk.Frame(self.profile_frame)
            feature_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(feature_frame, text="Features:").pack(side=tk.LEFT)
            self.feature_listbox = tk.Listbox(
                feature_frame,
                selectmode=tk.MULTIPLE,
                height=5
            )
            self.feature_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            # Scrollbar for feature listbox
            feature_scroll = ttk.Scrollbar(feature_frame, orient=tk.VERTICAL)
            feature_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Configure scrollbar
            self.feature_listbox.config(yscrollcommand=feature_scroll.set)
            feature_scroll.config(command=self.feature_listbox.yview)
            
            self.logger.info("Profile frame created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create profile frame: {str(e)}")
            raise

    def create_reduction_frame(self):
        """Create frame for dimensionality reduction visualization options"""
        try:
            self.reduction_frame = ttk.LabelFrame(self.viz_tab, text="Dimensionality Reduction Options")
            
            # Method selection
            method_frame = ttk.Frame(self.reduction_frame)
            method_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(method_frame, text="Method:").pack(side=tk.LEFT)
            self.reduction_method = tk.StringVar(value="pca")
            ttk.OptionMenu(
                method_frame,
                self.reduction_method,
                "pca",
                "pca", "tsne", "umap",
                command=self.update_reduction_options
            ).pack(side=tk.LEFT, padx=5)
            
            # Parameters frame
            self.reduction_params = ttk.Frame(self.reduction_frame)
            self.reduction_params.pack(fill=tk.X, padx=5, pady=2)
            
            # Components
            comp_frame = ttk.Frame(self.reduction_params)
            comp_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(comp_frame, text="Components:").pack(side=tk.LEFT)
            self.n_components = tk.IntVar(value=2)
            ttk.Entry(
                comp_frame,
                textvariable=self.n_components,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Additional parameters (method-specific)
            self.additional_params = ttk.Frame(self.reduction_params)
            self.additional_params.pack(fill=tk.X, pady=2)
            
            self.logger.info("Reduction frame created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create reduction frame: {str(e)}")
            raise

    def create_importance_frame(self):
        """Create frame for feature importance visualization options"""
        try:
            self.importance_frame = ttk.LabelFrame(self.viz_tab, text="Feature Importance Options")
            
            # Method selection
            method_frame = ttk.Frame(self.importance_frame)
            method_frame.pack(fill=tk.X, padx=5, pady=2)
            
            self.importance_method = tk.StringVar(value="statistical")
            ttk.Radiobutton(
                method_frame,
                text="Statistical Tests",
                variable=self.importance_method,
                value="statistical"
            ).pack(side=tk.LEFT, padx=5)
            ttk.Radiobutton(
                method_frame,
                text="Random Forest",
                variable=self.importance_method,
                value="random_forest"
            ).pack(side=tk.LEFT, padx=5)
            
            # Parameters frame
            params_frame = ttk.Frame(self.importance_frame)
            params_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Number of features
            ttk.Label(params_frame, text="Number of top features:").pack(side=tk.LEFT)
            self.n_top_features = tk.IntVar(value=10)
            ttk.Entry(
                params_frame,
                textvariable=self.n_top_features,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            self.logger.info("Feature importance analysis frame created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create feature importance analysis frame: {str(e)}")
            raise

    def create_statistical_frame(self):
        """Create frame for statistical analysis options"""
        try:
            self.stat_frame = ttk.LabelFrame(self.analysis_tab, text="Statistical Analysis Options")
            
            # Test selection
            test_frame = ttk.Frame(self.stat_frame)
            test_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(test_frame, text="Statistical Test:").pack(side=tk.LEFT)
            self.stat_test = tk.StringVar(value="anova")
            ttk.OptionMenu(
                test_frame,
                self.stat_test,
                "anova",
                "anova", "kruskal", "chi2",
                command=self.update_stat_options
            ).pack(side=tk.LEFT, padx=5)
            
            # Significance level
            alpha_frame = ttk.Frame(self.stat_frame)
            alpha_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(alpha_frame, text="Significance level (α):").pack(side=tk.LEFT)
            self.alpha = tk.DoubleVar(value=0.05)
            ttk.Entry(
                alpha_frame,
                textvariable=self.alpha,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Multiple testing correction
            correction_frame = ttk.Frame(self.stat_frame)
            correction_frame.pack(fill=tk.X, padx=5, pady=2)
            
            self.correction = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                correction_frame,
                text="Apply multiple testing correction",
                variable=self.correction
            ).pack(side=tk.LEFT)
            
            self.correction_method = tk.StringVar(value="bonferroni")
            ttk.OptionMenu(
                correction_frame,
                self.correction_method,
                "bonferroni",
                "bonferroni", "fdr", "holm"
            ).pack(side=tk.LEFT, padx=5)
            
            self.logger.info("Statistical frame created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create statistical frame: {str(e)}")
            raise

    def create_importance_analysis_frame(self):
        """Create frame for feature importance analysis options"""
        try:
            self.importance_analysis_frame = ttk.LabelFrame(
                self.analysis_tab,
                text="Feature Importance Analysis"
            )
            
            # Method selection
            method_frame = ttk.Frame(self.importance_analysis_frame)
            method_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(method_frame, text="Method:").pack(side=tk.LEFT)
            self.importance_analysis_method = tk.StringVar(value="random_forest")
            ttk.OptionMenu(
                method_frame,
                self.importance_analysis_method,
                "random_forest",
                "random_forest", "mutual_info", "chi2"
            ).pack(side=tk.LEFT, padx=5)
            
            # Parameters frame
            params_frame = ttk.Frame(self.importance_analysis_frame)
            params_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Number of features
            ttk.Label(params_frame, text="Number of top features:").pack(side=tk.LEFT)
            self.n_top_features = tk.IntVar(value=10)
            ttk.Entry(
                params_frame,
                textvariable=self.n_top_features,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            self.logger.info("Feature importance analysis frame created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create feature importance analysis frame: {str(e)}")
            raise

    def create_stability_frame(self):
        """Create frame for cluster stability analysis options"""
        try:
            self.stability_frame = ttk.LabelFrame(self.analysis_tab, text="Cluster Stability Analysis")
            
            # Number of iterations
            iter_frame = ttk.Frame(self.stability_frame)
            iter_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(iter_frame, text="Number of iterations:").pack(side=tk.LEFT)
            self.n_iterations = tk.IntVar(value=100)
            ttk.Entry(
                iter_frame,
                textvariable=self.n_iterations,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Subsample size
            sample_frame = ttk.Frame(self.stability_frame)
            sample_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(sample_frame, text="Subsample size (%):").pack(side=tk.LEFT)
            self.subsample_size = tk.IntVar(value=80)
            ttk.Entry(
                sample_frame,
                textvariable=self.subsample_size,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            self.logger.info("Stability frame created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create stability frame: {str(e)}")
            raise

    def create_validation_frame(self):
        """Create frame for cluster validation metrics"""
        try:
            self.validation_frame = ttk.LabelFrame(self.analysis_tab, text="Cluster Validation")
            
            # Metrics selection
            metrics_frame = ttk.Frame(self.validation_frame)
            metrics_frame.pack(fill=tk.X, padx=5, pady=2)
            
            self.silhouette = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                metrics_frame,
                text="Silhouette Score",
                variable=self.silhouette
            ).pack(side=tk.LEFT, padx=5)
            
            self.calinski = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                metrics_frame,
                text="Calinski-Harabasz Score",
                variable=self.calinski
            ).pack(side=tk.LEFT, padx=5)
            
            self.davies = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                metrics_frame,
                text="Davies-Bouldin Score",
                variable=self.davies
            ).pack(side=tk.LEFT, padx=5)
            
            # Cross-validation options
            cv_frame = ttk.Frame(self.validation_frame)
            cv_frame.pack(fill=tk.X, padx=5, pady=2)
            
            self.use_cv = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                cv_frame,
                text="Use Cross-validation",
                variable=self.use_cv,
                command=self.update_cv_options
            ).pack(side=tk.LEFT)
            
            self.n_splits = tk.IntVar(value=5)
            self.splits_entry = ttk.Entry(
                cv_frame,
                textvariable=self.n_splits,
                width=5,
                state='disabled'
            )
            self.splits_entry.pack(side=tk.LEFT, padx=5)
            
            self.logger.info("Validation frame created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create validation frame: {str(e)}")
            raise

if __name__ == "__main__":
    root = tk.Tk()
    app = ClusterAnalysisGUI(root)
    root.mainloop()
