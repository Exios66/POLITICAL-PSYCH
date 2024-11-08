import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.metrics import silhouette_score, calinski_harabasz_score
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
        
        self.notebook.add(self.data_tab, text="Data Processing")
        self.notebook.add(self.cluster_tab, text="Clustering")
        self.notebook.add(self.viz_tab, text="Visualization")
        
        # Initialize components
        self.create_data_tab()
        self.create_cluster_tab()
        self.create_viz_tab()
        
        # Initialize analyzer
        self.analyzer = None
        
        # Create status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(
            self.master, 
            textvariable=self.status_var,
            relief=tk.SUNKEN
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def create_data_tab(self):
        """Create data processing tab components"""
        # File selection
        file_frame = ttk.LabelFrame(self.data_tab, text="Data File")
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.file_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path).pack(
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
        
        self.handle_missing = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            process_frame,
            text="Handle Missing Values",
            variable=self.handle_missing
        ).pack(anchor=tk.W)
        
        self.remove_outliers = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            process_frame,
            text="Remove Outliers",
            variable=self.remove_outliers
        ).pack(anchor=tk.W)
        
        self.normalize = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            process_frame,
            text="Normalize Features",
            variable=self.normalize
        ).pack(anchor=tk.W)
        
        # Process button
        ttk.Button(
            self.data_tab,
            text="Process Data",
            command=self.process_data
        ).pack(pady=10)
        
        # Data preview
        preview_frame = ttk.LabelFrame(self.data_tab, text="Data Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.preview_text = tk.Text(preview_frame)
        self.preview_text.pack(fill=tk.BOTH, expand=True)
        
    def create_cluster_tab(self):
        """Create clustering tab components"""
        # Clustering options
        options_frame = ttk.LabelFrame(self.cluster_tab, text="Clustering Options")
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(options_frame, text="Method:").grid(row=0, column=0, padx=5, pady=5)
        self.cluster_method = tk.StringVar(value="kmeans")
        method_combo = ttk.Combobox(
            options_frame,
            textvariable=self.cluster_method,
            values=["kmeans", "dbscan", "hierarchical"]
        )
        method_combo.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(options_frame, text="Number of Clusters:").grid(
            row=1, column=0, padx=5, pady=5
        )
        self.n_clusters = tk.IntVar(value=3)
        ttk.Entry(
            options_frame,
            textvariable=self.n_clusters
        ).grid(row=1, column=1, padx=5, pady=5)
        
        # Find optimal clusters button
        ttk.Button(
            options_frame,
            text="Find Optimal Clusters",
            command=self.find_optimal_clusters
        ).grid(row=2, column=0, columnspan=2, pady=10)
        
        # Perform clustering button
        ttk.Button(
            options_frame,
            text="Perform Clustering",
            command=self.perform_clustering
        ).grid(row=3, column=0, columnspan=2, pady=10)
        
        # Results frame
        results_frame = ttk.LabelFrame(self.cluster_tab, text="Clustering Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.results_text = tk.Text(results_frame)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
    def create_viz_tab(self):
        """Create visualization tab components"""
        # Visualization options
        options_frame = ttk.LabelFrame(self.viz_tab, text="Visualization Options")
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(options_frame, text="Plot Type:").grid(row=0, column=0, padx=5, pady=5)
        self.plot_type = tk.StringVar(value="distribution")
        plot_combo = ttk.Combobox(
            options_frame,
            textvariable=self.plot_type,
            values=["distribution", "profile", "reduction"]
        )
        plot_combo.grid(row=0, column=1, padx=5, pady=5)
        
        # Plot button
        ttk.Button(
            options_frame,
            text="Generate Plot",
            command=self.generate_plot
        ).grid(row=1, column=0, columnspan=2, pady=10)
        
        # Plot frame
        self.plot_frame = ttk.Frame(self.viz_tab)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def browse_file(self):
        """Open file dialog to select data file"""
        filename = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv")]
        )
        if filename:
            self.file_path.set(filename)
            
    def process_data(self):
        """Process the data with selected options"""
        try:
            if not self.file_path.get():
                messagebox.showerror("Error", "Please select a data file")
                return
                
            self.analyzer = ClusterAnalysis(
                file_path=self.file_path.get(),
                handle_missing=self.handle_missing.get(),
                remove_outliers=self.remove_outliers.get(),
                normalize=self.normalize.get()
            )
            
            self.analyzer.load_data()
            if self.handle_missing.get():
                self.analyzer.clean_data()
            if self.normalize.get():
                self.analyzer.normalize_features()
                
            # Update preview
            self.preview_text.delete('1.0', tk.END)
            self.preview_text.insert('1.0', str(self.analyzer.data.head()))
            
            self.status_var.set("Data processed successfully")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            logger.error(f"Error processing data: {str(e)}")
            
    def find_optimal_clusters(self):
        """Find optimal number of clusters"""
        try:
            if self.analyzer is None:
                messagebox.showerror("Error", "Please process data first")
                return
                
            optimal_k = self.analyzer.find_optimal_clusters(
                method='elbow',
                max_k=10
            )
            
            self.n_clusters.set(optimal_k)
            self.status_var.set(f"Optimal number of clusters: {optimal_k}")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            logger.error(f"Error finding optimal clusters: {str(e)}")
            
    def perform_clustering(self):
        """Perform clustering with selected options"""
        try:
            if self.analyzer is None:
                messagebox.showerror("Error", "Please process data first")
                return
                
            self.analyzer.perform_clustering(
                method=self.cluster_method.get(),
                n_clusters=self.n_clusters.get()
            )
            
            # Update results
            self.results_text.delete('1.0', tk.END)
            self.results_text.insert('1.0', self.analyzer.get_cluster_summary())
            
            self.status_var.set("Clustering completed successfully")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            logger.error(f"Error performing clustering: {str(e)}")
            
    def generate_plot(self):
        """Generate selected visualization"""
        try:
            if self.analyzer is None or self.analyzer.clusters is None:
                messagebox.showerror("Error", "Please perform clustering first")
                return
                
            # Clear previous plot
            for widget in self.plot_frame.winfo_children():
                widget.destroy()
                
            fig = self.analyzer.visualize(plot_type=self.plot_type.get())
            
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
            toolbar.update()
            
            self.status_var.set(f"Generated {self.plot_type.get()} plot")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            logger.error(f"Error generating plot: {str(e)}")

class ClusterAnalysis:
    def __init__(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
        handle_missing: bool = True,
        remove_outliers: bool = True,
        normalize: bool = True,
        random_state: int = 42
    ):
        """
        Initialize ClusterAnalysis with data file path and processing options.
        
        Args:
            file_path: Path to the data file
            output_dir: Optional path to output directory for results
            handle_missing: Whether to handle missing values
            remove_outliers: Whether to remove outliers 
            normalize: Whether to normalize features
            random_state: Random seed for reproducibility
        """
        self.file_path = Path(file_path)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path('data/processed_data') / self.timestamp
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.plots_dir = self.output_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
        
        self.models_dir = self.output_dir / 'models'
        self.models_dir.mkdir(exist_ok=True)
        
        self.results_dir = self.output_dir / 'results'
        self.results_dir.mkdir(exist_ok=True)
        
        # Processing flags
        self.handle_missing = handle_missing
        self.remove_outliers = remove_outliers 
        self.normalize = normalize
        self.random_state = random_state
        
        # Initialize data attributes
        self.data = None
        self.cleaned_data = None
        self.normalized_data = None
        self.clusters = None
        self.n_clusters = None
        self.feature_importance = None
        self.cluster_profiles = None
        self.cluster_metrics = None
        self.model = None
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Add file handler
        log_file = self.output_dir / 'cluster_analysis.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"ClusterAnalysis initialized with output directory: {self.output_dir}")

    def load_data(self) -> None:
        """
        Load data from the specified file path.
        
        Raises:
            FileNotFoundError: If the data file doesn't exist
            pd.errors.EmptyDataError: If the data file is empty
            ValueError: If data format is invalid
        """
        try:
            # Check file extension
            extension = self.file_path.suffix.lower()
            
            if extension == '.csv':
                self.data = pd.read_csv(self.file_path)
            elif extension in ['.xls', '.xlsx']:
                self.data = pd.read_excel(self.file_path)
            elif extension == '.json':
                self.data = pd.read_json(self.file_path)
            else:
                raise ValueError(f"Unsupported file format: {extension}")
            
            # Validate data
            if self.data.empty:
                raise pd.errors.EmptyDataError("Data file is empty")
                
            # Check for numeric columns
            numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in data")
            
            # Convert date columns
            date_cols = self.data.select_dtypes(include=['object']).apply(
                lambda x: pd.to_datetime(x, errors='coerce').notna().all()
            )
            for col in date_cols[date_cols].index:
                self.data[col] = pd.to_datetime(self.data[col])
            
            # Save raw data copy
            self.data.to_csv(self.output_dir / 'raw_data.csv', index=False)
            
            self.logger.info(f"Data loaded successfully from {self.file_path}")
            self.logger.info(f"Shape: {self.data.shape}")
            self.logger.info(f"Columns: {list(self.data.columns)}")
            
        except FileNotFoundError:
            self.logger.error(f"Data file not found: {self.file_path}")
            raise
            
        except pd.errors.EmptyDataError:
            self.logger.error(f"Data file is empty: {self.file_path}")
            raise
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def clean_data(self) -> None:
        """
        Clean the loaded data by handling missing values and outliers.
        
        Raises:
            ValueError: If data is None or empty
            RuntimeError: If cleaning fails
        """
        if self.data is None or self.data.empty:
            raise ValueError("No data to clean. Load data first.")
            
        try:
            # Create copy of raw data
            self.cleaned_data = self.data.copy()
            
            # Get column types
            numeric_cols = self.cleaned_data.select_dtypes(
                include=['float64', 'int64']
            ).columns
            cat_cols = self.cleaned_data.select_dtypes(
                include=['object']
            ).columns
            date_cols = self.cleaned_data.select_dtypes(
                include=['datetime64']
            ).columns
            
            # Handle missing values
            if self.handle_missing:
                # For numeric columns
                if len(numeric_cols) > 0:
                    # Check if more than 50% missing
                    missing_pct = self.cleaned_data[numeric_cols].isnull().mean()
                    cols_to_drop = missing_pct[missing_pct > 0.5].index
                    
                    if len(cols_to_drop) > 0:
                        self.logger.warning(
                            f"Dropping columns with >50% missing values: {list(cols_to_drop)}"
                        )
                        self.cleaned_data = self.cleaned_data.drop(columns=cols_to_drop)
                        numeric_cols = numeric_cols.drop(cols_to_drop)
                    
                    # Fill remaining missing with median
                    self.cleaned_data[numeric_cols] = self.cleaned_data[numeric_cols].fillna(
                        self.cleaned_data[numeric_cols].median()
                    )
                
                # For categorical columns
                if len(cat_cols) > 0:
                    self.cleaned_data[cat_cols] = self.cleaned_data[cat_cols].fillna(
                        self.cleaned_data[cat_cols].mode().iloc[0]
                    )
                
                # For date columns
                if len(date_cols) > 0:
                    self.cleaned_data[date_cols] = self.cleaned_data[date_cols].fillna(
                        method='ffill'
                    )
            
            # Remove outliers using IQR method
            if self.remove_outliers:
                if len(numeric_cols) > 0:
                    Q1 = self.cleaned_data[numeric_cols].quantile(0.25)
                    Q3 = self.cleaned_data[numeric_cols].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    outlier_mask = ~((self.cleaned_data[numeric_cols] < (Q1 - 1.5 * IQR)) | 
                                    (self.cleaned_data[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
                    
                    n_outliers = (~outlier_mask).sum()
                    if n_outliers > 0:
                        self.logger.info(f"Removing {n_outliers} outlier rows")
                        self.cleaned_data = self.cleaned_data[outlier_mask]
            
            # Save cleaned data
            self.cleaned_data.to_csv(self.output_dir / 'cleaned_data.csv', index=False)
            
            # Log cleaning summary
            self.logger.info("Data cleaned successfully")
            self.logger.info(f"Original shape: {self.data.shape}")
            self.logger.info(f"Cleaned shape: {self.cleaned_data.shape}")
            
        except Exception as e:
            self.logger.error(f"Error cleaning data: {str(e)}")
            raise RuntimeError(f"Data cleaning failed: {str(e)}")

    def normalize_features(self) -> None:
        """
        Normalize features using StandardScaler.
        
        Raises:
            ValueError: If cleaned_data is None or empty
            RuntimeError: If normalization fails
        """
        if self.cleaned_data is None or self.cleaned_data.empty:
            raise ValueError("No cleaned data available. Clean data first.")
            
        try:
            # Get numeric columns
            numeric_cols = self.cleaned_data.select_dtypes(
                include=['float64', 'int64']
            ).columns
            
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns to normalize")
            
            # Initialize scaler
            scaler = StandardScaler()
            
            # Fit and transform numeric data
            normalized = scaler.fit_transform(self.cleaned_data[numeric_cols])
            
            # Create normalized dataframe
            self.normalized_data = pd.DataFrame(
                normalized,
                columns=numeric_cols,
                index=self.cleaned_data.index
            )
            
            # Add back non-numeric columns
            for col in self.cleaned_data.columns:
                if col not in numeric_cols:
                    self.normalized_data[col] = self.cleaned_data[col]
            
            # Save scaler
            joblib.dump(
                scaler,
                self.models_dir / 'scaler.joblib'
            )
            
            # Save normalized data
            self.normalized_data.to_csv(
                self.output_dir / 'normalized_data.csv',
                index=False
            )
            
            self.logger.info("Features normalized successfully")
            self.logger.info(f"Normalized columns: {list(numeric_cols)}")
            
        except Exception as e:
            self.logger.error(f"Error normalizing features: {str(e)}")
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
                        pbar.update(1)
                        
                # Calculate first and second derivatives
                d1 = np.diff(distortions)
                d2 = np.diff(d1)
                
                # Find elbow using kneedle algorithm
                elbow_idx = KneeLocator(
                    list(K),
                    distortions,
                    curve='convex',
                    direction='decreasing'
                ).elbow_y
                
                optimal_k = K[distortions.index(elbow_idx)]
                
                # Plot elbow curve with derivatives
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                # Main elbow plot
                ax1.plot(K, distortions, 'bo-', label='Distortion')
                ax1.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
                ax1.set_xlabel('Number of Clusters (k)')
                ax1.set_ylabel('Distortion')
                ax1.set_title('Elbow Method Analysis')
                ax1.legend()
                ax1.grid(True)
                
                # Derivatives plot
                ax2.plot(K[1:], d1, 'g.-', label='First Derivative')
                ax2.plot(K[2:], d2, 'm.-', label='Second Derivative')
                ax2.set_xlabel('Number of Clusters (k)')
                ax2.set_ylabel('Rate of Change')
                ax2.set_title('Derivatives Analysis')
                ax2.legend()
                ax2.grid(True)
                
                plt.tight_layout()
                plt.savefig(self.plots_dir / 'elbow_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
                
            elif method == 'silhouette':
                silhouette_scores = []
                sample_silhouettes = []
                
                with tqdm(total=len(K), desc="Computing silhouette scores") as pbar:
                    for k in K:
                        kmeans = KMeans(
                            n_clusters=k,
                            n_init=n_init,
                            random_state=self.random_state
                        )
                        labels = kmeans.fit_predict(numeric_data)
                        
                        # Calculate full silhouette score
                        silhouette_avg = silhouette_score(numeric_data, labels)
                        silhouette_scores.append(silhouette_avg)
                        
                        # Calculate per-sample silhouette scores
                        sample_silhouette_values = silhouette_samples(numeric_data, labels)
                        sample_silhouettes.append(sample_silhouette_values)
                        
                        pbar.update(1)
                        
                optimal_k = K[np.argmax(silhouette_scores)]
                
                # Create detailed silhouette plot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
                
                # Plot average silhouette scores
                ax1.plot(K, silhouette_scores, 'bo-')
                ax1.axvline(x=optimal_k, color='r', linestyle='--', 
                          label=f'Optimal k={optimal_k}')
                ax1.set_xlabel('Number of Clusters (k)')
                ax1.set_ylabel('Silhouette Score')
                ax1.set_title('Silhouette Analysis')
                ax1.legend()
                ax1.grid(True)
                
                # Plot detailed silhouette plot for optimal k
                optimal_silhouettes = sample_silhouettes[K.index(optimal_k)]
                optimal_labels = KMeans(
                    n_clusters=optimal_k,
                    n_init=n_init,
                    random_state=self.random_state
                ).fit_predict(numeric_data)
                
                y_lower = 10
                for i in range(optimal_k):
                    cluster_silhouettes = optimal_silhouettes[optimal_labels == i]
                    cluster_silhouettes.sort()
                    
                    size = cluster_silhouettes.shape[0]
                    y_upper = y_lower + size
                    
                    color = plt.cm.nipy_spectral(float(i) / optimal_k)
                    ax2.fill_betweenx(np.arange(y_lower, y_upper),
                                    0, cluster_silhouettes,
                                    facecolor=color, alpha=0.7)
                    y_lower = y_upper + 10
                    
                ax2.set_xlabel('Silhouette Coefficient')
                ax2.set_ylabel('Cluster Label')
                ax2.set_title('Silhouette Plot for Optimal k')
                ax2.axvline(x=np.mean(optimal_silhouettes), color='r', linestyle='--')
                
                plt.tight_layout()
                plt.savefig(self.plots_dir / 'silhouette_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
                
            elif method == 'gap':
                gap_stats = []
                gap_std = []
                reference_datasets = 5
                
                with tqdm(total=len(K) * reference_datasets, desc="Computing gap statistic") as pbar:
                    for k in K:
                        # Store reference inertias for this k
                        ref_inertias = []
                        
                        # Generate reference datasets and compute their inertias
                        for _ in range(reference_datasets):
                            # Generate random uniform data with same shape and bounds
                            rand_data = np.random.uniform(
                                low=numeric_data.min().values,
                                high=numeric_data.max().values,
                                size=numeric_data.shape
                            )
                            
                            # Fit k-means and store inertia
                            kmeans_ref = KMeans(
                                n_clusters=k,
                                n_init=n_init,
                                random_state=self.random_state
                            )
                            kmeans_ref.fit(rand_data)
                            ref_inertias.append(kmeans_ref.inertia_)
                            pbar.update(1)
                            
                        # Fit k-means on real data
                        kmeans_real = KMeans(
                            n_clusters=k,
                            n_init=n_init,
                            random_state=self.random_state
                        )
                        kmeans_real.fit(numeric_data)
                        
                        # Compute gap statistic
                        gap = np.log(np.mean(ref_inertias)) - np.log(kmeans_real.inertia_)
                        gap_stats.append(gap)
                        
                        # Compute standard deviation
                        sdk = np.std(np.log(ref_inertias)) * np.sqrt(1 + 1/reference_datasets)
                        gap_std.append(sdk)
                        
                # Find optimal k using gap statistic criterion
                gap_stats = np.array(gap_stats)
                gap_std = np.array(gap_std)
                optimal_k = K[np.argmax(gap_stats)]
                
                # Plot gap statistic results
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                # Gap statistic plot
                ax1.errorbar(K, gap_stats, yerr=gap_std, fmt='bo-', capsize=5)
                ax1.axvline(x=optimal_k, color='r', linestyle='--',
                          label=f'Optimal k={optimal_k}')
                ax1.set_xlabel('Number of Clusters (k)')
                ax1.set_ylabel('Gap Statistic')
                ax1.set_title('Gap Statistic Analysis')
                ax1.legend()
                ax1.grid(True)
                
                # Standard deviation plot
                ax2.plot(K, gap_std, 'go-', label='Standard Deviation')
                ax2.set_xlabel('Number of Clusters (k)')
                ax2.set_ylabel('Standard Deviation')
                ax2.set_title('Gap Statistic Standard Deviation')
                ax2.legend()
                ax2.grid(True)
                
                plt.tight_layout()
                plt.savefig(self.plots_dir / 'gap_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
                
            else:
                raise ValueError(f"Invalid method: {method}. Must be one of: 'elbow', 'silhouette', 'gap'")
            
            # Save optimization results
            optimization_results = {
                'method': method,
                'optimal_k': optimal_k,
                'min_k': min_k,
                'max_k': max_k,
                'n_init': n_init,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.results_dir / 'optimization_results.json', 'w') as f:
                json.dump(optimization_results, f, indent=4)
            
            self.logger.info(f"Optimal number of clusters found: {optimal_k}")
            self.logger.info(f"Method used: {method}")
            self.logger.info(f"Results saved to {self.results_dir / 'optimization_results.json'}")
            
            return optimal_k
            
        except Exception as e:
            self.logger.error(f"Error finding optimal clusters: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise RuntimeError(f"Cluster optimization failed: {str(e)}")

    def perform_clustering(
        self,
        method: str = 'kmeans',
        n_clusters: int = 3,
        **kwargs
    ) -> None:
        """
        Perform clustering using specified method.
        
        Args:
            method: Clustering method ('kmeans', 'dbscan', 'hierarchical', or 'gmm')
            n_clusters: Number of clusters
            **kwargs: Additional parameters for clustering algorithms
            
        Raises:
            ValueError: If normalized_data is None or empty
            ValueError: If invalid clustering method specified
            RuntimeError: If clustering fails
        """
        if self.normalized_data is None or self.normalized_data.empty:
            raise ValueError("No normalized data available")
            
        try:
            # Get numeric data
            numeric_data = self.normalized_data.select_dtypes(
                include=['float64', 'int64']
            )
            
            if method == 'kmeans':
                model = KMeans(
                    n_clusters=n_clusters,
                    random_state=self.random_state,
                    **kwargs
                )
                self.clusters = model.fit_predict(numeric_data)
                self.n_clusters = n_clusters
                self.model = model
                
                # Calculate feature importance
                self.feature_importance = pd.DataFrame({
                    'feature': numeric_data.columns,
                    'importance': np.abs(model.cluster_centers_).mean(axis=0)
                }).sort_values('importance', ascending=False)
                
            elif method == 'dbscan':
                model = DBSCAN(**kwargs)
                self.clusters = model.fit_predict(numeric_data)
                self.n_clusters = len(set(self.clusters)) - (1 if -1 in self.clusters else 0)
                self.model = model
                
            elif method == 'hierarchical':
                model = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    **kwargs
                )
                self.clusters = model.fit_predict(numeric_data)
                self.n_clusters = n_clusters
                
            else:
                raise ValueError(f"Invalid clustering method: {method}")
            
            # Calculate cluster profiles
            self.calculate_cluster_profiles()
            
            # Save results
            self.save_results()
            
            logger.info(f"Clustering completed successfully using {method}")
            logger.info(f"Number of clusters: {self.n_clusters}")
            
        except Exception as e:
            logger.error(f"Error performing clustering: {str(e)}")
            raise

    def calculate_cluster_profiles(self) -> None:
        """Calculate profiles for each cluster"""
        try:
            profiles = []
            
            for cluster in range(self.n_clusters):
                cluster_data = self.normalized_data[self.clusters == cluster]
                profile = cluster_data.mean()
                profiles.append(profile)
                
            self.cluster_profiles = pd.DataFrame(profiles)
            
        except Exception as e:
            logger.error(f"Error calculating cluster profiles: {str(e)}")
            raise

    def save_results(self) -> None:
        """Save clustering results to files"""
        try:
            # Save cluster assignments
            pd.DataFrame({
                'cluster': self.clusters
            }).to_csv(self.output_dir / 'cluster_assignments.csv', index=False)
            
            # Save feature importance
            if self.feature_importance is not None:
                self.feature_importance.to_csv(
                    self.output_dir / 'feature_importance.csv',
                    index=False
                )
                
            # Save cluster profiles
            if self.cluster_profiles is not None:
                self.cluster_profiles.to_csv(
                    self.output_dir / 'cluster_profiles.csv'
                )
                
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

    def get_cluster_summary(self) -> str:
        """Get text summary of clustering results"""
        try:
            summary = []
            
            summary.append(f"Number of clusters: {self.n_clusters}")
            summary.append("\nCluster sizes:")
            sizes = pd.Series(self.clusters).value_counts().sort_index()
            for cluster, size in sizes.items():
                summary.append(
                    f"Cluster {cluster}: {size} samples "
                    f"({size/len(self.clusters)*100:.1f}%)"
                )
                
            if self.feature_importance is not None:
                summary.append("\nTop features by importance:")
                for _, row in self.feature_importance.head().iterrows():
                    summary.append(
                        f"{row['feature']}: {row['importance']:.3f}"
                    )
                    
            return "\n".join(summary)
            
        except Exception as e:
            logger.error(f"Error getting cluster summary: {str(e)}")
            raise

    def visualize(
        self,
        plot_type: str = 'distribution'
    ) -> plt.Figure:
        """
        Generate visualization of clustering results.
        
        Args:
            plot_type: Type of plot to generate
            
        Returns:
            Matplotlib figure
            
        Raises:
            ValueError: If clusters is None
            ValueError: If invalid plot type specified
        """
        if self.clusters is None:
            raise ValueError("No clustering results available")
            
        try:
            if plot_type == 'distribution':
                fig = plt.figure(figsize=(12, 8))
                
                numeric_cols = self.normalized_data.select_dtypes(
                    include=['float64', 'int64']
                ).columns
                
                for i, col in enumerate(numeric_cols):
                    plt.subplot(3, 4, i+1)
                    for cluster in range(self.n_clusters):
                        cluster_data = self.normalized_data[
                            self.clusters == cluster
                        ][col]
                        sns.kdeplot(
                            data=cluster_data,
                            label=f'Cluster {cluster}'
                        )
                    plt.title(col)
                    if i == 0:
                        plt.legend()
                        
                plt.tight_layout()
                
            elif plot_type == 'profile':
                fig = plt.figure(figsize=(10, 6))
                
                for cluster in range(self.n_clusters):
                    plt.plot(
                        self.cluster_profiles.columns,
                        self.cluster_profiles.iloc[cluster],
                        label=f'Cluster {cluster}',
                        marker='o'
                    )
                    
                plt.xticks(rotation=45, ha='right')
                plt.legend()
                plt.title("Cluster Profiles")
                plt.tight_layout()
                
            elif plot_type == 'reduction':
                # Perform PCA
                pca = PCA(n_components=2)
                numeric_data = self.normalized_data.select_dtypes(
                    include=['float64', 'int64']
                )
                coords = pca.fit_transform(numeric_data)
                
                fig = plt.figure(figsize=(8, 6))
                
                scatter = plt.scatter(
                    coords[:, 0],
                    coords[:, 1],
                    c=self.clusters,
                    cmap='viridis'
                )
                plt.colorbar(scatter)
                plt.xlabel("First Principal Component")
                plt.ylabel("Second Principal Component")
                plt.title("PCA Visualization of Clusters")
                
            else:
                raise ValueError(f"Invalid plot type: {plot_type}")
                
            return fig
            
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")
            raise VisualizationError(f"Failed to generate visualization: {str(e)}")

def main():
    """Main function to run GUI application"""
    try:
        root = tk.Tk()
        app = ClusterAnalysisGUI(root)
        root.mainloop()
        
    except Exception as e:
        logger.error("An error occurred in the application")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
    logger.info("Application closed")
