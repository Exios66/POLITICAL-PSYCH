import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
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
import umap.umap_ as umap
from statsmodels.stats.multitest import multipletests
import sys
from scipy.cluster.hierarchy import linkage, dendrogram

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
        # Store root window reference
        self.master = master
        self.master.title("Political Psychology Cluster Analysis")
        self.master.geometry("1200x800")
        
        # Configure basic logging first
        self._setup_logging()
        
        try:
            # Initialize core variables
            self._initialize_variables()
            
            # Configure window minimum size
            self.master.minsize(800, 600)
            
            # Configure grid weights for proper scaling
            self.master.grid_rowconfigure(0, weight=1)
            self.master.grid_columnconfigure(0, weight=1)
            
            # Setup directories
            self._setup_directories()
            
            # Initialize data variables
            self.cleaned_data = None
            self.normalized_data = None
            self.random_state = 42
            
            # Create main container with tabs
            self.notebook = ttk.Notebook(self.master)
            self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Initialize tabs
            self._initialize_tabs()
            
            # Create status and progress bars
            self._create_status_bars()
            
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize application: {str(e)}")
            raise

    def _setup_logging(self):
        """Set up logging configuration"""
        # Create logs directory
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

    def _initialize_variables(self):
        """Initialize all GUI variables"""
        try:
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
            
            # Analysis variables
            self.analysis_type = tk.StringVar(value="statistical")
            self.stat_test = tk.StringVar(value="anova")
            
            # Plot variables
            self.plot_type = tk.StringVar(value="distribution")
            self.dist_feature = tk.StringVar()
            self.dist_type = tk.StringVar(value="histogram")
            self.profile_type = tk.StringVar(value="heatmap")
            self.reduction_method = tk.StringVar(value="pca")
            self.n_components = tk.IntVar(value=2)
            self.importance_method = tk.StringVar(value="random_forest")
            self.n_top_features = tk.IntVar(value=10)
            
            # Status variables
            self.status_var = tk.StringVar(value="Ready")
            self.progress_var = tk.DoubleVar(value=0)
            
            self.logger.info("Variables initialized successfully")
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Failed to initialize variables: {str(e)}")
            raise

    def _setup_directories(self):
        """Create necessary directories"""
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
            
            self.logger.info("Directories created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create directories: {str(e)}")
            raise

    def _initialize_tabs(self):
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
            
            # Create frames for clustering methods before creating cluster tab
            self.create_kmeans_frame()
            self.create_dbscan_frame()
            self.create_hierarchical_frame()
            
            # Initialize tab contents
            self.create_data_tab()
            self.create_cluster_tab()
            self.create_viz_tab()
            self.create_analysis_tab()
            
            self.logger.info("Tabs initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize tabs: {str(e)}")
            raise

    def _create_status_bars(self):
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
        pass

    def create_kmeans_frame(self):
        """Create and configure k-means clustering frame"""
        pass

    def create_dbscan_frame(self):
        """Create and configure DBSCAN clustering frame"""
        pass

    def create_hierarchical_frame(self):
        """Create and configure hierarchical clustering frame"""
        pass

    def create_cluster_tab(self):
        """Create and configure clustering tab"""
        pass

    def create_viz_tab(self):
        """Create and configure visualization tab"""
        pass

    def create_analysis_tab(self):
        """Create and configure analysis tab"""
        pass

    def browse_file(self):
        """Browse and load data file"""
        pass

    def update_missing_options(self):
        """Update missing value handling options"""
        pass

    def update_outlier_options(self):
        """Update outlier removal options"""
        pass

    def update_norm_options(self):
        """Update normalization options"""
        pass

    def load_data(self):
        """Load data from file"""
        pass

    def handle_missing_values(self):
        """Handle missing values based on selected method"""
        pass

    def remove_outliers(self):
        """Remove outliers based on selected method"""
        pass

    def normalize_features(self):
        """Normalize features based on selected method"""
        pass

    def cluster_data(self):
        """Perform clustering based on selected method"""
        pass

    def analyze_clusters(self):
        """Analyze clusters based on selected method"""
        pass

    def visualize_clusters(self):
        """Visualize clusters based on selected method"""
        pass

    def save_results(self):
        """Save results to file"""
        pass

    def run_analysis(self):
        """Run full analysis pipeline"""
        pass

if __name__ == "__main__":
    root = tk.Tk()
    app = ClusterAnalysisGUI(root)
    root.mainloop()
</original_file>
class ClusterAnalysisGUI:
    def __init__(self, master):
        """Initialize GUI components with enhanced structure and error handling"""
        # Store root window reference
        self.master = master
        self.master.title("Political Psychology Cluster Analysis")
        self.master.geometry("1200x800")
        
        # Configure basic logging first
        self._setup_logging()
        
        try:
            # Initialize core variables
            self._initialize_variables()
            
            # Configure window minimum size
            self.master.minsize(800, 600)
            
            # Configure grid weights for proper scaling
            self.master.grid_rowconfigure(0, weight=1)
            self.master.grid_columnconfigure(0, weight=1)
            
            # Setup directories
            self._setup_directories()
            
            # Initialize data variables
            self.cleaned_data = None
            self.normalized_data = None
            self.random_state = 42
            
            # Create main container with tabs
            self.notebook = ttk.Notebook(self.master)
            self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Initialize tabs
            self._initialize_tabs()
            
            # Create status and progress bars
            self._create_status_bars()
            
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize application: {str(e)}")
            raise

    def _setup_logging(self):
        """Set up logging configuration"""
        # Create logs directory
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

    def _initialize_variables(self):
        """Initialize all GUI variables"""
        try:
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
            
            # Analysis variables
            self.analysis_type = tk.StringVar(value="statistical")
            self.stat_test = tk.StringVar(value="anova")
            
            # Plot variables
            self.plot_type = tk.StringVar(value="distribution")
            self.dist_feature = tk.StringVar()
            self.dist_type = tk.StringVar(value="histogram")
            self.profile_type = tk.StringVar(value="heatmap")
            self.reduction_method = tk.StringVar(value="pca")
            self.n_components = tk.IntVar(value=2)
            self.importance_method = tk.StringVar(value="random_forest")
            self.n_top_features = tk.IntVar(value=10)
            
            # Status variables
            self.status_var = tk.StringVar(value="Ready")
            self.progress_var = tk.DoubleVar(value=0)
            
            self.logger.info("Variables initialized successfully")
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Failed to initialize variables: {str(e)}")
            raise

    def _setup_directories(self):
        """Create necessary directories"""
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
            
            self.logger.info("Directories created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create directories: {str(e)}")
            raise

    def _initialize_tabs(self):
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
            
            # Create frames for clustering methods before creating cluster tab
            self.create_kmeans_frame()
            self.create_dbscan_frame()
            self.create_hierarchical_frame()
            
            # Initialize tab contents
            self.create_data_tab()
            self.create_cluster_tab()
            self.create_viz_tab()
            self.create_analysis_tab()
            
            self.logger.info("Tabs initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize tabs: {str(e)}")
            raise

    def _create_status_bars(self):
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
```
</rewritten_file>
    ) -> int:
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
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
import umap.umap_ as umap
from statsmodels.stats.multitest import multipletests
import sys
from scipy.cluster.hierarchy import linkage, dendrogram

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
        # Store root window reference
        self.master = master
        self.master.title("Political Psychology Cluster Analysis")
        self.master.geometry("1200x800")
        
        # Configure basic logging first
        self._setup_logging()
        
        try:
            # Initialize core variables
            self._initialize_variables()
            
            # Configure window minimum size
            self.master.minsize(800, 600)
            
            # Configure grid weights for proper scaling
            self.master.grid_rowconfigure(0, weight=1)
            self.master.grid_columnconfigure(0, weight=1)
            
            # Setup directories
            self._setup_directories()
            
            # Initialize data variables
            self.cleaned_data = None
            self.normalized_data = None
            self.random_state = 42
            
            # Create main container with tabs
            self.notebook = ttk.Notebook(self.master)
            self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Initialize tabs
            self._initialize_tabs()
            
            # Create status and progress bars
            self._create_status_bars()
            
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize application: {str(e)}")
            raise

    def _setup_logging(self):
        """Set up logging configuration"""
        # Create logs directory
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

    def _initialize_variables(self):
        """Initialize all GUI variables"""
        try:
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
            
            # Analysis variables
            self.analysis_type = tk.StringVar(value="statistical")
            self.stat_test = tk.StringVar(value="anova")
            
            # Plot variables
            self.plot_type = tk.StringVar(value="distribution")
            self.dist_feature = tk.StringVar()
            self.dist_type = tk.StringVar(value="histogram")
            self.profile_type = tk.StringVar(value="heatmap")
            self.reduction_method = tk.StringVar(value="pca")
            self.n_components = tk.IntVar(value=2)
            self.importance_method = tk.StringVar(value="random_forest")
            self.n_top_features = tk.IntVar(value=10)
            
            # Status variables
            self.status_var = tk.StringVar(value="Ready")
            self.progress_var = tk.DoubleVar(value=0)
            
            self.logger.info("Variables initialized successfully")
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Failed to initialize variables: {str(e)}")
            raise

    def _setup_directories(self):
        """Create necessary directories"""
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
            
            self.logger.info("Directories created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create directories: {str(e)}")
            raise

    def _initialize_tabs(self):
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
            
            # Create frames for clustering methods before creating cluster tab
            self.create_kmeans_frame()
            self.create_dbscan_frame()
            self.create_hierarchical_frame()
            
            # Initialize tab contents
            self.create_data_tab()
            self.create_cluster_tab()
            self.create_viz_tab()
            self.create_analysis_tab()
            
            self.logger.info("Tabs initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize tabs: {str(e)}")
            raise

    def _create_status_bars(self):
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
        pass

    def create_kmeans_frame(self):
        """Create and configure k-means clustering frame"""
        pass

    def create_dbscan_frame(self):
        """Create and configure DBSCAN clustering frame"""
        pass

    def create_hierarchical_frame(self):
        """Create and configure hierarchical clustering frame"""
        pass

    def create_cluster_tab(self):
        """Create and configure clustering tab"""
        pass

    def create_viz_tab(self):
        """Create and configure visualization tab"""
        pass

    def create_analysis_tab(self):
        """Create and configure analysis tab"""
        pass

    def browse_file(self):
        """Browse and load data file"""
        pass

    def update_missing_options(self):
        """Update missing value handling options"""
        pass

    def update_outlier_options(self):
        """Update outlier removal options"""
        pass

    def update_norm_options(self):
        """Update normalization options"""
        pass

    def load_data(self):
        """Load data from file"""
        pass

    def handle_missing_values(self):
        """Handle missing values based on selected method"""
        pass

    def remove_outliers(self):
        """Remove outliers based on selected method"""
        pass

    def normalize_features(self):
        """Normalize features based on selected method"""
        pass

    def cluster_data(self):
        """Perform clustering based on selected method"""
        pass

    def analyze_clusters(self):
        """Analyze clusters based on selected method"""
        pass

    def visualize_clusters(self):
        """Visualize clusters based on selected method"""
        pass

    def save_results(self):
        """Save results to file"""
        pass

    def run_analysis(self):
        """Run full analysis pipeline"""
        pass

if __name__ == "__main__":
    root = tk.Tk()
    app = ClusterAnalysisGUI(root)
    root.mainloop()
```
</rewritten_file>
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
        pass
```
</rewritten_file>

    def perform_clustering(self):
        """Perform clustering analysis with selected parameters"""
        if self.normalized_data is None:
            raise ValueError("Data must be normalized before clustering")
            
        if self.clustering_method not in ['kmeans', 'dbscan', 'hierarchical']:
            raise ValueError(f"Invalid clustering method: {self.clustering_method}")
            
        try:
            if self.clustering_method == 'kmeans':
                self.clusterer = KMeans(
                    n_clusters=self.n_clusters,
                    random_state=42,
                    n_init=10
                )
            elif self.clustering_method == 'dbscan':
                self.clusterer = DBSCAN(
                    eps=self.eps,
                    min_samples=self.min_samples
                )
            else:  # hierarchical
                self.clusterer = AgglomerativeClustering(
                    n_clusters=self.n_clusters,
                    linkage='ward'
                )
                
            self.cluster_labels = self.clusterer.fit_predict(self.normalized_data)
            logger.info(f"Clustering completed using {self.clustering_method}")
            
        except Exception as e:
            logger.error(f"Clustering failed: {str(e)}")
            raise RuntimeError(f"Clustering failed: {str(e)}")
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
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
import umap.umap_ as umap
from statsmodels.stats.multitest import multipletests
import sys
from scipy.cluster.hierarchy import linkage, dendrogram

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
        # Store root window reference
        self.master = master
        self.master.title("Political Psychology Cluster Analysis")
        self.master.geometry("1200x800")
        
        # Configure basic logging first
        self._setup_logging()
        
        try:
            # Initialize core variables
            self._initialize_variables()
            
            # Configure window minimum size
            self.master.minsize(800, 600)
            
            # Configure grid weights for proper scaling
            self.master.grid_rowconfigure(0, weight=1)
            self.master.grid_columnconfigure(0, weight=1)
            
            # Setup directories
            self._setup_directories()
            
            # Initialize data variables
            self.cleaned_data = None
            self.normalized_data = None
            self.random_state = 42
            
            # Create main container with tabs
            self.notebook = ttk.Notebook(self.master)
            self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Initialize tabs
            self._initialize_tabs()
            
            # Create status and progress bars
            self._create_status_bars()
            
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize application: {str(e)}")
            raise

    def _setup_logging(self):
        """Set up logging configuration"""
        # Create logs directory
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

    def _initialize_variables(self):
        """Initialize all GUI variables"""
        try:
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
            
            # Analysis variables
            self.analysis_type = tk.StringVar(value="statistical")
            self.stat_test = tk.StringVar(value="anova")
            
            # Plot variables
            self.plot_type = tk.StringVar(value="distribution")
            self.dist_feature = tk.StringVar()
            self.dist_type = tk.StringVar(value="histogram")
            self.profile_type = tk.StringVar(value="heatmap")
            self.reduction_method = tk.StringVar(value="pca")
            self.n_components = tk.IntVar(value=2)
            self.importance_method = tk.StringVar(value="random_forest")
            self.n_top_features = tk.IntVar(value=10)
            
            # Status variables
            self.status_var = tk.StringVar(value="Ready")
            self.progress_var = tk.DoubleVar(value=0)
            
            self.logger.info("Variables initialized successfully")
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Failed to initialize variables: {str(e)}")
            raise

    def _setup_directories(self):
        """Create necessary directories"""
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
            
            self.logger.info("Directories created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create directories: {str(e)}")
            raise

    def _initialize_tabs(self):
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
            
            # Create frames for clustering methods before creating cluster tab
            self.create_kmeans_frame()
            self.create_dbscan_frame()
            self.create_hierarchical_frame()
            
            # Initialize tab contents
            self.create_data_tab()
            self.create_cluster_tab()
            self.create_viz_tab()
            self.create_analysis_tab()
            
            self.logger.info("Tabs initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize tabs: {str(e)}")
            raise

    def _create_status_bars(self):
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
            self.cluster_method = tk.StringVar(value="kmeans")
            self.cluster_method_menu = ttk.OptionMenu(
                method_frame,
                self.cluster_method,
                "kmeans",
                "kmeans", "dbscan", "hierarchical",
                command=self.update_cluster_options
            )
            self.cluster_method_menu.pack(side=tk.LEFT, padx=5)
            
            # Initially show kmeans frame
            self.kmeans_frame.pack(fill=tk.X, padx=5, pady=5)
            
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

    def save_clustering_results(self):
        """Save clustering results to file"""
        try:
            if not hasattr(self, 'normalized_data') or self.normalized_data is None:
                raise ValueError("No clustering results available to save")
                
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if file_path:
                # Save results
                results_df = self.normalized_data.copy()
                results_df['Cluster'] = self.cluster_labels
                results_df.to_csv(file_path, index=False)
                
                self.status_var.set(f"Results saved to {file_path}")
                self.logger.info(f"Clustering results saved to {file_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to save clustering results: {str(e)}")
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")

    def update_cluster_options(self, *args):
        """Update clustering options based on selected method."""
        try:
            method = self.cluster_method.get()
            
            # Hide all parameter frames initially
            self.kmeans_frame.pack_forget()
            self.dbscan_frame.pack_forget()
            self.hierarchical_frame.pack_forget()
            
            # Show the relevant frame based on the selected method
            if method == "kmeans":
                self.kmeans_frame.pack(fill=tk.X, padx=5, pady=5)
            elif method == "dbscan":
                self.dbscan_frame.pack(fill=tk.X, padx=5, pady=5)
            elif method == "hierarchical":
                self.hierarchical_frame.pack(fill=tk.X, padx=5, pady=5)
            
            self.logger.info(f"Updated cluster options for method: {method}")
            
        except Exception as e:
            self.logger.error(f"Failed to update cluster options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update cluster options: {str(e)}")

    def create_distribution_frame(self):
        """Create frame for distribution visualization options"""
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
                # options will be populated when data is loaded
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
            
            # Additional options frame
            options_frame = ttk.Frame(self.dist_frame)
            options_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Bin count for histogram
            self.bin_count = tk.IntVar(value=30)
            ttk.Label(options_frame, text="Bins:").pack(side=tk.LEFT)
            ttk.Entry(
                options_frame,
                textvariable=self.bin_count,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # KDE bandwidth
            self.kde_bandwidth = tk.DoubleVar(value=0.2)
            ttk.Label(options_frame, text="Bandwidth:").pack(side=tk.LEFT)
            ttk.Entry(
                options_frame,
                textvariable=self.kde_bandwidth,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            self.logger.info("Distribution frame created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create distribution frame: {str(e)}")
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
            
            # Create all visualization frames
            self.create_distribution_frame()
            self.create_profile_frame()
            self.create_reduction_frame()
            self.create_importance_frame()
            
            # Initially show distribution frame
            self.dist_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Plot frame
            plot_container = ttk.LabelFrame(self.viz_tab, text="Visualization")
            plot_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Toolbar frame
            self.toolbar_frame = ttk.Frame(plot_container)
            self.toolbar_frame.pack(fill=tk.X)
            
            # Plot frame
            self.plot_frame = ttk.Frame(plot_container)
            self.plot_frame.pack(fill=tk.BOTH, expand=True)
            
            # Initialize matplotlib figure and canvas
            self.fig = Figure(figsize=(8, 6), dpi=100)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add matplotlib toolbar
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
            self.toolbar.update()
            
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

    def update_feature_menu(self):
        """Update feature menu options when data is loaded"""
        try:
            if self.normalized_data is not None:
                # Get list of numeric columns
                features = list(self.normalized_data.select_dtypes(
                    include=['float64', 'int64']).columns)
                
                # Update feature menu
                menu = self.feature_menu["menu"]
                menu.delete(0, "end")
                for feature in features:
                    menu.add_command(
                        label=feature,
                        command=lambda f=feature: self.dist_feature.set(f))
                
                # Set default feature if none selected
                if not self.dist_feature.get() and features:
                    self.dist_feature.set(features[0])
                    
                self.logger.info("Feature menu updated successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to update feature menu: {str(e)}")
            messagebox.showerror("Error", f"Failed to update feature menu: {str(e)}")

    def generate_plot(self):
        """Generate visualization based on selected plot type and options"""
        try:
            # Clear previous plot
            self.fig.clear()
            
            # Get selected plot type
            plot_type = self.plot_type.get()
            
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
            
            self.logger.info(f"Generated {plot_type} plot successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to generate plot: {str(e)}")
            messagebox.showerror("Error", f"Failed to generate plot: {str(e)}")

    def _generate_distribution_plot(self):
        """Generate distribution plot for selected feature"""
        feature = self.dist_feature.get()
        plot_type = self.dist_type.get()
        
        if not feature:
            raise ValueError("No feature selected")
            
        data = self.data[feature]
        labels = self.cluster_labels
        
        ax = self.fig.add_subplot(111)
        
        if plot_type == "histogram":
            for label in np.unique(labels):
                ax.hist(data[labels == label], alpha=0.5, 
                       label=f'Cluster {label}', bins=30)
        elif plot_type == "kde":
            for label in np.unique(labels):
                sns.kdeplot(data[labels == label], ax=ax,
                          label=f'Cluster {label}')
        else:  # box plot
            ax.boxplot([data[labels == label] 
                       for label in np.unique(labels)],
                      labels=[f'Cluster {label}' 
                             for label in np.unique(labels)])
            
        ax.set_title(f'{feature} Distribution by Cluster')
        ax.set_xlabel(feature)
        ax.legend()
        
        self.plot_data = pd.DataFrame({
            'Feature': data,
            'Cluster': labels
        })

    def _generate_profile_plot(self):
        """Generate cluster profile visualization"""
        plot_type = self.profile_type.get()
        features = [self.feature_listbox.get(i) 
                   for i in self.feature_listbox.curselection()]
        
        if not features:
            raise ValueError("No features selected")
            
        data = self.data[features]
        labels = self.cluster_labels
        
        if plot_type == "heatmap":
            cluster_means = pd.DataFrame([
                data[labels == label].mean() 
                for label in np.unique(labels)
            ], index=[f'Cluster {label}' for label in np.unique(labels)])
            
            ax = self.fig.add_subplot(111)
            sns.heatmap(cluster_means, cmap='coolwarm', center=0,
                       annot=True, ax=ax)
            ax.set_title('Cluster Profiles Heatmap')
            
        elif plot_type == "parallel":
            ax = self.fig.add_subplot(111)
            pd.plotting.parallel_coordinates(
                pd.concat([data, pd.Series(labels, name='Cluster')], axis=1),
                'Cluster', ax=ax)
            ax.set_title('Parallel Coordinates Plot')
            
        else:  # radar chart
            angles = np.linspace(0, 2*np.pi, len(features), endpoint=False)
            cluster_means = np.array([
                data[labels == label].mean() 
                for label in np.unique(labels)
            ])
            
            ax = self.fig.add_subplot(111, projection='polar')
            for i, means in enumerate(cluster_means):
                values = np.concatenate((means, [means[0]]))
                angles_plot = np.concatenate((angles, [angles[0]]))
                ax.plot(angles_plot, values, label=f'Cluster {i}')
                
            ax.set_xticks(angles)
            ax.set_xticklabels(features)
            ax.set_title('Radar Chart of Cluster Profiles')
            ax.legend()
            
        self.plot_data = pd.DataFrame({
            'Feature': np.repeat(features, len(np.unique(labels))),
            'Cluster': np.tile(np.unique(labels), len(features)),
            'Value': cluster_means.flatten()
        })

    def _generate_reduction_plot(self):
        """Generate dimensionality reduction visualization"""
        method = self.reduction_method.get()
        n_components = self.n_components.get()
        
        if method == "pca":
            reducer = PCA(n_components=n_components)
        elif method == "tsne":
            reducer = TSNE(n_components=n_components)
        else:  # umap
            reducer = umap.UMAP(n_components=n_components)
            
        reduced_data = reducer.fit_transform(self.data)
        
        if n_components == 2:
            ax = self.fig.add_subplot(111)
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1],
                               c=self.cluster_labels, cmap='viridis')
            ax.set_title(f'{method.upper()} Projection')
            self.fig.colorbar(scatter, label='Cluster')
            
        elif n_components == 3:
            ax = self.fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1],
                               reduced_data[:, 2], c=self.cluster_labels,
                               cmap='viridis')
            ax.set_title(f'{method.upper()} Projection')
            self.fig.colorbar(scatter, label='Cluster')
            
        self.plot_data = pd.DataFrame(
            reduced_data,
            columns=[f'Component {i+1}' for i in range(n_components)])
        self.plot_data['Cluster'] = self.cluster_labels

    def _generate_importance_plot(self):
        """Generate feature importance visualization"""
        method = self.importance_method.get()
        n_features = self.n_top_features.get()
        
        if method == "statistical":
            importances = []
            pvalues = []
            for col in self.data.columns:
                f_stat, p_val = f_oneway(*[
                    self.data[col][self.cluster_labels == label]
                    for label in np.unique(self.cluster_labels)
                ])
                importances.append(f_stat)
                pvalues.append(p_val)
                
            importance_df = pd.DataFrame({
                'Feature': self.data.columns,
                'Importance': importances,
                'p-value': pvalues
            })
            
        else:  # random forest
            rf = RandomForestClassifier(n_estimators=100)
            rf.fit(self.data, self.cluster_labels)
            
            importance_df = pd.DataFrame({
                'Feature': self.data.columns,
                'Importance': rf.feature_importances_
            })
            
        importance_df = importance_df.nlargest(n_features, 'Importance')
        
        ax = self.fig.add_subplot(111)
        sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax)
        ax.set_title('Feature Importance')
        
        self.plot_data = importance_df

    def save_plot(self):
        """Save current plot to file"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("SVG files", "*.svg")
                ]
            )
            
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
            
            # Components
            comp_frame = ttk.Frame(self.reduction_frame)
            comp_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(comp_frame, text="Number of components:").pack(side=tk.LEFT)
            self.n_components = tk.IntVar(value=2)
            ttk.Entry(
                comp_frame,
                textvariable=self.n_components,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Method-specific parameters frame
            self.reduction_params_frame = ttk.Frame(self.reduction_frame)
            self.reduction_params_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Initialize with PCA parameters
            self.create_pca_params()
            
            self.logger.info("Reduction frame created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create reduction frame: {str(e)}")
            raise

    def update_reduction_options(self, *args):
        """Update reduction options based on selected method"""
        try:
            method = self.reduction_method.get()
            
            # Clear existing parameters
            for widget in self.reduction_params_frame.winfo_children():
                widget.destroy()
            
            # Show method-specific parameters
            if method == "pca":
                self.create_pca_params()
            elif method == "tsne":
                self.create_tsne_params()
            elif method == "umap":
                self.create_umap_params()
            
            self.logger.info(f"Updated reduction options for method: {method}")
            
        except Exception as e:
            self.logger.error(f"Failed to update reduction options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update reduction options: {str(e)}")

    def create_pca_params(self):
        """Create parameter inputs for PCA"""
        try:
            # Explained variance ratio threshold
            var_frame = ttk.Frame(self.reduction_params_frame)
            var_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(var_frame, text="Min explained variance:").pack(side=tk.LEFT)
            self.min_variance = tk.DoubleVar(value=0.95)
            ttk.Entry(
                var_frame,
                textvariable=self.min_variance,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Whitening option
            self.whiten = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                self.reduction_params_frame,
                text="Apply whitening",
                variable=self.whiten
            ).pack(anchor=tk.W)
            
        except Exception as e:
            self.logger.error(f"Failed to create PCA parameters: {str(e)}")
            raise

    def create_tsne_params(self):
        """Create parameter inputs for t-SNE"""
        try:
            # Perplexity
            perp_frame = ttk.Frame(self.reduction_params_frame)
            perp_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(perp_frame, text="Perplexity:").pack(side=tk.LEFT)
            self.perplexity = tk.DoubleVar(value=30.0)
            ttk.Entry(
                perp_frame,
                textvariable=self.perplexity,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Learning rate
            lr_frame = ttk.Frame(self.reduction_params_frame)
            lr_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(lr_frame, text="Learning rate:").pack(side=tk.LEFT)
            self.learning_rate = tk.DoubleVar(value=200.0)
            ttk.Entry(
                lr_frame,
                textvariable=self.learning_rate,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Number of iterations
            iter_frame = ttk.Frame(self.reduction_params_frame)
            iter_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(iter_frame, text="Iterations:").pack(side=tk.LEFT)
            self.n_iter_tsne = tk.IntVar(value=1000)
            ttk.Entry(
                iter_frame,
                textvariable=self.n_iter_tsne,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            self.logger.error(f"Failed to create t-SNE parameters: {str(e)}")
            raise

    def create_umap_params(self):
        """Create parameter inputs for UMAP"""
        try:
            # Number of neighbors
            neighbors_frame = ttk.Frame(self.reduction_params_frame)
            neighbors_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(neighbors_frame, text="Number of neighbors:").pack(side=tk.LEFT)
            self.n_neighbors = tk.IntVar(value=15)
            ttk.Entry(
                neighbors_frame,
                textvariable=self.n_neighbors,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Minimum distance
            dist_frame = ttk.Frame(self.reduction_params_frame)
            dist_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(dist_frame, text="Minimum distance:").pack(side=tk.LEFT)
            self.min_dist = tk.DoubleVar(value=0.1)
            ttk.Entry(
                dist_frame,
                textvariable=self.min_dist,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Metric
            metric_frame = ttk.Frame(self.reduction_params_frame)
            metric_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(metric_frame, text="Distance metric:").pack(side=tk.LEFT)
            self.metric = tk.StringVar(value="euclidean")
            ttk.OptionMenu(
                metric_frame,
                self.metric,
                "euclidean",
                "euclidean", "manhattan", "cosine"
            ).pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            self.logger.error(f"Failed to create UMAP parameters: {str(e)}")
            raise

    def create_importance_frame(self):
        """Create frame for feature importance visualization options"""
        try:
            self.importance_frame = ttk.LabelFrame(self.viz_tab, text="Feature Importance Options")
            
            # Method selection
            method_frame = ttk.Frame(self.importance_frame)
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

    def export_plot_data(self):
        """Export data used to generate current plot"""
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
                width=5,
                validate='key',
                validatecommand=(self.register(self.validate_alpha), '%P')
            ).pack(side=tk.LEFT, padx=5)
            
            # Multiple testing correction
            correction_frame = ttk.Frame(self.stat_frame)
            correction_frame.pack(fill=tk.X, padx=5, pady=2)
            
            self.correction = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                correction_frame,
                text="Apply multiple testing correction",
                variable=self.correction,
                command=self.update_correction_options
            ).pack(side=tk.LEFT)
            
            self.correction_method = tk.StringVar(value="bonferroni")
            self.correction_menu = ttk.OptionMenu(
                correction_frame,
                self.correction_method,
                "bonferroni",
                "bonferroni", "fdr", "holm"
            )
            self.correction_menu.pack(side=tk.LEFT, padx=5)
            
            # Feature selection
            feature_frame = ttk.LabelFrame(self.analysis_tab, text="Feature Selection")
            feature_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Feature listbox with scrollbar
            listbox_frame = ttk.Frame(feature_frame)
            listbox_frame.pack(fill=tk.X, padx=5, pady=2)
            
            self.feature_listbox = tk.Listbox(
                listbox_frame,
                selectmode=tk.MULTIPLE,
                height=6
            )
            self.feature_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            feature_scroll = ttk.Scrollbar(
                listbox_frame,
                orient=tk.VERTICAL,
                command=self.feature_listbox.yview
            )
            feature_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            self.feature_listbox.configure(yscrollcommand=feature_scroll.set)
            
            # Results frame
            results_frame = ttk.LabelFrame(self.analysis_tab, text="Analysis Results")
            results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Results text with scrollbars
            text_frame = ttk.Frame(results_frame)
            text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            self.analysis_text = tk.Text(
                text_frame,
                wrap=tk.WORD,
                height=10
            )
            self.analysis_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            text_scroll = ttk.Scrollbar(
                text_frame,
                orient=tk.VERTICAL,
                command=self.analysis_text.yview
            )
            text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            self.analysis_text.configure(yscrollcommand=text_scroll.set)
            
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
                text="Clear Results",
                command=lambda: self.analysis_text.delete(1.0, tk.END)
            ).pack(side=tk.LEFT, padx=5)
            
            # Initialize analysis variables
            self.analysis_results = None
            
            self.logger.info("Analysis tab created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create analysis tab: {str(e)}")
            raise

    def update_stat_options(self, *args):
        """Update statistical analysis options based on selected test"""
        try:
            test = self.stat_test.get()
            
            # Enable/disable correction options based on test type
            if test == "chi2":
                self.correction.set(False)
                self.correction_menu.configure(state='disabled')
            else:
                self.correction_menu.configure(state='normal')
                
            self.logger.info(f"Updated statistical options for test: {test}")
            
        except Exception as e:
            self.logger.error(f"Failed to update statistical options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update options: {str(e)}")

    def update_correction_options(self):
        """Enable/disable correction method based on correction checkbox"""
        try:
            if self.correction.get():
                self.correction_menu.configure(state='normal')
            else:
                self.correction_menu.configure(state='disabled')
                
            self.logger.info(f"Updated correction options: {self.correction.get()}")
            
        except Exception as e:
            self.logger.error(f"Failed to update correction options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update options: {str(e)}")

    def validate_alpha(self, value):
        """Validate that alpha value is between 0 and 1"""
        if value == "":
            return True
        try:
            alpha = float(value)
            return 0 < alpha < 1
        except ValueError:
            return False

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
            self.cluster_method = tk.StringVar(value="kmeans")
            self.cluster_method_menu = ttk.OptionMenu(
                method_frame,
                self.cluster_method,
                "kmeans",
                "kmeans", "dbscan", "hierarchical",
                command=self.update_cluster_options
            )
            self.cluster_method_menu.pack(side=tk.LEFT, padx=5)
            
            # Initially show kmeans frame
            self.kmeans_frame.pack(fill=tk.X, padx=5, pady=5)
            
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

    def save_clustering_results(self):
        """Save clustering results to file"""
        try:
            if not hasattr(self, 'normalized_data') or self.normalized_data is None:
                raise ValueError("No clustering results available to save")
                
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if file_path:
                # Save results
                results_df = self.normalized_data.copy()
                results_df['Cluster'] = self.cluster_labels
                results_df.to_csv(file_path, index=False)
                
                self.status_var.set(f"Results saved to {file_path}")
                self.logger.info(f"Clustering results saved to {file_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to save clustering results: {str(e)}")
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")

    def update_cluster_options(self, *args):
        """Update clustering options based on selected method."""
        try:
            method = self.cluster_method.get()
            
            # Hide all parameter frames initially
            self.kmeans_frame.pack_forget()
            self.dbscan_frame.pack_forget()
            self.hierarchical_frame.pack_forget()
            
            # Show the relevant frame based on the selected method
            if method == "kmeans":
                self.kmeans_frame.pack(fill=tk.X, padx=5, pady=5)
            elif method == "dbscan":
                self.dbscan_frame.pack(fill=tk.X, padx=5, pady=5)
            elif method == "hierarchical":
                self.hierarchical_frame.pack(fill=tk.X, padx=5, pady=5)
            
            self.logger.info(f"Updated cluster options for method: {method}")
            
        except Exception as e:
            self.logger.error(f"Failed to update cluster options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update cluster options: {str(e)}")

    def create_distribution_frame(self):
        """Create frame for distribution visualization options"""
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
                # options will be populated when data is loaded
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
            
            # Additional options frame
            options_frame = ttk.Frame(self.dist_frame)
            options_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Bin count for histogram
            self.bin_count = tk.IntVar(value=30)
            ttk.Label(options_frame, text="Bins:").pack(side=tk.LEFT)
            ttk.Entry(
                options_frame,
                textvariable=self.bin_count,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # KDE bandwidth
            self.kde_bandwidth = tk.DoubleVar(value=0.2)
            ttk.Label(options_frame, text="Bandwidth:").pack(side=tk.LEFT)
            ttk.Entry(
                options_frame,
                textvariable=self.kde_bandwidth,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            self.logger.info("Distribution frame created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create distribution frame: {str(e)}")
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
            
            # Create all visualization frames
            self.create_distribution_frame()
            self.create_profile_frame()
            self.create_reduction_frame()
            self.create_importance_frame()
            
            # Initially show distribution frame
            self.dist_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Plot frame
            plot_container = ttk.LabelFrame(self.viz_tab, text="Visualization")
            plot_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Toolbar frame
            self.toolbar_frame = ttk.Frame(plot_container)
            self.toolbar_frame.pack(fill=tk.X)
            
            # Plot frame
            self.plot_frame = ttk.Frame(plot_container)
            self.plot_frame.pack(fill=tk.BOTH, expand=True)
            
            # Initialize matplotlib figure and canvas
            self.fig = Figure(figsize=(8, 6), dpi=100)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add matplotlib toolbar
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
            self.toolbar.update()
            
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

    def update_feature_menu(self):
        """Update feature menu options when data is loaded"""
        try:
            if self.normalized_data is not None:
                # Get list of numeric columns
                features = list(self.normalized_data.select_dtypes(
                    include=['float64', 'int64']).columns)
                
                # Update feature menu
                menu = self.feature_menu["menu"]
                menu.delete(0, "end")
                for feature in features:
                    menu.add_command(
                        label=feature,
                        command=lambda f=feature: self.dist_feature.set(f))
                
                # Set default feature if none selected
                if not self.dist_feature.get() and features:
                    self.dist_feature.set(features[0])
                    
                self.logger.info("Feature menu updated successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to update feature menu: {str(e)}")
            messagebox.showerror("Error", f"Failed to update feature menu: {str(e)}")

    def generate_plot(self):
        """Generate visualization based on selected plot type and options"""
        try:
            # Clear previous plot
            self.fig.clear()
            
            # Get selected plot type
            plot_type = self.plot_type.get()
            
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
            
            self.logger.info(f"Generated {plot_type} plot successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to generate plot: {str(e)}")
            messagebox.showerror("Error", f"Failed to generate plot: {str(e)}")

    def _generate_distribution_plot(self):
        """Generate distribution plot for selected feature"""
        feature = self.dist_feature.get()
        plot_type = self.dist_type.get()
        
        if not feature:
            raise ValueError("No feature selected")
            
        data = self.data[feature]
        labels = self.cluster_labels
        
        ax = self.fig.add_subplot(111)
        
        if plot_type == "histogram":
            for label in np.unique(labels):
                ax.hist(data[labels == label], alpha=0.5, 
                       label=f'Cluster {label}', bins=30)
        elif plot_type == "kde":
            for label in np.unique(labels):
                sns.kdeplot(data[labels == label], ax=ax,
                          label=f'Cluster {label}')
        else:  # box plot
            ax.boxplot([data[labels == label] 
                       for label in np.unique(labels)],
                      labels=[f'Cluster {label}' 
                             for label in np.unique(labels)])
            
        ax.set_title(f'{feature} Distribution by Cluster')
        ax.set_xlabel(feature)
        ax.legend()
        
        self.plot_data = pd.DataFrame({
            'Feature': data,
            'Cluster': labels
        })

    def _generate_profile_plot(self):
        """Generate cluster profile visualization"""
        plot_type = self.profile_type.get()
        features = [self.feature_listbox.get(i) 
                   for i in self.feature_listbox.curselection()]
        
        if not features:
            raise ValueError("No features selected")
            
        data = self.data[features]
        labels = self.cluster_labels
        
        if plot_type == "heatmap":
            cluster_means = pd.DataFrame([
                data[labels == label].mean() 
                for label in np.unique(labels)
            ], index=[f'Cluster {label}' for label in np.unique(labels)])
            
            ax = self.fig.add_subplot(111)
            sns.heatmap(cluster_means, cmap='coolwarm', center=0,
                       annot=True, ax=ax)
            ax.set_title('Cluster Profiles Heatmap')
            
        elif plot_type == "parallel":
            ax = self.fig.add_subplot(111)
            pd.plotting.parallel_coordinates(
                pd.concat([data, pd.Series(labels, name='Cluster')], axis=1),
                'Cluster', ax=ax)
            ax.set_title('Parallel Coordinates Plot')
            
        else:  # radar chart
            angles = np.linspace(0, 2*np.pi, len(features), endpoint=False)
            cluster_means = np.array([
                data[labels == label].mean() 
                for label in np.unique(labels)
            ])
            
            ax = self.fig.add_subplot(111, projection='polar')
            for i, means in enumerate(cluster_means):
                values = np.concatenate((means, [means[0]]))
                angles_plot = np.concatenate((angles, [angles[0]]))
                ax.plot(angles_plot, values, label=f'Cluster {i}')
                
            ax.set_xticks(angles)
            ax.set_xticklabels(features)
            ax.set_title('Radar Chart of Cluster Profiles')
            ax.legend()
            
        self.plot_data = pd.DataFrame({
            'Feature': np.repeat(features, len(np.unique(labels))),
            'Cluster': np.tile(np.unique(labels), len(features)),
            'Value': cluster_means.flatten()
        })

    def _generate_reduction_plot(self):
        """Generate dimensionality reduction visualization"""
        method = self.reduction_method.get()
        n_components = self.n_components.get()
        
        if method == "pca":
            reducer = PCA(n_components=n_components)
        elif method == "tsne":
            reducer = TSNE(n_components=n_components)
        else:  # umap
            reducer = umap.UMAP(n_components=n_components)
            
        reduced_data = reducer.fit_transform(self.data)
        
        if n_components == 2:
            ax = self.fig.add_subplot(111)
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1],
                               c=self.cluster_labels, cmap='viridis')
            ax.set_title(f'{method.upper()} Projection')
            self.fig.colorbar(scatter, label='Cluster')
            
        elif n_components == 3:
            ax = self.fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1],
                               reduced_data[:, 2], c=self.cluster_labels,
                               cmap='viridis')
            ax.set_title(f'{method.upper()} Projection')
            self.fig.colorbar(scatter, label='Cluster')
            
        self.plot_data = pd.DataFrame(
            reduced_data,
            columns=[f'Component {i+1}' for i in range(n_components)])
        self.plot_data['Cluster'] = self.cluster_labels

    def _generate_importance_plot(self):
        """Generate feature importance visualization"""
        method = self.importance_method.get()
        n_features = self.n_top_features.get()
        
        if method == "statistical":
            importances = []
            pvalues = []
            for col in self.data.columns:
                f_stat, p_val = f_oneway(*[
                    self.data[col][self.cluster_labels == label]
                    for label in np.unique(self.cluster_labels)
                ])
                importances.append(f_stat)
                pvalues.append(p_val)
                
            importance_df = pd.DataFrame({
                'Feature': self.data.columns,
                'Importance': importances,
                'p-value': pvalues
            })
            
        else:  # random forest
            rf = RandomForestClassifier(n_estimators=100)
            rf.fit(self.data, self.cluster_labels)
            
            importance_df = pd.DataFrame({
                'Feature': self.data.columns,
                'Importance': rf.feature_importances_
            })
            
        importance_df = importance_df.nlargest(n_features, 'Importance')
        
        ax = self.fig.add_subplot(111)
        sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax)
        ax.set_title('Feature Importance')
        
        self.plot_data = importance_df

    def save_plot(self):
        """Save current plot to file"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("SVG files", "*.svg")
                ]
            )
            
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
            
            # Components
            comp_frame = ttk.Frame(self.reduction_frame)
            comp_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(comp_frame, text="Number of components:").pack(side=tk.LEFT)
            self.n_components = tk.IntVar(value=2)
            ttk.Entry(
                comp_frame,
                textvariable=self.n_components,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Method-specific parameters frame
            self.reduction_params_frame = ttk.Frame(self.reduction_frame)
            self.reduction_params_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Initialize with PCA parameters
            self.create_pca_params()
            
            self.logger.info("Reduction frame created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create reduction frame: {str(e)}")
            raise

    def update_reduction_options(self, *args):
        """Update reduction options based on selected method"""
        try:
            method = self.reduction_method.get()
            
            # Clear existing parameters
            for widget in self.reduction_params_frame.winfo_children():
                widget.destroy()
            
            # Show method-specific parameters
            if method == "pca":
                self.create_pca_params()
            elif method == "tsne":
                self.create_tsne_params()
            elif method == "umap":
                self.create_umap_params()
            
            self.logger.info(f"Updated reduction options for method: {method}")
            
        except Exception as e:
            self.logger.error(f"Failed to update reduction options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update reduction options: {str(e)}")

    def create_pca_params(self):
        """Create parameter inputs for PCA"""
        try:
            # Explained variance ratio threshold
            var_frame = ttk.Frame(self.reduction_params_frame)
            var_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(var_frame, text="Min explained variance:").pack(side=tk.LEFT)
            self.min_variance = tk.DoubleVar(value=0.95)
            ttk.Entry(
                var_frame,
                textvariable=self.min_variance,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Whitening option
            self.whiten = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                self.reduction_params_frame,
                text="Apply whitening",
                variable=self.whiten
            ).pack(anchor=tk.W)
            
        except Exception as e:
            self.logger.error(f"Failed to create PCA parameters: {str(e)}")
            raise

    def create_tsne_params(self):
        """Create parameter inputs for t-SNE"""
        try:
            # Perplexity
            perp_frame = ttk.Frame(self.reduction_params_frame)
            perp_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(perp_frame, text="Perplexity:").pack(side=tk.LEFT)
            self.perplexity = tk.DoubleVar(value=30.0)
            ttk.Entry(
                perp_frame,
                textvariable=self.perplexity,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Learning rate
            lr_frame = ttk.Frame(self.reduction_params_frame)
            lr_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(lr_frame, text="Learning rate:").pack(side=tk.LEFT)
            self.learning_rate = tk.DoubleVar(value=200.0)
            ttk.Entry(
                lr_frame,
                textvariable=self.learning_rate,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Number of iterations
            iter_frame = ttk.Frame(self.reduction_params_frame)
            iter_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(iter_frame, text="Iterations:").pack(side=tk.LEFT)
            self.n_iter_tsne = tk.IntVar(value=1000)
            ttk.Entry(
                iter_frame,
                textvariable=self.n_iter_tsne,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            self.logger.error(f"Failed to create t-SNE parameters: {str(e)}")
            raise

    def create_umap_params(self):
        """Create parameter inputs for UMAP"""
        try:
            # Number of neighbors
            neighbors_frame = ttk.Frame(self.reduction_params_frame)
            neighbors_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(neighbors_frame, text="Number of neighbors:").pack(side=tk.LEFT)
            self.n_neighbors = tk.IntVar(value=15)
            ttk.Entry(
                neighbors_frame,
                textvariable=self.n_neighbors,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Minimum distance
            dist_frame = ttk.Frame(self.reduction_params_frame)
            dist_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(dist_frame, text="Minimum distance:").pack(side=tk.LEFT)
            self.min_dist = tk.DoubleVar(value=0.1)
            ttk.Entry(
                dist_frame,
                textvariable=self.min_dist,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Metric
            metric_frame = ttk.Frame(self.reduction_params_frame)
            metric_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(metric_frame, text="Distance metric:").pack(side=tk.LEFT)
            self.metric = tk.StringVar(value="euclidean")
            ttk.OptionMenu(
                metric_frame,
                self.metric,
                "euclidean",
                "euclidean", "manhattan", "cosine"
            ).pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            self.logger.error(f"Failed to create UMAP parameters: {str(e)}")
            raise

    def create_importance_frame(self):
        """Create frame for feature importance visualization options"""
        try:
            self.importance_frame = ttk.LabelFrame(self.viz_tab, text="Feature Importance Options")
            
            # Method selection
            method_frame = ttk.Frame(self.importance_frame)
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

    def export_plot_data(self):
        """Export data used to generate current plot"""
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
                width=5,
                validate='key',
                validatecommand=(self.register(self.validate_alpha), '%P')
            ).pack(side=tk.LEFT, padx=5)
            
            # Multiple testing correction
            correction_frame = ttk.Frame(self.stat_frame)
            correction_frame.pack(fill=tk.X, padx=5, pady=2)
            
            self.correction = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                correction_frame,
                text="Apply multiple testing correction",
                variable=self.correction,
                command=self.update_correction_options
            ).pack(side=tk.LEFT)
            
            self.correction_method = tk.StringVar(value="bonferroni")
            self.correction_menu = ttk.OptionMenu(
                correction_frame,
                self.correction_method,
                "bonferroni",
                "bonferroni", "fdr", "holm"
            )
            self.correction_menu.pack(side=tk.LEFT, padx=5)
            
            # Feature selection
            feature_frame = ttk.LabelFrame(self.analysis_tab, text="Feature Selection")
            feature_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Feature listbox with scrollbar
            listbox_frame = ttk.Frame(feature_frame)
            listbox_frame.pack(fill=tk.X, padx=5, pady=2)
            
            self.feature_listbox = tk.Listbox(
                listbox_frame,
                selectmode=tk.MULTIPLE,
                height=6
            )
            self.feature_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            feature_scroll = ttk.Scrollbar(
                listbox_frame,
                orient=tk.VERTICAL,
                command=self.feature_listbox.yview
            )
            feature_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            self.feature_listbox.configure(yscrollcommand=feature_scroll.set)
            
            # Results frame
            results_frame = ttk.LabelFrame(self.analysis_tab, text="Analysis Results")
            results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Results text with scrollbars
            text_frame = ttk.Frame(results_frame)
            text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            self.analysis_text = tk.Text(
                text_frame,
                wrap=tk.WORD,
                height=10
            )
            self.analysis_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            text_scroll = ttk.Scrollbar(
                text_frame,
                orient=tk.VERTICAL,
                command=self.analysis_text.yview
            )
            text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            self.analysis_text.configure(yscrollcommand=text_scroll.set)
            
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
                text="Clear Results",
                command=lambda: self.analysis_text.delete(1.0, tk.END)
            ).pack(side=tk.LEFT, padx=5)
            
            # Initialize analysis variables
            self.analysis_results = None
            
            self.logger.info("Analysis tab created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create analysis tab: {str(e)}")
            raise

    def validate_alpha(self, value):
        """Validate that alpha value is between 0 and 1"""
        if value == "":
            return True
        try:
            alpha = float(value)
            return 0 < alpha < 1
        except ValueError:
            return False

    def update_stat_options(self, *args):
        """Update statistical analysis options based on selected test"""
        try:
            test = self.stat_test.get()
            
            # Enable/disable correction options based on test type
            if test == "chi2":
                self.correction.set(False)
                self.correction_menu.configure(state='disabled')
            else:
                self.correction_menu.configure(state='normal')
                
            self.logger.info(f"Updated statistical options for test: {test}")
            
        except Exception as e:
            self.logger.error(f"Failed to update statistical options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update options: {str(e)}")

    def update_correction_options(self):
        """Enable/disable correction method based on correction checkbox"""
        try:
            if self.correction.get():
                self.correction_menu.configure(state='normal')
            else:
                self.correction_menu.configure(state='disabled')
                
            self.logger.info(f"Updated correction options: {self.correction.get()}")
            
        except Exception as e:
            self.logger.error(f"Failed to update correction options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update options: {str(e)}")

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
            self.cluster_method = tk.StringVar(value="kmeans")
            self.cluster_method_menu = ttk.OptionMenu(
                method_frame,
                self.cluster_method,
                "kmeans",
                "kmeans", "dbscan", "hierarchical",
                command=self.update_cluster_options
            )
            self.cluster_method_menu.pack(side=tk.LEFT, padx=5)
            
            # Initially show kmeans frame
            self.kmeans_frame.pack(fill=tk.X, padx=5, pady=5)
            
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

    def save_clustering_results(self):
        """Save clustering results to file"""
        try:
            if not hasattr(self, 'normalized_data') or self.normalized_data is None:
                raise ValueError("No clustering results available to save")
                
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if file_path:
                # Save results
                results_df = self.normalized_data.copy()
                results_df['Cluster'] = self.cluster_labels
                results_df.to_csv(file_path, index=False)
                
                self.status_var.set(f"Results saved to {file_path}")
                self.logger.info(f"Clustering results saved to {file_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to save clustering results: {str(e)}")
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")

    def update_cluster_options(self, *args):
        """Update clustering options based on selected method."""
        try:
            method = self.cluster_method.get()
            
            # Hide all parameter frames initially
            self.kmeans_frame.pack_forget()
            self.dbscan_frame.pack_forget()
            self.hierarchical_frame.pack_forget()
            
            # Show the relevant frame based on the selected method
            if method == "kmeans":
                self.kmeans_frame.pack(fill=tk.X, padx=5, pady=5)
            elif method == "dbscan":
                self.dbscan_frame.pack(fill=tk.X, padx=5, pady=5)
            elif method == "hierarchical":
                self.hierarchical_frame.pack(fill=tk.X, padx=5, pady=5)
            
            self.logger.info(f"Updated cluster options for method: {method}")
            
        except Exception as e:
            self.logger.error(f"Failed to update cluster options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update cluster options: {str(e)}")

    def create_distribution_frame(self):
        """Create frame for distribution visualization options"""
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
                # options will be populated when data is loaded
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
            
            # Additional options frame
            options_frame = ttk.Frame(self.dist_frame)
            options_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Bin count for histogram
            self.bin_count = tk.IntVar(value=30)
            ttk.Label(options_frame, text="Bins:").pack(side=tk.LEFT)
            ttk.Entry(
                options_frame,
                textvariable=self.bin_count,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # KDE bandwidth
            self.kde_bandwidth = tk.DoubleVar(value=0.2)
            ttk.Label(options_frame, text="Bandwidth:").pack(side=tk.LEFT)
            ttk.Entry(
                options_frame,
                textvariable=self.kde_bandwidth,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            self.logger.info("Distribution frame created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create distribution frame: {str(e)}")
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
            
            # Create all visualization frames
            self.create_distribution_frame()
            self.create_profile_frame()
            self.create_reduction_frame()
            self.create_importance_frame()
            
            # Initially show distribution frame
            self.dist_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Plot frame
            plot_container = ttk.LabelFrame(self.viz_tab, text="Visualization")
            plot_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Toolbar frame
            self.toolbar_frame = ttk.Frame(plot_container)
            self.toolbar_frame.pack(fill=tk.X)
            
            # Plot frame
            self.plot_frame = ttk.Frame(plot_container)
            self.plot_frame.pack(fill=tk.BOTH, expand=True)
            
            # Initialize matplotlib figure and canvas
            self.fig = Figure(figsize=(8, 6), dpi=100)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add matplotlib toolbar
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
            self.toolbar.update()
            
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

    def update_feature_menu(self):
        """Update feature menu options when data is loaded"""
        try:
            if self.normalized_data is not None:
                # Get list of numeric columns
                features = list(self.normalized_data.select_dtypes(
                    include=['float64', 'int64']).columns)
                
                # Update feature menu
                menu = self.feature_menu["menu"]
                menu.delete(0, "end")
                
                for feature in features:
                    menu.add_command(
                        label=feature,
                        command=lambda f=feature: self.dist_feature.set(f)
                    )
                
                # Set default feature if none selected
                if not self.dist_feature.get() and features:
                    self.dist_feature.set(features[0])
                    
                self.logger.info("Feature menu updated successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to update feature menu: {str(e)}")
            messagebox.showerror("Error", f"Failed to update feature menu: {str(e)}")

    def generate_plot(self):
        """Generate visualization based on selected plot type and options"""
        try:
            # Clear previous plot
            self.fig.clear()
            
            # Get selected plot type
            plot_type = self.plot_type.get()
            
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
            
            self.logger.info(f"Generated {plot_type} plot successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to generate plot: {str(e)}")
            messagebox.showerror("Error", f"Failed to generate plot: {str(e)}")

    def _generate_distribution_plot(self):
        """Generate distribution plot for selected feature"""
        feature = self.dist_feature.get()
        plot_type = self.dist_type.get()
        
        if not feature:
            raise ValueError("No feature selected")
            
        data = self.data[feature]
        labels = self.cluster_labels
        
        ax = self.fig.add_subplot(111)
        
        if plot_type == "histogram":
            for label in np.unique(labels):
                ax.hist(data[labels == label], alpha=0.5, 
                       label=f'Cluster {label}', bins=30)
        elif plot_type == "kde":
            for label in np.unique(labels):
                sns.kdeplot(data[labels == label], ax=ax,
                          label=f'Cluster {label}')
        else:  # box plot
            ax.boxplot([data[labels == label] 
                       for label in np.unique(labels)],
                      labels=[f'Cluster {label}' 
                             for label in np.unique(labels)])
            
        ax.set_title(f'{feature} Distribution by Cluster')
        ax.set_xlabel(feature)
        ax.legend()
        
        self.plot_data = pd.DataFrame({
            'Feature': data,
            'Cluster': labels
        })

    def _generate_profile_plot(self):
        """Generate cluster profile visualization"""
        plot_type = self.profile_type.get()
        features = [self.feature_listbox.get(i) 
                   for i in self.feature_listbox.curselection()]
        
        if not features:
            raise ValueError("No features selected")
            
        data = self.data[features]
        labels = self.cluster_labels
        
        if plot_type == "heatmap":
            cluster_means = pd.DataFrame([
                data[labels == label].mean() 
                for label in np.unique(labels)
            ], index=[f'Cluster {label}' for label in np.unique(labels)])
            
            ax = self.fig.add_subplot(111)
            sns.heatmap(cluster_means, cmap='coolwarm', center=0,
                       annot=True, ax=ax)
            ax.set_title('Cluster Profiles Heatmap')
            
        elif plot_type == "parallel":
            ax = self.fig.add_subplot(111)
            pd.plotting.parallel_coordinates(
                pd.concat([data, pd.Series(labels, name='Cluster')], axis=1),
                'Cluster', ax=ax)
            ax.set_title('Parallel Coordinates Plot')
            
        else:  # radar chart
            angles = np.linspace(0, 2*np.pi, len(features), endpoint=False)
            cluster_means = np.array([
                data[labels == label].mean() 
                for label in np.unique(labels)
            ])
            
            ax = self.fig.add_subplot(111, projection='polar')
            for i, means in enumerate(cluster_means):
                values = np.concatenate((means, [means[0]]))
                angles_plot = np.concatenate((angles, [angles[0]]))
                ax.plot(angles_plot, values, label=f'Cluster {i}')
                
            ax.set_xticks(angles)
            ax.set_xticklabels(features)
            ax.set_title('Radar Chart of Cluster Profiles')
            ax.legend()
            
        self.plot_data = pd.DataFrame({
            'Feature': np.repeat(features, len(np.unique(labels))),
            'Cluster': np.tile(np.unique(labels), len(features)),
            'Value': cluster_means.flatten()
        })

    def _generate_reduction_plot(self):
        """Generate dimensionality reduction visualization"""
        method = self.reduction_method.get()
        n_components = self.n_components.get()
        
        if method == "pca":
            reducer = PCA(n_components=n_components)
        elif method == "tsne":
            reducer = TSNE(n_components=n_components)
        else:  # umap
            reducer = umap.UMAP(n_components=n_components)
            
        reduced_data = reducer.fit_transform(self.data)
        
        if n_components == 2:
            ax = self.fig.add_subplot(111)
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1],
                               c=self.cluster_labels, cmap='viridis')
            ax.set_title(f'{method.upper()} Projection')
            self.fig.colorbar(scatter, label='Cluster')
            
        elif n_components == 3:
            ax = self.fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1],
                               reduced_data[:, 2], c=self.cluster_labels,
                               cmap='viridis')
            ax.set_title(f'{method.upper()} Projection')
            self.fig.colorbar(scatter, label='Cluster')
            
        self.plot_data = pd.DataFrame(
            reduced_data,
            columns=[f'Component {i+1}' for i in range(n_components)])
        self.plot_data['Cluster'] = self.cluster_labels

    def _generate_importance_plot(self):
        """Generate feature importance visualization"""
        method = self.importance_method.get()
        n_features = self.n_top_features.get()
        
        if method == "statistical":
            importances = []
            pvalues = []
            for col in self.data.columns:
                f_stat, p_val = f_oneway(*[
                    self.data[col][self.cluster_labels == label]
                    for label in np.unique(self.cluster_labels)
                ])
                importances.append(f_stat)
                pvalues.append(p_val)
                
            importance_df = pd.DataFrame({
                'Feature': self.data.columns,
                'Importance': importances,
                'p-value': pvalues
            })
            
        else:  # random forest
            rf = RandomForestClassifier(n_estimators=100)
            rf.fit(self.data, self.cluster_labels)
            
            importance_df = pd.DataFrame({
                'Feature': self.data.columns,
                'Importance': rf.feature_importances_
            })
            
        importance_df = importance_df.nlargest(n_features, 'Importance')
        
        ax = self.fig.add_subplot(111)
        sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax)
        ax.set_title('Feature Importance')
        
        self.plot_data = importance_df

    def save_plot(self):
        """Save current plot to file"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("SVG files", "*.svg")
                ]
            )
            
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
            
            # Components
            comp_frame = ttk.Frame(self.reduction_frame)
            comp_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(comp_frame, text="Number of components:").pack(side=tk.LEFT)
            self.n_components = tk.IntVar(value=2)
            ttk.Entry(
                comp_frame,
                textvariable=self.n_components,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Method-specific parameters frame
            self.reduction_params_frame = ttk.Frame(self.reduction_frame)
            self.reduction_params_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Initialize with PCA parameters
            self.create_pca_params()
            
            self.logger.info("Reduction frame created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create reduction frame: {str(e)}")
            raise

    def update_reduction_options(self, *args):
        """Update reduction options based on selected method"""
        try:
            method = self.reduction_method.get()
            
            # Clear existing parameters
            for widget in self.reduction_params_frame.winfo_children():
                widget.destroy()
            
            # Show method-specific parameters
            if method == "pca":
                self.create_pca_params()
            elif method == "tsne":
                self.create_tsne_params()
            elif method == "umap":
                self.create_umap_params()
            
            self.logger.info(f"Updated reduction options for method: {method}")
            
        except Exception as e:
            self.logger.error(f"Failed to update reduction options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update reduction options: {str(e)}")

    def create_pca_params(self):
        """Create parameter inputs for PCA"""
        try:
            # Explained variance ratio threshold
            var_frame = ttk.Frame(self.reduction_params_frame)
            var_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(var_frame, text="Min explained variance:").pack(side=tk.LEFT)
            self.min_variance = tk.DoubleVar(value=0.95)
            ttk.Entry(
                var_frame,
                textvariable=self.min_variance,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Whitening option
            self.whiten = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                self.reduction_params_frame,
                text="Apply whitening",
                variable=self.whiten
            ).pack(anchor=tk.W)
            
        except Exception as e:
            self.logger.error(f"Failed to create PCA parameters: {str(e)}")
            raise

    def create_tsne_params(self):
        """Create parameter inputs for t-SNE"""
        try:
            # Perplexity
            perp_frame = ttk.Frame(self.reduction_params_frame)
            perp_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(perp_frame, text="Perplexity:").pack(side=tk.LEFT)
            self.perplexity = tk.DoubleVar(value=30.0)
            ttk.Entry(
                perp_frame,
                textvariable=self.perplexity,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Learning rate
            lr_frame = ttk.Frame(self.reduction_params_frame)
            lr_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(lr_frame, text="Learning rate:").pack(side=tk.LEFT)
            self.learning_rate = tk.DoubleVar(value=200.0)
            ttk.Entry(
                lr_frame,
                textvariable=self.learning_rate,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Number of iterations
            iter_frame = ttk.Frame(self.reduction_params_frame)
            iter_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(iter_frame, text="Iterations:").pack(side=tk.LEFT)
            self.n_iter_tsne = tk.IntVar(value=1000)
            ttk.Entry(
                iter_frame,
                textvariable=self.n_iter_tsne,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            self.logger.error(f"Failed to create t-SNE parameters: {str(e)}")
            raise

    def create_umap_params(self):
        """Create parameter inputs for UMAP"""
        try:
            # Number of neighbors
            neighbors_frame = ttk.Frame(self.reduction_params_frame)
            neighbors_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(neighbors_frame, text="Number of neighbors:").pack(side=tk.LEFT)
            self.n_neighbors = tk.IntVar(value=15)
            ttk.Entry(
                neighbors_frame,
                textvariable=self.n_neighbors,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Minimum distance
            dist_frame = ttk.Frame(self.reduction_params_frame)
            dist_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(dist_frame, text="Minimum distance:").pack(side=tk.LEFT)
            self.min_dist = tk.DoubleVar(value=0.1)
            ttk.Entry(
                dist_frame,
                textvariable=self.min_dist,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Metric
            metric_frame = ttk.Frame(self.reduction_params_frame)
            metric_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(metric_frame, text="Distance metric:").pack(side=tk.LEFT)
            self.metric = tk.StringVar(value="euclidean")
            ttk.OptionMenu(
                metric_frame,
                self.metric,
                "euclidean",
                "euclidean", "manhattan", "cosine"
            ).pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            self.logger.error(f"Failed to create UMAP parameters: {str(e)}")
            raise

    def create_importance_frame(self):
        """Create frame for feature importance visualization options"""
        try:
            self.importance_frame = ttk.LabelFrame(self.viz_tab, text="Feature Importance Options")
            
            # Method selection
            method_frame = ttk.Frame(self.importance_frame)
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

    def export_plot_data(self):
        """Export data used to generate current plot"""
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
                width=5,
                validate='key',
                validatecommand=(self.register(self.validate_alpha), '%P')
            ).pack(side=tk.LEFT, padx=5)
            
            # Multiple testing correction
            correction_frame = ttk.Frame(self.stat_frame)
            correction_frame.pack(fill=tk.X, padx=5, pady=2)
            
            self.correction = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                correction_frame,
                text="Apply multiple testing correction",
                variable=self.correction,
                command=self.update_correction_options
            ).pack(side=tk.LEFT)
            
            self.correction_method = tk.StringVar(value="bonferroni")
            self.correction_menu = ttk.OptionMenu(
                correction_frame,
                self.correction_method,
                "bonferroni",
                "bonferroni", "fdr", "holm"
            )
            self.correction_menu.pack(side=tk.LEFT, padx=5)
            
            # Feature selection
            feature_frame = ttk.LabelFrame(self.analysis_tab, text="Feature Selection")
            feature_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Feature listbox with scrollbar
            listbox_frame = ttk.Frame(feature_frame)
            listbox_frame.pack(fill=tk.X, padx=5, pady=2)
            
            self.feature_listbox = tk.Listbox(
                listbox_frame,
                selectmode=tk.MULTIPLE,
                height=6
            )
            self.feature_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            feature_scroll = ttk.Scrollbar(
                listbox_frame,
                orient=tk.VERTICAL,
                command=self.feature_listbox.yview
            )
            feature_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            self.feature_listbox.configure(yscrollcommand=feature_scroll.set)
            
            # Results frame
            results_frame = ttk.LabelFrame(self.analysis_tab, text="Analysis Results")
            results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Results text with scrollbars
            text_frame = ttk.Frame(results_frame)
            text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            self.analysis_text = tk.Text(
                text_frame,
                wrap=tk.WORD,
                height=10
            )
            self.analysis_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            text_scroll = ttk.Scrollbar(
                text_frame,
                orient=tk.VERTICAL,
                command=self.analysis_text.yview
            )
            text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            self.analysis_text.configure(yscrollcommand=text_scroll.set)
            
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
                text="Clear Results",
                command=lambda: self.analysis_text.delete(1.0, tk.END)
            ).pack(side=tk.LEFT, padx=5)
            
            # Initialize analysis variables
            self.analysis_results = None
            
            self.logger.info("Analysis tab created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create analysis tab: {str(e)}")
            raise

    def validate_alpha(self, value):
        """Validate that alpha value is between 0 and 1"""
        if value == "":
            return True
        try:
            alpha = float(value)
            return 0 < alpha < 1
        except ValueError:
            return False

    def update_stat_options(self, *args):
        """Update statistical analysis options based on selected test"""
        try:
            test = self.stat_test.get()
            
            # Enable/disable correction options based on test type
            if test == "chi2":
                self.correction.set(False)
                self.correction_menu.configure(state='disabled')
            else:
                self.correction_menu.configure(state='normal')
                
            self.logger.info(f"Updated statistical options for test: {test}")
            
        except Exception as e:
            self.logger.error(f"Failed to update statistical options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update options: {str(e)}")

    def update_correction_options(self):
        """Enable/disable correction method based on correction checkbox"""
        try:
            if self.correction.get():
                self.correction_menu.configure(state='normal')
            else:
                self.correction_menu.configure(state='disabled')
                
            self.logger.info(f"Updated correction options: {self.correction.get()}")
            
        except Exception as e:
            self.logger.error(f"Failed to update correction options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update options: {str(e)}")

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
            self.cluster_method = tk.StringVar(value="kmeans")
            self.cluster_method_menu = ttk.OptionMenu(
                method_frame,
                self.cluster_method,
                "kmeans",
                "kmeans", "dbscan", "hierarchical",
                command=self.update_cluster_options
            )
            self.cluster_method_menu.pack(side=tk.LEFT, padx=5)
            
            # Initially show kmeans frame
            self.kmeans_frame.pack(fill=tk.X, padx=5, pady=5)
            
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

    def save_clustering_results(self):
        """Save clustering results to file"""
        try:
            if not hasattr(self, 'normalized_data') or self.normalized_data is None:
                raise ValueError("No clustering results available to save")
                
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if file_path:
                # Save results
                results_df = self.normalized_data.copy()
                results_df['Cluster'] = self.cluster_labels
                results_df.to_csv(file_path, index=False)
                
                self.status_var.set(f"Results saved to {file_path}")
                self.logger.info(f"Clustering results saved to {file_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to save clustering results: {str(e)}")
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")

    def update_cluster_options(self, *args):
        """Update clustering options based on selected method."""
        try:
            method = self.cluster_method.get()
            
            # Hide all parameter frames initially
            self.kmeans_frame.pack_forget()
            self.dbscan_frame.pack_forget()
            self.hierarchical_frame.pack_forget()
            
            # Show the relevant frame based on the selected method
            if method == "kmeans":
                self.kmeans_frame.pack(fill=tk.X, padx=5, pady=5)
            elif method == "dbscan":
                self.dbscan_frame.pack(fill=tk.X, padx=5, pady=5)
            elif method == "hierarchical":
                self.hierarchical_frame.pack(fill=tk.X, padx=5, pady=5)
            
            self.logger.info(f"Updated cluster options for method: {method}")
            
        except Exception as e:
            self.logger.error(f"Failed to update cluster options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update cluster options: {str(e)}")

    def create_distribution_frame(self):
        """Create frame for distribution visualization options"""
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
                # options will be populated when data is loaded
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
            
            # Additional options frame
            options_frame = ttk.Frame(self.dist_frame)
            options_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Bin count for histogram
            self.bin_count = tk.IntVar(value=30)
            ttk.Label(options_frame, text="Bins:").pack(side=tk.LEFT)
            ttk.Entry(
                options_frame,
                textvariable=self.bin_count,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # KDE bandwidth
            self.kde_bandwidth = tk.DoubleVar(value=0.2)
            ttk.Label(options_frame, text="Bandwidth:").pack(side=tk.LEFT)
            ttk.Entry(
                options_frame,
                textvariable=self.kde_bandwidth,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            self.logger.info("Distribution frame created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create distribution frame: {str(e)}")
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
            
            # Create all visualization frames
            self.create_distribution_frame()
            self.create_profile_frame()
            self.create_reduction_frame()
            self.create_importance_frame()
            
            # Initially show distribution frame
            self.dist_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Plot frame
            plot_container = ttk.LabelFrame(self.viz_tab, text="Visualization")
            plot_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Toolbar frame
            self.toolbar_frame = ttk.Frame(plot_container)
            self.toolbar_frame.pack(fill=tk.X)
            
            # Plot frame
            self.plot_frame = ttk.Frame(plot_container)
            self.plot_frame.pack(fill=tk.BOTH, expand=True)
            
            # Initialize matplotlib figure and canvas
            self.fig = Figure(figsize=(8, 6), dpi=100)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add matplotlib toolbar
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
            self.toolbar.update()
            
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

    def update_feature_menu(self):
        """Update feature menu options when data is loaded"""
        try:
            if self.normalized_data is not None:
                # Get list of numeric columns
                features = list(self.normalized_data.select_dtypes(
                    include=['float64', 'int64']).columns)
                
                # Update feature menu
                menu = self.feature_menu["menu"]
                menu.delete(0, "end")
                
                for feature in features:
                    menu.add_command(
                        label=feature,
                        command=lambda f=feature: self.dist_feature.set(f)
                    )
                
                # Set default feature if none selected
                if not self.dist_feature.get() and features:
                    self.dist_feature.set(features[0])
                    
                self.logger.info("Feature menu updated successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to update feature menu: {str(e)}")
            messagebox.showerror("Error", f"Failed to update feature menu: {str(e)}")

    def generate_plot(self):
        """Generate visualization based on selected plot type and options"""
        try:
            # Clear previous plot
            self.fig.clear()
            
            # Get selected plot type
            plot_type = self.plot_type.get()
            
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
            
            self.logger.info(f"Generated {plot_type} plot successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to generate plot: {str(e)}")
            messagebox.showerror("Error", f"Failed to generate plot: {str(e)}")

    def _generate_distribution_plot(self):
        """Generate distribution plot for selected feature"""
        feature = self.dist_feature.get()
        plot_type = self.dist_type.get()
        
        if not feature:
            raise ValueError("No feature selected")
            
        data = self.data[feature]
        labels = self.cluster_labels
        
        ax = self.fig.add_subplot(111)
        
        if plot_type == "histogram":
            for label in np.unique(labels):
                ax.hist(data[labels == label], alpha=0.5, 
                       label=f'Cluster {label}', bins=30)
        elif plot_type == "kde":
            for label in np.unique(labels):
                sns.kdeplot(data[labels == label], ax=ax,
                          label=f'Cluster {label}')
        else:  # box plot
            ax.boxplot([data[labels == label] 
                       for label in np.unique(labels)],
                      labels=[f'Cluster {label}' 
                             for label in np.unique(labels)])
            
        ax.set_title(f'{feature} Distribution by Cluster')
        ax.set_xlabel(feature)
        ax.legend()
        
        self.plot_data = pd.DataFrame({
            'Feature': data,
            'Cluster': labels
        })

    def _generate_profile_plot(self):
        """Generate cluster profile visualization"""
        plot_type = self.profile_type.get()
        features = [self.feature_listbox.get(i) 
                   for i in self.feature_listbox.curselection()]
        
        if not features:
            raise ValueError("No features selected")
            
        data = self.data[features]
        labels = self.cluster_labels
        
        if plot_type == "heatmap":
            cluster_means = pd.DataFrame([
                data[labels == label].mean() 
                for label in np.unique(labels)
            ], index=[f'Cluster {label}' for label in np.unique(labels)])
            
            ax = self.fig.add_subplot(111)
            sns.heatmap(cluster_means, cmap='coolwarm', center=0,
                       annot=True, ax=ax)
            ax.set_title('Cluster Profiles Heatmap')
            
        elif plot_type == "parallel":
            ax = self.fig.add_subplot(111)
            pd.plotting.parallel_coordinates(
                pd.concat([data, pd.Series(labels, name='Cluster')], axis=1),
                'Cluster', ax=ax)
            ax.set_title('Parallel Coordinates Plot')
            
        else:  # radar chart
            angles = np.linspace(0, 2*np.pi, len(features), endpoint=False)
            cluster_means = np.array([
                data[labels == label].mean() 
                for label in np.unique(labels)
            ])
            
            ax = self.fig.add_subplot(111, projection='polar')
            for i, means in enumerate(cluster_means):
                values = np.concatenate((means, [means[0]]))
                angles_plot = np.concatenate((angles, [angles[0]]))
                ax.plot(angles_plot, values, label=f'Cluster {i}')
                
            ax.set_xticks(angles)
            ax.set_xticklabels(features)
            ax.set_title('Radar Chart of Cluster Profiles')
            ax.legend()
            
        self.plot_data = pd.DataFrame({
            'Feature': np.repeat(features, len(np.unique(labels))),
            'Cluster': np.tile(np.unique(labels), len(features)),
            'Value': cluster_means.flatten()
        })

    def _generate_reduction_plot(self):
        """Generate dimensionality reduction visualization"""
        method = self.reduction_method.get()
        n_components = self.n_components.get()
        
        if method == "pca":
            reducer = PCA(n_components=n_components)
        elif method == "tsne":
            reducer = TSNE(n_components=n_components)
        else:  # umap
            reducer = umap.UMAP(n_components=n_components)
            
        reduced_data = reducer.fit_transform(self.data)
        
        if n_components == 2:
            ax = self.fig.add_subplot(111)
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1],
                               c=self.cluster_labels, cmap='viridis')
            ax.set_title(f'{method.upper()} Projection')
            self.fig.colorbar(scatter, label='Cluster')
            
        elif n_components == 3:
            ax = self.fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1],
                               reduced_data[:, 2], c=self.cluster_labels,
                               cmap='viridis')
            ax.set_title(f'{method.upper()} Projection')
            self.fig.colorbar(scatter, label='Cluster')
            
        self.plot_data = pd.DataFrame(
            reduced_data,
            columns=[f'Component {i+1}' for i in range(n_components)])
        self.plot_data['Cluster'] = self.cluster_labels

    def _generate_importance_plot(self):
        """Generate feature importance visualization"""
        method = self.importance_method.get()
        n_features = self.n_top_features.get()
        
        if method == "statistical":
            importances = []
            pvalues = []
            for col in self.data.columns:
                f_stat, p_val = f_oneway(*[
                    self.data[col][self.cluster_labels == label]
                    for label in np.unique(self.cluster_labels)
                ])
                importances.append(f_stat)
                pvalues.append(p_val)
                
            importance_df = pd.DataFrame({
                'Feature': self.data.columns,
                'Importance': importances,
                'p-value': pvalues
            })
            
        else:  # random forest
            rf = RandomForestClassifier(n_estimators=100)
            rf.fit(self.data, self.cluster_labels)
            
            importance_df = pd.DataFrame({
                'Feature': self.data.columns,
                'Importance': rf.feature_importances_
            })
            
        importance_df = importance_df.nlargest(n_features, 'Importance')
        
        ax = self.fig.add_subplot(111)
        sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax)
        ax.set_title('Feature Importance')
        
        self.plot_data = importance_df

    def save_plot(self):
        """Save current plot to file"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("SVG files", "*.svg")
                ]
            )
            
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
            
            # Components
            comp_frame = ttk.Frame(self.reduction_frame)
            comp_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(comp_frame, text="Number of components:").pack(side=tk.LEFT)
            self.n_components = tk.IntVar(value=2)
            ttk.Entry(
                comp_frame,
                textvariable=self.n_components,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Method-specific parameters frame
            self.reduction_params_frame = ttk.Frame(self.reduction_frame)
            self.reduction_params_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Initialize with PCA parameters
            self.create_pca_params()
            
            self.logger.info("Reduction frame created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create reduction frame: {str(e)}")
            raise

    def update_reduction_options(self, *args):
        """Update reduction options based on selected method"""
        try:
            method = self.reduction_method.get()
            
            # Clear existing parameters
            for widget in self.reduction_params_frame.winfo_children():
                widget.destroy()
            
            # Show method-specific parameters
            if method == "pca":
                self.create_pca_params()
            elif method == "tsne":
                self.create_tsne_params()
            elif method == "umap":
                self.create_umap_params()
            
            self.logger.info(f"Updated reduction options for method: {method}")
            
        except Exception as e:
            self.logger.error(f"Failed to update reduction options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update reduction options: {str(e)}")

    def create_pca_params(self):
        """Create parameter inputs for PCA"""
        try:
            # Explained variance ratio threshold
            var_frame = ttk.Frame(self.reduction_params_frame)
            var_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(var_frame, text="Min explained variance:").pack(side=tk.LEFT)
            self.min_variance = tk.DoubleVar(value=0.95)
            ttk.Entry(
                var_frame,
                textvariable=self.min_variance,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Whitening option
            self.whiten = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                self.reduction_params_frame,
                text="Apply whitening",
                variable=self.whiten
            ).pack(anchor=tk.W)
            
        except Exception as e:
            self.logger.error(f"Failed to create PCA parameters: {str(e)}")
            raise

    def create_tsne_params(self):
        """Create parameter inputs for t-SNE"""
        try:
            # Perplexity
            perp_frame = ttk.Frame(self.reduction_params_frame)
            perp_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(perp_frame, text="Perplexity:").pack(side=tk.LEFT)
            self.perplexity = tk.DoubleVar(value=30.0)
            ttk.Entry(
                perp_frame,
                textvariable=self.perplexity,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Learning rate
            lr_frame = ttk.Frame(self.reduction_params_frame)
            lr_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(lr_frame, text="Learning rate:").pack(side=tk.LEFT)
            self.learning_rate = tk.DoubleVar(value=200.0)
            ttk.Entry(
                lr_frame,
                textvariable=self.learning_rate,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Number of iterations
            iter_frame = ttk.Frame(self.reduction_params_frame)
            iter_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(iter_frame, text="Iterations:").pack(side=tk.LEFT)
            self.n_iter_tsne = tk.IntVar(value=1000)
            ttk.Entry(
                iter_frame,
                textvariable=self.n_iter_tsne,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            self.logger.error(f"Failed to create t-SNE parameters: {str(e)}")
            raise

    def create_umap_params(self):
        """Create parameter inputs for UMAP"""
        try:
            # Number of neighbors
            neighbors_frame = ttk.Frame(self.reduction_params_frame)
            neighbors_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(neighbors_frame, text="Number of neighbors:").pack(side=tk.LEFT)
            self.n_neighbors = tk.IntVar(value=15)
            ttk.Entry(
                neighbors_frame,
                textvariable=self.n_neighbors,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Minimum distance
            dist_frame = ttk.Frame(self.reduction_params_frame)
            dist_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(dist_frame, text="Minimum distance:").pack(side=tk.LEFT)
            self.min_dist = tk.DoubleVar(value=0.1)
            ttk.Entry(
                dist_frame,
                textvariable=self.min_dist,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Metric
            metric_frame = ttk.Frame(self.reduction_params_frame)
            metric_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(metric_frame, text="Distance metric:").pack(side=tk.LEFT)
            self.metric = tk.StringVar(value="euclidean")
            ttk.OptionMenu(
                metric_frame,
                self.metric,
                "euclidean",
                "euclidean", "manhattan", "cosine"
            ).pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            self.logger.error(f"Failed to create UMAP parameters: {str(e)}")
            raise

    def create_importance_frame(self):
        """Create frame for feature importance visualization options"""
        try:
            self.importance_frame = ttk.LabelFrame(self.viz_tab, text="Feature Importance Options")
            
            # Method selection
            method_frame = ttk.Frame(self.importance_frame)
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

    def export_plot_data(self):
        """Export data used to generate current plot"""
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
                width=5,
                validate='key',
                validatecommand=(self.register(self.validate_alpha), '%P')
            ).pack(side=tk.LEFT, padx=5)
            
            # Multiple testing correction
            correction_frame = ttk.Frame(self.stat_frame)
            correction_frame.pack(fill=tk.X, padx=5, pady=2)
            
            self.correction = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                correction_frame,
                text="Apply multiple testing correction",
                variable=self.correction,
                command=self.update_correction_options
            ).pack(side=tk.LEFT)
            
            self.correction_method = tk.StringVar(value="bonferroni")
            self.correction_menu = ttk.OptionMenu(
                correction_frame,
                self.correction_method,
                "bonferroni",
                "bonferroni", "fdr", "holm"
            )
            self.correction_menu.pack(side=tk.LEFT, padx=5)
            
            # Feature selection
            feature_frame = ttk.LabelFrame(self.analysis_tab, text="Feature Selection")
            feature_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Feature listbox with scrollbar
            listbox_frame = ttk.Frame(feature_frame)
            listbox_frame.pack(fill=tk.X, padx=5, pady=2)
            
            self.feature_listbox = tk.Listbox(
                listbox_frame,
                selectmode=tk.MULTIPLE,
                height=6
            )
            self.feature_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            feature_scroll = ttk.Scrollbar(
                listbox_frame,
                orient=tk.VERTICAL,
                command=self.feature_listbox.yview
            )
            feature_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            self.feature_listbox.configure(yscrollcommand=feature_scroll.set)
            
            # Results frame
            results_frame = ttk.LabelFrame(self.analysis_tab, text="Analysis Results")
            results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Results text with scrollbars
            text_frame = ttk.Frame(results_frame)
            text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            self.analysis_text = tk.Text(
                text_frame,
                wrap=tk.WORD,
                height=10
            )
            self.analysis_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            text_scroll = ttk.Scrollbar(
                text_frame,
                orient=tk.VERTICAL,
                command=self.analysis_text.yview
            )
            text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            self.analysis_text.configure(yscrollcommand=text_scroll.set)
            
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
                text="Clear Results",
                command=lambda: self.analysis_text.delete(1.0, tk.END)
            ).pack(side=tk.LEFT, padx=5)
            
            # Initialize analysis variables
            self.analysis_results = None
            
            self.logger.info("Analysis tab created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create analysis tab: {str(e)}")
            raise

    def validate_alpha(self, value):
        """Validate that alpha value is between 0 and 1"""
        if value == "":
            return True
        try:
            alpha = float(value)
            return 0 < alpha < 1
        except ValueError:
            return False

    def update_stat_options(self, *args):
        """Update statistical analysis options based on selected test"""
        try:
            test = self.stat_test.get()
            
            # Enable/disable correction options based on test type
            if test == "chi2":
                self.correction.set(False)
                self.correction_menu.configure(state='disabled')
            else:
                self.correction_menu.configure(state='normal')
                
            self.logger.info(f"Updated statistical options for test: {test}")
            
        except Exception as e:
            self.logger.error(f"Failed to update statistical options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update options: {str(e)}")

    def update_correction_options(self):
        """Enable/disable correction method based on correction checkbox"""
        try:
            if self.correction.get():
                self.correction_menu.configure(state='normal')
            else:
                self.correction_menu.configure(state='disabled')
                
            self.logger.info(f"Updated correction options: {self.correction.get()}")
            
        except Exception as e:
            self.logger.error(f"Failed to update correction options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update options: {str(e)}")

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
            self.cluster_method = tk.StringVar(value="kmeans")
            self.cluster_method_menu = ttk.OptionMenu(
                method_frame,
                self.cluster_method,
                "kmeans",
                "kmeans", "dbscan", "hierarchical",
                command=self.update_cluster_options
            )
            self.cluster_method_menu.pack(side=tk.LEFT, padx=5)
            
            # Initially show kmeans frame
            self.kmeans_frame.pack(fill=tk.X, padx=5, pady=5)
            
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

    def save_clustering_results(self):
        """Save clustering results to file"""
        try:
            if not hasattr(self, 'normalized_data') or self.normalized_data is None:
                raise ValueError("No clustering results available to save")
                
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if file_path:
                # Save results
                results_df = self.normalized_data.copy()
                results_df['Cluster'] = self.cluster_labels
                results_df.to_csv(file_path, index=False)
                
                self.status_var.set(f"Results saved to {file_path}")
                self.logger.info(f"Clustering results saved to {file_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to save clustering results: {str(e)}")
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")

    def update_cluster_options(self, *args):
        """Update clustering options based on selected method."""
        try:
            method = self.cluster_method.get()
            
            # Hide all parameter frames initially
            self.kmeans_frame.pack_forget()
            self.dbscan_frame.pack_forget()
            self.hierarchical_frame.pack_forget()
            
            # Show the relevant frame based on the selected method
            if method == "kmeans":
                self.kmeans_frame.pack(fill=tk.X, padx=5, pady=5)
            elif method == "dbscan":
                self.dbscan_frame.pack(fill=tk.X, padx=5, pady=5)
            elif method == "hierarchical":
                self.hierarchical_frame.pack(fill=tk.X, padx=5, pady=5)
            
            self.logger.info(f"Updated cluster options for method: {method}")
            
        except Exception as e:
            self.logger.error(f"Failed to update cluster options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update cluster options: {str(e)}")

    def create_distribution_frame(self):
        """Create frame for distribution visualization options"""
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
                # options will be populated when data is loaded
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
            
            # Additional options frame
            options_frame = ttk.Frame(self.dist_frame)
            options_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Bin count for histogram
            self.bin_count = tk.IntVar(value=30)
            ttk.Label(options_frame, text="Bins:").pack(side=tk.LEFT)
            ttk.Entry(
                options_frame,
                textvariable=self.bin_count,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # KDE bandwidth
            self.kde_bandwidth = tk.DoubleVar(value=0.2)
            ttk.Label(options_frame, text="Bandwidth:").pack(side=tk.LEFT)
            ttk.Entry(
                options_frame,
                textvariable=self.kde_bandwidth,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            self.logger.info("Distribution frame created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create distribution frame: {str(e)}")
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
            
            # Create all visualization frames
            self.create_distribution_frame()
            self.create_profile_frame()
            self.create_reduction_frame()
            self.create_importance_frame()
            
            # Initially show distribution frame
            self.dist_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Plot frame
            plot_container = ttk.LabelFrame(self.viz_tab, text="Visualization")
            plot_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Toolbar frame
            self.toolbar_frame = ttk.Frame(plot_container)
            self.toolbar_frame.pack(fill=tk.X)
            
            # Plot frame
            self.plot_frame = ttk.Frame(plot_container)
            self.plot_frame.pack(fill=tk.BOTH, expand=True)
            
            # Initialize matplotlib figure and canvas
            self.fig = Figure(figsize=(8, 6), dpi=100)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add matplotlib toolbar
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
            self.toolbar.update()
            
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

    def update_feature_menu(self):
        """Update feature menu options when data is loaded"""
        try:
            if self.normalized_data is not None:
                # Get list of numeric columns
                features = list(self.normalized_data.select_dtypes(
                    include=['float64', 'int64']).columns)
                
                # Update feature menu
                menu = self.feature_menu["menu"]
                menu.delete(0, "end")
                
                for feature in features:
                    menu.add_command(
                        label=feature,
                        command=lambda f=feature: self.dist_feature.set(f)
                    )
                
                # Set default feature if none selected
                if not self.dist_feature.get() and features:
                    self.dist_feature.set(features[0])
                    
                self.logger.info("Feature menu updated successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to update feature menu: {str(e)}")
            messagebox.showerror("Error", f"Failed to update feature menu: {str(e)}")

    def generate_plot(self):
        """Generate visualization based on selected plot type and options"""
        try:
            # Clear previous plot
            self.fig.clear()
            
            # Get selected plot type
            plot_type = self.plot_type.get()
            
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
            
            self.logger.info(f"Generated {plot_type} plot successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to generate plot: {str(e)}")
            messagebox.showerror("Error", f"Failed to generate plot: {str(e)}")

    def _generate_distribution_plot(self):
        """Generate distribution plot for selected feature"""
        feature = self.dist_feature.get()
        plot_type = self.dist_type.get()
        
        if not feature:
            raise ValueError("No feature selected")
            
        data = self.data[feature]
        labels = self.cluster_labels
        
        ax = self.fig.add_subplot(111)
        
        if plot_type == "histogram":
            for label in np.unique(labels):
                ax.hist(data[labels == label], alpha=0.5, 
                       label=f'Cluster {label}', bins=30)
        elif plot_type == "kde":
            for label in np.unique(labels):
                sns.kdeplot(data[labels == label], ax=ax,
                          label=f'Cluster {label}')
        else:  # box plot
            ax.boxplot([data[labels == label] 
                       for label in np.unique(labels)],
                      labels=[f'Cluster {label}' 
                             for label in np.unique(labels)])
            
        ax.set_title(f'{feature} Distribution by Cluster')
        ax.set_xlabel(feature)
        ax.legend()
        
        self.plot_data = pd.DataFrame({
            'Feature': data,
            'Cluster': labels
        })

    def _generate_profile_plot(self):
        """Generate cluster profile visualization"""
        plot_type = self.profile_type.get()
        features = [self.feature_listbox.get(i) 
                   for i in self.feature_listbox.curselection()]
        
        if not features:
            raise ValueError("No features selected")
            
        data = self.data[features]
        labels = self.cluster_labels
        
        if plot_type == "heatmap":
            cluster_means = pd.DataFrame([
                data[labels == label].mean() 
                for label in np.unique(labels)
            ], index=[f'Cluster {label}' for label in np.unique(labels)])
            
            ax = self.fig.add_subplot(111)
            sns.heatmap(cluster_means, cmap='coolwarm', center=0,
                       annot=True, ax=ax)
            ax.set_title('Cluster Profiles Heatmap')
            
        elif plot_type == "parallel":
            ax = self.fig.add_subplot(111)
            pd.plotting.parallel_coordinates(
                pd.concat([data, pd.Series(labels, name='Cluster')], axis=1),
                'Cluster', ax=ax)
            ax.set_title('Parallel Coordinates Plot')
            
        else:  # radar chart
            angles = np.linspace(0, 2*np.pi, len(features), endpoint=False)
            cluster_means = np.array([
                data[labels == label].mean() 
                for label in np.unique(labels)
            ])
            
            ax = self.fig.add_subplot(111, projection='polar')
            for i, means in enumerate(cluster_means):
                values = np.concatenate((means, [means[0]]))
                angles_plot = np.concatenate((angles, [angles[0]]))
                ax.plot(angles_plot, values, label=f'Cluster {i}')
                
            ax.set_xticks(angles)
            ax.set_xticklabels(features)
            ax.set_title('Radar Chart of Cluster Profiles')
            ax.legend()
            
        self.plot_data = pd.DataFrame({
            'Feature': np.repeat(features, len(np.unique(labels))),
            'Cluster': np.tile(np.unique(labels), len(features)),
            'Value': cluster_means.flatten()
        })

    def _generate_reduction_plot(self):
        """Generate dimensionality reduction visualization"""
        method = self.reduction_method.get()
        n_components = self.n_components.get()
        
        if method == "pca":
            reducer = PCA(n_components=n_components)
        elif method == "tsne":
            reducer = TSNE(n_components=n_components)
        else:  # umap
            reducer = umap.UMAP(n_components=n_components)
            
        reduced_data = reducer.fit_transform(self.data)
        
        if n_components == 2:
            ax = self.fig.add_subplot(111)
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1],
                               c=self.cluster_labels, cmap='viridis')
            ax.set_title(f'{method.upper()} Projection')
            self.fig.colorbar(scatter, label='Cluster')
            
        elif n_components == 3:
            ax = self.fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1],
                               reduced_data[:, 2], c=self.cluster_labels,
                               cmap='viridis')
            ax.set_title(f'{method.upper()} Projection')
            self.fig.colorbar(scatter, label='Cluster')
            
        self.plot_data = pd.DataFrame(
            reduced_data,
            columns=[f'Component {i+1}' for i in range(n_components)])
        self.plot_data['Cluster'] = self.cluster_labels

    def _generate_importance_plot(self):
        """Generate feature importance visualization"""
        method = self.importance_method.get()
        n_features = self.n_top_features.get()
        
        if method == "statistical":
            importances = []
            pvalues = []
            for col in self.data.columns:
                f_stat, p_val = f_oneway(*[
                    self.data[col][self.cluster_labels == label]
                    for label in np.unique(self.cluster_labels)
                ])
                importances.append(f_stat)
                pvalues.append(p_val)
                
            importance_df = pd.DataFrame({
                'Feature': self.data.columns,
                'Importance': importances,
                'p-value': pvalues
            })
            
        else:  # random forest
            rf = RandomForestClassifier(n_estimators=100)
            rf.fit(self.data, self.cluster_labels)
            
            importance_df = pd.DataFrame({
                'Feature': self.data.columns,
                'Importance': rf.feature_importances_
            })
            
        importance_df = importance_df.nlargest(n_features, 'Importance')
        
        ax = self.fig.add_subplot(111)
        sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax)
        ax.set_title('Feature Importance')
        
        self.plot_data = importance_df

    def save_plot(self):
        """Save current plot to file"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("SVG files", "*.svg")
                ]
            )
            
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
            
            # Components
            comp_frame = ttk.Frame(self.reduction_frame)
            comp_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(comp_frame, text="Number of components:").pack(side=tk.LEFT)
            self.n_components = tk.IntVar(value=2)
            ttk.Entry(
                comp_frame,
                textvariable=self.n_components,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Method-specific parameters frame
            self.reduction_params_frame = ttk.Frame(self.reduction_frame)
            self.reduction_params_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Initialize with PCA parameters
            self.create_pca_params()
            
            self.logger.info("Reduction frame created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create reduction frame: {str(e)}")
            raise

    def update_reduction_options(self, *args):
        """Update reduction options based on selected method"""
        try:
            method = self.reduction_method.get()
            
            # Clear existing parameters
            for widget in self.reduction_params_frame.winfo_children():
                widget.destroy()
            
            # Show method-specific parameters
            if method == "pca":
                self.create_pca_params()
            elif method == "tsne":
                self.create_tsne_params()
            elif method == "umap":
                self.create_umap_params()
            
            self.logger.info(f"Updated reduction options for method: {method}")
            
        except Exception as e:
            self.logger.error(f"Failed to update reduction options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update reduction options: {str(e)}")

    def create_pca_params(self):
        """Create parameter inputs for PCA"""
        try:
            # Explained variance ratio threshold
            var_frame = ttk.Frame(self.reduction_params_frame)
            var_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(var_frame, text="Min explained variance:").pack(side=tk.LEFT)
            self.min_variance = tk.DoubleVar(value=0.95)
            ttk.Entry(
                var_frame,
                textvariable=self.min_variance,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Whitening option
            self.whiten = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                self.reduction_params_frame,
                text="Apply whitening",
                variable=self.whiten
            ).pack(anchor=tk.W)
            
        except Exception as e:
            self.logger.error(f"Failed to create PCA parameters: {str(e)}")
            raise

    def create_tsne_params(self):
        """Create parameter inputs for t-SNE"""
        try:
            # Perplexity
            perp_frame = ttk.Frame(self.reduction_params_frame)
            perp_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(perp_frame, text="Perplexity:").pack(side=tk.LEFT)
            self.perplexity = tk.DoubleVar(value=30.0)
            ttk.Entry(
                perp_frame,
                textvariable=self.perplexity,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Learning rate
            lr_frame = ttk.Frame(self.reduction_params_frame)
            lr_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(lr_frame, text="Learning rate:").pack(side=tk.LEFT)
            self.learning_rate = tk.DoubleVar(value=200.0)
            ttk.Entry(
                lr_frame,
                textvariable=self.learning_rate,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Number of iterations
            iter_frame = ttk.Frame(self.reduction_params_frame)
            iter_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(iter_frame, text="Iterations:").pack(side=tk.LEFT)
            self.n_iter_tsne = tk.IntVar(value=1000)
            ttk.Entry(
                iter_frame,
                textvariable=self.n_iter_tsne,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            self.logger.error(f"Failed to create t-SNE parameters: {str(e)}")
            raise

    def create_umap_params(self):
        """Create parameter inputs for UMAP"""
        try:
            # Number of neighbors
            neighbors_frame = ttk.Frame(self.reduction_params_frame)
            neighbors_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(neighbors_frame, text="Number of neighbors:").pack(side=tk.LEFT)
            self.n_neighbors = tk.IntVar(value=15)
            ttk.Entry(
                neighbors_frame,
                textvariable=self.n_neighbors,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Minimum distance
            dist_frame = ttk.Frame(self.reduction_params_frame)
            dist_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(dist_frame, text="Minimum distance:").pack(side=tk.LEFT)
            self.min_dist = tk.DoubleVar(value=0.1)
            ttk.Entry(
                dist_frame,
                textvariable=self.min_dist,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Metric
            metric_frame = ttk.Frame(self.reduction_params_frame)
            metric_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(metric_frame, text="Distance metric:").pack(side=tk.LEFT)
            self.metric = tk.StringVar(value="euclidean")
            ttk.OptionMenu(
                metric_frame,
                self.metric,
                "euclidean",
                "euclidean", "manhattan", "cosine"
            ).pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            self.logger.error(f"Failed to create UMAP parameters: {str(e)}")
            raise

    def create_importance_frame(self):
        """Create frame for feature importance visualization options"""
        try:
            self.importance_frame = ttk.LabelFrame(self.viz_tab, text="Feature Importance Options")
            
            # Method selection
            method_frame = ttk.Frame(self.importance_frame)
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

    def export_plot_data(self):
        """Export data used to generate current plot"""
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
                width=5,
                validate='key',
                validatecommand=(self.register(self.validate_alpha), '%P')
            ).pack(side=tk.LEFT, padx=5)
            
            # Multiple testing correction
            correction_frame = ttk.Frame(self.stat_frame)
            correction_frame.pack(fill=tk.X, padx=5, pady=2)
            
            self.correction = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                correction_frame,
                text="Apply multiple testing correction",
                variable=self.correction,
                command=self.update_correction_options
            ).pack(side=tk.LEFT)
            
            self.correction_method = tk.StringVar(value="bonferroni")
            self.correction_menu = ttk.OptionMenu(
                correction_frame,
                self.correction_method,
                "bonferroni",
                "bonferroni", "fdr", "holm"
            )
            self.correction_menu.pack(side=tk.LEFT, padx=5)
            
            # Feature selection
            feature_frame = ttk.LabelFrame(self.analysis_tab, text="Feature Selection")
            feature_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Feature listbox with scrollbar
            listbox_frame = ttk.Frame(feature_frame)
            listbox_frame.pack(fill=tk.X, padx=5, pady=2)
            
            self.feature_listbox = tk.Listbox(
                listbox_frame,
                selectmode=tk.MULTIPLE,
                height=6
            )
            self.feature_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            feature_scroll = ttk.Scrollbar(
                listbox_frame,
                orient=tk.VERTICAL,
                command=self.feature_listbox.yview
            )
            feature_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            self.feature_listbox.configure(yscrollcommand=feature_scroll.set)
            
            # Results frame
            results_frame = ttk.LabelFrame(self.analysis_tab, text="Analysis Results")
            results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Results text with scrollbars
            text_frame = ttk.Frame(results_frame)
            text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            self.analysis_text = tk.Text(
                text_frame,
                wrap=tk.WORD,
                height=10
            )
            self.analysis_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            text_scroll = ttk.Scrollbar(
                text_frame,
                orient=tk.VERTICAL,
                command=self.analysis_text.yview
            )
            text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            self.analysis_text.configure(yscrollcommand=text_scroll.set)
            
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
                text="Clear Results",
                command=lambda: self.analysis_text.delete(1.0, tk.END)
            ).pack(side=tk.LEFT, padx=5)
            
            # Initialize analysis variables
            self.analysis_results = None
            
            self.logger.info("Analysis tab created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create analysis tab: {str(e)}")
            raise

    def validate_alpha(self, value):
        """Validate that alpha value is between 0 and 1"""
        if value == "":
            return True
        try:
            alpha = float(value)
            return 0 < alpha < 1
        except ValueError:
            return False

    def update_stat_options(self, *args):
        """Update statistical analysis options based on selected test"""
        try:
            test = self.stat_test.get()
            
            # Enable/disable correction options based on test type
            if test == "chi2":
                self.correction.set(False)
                self.correction_menu.configure(state='disabled')
            else:
                self.correction_menu.configure(state='normal')
                
            self.logger.info(f"Updated statistical options for test: {test}")
            
        except Exception as e:
            self.logger.error(f"Failed to update statistical options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update options: {str(e)}")

    def update_correction_options(self):
        """Enable/disable correction method based on correction checkbox"""
        try:
            if self.correction.get():
                self.correction_menu.configure(state='normal')
            else:
                self.correction_menu.configure(state='disabled')
                
            self.logger.info(f"Updated correction options: {self.correction.get()}")
            
        except Exception as e:
            self.logger.error(f"Failed to update correction options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update options: {str(e)}")

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
            self.cluster_method = tk.StringVar(value="kmeans")
            self.cluster_method_menu = ttk.OptionMenu(
                method_frame,
                self.cluster_method,
                "kmeans",
                "kmeans", "dbscan", "hierarchical",
                command=self.update_cluster_options
            )
            self.cluster_method_menu.pack(side=tk.LEFT, padx=5)
            
            # Initially show kmeans frame
            self.kmeans_frame.pack(fill=tk.X, padx=5, pady=5)
            
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

    def save_clustering_results(self):
        """Save clustering results to file"""
        try:
            if not hasattr(self, 'normalized_data') or self.normalized_data is None:
                raise ValueError("No clustering results available to save")
                
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if file_path:
                # Save results
                results_df = self.normalized_data.copy()
                results_df['Cluster'] = self.cluster_labels
                results_df.to_csv(file_path, index=False)
                
                self.status_var.set(f"Results saved to {file_path}")
                self.logger.info(f"Clustering results saved to {file_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to save clustering results: {str(e)}")
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")

    def update_cluster_options(self, *args):
        """Update clustering options based on selected method."""
        try:
            method = self.cluster_method.get()
            
            # Hide all parameter frames initially
            self.kmeans_frame.pack_forget()
            self.dbscan_frame.pack_forget()
            self.hierarchical_frame.pack_forget()
            
            # Show the relevant frame based on the selected method
            if method == "kmeans":
                self.kmeans_frame.pack(fill=tk.X, padx=5, pady=5)
            elif method == "dbscan":
                self.dbscan_frame.pack(fill=tk.X, padx=5, pady=5)
            elif method == "hierarchical":
                self.hierarchical_frame.pack(fill=tk.X, padx=5, pady=5)
            
            self.logger.info(f"Updated cluster options for method: {method}")
            
        except Exception as e:
            self.logger.error(f"Failed to update cluster options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update cluster options: {str(e)}")

    def create_distribution_frame(self):
        """Create frame for distribution visualization options"""
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
                # options will be populated when data is loaded
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
            
            # Additional options frame
            options_frame = ttk.Frame(self.dist_frame)
            options_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Bin count for histogram
            self.bin_count = tk.IntVar(value=30)
            ttk.Label(options_frame, text="Bins:").pack(side=tk.LEFT)
            ttk.Entry(
                options_frame,
                textvariable=self.bin_count,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # KDE bandwidth
            self.kde_bandwidth = tk.DoubleVar(value=0.2)
            ttk.Label(options_frame, text="Bandwidth:").pack(side=tk.LEFT)
            ttk.Entry(
                options_frame,
                textvariable=self.kde_bandwidth,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            self.logger.info("Distribution frame created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create distribution frame: {str(e)}")
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
            
            # Create all visualization frames
            self.create_distribution_frame()
            self.create_profile_frame()
            self.create_reduction_frame()
            self.create_importance_frame()
            
            # Initially show distribution frame
            self.dist_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Plot frame
            plot_container = ttk.LabelFrame(self.viz_tab, text="Visualization")
            plot_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Toolbar frame
            self.toolbar_frame = ttk.Frame(plot_container)
            self.toolbar_frame.pack(fill=tk.X)
            
            # Plot frame
            self.plot_frame = ttk.Frame(plot_container)
            self.plot_frame.pack(fill=tk.BOTH, expand=True)
            
            # Initialize matplotlib figure and canvas
            self.fig = Figure(figsize=(8, 6), dpi=100)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add matplotlib toolbar
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
            self.toolbar.update()
            
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

    def update_feature_menu(self):
        """Update feature menu options when data is loaded"""
        try:
            if self.normalized_data is not None:
                # Get list of numeric columns
                features = list(self.normalized_data.select_dtypes(
                    include=['float64', 'int64']).columns)
                
                # Update feature menu
                menu = self.feature_menu["menu"]
                menu.delete(0, "end")
                
                for feature in features:
                    menu.add_command(
                        label=feature,
                        command=lambda f=feature: self.dist_feature.set(f)
                    )
                
                # Set default feature if none selected
                if not self.dist_feature.get() and features:
                    self.dist_feature.set(features[0])
                    
                self.logger.info("Feature menu updated successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to update feature menu: {str(e)}")
            messagebox.showerror("Error", f"Failed to update feature menu: {str(e)}")

    def generate_plot(self):
        """Generate visualization based on selected plot type and options"""
        try:
            # Clear previous plot
            self.fig.clear()
            
            # Get selected plot type
            plot_type = self.plot_type.get()
            
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
            
            self.logger.info(f"Generated {plot_type} plot successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to generate plot: {str(e)}")
            messagebox.showerror("Error", f"Failed to generate plot: {str(e)}")

    def _generate_distribution_plot(self):
        """Generate distribution plot for selected feature"""
        feature = self.dist_feature.get()
        plot_type = self.dist_type.get()
        
        if not feature:
            raise ValueError("No feature selected")
            
        data = self.data[feature]
        labels = self.cluster_labels
        
        ax = self.fig.add_subplot(111)
        
        if plot_type == "histogram":
            for label in np.unique(labels):
                ax.hist(data[labels == label], alpha=0.5, 
                       label=f'Cluster {label}', bins=30)
        elif plot_type == "kde":
            for label in np.unique(labels):
                sns.kdeplot(data[labels == label], ax=ax,
                          label=f'Cluster {label}')
        else:  # box plot
            ax.boxplot([data[labels == label] 
                       for label in np.unique(labels)],
                      labels=[f'Cluster {label}' 
                             for label in np.unique(labels)])
            
        ax.set_title(f'{feature} Distribution by Cluster')
        ax.set_xlabel(feature)
        ax.legend()
        
        self.plot_data = pd.DataFrame({
            'Feature': data,
            'Cluster': labels
        })

    def _generate_profile_plot(self):
        """Generate cluster profile visualization"""
        plot_type = self.profile_type.get()
        features = [self.feature_listbox.get(i) 
                   for i in self.feature_listbox.curselection()]
        
        if not features:
            raise ValueError("No features selected")
            
        data = self.data[features]
        labels = self.cluster_labels
        
        if plot_type == "heatmap":
            cluster_means = pd.DataFrame([
                data[labels == label].mean() 
                for label in np.unique(labels)
            ], index=[f'Cluster {label}' for label in np.unique(labels)])
            
            ax = self.fig.add_subplot(111)
            sns.heatmap(cluster_means, cmap='coolwarm', center=0,
                       annot=True, ax=ax)
            ax.set_title('Cluster Profiles Heatmap')
            
        elif plot_type == "parallel":
            ax = self.fig.add_subplot(111)
            pd.plotting.parallel_coordinates(
                pd.concat([data, pd.Series(labels, name='Cluster')], axis=1),
                'Cluster', ax=ax)
            ax.set_title('Parallel Coordinates Plot')
            
        else:  # radar chart
            angles = np.linspace(0, 2*np.pi, len(features), endpoint=False)
            cluster_means = np.array([
                data[labels == label].mean() 
                for label in np.unique(labels)
            ])
            
            ax = self.fig.add_subplot(111, projection='polar')
            for i, means in enumerate(cluster_means):
                values = np.concatenate((means, [means[0]]))
                angles_plot = np.concatenate((angles, [angles[0]]))
                ax.plot(angles_plot, values, label=f'Cluster {i}')
                
            ax.set_xticks(angles)
            ax.set_xticklabels(features)
            ax.set_title('Radar Chart of Cluster Profiles')
            ax.legend()
            
        self.plot_data = pd.DataFrame({
            'Feature': np.repeat(features, len(np.unique(labels))),
            'Cluster': np.tile(np.unique(labels), len(features)),
            'Value': cluster_means.flatten()
        })

    def _generate_reduction_plot(self):
        """Generate dimensionality reduction visualization"""
        method = self.reduction_method.get()
        n_components = self.n_components.get()
        
        if method == "pca":
            reducer = PCA(n_components=n_components)
        elif method == "tsne":
            reducer = TSNE(n_components=n_components)
        else:  # umap
            reducer = umap.UMAP(n_components=n_components)
            
        reduced_data = reducer.fit_transform(self.data)
        
        if n_components == 2:
            ax = self.fig.add_subplot(111)
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1],
                               c=self.cluster_labels, cmap='viridis')
            ax.set_title(f'{method.upper()} Projection')
            self.fig.colorbar(scatter, label='Cluster')
            
        elif n_components == 3:
            ax = self.fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1],
                               reduced_data[:, 2], c=self.cluster_labels,
                               cmap='viridis')
            ax.set_title(f'{method.upper()} Projection')
            self.fig.colorbar(scatter, label='Cluster')
            
        self.plot_data = pd.DataFrame(
            reduced_data,
            columns=[f'Component {i+1}' for i in range(n_components)])
        self.plot_data['Cluster'] = self.cluster_labels

    def _generate_importance_plot(self):
        """Generate feature importance visualization"""
        method = self.importance_method.get()
        n_features = self.n_top_features.get()
        
        if method == "statistical":
            importances = []
            pvalues = []
            for col in self.data.columns:
                f_stat, p_val = f_oneway(*[
                    self.data[col][self.cluster_labels == label]
                    for label in np.unique(self.cluster_labels)
                ])
                importances.append(f_stat)
                pvalues.append(p_val)
                
            importance_df = pd.DataFrame({
                'Feature': self.data.columns,
                'Importance': importances,
                'p-value': pvalues
            })
            
        else:  # random forest
            rf = RandomForestClassifier(n_estimators=100)
            rf.fit(self.data, self.cluster_labels)
            
            importance_df = pd.DataFrame({
                'Feature': self.data.columns,
                'Importance': rf.feature_importances_
            })
            
        importance_df = importance_df.nlargest(n_features, 'Importance')
        
        ax = self.fig.add_subplot(111)
        sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax)
        ax.set_title('Feature Importance')
        
        self.plot_data = importance_df

    def save_plot(self):
        """Save current plot to file"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("SVG files", "*.svg")
                ]
            )
            
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
            
            # Components
            comp_frame = ttk.Frame(self.reduction_frame)
            comp_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(comp_frame, text="Number of components:").pack(side=tk.LEFT)
            self.n_components = tk.IntVar(value=2)
            ttk.Entry(
                comp_frame,
                textvariable=self.n_components,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Method-specific parameters frame
            self.reduction_params_frame = ttk.Frame(self.reduction_frame)
            self.reduction_params_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Initialize with PCA parameters
            self.create_pca_params()
            
            self.logger.info("Reduction frame created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create reduction frame: {str(e)}")
            raise

    def update_reduction_options(self, *args):
        """Update reduction options based on selected method"""
        try:
            method = self.reduction_method.get()
            
            # Clear existing parameters
            for widget in self.reduction_params_frame.winfo_children():
                widget.destroy()
            
            # Show method-specific parameters
            if method == "pca":
                self.create_pca_params()
            elif method == "tsne":
                self.create_tsne_params()
            elif method == "umap":
                self.create_umap_params()
            
            self.logger.info(f"Updated reduction options for method: {method}")
            
        except Exception as e:
            self.logger.error(f"Failed to update reduction options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update reduction options: {str(e)}")

    def create_pca_params(self):
        """Create parameter inputs for PCA"""
        try:
            # Explained variance ratio threshold
            var_frame = ttk.Frame(self.reduction_params_frame)
            var_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(var_frame, text="Min explained variance:").pack(side=tk.LEFT)
            self.min_variance = tk.DoubleVar(value=0.95)
            ttk.Entry(
                var_frame,
                textvariable=self.min_variance,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Whitening option
            self.whiten = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                self.reduction_params_frame,
                text="Apply whitening",
                variable=self.whiten
            ).pack(anchor=tk.W)
            
        except Exception as e:
            self.logger.error(f"Failed to create PCA parameters: {str(e)}")
            raise

    def create_tsne_params(self):
        """Create parameter inputs for t-SNE"""
        try:
            # Perplexity
            perp_frame = ttk.Frame(self.reduction_params_frame)
            perp_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(perp_frame, text="Perplexity:").pack(side=tk.LEFT)
            self.perplexity = tk.DoubleVar(value=30.0)
            ttk.Entry(
                perp_frame,
                textvariable=self.perplexity,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Learning rate
            lr_frame = ttk.Frame(self.reduction_params_frame)
            lr_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(lr_frame, text="Learning rate:").pack(side=tk.LEFT)
            self.learning_rate = tk.DoubleVar(value=200.0)
            ttk.Entry(
                lr_frame,
                textvariable=self.learning_rate,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Number of iterations
            iter_frame = ttk.Frame(self.reduction_params_frame)
            iter_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(iter_frame, text="Iterations:").pack(side=tk.LEFT)
            self.n_iter_tsne = tk.IntVar(value=1000)
            ttk.Entry(
                iter_frame,
                textvariable=self.n_iter_tsne,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            self.logger.error(f"Failed to create t-SNE parameters: {str(e)}")
            raise

    def create_umap_params(self):
        """Create parameter inputs for UMAP"""
        try:
            # Number of neighbors
            neighbors_frame = ttk.Frame(self.reduction_params_frame)
            neighbors_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(neighbors_frame, text="Number of neighbors:").pack(side=tk.LEFT)
            self.n_neighbors = tk.IntVar(value=15)
            ttk.Entry(
                neighbors_frame,
                textvariable=self.n_neighbors,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Minimum distance
            dist_frame = ttk.Frame(self.reduction_params_frame)
            dist_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(dist_frame, text="Minimum distance:").pack(side=tk.LEFT)
            self.min_dist = tk.DoubleVar(value=0.1)
            ttk.Entry(
                dist_frame,
                textvariable=self.min_dist,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Metric
            metric_frame = ttk.Frame(self.reduction_params_frame)
            metric_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(metric_frame, text="Distance metric:").pack(side=tk.LEFT)
            self.metric = tk.StringVar(value="euclidean")
            ttk.OptionMenu(
                metric_frame,
                self.metric,
                "euclidean",
                "euclidean", "manhattan", "cosine"
            ).pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            self.logger.error(f"Failed to create UMAP parameters: {str(e)}")
            raise

    def create_importance_frame(self):
        """Create frame for feature importance visualization options"""
        try:
            self.importance_frame = ttk.LabelFrame(self.viz_tab, text="Feature Importance Options")
            
            # Method selection
            method_frame = ttk.Frame(self.importance_frame)
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

    def export_plot_data(self):
        """Export data used to generate current plot"""
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
                width=5,
                validate='key',
                validatecommand=(self.register(self.validate_alpha), '%P')
            ).pack(side=tk.LEFT, padx=5)
            
            # Multiple testing correction
            correction_frame = ttk.Frame(self.stat_frame)
            correction_frame.pack(fill=tk.X, padx=5, pady=2)
            
            self.correction = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                correction_frame,
                text="Apply multiple testing correction",
                variable=self.correction,
                command=self.update_correction_options
            ).pack(side=tk.LEFT)
            
            self.correction_method = tk.StringVar(value="bonferroni")
            self.correction_menu = ttk.OptionMenu(
                correction_frame,
                self.correction_method,
                "bonferroni",
                "bonferroni", "fdr", "holm"
            )
            self.correction_menu.pack(side=tk.LEFT, padx=5)
            
            # Feature selection
            feature_frame = ttk.LabelFrame(self.analysis_tab, text="Feature Selection")
            feature_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Feature listbox with scrollbar
            listbox_frame = ttk.Frame(feature_frame)
            listbox_frame.pack(fill=tk.X, padx=5, pady=2)
            
            self.feature_listbox = tk.Listbox(
                listbox_frame,
                selectmode=tk.MULTIPLE,
                height=6
            )
            self.feature_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            feature_scroll = ttk.Scrollbar(
                listbox_frame,
                orient=tk.VERTICAL,
                command=self.feature_listbox.yview
            )
            feature_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            self.feature_listbox.configure(yscrollcommand=feature_scroll.set)
            
            # Results frame
            results_frame = ttk.LabelFrame(self.analysis_tab, text="Analysis Results")
            results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Results text with scrollbars
            text_frame = ttk.Frame(results_frame)
            text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            self.analysis_text = tk.Text(
                text_frame,
                wrap=tk.WORD,
                height=10
            )
            self.analysis_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            text_scroll = ttk.Scrollbar(
                text_frame,
                orient=tk.VERTICAL,
                command=self.analysis_text.yview
            )
            text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            self.analysis_text.configure(yscrollcommand=text_scroll.set)
            
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
                text="Clear Results",
                command=lambda: self.analysis_text.delete(1.0, tk.END)
            ).pack(side=tk.LEFT, padx=5)
            
            # Initialize analysis variables
            self.analysis_results = None
            
            self.logger.info("Analysis tab created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create analysis tab: {str(e)}")
            raise

    def validate_alpha(self, value):
        """Validate that alpha value is between 0 and 1"""
        if value == "":
            return True
        try:
            alpha = float(value)
            return 0 < alpha < 1
        except ValueError:
            return False

    def update_stat_options(self, *args):
        """Update statistical analysis options based on selected test"""
        try:
            test = self.stat_test.get()
            
            # Enable/disable correction options based on test type
            if test == "chi2":
                self.correction.set(False)
                self.correction_menu.configure(state='disabled')
            else:
                self.correction_menu.configure(state='normal')
                
            self.logger.info(f"Updated statistical options for test: {test}")
            
        except Exception as e:
            self.logger.error(f"Failed to update statistical options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update options: {str(e)}")

    def update_correction_options(self):
        """Enable/disable correction method based on correction checkbox"""
        try:
            if self.correction.get():
                self.correction_menu.configure(state='normal')
            else:
                self.correction_menu.configure(state='disabled')
                
            self.logger.info(f"Updated correction options: {self.correction.get()}")
            
        except Exception as e:
            self.logger.error(f"Failed to update correction options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update options: {str(e)}")

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
            self.cluster_method = tk.StringVar(value="kmeans")
            self.cluster_method_menu = ttk.OptionMenu(
                method_frame,
                self.cluster_method,
                "kmeans",
                "kmeans", "dbscan", "hierarchical",
                command=self.update_cluster_options
            )
            self.cluster_method_menu.pack(side=tk.LEFT, padx=5)
            
            # Initially show kmeans frame
            self.kmeans_frame.pack(fill=tk.X, padx=5, pady=5)
            
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

    def save_clustering_results(self):
        """Save clustering results to file"""
        try:
            if not hasattr(self, 'normalized_data') or self.normalized_data is None:
                raise ValueError("No clustering results available to save")
                
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if file_path:
                # Save results
                results_df = self.normalized_data.copy()
                results_df['Cluster'] = self.cluster_labels
                results_df.to_csv(file_path, index=False)
                
                self.status_var.set(f"Results saved to {file_path}")
                self.logger.info(f"Clustering results saved to {file_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to save clustering results: {str(e)}")
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")

    def update_cluster_options(self, *args):
        """Update clustering options based on selected method."""
        try:
            method = self.cluster_method.get()
            
            # Hide all parameter frames initially
            self.kmeans_frame.pack_forget()
            self.dbscan_frame.pack_forget()
            self.hierarchical_frame.pack_forget()
            
            # Show the relevant frame based on the selected method
            if method == "kmeans":
                self.kmeans_frame.pack(fill=tk.X, padx=5, pady=5)
            elif method == "dbscan":
                self.dbscan_frame.pack(fill=tk.X, padx=5, pady=5)
            elif method == "hierarchical":
                self.hierarchical_frame.pack(fill=tk.X, padx=5, pady=5)
            
            self.logger.info(f"Updated cluster options for method: {method}")
            
        except Exception as e:
            self.logger.error(f"Failed to update cluster options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update cluster options: {str(e)}")

    def create_distribution_frame(self):
        """Create frame for distribution visualization options"""
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
                # options will be populated when data is loaded
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
            
            # Additional options frame
            options_frame = ttk.Frame(self.dist_frame)
            options_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Bin count for histogram
            self.bin_count = tk.IntVar(value=30)
            ttk.Label(options_frame, text="Bins:").pack(side=tk.LEFT)
            ttk.Entry(
                options_frame,
                textvariable=self.bin_count,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # KDE bandwidth
            self.kde_bandwidth = tk.DoubleVar(value=0.2)
            ttk.Label(options_frame, text="Bandwidth:").pack(side=tk.LEFT)
            ttk.Entry(
                options_frame,
                textvariable=self.kde_bandwidth,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            self.logger.info("Distribution frame created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create distribution frame: {str(e)}")
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
            
            # Create all visualization frames
            self.create_distribution_frame()
            self.create_profile_frame()
            self.create_reduction_frame()
            self.create_importance_frame()
            
            # Initially show distribution frame
            self.dist_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Plot frame
            plot_container = ttk.LabelFrame(self.viz_tab, text="Visualization")
            plot_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Toolbar frame
            self.toolbar_frame = ttk.Frame(plot_container)
            self.toolbar_frame.pack(fill=tk.X)
            
            # Plot frame
            self.plot_frame = ttk.Frame(plot_container)
            self.plot_frame.pack(fill=tk.BOTH, expand=True)
            
            # Initialize matplotlib figure and canvas
            self.fig = Figure(figsize=(8, 6), dpi=100)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add matplotlib toolbar
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
            self.toolbar.update()
            
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

    def update_feature_menu(self):
        """Update feature menu options when data is loaded"""
        try:
            if self.normalized_data is not None:
                # Get list of numeric columns
                features = list(self.normalized_data.select_dtypes(
                    include=['float64', 'int64']).columns)
                
                # Update feature menu
                menu = self.feature_menu["menu"]
                menu.delete(0, "end")
                
                for feature in features:
                    menu.add_command(
                        label=feature,
                        command=lambda f=feature: self.dist_feature.set(f)
                    )
                
                # Set default feature if none selected
                if not self.dist_feature.get() and features:
                    self.dist_feature.set(features[0])
                    
                self.logger.info("Feature menu updated successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to update feature menu: {str(e)}")
            messagebox.showerror("Error", f"Failed to update feature menu: {str(e)}")

    def generate_plot(self):
        """Generate visualization based on selected plot type and options"""
        try:
            # Clear previous plot
            self.fig.clear()
            
            # Get selected plot type
            plot_type = self.plot_type.get()
            
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
            
            self.logger.info(f"Generated {plot_type} plot successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to generate plot: {str(e)}")
            messagebox.showerror("Error", f"Failed to generate plot: {str(e)}")

    def _generate_distribution_plot(self):
        """Generate distribution plot for selected feature"""
        feature = self.dist_feature.get()
        plot_type = self.dist_type.get()
        
        if not feature:
            raise ValueError("No feature selected")
            
        data = self.data[feature]
        labels = self.cluster_labels
        
        ax = self.fig.add_subplot(111)
        
        if plot_type == "histogram":
            for label in np.unique(labels):
                ax.hist(data[labels == label], alpha=0.5, 
                       label=f'Cluster {label}', bins=30)
        elif plot_type == "kde":
            for label in np.unique(labels):
                sns.kdeplot(data[labels == label], ax=ax,
                          label=f'Cluster {label}')
        else:  # box plot
            ax.boxplot([data[labels == label] 
                       for label in np.unique(labels)],
                      labels=[f'Cluster {label}' 
                             for label in np.unique(labels)])
            
        ax.set_title(f'{feature} Distribution by Cluster')
        ax.set_xlabel(feature)
        ax.legend()
        
        self.plot_data = pd.DataFrame({
            'Feature': data,
            'Cluster': labels
        })

    def _generate_profile_plot(self):
        """Generate cluster profile visualization"""
        plot_type = self.profile_type.get()
        features = [self.feature_listbox.get(i) 
                   for i in self.feature_listbox.curselection()]
        
        if not features:
            raise ValueError("No features selected")
            
        data = self.data[features]
        labels = self.cluster_labels
        
        if plot_type == "heatmap":
            cluster_means = pd.DataFrame([
                data[labels == label].mean() 
                for label in np.unique(labels)
            ], index=[f'Cluster {label}' for label in np.unique(labels)])
            
            ax = self.fig.add_subplot(111)
            sns.heatmap(cluster_means, cmap='coolwarm', center=0,
                       annot=True, ax=ax)
            ax.set_title('Cluster Profiles Heatmap')
            
        elif plot_type == "parallel":
            ax = self.fig.add_subplot(111)
            pd.plotting.parallel_coordinates(
                pd.concat([data, pd.Series(labels, name='Cluster')], axis=1),
                'Cluster', ax=ax)
            ax.set_title('Parallel Coordinates Plot')
            
        else:  # radar chart
            angles = np.linspace(0, 2*np.pi, len(features), endpoint=False)
            cluster_means = np.array([
                data[labels == label].mean() 
                for label in np.unique(labels)
            ])
            
            ax = self.fig.add_subplot(111, projection='polar')
            for i, means in enumerate(cluster_means):
                values = np.concatenate((means, [means[0]]))
                angles_plot = np.concatenate((angles, [angles[0]]))
                ax.plot(angles_plot, values, label=f'Cluster {i}')
                
            ax.set_xticks(angles)
            ax.set_xticklabels(features)
            ax.set_title('Radar Chart of Cluster Profiles')
            ax.legend()
            
        self.plot_data = pd.DataFrame({
            'Feature': np.repeat(features, len(np.unique(labels))),
            'Cluster': np.tile(np.unique(labels), len(features)),
            'Value': cluster_means.flatten()
        })

    def _generate_reduction_plot(self):
        """Generate dimensionality reduction visualization"""
        method = self.reduction_method.get()
        n_components = self.n_components.get()
        
        if method == "pca":
            reducer = PCA(n_components=n_components)
        elif method == "tsne":
            reducer = TSNE(n_components=n_components)
        else:  # umap
            reducer = umap.UMAP(n_components=n_components)
            
        reduced_data = reducer.fit_transform(self.data)
        
        if n_components == 2:
            ax = self.fig.add_subplot(111)
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1],
                               c=self.cluster_labels, cmap='viridis')
            ax.set_title(f'{method.upper()} Projection')
            self.fig.colorbar(scatter, label='Cluster')
            
        elif n_components == 3:
            ax = self.fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1],
                               reduced_data[:, 2], c=self.cluster_labels,
                               cmap='viridis')
            ax.set_title(f'{method.upper()} Projection')
            self.fig.colorbar(scatter, label='Cluster')
            
        self.plot_data = pd.DataFrame(
            reduced_data,
            columns=[f'Component {i+1}' for i in range(n_components)])
        self.plot_data['Cluster'] = self.cluster_labels

    def _generate_importance_plot(self):
        """Generate feature importance visualization"""
        method = self.importance_method.get()
        n_features = self.n_top_features.get()
        
        if method == "statistical":
            importances = []
            pvalues = []
            for col in self.data.columns:
                f_stat, p_val = f_oneway(*[
                    self.data[col][self.cluster_labels == label]
                    for label in np.unique(self.cluster_labels)
                ])
                importances.append(f_stat)
                pvalues.append(p_val)
                
            importance_df = pd.DataFrame({
                'Feature': self.data.columns,
                'Importance': importances,
                'p-value': pvalues
            })
            
        else:  # random forest
            rf = RandomForestClassifier(n_estimators=100)
            rf.fit(self.data, self.cluster_labels)
            
            importance_df = pd.DataFrame({
                'Feature': self.data.columns,
                'Importance': rf.feature_importances_
            })
            
        importance_df = importance_df.nlargest(n_features, 'Importance')
        
        ax = self.fig.add_subplot(111)
        sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax)
        ax.set_title('Feature Importance')
        
        self.plot_data = importance_df

    def save_plot(self):
        """Save current plot to file"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("SVG files", "*.svg")
                ]
            )
            
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
            
            # Components
            comp_frame = ttk.Frame(self.reduction_frame)
            comp_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(comp_frame, text="Number of components:").pack(side=tk.LEFT)
            self.n_components = tk.IntVar(value=2)
            ttk.Entry(
                comp_frame,
                textvariable=self.n_components,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Method-specific parameters frame
            self.reduction_params_frame = ttk.Frame(self.reduction_frame)
            self.reduction_params_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Initialize with PCA parameters
            self.create_pca_params()
            
            self.logger.info("Reduction frame created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create reduction frame: {str(e)}")
            raise

    def update_reduction_options(self, *args):
        """Update reduction options based on selected method"""
        try:
            method = self.reduction_method.get()
            
            # Clear existing parameters
            for widget in self.reduction_params_frame.winfo_children():
                widget.destroy()
            
            # Show method-specific parameters
            if method == "pca":
                self.create_pca_params()
            elif method == "tsne":
                self.create_tsne_params()
            elif method == "umap":
                self.create_umap_params()
            
            self.logger.info(f"Updated reduction options for method: {method}")
            
        except Exception as e:
            self.logger.error(f"Failed to update reduction options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update reduction options: {str(e)}")

    def create_pca_params(self):
        """Create parameter inputs for PCA"""
        try:
            # Explained variance ratio threshold
            var_frame = ttk.Frame(self.reduction_params_frame)
            var_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(var_frame, text="Min explained variance:").pack(side=tk.LEFT)
            self.min_variance = tk.DoubleVar(value=0.95)
            ttk.Entry(
                var_frame,
                textvariable=self.min_variance,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Whitening option
            self.whiten = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                self.reduction_params_frame,
                text="Apply whitening",
                variable=self.whiten
            ).pack(anchor=tk.W)
            
        except Exception as e:
            self.logger.error(f"Failed to create PCA parameters: {str(e)}")
            raise

    def create_tsne_params(self):
        """Create parameter inputs for t-SNE"""
        try:
            # Perplexity
            perp_frame = ttk.Frame(self.reduction_params_frame)
            perp_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(perp_frame, text="Perplexity:").pack(side=tk.LEFT)
            self.perplexity = tk.DoubleVar(value=30.0)
            ttk.Entry(
                perp_frame,
                textvariable=self.perplexity,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Learning rate
            lr_frame = ttk.Frame(self.reduction_params_frame)
            lr_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(lr_frame, text="Learning rate:").pack(side=tk.LEFT)
            self.learning_rate = tk.DoubleVar(value=200.0)
            ttk.Entry(
                lr_frame,
                textvariable=self.learning_rate,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Number of iterations
            iter_frame = ttk.Frame(self.reduction_params_frame)
            iter_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(iter_frame, text="Iterations:").pack(side=tk.LEFT)
            self.n_iter_tsne = tk.IntVar(value=1000)
            ttk.Entry(
                iter_frame,
                textvariable=self.n_iter_tsne,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            self.logger.error(f"Failed to create t-SNE parameters: {str(e)}")
            raise

    def create_umap_params(self):
        """Create parameter inputs for UMAP"""
        try:
            # Number of neighbors
            neighbors_frame = ttk.Frame(self.reduction_params_frame)
            neighbors_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(neighbors_frame, text="Number of neighbors:").pack(side=tk.LEFT)
            self.n_neighbors = tk.IntVar(value=15)
            ttk.Entry(
                neighbors_frame,
                textvariable=self.n_neighbors,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Minimum distance
            dist_frame = ttk.Frame(self.reduction_params_frame)
            dist_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(dist_frame, text="Minimum distance:").pack(side=tk.LEFT)
            self.min_dist = tk.DoubleVar(value=0.1)
            ttk.Entry(
                dist_frame,
                textvariable=self.min_dist,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Metric
            metric_frame = ttk.Frame(self.reduction_params_frame)
            metric_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(metric_frame, text="Distance metric:").pack(side=tk.LEFT)
            self.metric = tk.StringVar(value="euclidean")
            ttk.OptionMenu(
                metric_frame,
                self.metric,
                "euclidean",
                "euclidean", "manhattan", "cosine"
            ).pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            self.logger.error(f"Failed to create UMAP parameters: {str(e)}")
            raise

    def create_importance_frame(self):
                try:
                    pass  # Add your code here
                except Exception as e:
                    self.logger.error(f"An error occurred: {str(e)}")
                    raise
                    
    def create_importance_frame(self):
        """Create frame for feature importance visualization options"""
        try:
            self.importance_frame = ttk.LabelFrame(self.viz_tab, text="Feature Importance Options")
            
            # Method selection
            method_frame = ttk.Frame(self.importance_frame)
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

    def export_plot_data(self):
        """Export data used to generate current plot"""
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
                width=5,
                validate='key',
                validatecommand=(self.register(self.validate_alpha), '%P')
            ).pack(side=tk.LEFT, padx=5)
            
            # Multiple testing correction
            correction_frame = ttk.Frame(self.stat_frame)
            correction_frame.pack(fill=tk.X, padx=5, pady=2)
            
            self.correction = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                correction_frame,
                text="Apply multiple testing correction",
                variable=self.correction,
                command=self.update_correction_options
            ).pack(side=tk.LEFT)
            
            self.correction_method = tk.StringVar(value="bonferroni")
            self.correction_menu = ttk.OptionMenu(
                correction_frame,
                self.correction_method,
                "bonferroni",
                "bonferroni", "fdr", "holm"
            )
            self.correction_menu.pack(side=tk.LEFT, padx=5)
            
            # Feature selection
            feature_frame = ttk.LabelFrame(self.analysis_tab, text="Feature Selection")
            feature_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Feature listbox with scrollbar
            listbox_frame = ttk.Frame(feature_frame)
            listbox_frame.pack(fill=tk.X, padx=5, pady=2)
            
            self.feature_listbox = tk.Listbox(
                listbox_frame,
                selectmode=tk.MULTIPLE,
                height=6
            )
            self.feature_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            feature_scroll = ttk.Scrollbar(
                listbox_frame,
                orient=tk.VERTICAL,
                command=self.feature_listbox.yview
            )
            feature_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            self.feature_listbox.configure(yscrollcommand=feature_scroll.set)
            
            # Results frame
            results_frame = ttk.LabelFrame(self.analysis_tab, text="Analysis Results")
            results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Results text with scrollbars
            text_frame = ttk.Frame(results_frame)
            text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            self.analysis_text = tk.Text(
                text_frame,
                wrap=tk.WORD,
                height=10
            )
            self.analysis_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            text_scroll = ttk.Scrollbar(
                text_frame,
                orient=tk.VERTICAL,
                command=self.analysis_text.yview
            )
            text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            self.analysis_text.configure(yscrollcommand=text_scroll.set)
            
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
                text="Clear Results",
                command=lambda: self.analysis_text.delete(1.0, tk.END)
            ).pack(side=tk.LEFT, padx=5)
            
            # Initialize analysis variables
            self.analysis_results = None
            
            self.logger.info("Analysis tab created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create analysis tab: {str(e)}")
            raise

    def validate_alpha(self, value):
        """Validate that alpha value is between 0 and 1"""
        if value == "":
            return True
        try:
            alpha = float(value)
            return 0 < alpha < 1
        except ValueError:
            return False

    def update_stat_options(self, *args):
        """Update statistical analysis options based on selected test"""
        try:
            test = self.stat_test.get()
            
            # Enable/disable correction options based on test type
            if test == "chi2":
                self.correction.set(False)
                self.correction_menu.configure(state='disabled')
            else:
                self.correction_menu.configure(state='normal')
                
            self.logger.info(f"Updated statistical options for test: {test}")
            
        except Exception as e:
            self.logger.error(f"Failed to update statistical options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update options: {str(e)}")

    def update_correction_options(self):
        """Enable/disable correction method based on correction checkbox"""
        try:
            if self.correction.get():
                self.correction_menu.configure(state='normal')
            else:
                self.correction_menu.configure(state='disabled')
                
            self.logger.info(f"Updated correction options: {self.correction.get()}")
            
        except Exception as e:
            self.logger.error(f"Failed to update correction options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update options: {str(e)}")

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
            self.cluster_method = tk.StringVar(value="kmeans")
            self.cluster_method_menu = ttk.OptionMenu(
                method_frame,
                self.cluster_method,
                "kmeans",
                "kmeans", "dbscan", "hierarchical",
                command=self.update_cluster_options
            )
            self.cluster_method_menu.pack(side=tk.LEFT, padx=5)
            
            # Initially show kmeans frame
            self.kmeans_frame.pack(fill=tk.X, padx=5, pady=5)
            
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

    def save_clustering_results(self):
        """Save clustering results to file"""
        try:
            if not hasattr(self, 'normalized_data') or self.normalized_data is None:
                raise ValueError("No clustering results available to save")
                
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if file_path:
                # Save results
                results_df = self.normalized_data.copy()
                results_df['Cluster'] = self.cluster_labels
                results_df.to_csv(file_path, index=False)
                
                self.status_var.set(f"Results saved to {file_path}")
                self.logger.info(f"Clustering results saved to {file_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to save clustering results: {str(e)}")
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")

    def update_cluster_options(self, *args):
        """Update clustering options based on selected method."""
        try:
            method = self.cluster_method.get()
            
            # Hide all parameter frames initially
            self.kmeans_frame.pack_forget()
            self.dbscan_frame.pack_forget()
            self.hierarchical_frame.pack_forget()
            
            # Show the relevant frame based on the selected method
            if method == "kmeans":
                self.kmeans_frame.pack(fill=tk.X, padx=5, pady=5)
            elif method == "dbscan":
                self.dbscan_frame.pack(fill=tk.X, padx=5, pady=5)
            elif method == "hierarchical":
                self.hierarchical_frame.pack(fill=tk.X, padx=5, pady=5)
            
            self.logger.info(f"Updated cluster options for method: {method}")
            
        except Exception as e:
            self.logger.error(f"Failed to update cluster options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update cluster options: {str(e)}")

    def create_distribution_frame(self):
        """Create frame for distribution visualization options"""
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
                # options will be populated when data is loaded
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
            
            # Additional options frame
            options_frame = ttk.Frame(self.dist_frame)
            options_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Bin count for histogram
            self.bin_count = tk.IntVar(value=30)
            ttk.Label(options_frame, text="Bins:").pack(side=tk.LEFT)
            ttk.Entry(
                options_frame,
                textvariable=self.bin_count,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # KDE bandwidth
            self.kde_bandwidth = tk.DoubleVar(value=0.2)
            ttk.Label(options_frame, text="Bandwidth:").pack(side=tk.LEFT)
            ttk.Entry(
                options_frame,
                textvariable=self.kde_bandwidth,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            self.logger.info("Distribution frame created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create distribution frame: {str(e)}")
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
            
            # Create all visualization frames
            self.create_distribution_frame()
            self.create_profile_frame()
            self.create_reduction_frame()
            self.create_importance_frame()
            
            # Initially show distribution frame
            self.dist_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Plot frame
            plot_container = ttk.LabelFrame(self.viz_tab, text="Visualization")
            plot_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Toolbar frame
            self.toolbar_frame = ttk.Frame(plot_container)
            self.toolbar_frame.pack(fill=tk.X)
            
            # Plot frame
            self.plot_frame = ttk.Frame(plot_container)
            self.plot_frame.pack(fill=tk.BOTH, expand=True)
            
            # Initialize matplotlib figure and canvas
            self.fig = Figure(figsize=(8, 6), dpi=100)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add matplotlib toolbar
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
            self.toolbar.update()
            
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

    def update_feature_menu(self):
        """Update feature menu options when data is loaded"""
        try:
            if self.normalized_data is not None:
                # Get list of numeric columns
                features = list(self.normalized_data.select_dtypes(
                    include=['float64', 'int64']).columns)
                
                # Update feature menu
                menu = self.feature_menu["menu"]
                menu.delete(0, "end")
                
                for feature in features:
                    menu.add_command(
                        label=feature,
                        command=lambda f=feature: self.dist_feature.set(f)
                    )
                
                # Set default feature if none selected
                if not self.dist_feature.get() and features:
                    self.dist_feature.set(features[0])
                    
                self.logger.info("Feature menu updated successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to update feature menu: {str(e)}")
            messagebox.showerror("Error", f"Failed to update feature menu: {str(e)}")

    def generate_plot(self):
        """Generate visualization based on selected plot type and options"""
        try:
            # Clear previous plot
            self.fig.clear()
            
            # Get selected plot type
            plot_type = self.plot_type.get()
            
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
            
            self.logger.info(f"Generated {plot_type} plot successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to generate plot: {str(e)}")
            messagebox.showerror("Error", f"Failed to generate plot: {str(e)}")

    def _generate_distribution_plot(self):
        """Generate distribution plot for selected feature"""
        feature = self.dist_feature.get()
        plot_type = self.dist_type.get()
        
        if not feature:
            raise ValueError("No feature selected")
            
        data = self.data[feature]
        labels = self.cluster_labels
        
        ax = self.fig.add_subplot(111)
        
        if plot_type == "histogram":
            for label in np.unique(labels):
                ax.hist(data[labels == label], alpha=0.5, 
                       label=f'Cluster {label}', bins=30)
        elif plot_type == "kde":
            for label in np.unique(labels):
                sns.kdeplot(data[labels == label], ax=ax,
                          label=f'Cluster {label}')
        else:  # box plot
            ax.boxplot([data[labels == label] 
                       for label in np.unique(labels)],
                      labels=[f'Cluster {label}' 
                             for label in np.unique(labels)])
            
        ax.set_title(f'{feature} Distribution by Cluster')
        ax.set_xlabel(feature)
        ax.legend()
        
        self.plot_data = pd.DataFrame({
            'Feature': data,
            'Cluster': labels
        })

    def _generate_profile_plot(self):
        """Generate cluster profile visualization"""
        plot_type = self.profile_type.get()
        features = [self.feature_listbox.get(i) 
                   for i in self.feature_listbox.curselection()]
        
        if not features:
            raise ValueError("No features selected")
            
        data = self.data[features]
        labels = self.cluster_labels
        
        if plot_type == "heatmap":
            cluster_means = pd.DataFrame([
                data[labels == label].mean() 
                for label in np.unique(labels)
            ], index=[f'Cluster {label}' for label in np.unique(labels)])
            
            ax = self.fig.add_subplot(111)
            sns.heatmap(cluster_means, cmap='coolwarm', center=0,
                       annot=True, ax=ax)
            ax.set_title('Cluster Profiles Heatmap')
            
        elif plot_type == "parallel":
            ax = self.fig.add_subplot(111)
            pd.plotting.parallel_coordinates(
                pd.concat([data, pd.Series(labels, name='Cluster')], axis=1),
                'Cluster', ax=ax)
            ax.set_title('Parallel Coordinates Plot')
            
        else:  # radar chart
            angles = np.linspace(0, 2*np.pi, len(features), endpoint=False)
            cluster_means = np.array([
                data[labels == label].mean() 
                for label in np.unique(labels)
            ])
            
            ax = self.fig.add_subplot(111, projection='polar')
            for i, means in enumerate(cluster_means):
                values = np.concatenate((means, [means[0]]))
                angles_plot = np.concatenate((angles, [angles[0]]))
                ax.plot(angles_plot, values, label=f'Cluster {i}')
                
            ax.set_xticks(angles)
            ax.set_xticklabels(features)
            ax.set_title('Radar Chart of Cluster Profiles')
            ax.legend()
            
        self.plot_data = pd.DataFrame({
            'Feature': np.repeat(features, len(np.unique(labels))),
            'Cluster': np.tile(np.unique(labels), len(features)),
            'Value': cluster_means.flatten()
        })

    def _generate_reduction_plot(self):
        """Generate dimensionality reduction visualization"""
        method = self.reduction_method.get()
        n_components = self.n_components.get()
        
        if method == "pca":
            reducer = PCA(n_components=n_components)
        elif method == "tsne":
            reducer = TSNE(n_components=n_components)
        else:  # umap
            reducer = umap.UMAP(n_components=n_components)
            
        reduced_data = reducer.fit_transform(self.data)
        
        if n_components == 2:
            ax = self.fig.add_subplot(111)
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1],
                               c=self.cluster_labels, cmap='viridis')
            ax.set_title(f'{method.upper()} Projection')
            self.fig.colorbar(scatter, label='Cluster')
            
        elif n_components == 3:
            ax = self.fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1],
                               reduced_data[:, 2], c=self.cluster_labels,
                               cmap='viridis')
            ax.set_title(f'{method.upper()} Projection')
            self.fig.colorbar(scatter, label='Cluster')
            
        self.plot_data = pd.DataFrame(
            reduced_data,
            columns=[f'Component {i+1}' for i in range(n_components)])
        self.plot_data['Cluster'] = self.cluster_labels

    def _generate_importance_plot(self):
        """Generate feature importance visualization"""
        method = self.importance_method.get()
        n_features = self.n_top_features.get()
        
        if method == "statistical":
            importances = []
            pvalues = []
            for col in self.data.columns:
                f_stat, p_val = f_oneway(*[
                    self.data[col][self.cluster_labels == label]
                    for label in np.unique(self.cluster_labels)
                ])
                importances.append(f_stat)
                pvalues.append(p_val)
                
            importance_df = pd.DataFrame({
                'Feature': self.data.columns,
                'Importance': importances,
                'p-value': pvalues
            })
            
        else:  # random forest
            rf = RandomForestClassifier(n_estimators=100)
            rf.fit(self.data, self.cluster_labels)
            
            importance_df = pd.DataFrame({
                'Feature': self.data.columns,
                'Importance': rf.feature_importances_
            })
            
        importance_df = importance_df.nlargest(n_features, 'Importance')
        
        ax = self.fig.add_subplot(111)
        sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax)
        ax.set_title('Feature Importance')
        
        self.plot_data = importance_df

    def save_plot(self):
        """Save current plot to file"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("SVG files", "*.svg")
                ]
            )
            
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
            
            # Components
            comp_frame = ttk.Frame(self.reduction_frame)
            comp_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(comp_frame, text="Number of components:").pack(side=tk.LEFT)
            self.n_components = tk.IntVar(value=2)
            ttk.Entry(
                comp_frame,
                textvariable=self.n_components,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Method-specific parameters frame
            self.reduction_params_frame = ttk.Frame(self.reduction_frame)
            self.reduction_params_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Initialize with PCA parameters
            self.create_pca_params()
            
            self.logger.info("Reduction frame created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create reduction frame: {str(e)}")
            raise

    def update_reduction_options(self, *args):
        """Update reduction options based on selected method"""
        try:
            method = self.reduction_method.get()
            
            # Clear existing parameters
            for widget in self.reduction_params_frame.winfo_children():
                widget.destroy()
            
            # Show method-specific parameters
            if method == "pca":
                self.create_pca_params()
            elif method == "tsne":
                self.create_tsne_params()
            elif method == "umap":
                self.create_umap_params()
            
            self.logger.info(f"Updated reduction options for method: {method}")
            
        except Exception as e:
            self.logger.error(f"Failed to update reduction options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update reduction options: {str(e)}")

    def create_pca_params(self):
        """Create parameter inputs for PCA"""
        try:
            # Explained variance ratio threshold
            var_frame = ttk.Frame(self.reduction_params_frame)
            var_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(var_frame, text="Min explained variance:").pack(side=tk.LEFT)
            self.min_variance = tk.DoubleVar(value=0.95)
            ttk.Entry(
                var_frame,
                textvariable=self.min_variance,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Whitening option
            self.whiten = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                self.reduction_params_frame,
                text="Apply whitening",
                variable=self.whiten
            ).pack(anchor=tk.W)
            
        except Exception as e:
            self.logger.error(f"Failed to create PCA parameters: {str(e)}")
            raise

    def create_tsne_params(self):
        """Create parameter inputs for t-SNE"""
        try:
            # Perplexity
            perp_frame = ttk.Frame(self.reduction_params_frame)
            perp_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(perp_frame, text="Perplexity:").pack(side=tk.LEFT)
            self.perplexity = tk.DoubleVar(value=30.0)
            ttk.Entry(
                perp_frame,
                textvariable=self.perplexity,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Learning rate
            lr_frame = ttk.Frame(self.reduction_params_frame)
            lr_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(lr_frame, text="Learning rate:").pack(side=tk.LEFT)
            self.learning_rate = tk.DoubleVar(value=200.0)
            ttk.Entry(
                lr_frame,
                textvariable=self.learning_rate,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Number of iterations
            iter_frame = ttk.Frame(self.reduction_params_frame)
            iter_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(iter_frame, text="Iterations:").pack(side=tk.LEFT)
            self.n_iter_tsne = tk.IntVar(value=1000)
            ttk.Entry(
                iter_frame,
                textvariable=self.n_iter_tsne,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            self.logger.error(f"Failed to create t-SNE parameters: {str(e)}")
            raise

    def create_umap_params(self):
        """Create parameter inputs for UMAP"""
        try:
            # Number of neighbors
            neighbors_frame = ttk.Frame(self.reduction_params_frame)
            neighbors_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(neighbors_frame, text="Number of neighbors:").pack(side=tk.LEFT)
            self.n_neighbors = tk.IntVar(value=15)
            ttk.Entry(
                neighbors_frame,
                textvariable=self.n_neighbors,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Minimum distance
            dist_frame = ttk.Frame(self.reduction_params_frame)
            dist_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(dist_frame, text="Minimum distance:").pack(side=tk.LEFT)
            self.min_dist = tk.DoubleVar(value=0.1)
            ttk.Entry(
                dist_frame,
                textvariable=self.min_dist,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Metric
            metric_frame = ttk.Frame(self.reduction_params_frame)
            metric_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(metric_frame, text="Distance metric:").pack(side=tk.LEFT)
            self.metric = tk.StringVar(value="euclidean")
            ttk.OptionMenu(
                metric_frame,
                self.metric,
                "euclidean",
                "euclidean", "manhattan", "cosine"
            ).pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            self.logger.error(f"Failed to create UMAP parameters: {str(e)}")
            raise

    def create_importance_frame(self):
        """Create frame for feature importance visualization options"""
        try:
            self.importance_frame = ttk.LabelFrame(self.viz_tab, text="Feature Importance Options")
            
            # Method selection
            method_frame = ttk.Frame(self.importance_frame)
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

    def export_plot_data(self):
        """Export data used to generate current plot"""
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
                width=5,
                validate='key',
                validatecommand=(self.register(self.validate_alpha), '%P')
            ).pack(side=tk.LEFT, padx=5)
            
            # Multiple testing correction
            correction_frame = ttk.Frame(self.stat_frame)
            correction_frame.pack(fill=tk.X, padx=5, pady=2)
            
            self.correction = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                correction_frame,
                text="Apply multiple testing correction",
                variable=self.correction,
                command=self.update_correction_options
            ).pack(side=tk.LEFT)
            
            self.correction_method = tk.StringVar(value="bonferroni")
            self.correction_menu = ttk.OptionMenu(
                correction_frame,
                self.correction_method,
                "bonferroni",
                "bonferroni", "fdr", "holm"
            )
            self.correction_menu.pack(side=tk.LEFT, padx=5)
            
            # Feature selection
            feature_frame = ttk.LabelFrame(self.analysis_tab, text="Feature Selection")
            feature_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Feature listbox with scrollbar
            listbox_frame = ttk.Frame(feature_frame)
            listbox_frame.pack(fill=tk.X, padx=5, pady=2)
            
            self.feature_listbox = tk.Listbox(
                listbox_frame,
                selectmode=tk.MULTIPLE,
                height=6
            )
            self.feature_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            feature_scroll = ttk.Scrollbar(
                listbox_frame,
                orient=tk.VERTICAL,
                command=self.feature_listbox.yview
            )
            feature_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            self.feature_listbox.configure(yscrollcommand=feature_scroll.set)
            
            # Results frame
            results_frame = ttk.LabelFrame(self.analysis_tab, text="Analysis Results")
            results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Results text with scrollbars
            text_frame = ttk.Frame(results_frame)
            text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            self.analysis_text = tk.Text(
                text_frame,
                wrap=tk.WORD,
                height=10
            )
            self.analysis_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            text_scroll = ttk.Scrollbar(
                text_frame,
                orient=tk.VERTICAL,
                command=self.analysis_text.yview
            )
            text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            self.analysis_text.configure(yscrollcommand=text_scroll.set)
            
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
                text="Clear Results",
                command=lambda: self.analysis_text.delete(1.0, tk.END)
            ).pack(side=tk.LEFT, padx=5)
            
            # Initialize analysis variables
            self.analysis_results = None
            
            self.logger.info("Analysis tab created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create analysis tab: {str(e)}")
            raise

    def validate_alpha(self, value):
        """Validate that alpha value is between 0 and 1"""
        if value == "":
            return True
        try:
            alpha = float(value)
            return 0 < alpha < 1
        except ValueError:
            return False

    def update_stat_options(self, *args):
        """Update statistical analysis options based on selected test"""
        try:
            test = self.stat_test.get()
            
            # Enable/disable correction options based on test type
            if test == "chi2":
                self.correction.set(False)
                self.correction_menu.configure(state='disabled')
            else:
                self.correction_menu.configure(state='normal')
                
            self.logger.info(f"Updated statistical options for test: {test}")
            
        except Exception as e:
            self.logger.error(f"Failed to update statistical options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update options: {str(e)}")

    def update_correction_options(self):
        """Enable/disable correction method based on correction checkbox"""
        try:
            if self.correction.get():
                self.correction_menu.configure(state='normal')
            else:
                self.correction_menu.configure(state='disabled')
                
            self.logger.info(f"Updated correction options: {self.correction.get()}")
            
        except Exception as e:
            self.logger.error(f"Failed to update correction options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update options: {str(e)}")

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
            self.cluster_method = tk.StringVar(value="kmeans")
            self.cluster_method_menu = ttk.OptionMenu(
                method_frame,
                self.cluster_method,
                "kmeans",
                "kmeans", "dbscan", "hierarchical",
                command=self.update_cluster_options
            )
            self.cluster_method_menu.pack(side=tk.LEFT, padx=5)
            
            # Initially show kmeans frame
            self.kmeans_frame.pack(fill=tk.X, padx=5, pady=5)
            
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

    def save_clustering_results(self):
        """Save clustering results to file"""
        try:
            if not hasattr(self, 'normalized_data') or self.normalized_data is None:
                raise ValueError("No clustering results available to save")
                
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if file_path:
                # Save results
                results_df = self.normalized_data.copy()
                results_df['Cluster'] = self.cluster_labels
                results_df.to_csv(file_path, index=False)
                
                self.status_var.set(f"Results saved to {file_path}")
                self.logger.info(f"Clustering results saved to {file_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to save clustering results: {str(e)}")
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")

    def update_cluster_options(self, *args):
        """Update clustering options based on selected method."""
        try:
            method = self.cluster_method.get()
            
            # Hide all parameter frames initially
            self.kmeans_frame.pack_forget()
            self.dbscan_frame.pack_forget()
            self.hierarchical_frame.pack_forget()
            
            # Show the relevant frame based on the selected method
            if method == "kmeans":
                self.kmeans_frame.pack(fill=tk.X, padx=5, pady=5)
            elif method == "dbscan":
                self.dbscan_frame.pack(fill=tk.X, padx=5, pady=5)
            elif method == "hierarchical":
                self.hierarchical_frame.pack(fill=tk.X, padx=5, pady=5)
            
            self.logger.info(f"Updated cluster options for method: {method}")
            
        except Exception as e:
            self.logger.error(f"Failed to update cluster options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update cluster options: {str(e)}")

    def create_distribution_frame(self):
        """Create frame for distribution visualization options"""
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
                # options will be populated when data is loaded
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
            
            # Additional options frame
            options_frame = ttk.Frame(self.dist_frame)
            options_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Bin count for histogram
            self.bin_count = tk.IntVar(value=30)
            ttk.Label(options_frame, text="Bins:").pack(side=tk.LEFT)
            ttk.Entry(
                options_frame,
                textvariable=self.bin_count,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # KDE bandwidth
            self.kde_bandwidth = tk.DoubleVar(value=0.2)
            ttk.Label(options_frame, text="Bandwidth:").pack(side=tk.LEFT)
            ttk.Entry(
                options_frame,
                textvariable=self.kde_bandwidth,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            self.logger.info("Distribution frame created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create distribution frame: {str(e)}")
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
            
            # Create all visualization frames
            self.create_distribution_frame()
            self.create_profile_frame()
            self.create_reduction_frame()
            self.create_importance_frame()
            
            # Initially show distribution frame
            self.dist_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Plot frame
            plot_container = ttk.LabelFrame(self.viz_tab, text="Visualization")
            plot_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Toolbar frame
            self.toolbar_frame = ttk.Frame(plot_container)
            self.toolbar_frame.pack(fill=tk.X)
            
            # Plot frame
            self.plot_frame = ttk.Frame(plot_container)
            self.plot_frame.pack(fill=tk.BOTH, expand=True)
            
            # Initialize matplotlib figure and canvas
            self.fig = Figure(figsize=(8, 6), dpi=100)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add matplotlib toolbar
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
            self.toolbar.update()
            
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

    def update_feature_menu(self):
        """Update feature menu options when data is loaded"""
        try:
            if self.normalized_data is not None:
                # Get list of numeric columns
                features = list(self.normalized_data.select_dtypes(
                    include=['float64', 'int64']).columns)
                
                # Update feature menu
                menu = self.feature_menu["menu"]
                menu.delete(0, "end")
                
                for feature in features:
                    menu.add_command(
                        label=feature,
                        command=lambda f=feature: self.dist_feature.set(f)
                    )
                
                # Set default feature if none selected
                if not self.dist_feature.get() and features:
                    self.dist_feature.set(features[0])
                    
                self.logger.info("Feature menu updated successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to update feature menu: {str(e)}")
            messagebox.showerror("Error", f"Failed to update feature menu: {str(e)}")

    def generate_plot(self):
        """Generate visualization based on selected plot type and options"""
        try:
            # Clear previous plot
            self.fig.clear()
            
            # Get selected plot type
            plot_type = self.plot_type.get()
            
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
            
            self.logger.info(f"Generated {plot_type} plot successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to generate plot: {str(e)}")
            messagebox.showerror("Error", f"Failed to generate plot: {str(e)}")

    def _generate_distribution_plot(self):
        """Generate distribution plot for selected feature"""
        feature = self.dist_feature.get()
        plot_type = self.dist_type.get()
        
        if not feature:
            raise ValueError("No feature selected")
            
        data = self.data[feature]
        labels = self.cluster_labels
        
        ax = self.fig.add_subplot(111)
        
        if plot_type == "histogram":
            for label in np.unique(labels):
                ax.hist(data[labels == label], alpha=0.5, 
                       label=f'Cluster {label}', bins=30)
        elif plot_type == "kde":
            for label in np.unique(labels):
                sns.kdeplot(data[labels == label], ax=ax,
                          label=f'Cluster {label}')
        else:  # box plot
            ax.boxplot([data[labels == label] 
                       for label in np.unique(labels)],
                      labels=[f'Cluster {label}' 
                             for label in np.unique(labels)])
            
        ax.set_title(f'{feature} Distribution by Cluster')
        ax.set_xlabel(feature)
        ax.legend()
        
        self.plot_data = pd.DataFrame({
            'Feature': data,
            'Cluster': labels
        })

    def _generate_profile_plot(self):
        """Generate cluster profile visualization"""
        plot_type = self.profile_type.get()
        features = [self.feature_listbox.get(i) 
                   for i in self.feature_listbox.curselection()]
        
        if not features:
            raise ValueError("No features selected")
            
        data = self.data[features]
        labels = self.cluster_labels
        
        if plot_type == "heatmap":
            cluster_means = pd.DataFrame([
                data[labels == label].mean() 
                for label in np.unique(labels)
            ], index=[f'Cluster {label}' for label in np.unique(labels)])
            
            ax = self.fig.add_subplot(111)
            sns.heatmap(cluster_means, cmap='coolwarm', center=0,
                       annot=True, ax=ax)
            ax.set_title('Cluster Profiles Heatmap')
            
        elif plot_type == "parallel":
            ax = self.fig.add_subplot(111)
            pd.plotting.parallel_coordinates(
                pd.concat([data, pd.Series(labels, name='Cluster')], axis=1),
                'Cluster', ax=ax)
            ax.set_title('Parallel Coordinates Plot')
            
        else:  # radar chart
            angles = np.linspace(0, 2*np.pi, len(features), endpoint=False)
            cluster_means = np.array([
                data[labels == label].mean() 
                for label in np.unique(labels)
            ])
            
            ax = self.fig.add_subplot(111, projection='polar')
            for i, means in enumerate(cluster_means):
                values = np.concatenate((means, [means[0]]))
                angles_plot = np.concatenate((angles, [angles[0]]))
                ax.plot(angles_plot, values, label=f'Cluster {i}')
                
            ax.set_xticks(angles)
            ax.set_xticklabels(features)
            ax.set_title('Radar Chart of Cluster Profiles')
            ax.legend()
            
        self.plot_data = pd.DataFrame({
            'Feature': np.repeat(features, len(np.unique(labels))),
            'Cluster': np.tile(np.unique(labels), len(features)),
            'Value': cluster_means.flatten()
        })

    def _generate_reduction_plot(self):
        """Generate dimensionality reduction visualization"""
        method = self.reduction_method.get()
        n_components = self.n_components.get()
        
        if method == "pca":
            reducer = PCA(n_components=n_components)
        elif method == "tsne":
            reducer = TSNE(n_components=n_components)
        else:  # umap
            reducer = umap.UMAP(n_components=n_components)
            
        reduced_data = reducer.fit_transform(self.data)
        
        if n_components == 2:
            ax = self.fig.add_subplot(111)
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1],
                               c=self.cluster_labels, cmap='viridis')
            ax.set_title(f'{method.upper()} Projection')
            self.fig.colorbar(scatter, label='Cluster')
            
        elif n_components == 3:
            ax = self.fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1],
                               reduced_data[:, 2], c=self.cluster_labels,
                               cmap='viridis')
            ax.set_title(f'{method.upper()} Projection')
            self.fig.colorbar(scatter, label='Cluster')
            
        self.plot_data = pd.DataFrame(
            reduced_data,
            columns=[f'Component {i+1}' for i in range(n_components)])
        self.plot_data['Cluster'] = self.cluster_labels

    def _generate_importance_plot(self):
        """Generate feature importance visualization"""
        method = self.importance_method.get()
        n_features = self.n_top_features.get()
        
        if method == "statistical":
            importances = []
            pvalues = []
            for col in self.data.columns:
                f_stat, p_val = f_oneway(*[
                    self.data[col][self.cluster_labels == label]
                    for label in np.unique(self.cluster_labels)
                ])
                importances.append(f_stat)
                pvalues.append(p_val)
                
            importance_df = pd.DataFrame({
                'Feature': self.data.columns,
                'Importance': importances,
                'p-value': pvalues
            })
            
        else:  # random forest
            rf = RandomForestClassifier(n_estimators=100)
            rf.fit(self.data, self.cluster_labels)
            
            importance_df = pd.DataFrame({
                'Feature': self.data.columns,
                'Importance': rf.feature_importances_
            })
            
        importance_df = importance_df.nlargest(n_features, 'Importance')
        
        ax = self.fig.add_subplot(111)
        sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax)
        ax.set_title('Feature Importance')
        
        self.plot_data = importance_df

    def save_plot(self):
        """Save current plot to file"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("SVG files", "*.svg")
                ]
            )
            
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
            
            # Components
            comp_frame = ttk.Frame(self.reduction_frame)
            comp_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(comp_frame, text="Number of components:").pack(side=tk.LEFT)
            self.n_components = tk.IntVar(value=2)
            ttk.Entry(
                comp_frame,
                textvariable=self.n_components,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Method-specific parameters frame
            self.reduction_params_frame = ttk.Frame(self.reduction_frame)
            self.reduction_params_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Initialize with PCA parameters
            self.create_pca_params()
            
            self.logger.info("Reduction frame created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create reduction frame: {str(e)}")
            raise

    def update_reduction_options(self, *args):
        """Update reduction options based on selected method"""
        try:
            method = self.reduction_method.get()
            
            # Clear existing parameters
            for widget in self.reduction_params_frame.winfo_children():
                widget.destroy()
            
            # Show method-specific parameters
            if method == "pca":
                self.create_pca_params()
            elif method == "tsne":
                self.create_tsne_params()
            elif method == "umap":
                self.create_umap_params()
            
            self.logger.info(f"Updated reduction options for method: {method}")
            
        except Exception as e:
            self.logger.error(f"Failed to update reduction options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update reduction options: {str(e)}")

    def create_pca_params(self):
        """Create parameter inputs for PCA"""
        try:
            # Explained variance ratio threshold
            var_frame = ttk.Frame(self.reduction_params_frame)
            var_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(var_frame, text="Min explained variance:").pack(side=tk.LEFT)
            self.min_variance = tk.DoubleVar(value=0.95)
            ttk.Entry(
                var_frame,
                textvariable=self.min_variance,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Whitening option
            self.whiten = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                self.reduction_params_frame,
                text="Apply whitening",
                variable=self.whiten
            ).pack(anchor=tk.W)
            
        except Exception as e:
            self.logger.error(f"Failed to create PCA parameters: {str(e)}")
            raise

    def create_tsne_params(self):
        """Create parameter inputs for t-SNE"""
        try:
            # Perplexity
            perp_frame = ttk.Frame(self.reduction_params_frame)
            perp_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(perp_frame, text="Perplexity:").pack(side=tk.LEFT)
            self.perplexity = tk.DoubleVar(value=30.0)
            ttk.Entry(
                perp_frame,
                textvariable=self.perplexity,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Learning rate
            lr_frame = ttk.Frame(self.reduction_params_frame)
            lr_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(lr_frame, text="Learning rate:").pack(side=tk.LEFT)
            self.learning_rate = tk.DoubleVar(value=200.0)
            ttk.Entry(
                lr_frame,
                textvariable=self.learning_rate,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Number of iterations
            iter_frame = ttk.Frame(self.reduction_params_frame)
            iter_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(iter_frame, text="Iterations:").pack(side=tk.LEFT)
            self.n_iter_tsne = tk.IntVar(value=1000)
            ttk.Entry(
                iter_frame,
                textvariable=self.n_iter_tsne,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            self.logger.error(f"Failed to create t-SNE parameters: {str(e)}")
            raise

    def create_umap_params(self):
        """Create parameter inputs for UMAP"""
        try:
            # Number of neighbors
            neighbors_frame = ttk.Frame(self.reduction_params_frame)
            neighbors_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(neighbors_frame, text="Number of neighbors:").pack(side=tk.LEFT)
            self.n_neighbors = tk.IntVar(value=15)
            ttk.Entry(
                neighbors_frame,
                textvariable=self.n_neighbors,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Minimum distance
            dist_frame = ttk.Frame(self.reduction_params_frame)
            dist_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(dist_frame, text="Minimum distance:").pack(side=tk.LEFT)
            self.min_dist = tk.DoubleVar(value=0.1)
            ttk.Entry(
                dist_frame,
                textvariable=self.min_dist,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Metric
            metric_frame = ttk.Frame(self.reduction_params_frame)
            metric_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(metric_frame, text="Distance metric:").pack(side=tk.LEFT)
            self.metric = tk.StringVar(value="euclidean")
            ttk.OptionMenu(
                metric_frame,
                self.metric,
                "euclidean",
                "euclidean", "manhattan", "cosine"
            ).pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            self.logger.error(f"Failed to create UMAP parameters: {str(e)}")
            raise

    def create_importance_frame(self):
        """Create frame for feature importance visualization options"""
        try:
            self.importance_frame = ttk.LabelFrame(self.viz_tab, text="Feature Importance Options")
            
            # Method selection
            method_frame = ttk.Frame(self.importance_frame)
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

    def export_plot_data(self):
        """Export data used to generate current plot"""
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
                width=5,
                validate='key',
                validatecommand=(self.register(self.validate_alpha), '%P')
            ).pack(side=tk.LEFT, padx=5)
            
            # Multiple testing correction
            correction_frame = ttk.Frame(self.stat_frame)
            correction_frame.pack(fill=tk.X, padx=5, pady=2)
            
            self.correction = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                correction_frame,
                text="Apply multiple testing correction",
                variable=self.correction,
                command=self.update_correction_options
            ).pack(side=tk.LEFT)
            
            self.correction_method = tk.StringVar(value="bonferroni")
            self.correction_menu = ttk.OptionMenu(
                correction_frame,
                self.correction_method,
                "bonferroni",
                "bonferroni", "fdr", "holm"
            )
            self.correction_menu.pack(side=tk.LEFT, padx=5)
            
            # Feature selection
            feature_frame = ttk.LabelFrame(self.analysis_tab, text="Feature Selection")
            feature_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Feature listbox with scrollbar
            listbox_frame = ttk.Frame(feature_frame)
            listbox_frame.pack(fill=tk.X, padx=5, pady=2)
            
            self.feature_listbox = tk.Listbox(
                listbox_frame,
                selectmode=tk.MULTIPLE,
                height=6
            )
            self.feature_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            feature_scroll = ttk.Scrollbar(
                listbox_frame,
                orient=tk.VERTICAL,
                command=self.feature_listbox.yview
            )
            feature_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            self.feature_listbox.configure(yscrollcommand=feature_scroll.set)
            
            # Results frame
            results_frame = ttk.LabelFrame(self.analysis_tab, text="Analysis Results")
            results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Results text with scrollbars
            text_frame = ttk.Frame(results_frame)
            text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            self.analysis_text = tk.Text(
                text_frame,
                wrap=tk.WORD,
                height=10
            )
            self.analysis_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            text_scroll = ttk.Scrollbar(
                text_frame,
                orient=tk.VERTICAL,
                command=self.analysis_text.yview
            )
            text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            self.analysis_text.configure(yscrollcommand=text_scroll.set)
            
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
                text="Clear Results",
                command=lambda: self.analysis_text.delete(1.0, tk.END)
            ).pack(side=tk.LEFT, padx=5)
            
            # Initialize analysis variables
            self.analysis_results = None
            
            self.logger.info("Analysis tab created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create analysis tab: {str(e)}")
            raise

    def validate_alpha(self, value):
        """Validate that alpha value is between 0 and 1"""
        if value == "":
            return True
        try:
            alpha = float(value)
            return 0 < alpha < 1
        except ValueError:
            return False

    def update_stat_options(self, *args):
        """Update statistical analysis options based on selected test"""
        try:
            test = self.stat_test.get()
            
            # Enable/disable correction options based on test type
            if test == "chi2":
                self.correction.set(False)
                self.correction_menu.configure(state='disabled')
            else:
                self.correction_menu.configure(state='normal')
                
            self.logger.info(f"Updated statistical options for test: {test}")
            
        except Exception as e:
            self.logger.error(f"Failed to update statistical options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update options: {str(e)}")

    def update_correction_options(self):
        """Enable/disable correction method based on correction checkbox"""
        try:
            if self.correction.get():
                self.correction_menu.configure(state='normal')
            else:
                self.correction_menu.configure(state='disabled')
                
            self.logger.info(f"Updated correction options: {self.correction.get()}")
            
        except Exception as e:
            self.logger.error(f"Failed to update correction options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update options: {str(e)}")

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
            self.cluster_method = tk.StringVar(value="kmeans")
            self.cluster_method_menu = ttk.OptionMenu(
                method_frame,
                self.cluster_method,
                "kmeans",
                "kmeans", "dbscan", "hierarchical",
                command=self.update_cluster_options
            )
            self.cluster_method_menu.pack(side=tk.LEFT, padx=5)
            
            # Initially show kmeans frame
            self.kmeans_frame.pack(fill=tk.X, padx=5, pady=5)
            
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

    def save_clustering_results(self):
        """Save clustering results to file"""
        try:
            if not hasattr(self, 'normalized_data') or self.normalized_data is None:
                raise ValueError("No clustering results available to save")
                
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if file_path:
                # Save results
                results_df = self.normalized_data.copy()
                results_df['Cluster'] = self.cluster_labels
                results_df.to_csv(file_path, index=False)
                
                self.status_var.set(f"Results saved to {file_path}")
                self.logger.info(f"Clustering results saved to {file_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to save clustering results: {str(e)}")
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")

    def update_cluster_options(self, *args):
        """Update clustering options based on selected method."""
        try:
            method = self.cluster_method.get()
            
            # Hide all parameter frames initially
            self.kmeans_frame.pack_forget()
            self.dbscan_frame.pack_forget()
            self.hierarchical_frame.pack_forget()
            
            # Show the relevant frame based on the selected method
            if method == "kmeans":
                self.kmeans_frame.pack(fill=tk.X, padx=5, pady=5)
            elif method == "dbscan":
                self.dbscan_frame.pack(fill=tk.X, padx=5, pady=5)
            elif method == "hierarchical":
                self.hierarchical_frame.pack(fill=tk.X, padx=5, pady=5)
            
            self.logger.info(f"Updated cluster options for method: {method}")
            
        except Exception as e:
            self.logger.error(f"Failed to update cluster options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update cluster options: {str(e)}")

    def create_distribution_frame(self):
        """Create frame for distribution visualization options"""
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
                # options will be populated when data is loaded
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
            
            # Additional options frame
            options_frame = ttk.Frame(self.dist_frame)
            options_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Bin count for histogram
            self.bin_count = tk.IntVar(value=30)
            ttk.Label(options_frame, text="Bins:").pack(side=tk.LEFT)
            ttk.Entry(
                options_frame,
                textvariable=self.bin_count,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # KDE bandwidth
            self.kde_bandwidth = tk.DoubleVar(value=0.2)
            ttk.Label(options_frame, text="Bandwidth:").pack(side=tk.LEFT)
            ttk.Entry(
                options_frame,
                textvariable=self.kde_bandwidth,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            self.logger.info("Distribution frame created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create distribution frame: {str(e)}")
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
            
            # Create all visualization frames
            self.create_distribution_frame()
            self.create_profile_frame()
            self.create_reduction_frame()
            self.create_importance_frame()
            
            # Initially show distribution frame
            self.dist_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Plot frame
            plot_container = ttk.LabelFrame(self.viz_tab, text="Visualization")
            plot_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Toolbar frame
            self.toolbar_frame = ttk.Frame(plot_container)
            self.toolbar_frame.pack(fill=tk.X)
            
            # Plot frame
            self.plot_frame = ttk.Frame(plot_container)
            self.plot_frame.pack(fill=tk.BOTH, expand=True)
            
            # Initialize matplotlib figure and canvas
            self.fig = Figure(figsize=(8, 6), dpi=100)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add matplotlib toolbar
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
            self.toolbar.update()
            
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

    def update_feature_menu(self):
        """Update feature menu options when data is loaded"""
        try:
            if self.normalized_data is not None:
                # Get list of numeric columns
                features = list(self.normalized_data.select_dtypes(
                    include=['float64', 'int64']).columns)
                
                # Update feature menu
                menu = self.feature_menu["menu"]
                menu.delete(0, "end")
                
                for feature in features:
                    menu.add_command(
                        label=feature,
                        command=lambda f=feature: self.dist_feature.set(f)
                    )
                
                # Set default feature if none selected
                if not self.dist_feature.get() and features:
                    self.dist_feature.set(features[0])
                    
                self.logger.info("Feature menu updated successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to update feature menu: {str(e)}")
            messagebox.showerror("Error", f"Failed to update feature menu: {str(e)}")

    def generate_plot(self):
        """Generate visualization based on selected plot type and options"""
        try:
            # Clear previous plot
            self.fig.clear()
            
            # Get selected plot type
            plot_type = self.plot_type.get()
            
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
            
            self.logger.info(f"Generated {plot_type} plot successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to generate plot: {str(e)}")
            messagebox.showerror("Error", f"Failed to generate plot: {str(e)}")

    def _generate_distribution_plot(self):
        """Generate distribution plot for selected feature"""
        feature = self.dist_feature.get()
        plot_type = self.dist_type.get()
        
        if not feature:
            raise ValueError("No feature selected")
            
        data = self.data[feature]
        labels = self.cluster_labels
        
        ax = self.fig.add_subplot(111)
        
        if plot_type == "histogram":
            for label in np.unique(labels):
                ax.hist(data[labels == label], alpha=0.5, 
                       label=f'Cluster {label}', bins=30)
        elif plot_type == "kde":
            for label in np.unique(labels):
                sns.kdeplot(data[labels == label], ax=ax,
                          label=f'Cluster {label}')
        else:  # box plot
            ax.boxplot([data[labels == label] 
                       for label in np.unique(labels)],
                      labels=[f'Cluster {label}' 
                             for label in np.unique(labels)])
            
        ax.set_title(f'{feature} Distribution by Cluster')
        ax.set_xlabel(feature)
        ax.legend()
        
        self.plot_data = pd.DataFrame({
            'Feature': data,
            'Cluster': labels
        })

    def _generate_profile_plot(self):
        """Generate cluster profile visualization"""
        plot_type = self.profile_type.get()
        features = [self.feature_listbox.get(i) 
                   for i in self.feature_listbox.curselection()]
        
        if not features:
            raise ValueError("No features selected")
            
        data = self.data[features]
        labels = self.cluster_labels
        
        if plot_type == "heatmap":
            cluster_means = pd.DataFrame([
                data[labels == label].mean() 
                for label in np.unique(labels)
            ], index=[f'Cluster {label}' for label in np.unique(labels)])
            
            ax = self.fig.add_subplot(111)
            sns.heatmap(cluster_means, cmap='coolwarm', center=0,
                       annot=True, ax=ax)
            ax.set_title('Cluster Profiles Heatmap')
            
        elif plot_type == "parallel":
            ax = self.fig.add_subplot(111)
            pd.plotting.parallel_coordinates(
                pd.concat([data, pd.Series(labels, name='Cluster')], axis=1),
                'Cluster', ax=ax)
            ax.set_title('Parallel Coordinates Plot')
            
        else:  # radar chart
            angles = np.linspace(0, 2*np.pi, len(features), endpoint=False)
            cluster_means = np.array([
                data[labels == label].mean() 
                for label in np.unique(labels)
            ])
            
            ax = self.fig.add_subplot(111, projection='polar')
            for i, means in enumerate(cluster_means):
                values = np.concatenate((means, [means[0]]))
                angles_plot = np.concatenate((angles, [angles[0]]))
                ax.plot(angles_plot, values, label=f'Cluster {i}')
                
            ax.set_xticks(angles)
            ax.set_xticklabels(features)
            ax.set_title('Radar Chart of Cluster Profiles')
            ax.legend()
            
        self.plot_data = pd.DataFrame({
            'Feature': np.repeat(features, len(np.unique(labels))),
            'Cluster': np.tile(np.unique(labels), len(features)),
            'Value': cluster_means.flatten()
        })

    def _generate_reduction_plot(self):
        """Generate dimensionality reduction visualization"""
        method = self.reduction_method.get()
        n_components = self.n_components.get()
        
        if method == "pca":
            reducer = PCA(n_components=n_components)
        elif method == "tsne":
            reducer = TSNE(n_components=n_components)
        else:  # umap
            reducer = umap.UMAP(n_components=n_components)
            
        reduced_data = reducer.fit_transform(self.data)
        
        if n_components == 2:
            ax = self.fig.add_subplot(111)
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1],
                               c=self.cluster_labels, cmap='viridis')
            ax.set_title(f'{method.upper()} Projection')
            self.fig.colorbar(scatter, label='Cluster')
            
        elif n_components == 3:
            ax = self.fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1],
                               reduced_data[:, 2], c=self.cluster_labels,
                               cmap='viridis')
            ax.set_title(f'{method.upper()} Projection')
            self.fig.colorbar(scatter, label='Cluster')
            
        self.plot_data = pd.DataFrame(
            reduced_data,
            columns=[f'Component {i+1}' for i in range(n_components)])
        self.plot_data['Cluster'] = self.cluster_labels

    def _generate_importance_plot(self):
        """Generate feature importance visualization"""
        method = self.importance_method.get()
        n_features = self.n_top_features.get()
        
        if method == "statistical":
            importances = []
            pvalues = []
            for col in self.data.columns:
                f_stat, p_val = f_oneway(*[
                    self.data[col][self.cluster_labels == label]
                    for label in np.unique(self.cluster_labels)
                ])
                importances.append(f_stat)
                pvalues.append(p_val)
                
            importance_df = pd.DataFrame({
                'Feature': self.data.columns,
                'Importance': importances,
                'p-value': pvalues
            })
            
        else:  # random forest
            rf = RandomForestClassifier(n_estimators=100)
            rf.fit(self.data, self.cluster_labels)
            
            importance_df = pd.DataFrame({
                'Feature': self.data.columns,
                'Importance': rf.feature_importances_
            })
            
        importance_df = importance_df.nlargest(n_features, 'Importance')
        
        ax = self.fig.add_subplot(111)
        sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax)
        ax.set_title('Feature Importance')
        
        self.plot_data = importance_df

    def save_plot(self):
        """Save current plot to file"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("SVG files", "*.svg")
                ]
            )
            
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
            
            # Components
            comp_frame = ttk.Frame(self.reduction_frame)
            comp_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(comp_frame, text="Number of components:").pack(side=tk.LEFT)
            self.n_components = tk.IntVar(value=2)
            ttk.Entry(
                comp_frame,
                textvariable=self.n_components,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Method-specific parameters frame
            self.reduction_params_frame = ttk.Frame(self.reduction_frame)
            self.reduction_params_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Initialize with PCA parameters
            self.create_pca_params()
            
            self.logger.info("Reduction frame created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create reduction frame: {str(e)}")
            raise

    def update_reduction_options(self, *args):
        """Update reduction options based on selected method"""
        try:
            method = self.reduction_method.get()
            
            # Clear existing parameters
            for widget in self.reduction_params_frame.winfo_children():
                widget.destroy()
            
            # Show method-specific parameters
            if method == "pca":
                self.create_pca_params()
            elif method == "tsne":
                self.create_tsne_params()
            elif method == "umap":
                self.create_umap_params()
            
            self.logger.info(f"Updated reduction options for method: {method}")
            
        except Exception as e:
            self.logger.error(f"Failed to update reduction options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update reduction options: {str(e)}")

    def create_pca_params(self):
        """Create parameter inputs for PCA"""
        try:
            # Explained variance ratio threshold
            var_frame = ttk.Frame(self.reduction_params_frame)
            var_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(var_frame, text="Min explained variance:").pack(side=tk.LEFT)
            self.min_variance = tk.DoubleVar(value=0.95)
            ttk.Entry(
                var_frame,
                textvariable=self.min_variance,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Whitening option
            self.whiten = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                self.reduction_params_frame,
                text="Apply whitening",
                variable=self.whiten
            ).pack(anchor=tk.W)
            
        except Exception as e:
            self.logger.error(f"Failed to create PCA parameters: {str(e)}")
            raise

    def create_tsne_params(self):
        """Create parameter inputs for t-SNE"""
        try:
            # Perplexity
            perp_frame = ttk.Frame(self.reduction_params_frame)
            perp_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(perp_frame, text="Perplexity:").pack(side=tk.LEFT)
            self.perplexity = tk.DoubleVar(value=30.0)
            ttk.Entry(
                perp_frame,
                textvariable=self.perplexity,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Learning rate
            lr_frame = ttk.Frame(self.reduction_params_frame)
            lr_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(lr_frame, text="Learning rate:").pack(side=tk.LEFT)
            self.learning_rate = tk.DoubleVar(value=200.0)
            ttk.Entry(
                lr_frame,
                textvariable=self.learning_rate,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Number of iterations
            iter_frame = ttk.Frame(self.reduction_params_frame)
            iter_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(iter_frame, text="Iterations:").pack(side=tk.LEFT)
            self.n_iter_tsne = tk.IntVar(value=1000)
            ttk.Entry(
                iter_frame,
                textvariable=self.n_iter_tsne,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            self.logger.error(f"Failed to create t-SNE parameters: {str(e)}")
            raise

    def create_umap_params(self):
        """Create parameter inputs for UMAP"""
        try:
            # Number of neighbors
            neighbors_frame = ttk.Frame(self.reduction_params_frame)
            neighbors_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(neighbors_frame, text="Number of neighbors:").pack(side=tk.LEFT)
            self.n_neighbors = tk.IntVar(value=15)
            ttk.Entry(
                neighbors_frame,
                textvariable=self.n_neighbors,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Minimum distance
            dist_frame = ttk.Frame(self.reduction_params_frame)
            dist_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(dist_frame, text="Minimum distance:").pack(side=tk.LEFT)
            self.min_dist = tk.DoubleVar(value=0.1)
            ttk.Entry(
                dist_frame,
                textvariable=self.min_dist,
                width=5
            ).pack(side=tk.LEFT, padx=5)
            
            # Metric
            metric_frame = ttk.Frame(self.reduction_params_frame)
            metric_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(metric_frame, text="Distance metric:").pack(side=tk.LEFT)
            self.metric = tk.StringVar(value="euclidean")
            ttk.OptionMenu(
                metric_frame,
                self.metric,
                "euclidean",
                "euclidean", "manhattan", "cosine"
            ).pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            self.logger.error(f"Failed to create UMAP parameters: {str(e)}")
            raise

    def create_importance_frame(self):
        """Create frame for feature importance visualization options"""
        try:
            self.importance_frame = ttk.LabelFrame(self.viz_tab, text="Feature Importance Options")
            
            # Method selection
            method_frame = ttk.Frame(self.importance_frame)
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

    def export_plot_data(self):
        """Export data used to generate current plot"""
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
                width=5,
                validate='key',
                validatecommand=(self.register(self.validate_alpha), '%P')
            ).pack(side=tk.LEFT, padx=5)
            
            # Multiple testing correction
            correction_frame = ttk.Frame(self.stat_frame)
            correction_frame.pack(fill=tk.X, padx=5, pady=2)
            
            self.correction = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                correction_frame,
                text="Apply multiple testing correction",
                variable=self.correction,
                command=self.update_correction_options
            ).pack(side=tk.LEFT)
            
            self.correction_method = tk.StringVar(value="bonferroni")
            self.correction_menu = ttk.OptionMenu(
                correction_frame,
                self.correction_method,
                "bonferroni",
                "bonferroni", "fdr", "holm"
            )
            self.correction_menu.pack(side=tk.LEFT, padx=5)
            
            # Feature selection
            feature_frame = ttk.LabelFrame(self.analysis_tab, text="Feature Selection")
            feature_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Feature listbox with scrollbar
            listbox_frame = ttk.Frame(feature_frame)
            listbox_frame.pack(fill=tk.X, padx=5, pady=2)
            
            self.feature_listbox = tk.Listbox(
                listbox_frame,
                selectmode=tk.MULTIPLE,
                height=6
            )
            self.feature_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            feature_scroll = ttk.Scrollbar(
                listbox_frame,
                orient=tk.VERTICAL,
                command=self.feature_listbox.yview
            )
            feature_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            self.feature_listbox.configure(yscrollcommand=feature_scroll.set)
            
            # Results frame
            results_frame = ttk.LabelFrame(self.analysis_tab, text="Analysis Results")
            results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Results text with scrollbars
            text_frame = ttk.Frame(results_frame)
            text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            self.analysis_text = tk.Text(
                text_frame,
                wrap=tk.WORD,
                height=10
            )
            self.analysis_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            text_scroll = ttk.Scrollbar(
                text_frame,
                orient=tk.VERTICAL,
                command=self.analysis_text.yview
            )
            text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            self.analysis_text.configure(yscrollcommand=text_scroll.set)
            
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
                text="Clear Results",
                command=lambda: self.analysis_text.delete(1.0, tk.END)
            ).pack(side=tk.LEFT, padx=5)
            
            # Initialize analysis variables
            self.analysis_results = None
            
            self.logger.info("Analysis tab created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create analysis tab: {str(e)}")
            raise

    def validate_alpha(self, value):
        """Validate that alpha value is between 0 and 1"""
        if value == "":
            return True
        try:
            alpha = float(value)
            return 0 < alpha < 1
        except ValueError:
            return False

    def update_stat_options(self, *args):
        """Update statistical analysis options based on selected test"""
        try:
            test = self.stat_test.get()
            
            # Enable/disable correction options based on test type
            if test == "chi2":
                self.correction.set(False)
                self.correction_menu.configure(state='disabled')
            else:
                self.correction_menu.configure(state='normal')
                
            self.logger.info(f"Updated statistical options for test: {test}")
            
        except Exception as e:
            self.logger.error(f"Failed to update statistical options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update options: {str(e)}")

    def update_correction_options(self):
        """Enable/disable correction method based on correction checkbox"""
        try:
            if self.correction.get():
                self.correction_menu.configure(state='normal')
            else:
                self.correction_menu.configure(state='disabled')
                
            self.logger.info(f"Updated correction options: {self.correction.get()}")
            
        except Exception as e:
            self.logger.error(f"Failed to update correction options: {str(e)}")
            messagebox.showerror("Error", f"Failed to update options: {str(e)}")

    def run_analysis(self):
        """Run selected analysis on clustering results"""
        try:
            if self.normalized_data is None:
                raise ValueError("No processed data available")
                
            analysis_type = self.analysis_type.get()
        except Exception as e:
            self.logger.error(f"Failed to run analysis: {str(e)}")
            raise
            
            if analysis_type == 'statistical':
                # Perform statistical tests between clusters
                test = self.stat_test.get()
                results = []
                
                for column in self.normalized_data.select_dtypes(include=['float64', 'int64']):
