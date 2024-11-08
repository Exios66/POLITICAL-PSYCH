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
        normalize: bool = True
    ):
        """
        Initialize ClusterAnalysis with data file path and processing options.
        
        Args:
            file_path: Path to the data file
            output_dir: Optional path to output directory for results
            handle_missing: Whether to handle missing values
            remove_outliers: Whether to remove outliers
            normalize: Whether to normalize features
        """
        self.file_path = Path(file_path)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path('data/processed_data') / self.timestamp
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.handle_missing = handle_missing
        self.remove_outliers = remove_outliers
        self.normalize = normalize
        
        self.data = None
        self.cleaned_data = None
        self.normalized_data = None
        self.clusters = None
        self.n_clusters = None
        self.feature_importance = None
        self.cluster_profiles = None
        
        logger.info(f"ClusterAnalysis initialized with output directory: {self.output_dir}")

    def load_data(self) -> None:
        """
        Load data from the specified file path.
        
        Raises:
            FileNotFoundError: If the data file doesn't exist
            pd.errors.EmptyDataError: If the data file is empty
        """
        try:
            self.data = pd.read_csv(self.file_path)
            
            # Convert survey_date to datetime
            if 'survey_date' in self.data.columns:
                self.data['survey_date'] = pd.to_datetime(self.data['survey_date'])
            
            logger.info(f"Data loaded successfully from {self.file_path}")
            
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.file_path}")
            raise
            
        except pd.errors.EmptyDataError:
            logger.error(f"Data file is empty: {self.file_path}")
            raise
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def clean_data(self) -> None:
        """
        Clean the loaded data by handling missing values and outliers.
        
        Raises:
            ValueError: If data is None or empty
        """
        if self.data is None or self.data.empty:
            raise ValueError("No data to clean. Load data first.")
            
        try:
            # Create copy of raw data
            self.cleaned_data = self.data.copy()
            
            # Handle missing values
            if self.handle_missing:
                # For numeric columns, fill with median
                numeric_cols = self.cleaned_data.select_dtypes(
                    include=['float64', 'int64']
                ).columns
                self.cleaned_data[numeric_cols] = self.cleaned_data[numeric_cols].fillna(
                    self.cleaned_data[numeric_cols].median()
                )
                
                # For categorical columns, fill with mode
                cat_cols = self.cleaned_data.select_dtypes(
                    include=['object']
                ).columns
                self.cleaned_data[cat_cols] = self.cleaned_data[cat_cols].fillna(
                    self.cleaned_data[cat_cols].mode().iloc[0]
                )
            
            # Remove outliers using IQR method
            if self.remove_outliers:
                numeric_cols = self.cleaned_data.select_dtypes(
                    include=['float64', 'int64']
                ).columns
                
                Q1 = self.cleaned_data[numeric_cols].quantile(0.25)
                Q3 = self.cleaned_data[numeric_cols].quantile(0.75)
                IQR = Q3 - Q1
                
                outlier_mask = ~((self.cleaned_data[numeric_cols] < (Q1 - 1.5 * IQR)) | 
                                (self.cleaned_data[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
                
                self.cleaned_data = self.cleaned_data[outlier_mask]
            
            logger.info("Data cleaned successfully")
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise

    def normalize_features(self) -> None:
        """
        Normalize features using StandardScaler.
        
        Raises:
            ValueError: If cleaned_data is None or empty
        """
        if self.cleaned_data is None or self.cleaned_data.empty:
            raise ValueError("No cleaned data available. Clean data first.")
            
        try:
            numeric_cols = self.cleaned_data.select_dtypes(
                include=['float64', 'int64']
            ).columns
            
            scaler = StandardScaler()
            normalized = scaler.fit_transform(self.cleaned_data[numeric_cols])
            
            self.normalized_data = pd.DataFrame(
                normalized,
                columns=numeric_cols,
                index=self.cleaned_data.index
            )
            
            # Add back non-numeric columns
            for col in self.cleaned_data.columns:
                if col not in numeric_cols:
                    self.normalized_data[col] = self.cleaned_data[col]
            
            logger.info("Features normalized successfully")
            
        except Exception as e:
            logger.error(f"Error normalizing features: {str(e)}")
            raise

    def find_optimal_clusters(
        self,
        method: str = 'elbow',
        max_k: int = 10
    ) -> int:
        """
        Find optimal number of clusters using specified method.
        
        Args:
            method: Method to use ('elbow' or 'silhouette')
            max_k: Maximum number of clusters to try
            
        Returns:
            Optimal number of clusters
            
        Raises:
            ValueError: If normalized_data is None or empty
        """
        if self.normalized_data is None or self.normalized_data.empty:
            raise ValueError("No normalized data available")
            
        try:
            numeric_data = self.normalized_data.select_dtypes(
                include=['float64', 'int64']
            )
            
            if method == 'elbow':
                distortions = []
                K = range(1, max_k + 1)
                
                for k in K:
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    kmeans.fit(numeric_data)
                    distortions.append(kmeans.inertia_)
                    
                # Find elbow point
                diffs = np.diff(distortions)
                optimal_k = np.argmin(diffs) + 2
                
            elif method == 'silhouette':
                silhouette_scores = []
                K = range(2, max_k + 1)
                
                for k in K:
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    labels = kmeans.fit_predict(numeric_data)
                    silhouette_scores.append(
                        silhouette_score(numeric_data, labels)
                    )
                    
                optimal_k = K[np.argmax(silhouette_scores)]
                
            else:
                raise ValueError(f"Invalid method: {method}")
                
            logger.info(f"Optimal number of clusters found: {optimal_k}")
            return optimal_k
            
        except Exception as e:
            logger.error(f"Error finding optimal clusters: {str(e)}")
            raise

    def perform_clustering(
        self,
        method: str = 'kmeans',
        n_clusters: int = 3,
        **kwargs
    ) -> None:
        """
        Perform clustering using specified method.
        
        Args:
            method: Clustering method ('kmeans', 'dbscan', or 'hierarchical')
            n_clusters: Number of clusters
            **kwargs: Additional parameters for clustering algorithms
            
        Raises:
            ValueError: If normalized_data is None or empty
            ValueError: If invalid clustering method specified
        """
        if self.normalized_data is None or self.normalized_data.empty:
            raise ValueError("No normalized data available")
            
        try:
            numeric_data = self.normalized_data.select_dtypes(
                include=['float64', 'int64']
            )
            
            if method == 'kmeans':
                model = KMeans(n_clusters=n_clusters, random_state=42, **kwargs)
                self.clusters = model.fit_predict(numeric_data)
                self.n_clusters = n_clusters
                
                # Calculate feature importance
                self.feature_importance = pd.DataFrame({
                    'feature': numeric_data.columns,
                    'importance': np.abs(model.cluster_centers_).mean(axis=0)
                }).sort_values('importance', ascending=False)
                
            elif method == 'dbscan':
                model = DBSCAN(**kwargs)
                self.clusters = model.fit_predict(numeric_data)
                self.n_clusters = len(set(self.clusters)) - (1 if -1 in self.clusters else 0)
                
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
