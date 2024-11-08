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
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

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

class ClusterAnalysis:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None
        self.cleaned_data = None
        self.normalized_data = None
        self.clusters = None
        self.n_clusters = None
        self.model = None
        
    def load_data(self):
        """Load data from CSV file"""
        self.data = pd.read_csv(self.file_path)
        return self.data
        
    def clean_data(self):
        """Clean the loaded data"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        # Make a copy of the data
        self.cleaned_data = self.data.copy()
        
        # Remove datetime column (assuming it's the last column)
        datetime_col = self.cleaned_data.columns[-1]
        self.cleaned_data = self.cleaned_data.drop(columns=[datetime_col])
        
        # Convert all values to numeric, replacing non-numeric with NaN
        for col in self.cleaned_data.columns:
            self.cleaned_data[col] = pd.to_numeric(self.cleaned_data[col], errors='coerce')
        
        # Drop rows with missing values
        self.cleaned_data = self.cleaned_data.dropna()
        
        return self.cleaned_data
        
    def normalize_features(self):
        """Normalize the features using StandardScaler"""
        if self.cleaned_data is None:
            raise ValueError("Data not cleaned. Call clean_data() first.")
            
        scaler = StandardScaler()
        self.normalized_data = pd.DataFrame(
            scaler.fit_transform(self.cleaned_data),
            columns=self.cleaned_data.columns
        )
        return self.normalized_data
        
    def perform_clustering(self, method='kmeans', n_clusters=3, **kwargs):
        """Perform clustering with specified method"""
        if self.normalized_data is None:
            raise ValueError("Data not normalized. Call normalize_features() first.")
            
        self.n_clusters = n_clusters
        method = method.lower()
        
        if method == 'kmeans':
            self.model = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == 'hierarchical':
            self.model = AgglomerativeClustering(n_clusters=n_clusters)
        elif method == 'dbscan':
            self.model = DBSCAN(eps=kwargs.get('eps', 0.5), 
                              min_samples=kwargs.get('min_samples', 5))
        elif method == 'spectral':
            self.model = SpectralClustering(n_clusters=n_clusters, random_state=42)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
            
        self.clusters = self.model.fit_predict(self.normalized_data)
        return self.clusters
        
    def plot_cluster_distribution(self):
        """Plot cluster size distribution"""
        if self.clusters is None:
            raise ValueError("Clustering not performed yet")
            
        plt.figure(figsize=(10, 6))
        pd.Series(self.clusters).value_counts().sort_index().plot(kind='bar')
        plt.title('Cluster Size Distribution')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Samples')
        plt.tight_layout()
        return plt.gcf()
        
    def plot_feature_importance(self):
        """Plot feature importance heatmap"""
        if self.clusters is None:
            raise ValueError("Clustering not performed yet")
            
        feature_importance = []
        for i in range(self.n_clusters):
            cluster_data = self.cleaned_data[self.clusters == i]
            importance = self._calculate_feature_importance(cluster_data)
            feature_importance.append(importance)
            
        plt.figure(figsize=(12, 8))
        sns.heatmap(pd.DataFrame(feature_importance), 
                   cmap='RdBu', center=0, annot=True, fmt='.2f')
        plt.title('Feature Importance by Cluster')
        plt.tight_layout()
        return plt.gcf()
        
    def _calculate_feature_importance(self, cluster_data):
        """Calculate feature importance for a cluster"""
        overall_mean = self.cleaned_data.mean()
        overall_std = self.cleaned_data.std()
        cluster_mean = cluster_data.mean()
        
        importance = {}
        for feature in self.cleaned_data.columns:
            z_score = (cluster_mean[feature] - overall_mean[feature]) / overall_std[feature]
            importance[feature] = z_score
            
        return importance
        
    def plot_pca(self):
        """Plot PCA visualization"""
        if self.clusters is None:
            raise ValueError("Clustering not performed yet")
            
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self.normalized_data)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], c=self.clusters, cmap='viridis')
        plt.title('PCA Visualization of Clusters')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.colorbar(label='Cluster')
        plt.tight_layout()
        return plt.gcf()
        
    def plot_umap(self):
        """Plot UMAP visualization"""
        if self.clusters is None:
            raise ValueError("Clustering not performed yet")
            
        reducer = umap.UMAP(random_state=42)
        umap_result = reducer.fit_transform(self.normalized_data)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(umap_result[:, 0], umap_result[:, 1], c=self.clusters, cmap='viridis')
        plt.title('UMAP Visualization of Clusters')
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.colorbar(label='Cluster')
        plt.tight_layout()
        return plt.gcf()

class ClusteringGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Cluster Analysis Tool")
        self.root.geometry("800x600")
        self.analyzer = None
        self.results = None
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Initialize buttons as None
        self.plot_button = None
        self.export_button = None
        self.file_button = None
        self.run_button = None
        
        # Initialize GUI components
        self._create_menu()
        self._create_input_section()
        self._create_analysis_section()
        self._create_visualization_section()
        self._create_status_bar()
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

    def _run_clustering(self):
        """Run clustering analysis"""
        if not self.analyzer:
            messagebox.showwarning("Warning", "Please load data first!")
            return
            
        try:
            # Get parameters from GUI
            method = self.cluster_method.get().lower()
            n_clusters = int(self.n_clusters.get())
            
            # Prepare data
            self.analyzer.clean_data()
            self.analyzer.normalize_features()
            
            # Run clustering
            self.results = self.analyzer.perform_clustering(
                method=method,
                n_clusters=n_clusters
            )
            
            # Update status and show success message
            self.status_var.set("Clustering completed successfully")
            messagebox.showinfo("Success", "Clustering completed successfully!")
            
            # Enable visualization and export buttons
            self.plot_button.config(state='normal')
            self.export_button.config(state='normal')
            
        except Exception as e:
            error_msg = f"Error in clustering: {str(e)}"
            self.status_var.set(error_msg)
            messagebox.showerror("Error", error_msg)
            logging.error(error_msg)
            
    def setup_gui(self):
        """Setup GUI elements"""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # File selection
        ttk.Label(main_frame, text="Data File:").grid(row=0, column=0, sticky=tk.W)
        self.file_button = ttk.Button(main_frame, text="Select File", command=self.open_file)
        self.file_button.grid(row=0, column=1, sticky=tk.W)
        
        # Clustering method selection
        ttk.Label(main_frame, text="Clustering Method:").grid(row=1, column=0, sticky=tk.W)
        self.cluster_method = ttk.Combobox(main_frame, 
            values=["KMeans", "Hierarchical", "DBSCAN", "Spectral"])
        self.cluster_method.set("KMeans")
        self.cluster_method.grid(row=1, column=1, sticky=tk.W)
        
        # Number of clusters
        ttk.Label(main_frame, text="Number of Clusters:").grid(row=2, column=0, sticky=tk.W)
        self.n_clusters = ttk.Entry(main_frame)
        self.n_clusters.insert(0, "3")
        self.n_clusters.grid(row=2, column=1, sticky=tk.W)
        
        # Run button
        self.run_button = ttk.Button(main_frame, text="Run Clustering", 
                                   command=self._run_clustering)
        self.run_button.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Plot type selection
        ttk.Label(main_frame, text="Plot Type:").grid(row=4, column=0, sticky=tk.W)
        self.plot_type = ttk.Combobox(main_frame, 
            values=["Cluster Distribution", "PCA", "UMAP", "Feature Importance"])
        self.plot_type.set("Cluster Distribution")
        self.plot_type.grid(row=4, column=1, sticky=tk.W)
        
        # Plot button
        self.plot_button = ttk.Button(main_frame, text="Generate Plot", 
                                    command=self._generate_plot)
        self.plot_button.grid(row=5, column=0, columnspan=2, pady=5)
        self.plot_button.config(state='disabled')
        
        # Export button
        self.export_button = ttk.Button(main_frame, text="Export Results", 
                                      command=self._export_results)
        self.export_button.grid(row=6, column=0, columnspan=2, pady=5)
        self.export_button.config(state='disabled')
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(main_frame, textvariable=self.status_var)
        self.status_bar.grid(row=7, column=0, columnspan=2, pady=5)
        
    def open_file(self):
        """Open file dialog and load data"""
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.analyzer = ClusterAnalysis(file_path)
                self.analyzer.load_data()
                self.status_var.set(f"Loaded data from {file_path}")
                messagebox.showinfo("Success", "Data loaded successfully!")
            except Exception as e:
                error_msg = f"Error loading file: {str(e)}"
                self.status_var.set(error_msg)
                messagebox.showerror("Error", error_msg)
                logging.error(error_msg)

    def _generate_plot(self):
        """Generate and display selected plot type"""
        if not self.analyzer or self.analyzer.clusters is None:
            messagebox.showwarning("Warning", "Please run clustering first!")
            return
            
        try:
            plot_type = self.plot_type.get()
            fig = None
            
            if plot_type == "Cluster Distribution":
                fig = self.analyzer.plot_cluster_distribution()
            elif plot_type == "PCA":
                fig = self.analyzer.plot_pca()
            elif plot_type == "UMAP":
                fig = self.analyzer.plot_umap()
            elif plot_type == "Feature Importance":
                fig = self.analyzer.plot_feature_importance()
            
            if fig:
                # Create plots directory if it doesn't exist
                plots_dir = Path("results/plots")
                plots_dir.mkdir(parents=True, exist_ok=True)
                
                # Save plot
                plot_path = plots_dir / f"{plot_type.lower().replace(' ', '_')}.png"
                fig.savefig(plot_path)
                plt.close(fig)
                
                # Show success message
                self.status_var.set(f"Plot saved to {plot_path}")
                messagebox.showinfo("Success", f"Plot saved to:\n{plot_path}")
                
                # Open plot in default viewer
                if sys.platform == "win32":
                    os.startfile(plot_path)
                else:
                    import subprocess
                    subprocess.call(["open" if sys.platform == "darwin" else "xdg-open", plot_path])
                    
        except Exception as e:
            error_msg = f"Error generating plot: {str(e)}"
            self.status_var.set(error_msg)
            messagebox.showerror("Error", error_msg)
            logging.error(error_msg)

    def _generate_report(self):
        """Generate and save analysis report"""
        if not self.analyzer or self.analyzer.clusters is None:
            messagebox.showwarning("Warning", "Please run clustering first!")
            return
            
        try:
            # Create reports directory if it doesn't exist
            reports_dir = Path("results/reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp for unique filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = reports_dir / f"cluster_analysis_report_{timestamp}"
            
            # Generate HTML report
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Cluster Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .section {{ margin-bottom: 30px; }}
                    .metric {{ margin: 20px 0; }}
                    .visualization {{ margin: 30px 0; }}
                    img {{ max-width: 100%; }}
                </style>
            </head>
            <body>
                {self._generate_metadata_html()}
                {self._generate_metrics_html()}
                {self._generate_cluster_details_html()}
                {self._generate_visualizations_html()}
            </body>
            </html>
            """
            
            # Save HTML report
            with open(f"{report_path}.html", 'w') as f:
                f.write(html_content)
            
            # Generate and save PDF version if possible
            try:
                import pdfkit
                pdfkit.from_string(html_content, f"{report_path}.pdf")
            except Exception as pdf_error:
                logging.warning(f"Could not generate PDF: {pdf_error}")
            
            self.status_var.set("Report generated successfully")
            messagebox.showinfo("Success", f"Report saved to:\n{report_path}.html")
            
        except Exception as e:
            error_msg = f"Error generating report: {str(e)}"
            self.status_var.set(error_msg)
            messagebox.showerror("Error", error_msg)
            logging.error(error_msg)

    def _update_status(self, message: str, level: str = "info"):
        """Update status bar with message"""
        self.status_var.set(message)
        if level == "error":
            logging.error(message)
        else:
            logging.info(message)

    def _validate_inputs(self) -> bool:
        """Validate user inputs before running analysis"""
        try:
            n_clusters = int(self.n_clusters.get())
            if n_clusters < 2:
                raise ValueError("Number of clusters must be at least 2")
            return True
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))
            return False

    def _save_configuration(self):
        """Save current configuration to file"""
        config = {
            'clustering_method': self.cluster_method.get(),
            'n_clusters': self.n_clusters.get(),
            'plot_type': self.plot_type.get()
        }
        
        try:
            with open('config/last_session.json', 'w') as f:
                json.dump(config, f)
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")

    def _load_configuration(self):
        """Load last used configuration"""
        try:
            with open('config/last_session.json', 'r') as f:
                config = json.load(f)
                
            self.cluster_method.set(config.get('clustering_method', 'KMeans'))
            self.n_clusters.delete(0, tk.END)
            self.n_clusters.insert(0, config.get('n_clusters', '3'))
            self.plot_type.set(config.get('plot_type', 'Cluster Distribution'))
        except Exception as e:
            logging.warning(f"Could not load last configuration: {e}")

    def _create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Import Data", command=self._import_data)
        file_menu.add_command(label="Export Results", command=self._export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Run Clustering", command=self._run_clustering)
        analysis_menu.add_command(label="Generate Report", command=self._generate_report)

    def _create_input_section(self):
        """Create data input section"""
        input_frame = ttk.LabelFrame(self.main_frame, text="Data Input", padding="5")
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        ttk.Button(input_frame, text="Select Data File", command=self._import_data).grid(row=0, column=0, padx=5, pady=5)
        self.file_label = ttk.Label(input_frame, text="No file selected")
        self.file_label.grid(row=0, column=1, padx=5, pady=5)

    def _create_analysis_section(self):
        """Create analysis options section"""
        analysis_frame = ttk.LabelFrame(self.main_frame, text="Analysis Options", padding="5")
        analysis_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Clustering method selection
        ttk.Label(analysis_frame, text="Clustering Method:").grid(row=0, column=0, padx=5, pady=5)
        self.cluster_method = ttk.Combobox(analysis_frame, 
                                         values=["KMeans", "Hierarchical", "DBSCAN", "Spectral"])
        self.cluster_method.grid(row=0, column=1, padx=5, pady=5)
        self.cluster_method.set("KMeans")
        
        # Number of clusters
        ttk.Label(analysis_frame, text="Number of Clusters:").grid(row=1, column=0, padx=5, pady=5)
        self.n_clusters = ttk.Entry(analysis_frame)
        self.n_clusters.grid(row=1, column=1, padx=5, pady=5)
        self.n_clusters.insert(0, "3")
        
        # Run button
        ttk.Button(analysis_frame, text="Run Analysis", 
                  command=self._run_clustering).grid(row=2, column=0, columnspan=2, pady=10)

    def _create_visualization_section(self):
        """Create visualization options section"""
        viz_frame = ttk.LabelFrame(self.main_frame, text="Visualization", padding="5")
        viz_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Visualization type selection
        ttk.Label(viz_frame, text="Plot Type:").grid(row=0, column=0, padx=5, pady=5)
        self.plot_type = ttk.Combobox(viz_frame, 
                                    values=["Cluster Distribution", "PCA", "UMAP", "Feature Importance"])
        self.plot_type.grid(row=0, column=1, padx=5, pady=5)
        self.plot_type.set("Cluster Distribution")
        
        # Generate plot button - Store as instance variable
        self.plot_button = ttk.Button(viz_frame, text="Generate Plot", 
                                    command=self._generate_plot)
        self.plot_button.grid(row=1, column=0, columnspan=2, pady=10)
        self.plot_button.config(state='disabled')  # Initially disabled
        
        # Export button - Store as instance variable
        self.export_button = ttk.Button(viz_frame, text="Export Results", 
                                      command=self._export_results)
        self.export_button.grid(row=2, column=0, columnspan=2, pady=5)
        self.export_button.config(state='disabled')  # Initially disabled

    def _create_status_bar(self):
        """Create status bar"""
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))

    def _import_data(self):
        """Import data file"""
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if not file_path:  # User cancelled the dialog
                return
            
            if not os.path.exists(file_path):
                messagebox.showerror("Error", f"Selected file does not exist: {file_path}")
                return
            
            print(f"Selected file: {file_path}")  # Debug print
            self.analyzer = ClusterAnalysis(file_path)
            self.analyzer.load_data()
            self.file_label.config(text=os.path.basename(file_path))
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
            print(f"Error details: {e}")  # Debug print

    def _export_results(self):
        """Export clustering results with detailed interpretation in CSV and Markdown formats"""
        if not hasattr(self, 'results'):
            messagebox.showwarning("Warning", "Please run clustering first!")
            return
            
        try:
            # Create timestamped directory path
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_path = Path("data/processed_data")
            export_dir = base_path / timestamp
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Export formatted cluster assignments as CSV
            if self.analyzer.clusters is not None:
                results_df = self._format_cluster_results()
                results_df.to_csv(export_dir / 'cluster_assignments.csv', index=False)
                
                # Export summary statistics as CSV
                self._export_summary_statistics(results_df, export_dir)
                
            # 2. Generate and export cluster insights as Markdown
            insights = self._generate_cluster_insights()
            with open(export_dir / 'cluster_insights.md', 'w') as f:
                f.write(insights)
                
            # 3. Export visualization with annotations
            self._export_visualizations(export_dir)
                
            # 4. Generate detailed markdown report
            self._generate_markdown_report(export_dir)
            
            self.status_var.set(f"Results exported to {export_dir}")
            messagebox.showinfo("Success", f"Results exported to:\n{export_dir}")
            
        except Exception as e:
            error_msg = f"Error exporting results: {str(e)}"
            logging.error(error_msg)
            self.status_var.set(error_msg)
            messagebox.showerror("Error", error_msg)

    def _export_summary_statistics(self, df, export_dir):
        """Export summary statistics for each cluster"""
        # Cluster-wise statistics
        cluster_stats = df.groupby('Cluster').describe()
        cluster_stats.to_csv(export_dir / 'cluster_statistics.csv')
        
        # Feature importance by cluster
        feature_importance_df = pd.DataFrame([
            self._calculate_feature_importance(
                self.analyzer.cleaned_data[self.analyzer.clusters == i]
            )
            for i in range(self.analyzer.n_clusters)
        ])
        feature_importance_df.to_csv(export_dir / 'feature_importance.csv')

    def _generate_markdown_report(self, export_dir):
        """Generate comprehensive Markdown report"""
        report = f"""# Cluster Analysis Report
Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Analysis Overview
- Clustering Method: {self.cluster_method.get()}
- Number of Clusters: {self.analyzer.n_clusters}
- Total Samples: {len(self.analyzer.clusters)}

## Clustering Metrics
"""
        
        try:
            silhouette = silhouette_score(self.analyzer.normalized_data, self.analyzer.clusters)
            davies_bouldin = davies_bouldin_score(self.analyzer.normalized_data, self.analyzer.clusters)
            calinski_harabasz = calinski_harabasz_score(self.analyzer.normalized_data, self.analyzer.clusters)
            
            report += f"""
- Silhouette Score: {silhouette:.3f}
  - Measures cluster cohesion and separation (range: [-1, 1], higher is better)
- Davies-Bouldin Score: {davies_bouldin:.3f}
  - Measures average similarity between clusters (lower is better)
- Calinski-Harabasz Score: {calinski_harabasz:.3f}
  - Measures cluster density and separation (higher is better)

"""
        except Exception as e:
            report += f"\nError calculating metrics: {str(e)}\n\n"

        # Add cluster details
        report += "## Cluster Details\n\n"
        for i in range(self.analyzer.n_clusters):
            cluster_data = self.analyzer.cleaned_data[self.analyzer.clusters == i]
            feature_importance = self._calculate_feature_importance(cluster_data)
            
            report += f"""### Cluster {i}
- Size: {len(cluster_data)} samples ({len(cluster_data)/len(self.analyzer.clusters)*100:.1f}%)
- Key Characteristics:
"""
            
            # Add top features
            for feature, importance in sorted(feature_importance.items(), 
                                           key=lambda x: abs(x[1]), 
                                           reverse=True)[:5]:
                report += f"  - {feature}: {importance:+.2f} standard deviations "
                report += f"({'above' if importance > 0 else 'below'} average)\n"
                
            report += "\n"

        # Add visualization references
        report += """## Visualizations
The following visualizations are available in the 'visualizations' directory:
1. cluster_distribution.png - Shows the size distribution of clusters
2. feature_importance.png - Heatmap showing feature importance by cluster

## Data Files
The following CSV files contain detailed analysis results:
1. cluster_assignments.csv - Original data with cluster assignments
2. cluster_statistics.csv - Statistical summary for each cluster
3. feature_importance.csv - Importance scores for features in each cluster
"""

        # Save report
        with open(export_dir / 'detailed_report.md', 'w') as f:
            f.write(report)

    def _format_cluster_results(self):
        """Format cluster results with normalized and readable values"""
        results_df = self.analyzer.cleaned_data.copy()
        
        # Add cluster assignments
        results_df['Cluster'] = self.analyzer.clusters
        
        # Calculate normalized values
        scaler = MinMaxScaler()
        normalized_values = scaler.fit_transform(results_df.drop('Cluster', axis=1))
        normalized_df = pd.DataFrame(
            normalized_values,
            columns=[f"{col}_normalized" for col in results_df.drop('Cluster', axis=1).columns]
        )
        
        # Combine original and normalized values
        final_df = pd.concat([results_df, normalized_df], axis=1)
        
        # Add cluster descriptions
        cluster_descriptions = self._generate_cluster_descriptions()
        final_df['Cluster_Description'] = final_df['Cluster'].map(cluster_descriptions)
        
        return final_df

    def _generate_cluster_insights(self):
        """Generate detailed insights about the clustering results"""
        insights = f"""# Cluster Analysis Results
Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overview
- Number of clusters: {self.analyzer.n_clusters}
- Clustering method: {self.cluster_method.get()}
- Total samples: {len(self.analyzer.clusters)}

## Cluster Characteristics
"""
        
        for i in range(self.analyzer.n_clusters):
            cluster_data = self.analyzer.cleaned_data[self.analyzer.clusters == i]
            insights += f"\n### Cluster {i}\n"
            insights += f"- Size: {len(cluster_data)} samples ({len(cluster_data)/len(self.analyzer.clusters)*100:.1f}%)\n"
            insights += "- Key features:\n"
            
            # Add feature importance for this cluster
            feature_importance = self._calculate_feature_importance(cluster_data)
            for feature, importance in feature_importance.items():
                insights += f"  - {feature}: {importance:.2f}\n"
                
        return insights

    def _generate_metadata_html(self):
        """Generate HTML for metadata section"""
        return f"""
        <div class="metadata">
            <h2>Analysis Metadata</h2>
            <ul>
                <li>Analysis Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</li>
                <li>Clustering Method: {self.cluster_method.get()}</li>
                <li>Number of Clusters: {self.analyzer.n_clusters}</li>
                <li>Total Samples: {len(self.analyzer.clusters)}</li>
            </ul>
        </div>
        """

    def _generate_metrics_html(self):
        """Generate HTML for metrics section"""
        try:
            silhouette = silhouette_score(self.analyzer.normalized_data, self.analyzer.clusters)
            davies_bouldin = davies_bouldin_score(self.analyzer.normalized_data, self.analyzer.clusters)
            calinski_harabasz = calinski_harabasz_score(self.analyzer.normalized_data, self.analyzer.clusters)
            
            return f"""
            <div class="metrics">
                <h2>Clustering Metrics</h2>
                <div class="metric">
                    <h3>Silhouette Score: {silhouette:.3f}</h3>
                    <p>Measures how similar an object is to its own cluster compared to other clusters.
                       Range: [-1, 1], higher is better.</p>
                </div>
                <div class="metric">
                    <h3>Davies-Bouldin Score: {davies_bouldin:.3f}</h3>
                    <p>Measures average similarity between clusters. Lower is better.</p>
                </div>
                <div class="metric">
                    <h3>Calinski-Harabasz Score: {calinski_harabasz:.3f}</h3>
                    <p>Ratio of between-cluster variance to within-cluster variance. Higher is better.</p>
                </div>
            </div>
            """
        except Exception as e:
            logging.error(f"Error calculating metrics: {str(e)}")
            return "<div class='metrics'><p>Error calculating metrics</p></div>"

    def _generate_cluster_details_html(self):
        """Generate HTML for cluster details section"""
        details = "<h2>Cluster Details</h2>"
        
        for i in range(self.analyzer.n_clusters):
            cluster_data = self.analyzer.cleaned_data[self.analyzer.clusters == i]
            feature_importance = self._calculate_feature_importance(cluster_data)
            
            details += f"""
            <div class="cluster-section">
                <h3>Cluster {i}</h3>
                <p>Size: {len(cluster_data)} samples 
                   ({len(cluster_data)/len(self.analyzer.clusters)*100:.1f}%)</p>
                <h4>Key Characteristics:</h4>
                <ul>
            """
            
            # Add feature importance details
            for feature, importance in sorted(feature_importance.items(), 
                                           key=lambda x: abs(x[1]), 
                                           reverse=True)[:5]:
                details += f"""
                    <li>{feature}: {importance:+.2f} standard deviations from mean
                        ({'above' if importance > 0 else 'below'} average)</li>
                """
                
            details += "</ul></div>"
            
        return details

    def _generate_visualizations_html(self):
        """Generate HTML for visualizations section"""
        return f"""
        <div class="visualizations">
            <h2>Visualizations</h2>
            <div class="visualization">
                <h3>Cluster Distribution</h3>
                <img src="visualizations/cluster_distribution.png" alt="Cluster Distribution">
            </div>
            <div class="visualization">
                <h3>Feature Importance</h3>
                <img src="visualizations/feature_importance.png" alt="Feature Importance">
            </div>
        </div>
        """

    def _export_visualizations(self, export_dir):
        """Export all visualizations"""
        viz_dir = export_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # Create and save cluster distribution plot
        plt.figure(figsize=(10, 6))
        cluster_sizes = pd.Series(self.analyzer.clusters).value_counts().sort_index()
        cluster_sizes.plot(kind='bar')
        plt.title('Cluster Size Distribution')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Samples')
        plt.tight_layout()
        plt.savefig(viz_dir / 'cluster_distribution.png')
        plt.close()
        
        # Create and save feature importance plot
        plt.figure(figsize=(12, 8))
        feature_importance = pd.DataFrame([
            self._calculate_feature_importance(
                self.analyzer.cleaned_data[self.analyzer.clusters == i]
            )
            for i in range(self.analyzer.n_clusters)
        ])
        
        sns.heatmap(feature_importance, cmap='RdBu', center=0, annot=True, fmt='.2f')
        plt.title('Feature Importance by Cluster')
        plt.tight_layout()
        plt.savefig(viz_dir / 'feature_importance.png')
        plt.close()

    def _generate_cluster_descriptions(self) -> Dict[int, str]:
        """Generate descriptive labels for each cluster based on their characteristics"""
        descriptions = {}
        
        for cluster_id in range(self.analyzer.n_clusters):
            # Get data for this cluster
            cluster_mask = self.analyzer.clusters == cluster_id
            cluster_data = self.analyzer.cleaned_data[cluster_mask]
            
            # Calculate feature importance for this cluster
            feature_importance = self._calculate_feature_importance(cluster_data)
            
            # Sort features by absolute importance
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            # Get top 3 most distinctive features
            top_features = sorted_features[:3]
            
            # Generate description based on top features
            description_parts = []
            for feature, importance in top_features:
                if abs(importance) > 0.5:  # Only include significant features
                    direction = "high" if importance > 0 else "low"
                    feature_name = feature.replace('_', ' ').title()
                    description_parts.append(f"{direction} {feature_name}")
            
            # Create final description
            if description_parts:
                description = f"Cluster {cluster_id}: " + ", ".join(description_parts)
                
                # Add size information
                cluster_size = sum(cluster_mask)
                size_percentage = (cluster_size / len(self.analyzer.clusters)) * 100
                description += f" ({size_percentage:.1f}% of samples)"
            else:
                description = f"Cluster {cluster_id}: No distinctive features"
            
            descriptions[cluster_id] = description
            
            # Log the descriptions
            logging.info(f"Generated description for {description}")
        
        return descriptions

    def _calculate_feature_importance(self, cluster_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature importance scores for a cluster"""
        if self.analyzer is None or self.analyzer.cleaned_data is None:
            raise ValueError("Analyzer or cleaned data not initialized")
            
        importance_scores = {}
        
        # Calculate overall statistics
        overall_mean = self.analyzer.cleaned_data.mean()
        overall_std = self.analyzer.cleaned_data.std()
        
        # Calculate cluster statistics
        cluster_mean = cluster_data.mean()
        
        # Calculate normalized difference from overall mean
        for feature in self.analyzer.cleaned_data.columns:
            if overall_std[feature] != 0:
                z_score = (cluster_mean[feature] - overall_mean[feature]) / overall_std[feature]
                importance_scores[feature] = float(z_score)  # Convert numpy float to Python float
            else:
                importance_scores[feature] = 0.0
        
        return importance_scores

    def _get_cluster_profile(self, cluster_id: int) -> Dict[str, Any]:
        """Generate detailed profile for a specific cluster"""
        if self.analyzer is None or self.analyzer.clusters is None:
            raise ValueError("Clustering not performed yet")
            
        cluster_mask = self.analyzer.clusters == cluster_id
        cluster_data = self.analyzer.cleaned_data[cluster_mask]
        
        profile = {
            'size': int(sum(cluster_mask)),
            'percentage': float(sum(cluster_mask) / len(self.analyzer.clusters) * 100),
            'feature_stats': {},
            'distinctive_features': []
        }
        
        # Calculate feature statistics
        for column in self.analyzer.cleaned_data.columns:
            profile['feature_stats'][column] = {
                'mean': float(cluster_data[column].mean()),
                'std': float(cluster_data[column].std()),
                'median': float(cluster_data[column].median()),
                'min': float(cluster_data[column].min()),
                'max': float(cluster_data[column].max())
            }
        
        # Calculate feature importance
        importance_scores = self._calculate_feature_importance(cluster_data)
        
        # Identify distinctive features (those with absolute z-score > 0.5)
        distinctive_features = [
            {
                'feature': feature,
                'importance': score,
                'direction': 'high' if score > 0 else 'low'
            }
            for feature, score in importance_scores.items()
            if abs(score) > 0.5
        ]
        
        profile['distinctive_features'] = sorted(
            distinctive_features,
            key=lambda x: abs(x['importance']),
            reverse=True
        )
        
        return profile

    def _generate_cluster_summary(self) -> str:
        """Generate a text summary of all clusters"""
        if self.analyzer is None or self.analyzer.clusters is None:
            raise ValueError("Clustering not performed yet")
            
        summary = ["Cluster Analysis Summary", "=" * 30, ""]
        
        for cluster_id in range(self.analyzer.n_clusters):
            profile = self._get_cluster_profile(cluster_id)
            
            summary.append(f"\nCluster {cluster_id}:")
            summary.append("-" * 20)
            summary.append(f"Size: {profile['size']} samples ({profile['percentage']:.1f}%)")
            
            if profile['distinctive_features']:
                summary.append("\nDistinctive Features:")
                for feature in profile['distinctive_features'][:5]:  # Top 5 features
                    direction = "higher" if feature['importance'] > 0 else "lower"
                    summary.append(
                        f"- {feature['feature']}: {abs(feature['importance']):.2f} std. dev. {direction} than average"
                    )
            else:
                summary.append("\nNo highly distinctive features found.")
            
            summary.append("")  # Empty line between clusters
        
        return "\n".join(summary)

def main():
    root = tk.Tk()
    app = ClusteringGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()