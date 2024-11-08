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
        
        # Remove duplicates
        self.cleaned_data = self.cleaned_data.drop_duplicates()
        
        return self.cleaned_data
        
    def normalize_features(self):
        """Normalize the features using StandardScaler"""
        if self.cleaned_data is None:
            raise ValueError("Data not cleaned. Call clean_data() first.")
            
        # Ensure all data is numeric
        numeric_data = self.cleaned_data.select_dtypes(include=[np.number])
        
        scaler = StandardScaler()
        self.normalized_data = pd.DataFrame(
            scaler.fit_transform(numeric_data),
            columns=numeric_data.columns
        )
        return self.normalized_data
        
    def perform_clustering(self, method='kmeans', n_clusters=3, **kwargs):
        """Perform clustering with specified method
    
        Args:
            method (str): Clustering method ('kmeans', 'dbscan', 'hierarchical', 'spectral')
            n_clusters (int): Number of clusters (for applicable methods)
            **kwargs: Additional parameters for specific clustering methods
        """
        if self.normalized_data is None:
            raise ValueError("Data not normalized. Call normalize_features() first.")
            
        self.n_clusters = n_clusters
        method = method.lower()
        
        try:
            if method == 'kmeans':
                self.model = KMeans(n_clusters=n_clusters, random_state=42)
            elif method == 'dbscan':
                self.model = DBSCAN(eps=kwargs.get('eps', 0.5), 
                                  min_samples=kwargs.get('min_samples', 5))
            elif method == 'hierarchical':
                self.model = AgglomerativeClustering(n_clusters=n_clusters)
            elif method == 'spectral':
                self.model = SpectralClustering(n_clusters=n_clusters, random_state=42)
            else:
                raise ValueError(f"Unsupported clustering method: {method}")
                
            self.clusters = self.model.fit_predict(self.normalized_data)
            return self.clusters
            
        except Exception as e:
            logging.error(f"Clustering failed: {str(e)}")
            raise
        
    def get_cluster_centers(self):
        """Get cluster centers"""
        if self.model is None:
            raise ValueError("Clustering not performed. Call perform_clustering() first.")
        return self.model.cluster_centers_
        
    def get_cluster_labels(self):
        """Get cluster labels"""
        if self.clusters is None:
            raise ValueError("Clustering not performed. Call perform_clustering() first.")
        return self.clusters
        
    def calculate_silhouette_score(self):
        """Calculate silhouette score"""
        if self.clusters is None:
            raise ValueError("Clustering not performed. Call perform_clustering() first.")
        return silhouette_score(self.normalized_data, self.clusters)
        
    def plot_clusters(self, feature1, feature2):
        """Plot clusters for two selected features"""
        if self.clusters is None:
            raise ValueError("Clustering not performed. Call perform_clustering() first.")
            
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            self.normalized_data[feature1],
            self.normalized_data[feature2],
            c=self.clusters,
            cmap='viridis'
        )
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.title(f'Cluster Analysis: {feature1} vs {feature2}')
        plt.colorbar(scatter)
        return plt.gcf()
        
    def get_cluster_summary(self):
        """Get summary statistics for each cluster"""
        if self.clusters is None:
            raise ValueError("Clustering not performed. Call perform_clustering() first.")
            
        summary = {}
        for i in range(self.n_clusters):
            cluster_data = self.cleaned_data[self.clusters == i]
            summary[f'Cluster_{i}'] = {
                'size': len(cluster_data),
                'mean': cluster_data.mean(),
                'std': cluster_data.std()
            }
        return summary

class ClusteringGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Cluster Analysis Tool")
        self.root.geometry("800x600")
        
        # Initialize analyzer as None
        self.analyzer = None
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create GUI elements
        self._create_menu()
        self._create_input_section()
        self._create_analysis_section()
        self._create_visualization_section()
        self._create_status_bar()

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
        
        # Generate plot button
        ttk.Button(viz_frame, text="Generate Plot", 
                  command=self._generate_plot).grid(row=1, column=0, columnspan=2, pady=10)

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
        """Export clustering results with detailed interpretation and formatted output"""
        if not hasattr(self, 'results'):
            messagebox.showwarning("Warning", "Please run clustering first!")
            return
            
        try:
            # Create timestamped directory path
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_path = Path("data/processed_data")
            export_dir = base_path / timestamp
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Export formatted cluster assignments
            if self.analyzer.clusters is not None:
                results_df = self._format_cluster_results()
                results_df.to_csv(export_dir / 'cluster_assignments.csv', index=False)
                
                # Export Excel version with formatting
                self._export_excel_report(results_df, export_dir / 'detailed_results.xlsx')
                
            # 2. Generate and export cluster insights
            insights = self._generate_cluster_insights()
            with open(export_dir / 'cluster_insights.md', 'w') as f:
                f.write(insights)
                
            # 3. Export visualization with annotations
            self._export_visualizations(export_dir)
                
            # 4. Generate HTML report
            self._generate_html_report(export_dir)
            
            self.status_var.set(f"Results exported to {export_dir}")
            messagebox.showinfo("Success", f"Results exported to:\n{export_dir}")
            
        except Exception as e:
            error_msg = f"Error exporting results: {str(e)}"
            logging.error(error_msg)
            self.status_var.set(error_msg)
            messagebox.showerror("Error", error_msg)

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

    def _export_excel_report(self, df, filepath):
        """Export formatted Excel report"""
        writer = pd.ExcelWriter(filepath, engine='xlsxwriter')
        df.to_excel(writer, sheet_name='Cluster Results', index=False)
        
        workbook = writer.book
        worksheet = writer.sheets['Cluster Results']
        
        # Add formatting
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'bg_color': '#D9E1F2',
            'border': 1
        })
        
        # Format headers
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
            
        # Add conditional formatting for clusters
        worksheet.conditional_format(1, df.columns.get_loc('Cluster'), 
                                   len(df), df.columns.get_loc('Cluster'),
                                   {'type': '3_color_scale'})
        
        writer.close()

    def _generate_html_report(self, export_dir):
        """Generate comprehensive HTML report"""
        template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cluster Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 1200px; margin: auto; }
                .metric { margin: 20px 0; padding: 10px; background: #f5f5f5; }
                .cluster-section { margin: 30px 0; }
                .visualization { margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Cluster Analysis Report</h1>
                <div class="metadata">
                    {metadata}
                </div>
                <div class="metrics">
                    {metrics}
                </div>
                <div class="cluster-details">
                    {cluster_details}
                </div>
                <div class="visualizations">
                    {visualizations}
                </div>
            </div>
        </body>
        </html>
        """
        
        # Generate report sections
        metadata = self._generate_metadata_html()
        metrics = self._generate_metrics_html()
        cluster_details = self._generate_cluster_details_html()
        visualizations = self._generate_visualizations_html()
        
        # Compile report
        report = template.format(
            metadata=metadata,
            metrics=metrics,
            cluster_details=cluster_details,
            visualizations=visualizations
        )
        
        # Save report
        with open(export_dir / 'cluster_analysis_report.html', 'w') as f:
            f.write(report)

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
            
            # Run clustering with specified method
            self.results = self.analyzer.perform_clustering(
                method=method,
                n_clusters=n_clusters
            )
            
            self.status_var.set("Clustering completed successfully")
            messagebox.showinfo("Success", "Clustering completed successfully!")
        except Exception as e:
            self.status_var.set(f"Error in clustering: {str(e)}")
            messagebox.showerror("Error", f"Error in clustering: {str(e)}")

    def _generate_plot(self):
        """Generate visualization"""
        if not hasattr(self, 'results'):
            messagebox.showwarning("Warning", "Please run clustering first!")
            return
            
        try:
            plot_type = self.plot_type.get()
            
            if plot_type == "Cluster Distribution":
                self.analyzer.plot_cluster_distribution()
            elif plot_type == "PCA":
                self.analyzer.plot_pca()
            elif plot_type == "UMAP":
                self.analyzer.plot_umap()
            elif plot_type == "Feature Importance":
                self.analyzer.plot_feature_importance()
                
            self.status_var.set(f"Generated {plot_type} plot")
            messagebox.showinfo("Success", f"Generated {plot_type} plot!")
        except Exception as e:
            self.status_var.set(f"Error generating plot: {str(e)}")
            messagebox.showerror("Error", f"Error generating plot: {str(e)}")

    def _generate_report(self):
        """Generate analysis report"""
        if not hasattr(self, 'results'):
            messagebox.showwarning("Warning", "Please run clustering first!")
            return
            
        try:
            report = self.analyzer.generate_report()
            
            file_path = filedialog.asksaveasfilename(
                defaultextension=".html",
                filetypes=[("HTML files", "*.html"), ("All files", "*.*")]
            )
            
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(report)
                self.status_var.set("Report generated successfully")
                messagebox.showinfo("Success", "Report generated successfully!")
        except Exception as e:
            self.status_var.set(f"Error generating report: {str(e)}")
            messagebox.showerror("Error", f"Error generating report: {str(e)}")

def main():
    root = tk.Tk()
    app = ClusteringGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()