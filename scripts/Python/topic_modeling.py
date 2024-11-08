import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import logging
import re
import string
import warnings
import json
import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

import gensim
from gensim import corpora, models
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.phrases import Phrases, Phraser

import pyLDAvis
import pyLDAvis.gensim_models
import pyLDAvis.sklearn # type: ignore

import spacy
from textblob import TextBlob
from wordcloud import WordCloud

from topic_modeling_pipeline import TopicModelingPipeline

# Configure logging with more detailed formatting and file rotation
LOG_FILENAME = 'logs/topic_modeling.log'
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.handlers.RotatingFileHandler(
            LOG_FILENAME, 
            maxBytes=1024*1024,  # 1MB
            backupCount=5
        )
    ]
)

logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Download required NLTK resources with error handling and retry
NLTK_RESOURCES = [
    'punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger',
    'maxent_ne_chunker', 'words', 'omw-1.4'
]

def download_nltk_resources(max_retries: int = 3) -> None:
    """Download NLTK resources with retry mechanism"""
    for resource in NLTK_RESOURCES:
        for attempt in range(max_retries):
            try:
                nltk.download(resource, quiet=True)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to download NLTK resource {resource} after {max_retries} attempts: {str(e)}")
                else:
                    logger.warning(f"Attempt {attempt + 1} failed for {resource}, retrying...")
                    continue

download_nltk_resources()

# Load spaCy model with error handling and download if needed
def load_spacy_model(model_name: str = 'en_core_web_sm') -> spacy.language.Language:
    """Load spaCy model with automatic download if missing"""
    try:
        return spacy.load(model_name)
    except OSError:
        logger.info(f"Downloading spaCy model {model_name}...")
        try:
            os.system(f'python -m spacy download {model_name}')
            return spacy.load(model_name)
        except Exception as e:
            logger.error(f"Failed to download spaCy model: {str(e)}")
            raise

try:
    nlp = load_spacy_model()
except Exception as e:
    logger.critical(f"Could not initialize spaCy model: {str(e)}")
    sys.exit(1)

class TopicModelingGUI:
    """GUI interface for topic modeling analysis with enhanced error handling and validation"""
    
    def __init__(self):
        """Initialize the GUI with error handling"""
        try:
            self.root = tk.Tk()
            self.root.title("Topic Modeling Analysis")
            self.root.geometry("800x600")
            
            # Set minimum window size
            self.root.minsize(600, 400)
            
            # Configure grid weight
            self.root.grid_rowconfigure(0, weight=1)
            self.root.grid_columnconfigure(0, weight=1)
            
            # Create main frame with proper scaling
            self.main_frame = ttk.Frame(self.root, padding="10")
            self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            self.main_frame.grid_columnconfigure(0, weight=1)
            
            # Initialize pipeline and other attributes
            self.pipeline: Optional[TopicModelingPipeline] = None
            self.current_file: Optional[str] = None
            self.analysis_running = False
            
            # Create GUI sections
            self._create_menu()
            self._create_input_section()
            self._create_analysis_section()
            self._create_visualization_section()
            self._create_status_bar()
            
            # Load configuration
            self._load_configuration()
            
            # Set up autosave
            self._setup_autosave()
            
        except Exception as e:
            logger.critical(f"Failed to initialize GUI: {str(e)}")
            raise

    def _create_menu(self):
        """Create enhanced menu bar with additional options and shortcuts"""
        try:
            menubar = tk.Menu(self.root)
            self.root.config(menu=menubar)
            
            # File menu with shortcuts
            file_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="File", menu=file_menu)
            file_menu.add_command(label="Import Data", command=self._import_data, 
                                accelerator="Ctrl+O")
            file_menu.add_command(label="Export Results", command=self._export_results,
                                accelerator="Ctrl+S")
            file_menu.add_separator()
            file_menu.add_command(label="Settings", command=self._show_settings)
            file_menu.add_separator()
            file_menu.add_command(label="Exit", command=self._on_closing,
                                accelerator="Ctrl+Q")
            
            # Analysis menu
            analysis_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Analysis", menu=analysis_menu)
            analysis_menu.add_command(label="Run Topic Modeling", 
                                    command=self._run_analysis,
                                    accelerator="F5")
            analysis_menu.add_command(label="Generate Report",
                                    command=self._generate_report,
                                    accelerator="Ctrl+R")
            
            # Help menu
            help_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Help", menu=help_menu)
            help_menu.add_command(label="Documentation", 
                                command=self._show_documentation)
            help_menu.add_command(label="About", command=self._show_about)
            
            # Bind shortcuts
            self.root.bind('<Control-o>', lambda e: self._import_data())
            self.root.bind('<Control-s>', lambda e: self._export_results())
            self.root.bind('<Control-q>', lambda e: self._on_closing())
            self.root.bind('<F5>', lambda e: self._run_analysis())
            self.root.bind('<Control-r>', lambda e: self._generate_report())
            
        except Exception as e:
            logger.error(f"Failed to create menu: {str(e)}")
            raise

    def _create_input_section(self):
        """Create enhanced data input section with validation"""
        try:
            input_frame = ttk.LabelFrame(self.main_frame, text="Data Input", padding="5")
            input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), 
                           padx=5, pady=5)
            input_frame.grid_columnconfigure(1, weight=1)
            
            # File selection with drag-and-drop support
            ttk.Button(input_frame, text="Select Data File", 
                      command=self._import_data).grid(row=0, column=0, 
                                                    padx=5, pady=5)
            self.file_label = ttk.Label(input_frame, text="No file selected")
            self.file_label.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
            
            # Text column selection with validation
            ttk.Label(input_frame, text="Text Columns:").grid(row=1, column=0,
                                                            padx=5, pady=5)
            self.text_columns = ttk.Entry(input_frame)
            self.text_columns.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
            self.text_columns.insert(0, "open_ended_response_1,open_ended_response_2")
            
            # Add validation tooltip
            self._create_tooltip(self.text_columns, 
                               "Enter comma-separated column names")
            
            # Preview data button
            ttk.Button(input_frame, text="Preview Data",
                      command=self._preview_data).grid(row=2, column=0,
                                                     columnspan=2, pady=5)
            
        except Exception as e:
            logger.error(f"Failed to create input section: {str(e)}")
            raise

    def _create_analysis_section(self):
        """Create enhanced analysis options section with validation"""
        try:
            analysis_frame = ttk.LabelFrame(self.main_frame, text="Analysis Options",
                                          padding="5")
            analysis_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S),
                              padx=5, pady=5)
            analysis_frame.grid_columnconfigure(1, weight=1)
            
            # Model selection with tooltips
            ttk.Label(analysis_frame, text="Topic Model:").grid(row=0, column=0,
                                                              padx=5, pady=5)
            self.model_type = ttk.Combobox(analysis_frame,
                                         values=["LDA", "NMF", "HDP", "LSI"])
            self.model_type.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
            self.model_type.set("LDA")
            
            # Add model selection tooltip
            self._create_tooltip(self.model_type,
                               "Select topic modeling algorithm")
            
            # Topics range with validation
            ttk.Label(analysis_frame, text="Topics Range (min-max):").grid(
                row=1, column=0, padx=5, pady=5)
            self.topics_range = ttk.Entry(analysis_frame)
            self.topics_range.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
            self.topics_range.insert(0, "2-15")
            
            # Validate topics range input
            vcmd = (self.root.register(self._validate_topics_range), '%P')
            self.topics_range.config(validate='key', validatecommand=vcmd)
            
            # Advanced options button
            ttk.Button(analysis_frame, text="Advanced Options",
                      command=self._show_advanced_options).grid(
                          row=2, column=0, columnspan=2, pady=5)
            
            # Run button with progress indicator
            self.run_button = ttk.Button(analysis_frame, text="Run Analysis",
                                       command=self._run_analysis)
            self.run_button.grid(row=3, column=0, columnspan=2, pady=10)
            
            # Progress bar
            self.progress_var = tk.DoubleVar()
            self.progress_bar = ttk.Progressbar(analysis_frame, 
                                              variable=self.progress_var,
                                              maximum=100)
            self.progress_bar.grid(row=4, column=0, columnspan=2,
                                 sticky=tk.W+tk.E, padx=5)
            
        except Exception as e:
            logger.error(f"Failed to create analysis section: {str(e)}")
            raise

    def _create_visualization_section(self):
        """Create enhanced visualization options section"""
        try:
            viz_frame = ttk.LabelFrame(self.main_frame, text="Visualization",
                                     padding="5")
            viz_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S),
                         padx=5, pady=5)
            viz_frame.grid_columnconfigure(1, weight=1)
            
            # Visualization type selection with preview
            ttk.Label(viz_frame, text="Plot Type:").grid(row=0, column=0,
                                                       padx=5, pady=5)
            self.plot_type = ttk.Combobox(viz_frame,
                                        values=["Topic Distribution",
                                               "Word Clouds",
                                               "Coherence Scores",
                                               "Interactive Topics",
                                               "Topic Evolution",
                                               "Document Clustering"])
            self.plot_type.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
            self.plot_type.set("Topic Distribution")
            
            # Plot options frame
            self.plot_options_frame = ttk.Frame(viz_frame)
            self.plot_options_frame.grid(row=1, column=0, columnspan=2,
                                       sticky=tk.W+tk.E)
            
            # Bind plot type change
            self.plot_type.bind('<<ComboboxSelected>>', self._update_plot_options)
            
            # Generate plot button with error handling
            self.plot_button = ttk.Button(viz_frame, text="Generate Plot",
                                        command=self._generate_plot)
            self.plot_button.grid(row=2, column=0, columnspan=2, pady=10)
            self.plot_button.config(state='disabled')
            
            # Export options
            export_frame = ttk.Frame(viz_frame)
            export_frame.grid(row=3, column=0, columnspan=2, sticky=tk.W+tk.E)
            
            self.export_button = ttk.Button(export_frame, text="Export Results",
                                          command=self._export_results)
            self.export_button.pack(side=tk.LEFT, padx=5)
            self.export_button.config(state='disabled')
            
            self.export_format = ttk.Combobox(export_frame,
                                            values=["CSV", "JSON", "Excel", "HTML"])
            self.export_format.pack(side=tk.RIGHT, padx=5)
            self.export_format.set("CSV")
            
        except Exception as e:
            logger.error(f"Failed to create visualization section: {str(e)}")
            raise

    def _create_status_bar(self):
        """Create status bar for user feedback"""
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var,
                                  relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=1, column=0, sticky=tk.W+tk.E)
        self.status_var.set("Ready")

    def _setup_autosave(self):
        """Configure autosave functionality"""
        self.autosave_interval = 300000  # 5 minutes
        self.root.after(self.autosave_interval, self._autosave)

    def _autosave(self):
        """Perform autosave of current configuration and results"""
        try:
            if self.pipeline and not self.analysis_running:
                self._save_configuration()
                if hasattr(self.pipeline, 'save_results'):
                    self.pipeline.save_results(autosave=True)
            self.root.after(self.autosave_interval, self._autosave)
        except Exception as e:
            logger.warning(f"Autosave failed: {str(e)}")

    def _validate_topics_range(self, value: str) -> bool:
        """Validate topics range input"""
        if not value:
            return True
        pattern = r'^\d+(-\d+)?$'
        if not re.match(pattern, value):
            return False
        try:
            if '-' in value:
                min_topics, max_topics = map(int, value.split('-'))
                return 1 <= min_topics < max_topics <= 100
            return 1 <= int(value) <= 100
        except ValueError:
            return False

    def _create_tooltip(self, widget: tk.Widget, text: str):
        """Create tooltip for widget"""
        def show_tooltip(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            
            label = ttk.Label(tooltip, text=text, justify=tk.LEFT,
                            background="#ffffe0", relief=tk.SOLID, borderwidth=1)
            label.pack()
            
            def hide_tooltip():
                tooltip.destroy()
            
            tooltip.after(2000, hide_tooltip)
            
        widget.bind('<Enter>', show_tooltip)

    def _show_settings(self):
        """Show settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x300")
        settings_window.transient(self.root)
        
        # Add settings options here
        # ...

    def _show_documentation(self):
        """Show documentation window"""
        # Implementation for documentation viewer
        pass

    def _show_about(self):
        """Show about dialog"""
        messagebox.showinfo("About",
                          "Topic Modeling Analysis Tool\nVersion 1.0\n© 2024")

    def _preview_data(self):
        """Show data preview window"""
        if not self.pipeline or not hasattr(self.pipeline, 'data'):
            messagebox.showwarning("Warning", "No data loaded")
            return
            
        preview_window = tk.Toplevel(self.root)
        preview_window.title("Data Preview")
        preview_window.geometry("600x400")
        
        # Add data preview implementation
        # ...

    def _show_advanced_options(self):
        """Show advanced options dialog"""
        # Implementation for advanced options
        pass

    def _update_plot_options(self, event=None):
        """Update plot options based on selected plot type"""
        # Clear existing options
        for widget in self.plot_options_frame.winfo_children():
            widget.destroy()
            
        # Add new options based on plot type
        # ...

    def _on_closing(self):
        """Handle application closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self._save_configuration()
            self.root.quit()

    def run(self):
        """Start the GUI application with error handling"""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
            self.root.mainloop()
        except Exception as e:
            logger.critical(f"Application crashed: {str(e)}")
            raise

def main():
    """Main execution function with error handling."""
    try:
        app = TopicModelingGUI()
        app.run()
    except Exception as e:
        logger.critical(f"Application failed to start: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
