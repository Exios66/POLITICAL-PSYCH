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
from typing import List, Tuple, Dict, Any, Optional
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

# Configure logging with more detailed formatting
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('topic_modeling.log')
    ]
)

logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Download required NLTK resources
NLTK_RESOURCES = [
    'punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger',
    'maxent_ne_chunker', 'words', 'omw-1.4'
]

for resource in NLTK_RESOURCES:
    try:
        nltk.download(resource, quiet=True)
    except Exception as e:
        logger.warning(f"Failed to download NLTK resource {resource}: {str(e)}")

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    logger.info("Downloading spaCy model...")
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

class TopicModelingGUI:
    """GUI interface for topic modeling analysis"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Topic Modeling Analysis")
        self.root.geometry("800x600")
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Initialize pipeline
        self.pipeline = None
        
        # Create GUI sections
        self._create_menu()
        self._create_input_section()
        self._create_analysis_section()
        self._create_visualization_section()
        
        # Load last configuration
        self._load_configuration()

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
        analysis_menu.add_command(label="Run Topic Modeling", command=self._run_analysis)
        analysis_menu.add_command(label="Generate Report", command=self._generate_report)

    def _create_input_section(self):
        """Create data input section"""
        input_frame = ttk.LabelFrame(self.main_frame, text="Data Input", padding="5")
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        ttk.Button(input_frame, text="Select Data File", command=self._import_data).grid(row=0, column=0, padx=5, pady=5)
        self.file_label = ttk.Label(input_frame, text="No file selected")
        self.file_label.grid(row=0, column=1, padx=5, pady=5)
        
        # Text column selection
        ttk.Label(input_frame, text="Text Columns:").grid(row=1, column=0, padx=5, pady=5)
        self.text_columns = ttk.Entry(input_frame)
        self.text_columns.grid(row=1, column=1, padx=5, pady=5)
        self.text_columns.insert(0, "open_ended_response_1,open_ended_response_2")

    def _create_analysis_section(self):
        """Create analysis options section"""
        analysis_frame = ttk.LabelFrame(self.main_frame, text="Analysis Options", padding="5")
        analysis_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Model selection
        ttk.Label(analysis_frame, text="Topic Model:").grid(row=0, column=0, padx=5, pady=5)
        self.model_type = ttk.Combobox(analysis_frame, values=["LDA", "NMF", "HDP", "LSI"])
        self.model_type.grid(row=0, column=1, padx=5, pady=5)
        self.model_type.set("LDA")
        
        # Number of topics range
        ttk.Label(analysis_frame, text="Topics Range (min-max):").grid(row=1, column=0, padx=5, pady=5)
        self.topics_range = ttk.Entry(analysis_frame)
        self.topics_range.grid(row=1, column=1, padx=5, pady=5)
        self.topics_range.insert(0, "2-15")
        
        # Run button
        ttk.Button(analysis_frame, text="Run Analysis", 
                  command=self._run_analysis).grid(row=2, column=0, columnspan=2, pady=10)

    def _create_visualization_section(self):
        """Create visualization options section"""
        viz_frame = ttk.LabelFrame(self.main_frame, text="Visualization", padding="5")
        viz_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Visualization type selection
        ttk.Label(viz_frame, text="Plot Type:").grid(row=0, column=0, padx=5, pady=5)
        self.plot_type = ttk.Combobox(viz_frame, 
                                    values=["Topic Distribution", "Word Clouds", "Coherence Scores", "Interactive Topics"])
        self.plot_type.grid(row=0, column=1, padx=5, pady=5)
        self.plot_type.set("Topic Distribution")
        
        # Generate plot button
        self.plot_button = ttk.Button(viz_frame, text="Generate Plot", 
                                    command=self._generate_plot)
        self.plot_button.grid(row=1, column=0, columnspan=2, pady=10)
        self.plot_button.config(state='disabled')
        
        # Export button
        self.export_button = ttk.Button(viz_frame, text="Export Results",
                                      command=self._export_results)
        self.export_button.grid(row=2, column=0, columnspan=2, pady=10)
        self.export_button.config(state='disabled')

    def _save_configuration(self):
        """Save current configuration to file"""
        config = {
            'model_type': self.model_type.get(),
            'topics_range': self.topics_range.get(),
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
                self.model_type.set(config.get('model_type', 'LDA'))
                self.topics_range.delete(0, tk.END)
                self.topics_range.insert(0, config.get('topics_range', '2-15'))
                self.plot_type.set(config.get('plot_type', 'Topic Distribution'))
        except Exception as e:
            logging.warning(f"Could not load last configuration: {e}")

    def _import_data(self):
        """Import data file"""
        filename = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.file_label.config(text=os.path.basename(filename))
            self.pipeline = TopicModelingPipeline(
                input_file=filename,
                output_dir='results/topic_modeling',
                text_columns=self.text_columns.get().split(',')
            )
            self.plot_button.config(state='normal')

    def _run_analysis(self):
        """Run topic modeling analysis"""
        if not self.pipeline:
            messagebox.showerror("Error", "Please import data first")
            return
            
        try:
            min_topics, max_topics = map(int, self.topics_range.get().split('-'))
            self.pipeline.num_topics_range = (min_topics, max_topics)
            
            # Run pipeline
            self.pipeline.load_and_validate_data()
            self.pipeline.prepare_corpus()
            self.pipeline.train_models()
            self.pipeline.analyze_topics()
            
            self.export_button.config(state='normal')
            messagebox.showinfo("Success", "Analysis completed successfully")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _generate_plot(self):
        """Generate selected visualization"""
        if not self.pipeline:
            messagebox.showerror("Error", "Please run analysis first")
            return
            
        plot_type = self.plot_type.get()
        try:
            if plot_type == "Topic Distribution":
                self.pipeline._visualize_topics(
                    self.pipeline.models[self.model_type.get().lower()],
                    self.model_type.get().lower()
                )
            elif plot_type == "Word Clouds":
                self.pipeline._create_wordclouds(
                    self.pipeline.models[self.model_type.get().lower()],
                    self.model_type.get().lower()
                )
            messagebox.showinfo("Success", f"Generated {plot_type} visualization")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _export_results(self):
        """Export analysis results"""
        if not self.pipeline:
            messagebox.showerror("Error", "Please run analysis first")
            return
            
        try:
            export_dir = filedialog.askdirectory(title="Select Export Directory")
            if export_dir:
                self.pipeline.save_results()
                messagebox.showinfo("Success", f"Results exported to {export_dir}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _generate_report(self):
        """Generate detailed analysis report"""
        if not self.pipeline:
            messagebox.showerror("Error", "Please run analysis first")
            return
            
        try:
            self.pipeline.save_results()
            messagebox.showinfo("Success", "Report generated successfully")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

def main():
    """Main execution function."""
    app = TopicModelingGUI()
    app.run()

if __name__ == "__main__":
    main()
