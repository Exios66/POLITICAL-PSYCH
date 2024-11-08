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
from tkinter import ttk, messagebox, filedialog, font
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE, MDS
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer
from nltk.tag import pos_tag, averaged_perceptron_tagger
from nltk.chunk import ne_chunk
from nltk.probability import FreqDist
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder

import gensim
from gensim import corpora, models, similarities
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel, Word2Vec, FastText
from gensim.models.phrases import Phrases, Phraser
from gensim.models.callbacks import PerplexityMetric, CoherenceMetric
from gensim.corpora.dictionary import Dictionary
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation

import pyLDAvis
import pyLDAvis.gensim_models
import pyLDAvis.sklearn # type: ignore
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import spacy
from spacy.tokens import Doc, Token, Span
from spacy.language import Language
from spacy.pipeline import EntityRuler
from spacy.matcher import Matcher, PhraseMatcher

from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# NLP Libraries
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer
from nltk.tag import pos_tag, averaged_perceptron_tagger
from nltk.chunk import ne_chunk
from nltk.probability import FreqDist
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder

# Topic Modeling Libraries
import gensim
from gensim import corpora, models, similarities
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel, Word2Vec, FastText
from gensim.models.phrases import Phrases, Phraser
from gensim.models.callbacks import PerplexityMetric, CoherenceMetric
from gensim.corpora.dictionary import Dictionary
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation

# Visualization Libraries
import pyLDAvis
import pyLDAvis.gensim_models
import pyLDAvis.sklearn # type: ignore
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# SpaCy NLP Library
import spacy
from spacy.tokens import Doc, Token, Span
from spacy.language import Language
from spacy.pipeline import EntityRuler
from spacy.matcher import Matcher, PhraseMatcher

# Sentiment Analysis Libraries
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Custom Pipeline
from topic_modeling_pipeline import TopicModelingPipeline

# Configure comprehensive logging with detailed formatting and file rotation
LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)
LOG_FILENAME = LOG_DIR / f'topic_modeling_{datetime.datetime.now():%Y%m%d}.log'

# Create custom log formatter
class CustomFormatter(logging.Formatter):
    """Custom log formatter with color coding and extra context"""
    
    COLORS = {
        'DEBUG': '\033[0;36m',    # Cyan
        'INFO': '\033[0;32m',     # Green
        'WARNING': '\033[0;33m',  # Yellow
        'ERROR': '\033[0;31m',    # Red
        'CRITICAL': '\033[0;35m', # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add thread name and memory usage
        record.thread_name = f"{record.threadName}:{record.thread}"
        import psutil
        process = psutil.Process()
        record.memory_usage = f"{process.memory_info().rss / 1024 / 1024:.1f}MB"
        
        # Color code based on level
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.color = color
        record.reset = self.COLORS['RESET']
        
        return super().format(record)

# Configure logging with custom formatter
log_format = '%(asctime)s %(color)s[%(levelname)s]%(reset)s [%(thread_name)s][%(memory_usage)s] %(name)s - %(funcName)s:%(lineno)d - %(message)s'
formatter = CustomFormatter(log_format)

# Add handlers with rotation and compression
handlers = [
    logging.StreamHandler(sys.stdout),
    logging.handlers.RotatingFileHandler(
        LOG_FILENAME,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=10,
        encoding='utf-8'
    ),
    logging.handlers.TimedRotatingFileHandler(
        LOG_FILENAME,
        when='midnight',
        interval=1,
        backupCount=30,
        encoding='utf-8'
    )
]

for handler in handlers:
    handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.DEBUG,
    handlers=handlers
)

logger = logging.getLogger(__name__)

# Configure comprehensive warning filters
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='gensim')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')

# Expanded NLTK resources with error handling and validation
@dataclass
class NLTKResource:
    """Dataclass to manage NLTK resource metadata"""
    name: str
    required: bool = True
    downloaded: bool = False
    error_count: int = 0
    last_error: Optional[str] = None

NLTK_RESOURCES = [
    NLTKResource('punkt'),
    NLTKResource('wordnet'),
    NLTKResource('stopwords'),
    NLTKResource('averaged_perceptron_tagger'),
    NLTKResource('maxent_ne_chunker'),
    NLTKResource('words'),
    NLTKResource('omw-1.4'),
    # Additional resources
    NLTKResource('vader_lexicon', required=False),
    NLTKResource('sentiwordnet', required=False),
    NLTKResource('opinion_lexicon', required=False),
    NLTKResource('subjectivity', required=False)
]

def download_nltk_resources(max_retries: int = 3, timeout: int = 30) -> None:
    """
    Download NLTK resources with advanced retry mechanism and timeout
    
    Args:
        max_retries: Maximum number of download attempts
        timeout: Timeout in seconds for each download attempt
    """
    download_lock = Lock()
    
    def download_resource(resource: NLTKResource) -> None:
        for attempt in range(max_retries):
            try:
                with download_lock:
                    nltk.download(resource.name, quiet=True, raise_on_error=True, timeout=timeout)
                resource.downloaded = True
                logger.info(f"Successfully downloaded NLTK resource: {resource.name}")
                break
            except Exception as e:
                resource.error_count += 1
                resource.last_error = str(e)
                if attempt == max_retries - 1:
                    error_msg = f"Failed to download NLTK resource {resource.name} after {max_retries} attempts: {str(e)}"
                    if resource.required:
                        logger.error(error_msg)
                        raise RuntimeError(error_msg)
                    else:
                        logger.warning(f"Optional resource {resource.name} not downloaded: {error_msg}")
                else:
                    logger.warning(f"Attempt {attempt + 1} failed for {resource.name}, retrying...")
                    continue
    
    # Download resources in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(download_resource, NLTK_RESOURCES)

# Initialize NLTK downloads with error handling
try:
    download_nltk_resources()
except Exception as e:
    logger.critical(f"Critical error during NLTK resource initialization: {str(e)}")
    sys.exit(1)

# Enhanced spaCy model loader with pipeline customization
@dataclass
class SpacyModelConfig:
    """Configuration for spaCy model loading"""
    name: str
    disable: List[str] = field(default_factory=lambda: ['ner', 'parser'])
    enable: List[str] = field(default_factory=lambda: ['tagger', 'lemmatizer'])
    custom_components: List[Tuple[str, Any]] = field(default_factory=list)

def load_spacy_model(
    config: SpacyModelConfig = SpacyModelConfig('en_core_web_sm')
) -> Language:
    """
    Load spaCy model with advanced configuration and pipeline customization
    
    Args:
        config: SpacyModelConfig instance with model specifications
        
    Returns:
        Loaded and configured spaCy Language model
    """
    try:
        # Try loading existing model
        nlp = spacy.load(config.name, disable=config.disable)
        logger.info(f"Loaded existing spaCy model: {config.name}")
    except OSError:
        # Download if not found
        logger.info(f"Downloading spaCy model {config.name}...")
        try:
            os.system(f'python -m spacy download {config.name}')
            nlp = spacy.load(config.name, disable=config.disable)
        except Exception as e:
            logger.error(f"Failed to download spaCy model: {str(e)}")
            raise
    
    # Customize pipeline
    for component_name in config.enable:
        if component_name not in nlp.pipe_names:
            nlp.enable_pipe(component_name)
    
    # Add custom components
    for name, component in config.custom_components:
        if name not in nlp.pipe_names:
            nlp.add_pipe(component, name=name)
    
    return nlp

# Initialize spaCy with custom configuration
try:
    nlp = load_spacy_model(SpacyModelConfig(
        name='en_core_web_sm',
        disable=['ner', 'parser', 'textcat'],
        enable=['tagger', 'lemmatizer'],
        custom_components=[
            ('sentencizer', spacy.pipeline.Sentencizer()),
            ('entity_ruler', lambda nlp: EntityRuler(nlp))
        ]
    ))
except Exception as e:
    logger.critical(f"Could not initialize spaCy model: {str(e)}")
    sys.exit(1)
