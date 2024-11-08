import logging
import sys
from pathlib import Path
import traceback
from typing import Dict, List, Optional, Union, Any, Tuple
import warnings

from gensim.matutils import hellinger
from gensim.models import LdaModel, CoherenceModel
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import (
    STOPWORDS, 
    preprocess_string,
    strip_punctuation,
    strip_numeric,
    strip_tags,
    strip_multiple_whitespaces
)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn import metrics
from wordcloud import WordCloud
from scipy.stats import entropy
from sklearn.feature_extraction.text import CountVectorizer
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')

# Custom exceptions
class DataValidationError(Exception):
    """Raised when data validation fails"""
    pass

class ModelTrainingError(Exception):
    """Raised when model training fails"""
    pass

class AnalysisError(Exception):
    """Raised when topic analysis fails"""
    pass

class TopicModelingPipeline:
    """Pipeline for topic modeling analysis using LDA"""
    
    def __init__(
        self,
        input_file: Union[str, Path],
        output_dir: Union[str, Path],
        text_columns: List[str],
        min_topics: int = 2,
        max_topics: int = 15,
        log_file: str = 'topic_modeling.log',
        min_doc_length: int = 10,
        max_doc_length: Optional[int] = None,
        min_word_length: int = 3,
        max_word_length: int = 30,
        min_word_freq: int = 2,
        max_word_freq_pct: float = 0.9,
        max_vocab_size: int = 100000,
        random_seed: int = 42
    ):
        """
        Initialize the topic modeling pipeline.
        
        Args:
            input_file: Path to input CSV file
            output_dir: Directory to save outputs
            text_columns: List of column names containing text data
            min_topics: Minimum number of topics to try
            max_topics: Maximum number of topics to try
            log_file: Path to log file
            min_doc_length: Minimum document length in words
            max_doc_length: Maximum document length in words (None for no limit)
            min_word_length: Minimum word length to keep
            max_word_length: Maximum word length to keep
            min_word_freq: Minimum word frequency across documents
            max_word_freq_pct: Maximum word frequency as percentage of documents
            max_vocab_size: Maximum vocabulary size to keep
            random_seed: Random seed for reproducibility
            
        Raises:
            ValueError: If invalid parameters provided
        """
        # Validate parameters
        if min_topics < 2:
            raise ValueError("min_topics must be >= 2")
        if max_topics <= min_topics:
            raise ValueError("max_topics must be > min_topics")
        if min_doc_length < 1:
            raise ValueError("min_doc_length must be >= 1")
        if max_doc_length and max_doc_length < min_doc_length:
            raise ValueError("max_doc_length must be > min_doc_length")
        if not 0 < max_word_freq_pct < 1:
            raise ValueError("max_word_freq_pct must be between 0 and 1")
            
        # Store parameters
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.text_columns = text_columns
        self.models: Dict[int, Dict[str, Any]] = {}
        self.num_topics_range = (min_topics, max_topics)
        self.min_doc_length = min_doc_length
        self.max_doc_length = max_doc_length
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        self.min_word_freq = min_word_freq
        self.max_word_freq_pct = max_word_freq_pct
        self.max_vocab_size = max_vocab_size
        self.random_seed = random_seed
        
        # Initialize state
        self.data: Optional[pd.DataFrame] = None
        self.corpus: Optional[List] = None
        self.dictionary: Optional[Dictionary] = None
        self.analysis: Optional[Dict] = None
        self.vectorizer: Optional[CountVectorizer] = None
        
        # Set random seeds
        np.random.seed(random_seed)
        
        # Configure logging with both file and console handlers
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Create formatters and handlers
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Log initialization
        self.logger.info(
            f"Initialized TopicModelingPipeline with parameters:\n"
            f"Input file: {self.input_file}\n"
            f"Output directory: {self.output_dir}\n"
            f"Text columns: {self.text_columns}\n"
            f"Topics range: {self.num_topics_range}\n"
            f"Document length range: ({self.min_doc_length}, {self.max_doc_length})\n"
            f"Word length range: ({self.min_word_length}, {self.max_word_length})\n"
            f"Word frequency range: ({self.min_word_freq}, {self.max_word_freq_pct})\n"
            f"Max vocabulary size: {self.max_vocab_size}\n"
            f"Random seed: {self.random_seed}"
        )

    def load_and_validate_data(self) -> pd.DataFrame:
        """
        Load data from CSV file and validate contents.
        
        Returns:
            DataFrame containing validated data
            
        Raises:
            DataValidationError: If validation fails
            FileNotFoundError: If input file not found
            pd.errors.EmptyDataError: If file is empty
            pd.errors.ParserError: If file cannot be parsed
        """
        try:
            if not self.input_file.exists():
                raise FileNotFoundError(f"Input file not found: {self.input_file}")
                
            # Try reading with different encodings
            encodings = ['utf-8', 'latin1', 'iso-8859-1']
            for encoding in encodings:
                try:
                    self.data = pd.read_csv(self.input_file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise DataValidationError(f"Could not read file with encodings: {encodings}")
                
            self.logger.info(f"Loaded data with shape: {self.data.shape}")
            
            # Basic data validation
            if len(self.data) == 0:
                raise DataValidationError("Input file is empty")
                
            # Validate text columns exist
            missing_cols = [col for col in self.text_columns if col not in self.data.columns]
            if missing_cols:
                raise DataValidationError(f"Missing text columns: {missing_cols}")
                
            # Validate data types
            non_string_cols = [
                col for col in self.text_columns 
                if not pd.api.types.is_string_dtype(self.data[col])
            ]
            if non_string_cols:
                self.logger.warning(
                    f"Converting non-string columns to string: {non_string_cols}"
                )
                for col in non_string_cols:
                    self.data[col] = self.data[col].astype(str)
            
            # Remove rows with missing text
            initial_rows = len(self.data)
            self.data = self.data.dropna(subset=self.text_columns)
            dropped_rows = initial_rows - len(self.data)
            
            if dropped_rows > 0:
                self.logger.warning(f"Dropped {dropped_rows} rows with missing text")
            
            if len(self.data) == 0:
                raise DataValidationError("No valid text data found after cleaning")
                
            # Save raw data sample
            sample_path = self.output_dir / 'data_sample.csv'
            self.data.head(1000).to_csv(sample_path, index=False)
            
            self.logger.info(f"Successfully validated {len(self.data)} documents")
            return self.data
            
        except Exception as e:
            error_msg = f"Error loading/validating data: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            raise DataValidationError(error_msg)

    def prepare_corpus(self) -> List:
        """
        Preprocess text and create document corpus.
        
        Returns:
            List of processed documents in bag-of-words format
            
        Raises:
            ValueError: If data not loaded
            RuntimeError: If corpus preparation fails
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_and_validate_data() first.")
            
        try:
            # Combine text columns
            texts = self.data[self.text_columns].fillna('').agg(' '.join, axis=1)
            
            # Initialize preprocessing filters
            filters = [
                lambda x: x.lower(),
                strip_tags,
                strip_punctuation,
                strip_numeric,
                strip_multiple_whitespaces
            ]
            
            # Custom stopwords
            custom_stopwords = set(STOPWORDS)
            custom_stopwords.update(['www', 'http', 'https', 'com'])
            
            # Preprocess texts with progress tracking
            processed_texts = []
            total_docs = len(texts)
            
            for i, text in enumerate(texts, 1):
                if i % 1000 == 0:
                    self.logger.info(f"Processing document {i}/{total_docs}")
                    
                # Apply preprocessing steps with error handling
                try:
                    tokens = preprocess_string(text, filters=filters)
                except Exception as e:
                    self.logger.warning(f"Error preprocessing document {i}: {str(e)}")
                    tokens = []

                # Apply length and stopword filters with validation
                filtered_tokens = []
                for token in tokens:
                    # Validate token is string
                    if not isinstance(token, str):
                        continue
                        
                    # Apply filters
                    if (token not in custom_stopwords and 
                        self.min_word_length <= len(token) <= self.max_word_length):
                        filtered_tokens.append(token)

                # Validate document length
                if len(filtered_tokens) < self.min_doc_length:
                    self.logger.debug(
                        f"Document {i} filtered: {len(filtered_tokens)} tokens < "
                        f"minimum {self.min_doc_length}"
                    )
                    continue

                # Apply max length truncation if needed
                if self.max_doc_length:
                    if len(filtered_tokens) > self.max_doc_length:
                        self.logger.debug(
                            f"Document {i} truncated from {len(filtered_tokens)} to "
                            f"{self.max_doc_length} tokens"
                        )
                    filtered_tokens = filtered_tokens[:self.max_doc_length]

                # Add processed document
                processed_texts.append(filtered_tokens)

            # Validate corpus size
            if not processed_texts:
                error_msg = "No documents remained after preprocessing"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)

            # Create dictionary with validation
            try:
                self.dictionary = Dictionary(processed_texts)
                initial_vocab = len(self.dictionary)
                
                # Apply dictionary filters
                self.dictionary.filter_extremes(
                    no_below=max(1, self.min_word_freq),  # Ensure minimum of 1
                    no_above=min(1.0, self.max_word_freq_pct),  # Ensure maximum of 1.0
                    keep_n=self.max_vocab_size if self.max_vocab_size > 0 else None
                )
                
                final_vocab = len(self.dictionary)
                self.logger.info(
                    f"Dictionary filtered from {initial_vocab} to {final_vocab} terms"
                )
                
            except Exception as e:
                error_msg = f"Error creating dictionary: {str(e)}"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)

            # Create corpus with validation
            try:
                self.corpus = [
                    self.dictionary.doc2bow(text) 
                    for text in processed_texts
                ]
                
                # Validate corpus entries
                empty_docs = sum(1 for doc in self.corpus if len(doc) == 0)
                if empty_docs > 0:
                    self.logger.warning(
                        f"{empty_docs} documents have no terms after dictionary filtering"
                    )
                
            except Exception as e:
                error_msg = f"Error creating corpus: {str(e)}"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)

            # Save artifacts with error handling
            try:
                dict_path = self.output_dir / 'dictionary.pkl'
                corpus_path = self.output_dir / 'corpus.pkl'
                
                self.dictionary.save(str(dict_path))
                joblib.dump(self.corpus, corpus_path)
                
                vocab_size = len(self.dictionary)
                corpus_size = len(self.corpus)
                
                self.logger.info(
                    f"Prepared corpus with {corpus_size:,} documents and "
                    f"vocabulary size {vocab_size:,}\n"
                    f"Dictionary saved to {dict_path}\n"
                    f"Corpus saved to {corpus_path}"
                )
                
            except Exception as e:
                error_msg = f"Error saving corpus artifacts: {str(e)}"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)

            # Initialize metrics data list
            metrics_data = []
            
            # Iterate over analysis results to add average distances
            for num_topics, v in self.analysis.items():
                metrics = {}
                metrics.update({
                    f'avg_distance_{metric}': dist
                    for metric, dist in v['avg_distances'].items()
                })
                metrics_data.append(metrics)
            
            return self.corpus
                
            metrics_df = pd.DataFrame(metrics_data)
            metrics_path = tables_dir / 'model_metrics.csv'
            metrics_df.to_csv(metrics_path, index=False)
            self.logger.info(f"Saved model metrics to {metrics_path}")
            
            # Save detailed topic information
            for num_topics, results in self.analysis.items():
                topic_data = []
                for topic_id, topic_info in results['topics'].items():
                    topic_metrics = topic_info['metrics']
                    
                    for n_words in [10, 20, 50]:
                        topic_data.append({
                            'topic_id': topic_id,
                            'num_words': n_words,
                            'words': ', '.join(topic_info['words'][n_words]),
                            'word_probabilities': topic_info['probabilities'][n_words],
                            'entropy': topic_metrics['entropy'],
                            'distinctiveness': topic_metrics['distinctiveness'],
                            'coherence': topic_metrics['coherence']
                        })
                    
                topic_df = pd.DataFrame(topic_data)
                topic_path = tables_dir / f'topics_{num_topics}.csv'
                topic_df.to_csv(topic_path, index=False)
                self.logger.info(f"Saved topic details to {topic_path}")
                
            # Generate visualizations
            self._visualize_topics(plots_dir)
            self._create_wordclouds(plots_dir)
            
            self.logger.info(f"Successfully saved all results to {self.output_dir}")
            
        except Exception as e:
            error_msg = f"Error saving results: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _visualize_topics(self, plots_dir: Path) -> None:
        """
        Generate topic visualization plots.
        
        Args:
            plots_dir: Directory to save plots
            
        Raises:
            RuntimeError: If visualization fails
        """
        try:
            for num_topics, results in self.analysis.items():
                # Topic distance heatmaps for each metric
                for metric, distances in results['distances'].items():
                    plt.figure(figsize=(12, 10))
                    sns.heatmap(
                        distances,
                        annot=True,
                        fmt='.3f',
                        cmap='YlOrRd',
                        xticklabels=[f'Topic {i}' for i in range(num_topics)],
                        yticklabels=[f'Topic {i}' for i in range(num_topics)]
                    )
                    plt.title(f'Topic Distances - {metric} (k={num_topics})')
                    plt.tight_layout()
                    
                    heatmap_path = plots_dir / f'topic_distances_{metric}_{num_topics}.png'
                    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    self.logger.info(f"Saved topic distance heatmap to {heatmap_path}")
                
                # Topic metrics visualization
                topic_metrics = pd.DataFrame([
                    {
                        'Topic': f'Topic {topic_id}',
                        'Entropy': info['metrics']['entropy'],
                        'Distinctiveness': info['metrics']['distinctiveness'],
                        'Coherence': info['metrics']['coherence']
                    }
                    for topic_id, info in results['topics'].items()
                ])
                
                # Plot topic metrics
                plt.figure(figsize=(12, 6))
                topic_metrics.plot(x='Topic', kind='bar', width=0.8)
                plt.title(f'Topic Metrics (k={num_topics})')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                metrics_path = plots_dir / f'topic_metrics_{num_topics}.png'
                plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                self.logger.info(f"Saved topic metrics plot to {metrics_path}")
                
        except Exception as e:
            error_msg = f"Error creating visualizations: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _create_wordclouds(self, plots_dir: Path) -> None:
        """
        Generate word clouds for each topic.
        
        Args:
            plots_dir: Directory to save plots
            
        Raises:
            RuntimeError: If wordcloud generation fails
        """
        try:
            wordcloud_dir = plots_dir / 'wordclouds'
            wordcloud_dir.mkdir(exist_ok=True)
            
            # Configure word cloud settings
            wordcloud_config = {
                'width': 1200,
                'height': 800,
                'background_color': 'white',
                'max_words': 100,
                'prefer_horizontal': 0.7,
                'scale': 3,
                'colormap': 'viridis',
                'relative_scaling': 0.5,
                'min_font_size': 10,
                'max_font_size': 100
            }
            
            for num_topics, results in self.analysis.items():
                for topic_id, topic_info in results['topics'].items():
                    try:
                        # Create word frequency dict using top 50 words
                        # Handle potential index errors
                        if len(topic_info['words']) < 50 or len(topic_info['probabilities']) < 50:
                            word_freq = dict(zip(
                                topic_info['words'],
                                topic_info['probabilities']
                            ))
                        else:
                            word_freq = dict(zip(
                                topic_info['words'][:50],
                                topic_info['probabilities'][:50]
                            ))
                        
                        # Validate word frequencies
                        if not word_freq:
                            self.logger.warning(f"Empty word frequencies for topic {topic_id}")
                            continue
                            
                        # Generate wordcloud with error handling
                        try:
                            wordcloud = WordCloud(**wordcloud_config).generate_from_frequencies(word_freq)
                        except ValueError as e:
                            self.logger.error(f"WordCloud generation failed for topic {topic_id}: {str(e)}")
                            continue
                        
                        # Create figure with metadata
                        plt.figure(figsize=(12, 8))
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis('off')
                        
                        # Add topic metadata with validation
                        metrics = topic_info.get('metrics', {})
                        entropy_val = metrics.get('entropy', 0.0)
                        distinctiveness_val = metrics.get('distinctiveness', 0.0) 
                        coherence_val = metrics.get('coherence', 0.0)
                        
                        plt.title(
                            f'Topic {topic_id} Keywords (k={num_topics})\n'
                            f'Entropy: {entropy_val:.3f} | '
                            f'Distinctiveness: {distinctiveness_val:.3f} | '
                            f'Coherence: {coherence_val:.3f}',
                            fontsize=14,
                            pad=20
                        )
                        
                        # Save with error handling
                        wordcloud_path = wordcloud_dir / f'wordcloud_k{num_topics}_topic{topic_id}.png'
                        try:
                            plt.savefig(wordcloud_path, dpi=300, bbox_inches='tight')
                        except Exception as e:
                            self.logger.error(f"Failed to save wordcloud for topic {topic_id}: {str(e)}")
                        finally:
                            plt.close()
                            
                    except Exception as e:
                        self.logger.error(f"Error processing topic {topic_id}: {str(e)}")
                        continue
                        
            self.logger.info(f"Generated wordclouds in {wordcloud_dir}")
                    
        except Exception as e:
            error_msg = f"Error creating wordclouds: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
            
    @staticmethod
    def _calculate_topic_entropy(topic_words: List[Tuple[str, float]]) -> float:
        """
        Calculate entropy of topic probability distribution.
        
        Args:
            topic_words: List of (word, probability) tuples
            
        Returns:
            Entropy value
            
        Raises:
            ValueError: If topic_words is empty or contains invalid probabilities
        """
        if not topic_words:
            raise ValueError("Empty topic words list")
            
        probs = np.array([prob for _, prob in topic_words])
        
        # Validate probabilities
        if not np.all(probs >= 0):
            raise ValueError("Negative probabilities found")
        if not np.isclose(np.sum(probs), 1.0, rtol=1e-5):
            probs = probs / np.sum(probs)  # Normalize if needed
            
        return entropy(probs)
        
    @staticmethod
    def _calculate_topic_distinctiveness(
        model: LdaModel,
        topic_id: int,
        num_topics: int,
        num_words: int = 20
    ) -> float:
        """
        Calculate topic distinctiveness based on unique top words.
        
        Args:
            model: Trained LDA model
            topic_id: ID of topic to analyze
            num_topics: Total number of topics
            num_words: Number of top words to consider
            
        Returns:
            Distinctiveness score
            
        Raises:
            ValueError: If invalid parameters or model state
        """
        if not model:
            raise ValueError("Model not initialized")
        if topic_id >= num_topics:
            raise ValueError("Topic ID exceeds number of topics")
        if num_words < 1:
            raise ValueError("Number of words must be positive")
            
        try:
            # Get top words for target topic
            target_words = set(word for word, _ in model.show_topic(topic_id, topn=num_words))
            
            # Get top words for other topics
            other_words = set()
            for other_id in range(num_topics):
                if other_id != topic_id:
                    other_words.update(
                        word for word, _ in model.show_topic(other_id, topn=num_words)
                    )
            
            # Calculate distinctiveness as fraction of unique words
            if not target_words:
                return 0.0
                
            unique_words = target_words - other_words
            return len(unique_words) / len(target_words)
            
        except Exception as e:
            logging.error(f"Error calculating distinctiveness: {str(e)}")
            return 0.0
        
    @staticmethod
    def _calculate_topic_coherence(
        model: LdaModel,
        topic_id: int,
        corpus: List,
        num_words: int = 20
    ) -> float:
        """
        Calculate topic coherence using normalized PMI.
        
        Args:
            model: Trained LDA model
            topic_id: ID of topic to analyze
            corpus: Document corpus
            num_words: Number of top words to consider
            
        Returns:
            Coherence score
            
        Raises:
            ValueError: If invalid parameters
        """
        if not model or not corpus:
            raise ValueError("Model or corpus not initialized")
        if num_words < 1:
            raise ValueError("Number of words must be positive")
            
        try:
            coherence = CoherenceModel(
                model=model,
                corpus=corpus,
                topics=[model.show_topic(topic_id, topn=num_words)],
                coherence='c_npmi'
            )
            score = coherence.get_coherence()
            return max(0.0, min(1.0, score))  # Clamp between 0 and 1
            
        except Exception as e:
            logging.error(f"Error calculating coherence: {str(e)}")
            return 0.0
            
    @staticmethod
    def _calculate_topic_distances(
        model: LdaModel,
        num_topics: int,
        metric: str = 'hellinger'
    ) -> np.ndarray:
        """
        Calculate pairwise topic distances.
        
        Args:
            model: Trained LDA model
            num_topics: Number of topics
            metric: Distance metric to use ('hellinger' or 'jensen_shannon')
            
        Returns:
            Distance matrix
            
        Raises:
            ValueError: If invalid parameters or metric
        """
        if not model:
            raise ValueError("Model not initialized")
        if num_topics < 1:
            raise ValueError("Number of topics must be positive")
        if metric not in ['hellinger', 'jensen_shannon']:
            raise ValueError("Invalid distance metric")
            
        try:
            distances = np.zeros((num_topics, num_topics))
            
            for i in range(num_topics):
                for j in range(num_topics):
                    if metric == 'hellinger':
                        distances[i,j] = hellinger(
                            model.get_topic_terms(i),
                            model.get_topic_terms(j)
                        )
                    elif metric == 'jensen_shannon':
                        # Calculate Jensen-Shannon divergence
                        p = np.array([prob for id, prob in model.get_topic_terms(i)])
                        q = np.array([prob for id, prob in model.get_topic_terms(j)])
                        
                        # Ensure valid probability distributions
                        p = p / np.sum(p)
                        q = q / np.sum(q)
                        
                        # Calculate mean distribution
                        m = 0.5 * (p + q)
                        
                        # Calculate JS divergence
                        # JS = 0.5 * (KL(P||M) + KL(Q||M))
                        js_div = 0.5 * (
                            entropy(p, m, base=2) +
                            entropy(q, m, base=2)
                        )
                        
                        # Convert to distance metric
                        distances[i,j] = np.sqrt(js_div)
                        
            return distances
            
        except Exception as e:
            logging.error(f"Error calculating distances: {str(e)}")
            return np.zeros((num_topics, num_topics))