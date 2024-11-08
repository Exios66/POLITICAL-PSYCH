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

class TopicModelingPipeline:
    """
    A comprehensive pipeline for topic modeling of survey open-ended responses.
    """
    
    def __init__(
        self,
        input_file: str,
        output_dir: str,
        text_columns: List[str],
        min_df: float = 0.01,
        max_df: float = 0.95,
        num_topics_range: Tuple[int, int] = (2, 15),
        random_state: int = 42
    ):
        """
        Initialize the topic modeling pipeline.
        
        Args:
            input_file: Path to input CSV file
            output_dir: Directory for output files
            text_columns: List of column names containing text data
            min_df: Minimum document frequency for terms
            max_df: Maximum document frequency for terms
            num_topics_range: Range of topics to test (min, max)
            random_state: Random seed for reproducibility
        """
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.text_columns = text_columns
        self.min_df = min_df
        self.max_df = max_df
        self.num_topics_range = num_topics_range
        self.random_state = random_state
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize preprocessing tools
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(['would', 'could', 'should', 'might', 'must', 'need'])
        
        # Initialize containers for models and results
        self.data = None
        self.processed_docs = None
        self.dictionary = None
        self.corpus = None
        self.models = {
            'lda': None,
            'nmf': None,
            'hdp': None,
            'lsi': None
        }
        self.topic_distributions = {}
        
    def load_and_validate_data(self) -> pd.DataFrame:
        """
        Load and validate the input data.
        
        Returns:
            DataFrame with loaded data
        """
        try:
            data = pd.read_csv(self.input_file)
            logger.info(f"Loaded data with shape {data.shape}")
            
            # Validate text columns
            missing_cols = [col for col in self.text_columns if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")
            
            # Check for empty text columns
            empty_cols = data[self.text_columns].isna().all()
            if empty_cols.any():
                logger.warning(f"Columns with all missing values: {empty_cols[empty_cols].index.tolist()}")
            
            self.data = data
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def preprocess_text(self, text: str) -> List[str]:
        """
        Comprehensive text preprocessing pipeline.
        
        Args:
            text: Raw text string
            
        Returns:
            List of processed tokens
        """
        if pd.isna(text):
            return []
            
        # Convert to string
        text = str(text)
        
        # Basic cleaning
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords, numbers, and short tokens
        tokens = [
            token for token in tokens 
            if token not in self.stop_words
            and not token.isnumeric()
            and len(token) > 2
        ]
        
        # Lemmatization with POS tagging
        tokens = [
            self.lemmatizer.lemmatize(token, pos=self._get_wordnet_pos(token))
            for token in tokens
        ]
        
        # Named Entity Recognition
        doc = nlp(' '.join(tokens))
        ner_tokens = [
            token.text if not token.ent_type_ 
            else f"{token.ent_type_}_{token.text}"
            for token in doc
        ]
        
        return ner_tokens
        
    def _get_wordnet_pos(self, word: str) -> str:
        """Get POS tag for lemmatization."""
        tag = pos_tag([word])[0][1][0].upper()
        tag_dict = {
            "J": nltk.corpus.wordnet.ADJ,
            "N": nltk.corpus.wordnet.NOUN,
            "V": nltk.corpus.wordnet.VERB,
            "R": nltk.corpus.wordnet.ADV
        }
        return tag_dict.get(tag, nltk.corpus.wordnet.NOUN)

    def prepare_corpus(self) -> Tuple[List[List[str]], corpora.Dictionary, List[List[Tuple[int, float]]]]:
        """
        Prepare text corpus for topic modeling.
        
        Returns:
            Tuple of (processed documents, dictionary, corpus)
        """
        # Combine text columns
        combined_text = self.data[self.text_columns].fillna('').agg(' '.join, axis=1)
        
        # Preprocess documents
        self.processed_docs = [self.preprocess_text(text) for text in combined_text]
        
        # Build bigram and trigram models
        bigram = Phrases(self.processed_docs, min_count=5, threshold=100)
        trigram = Phrases(bigram[self.processed_docs], threshold=100)
        
        bigram_mod = Phraser(bigram)
        trigram_mod = Phraser(trigram)
        
        # Apply bigrams and trigrams
        self.processed_docs = [
            trigram_mod[bigram_mod[doc]] for doc in self.processed_docs
        ]
        
        # Create dictionary
        self.dictionary = corpora.Dictionary(self.processed_docs)
        
        # Filter extreme terms
        self.dictionary.filter_extremes(
            no_below=int(len(self.processed_docs) * self.min_df),
            no_above=self.max_df
        )
        
        # Create corpus
        self.corpus = [
            self.dictionary.doc2bow(doc) for doc in self.processed_docs
        ]
        
        return self.processed_docs, self.dictionary, self.corpus

    def train_models(self):
        """Train multiple topic models."""
        # LDA
        self.models['lda'] = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self._find_optimal_topics(),
            random_state=self.random_state,
            alpha='auto',
            per_word_topics=True,
            passes=20
        )
        
        # NMF (using sklearn)
        tfidf_vectorizer = TfidfVectorizer(
            max_df=self.max_df, 
            min_df=self.min_df,
            stop_words='english'
        )
        tfidf = tfidf_vectorizer.fit_transform(
            [' '.join(doc) for doc in self.processed_docs]
        )
        
        self.models['nmf'] = NMF(
            n_components=self._find_optimal_topics(),
            random_state=self.random_state
        ).fit(tfidf)
        
        # HDP
        self.models['hdp'] = HdpModel(
            corpus=self.corpus,
            id2word=self.dictionary
        )
        
        # LSI
        self.models['lsi'] = LsiModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self._find_optimal_topics()
        )

    def _find_optimal_topics(self) -> int:
        """
        Find optimal number of topics using coherence scores.
        
        Returns:
            Optimal number of topics
        """
        coherence_scores = []
        
        for num_topics in range(self.num_topics_range[0], self.num_topics_range[1] + 1):
            model = LdaModel(
                corpus=self.corpus,
                id2word=self.dictionary,
                num_topics=num_topics,
                random_state=self.random_state
            )
            
            coherence_model = CoherenceModel(
                model=model,
                texts=self.processed_docs,
                dictionary=self.dictionary,
                coherence='c_v'
            )
            
            coherence_scores.append(coherence_model.get_coherence())
            
        optimal_topics = self.num_topics_range[0] + np.argmax(coherence_scores)
        
        # Plot coherence scores
        plt.figure(figsize=(10, 6))
        plt.plot(
            range(self.num_topics_range[0], self.num_topics_range[1] + 1),
            coherence_scores,
            'bo-'
        )
        plt.xlabel('Number of Topics')
        plt.ylabel('Coherence Score')
        plt.title('Topic Coherence Scores')
        plt.savefig(self.output_dir / 'coherence_scores.png')
        plt.close()
        
        return optimal_topics

    def analyze_topics(self):
        """Generate comprehensive topic analysis."""
        for model_name, model in self.models.items():
            # Get topic distributions
            if model_name == 'lda':
                self.topic_distributions[model_name] = [
                    dict(model.get_document_topics(bow))
                    for bow in self.corpus
                ]
            
            # Generate topic visualizations
            if model_name in ['lda', 'nmf']:
                self._visualize_topics(model, model_name)
            
            # Generate topic summaries
            self._summarize_topics(model, model_name)
            
            # Create word clouds
            self._create_wordclouds(model, model_name)

    def _visualize_topics(self, model: Any, model_name: str):
        """Generate interactive topic visualizations."""
        if model_name == 'lda':
            vis = pyLDAvis.gensim_models.prepare(
                model, self.corpus, self.dictionary
            )
        else:
            vis = pyLDAvis.sklearn_.prepare(
                model, self.tfidf, self.tfidf_vectorizer
            )
            
        pyLDAvis.save_html(
            vis,
            str(self.output_dir / f'{model_name}_visualization.html')
        )

    def _summarize_topics(self, model: Any, model_name: str):
        """Generate topic summaries with key metrics."""
        summaries = []
        
        if model_name == 'lda':
            topics = model.show_topics(formatted=False)
            for topic_id, topic in topics:
                terms = [term for term, _ in topic]
                weights = [weight for _, weight in topic]
                
                summary = {
                    'topic_id': topic_id,
                    'top_terms': terms[:10],
                    'weights': weights[:10],
                    'coherence': self._calculate_topic_coherence(terms)
                }
                summaries.append(summary)
                
        pd.DataFrame(summaries).to_csv(
            self.output_dir / f'{model_name}_topic_summaries.csv',
            index=False
        )

    def _calculate_topic_coherence(self, terms: List[str]) -> float:
        """Calculate topic coherence score."""
        return CoherenceModel(
            topics=[terms],
            texts=self.processed_docs,
            dictionary=self.dictionary,
            coherence='c_v'
        ).get_coherence()

    def _create_wordclouds(self, model: Any, model_name: str):
        """Generate word clouds for each topic."""
        if model_name == 'lda':
            for topic_id in range(model.num_topics):
                topic_terms = dict(model.show_topic(topic_id, topn=50))
                
                wordcloud = WordCloud(
                    background_color='white',
                    width=800,
                    height=400
                ).generate_from_frequencies(topic_terms)
                
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'Topic {topic_id + 1}')
                plt.savefig(
                    self.output_dir / f'{model_name}_topic_{topic_id + 1}_wordcloud.png'
                )
                plt.close()

    def save_results(self):
        """Save all results and models."""
        # Save models
        for model_name, model in self.models.items():
            model.save(str(self.output_dir / f'{model_name}_model.pkl'))
        
        # Save topic distributions
        for model_name, distributions in self.topic_distributions.items():
            pd.DataFrame(distributions).to_csv(
                self.output_dir / f'{model_name}_document_topics.csv'
            )
        
        # Save processed documents
        with open(self.output_dir / 'processed_documents.txt', 'w') as f:
            for doc in self.processed_docs:
                f.write(' '.join(doc) + '\n')

def main():
    """Main execution function."""
    # Configuration
    config = {
        'input_file': 'data/survey_data.csv',
        'output_dir': 'results/topic_modeling',
        'text_columns': ['open_ended_response_1', 'open_ended_response_2'],
        'min_df': 0.01,
        'max_df': 0.95,
        'num_topics_range': (2, 15),
        'random_state': 42
    }
    
    try:
        # Initialize pipeline
        pipeline = TopicModelingPipeline(**config)
        
        # Execute pipeline
        pipeline.load_and_validate_data()
        pipeline.prepare_corpus()
        pipeline.train_models()
        pipeline.analyze_topics()
        pipeline.save_results()
        
        logger.info("Topic modeling pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
