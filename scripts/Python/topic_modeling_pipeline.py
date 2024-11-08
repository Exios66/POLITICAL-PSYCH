import logging
import sys
from zipfile import Path
from gensim.matutils import hellinger
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from wordcloud import WordCloud


class TopicModelingPipeline:
    def __init__(self, input_file, output_dir, text_columns):
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.text_columns = text_columns
        self.models = {}
        self.num_topics_range = (2, 15)
        self.data = None
        self.corpus = None
        self.dictionary = None
        self.model_metrics = {}
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('topic_modeling.log')
            ]
        )

    def load_and_validate_data(self):
        """Load data from CSV file and validate contents"""
        try:
            self.data = pd.read_csv(self.input_file)
            
            # Validate text columns exist
            missing_cols = [col for col in self.text_columns if col not in self.data.columns]
            if missing_cols:
                raise ValueError(f"Missing text columns: {missing_cols}")
                
            # Remove rows with missing text
            self.data = self.data.dropna(subset=self.text_columns)
            
            if len(self.data) == 0:
                raise ValueError("No valid text data found after cleaning")
                
            logging.info(f"Successfully loaded {len(self.data)} documents")
            return self.data
            
        except Exception as e:
            error_msg = f"Error loading data: {str(e)}"
            logging.error(error_msg)
            raise

    def prepare_corpus(self):
        """Preprocess text and create corpus"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_and_validate_data() first.")
            
        try:
            # Combine text columns
            texts = self.data[self.text_columns].fillna('').agg(' '.join, axis=1)
            
            # Preprocess texts
            processed_texts = []
            for text in texts:
                # Tokenize and clean
                tokens = text.lower().split()
                # Remove stopwords, punctuation, numbers
                tokens = [token for token in tokens if token.isalpha()]
                processed_texts.append(tokens)
                
            # Create dictionary and corpus
            self.dictionary = Dictionary(processed_texts)
            self.corpus = [self.dictionary.doc2bow(text) for text in processed_texts]
            
            logging.info("Successfully prepared corpus")
            return self.corpus
            
        except Exception as e:
            error_msg = f"Error preparing corpus: {str(e)}"
            logging.error(error_msg)
            raise

    def train_models(self):
        """Train topic models with different parameters"""
        if self.corpus is None:
            raise ValueError("Corpus not prepared. Call prepare_corpus() first.")
            
        try:
            for num_topics in range(self.num_topics_range[0], self.num_topics_range[1]+1):
                # Train LDA model
                lda_model = LdaModel(
                    corpus=self.corpus,
                    id2word=self.dictionary,
                    num_topics=num_topics,
                    random_state=42
                )
                
                # Calculate coherence score
                coherence_model = CoherenceModel(
                    model=lda_model, 
                    texts=self.corpus,
                    dictionary=self.dictionary,
                    coherence='c_v'
                )
                coherence_score = coherence_model.get_coherence()
                
                self.models[num_topics] = {
                    'model': lda_model,
                    'coherence': coherence_score
                }
                
                logging.info(f"Trained model with {num_topics} topics. Coherence: {coherence_score:.4f}")
                
            return self.models
            
        except Exception as e:
            error_msg = f"Error training models: {str(e)}"
            logging.error(error_msg)
            raise

    def analyze_topics(self):
        """Analyze topic quality and generate insights"""
        if not self.models:
            raise ValueError("No models trained. Call train_models() first.")
            
        try:
            analysis = {}
            
            for num_topics, model_info in self.models.items():
                model = model_info['model']
                
                # Get top words for each topic
                topics = {}
                for topic_id in range(num_topics):
                    top_words = model.show_topic(topic_id, topn=10)
                    topics[topic_id] = {
                        'words': [word for word, prob in top_words],
                        'probabilities': [prob for word, prob in top_words]
                    }
                    
                # Calculate topic distances
                topic_distances = np.zeros((num_topics, num_topics))
                for i in range(num_topics):
                    for j in range(num_topics):
                        distance = hellinger(
                            model.get_topic_terms(i),
                            model.get_topic_terms(j)
                        )
                        topic_distances[i,j] = distance
                        
                analysis[num_topics] = {
                    'topics': topics,
                    'distances': topic_distances,
                    'coherence': model_info['coherence']
                }
                
            self.analysis = analysis
            return analysis
            
        except Exception as e:
            error_msg = f"Error analyzing topics: {str(e)}"
            logging.error(error_msg)
            raise

    def save_results(self):
        """Save analysis results and visualizations"""
        if not hasattr(self, 'analysis'):
            raise ValueError("No analysis results. Call analyze_topics() first.")
            
        try:
            # Create output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model metrics
            metrics_df = pd.DataFrame([
                {
                    'num_topics': k,
                    'coherence': v['coherence']
                }
                for k,v in self.analysis.items()
            ])
            metrics_df.to_csv(self.output_dir / 'model_metrics.csv', index=False)
            
            # Save topic details
            for num_topics, results in self.analysis.items():
                topic_df = pd.DataFrame([
                    {
                        'topic_id': topic_id,
                        'words': ', '.join(topic_info['words']),
                        'probabilities': topic_info['probabilities']
                    }
                    for topic_id, topic_info in results['topics'].items()
                ])
                topic_df.to_csv(self.output_dir / f'topics_{num_topics}.csv', index=False)
                
            # Generate visualizations
            self._visualize_topics()
            self._create_wordclouds()
            
            logging.info(f"Results saved to {self.output_dir}")
            
        except Exception as e:
            error_msg = f"Error saving results: {str(e)}"
            logging.error(error_msg)
            raise

    def _visualize_topics(self):
        """Generate topic visualization plots"""
        try:
            for num_topics, results in self.analysis.items():
                # Topic distance heatmap
                plt.figure(figsize=(10,8))
                sys.heatmap(
                    results['distances'],
                    annot=True,
                    cmap='YlOrRd',
                    xticklabels=[f'Topic {i}' for i in range(num_topics)],
                    yticklabels=[f'Topic {i}' for i in range(num_topics)]
                )
                plt.title(f'Topic Distances (k={num_topics})')
                plt.tight_layout()
                plt.savefig(self.output_dir / f'topic_distances_{num_topics}.png')
                plt.close()
                
        except Exception as e:
            logging.error(f"Error creating visualizations: {str(e)}")
            raise

    def _create_wordclouds(self):
        """Generate word clouds for each topic"""
        try:
            for num_topics, results in self.analysis.items():
                for topic_id, topic_info in results['topics'].items():
                    # Create word frequency dict
                    word_freq = dict(zip(topic_info['words'], topic_info['probabilities']))
                    
                    # Generate wordcloud
                    wordcloud = WordCloud(
                        width=800,
                        height=400,
                        background_color='white'
                    ).generate_from_frequencies(word_freq)
                    
                    # Save plot
                    plt.figure(figsize=(10,5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    plt.title(f'Topic {topic_id} (k={num_topics})')
                    plt.savefig(self.output_dir / f'wordcloud_k{num_topics}_topic{topic_id}.png')
                    plt.close()
                    
        except Exception as e:
            logging.error(f"Error creating wordclouds: {str(e)}")
            raise