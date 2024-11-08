import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import logging
import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import gensim
from gensim import corpora
from gensim.models import CoherenceModel

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def load_data(file_path):
    """
    Load survey data from a CSV file.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Loaded data.
    """
    if not os.path.exists(file_path):
        logging.error(f"File {file_path} does not exist.")
        sys.exit(1)
    
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully with shape {data.shape}.")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        sys.exit(1)

def preprocess_text(text):
    """
    Preprocess the input text by cleaning, tokenizing, removing stopwords, and lemmatizing.

    Parameters:
    - text (str): The text to preprocess.

    Returns:
    - List[str]: List of cleaned tokens.
    """
    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords and single-character tokens, and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1]
    
    return tokens

def prepare_corpus(data, text_columns):
    """
    Prepare the corpus for topic modeling by preprocessing text and creating a dictionary and corpus.

    Parameters:
    - data (pd.DataFrame): The survey data.
    - text_columns (List[str]): List of columns containing open-ended responses.

    Returns:
    - List[List[str]]: Tokenized and cleaned documents.
    - corpora.Dictionary: Gensim dictionary.
    - List[List[tuple]]: Corpus in BoW format.
    """
    # Combine all text columns into a single list
    combined_text = data[text_columns].fillna('').agg(' '.join, axis=1).tolist()
    logging.info("Combined open-ended responses into a single text corpus.")

    # Preprocess each document
    processed_docs = [preprocess_text(doc) for doc in combined_text]
    logging.info("Completed text preprocessing (cleaning, tokenizing, stopword removal, lemmatizing).")

    # Create a dictionary representation of the documents.
    dictionary = corpora.Dictionary(processed_docs)
    # Filter out extremes to limit the number of features
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    logging.info(f"Created dictionary with {len(dictionary)} tokens after filtering.")

    # Create the Bag-of-Words corpus
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    logging.info("Created Bag-of-Words corpus.")
    
    return processed_docs, dictionary, corpus

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=1):
    """
    Compute c_v coherence for various number of topics.

    Parameters:
    - dictionary (corpora.Dictionary): Gensim dictionary.
    - corpus (List[List[tuple]]): Corpus in BoW format.
    - texts (List[List[str]]): Tokenized texts.
    - limit (int): Max number of topics.
    - start (int): Starting number of topics.
    - step (int): Step size.

    Returns:
    - model_list (List[gensim.models.LdaModel]): List of LDA models.
    - coherence_values (List[float]): Coherence scores corresponding to the LDA model list.
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit + 1, step):
        model = gensim.models.LdaModel(corpus=corpus,
                                       id2word=dictionary,
                                       num_topics=num_topics,
                                       random_state=42,
                                       update_every=1,
                                       chunksize=100,
                                       passes=10,
                                       alpha='auto',
                                       per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence = coherencemodel.get_coherence()
        coherence_values.append(coherence)
        logging.info(f"Computed coherence for {num_topics} topics: {coherence:.4f}")
    return model_list, coherence_values

def visualize_coherence(start, limit, step, coherence_values):
    """
    Plot coherence scores to help determine the optimal number of topics.

    Parameters:
    - start (int): Starting number of topics.
    - limit (int): Maximum number of topics.
    - step (int): Step size.
    - coherence_values (List[float]): Coherence scores.
    """
    x = range(start, limit + 1, step)
    plt.figure(figsize=(10,6))
    plt.plot(x, coherence_values, marker='o')
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.title("Coherence Scores by Number of Topics")
    plt.xticks(x)
    plt.grid(True)
    plt.savefig('cluster_analysis/coherence_scores.png')
    plt.close()
    logging.info("Coherence scores plot saved as 'cluster_analysis/coherence_scores.png'.")

def select_optimal_model(model_list, coherence_values, start, limit, step):
    """
    Select the LDA model with the highest coherence score.

    Parameters:
    - model_list (List[gensim.models.LdaModel]): List of LDA models.
    - coherence_values (List[float]): Coherence scores.
    - start (int): Starting number of topics.
    - limit (int): Maximum number of topics.
    - step (int): Step size.

    Returns:
    - gensim.models.LdaModel: Optimal LDA model.
    - int: Number of topics.
    """
    max_coherence = max(coherence_values)
    optimal_index = coherence_values.index(max_coherence)
    optimal_num_topics = start + optimal_index * step
    optimal_model = model_list[optimal_index]
    logging.info(f"Optimal number of topics selected: {optimal_num_topics} with coherence score {max_coherence:.4f}.")
    return optimal_model, optimal_num_topics

def assign_topic_distribution(optimal_model, corpus, num_topics):
    """
    Assign topic distribution to each document.

    Parameters:
    - optimal_model (gensim.models.LdaModel): The optimal LDA model.
    - corpus (List[List[tuple]]): Corpus in BoW format.
    - num_topics (int): Number of topics.

    Returns:
    - pd.DataFrame: DataFrame containing topic distribution for each document.
    """
    topic_distributions = []
    for bow in corpus:
        topic_probs = optimal_model.get_document_topics(bow, minimum_probability=0)
        topic_probs_sorted = sorted(topic_probs, key=lambda x: x[0])
        topic_probs_only = [prob for _, prob in topic_probs_sorted]
        topic_distributions.append(topic_probs_only)
    
    topic_df = pd.DataFrame(topic_distributions, columns=[f'Topic_{i+1}' for i in range(num_topics)])
    logging.info("Assigned topic distributions to each document.")
    return topic_df

def visualize_topics(optimal_model, corpus, dictionary, output_dir='cluster_analysis'):
    """
    Visualize the topics using pyLDAvis.

    Parameters:
    - optimal_model (gensim.models.LdaModel): The optimal LDA model.
    - corpus (List[List[tuple]]): Corpus in BoW format.
    - dictionary (corpora.Dictionary): Gensim dictionary.
    - output_dir (str): Directory to save the visualization.
    """
    lda_display = gensimvis.prepare(optimal_model, corpus, dictionary, sort_topics=False)
    pyLDAvis.save_html(lda_display, os.path.join(output_dir, 'lda_visualization.html'))
    logging.info("LDA visualization saved as 'cluster_analysis/lda_visualization.html'.")

def main():
    # File paths
    input_file = 'survey_data.csv'
    output_dir = 'cluster_analysis'
    topic_output_file = 'topic_distributions.csv'

    # Create output directory if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created directory '{output_dir}' for analysis outputs.")

    # Load data
    data = load_data(input_file)

    # Define open-ended response columns
    text_columns = ['Define_fake_news_prop', 'Detect_news_verification']
    missing_columns = [col for col in text_columns if col not in data.columns]
    if missing_columns:
        logging.error(f"The following required text columns are missing from the data: {missing_columns}")
        sys.exit(1)
    
    # Prepare corpus
    processed_docs, dictionary, corpus = prepare_corpus(data, text_columns)

    # Compute coherence values to determine optimal number of topics
    start, limit, step = 2, 10, 1
    model_list, coherence_values = compute_coherence_values(dictionary, corpus, processed_docs, limit, start, step)

    # Visualize coherence scores
    visualize_coherence(start, limit, step, coherence_values)

    # Select the optimal model
    optimal_model, optimal_num_topics = select_optimal_model(model_list, coherence_values, start, limit, step)

    # Visualize topics with pyLDAvis
    visualize_topics(optimal_model, corpus, dictionary, output_dir=output_dir)

    # Assign topic distributions to documents
    topic_df = assign_topic_distribution(optimal_model, corpus, optimal_num_topics)

    # Save topic distributions
    topic_df.to_csv(os.path.join(output_dir, topic_output_file), index=False)
    logging.info(f"Topic distributions saved as '{os.path.join(output_dir, topic_output_file)}'.")

    # Optionally, save the LDA model for future use
    optimal_model.save(os.path.join(output_dir, f'lda_model_{optimal_num_topics}_topics.model'))
    logging.info(f"Optimal LDA model saved as '{os.path.join(output_dir, f'lda_model_{optimal_num_topics}_topics.model')}'.")

    logging.info("Topic modeling completed successfully.")

if __name__ == "__main__":
    main()