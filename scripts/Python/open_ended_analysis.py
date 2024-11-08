import re
import logging
import sys
import traceback
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
from typing import Dict, List, Tuple, Union

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('open_ended_analysis.log')
    ]
)

logger = logging.getLogger(__name__)

def evaluate_open_response(response: str) -> Dict[str, Union[int, List[str], float]]:
    """
    Evaluate an open-ended response based on multiple metrics including:
    - Number of distinct steps
    - Key themes/topics
    - Sentiment analysis
    - Response complexity

    Parameters:
    - response (str): The response describing verification steps.

    Returns:
    - Dict containing analysis metrics:
        - steps_count: Number of distinct steps
        - key_themes: List of main themes identified
        - sentiment_score: Overall sentiment score
        - complexity_score: Measure of response complexity
    """
    try:
        # Initialize sentiment analyzer
        sia = SentimentIntensityAnalyzer()
        
        # Split into sentences and words
        sentences = sent_tokenize(response)
        words = word_tokenize(response.lower())
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words and w.isalnum()]
        
        # Count distinct steps
        steps_count = len(sentences)
        
        # Identify key themes (most common meaningful words)
        word_freq = Counter(words)
        key_themes = [word for word, count in word_freq.most_common(5)]
        
        # Calculate sentiment
        sentiment_scores = sia.polarity_scores(response)
        sentiment_score = sentiment_scores['compound']
        
        # Calculate complexity score based on:
        # - Average sentence length
        # - Vocabulary diversity
        # - Number of steps
        avg_sent_length = len(words) / len(sentences)
        vocab_diversity = len(set(words)) / len(words)
        complexity_score = (avg_sent_length * 0.3) + (vocab_diversity * 0.4) + (steps_count * 0.3)
        
        results = {
            'steps_count': steps_count,
            'key_themes': key_themes,
            'sentiment_score': sentiment_score,
            'complexity_score': round(complexity_score, 2)
        }
        
        logger.info(f"Response analysis complete: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Error in evaluate_open_response: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def compare_news_habits(news_source: int, news_frequency: int, 
                       additional_sources: List[int] = None) -> Dict[str, str]:
    """
    Compare news consumption habits and provide detailed analysis.

    Parameters:
    - news_source (int): Primary news source (1-6)
    - news_frequency (int): Time devoted to news daily (1-4)
    - additional_sources (List[int], optional): Secondary news sources

    Returns:
    - Dict containing:
        - evaluation: Overall evaluation
        - diversity_score: Source diversity assessment
        - recommendations: Suggested improvements
    """
    try:
        sources = {
            1: 'Print newspapers',
            2: 'Television',
            3: 'News websites',
            4: 'Social media',
            5: 'Radio',
            6: 'Podcasts'
        }
        
        frequencies = {
            1: 'Fewer than 30 minutes',
            2: 'Between 30 minutes and 1 hour',
            3: 'Between 1 and 2 hours',
            4: 'More than 2 hours'
        }

        # Validate inputs
        if news_source not in sources:
            raise ValueError(f"Invalid news source value: {news_source}")
        if news_frequency not in frequencies:
            raise ValueError(f"Invalid news frequency value: {news_frequency}")

        source = sources[news_source]
        frequency = frequencies[news_frequency]
        
        # Calculate source diversity score
        source_count = 1
        if additional_sources:
            source_count += len(set(additional_sources))
        diversity_score = min(source_count / len(sources), 1.0)
        
        # Generate evaluation
        if news_source in [3, 4] and news_frequency >= 3:
            evaluation = "High engagement with digital news sources."
            recommendations = "Consider incorporating more traditional sources for balance."
        elif news_source in [1, 2, 5, 6] and news_frequency <= 2:
            evaluation = "Moderate engagement with traditional news sources."
            recommendations = "Consider increasing news consumption and exploring digital sources."
        else:
            evaluation = "Mixed engagement with news sources."
            recommendations = "Focus on maintaining diverse news diet while increasing engagement."
            
        if diversity_score < 0.3:
            recommendations += " Strongly recommend diversifying news sources."
        
        results = {
            'evaluation': evaluation,
            'diversity_score': f"{diversity_score:.2f}",
            'recommendations': recommendations,
            'primary_source': source,
            'frequency': frequency
        }
        
        logger.info(f"News habits analysis complete: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Error in compare_news_habits: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def main():
    try:
        # Example open-ended response
        response = (
            "First, I check the source of the news. Then, I cross-check it with other news websites. "
            "I also verify if the information has been reported by any reputable news agency. "
            "Finally, I check the comments and discussions on social media to get different perspectives."
        )
        
        # Analyze open-ended response
        response_analysis = evaluate_open_response(response)
        
        # Example news consumption analysis
        news_source = 3  # News websites
        news_frequency = 4  # More than 2 hours
        additional_sources = [2, 4]  # TV and social media
        
        # Analyze news habits
        habits_analysis = compare_news_habits(
            news_source, 
            news_frequency,
            additional_sources
        )

        # Output results
        print("\nOpen Response Analysis:")
        print(f"Steps identified: {response_analysis['steps_count']}")
        print(f"Key themes: {', '.join(response_analysis['key_themes'])}")
        print(f"Sentiment score: {response_analysis['sentiment_score']:.2f}")
        print(f"Complexity score: {response_analysis['complexity_score']}")
        
        print("\nNews Habits Analysis:")
        print(f"Primary source: {habits_analysis['primary_source']}")
        print(f"Frequency: {habits_analysis['frequency']}")
        print(f"Source diversity score: {habits_analysis['diversity_score']}")
        print(f"Evaluation: {habits_analysis['evaluation']}")
        print(f"Recommendations: {habits_analysis['recommendations']}")

    except Exception as e:
        logger.error("An error occurred in open_ended_analysis.py")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
