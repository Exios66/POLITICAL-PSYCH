import re
import logging
import sys
import traceback
import nltk
from nltk.tokenize import sent_tokenize

# Download necessary NLTK data files
nltk.download('punkt')

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def evaluate_open_response(response):
    """
    Evaluate an open-ended response based on the number of distinct steps provided.

    Parameters:
    - response (str): The response describing verification steps.

    Returns:
    - int: Number of distinct steps mentioned in the response.
    """
    # Split the response into sentences using NLTK's sentence tokenizer for more accuracy
    sentences = sent_tokenize(response)
    steps_count = 0

    # Iterate over sentences and evaluate if each describes a distinct verification step
    for sentence in sentences:
        # Check if sentence has meaningful content
        if sentence.strip():
            # Increment steps count (assuming each meaningful sentence is a distinct step)
            steps_count += 1
            logging.debug(f"Identified step: {sentence.strip()}")

    logging.info(f"Total number of distinct steps identified: {steps_count}")
    return steps_count

def compare_news_habits(news_source, news_frequency):
    """
    Compare the news source and frequency to draw some inference about news consumption habits.

    Parameters:
    - news_source (int): The selected news source (1-6).
    - news_frequency (int): The time devoted to news each day (1-4).

    Returns:
    - str: Summary evaluation of news habits.
    """
    # Define news source and frequency interpretations
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
        logging.error(f"Invalid news source value: {news_source}")
        return "Invalid news source provided."
    if news_frequency not in frequencies:
        logging.error(f"Invalid news frequency value: {news_frequency}")
        return "Invalid news frequency provided."

    source = sources.get(news_source)
    frequency = frequencies.get(news_frequency)

    logging.info(f"News Source: {source}")
    logging.info(f"News Frequency: {frequency}")

    # Basic evaluation logic
    if news_source in [3, 4] and news_frequency >= 3:
        evaluation = "High engagement with digital news sources."
    elif news_source in [1, 2, 5, 6] and news_frequency <= 2:
        evaluation = "Moderate engagement with traditional news sources."
    else:
        evaluation = "Mixed engagement with news sources."

    logging.info(f"Evaluation of news habits: {evaluation}")
    return evaluation

def main():
    try:
        # Example open-ended response
        response = (
            "First, I check the source of the news. Then, I cross-check it with other news websites. "
            "I also verify if the information has been reported by any reputable news agency. "
            "Finally, I check the comments and discussions on social media to get different perspectives."
        )
        
        # Evaluate response
        steps_count = evaluate_open_response(response)
        
        # Example news source and frequency inputs with validation
        news_source = 3  # News websites
        news_frequency = 4  # More than 2 hours

        # Validate inputs before calling compare_news_habits
        if news_source not in range(1, 7):
            logging.error(f"Invalid news source value: {news_source}")
            print("Invalid news source provided.")
            return
        if news_frequency not in range(1, 5):
            logging.error(f"Invalid news frequency value: {news_frequency}")
            print("Invalid news frequency provided.")
            return
        
        # Compare news habits
        evaluation = compare_news_habits(news_source, news_frequency)

        # Output results
        print(f"Number of distinct steps identified: {steps_count}")
        print(f"Evaluation of news habits: {evaluation}")

    except Exception as e:
        logging.error("An error occurred in open_ended_analysis.py.")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
