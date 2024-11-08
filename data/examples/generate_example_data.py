import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
import datetime
from pathlib import Path

def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        'data',
        'data/raw',
        'data/processed',
        'data/examples',
        'results',
        'results/plots',
        'results/models',
        'results/reports',
        'config'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def generate_survey_data(n_samples=1000, random_state=42):
    """Generate synthetic survey data"""
    np.random.seed(random_state)
    
    # Generate base clusters
    X, y = make_blobs(n_samples=n_samples, n_features=5, centers=4, random_state=random_state)
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=['News_1', 'News_frequency', 'Trad_News_print', 
                                'Trad_News_online', 'Trad_News_TV'])
    
    # Add more features
    df['Trad_News_radio'] = np.random.normal(0, 1, n_samples)
    
    # Social media features
    for i in range(1, 7):
        df[f'SM_News_{i}'] = np.random.choice([0, 1, 2, 3, 4], n_samples)
    
    # Sharing behavior
    df['SM_Sharing'] = np.random.choice([0, 1, 2, 3], n_samples)
    
    # Add some missing values
    mask = np.random.random(df.shape) < 0.05
    df[mask] = np.nan
    
    # Add timestamps
    dates = pd.date_range(
        start='2023-01-01',
        end='2023-12-31',
        periods=n_samples
    )
    df['survey_date'] = dates
    
    return df

def main():
    # Ensure directories exist
    ensure_directories()
    
    # Generate example dataset
    df = generate_survey_data()
    
    # Save to CSV
    df.to_csv('data/examples/survey_example.csv', index=False)
    print(f"Saved example dataset to data/examples/survey_example.csv")
    
    # Generate some additional test cases
    small_df = generate_survey_data(n_samples=100)
    large_df = generate_survey_data(n_samples=5000)
    
    small_df.to_csv('data/examples/survey_small.csv', index=False)
    print(f"Saved small dataset to data/examples/survey_small.csv")
    
    large_df.to_csv('data/examples/survey_large.csv', index=False)
    print(f"Saved large dataset to data/examples/survey_large.csv")

if __name__ == "__main__":
    main() 