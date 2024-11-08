import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import logging

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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

def clean_data(data):
    """
    Clean the survey data by handling missing values and ensuring correct data types.

    Parameters:
    - data (pd.DataFrame): Raw data.

    Returns:
    - pd.DataFrame: Cleaned data.
    """
    # Identify columns to drop or impute
    # For simplicity, we'll drop rows with missing values
    initial_shape = data.shape
    data = data.dropna()
    final_shape = data.shape
    logging.info(f"Dropped {initial_shape[0] - final_shape[0]} rows due to missing values.")
    
    # Ensure categorical columns are of type integer
    categorical_columns = [
        'News_1',
        'News_frequency',
        'Trad_News_print',
        'Trad_News_online',
        'Trad_News_TV',
        'Trad_News_radio',
        'SM_News_1',
        'SM_News_2',
        'SM_News_3',
        'SM_News_4',
        'SM_News_5',
        'SM_News_6',
        'SM_Sharing'
    ]
    
    for col in categorical_columns:
        if col in data.columns:
            data[col] = data[col].astype(int)
        else:
            logging.warning(f"Expected column '{col}' not found in data.")
    
    return data

def select_features(data):
    """
    Select relevant features for clustering.

    Parameters:
    - data (pd.DataFrame): Cleaned data.

    Returns:
    - pd.DataFrame: Selected features.
    """
    feature_columns = [
        'News_1',
        'News_frequency',
        'Trad_News_print',
        'Trad_News_online',
        'Trad_News_TV',
        'Trad_News_radio',
        'SM_News_1',
        'SM_News_2',
        'SM_News_3',
        'SM_News_4',
        'SM_News_5',
        'SM_News_6',
        'SM_Sharing'
    ]
    
    missing_features = [col for col in feature_columns if col not in data.columns]
    if missing_features:
        logging.error(f"The following required features are missing from the data: {missing_features}")
        sys.exit(1)
    
    features = data[feature_columns]
    logging.info(f"Selected {features.shape[1]} features for clustering.")
    return features

def preprocess_features(features):
    """
    Normalize the feature data using StandardScaler.

    Parameters:
    - features (pd.DataFrame): Selected features.

    Returns:
    - np.ndarray: Normalized feature array.
    - StandardScaler: Fitted scaler object.
    """
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    logging.info("Features have been standardized (zero mean, unit variance).")
    return scaled_features, scaler

def determine_optimal_clusters(scaled_features, max_k=10):
    """
    Determine the optimal number of clusters using Elbow Method and Silhouette Score.

    Parameters:
    - scaled_features (np.ndarray): Normalized features.
    - max_k (int): Maximum number of clusters to test.

    Returns:
    - int: Optimal number of clusters.
    """
    wcss = []
    silhouette_scores = []
    
    logging.info("Determining the optimal number of clusters using Elbow Method and Silhouette Score.")
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        wcss.append(kmeans.inertia_)
        score = silhouette_score(scaled_features, kmeans.labels_)
        silhouette_scores.append(score)
        logging.info(f"k={k}: WCSS={kmeans.inertia_:.2f}, Silhouette Score={score:.4f}")
    
    # Plot Elbow Method
    plt.figure(figsize=(10, 5))
    plt.plot(range(2, max_k + 1), wcss, marker='o')
    plt.title('Elbow Method for Determining Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.xticks(range(2, max_k + 1))
    plt.grid(True)
    plt.savefig('elbow_method.png')
    plt.close()
    logging.info("Elbow Method plot saved as 'elbow_method.png'.")
    
    # Plot Silhouette Scores
    plt.figure(figsize=(10, 5))
    plt.plot(range(2, max_k + 1), silhouette_scores, marker='o', color='orange')
    plt.title('Silhouette Scores for Different k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.xticks(range(2, max_k + 1))
    plt.grid(True)
    plt.savefig('silhouette_scores.png')
    plt.close()
    logging.info("Silhouette Scores plot saved as 'silhouette_scores.png'.")
    
    # Determine optimal k (for simplicity, choose k with highest silhouette score)
    optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2  # since k starts at 2
    logging.info(f"Optimal number of clusters determined to be k={optimal_k}.")
    
    return optimal_k

def perform_clustering(scaled_features, n_clusters):
    """
    Perform K-Means clustering.

    Parameters:
    - scaled_features (np.ndarray): Normalized features.
    - n_clusters (int): Number of clusters.

    Returns:
    - KMeans: Fitted KMeans object.
    - np.ndarray: Cluster labels.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)
    logging.info(f"K-Means clustering performed with k={n_clusters}.")
    return kmeans, cluster_labels

def analyze_clusters(data, cluster_labels, output_dir='cluster_analysis'):
    """
    Analyze and visualize the clusters.

    Parameters:
    - data (pd.DataFrame): Original data with features.
    - cluster_labels (np.ndarray): Assigned cluster labels.
    - output_dir (str): Directory to save analysis outputs.

    Returns:
    - pd.DataFrame: Data with cluster labels.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created directory '{output_dir}' for analysis outputs.")
    
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = cluster_labels
    
    # Descriptive statistics
    cluster_summary = data_with_clusters.groupby('Cluster').mean().reset_index()
    cluster_summary.to_csv(os.path.join(output_dir, 'cluster_summary.csv'), index=False)
    logging.info("Cluster summary statistics saved as 'cluster_summary.csv'.")
    
    # Save cluster counts
    cluster_counts = data_with_clusters['Cluster'].value_counts().sort_index()
    cluster_counts.to_csv(os.path.join(output_dir, 'cluster_counts.csv'), header=['Count'])
    logging.info("Cluster counts saved as 'cluster_counts.csv'.")
    
    # Visualization: Radar Charts for each cluster
    feature_columns = data.columns.tolist()
    feature_columns.remove('Define_fake_news_prop')  # Exclude open-ended questions
    feature_columns.remove('Detect_news_verification')  # Exclude open-ended questions
    
    # Normalize for radar chart
    radar_data = cluster_summary[cluster_summary.columns[1:-1]]
    radar_data_normalized = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min())
    
    # Radar chart function
    def plot_radar(row, features, title, save_path):
        angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
        angles += angles[:1]
        values = row.tolist()
        values += values[:1]
        
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        plt.xticks(angles[:-1], features, color='grey', size=8)
        ax.plot(angles, values, linewidth=1, linestyle='solid')
        ax.fill(angles, values, alpha=0.4)
        plt.title(title, size=14, y=1.1)
        plt.savefig(save_path)
        plt.close()
    
    for _, row in cluster_summary.iterrows():
        cluster = row['Cluster']
        features = cluster_summary.columns[1:-1]
        values = radar_data_normalized.iloc[row.name].values
        plot_radar(
            row=values,
            features=features,
            title=f'Cluster {int(cluster)} Radar Chart',
            save_path=os.path.join(output_dir, f'cluster_{int(cluster)}_radar.png')
        )
        logging.info(f"Radar chart for Cluster {int(cluster)} saved as 'cluster_{int(cluster)}_radar.png'.")
    
    # Box plots for each feature by cluster
    for feature in feature_columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Cluster', y=feature, data=data_with_clusters)
        plt.title(f'Box Plot of {feature} by Cluster')
        plt.savefig(os.path.join(output_dir, f'boxplot_{feature}.png'))
        plt.close()
        logging.info(f"Box plot for '{feature}' saved as 'boxplot_{feature}.png'.")
    
    return data_with_clusters

def save_clustered_data(data_with_clusters, output_file='clustered_survey_data.csv'):
    """
    Save the clustered data to a CSV file.

    Parameters:
    - data_with_clusters (pd.DataFrame): Data with cluster labels.
    - output_file (str): Output CSV file name.
    """
    data_with_clusters.to_csv(output_file, index=False)
    logging.info(f"Clustered data saved as '{output_file}'.")

def main():
    # File paths
    input_file = 'survey_data.csv'
    clustered_output_file = 'clustered_survey_data.csv'
    analysis_output_dir = 'cluster_analysis'
    
    # Load data
    data = load_data(input_file)
    
    # Clean data
    data_clean = clean_data(data)
    
    # Select features
    features = select_features(data_clean)
    
    # Preprocess features
    scaled_features, scaler = preprocess_features(features)
    
    # Determine optimal number of clusters
    optimal_k = determine_optimal_clusters(scaled_features, max_k=10)
    
    # Perform clustering
    kmeans_model, cluster_labels = perform_clustering(scaled_features, n_clusters=optimal_k)
    
    # Analyze clusters
    data_with_clusters = analyze_clusters(data_clean, cluster_labels, output_dir=analysis_output_dir)
    
    # Save clustered data
    save_clustered_data(data_with_clusters, output_file=clustered_output_file)
    
    logging.info("Clustering and analysis completed successfully.")

if __name__ == "__main__":
    main()