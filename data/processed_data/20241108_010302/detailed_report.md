# Cluster Analysis Report

Generated on: 2024-11-08 01:03:03

## Analysis Overview

- Clustering Method: KMeans
- Number of Clusters: 3
- Total Samples: 494

## Clustering Metrics

- Silhouette Score: 0.196
  - Measures cluster cohesion and separation (range: [-1, 1], higher is better)
- Davies-Bouldin Score: 1.908
  - Measures average similarity between clusters (lower is better)
- Calinski-Harabasz Score: 94.904
  - Measures cluster density and separation (higher is better)

## Cluster Details

### Cluster 0

- Size: 244 samples (49.4%)
- Key Characteristics:
  - Trad_News_print: -0.80 standard deviations (below average)
  - News_1: +0.70 standard deviations (above average)
  - Trad_News_TV: -0.47 standard deviations (below average)
  - Trad_News_online: +0.30 standard deviations (above average)
  - News_frequency: +0.11 standard deviations (above average)

### Cluster 1

- Size: 125 samples (25.3%)
- Key Characteristics:
  - Trad_News_online: -1.45 standard deviations (below average)
  - News_1: -1.21 standard deviations (below average)
  - News_frequency: +1.02 standard deviations (above average)
  - Trad_News_TV: -0.71 standard deviations (below average)
  - Trad_News_print: +0.71 standard deviations (above average)

### Cluster 2

- Size: 125 samples (25.3%)
- Key Characteristics:
  - Trad_News_TV: +1.64 standard deviations (above average)
  - News_frequency: -1.24 standard deviations (below average)
  - Trad_News_online: +0.86 standard deviations (above average)
  - Trad_News_print: +0.86 standard deviations (above average)
  - News_1: -0.15 standard deviations (below average)

## Visualizations

The following visualizations are available in the 'visualizations' directory:

1. cluster_distribution.png - Shows the size distribution of clusters
2. feature_importance.png - Heatmap showing feature importance by cluster

## Data Files

The following CSV files contain detailed analysis results:

1. cluster_assignments.csv - Original data with cluster assignments
2. cluster_statistics.csv - Statistical summary for each cluster
3. feature_importance.csv - Importance scores for features in each cluster
