# Political Psychology Cluster Analysis

A comprehensive tool for analyzing political psychology survey data using advanced clustering techniques and interactive visualizations.

## Features

- **Data Processing**
  - Automated handling of missing values
  - Outlier detection and removal
  - Feature normalization
  - Support for various data formats

- **Clustering Analysis**
  - Multiple clustering algorithms (K-means, DBSCAN, Hierarchical)
  - Automatic optimal cluster detection
  - Feature importance analysis
  - Cluster profiling

- **Interactive GUI**
  - User-friendly interface for data analysis
  - Real-time visualization
  - Interactive plot manipulation
  - Comprehensive results display

- **Visualization**
  - Distribution plots
  - Cluster profiles
  - Dimensionality reduction visualizations
  - Feature importance plots

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/POLITICAL-PSYCH.git
cd POLITICAL-PSYCH
```

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### GUI Application

1. Launch the GUI:

```bash
python scripts/Python/cluster.py
```

2. Use the interface to:
   - Load and process survey data
   - Configure clustering parameters
   - Generate visualizations
   - Export results

### Data Processing Options

- **Handle Missing Values**: Automatically fills missing values using appropriate methods for numeric (median) and categorical (mode) data
- **Remove Outliers**: Identifies and removes statistical outliers using the IQR method
- **Normalize Features**: Standardizes numeric features to have zero mean and unit variance

### Clustering Options

- **K-means**: Traditional centroid-based clustering
- **DBSCAN**: Density-based clustering for non-spherical clusters
- **Hierarchical**: Agglomerative clustering with dendrograms

### Visualization Types

- **Distribution**: Shows feature distributions across clusters
- **Profile**: Displays cluster centroids across features
- **Reduction**: PCA-based 2D visualization of clusters

## Project Structure

```
POLITICAL-PSYCH/
├── config/
│   └── config.yaml           # Configuration settings
├── data/
│   ├── examples/             # Example datasets
│   ├── processed/            # Processed data outputs
│   └── raw/                  # Raw survey data
├── documentation/            # Project documentation
├── scripts/
│   └── Python/
│       ├── cluster.py        # Main clustering module
│       ├── visualization.py  # Visualization utilities
│       └── utils.py         # Helper functions
├── tests/                    # Unit tests
└── requirements.txt         # Project dependencies
```

## Survey Metrics

The tool analyzes various political psychology metrics:

### News Consumption

- Traditional news consumption (print, online, TV, radio)
- Social media news consumption
- News sharing behavior
- News consumption frequency

### Temporal Analysis

- Survey date patterns
- Temporal trends in responses

## Output Files

The analysis generates several output files in the `data/processed_data/` directory:

- `cluster_assignments.csv`: Cluster labels for each data point
- `feature_importance.csv`: Importance scores for each feature
- `cluster_profiles.csv`: Characteristic profiles of each cluster
- Visualization plots in various formats

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Survey data collection team
- Political psychology research group
- Open source libraries used in development

## Contact

For questions and support, please open an issue in the GitHub repository.
