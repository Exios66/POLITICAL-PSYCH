# Political Psychology Cluster Analysis

A comprehensive tool for analyzing political psychology survey data using advanced clustering techniques and interactive visualizations. This project provides researchers and analysts with a robust framework for uncovering patterns and insights in political psychology survey responses through machine learning and statistical analysis.

## Features

- **Data Processing**
  - Automated handling of missing values using advanced imputation techniques (KNN, median, mode)
  - Outlier detection and removal using Isolation Forest and IQR methods
  - Feature normalization with multiple scaling options (Standard, MinMax, Robust)
  - Support for various data formats (CSV, Excel, JSON, SPSS)
  - Data validation and quality checks
  - Automated feature selection and engineering

- **Clustering Analysis**
  - Multiple clustering algorithms:
    - K-means with automatic initialization
    - DBSCAN with adaptive epsilon selection
    - Hierarchical clustering with various linkage methods
    - Gaussian Mixture Models
    - Spectral Clustering
  - Automatic optimal cluster detection using:
    - Elbow method
    - Silhouette analysis
    - Gap statistic
  - Feature importance analysis using:
    - PCA loadings
    - Random Forest feature importance
    - SHAP values
  - Cluster profiling with statistical significance testing
  - Cross-validation of clustering results
  - Cluster stability analysis

- **Interactive GUI**
  - User-friendly interface for data analysis with:
    - Drag-and-drop data loading
    - Interactive parameter selection
    - Real-time progress tracking
  - Real-time visualization with zoom and pan capabilities
  - Interactive plot manipulation:
    - Feature selection
    - Cluster highlighting
    - Dynamic filtering
  - Comprehensive results display with:
    - Statistical summaries
    - Data tables
    - Export options

- **Visualization**
  - Distribution plots:
    - Histograms
    - Kernel density estimates
    - Box plots
    - Violin plots
  - Cluster profiles:
    - Radar charts
    - Heat maps
    - Parallel coordinates
  - Dimensionality reduction visualizations:
    - PCA
    - t-SNE
    - UMAP
  - Feature importance plots:
    - Bar charts
    - Tree maps
    - Network graphs
  - Interactive Plotly dashboards
  - Publication-ready figure export

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/POLITICAL-PSYCH.git
cd POLITICAL-PSYCH
```

1. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### GUI Application

1. Launch the GUI:

```bash
python scripts/Python/cluster.py
```

1. Use the interface to:
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

## Directory Layout

```bash
POLITICAL-PSYCH/
├── config/
│   └── config.yaml           # Configuration settings
├── data/
│   ├── examples/             # Example datasets
│   ├── processed/            # Processed data outputs
│   └── raw/                  # Raw survey data
├── documentation/            # Project documentation
├��─ scripts/
│   └── Python/
│       ├── cluster.py        # Main clustering module
│       ├── visualization.py  # Visualization utilities
│       └── utils.py          # Helper functions
├── tests/                    # Unit tests
└── requirements.txt          # Project dependencies
```

## Survey Metrics

The tool analyzes various political psychology metrics across multiple dimensions:

### News Consumption Patterns

- Traditional Media Consumption
  - Print newspapers and magazines
  - Online news websites and portals
  - Television news programs
  - Radio news broadcasts
  - Frequency and duration of consumption

- Social Media News Behavior
  - Platform-specific consumption (Facebook, Twitter, etc.)
  - News sharing and redistribution patterns
  - Engagement metrics (likes, comments, shares)
  - Content type preferences
  - Network effects in news dissemination

- News Consumption Habits
  - Daily/weekly frequency patterns
  - Time spent per medium
  - Cross-platform consumption behaviors
  - News source diversity metrics
  - Content topic preferences

### Temporal Analysis Features

- Longitudinal Patterns
  - Seasonal variations in news consumption
  - Event-driven consumption spikes
  - Long-term trend analysis
  - Day-of-week effects
  - Time-of-day patterns

- Response Evolution
  - Changes in media preferences over time
  - Platform adoption/abandonment trends
  - Content type preference shifts
  - Engagement pattern changes
  - Demographic cohort effects

## Output Files and Data Products

The analysis pipeline generates a comprehensive set of output files in the `data/processed_data/` directory:

### Core Analysis Files

- `cluster_assignments.csv`
  - Individual-level cluster assignments
  - Confidence scores for assignments
  - Distance to cluster centroids
  - Secondary cluster affiliations
  - Temporal stability metrics

- `feature_importance.csv`
  - Feature contribution scores
  - Statistical significance measures
  - Cross-validation stability metrics
  - Feature correlation matrices
  - Principal component loadings

- `cluster_profiles.csv`
  - Detailed cluster characteristics
  - Demographic breakdowns
  - Behavioral pattern summaries
  - Temporal evolution metrics
  - Inter-cluster distance measures

### Visualization Outputs

- Distribution Plots
  - Feature histograms by cluster
  - Density estimation plots
  - Box plots of key metrics
  - Violin plots for distributions
  - QQ plots for normality assessment

- Interactive Visualizations
  - Dynamic cluster exploration tools
  - Time series animations
  - Network visualization graphs
  - Geographic distribution maps
  - Feature correlation heatmaps

## Contributing

We welcome contributions from researchers and developers. Please follow these steps:

1. Fork the repository to your own GitHub account
2. Create a feature branch with a descriptive name
3. Implement your changes following our coding standards
4. Add appropriate tests and documentation
5. Submit a Pull Request with detailed description
6. Respond to code review feedback

### Development Guidelines

- Follow PEP 8 style guide for Python code
- Include docstrings for all functions and classes
- Write unit tests for new features
- Update documentation for API changes
- Maintain backwards compatibility where possible

## License

This project is licensed under the MIT License - see the LICENSE file for details. Key points:

- Permission to use, copy, modify, and distribute
- License and copyright notice must be included
- No warranty provided
- Authors not liable for damages

## Acknowledgments

### Research Teams

- Survey Design and Data Collection Team at Political Psychology Lab
- Statistical Analysis Group at Data Science Department
- UX Research Team for Interface Design

### Technology Partners

- Political Psychology Research Consortium
- Open Source Community Contributors
- Cloud Computing Partners

### Libraries and Tools

- Scientific Computing: NumPy, Pandas, SciPy
- Machine Learning: Scikit-learn, TensorFlow
- Visualization: Matplotlib, Plotly, Seaborn
- GUI Development: Tkinter, Qt

## Contact and Support

### Issue Reporting

- Use GitHub Issues for bug reports and feature requests
- Include reproducible examples when possible
- Tag issues appropriately (bug/feature/question)

### Communication Channels

- GitHub Discussions for general questions
- Project mailing list for announcements
- Slack channel for development coordination

### Documentation

- Full API documentation available on ReadTheDocs
- Tutorial notebooks in documentation/tutorials
- Regular webinars for major releases
