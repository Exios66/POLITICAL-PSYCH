import pytest
import numpy as np
import pandas as pd
from scripts.Python.cluster import ClusterAnalysis
from data.examples.generate_example_data import generate_survey_data

@pytest.fixture
def sample_data():
    """Generate sample data for testing"""
    return generate_survey_data(n_samples=100)

@pytest.fixture
def cluster_analyzer(sample_data):
    """Create ClusterAnalysis instance with sample data"""
    sample_data.to_csv('test_data.csv', index=False)
    analyzer = ClusterAnalysis('test_data.csv')
    return analyzer

def test_load_data(cluster_analyzer):
    """Test data loading functionality"""
    data = cluster_analyzer.load_data()
    assert isinstance(data, pd.DataFrame)
    assert not data.empty

def test_clean_data(cluster_analyzer):
    """Test data cleaning functionality"""
    cluster_analyzer.load_data()
    cleaned_data = cluster_analyzer.clean_data()
    assert cleaned_data.isnull().sum().sum() == 0

def test_normalize_features(cluster_analyzer):
    """Test feature normalization"""
    cluster_analyzer.load_data()
    cluster_analyzer.clean_data()
    normalized = cluster_analyzer.normalize_features()
    assert isinstance(normalized, np.ndarray)
    assert np.allclose(normalized.mean(axis=0), 0, atol=1e-10)

def test_perform_clustering(cluster_analyzer):
    """Test clustering functionality"""
    cluster_analyzer.load_data()
    cluster_analyzer.clean_data()
    cluster_analyzer.normalize_features()
    results = cluster_analyzer.perform_clustering(method='kmeans', n_clusters=3)
    assert hasattr(results, 'labels')
    assert len(np.unique(results.labels)) == 3 