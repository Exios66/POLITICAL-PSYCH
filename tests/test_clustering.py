import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from cluster import ClusterAnalysis

class TestClusterAnalysis(unittest.TestCase):

    def setUp(self):
        """Set up test variables and create sample test data"""
        # Create sample test data
        self.test_data_path = Path('tests/test_data.csv')
        self.sample_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.normal(0, 1, 100)
        })
        self.sample_data.to_csv(self.test_data_path, index=False)
        
        # Initialize cluster analysis
        self.cluster_analysis = ClusterAnalysis(file_path=str(self.test_data_path))
        self.cluster_analysis.load_data()
        self.cluster_analysis.clean_data()
        self.cluster_analysis.normalize_features()

    def tearDown(self):
        """Clean up test data file"""
        if self.test_data_path.exists():
            self.test_data_path.unlink()

    def test_load_data(self):
        """Test data loading functionality"""
        self.assertIsNotNone(self.cluster_analysis.data)
        self.assertFalse(self.cluster_analysis.data.empty)
        self.assertEqual(len(self.cluster_analysis.data.columns), 3)
        self.assertEqual(len(self.cluster_analysis.data), 100)

    def test_clean_data(self):
        """Test data cleaning functionality"""
        # Test with clean data
        self.assertIsNotNone(self.cluster_analysis.cleaned_data)
        self.assertFalse(self.cluster_analysis.cleaned_data.isnull().values.any())
        
        # Test with dirty data
        dirty_data = self.sample_data.copy()
        dirty_data.iloc[0,0] = np.nan
        self.cluster_analysis.data = dirty_data
        self.cluster_analysis.clean_data()
        self.assertFalse(self.cluster_analysis.cleaned_data.isnull().values.any())

    def test_normalize_features(self):
        """Test feature normalization functionality"""
        self.assertIsNotNone(self.cluster_analysis.normalized_data)
        means = self.cluster_analysis.normalized_data.mean()
        stds = self.cluster_analysis.normalized_data.std()
        
        # Check means are close to 0
        for mean in means:
            self.assertAlmostEqual(mean, 0.0, places=1)
            
        # Check standard deviations are close to 1
        for std in stds:
            self.assertAlmostEqual(std, 1.0, places=1)

    def test_find_optimal_clusters(self):
        """Test finding optimal clusters with different methods"""
        # Test elbow method
        optimal_k_elbow = self.cluster_analysis.find_optimal_clusters(method='elbow', max_k=10)
        self.assertIsInstance(optimal_k_elbow, int)
        self.assertGreater(optimal_k_elbow, 0)
        self.assertLessEqual(optimal_k_elbow, 10)
        
        # Test silhouette method
        optimal_k_silhouette = self.cluster_analysis.find_optimal_clusters(method='silhouette', max_k=10)
        self.assertIsInstance(optimal_k_silhouette, int)
        self.assertGreater(optimal_k_silhouette, 0)
        self.assertLessEqual(optimal_k_silhouette, 10)

    def test_perform_clustering(self):
        """Test clustering performance with different algorithms"""
        # Test K-means
        self.cluster_analysis.perform_clustering(method='kmeans', n_clusters=3)
        self.assertIsNotNone(self.cluster_analysis.clusters)
        self.assertEqual(len(self.cluster_analysis.clusters), len(self.cluster_analysis.normalized_data))
        
        # Test DBSCAN
        self.cluster_analysis.perform_clustering(method='dbscan', eps=0.5, min_samples=5)
        self.assertIsNotNone(self.cluster_analysis.clusters)
        self.assertEqual(len(self.cluster_analysis.clusters), len(self.cluster_analysis.normalized_data))
        
        # Test Hierarchical
        self.cluster_analysis.perform_clustering(method='hierarchical', n_clusters=3)
        self.assertIsNotNone(self.cluster_analysis.clusters)
        self.assertEqual(len(self.cluster_analysis.clusters), len(self.cluster_analysis.normalized_data))

    def test_perform_dimensionality_reduction(self):
        """Test dimensionality reduction with different methods"""
        # Test PCA
        pca_embeddings = self.cluster_analysis.perform_dimensionality_reduction(method='pca', n_components=2)
        self.assertEqual(pca_embeddings.shape[1], 2)
        
        # Test t-SNE
        tsne_embeddings = self.cluster_analysis.perform_dimensionality_reduction(method='tsne', n_components=2)
        self.assertEqual(tsne_embeddings.shape[1], 2)
        
        # Test UMAP
        umap_embeddings = self.cluster_analysis.perform_dimensionality_reduction(method='umap', n_components=2)
        self.assertEqual(umap_embeddings.shape[1], 2)

    def test_evaluate_clustering(self):
        """Test clustering evaluation metrics"""
        self.cluster_analysis.perform_clustering(method='kmeans', n_clusters=3)
        metrics = self.cluster_analysis.evaluate_clustering()
        
        self.assertIn('silhouette_score', metrics)
        self.assertIn('calinski_harabasz_score', metrics)
        self.assertIn('davies_bouldin_score', metrics)
        
        self.assertGreaterEqual(metrics['silhouette_score'], -1)
        self.assertLessEqual(metrics['silhouette_score'], 1)
        self.assertGreater(metrics['calinski_harabasz_score'], 0)
        self.assertGreater(metrics['davies_bouldin_score'], 0)

if __name__ == '__main__':
    unittest.main()