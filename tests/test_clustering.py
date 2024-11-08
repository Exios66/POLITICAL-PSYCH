import unittest
from cluster import ClusterAnalysis

class TestClusterAnalysis(unittest.TestCase):

    def setUp(self):
        """Set up test variables"""
        self.cluster_analysis = ClusterAnalysis(file_path='tests/test_data.csv')
        self.cluster_analysis.load_data()
        self.cluster_analysis.clean_data()
        self.cluster_analysis.normalize_features()

    def test_load_data(self):
        """Test data loading"""
        self.assertIsNotNone(self.cluster_analysis.data)
        self.assertFalse(self.cluster_analysis.data.empty)

    def test_clean_data(self):
        """Test data cleaning"""
        self.assertIsNotNone(self.cluster_analysis.cleaned_data)
        self.assertFalse(self.cluster_analysis.cleaned_data.isnull().values.any())

    def test_normalize_features(self):
        """Test feature normalization"""
        self.assertIsNotNone(self.cluster_analysis.normalized_data)
        means = self.cluster_analysis.normalized_data.mean()
        stds = self.cluster_analysis.normalized_data.std()
        for mean in means:
            self.assertAlmostEqual(mean, 0.0, places=1)
        for std in stds:
            self.assertAlmostEqual(std, 1.0, places=1)

    def test_find_optimal_clusters(self):
        """Test finding optimal clusters"""
        optimal_k = self.cluster_analysis.find_optimal_clusters(method='elbow', max_k=10)
        self.assertIsInstance(optimal_k, int)
        self.assertGreater(optimal_k, 0)

    def test_perform_clustering(self):
        """Test clustering performance"""
        self.cluster_analysis.perform_clustering(method='kmeans', n_clusters=3)
        self.assertIsNotNone(self.cluster_analysis.clusters)
        self.assertEqual(len(self.cluster_analysis.clusters), len(self.cluster_analysis.normalized_data))

    def test_perform_dimensionality_reduction(self):
        """Test dimensionality reduction"""
        embeddings = self.cluster_analysis.perform_dimensionality_reduction(method='pca', n_components=2)
        self.assertEqual(embeddings.shape[1], 2)

if __name__ == '__main__':
    unittest.main()