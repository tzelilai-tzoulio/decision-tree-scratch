"""
Proper test structure separating unit tests from integration tests
"""

import unittest
import numpy as np
import pandas as pd
from decision_tree import DecisionTree, Node, calculate_entropy, calculate_information_gain, find_best_split

# ============================================================================
# UNIT TESTS - Test individual functions in isolation
# ============================================================================

class TestEntropyCalculation(unittest.TestCase):
    """Unit tests for entropy calculation"""
    
    def test_entropy_pure_node(self):
        """Entropy of pure node should be 0"""
        y = np.array([1, 1, 1, 1])
        entropy = calculate_entropy(y)
        self.assertAlmostEqual(entropy, 0.0)
    
    def test_entropy_maximum_impurity(self):
        """Entropy of 50-50 split should be 1.0 (binary)"""
        y = np.array([0, 0, 1, 1])
        entropy = calculate_entropy(y)
        self.assertAlmostEqual(entropy, 1.0)
    
    def test_entropy_multiclass(self):
        """Entropy calculation for multiple classes"""
        y = np.array([0, 1, 2])  # Equal distribution of 3 classes
        entropy = calculate_entropy(y)
        expected = -3 * (1/3 * np.log2(1/3))  # ≈ 1.585
        self.assertAlmostEqual(entropy, expected, places=3)
    
    def test_entropy_empty_array(self):
        """Entropy of empty array should handle gracefully"""
        y = np.array([])
        entropy = calculate_entropy(y)
        self.assertEqual(entropy, 0.0)
    
    def test_entropy_single_sample(self):
        """Entropy of single sample should be 0"""
        y = np.array([1])
        entropy = calculate_entropy(y)
        self.assertEqual(entropy, 0.0)


class TestInformationGain(unittest.TestCase):
    """Unit tests for information gain calculation"""
    
    def test_information_gain_perfect_split(self):
        """Perfect split should give maximum information gain"""
        parent = np.array([0, 0, 1, 1])
        left = np.array([0, 0])
        right = np.array([1, 1])
        gain = calculate_information_gain(parent, left, right)
        self.assertAlmostEqual(gain, 1.0)
    
    def test_information_gain_no_split(self):
        """No split (all data to one side) should give 0 gain"""
        parent = np.array([0, 0, 1, 1])
        left = parent
        right = np.array([])
        gain = calculate_information_gain(parent, left, right)
        self.assertAlmostEqual(gain, 0.0)
    
    def test_information_gain_useless_split(self):
        """Split that doesn't separate classes should give 0 gain"""
        parent = np.array([0, 1, 0, 1])
        left = np.array([0, 1])
        right = np.array([0, 1])
        gain = calculate_information_gain(parent, left, right)
        self.assertAlmostEqual(gain, 0.0)
    
    def test_information_gain_always_non_negative(self):
        """Information gain should never be negative"""
        parent = np.array([0, 0, 1, 1, 2, 2])
        left = np.array([0, 1, 2])
        right = np.array([0, 1, 2])
        gain = calculate_information_gain(parent, left, right)
        self.assertGreaterEqual(gain, 0.0)


class TestFindBestSplit(unittest.TestCase):
    """Unit tests for finding the best split"""
    
    def test_find_best_split_single_perfect_feature(self):
        """Should find the one perfect feature"""
        X = np.array([
            [0, 1],
            [0, 1],
            [1, 0],
            [1, 0]
        ])
        y = np.array([0, 0, 1, 1])
        
        best_feature, best_gain = find_best_split(X, y)
        self.assertEqual(best_feature, 0)  # First feature perfectly separates
        self.assertAlmostEqual(best_gain, 1.0)
    
    def test_find_best_split_no_valid_features(self):
        """Should return None when all features are constant"""
        X = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]
        ])
        y = np.array([0, 1, 0, 1])
        
        best_feature, best_gain = find_best_split(X, y)
        self.assertIsNone(best_feature)
    
    def test_find_best_split_rejects_unbalanced(self):
        """Should not select features that send all data to one side"""
        X = np.array([
            [0, 0],  # Feature 0 is all 0s (unbalanced)
            [0, 1],
            [0, 0],
            [0, 1]
        ])
        y = np.array([0, 1, 0, 1])
        
        best_feature, best_gain = find_best_split(X, y)
        
        # Should select feature 1, not feature 0
        if best_feature is not None:
            self.assertEqual(best_feature, 1)
    
    def test_find_best_split_compares_all_features(self):
        """Should evaluate all features and pick the best"""
        X = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 1],
            [1, 1, 1]
        ])
        y = np.array([0, 0, 1, 1])
        
        best_feature, best_gain = find_best_split(X, y)
        
        # Feature 0 or 2 should be selected (both separate well)
        self.assertIn(best_feature, [0, 2])
        self.assertGreater(best_gain, 0.5)


class TestMajorityClass(unittest.TestCase):
    """Unit tests for majority class function"""
    
    def test_majority_class_clear_winner(self):
        """Should return most frequent class"""
        from your_module import majority_class
        y = np.array([0, 0, 0, 1, 1, 2])
        self.assertEqual(majority_class(y), 0)
    
    def test_majority_class_tie(self):
        """Should handle ties consistently"""
        from your_module import majority_class
        y = np.array([0, 0, 1, 1])
        result = majority_class(y)
        self.assertIn(result, [0, 1])  # Either is acceptable
    
    def test_majority_class_single_sample(self):
        """Should work with single sample"""
        from your_module import majority_class
        y = np.array([5])
        self.assertEqual(majority_class(y), 5)


class TestNodeClass(unittest.TestCase):
    """Unit tests for Node class"""
    
    def test_node_is_leaf(self):
        """Leaf node should be identified correctly"""
        node = Node(value=1)  # Leaf with class 1
        self.assertTrue(node.is_leaf())
    
    def test_node_is_not_leaf(self):
        """Internal node should not be a leaf"""
        left_child = Node(value=0)
        right_child = Node(value=1)
        node = Node(feature=0, left=left_child, right=right_child)
        self.assertFalse(node.is_leaf())
    
    def test_node_stores_feature(self):
        """Node should store split feature correctly"""
        node = Node(feature=3)
        self.assertEqual(node.feature, 3)


# ============================================================================
# INTEGRATION TESTS - Test the entire system working together
# ============================================================================

class TestDecisionTreeIntegration(unittest.TestCase):
    """Integration tests for the complete decision tree"""
    
    def test_fit_and_predict_perfect_data(self):
        """End-to-end test with perfectly separable data"""
        X = np.array([[0], [0], [1], [1]])
        y = np.array([0, 0, 1, 1])
        
        tree = DecisionTree()
        tree.fit(X, y)
        predictions = tree.predict(X)
        
        np.testing.assert_array_equal(predictions, y)
    
    def test_fit_and_predict_multiclass(self):
        """End-to-end test with multiple classes"""
        X = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])
        y = np.array([0, 1, 2, 1])
        
        tree = DecisionTree()
        tree.fit(X, y)
        predictions = tree.predict(X)
        
        accuracy = np.mean(predictions == y)
        self.assertGreater(accuracy, 0.5)
    
    def test_xor_problem(self):
        """Integration test: XOR problem requires depth >= 2"""
        X = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])
        y = np.array([0, 1, 1, 0])
        
        tree = DecisionTree()
        tree.fit(X, y)
        predictions = tree.predict(X)
        
        # Decision tree can solve XOR
        np.testing.assert_array_equal(predictions, y)
    
    def test_no_none_leaves_in_tree(self):
        """Integration test: verify no None leaves exist"""
        df = pd.DataFrame({
            "F1": [0, 0, 0, 1, 1, 1, 1, 0, 0, 1],
            "F2": [0, 0, 1, 0, 1, 1, 0, 1, 0, 1],
            "F3": [0, 1, 0, 0, 1, 0, 1, 1, 1, 0],
            "Target": [0, 1, 2, 0, 2, 1, 0, 2, 1, 0]
        })
        
        X = df.drop('Target', axis=1).values
        y = df['Target'].values
        
        tree = DecisionTree()
        tree.fit(X, y)
        
        # Check no None leaves
        self.assertFalse(tree.has_none_leaves())
        
        # Check predictions work
        predictions = tree.predict(X)
        self.assertTrue(all(p is not None for p in predictions))


class TestDecisionTreeHyperparameters(unittest.TestCase):
    """Integration tests for hyperparameter behavior"""
    
    def test_max_depth_limits_tree(self):
        """max_depth should limit tree depth"""
        X = np.array([[i] for i in range(16)])
        y = np.array([i % 4 for i in range(16)])
        
        tree = DecisionTree(max_depth=2)
        tree.fit(X, y)
        
        self.assertLessEqual(tree.get_depth(), 2)
    
    def test_min_samples_split_prevents_splitting(self):
        """min_samples_split should prevent splitting small nodes"""
        X = np.array([[0], [0], [1], [1]])
        y = np.array([0, 0, 1, 1])
        
        tree = DecisionTree(min_samples_split=10)
        tree.fit(X, y)
        
        # Should create single leaf (not enough samples)
        self.assertEqual(tree.get_depth(), 0)
    
    def test_min_samples_leaf_enforced(self):
        """min_samples_leaf should be enforced in all leaves"""
        X = np.array([[i] for i in range(20)])
        y = np.array([i % 3 for i in range(20)])
        
        tree = DecisionTree(min_samples_leaf=5)
        tree.fit(X, y)
        
        leaf_sizes = tree.get_leaf_sizes()
        self.assertTrue(all(size >= 5 for size in leaf_sizes))


# ============================================================================
# EDGE CASE TESTS - Boundary conditions and unusual inputs
# ============================================================================

class TestDecisionTreeEdgeCases(unittest.TestCase):
    """Tests for edge cases and boundary conditions"""
    
    def test_single_training_sample(self):
        """Should handle single training sample"""
        X = np.array([[1]])
        y = np.array([0])
        
        tree = DecisionTree()
        tree.fit(X, y)
        prediction = tree.predict(X)
        
        self.assertEqual(prediction[0], 0)
    
    def test_all_same_features_different_labels(self):
        """Should handle identical features with different labels"""
        X = np.array([[0], [0], [0], [0]])
        y = np.array([0, 1, 0, 1])
        
        tree = DecisionTree()
        tree.fit(X, y)
        predictions = tree.predict(X)
        
        # Should predict majority class
        majority = np.bincount(y).argmax()
        self.assertTrue(all(p == majority for p in predictions))
    
    def test_all_same_labels_different_features(self):
        """Should handle all same labels (pure from start)"""
        X = np.array([[0], [1], [0], [1]])
        y = np.array([1, 1, 1, 1])
        
        tree = DecisionTree()
        tree.fit(X, y)
        predictions = tree.predict(X)
        
        np.testing.assert_array_equal(predictions, np.array([1, 1, 1, 1]))
    
    def test_binary_features_many_classes(self):
        """Should handle more classes than feature combinations allow"""
        X = np.array([[0], [1]])
        y = np.array([0, 1])
        
        tree = DecisionTree()
        tree.fit(X, y)
        
        # Should still work
        predictions = tree.predict(X)
        self.assertEqual(len(predictions), 2)


# ============================================================================
# PROPERTY-BASED TESTS (Optional - requires hypothesis library)
# ============================================================================

try:
    from hypothesis import given, strategies as st
    
    class TestDecisionTreeProperties(unittest.TestCase):
        """Property-based tests using hypothesis"""
        
        @given(st.integers(min_value=10, max_value=100))
        def test_tree_handles_any_size_input(self, n_samples):
            """Tree should handle any reasonable input size"""
            X = np.random.randint(0, 2, size=(n_samples, 3))
            y = np.random.randint(0, 3, size=n_samples)
            
            tree = DecisionTree()
            tree.fit(X, y)
            predictions = tree.predict(X)
            
            self.assertEqual(len(predictions), n_samples)
        
        @given(st.integers(min_value=1, max_value=10))
        def test_tree_handles_any_depth(self, max_depth):
            """Tree should respect any reasonable max_depth"""
            X = np.random.randint(0, 2, size=(50, 5))
            y = np.random.randint(0, 3, size=50)
            
            tree = DecisionTree(max_depth=max_depth)
            tree.fit(X, y)
            
            self.assertLessEqual(tree.get_depth(), max_depth)

except ImportError:
    print("hypothesis not installed, skipping property-based tests")


if __name__ == '__main__':
    # Run unit tests first, then integration tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add unit tests
    suite.addTests(loader.loadTestsFromTestCase(TestEntropyCalculation))
    suite.addTests(loader.loadTestsFromTestCase(TestInformationGain))
    suite.addTests(loader.loadTestsFromTestCase(TestFindBestSplit))
    suite.addTests(loader.loadTestsFromTestCase(TestMajorityClass))
    suite.addTests(loader.loadTestsFromTestCase(TestNodeClass))
    
    # Add integration tests
    suite.addTests(loader.loadTestsFromTestCase(TestDecisionTreeIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestDecisionTreeHyperparameters))
    
    # Add edge case tests
    suite.addTests(loader.loadTestsFromTestCase(TestDecisionTreeEdgeCases))
    
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)