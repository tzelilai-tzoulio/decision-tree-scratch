import numpy as np
import pandas as pd 
from node import Node
from multiprocessing import Pool


class DecisionTree:
    def __init__(self, max_depth=None, impurity_metric="entropy"):
    
        if max_depth is not None and not isinstance(max_depth, int):
            raise TypeError("max_depth must be an integer or None")
        
        if not impurity_metric in ["gini", "entropy"]:
            raise TypeError("impurity_metric is not valid. Choose gini or entropy")
        
        self.impurity_metric = impurity_metric
        self.max_depth = max_depth
        self.current_depth = 0 
        self.root_node = None 
    
    def _validate_input(self, X, y):
        
        # Convert data to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        
        if isinstance(y, pd.DataFrame): 
            y = y.to_numpy()

        # Check if the data is numpy arrays 
        if not isinstance(X, np.ndarray): 
            raise TypeError("X is not a pandas DataFrame or numpy array")
        
        if not isinstance(y, np.ndarray):
            raise TypeError("y is not a pandas DataFrame or numpy array")
        
        # Check if numpy array is 2D 
        if not X.ndim == 2:
            raise TypeError("X is not a 2D array") 
        
        # Check for missing values 
        if np.isnan(X).any(): 
            raise ValueError("X contains Nan Values")
        
        # Check if All Values are Numeric  
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError("X contains non-numeric data. Convert or encode it first.")
        
    def _identify_columns(self, X):
        
        m,n = X.shape
        cat = []
        num = []

        # Check if column is categorical or numerical 
        # based on unique values 
        for i in range(n):
            # > 2 => Numerical because we have only binary categorical values 
            if len(np.unique(X[:, i])) > 2: 
                num.append(i)
            
            else: 
                cat.append(i)

        return num, cat
    
    def _cal_cat_gini(self, y, indices): 

        subset = y[indices]

        _, counts = np.unique(subset, return_counts=True)
        p = counts / counts.sum() 

        return 1 - np.sum(p ** 2)
    

    def _cal_cat_entropy(self, y, indices): 

        subset = y[indices]

        # Measure populatrity of each unique target value 
        _, counts = np.unique(subset, return_counts=True)
        p = counts / counts.sum()

        return -(p * np.log2(p)).sum()
        
    def _cal_info_gain(self, X, y, feature, indices):
        # Assuming 0 values -> left 
        #          1 values -> right

        left_indices = X[:,feature] == 0
        right_indices = X[:,feature] == 1

        left_indices = left_indices & indices 
        right_indices = right_indices & indices 

        if self.impurity_metric == "entropy":
            H_parent = self._cal_cat_entropy(y, indices)
            H_left = self._cal_cat_entropy(y, left_indices)
            H_right = self._cal_cat_entropy(y, right_indices)
        
        elif self.impurity_metric == "gini":
            H_parent = self._cal_cat_gini(y, indices)
            H_left = self._cal_cat_gini(y, left_indices)
            H_right = self._cal_cat_gini(y, right_indices)

        else: 
            raise ValueError("No impurity metric provided")
        
        w_left = left_indices.sum()/len(indices)
        w_right = right_indices.sum()/len(indices)

        return H_parent - (w_left*H_left + w_right*H_right)
    
    def _display_aux(self, node):
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        if node is None:
            return [], 0, 0, 0

        # Node label
        if node.value is not None:
            s = f"[{node.value}]"
        else:
            s = f"[{node.feature}]"

        # No children (leaf)
        if node.left_node is None and node.right_node is None:
            width = len(s)
            height = 1
            middle = width // 2
            return [s], width, height, middle

        # Only left child
        if node.right_node is None:
            left, n, p, x = self._display_aux(node.left_node)
            first_line = (x + 1) * " " + (n - x - 1) * "_" + s
            second_line = x * " " + "/" + (n - x - 1 + len(s)) * " "
            shifted_lines = [line + len(s) * " " for line in left]
            return [first_line, second_line] + shifted_lines, n + len(s), p + 2, n + len(s) // 2

        # Only right child
        if node.left_node is None:
            right, n, p, x = self._display_aux(node.right_node)
            first_line = s + x * "_" + (n - x) * " "
            second_line = (len(s) + x) * " " + "\\" + (n - x - 1) * " "
            shifted_lines = [len(s) * " " + line for line in right]
            return [first_line, second_line] + shifted_lines, n + len(s), p + 2, len(s) // 2

        # Two children
        left, n, p, x = self._display_aux(node.left_node)
        right, m, q, y = self._display_aux(node.right_node)

        first_line = (
            (x + 1) * " "
            + (n - x - 1) * "_"
            + s
            + y * "_"
            + (m - y) * " "
        )
        second_line = (
            x * " "
            + "/"
            + (n - x - 1 + len(s) + y) * " "
            + "\\"
            + (m - y - 1) * " "
        )

        if p < q:
            left += [" " * n] * (q - p)
        elif q < p:
            right += [" " * m] * (p - q)

        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + len(s) * " " + b for a, b in zipped_lines]

        return lines, n + m + len(s), max(p, q) + 2, n + len(s) // 2

    def _find_best_split(self, X, y, indices, available_features):
        max_info_gain = 0
        best_feature = None

        for feature in available_features:
            # Compute child indices for THIS node only
            left_indices = (X[:, feature] == 0) & indices
            right_indices = (X[:, feature] == 1) & indices

            # Skip invalid splits (empty child)
            if not np.any(left_indices) or not np.any(right_indices):
                continue

            # Compute information gain only for valid splits
            info_gain = self._cal_info_gain(X, y, feature, indices)

            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_feature = feature

        return max_info_gain, best_feature
    
    def _most_frequent_element(self, x): 
        # Get unique values and their counts
            unique_vals, counts = np.unique(x, return_counts=True)
 
            # Find the index of the maximum count
            max_count_index = np.argmax(counts)

            # Get the most frequent element
            most_frequent_element = unique_vals[max_count_index]

            return most_frequent_element
    
    def _build_tree(self, X, y, indices, available_features, node, current_depth = 0):
        
        # Incrementing current depth  
        current_depth += 1 

        values = np.unique(y[indices])

        # Find best feature for current Node 
        max_info_gain, best_feature = self._find_best_split(X, y, indices, available_features)

        # 1. Pure Node 
        if len(values) == 1:
            node.value = values[0]
            return
        
        # 2. No information gain from any feature
        if len(available_features) == 0 or max_info_gain <= 0 or current_depth == self.max_depth: 
            node.value = self._most_frequent_element(y[indices])
            return
        
        # Creating available features for children nodes
        child_features = [
            f for f in available_features if f != best_feature
        ]

        # Commit to the split
        node.feature = best_feature

        # Compute child indices (guaranteed non-empty)
        left_indices = (X[:, best_feature] == 0) & indices
        right_indices = (X[:, best_feature] == 1) & indices

        # Create child nodes
        node.left_node = Node()
        node.right_node = Node()

        # Recurse with reduced feature set
        self._build_tree(X, y, indices=left_indices, 
                        available_features=child_features,
                        node=node.left_node, 
                        current_depth=current_depth
                        )
        self._build_tree(X, y, indices=right_indices,
                        available_features=child_features, 
                        node=node.right_node, 
                        current_depth=current_depth
                        )

    def _predict_single_example(self, x, node): 
        
        if node.left_node != None and node.right_node != None: 
            if x[node.feature] == 0: 
                return self._predict_single_example(x, node.left_node)
            else:
                return self._predict_single_example(x, node.right_node)

        else: 
            return node.value 


    def train(self, X, y): 
        # Validate training data 
        self._validate_input(X,y)

        m,n = X.shape
        starting_indices = [True for i in range(m)]
        available_features = [i for i in range(n)]
        root_node = Node() 

        # start building the Tree
        print("Start building the Tree")
        self._build_tree(X, y, indices=starting_indices, available_features=available_features, node=root_node)
        self.root_node = root_node


    def predict(self, X): 
        # Validate inputs 
        y_pred = np.array([None for _ in X])

        for x in X: 
            y_pred = self._predict_single_example(x, self.root_node)

        return y_pred 

    def print_tree(self):
        lines, *_ = self._display_aux(self.root_node)
        for line in lines:
            print(line)     