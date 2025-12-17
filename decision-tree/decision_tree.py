import numpy as np
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder
from node import Node

class DecisionTree:
    def __init__(self, max_depth=None, impurity_metric="entropy"):
    
        if max_depth is not None and not isinstance(max_depth, int):
            raise TypeError("max_depth must be an integer or None")
        
        if not impurity_metric in ["gini", "entropy"]:
            raise TypeError("impurity_metric is not valid. Choose gini or entropy")
        
        self.max_depth = max_depth
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

        H_parent = self._cal_cat_entropy(y, indices)
        H_left = self._cal_cat_entropy(y, left_indices)
        H_right = self._cal_cat_entropy(y, right_indices)

        w_left = left_indices.sum()/len(indices)
        w_right = right_indices.sum()/len(indices)

        return H_parent - (w_left*H_left + w_right*H_right)
    

    def train(self, X, y): 
        # Validate training data 
        self._validate_input(X,y)

        m,n = X.shape
        starting_indices = [True for i in range(m)]
        available_features = [i for i in range(n)]
        root_node = Node() 

        # start building the Tree
        print("Start building the Tree")
        self.build_tree(X, y, indices=starting_indices, available_features=available_features, node=root_node)
        self.root_node = root_node


    def print_tree(self):
        lines, *_ = self._display_aux(self.root_node)
        for line in lines:
            print(line)

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

        # Find best feature for current Node 
        for feature in available_features: 
            info_gain = self._cal_info_gain(X, y, feature, indices = indices)

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
    

    def build_tree(self, X, y, indices, available_features, node):
        
        values = np.unique(y[indices])
        if len(values) == 0: 
            return 
        
        # 2. Pure Node 
        if len(values) == 1:
            node.value = values[0]
            return

        # 2. Available Features 
            
        # Find best feature for current Node 
        max_info_gain, best_feature = self._find_best_split(X, y, indices, available_features)

        # No information gain from any feature
        if len(available_features) == 0 or max_info_gain == 0: 
            node.value = self.most_frequent_element(y[indices])
            return

        # Removing used feature from the list 
        available_features.remove(best_feature)

        # Updating Node 
        node.feature = best_feature


        left_indices = X[:, best_feature] == 0 
        right_indices = X[:, best_feature] == 1

        left_indices = left_indices & indices
        right_indices = right_indices & indices

        left_values = np.unique(y[left_indices])
        right_values = np.unique(y[right_indices])

        # 1. Avoiding empty subsets creating new None nodes
        # Left/Right Subset contains no values  
        if len(left_values) == 0 or len(right_values) == 0:
            node.value = self._most_frequent_element(y[indices])
            return  
        else: 
            node.left_node = Node()
            node.right_node = Node()
            self.build_tree(X, y, indices=left_indices, available_features=available_features.copy(), node=node.left_node)
            self.build_tree(X, y, indices=right_indices, available_features=available_features.copy(), node=node.right_node)
        