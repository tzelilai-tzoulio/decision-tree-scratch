import numpy as np
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder


class DecisionTree:
    def __init__(self, max_depth=None, impurity_metric="entropy"):
    
        if max_depth is not None and not isinstance(max_depth, int):
            raise TypeError("max_depth must be an integer or None")
        
        if not impurity_metric in ["gini", "entropy"]:
            raise TypeError("impurity_metric is not valid. Choose gini or entropy")
        
        self.max_depth = max_depth

    
    def _validate_input(self, X, y):
        
        # Convert data to numpy arrays 
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        
        if isinstance(y, pd.DataFrame): 
            y = y.to_numpy()

        # Check if the data is numpy arrays 
        if not isinstance(X, np.array): 
            raise TypeError("X is not a pandas DataFrame or numpy array")
        
        if not isinstance(y, np.array):
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

        left_indices = X[indices,feature] == 0
        right_indices = X[indices,feature] == 1

        H_parent = self._cal_cat_entropy(y, indices)
        H_left = self._cal_cat_entropy(y, left_indices)
        H_right = self._cal_cat_entropy(y, right_indices)

        w_left = left_indices.sum()/len(indices)
        w_right = right_indices.sum()/len(indices)

        return H_parent - (w_left*H_left + w_right*H_right)
    