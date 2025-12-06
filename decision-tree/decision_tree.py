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
            raise TypeError("X is not a numpy array")
        
        if not isinstance(y, np.array):
            raise TypeError("y is not a numpy array")
        
        # Check for missing values in numpy arrays 
        