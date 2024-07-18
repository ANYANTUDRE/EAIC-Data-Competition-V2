from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif, chi2, mutual_info_classif, f_regression, mutual_info_regression
from mlxtend.feature_selection import SequentialFeatureSelector
import numpy as np
import pandas as pd

class UnivariateFeatureSelction:
    def __init__(self, n_features, problem_type, scoring):
        if problem_type == "classification":
            valid_scoring = {
                "f_classif": f_classif,
                "chi2": chi2,
                "mutual_info_classif": mutual_info_classif
            }
        else:
            valid_scoring = {
                "f_regression": f_regression,
                "mutual_info_regression": mutual_info_regression
            }
        
        if scoring not in valid_scoring:
            raise Exception("Invalid scoring function")
        
        if isinstance(n_features, int):
            self.selection = SelectKBest(
                valid_scoring[scoring],
                k=n_features
            )
        elif isinstance(n_features, float):
            self.selection = SelectPercentile(
                valid_scoring[scoring],
                percentile=int(n_features * 100)
            )
        else:
            raise Exception("Invalid type of feature")
        
    def fit(self, X, y):
        self.selection.fit(X, y)
        return self
    
    def transform(self, X):
        selected_features = self.selection.get_support()
        selected_column_names = X.columns[selected_features]
        return X[selected_column_names]
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
    
    def sequential_forward_selection(self, estimator, X, y, k_features=10, forward=True):
        sfs = SequentialFeatureSelector(estimator,
                                        k_features=k_features,
                                        forward=forward,
                                        scoring='accuracy',
                                        cv=5)
        sfs.fit(X, y)
        selected_column_names = list(X.columns[list(sfs.k_feature_idx_)])
        return X[selected_column_names]
    
    def sequential_backward_selection(self, estimator, X, y, k_features=10, forward=False):
        sbs = SequentialFeatureSelector(estimator,
                                        k_features=k_features,
                                        forward=forward,
                                        scoring='accuracy',
                                        cv=5)
        sbs.fit(X, y)
        selected_column_names = list(X.columns[list(sbs.k_feature_idx_)])
        return X[selected_column_names]
