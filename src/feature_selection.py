import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import config
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile


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
            # raise exception if we do not have a valid scoring method
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
        
    # same fit function
    def fit(self, X, y):
        return self.selection.fit(X, y)
    
    # same transform function
    def transform(self, X):
        return self.selection.transform(X)
    
    # same fit_transform function
    def fit_transform(self, X, y):
        return self.selection.fit_transform(X, y)

def VarianceThreshold(data):
    print("Shape before Variance Threshold", data.shape)

    var_thresh = VarianceThreshold(threshold=0.25)
    var_thresh.fit(data)
    dropcols = [col for col in data.columns if col not in data.columns[var_thresh.get_support()]]

    for features in dropcols:
        print(features)

    transformed_data = data.drop(dropcols, axis=1)
    # transformed data will have all columns with variance less than 0.1 removed
    transformed_data = VarianceThreshold()
    transformed_data.to_csv("../input/variance_thresh_train.csv", index=False)
    print("Shape after Variance Threshold", transformed_data.shape)


def CorrelationCoefficient(data):
    print("Shape before Correlation Coefficient", data.shape)
    # Creating correlation matrix
    cor_matrix = data.corr().abs()
    print(); print(cor_matrix)

    # Selecting upper triangle of correlation matrix
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    #print(); print(upper_tri)

    # Finding index of feature columns with correlation greater than 0.97
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    print(); print(to_drop)

    # Droping Marked Features
    df = data.drop(data[to_drop], axis=1)
    print(); print(df.head())
    df.to_csv("../input/correlation_drop_train.csv", index=False)
    print("Shape after Correlation Coefficient", df.shape)

    #return df

#CorrelationCoefficient(data)


if __name__ == "__main__":
    data  = pd.read_csv(config.ORIG_TRAIN_FILE, index_col=0)
    test  = pd.read_csv(config.TEST_FILE, index_col=0)
    X = data.drop("target", axis=1)
    y = data.target

    ufs = UnivariateFeatureSelction(n_features=7, 
                                    problem_type="classification",
                                    scoring="mutual_info_classif")

    ufs.fit(X, y)
    X_transformed = ufs.transform(X)
    test_transformed = ufs.transform(test)

    print(X_transformed)
    print(test_transformed)
    

    
