# rf_gp_minimize.py
import numpy as np
import pandas as pd
from functools import partial
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from skopt import gp_minimize
from skopt import space
import config

def optimize(params, param_names, x, y):
    # convert params to dictionary
    params = dict(zip(param_names, params))
    # initialize model with current parameters
    model = ensemble.RandomForestClassifier(**params)
    # initialize stratified k-fold
    kf = model_selection.StratifiedKFold(n_splits=5)
    # initialize accuracy list
    accuracies = []
    # loop over all folds
    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x[train_idx]
        ytrain = y[train_idx]
        xtest  = x[test_idx]
        ytest  = y[test_idx]
        # fit model for current fold
        model.fit(xtrain, ytrain)
        #create predictions
        preds = model.predict(xtest)
        # calculate and append accuracy
        fold_accuracy = metrics.accuracy_score(ytest, preds)
        accuracies.append(fold_accuracy)

    # return negative accuracy
    return -1 * np.mean(accuracies)


if __name__ == "__main__":
    # read the training data
    df = pd.read_csv(config.MUTUAL_INF_TRAIN, index_col=0)
    # here we have training features
    X = df.drop(["target", "kfold"], axis=1).values
    # and the targets
    y = df.target.values

    # define a parameter space
    param_space = [
        space.Integer(3, 15, name="max_depth"),
        space.Integer(100, 1500, name="n_estimators"),
        space.Categorical(["gini", "entropy", "log_loss"], name="criterion"),
        space.Real(0.01, 1, prior="uniform", name="max_features"),
        space.Integer(1, 100, name="min_samples_split"),
        space.Integer(1, 100, name="min_samples_leaf"),
        space.Categorical(["balanced", "balanced_subsample", None], name="class_weight")
        ]
    
    param_names = ["max_depth", "n_estimators", "criterion", "max_features", "min_samples_split", "min_samples_leaf", "class_weight"]

    np.int = int
    optimization_function = partial(optimize,
                                    param_names=param_names,
                                    x=X, y=y )

    result = gp_minimize(   optimization_function,
                            dimensions=param_space,
                            n_calls=30,
                            n_random_starts=10,
                            verbose=10
                            )

    # create best params dict and print it
    best_params = dict( zip(param_names, result.x))
    print(best_params)

    from skopt.plots import plot_convergence
    plot_convergence(result)