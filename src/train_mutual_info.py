import joblib
import os
import argparse
import pandas as pd
from sklearn import metrics

import config
import model_dispatcher

def run(fold, model):
    # read the training data with folds
    df = pd.read_csv(config.MUTUAL_INF_TRAIN)
      
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train.drop(["target", "kfold"], axis=1).values
    y_train = df_train.target.values
    
    x_valid = df_valid.drop(["target", "kfold"], axis=1).values
    y_valid = df_valid.target.values
    
    clf = model_dispatcher.models[model]

    # fit the model on training data
    clf.fit(x_train, y_train)

    # create predictions for validation samples
    preds = clf.predict(x_valid)

    # calculate & print accuracy
    accuracy = metrics.accuracy_score(y_valid, preds)
    f1 = metrics.f1_score(y_valid, preds, average='weighted')
    print(f"Fold={fold}, Model={model}, Accuracy={accuracy}, F1-Score={f1}\n")
    print(f"Confusion Matrix:\n {metrics.confusion_matrix(y_valid, preds)}")

    # save the model
    joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, f"mutual_info_10fold_{model}_{fold}.pkl"))


if __name__ == "__main__":
    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # add the different arguments you need and their types
    parser.add_argument(
        "--fold",
        type=int
    )
    parser.add_argument(
        "--model",
        type=str
    )
    # read the arguments from the command line
    args = parser.parse_args()

    # run the fold specified by command line arguments
    run(fold=args.fold, model=args.model)