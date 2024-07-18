import joblib
import os
import numpy as np
import pandas as pd
import config
#model = "xgb"

def majority_vote(predictions):
    num_folds = len(predictions)
    num_samples = len(predictions[0])
    final_predictions = []
    for i in range(num_samples):
        sample_votes = [pred[i] for pred in predictions]
        unique_classes, class_counts = np.unique(sample_votes, return_counts=True)
        # Check if there is a clear majority
        max_count = np.max(class_counts)
        if np.sum(class_counts == max_count) == 1:
            # If a clear majority exists, select the majority class
            majority_class = unique_classes[np.argmax(class_counts)]
        else:
            # If there is a tie or no clear majority, choose randomly among the tied classes
            tied_classes = unique_classes[class_counts == max_count]
            majority_class = np.random.choice(tied_classes)
        final_predictions.append(majority_class)
    return np.array(final_predictions)


def predict(model):
    test  = pd.read_csv(config.MUTUAL_INF_TEST)
    sample = pd.read_csv(config.SAMPLE_FILE, index_col=0)
    predictions = []
    for fold in range(10):
        clf = joblib.load(os.path.join(config.MODEL_OUTPUT, f"baseline_{model}_10{fold}.pkl"))
        fold_predictions = clf.predict(test.values)
        predictions.append(fold_predictions)
    # Perform majority vote as final prediction
    final_predictions = majority_vote(predictions)
    sample.target = final_predictions
    print(sample.head())
    return sample


def predict_ensemble(models):
    test = pd.read_csv(config.MUTUAL_INF_TEST, index_col=0)
    sample = pd.read_csv(config.SAMPLE_FILE, index_col=0)
    predictions = []
    for model in models:
        model_predictions = []
        for fold in range(10):
            clf = joblib.load(os.path.join(config.MODEL_OUTPUT, f"mutual_info_10fold_{model}_{fold}.pkl"))
            fold_predictions = clf.predict(test.values)
            model_predictions.append(fold_predictions)
        predictions.append(model_predictions)
    # Perform majority vote for each model's predictions
    final_predictions = []
    for model_preds in predictions:
        final_predictions.append(majority_vote(model_preds))
    # Perform majority vote across all models
    final_predictions = majority_vote(final_predictions)

    sample.target = final_predictions
    print(sample.head())

    return sample


if __name__ == "__main__":
    models = ["xgb", "hist", "extra", "cat"]
    submission = predict_ensemble(models)
    submission.to_csv(f"../output/last_mutual_info_ensemble_10folds.csv")
