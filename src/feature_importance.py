import pandas as pd
import argparse, os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import config


def plot_feature_importance(model, feature_names):
    # Extract feature importances from the trained model
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    
    # Sort features based on importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    most_important_features = feature_importance_df.loc[:20, "Feature"]
    least_important_features = feature_importance_df.tail(20)["Feature"]

    least_important_threshold_features = feature_importance_df[feature_importance_df['Importance'] <= 0.0025]["Feature"].tolist()
    print(f"Features with zero importance: {least_important_threshold_features}\n. Nombre: {len(least_important_threshold_features)}")


    # Plot feature importance
    plt.figure(figsize=(300, 100))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

    return most_important_features, least_important_features


def feature_importance_analysis(model_name):
    # Load the trained model
    model_path = os.path.join(config.MODEL_OUTPUT, f"5fold_{model_name}_0.pkl")  # Loading model from fold 0
    clf = joblib.load(model_path)
    
    # Load the dataset
    df = pd.read_csv(config.TRAINING_FILE, index_col=0)
    
    # Extract feature names
    feature_names = df.drop(["target", "kfold"], axis=1).columns
    
    # Perform feature importance analysis
    most_important_features, least_important_features = plot_feature_importance(clf, feature_names)

    print(f"Most important features: {most_important_features}.\n Least important features: {least_important_features}\n")


if __name__ == "__main__":
    # Initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # Add the argument for the model name
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model for which feature importance analysis will be performed"
    )
    # Read the argument from the command line
    args = parser.parse_args()

    # Perform feature importance analysis for the specified model
    feature_importance_analysis(args.model)
