import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import config

# Initialize StandardScaler and PCA instances
scaler = StandardScaler()
pca = PCA(n_components=30)  

def perform_pca(df, dataset):
    global scaler, pca  # Use the global instances

    if dataset == 'train':  # Train set
        X = df.drop(columns=["target"])
        y = df["target"]
        # Fit StandardScaler and PCA on the training data
        X_scaled = scaler.fit_transform(X)
        X_pca = pca.fit_transform(X_scaled)
    else:  # Test set
        X = df
        # Only transform using the fitted StandardScaler and PCA
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)

    # Save PCA-transformed features along with labels (if applicable)
    pca_df = pd.DataFrame(X_pca, columns=[f"pca_{i+1}" for i in range(X_pca.shape[1])])
    if dataset == 'train':
        pca_df["target"] = y.values
    return pca_df

def main():
    # Read the dataset with kfold information
    df = pd.read_csv("../input/data.csv")

    # Perform PCA on train data
    pca_df_train = perform_pca(df, dataset='train')

    # Save the final PCA-transformed features for train set
    final_output_file_train = os.path.join(config.PCA_OUTPUT_DIR, "orig_train_pca.csv")
    pca_df_train.to_csv(final_output_file_train, index=False)
    print("PCA transformation completed on train.")

    # Apply PCA transformations to the test dataset
    test_df = pd.read_csv(config.TEST_FILE)

    # Perform PCA on test data
    pca_df_test = perform_pca(test_df, dataset='test')

    # Save the final PCA-transformed features for test set
    test_output_file = os.path.join(config.PCA_OUTPUT_DIR, "orig_test_pca.csv")
    pca_df_test.to_csv(test_output_file, index=False)

    print("PCA transformation applied to the test dataset.")

if __name__ == "__main__":
    main()
