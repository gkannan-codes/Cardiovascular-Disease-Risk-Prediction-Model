# %%
# Import necessary libraries in order to perform data manipulation, visualization, and machine learning tasks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV  # Regularized Logistic Regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Function for standardization (z-score)
def standardize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Function for normalization (min-max scaling)
def normalize(X):
    return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

# Hypothesis function for logistic regression (sigmoid)
def Hypothesis(X, w):
    z = X.dot(w)
    H = 1 / (1 + np.exp(-z))
    return np.clip(H, 1e-15, 1 - 1e-15)

# Cost function for logistic regression with regularization
def cost(X, w, y, lambda_reg=0.1):
    H = Hypothesis(X, w)  # Getting the hypothesis
    regularization_term = (lambda_reg / (2 * X.shape[0])) * np.sum(w[1:] ** 2)  # L2 regularization (penalty)
    return -np.mean((y * np.log(H)) + ((1 - y) * np.log(1 - H))) + regularization_term

# Function for weight update in logistic regression with regularization
def new_w(w, alpha, X, y, lambda_reg=0.1):
    H = Hypothesis(X, w)  # Getting the hypothesis
    regularization_term = lambda_reg * w  # Regularization for weights (excluding bias term)
    regularization_term[0] = 0  # No regularization for bias term
    return w - ((alpha / X.shape[0]) * (X.T.dot((H - y)))) + regularization_term  # Updating the weights

# Feature selection using RFE (Recursive Feature Elimination)
def feature_selection(X, y, n_features=20):
    selector = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=n_features, step=1)
    X_selected = selector.fit_transform(X, y)
    return X_selected

# PCA Logistic Regression Classifier
def pca_logistic_classifier(X_train, y_train, X_test, n_components=5):
    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Apply PCA
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    # Get PCA component loadings
    loadings = pca.components_  # Shape: (n_components, n_features)

    # Create a DataFrame for readability
    feature_names = data.drop('target', axis=1).columns  # Assuming you pass in original feature names
    loading_df = pd.DataFrame(loadings.T, columns=[f'PC{i+1}' for i in range(pca.n_components_)], index=feature_names)

    # Compute the absolute value of the loadings for contribution strength
    loading_strength = loading_df.abs()

    # Optionally rank the features by their contribution to each principal component
    for pc in loading_strength.columns:
        print(f"\nTop contributing features to {pc}:")
        print(loading_strength[pc].sort_values(ascending=False).head(10))

    X_test_pca = pca.transform(X_test)

    # Train Logistic Regression model
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train_pca, y_train)

    # Predict probabilities
    preds = model.predict_proba(X_test_pca)[:, 1]
    
    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_train, preds, pos_label=1)  # Specify pos_label explicitly
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc

# Cross-validation function with ROC curve generation
def cross_validate(X, y, method, use_feature_selection=False, lambda_reg=0.1, n_features=20, n_components=5):
    # Apply standardization, normalization or raw data
    if method == 'standardize':
        X_processed = standardize(X)
    elif method == 'normalize':
        X_processed = normalize(X)
    elif method == 'raw':
        X_processed = X  # No preprocessing for raw data
    elif method == 'regularized':
        X_processed = X  # No preprocessing for raw data (applies regularization directly)
    elif method == 'pca':
        return pca_logistic_classifier(X, y, X, n_components)  # PCA method does not need cross-validation

    # Feature selection: Recursive Feature Elimination (RFE)
    if use_feature_selection and method != 'regularized' and method != 'pca':
        X_processed = feature_selection(X_processed, y, n_features)
    
    # Add the bias term (1s) to the processed data
    X_processed = np.concatenate([np.ones((X_processed.shape[0], 1)), X_processed], axis=1)

    # Shuffle the data
    m = X_processed.shape[0]
    indices = np.arange(m)
    np.random.shuffle(indices)
    X_processed = X_processed[indices]
    y = y[indices]

    # Split indices into 10 folds
    fold_size = m // 10
    folds = [(i * fold_size, (i + 1) * fold_size) for i in range(10)]

    errors = []
    all_preds = []
    all_true = []

    for fold_idx, (start, end) in enumerate(folds):
        print(f"Fold {fold_idx + 1}")

        # Validation set
        X_val = X_processed[start:end]
        y_val = y[start:end]

        # Training set
        X_train = np.concatenate((X_processed[:start], X_processed[end:]), axis=0)
        y_train = np.concatenate((y[:start], y[end:]), axis=0)

        # Initialize weights
        w = np.zeros((X_processed.shape[1], 1))

        # Train the model
        alpha = 0.1  # Learning rate
        iterations = 5000  # You can increase repetitions to converge better
        for i in range(iterations):
            if method == 'regularized':
                # Regularized logistic regression
                clf = LogisticRegression(C=1/lambda_reg, solver='lbfgs', max_iter=1000)
                clf.fit(X_train, y_train.flatten())
                preds = clf.predict_proba(X_val)[:, 1]
                break
            else:
                w = new_w(w, alpha, X_train, y_train, 0)

        # Evaluate model on validation set
        if method != 'regularized' and method != 'pca':
            preds = Hypothesis(X_val, w)
        all_preds.append(preds.flatten())  # Flatten to 1D
        all_true.append(y_val.flatten())  # Flatten to 1D

    # Convert list to numpy array
    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(all_true, all_preds, pos_label=1)  # Specify pos_label explicitly
    roc_auc = auc(fpr, tpr)

    print(f"Method: {method}, W: {w}")
    return fpr, tpr, roc_auc

# %% Load and preprocess the data
data = pd.read_csv('heart.csv')

# Convert diagnosis to binary labels: Malignant -> 1, Benign -> 0
# y = data['diagnosis'].replace({'M': 1, 'B': 0}).to_numpy().reshape(-1, 1)
y = data['target'].to_numpy().reshape(-1, 1)
X = data.drop('target', axis=1).dropna(axis=1).to_numpy()  # Dropping NaN values and converting to numpy

# %% Run cross-validation and plot ROC for all methods
fpr_raw, tpr_raw, roc_auc_raw = cross_validate(X, y, 'raw', use_feature_selection=False, lambda_reg=0.1)
fpr_normalized, tpr_normalized, roc_auc_normalized = cross_validate(X, y, 'normalize', use_feature_selection=False, lambda_reg=0.1)
fpr_standardized, tpr_standardized, roc_auc_standardized = cross_validate(X, y, 'standardize', use_feature_selection=False, lambda_reg=0.1)
fpr_feature_selected, tpr_feature_selected, roc_auc_feature_selected = cross_validate(X, y, 'raw', use_feature_selection=True, lambda_reg=0.1)
fpr_regularized, tpr_regularized, roc_auc_regularized = cross_validate(X, y, 'regularized', use_feature_selection=False, lambda_reg=0.1)
fpr_pca, tpr_pca, roc_auc_pca = cross_validate(X, y, 'pca', use_feature_selection=False, lambda_reg=0.1)

# Plot ROC curves for all methods
plt.figure(figsize=(10, 6))
plt.plot(fpr_raw, tpr_raw, color='blue', lw=2, label=f'Raw Data (AUC = {roc_auc_raw:.2f})')
plt.plot(fpr_normalized, tpr_normalized, color='red', lw=2, label=f'Normalized (AUC = {roc_auc_normalized:.2f})')
plt.plot(fpr_standardized, tpr_standardized, color='green', lw=2, label=f'Standardized (AUC = {roc_auc_standardized:.2f})')
plt.plot(fpr_feature_selected, tpr_feature_selected, color='purple', lw=2, label=f'Feature Selected (AUC = {roc_auc_feature_selected:.2f})')
plt.plot(fpr_regularized, tpr_regularized, color='orange', lw=2, label=f'Regularized (AUC = {roc_auc_regularized:.2f})')
plt.plot(fpr_pca, tpr_pca, color='brown', lw=2, label=f'PCA (AUC = {roc_auc_pca:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('ROC Curve Comparison (Logistic Regression Methods)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Print AUC values for all methods
print(f"Raw Data AUC: {roc_auc_raw*100:.2f}%")
print(f"Normalized AUC: {roc_auc_normalized*100:.2f}%")
print(f"Standardized AUC: {roc_auc_standardized*100:.2f}%")
print(f"Feature Selected AUC: {roc_auc_feature_selected*100:.2f}%")
print(f"Regularized AUC: {roc_auc_regularized*100:.2f}%")
print(f"PCA AUC: {roc_auc_pca*100:.2f}%")

# %%
