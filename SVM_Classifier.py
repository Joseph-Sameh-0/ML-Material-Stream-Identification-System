import os
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from FeatureLead import load_features

# ---------------------------------------------------------------
# first step => load extracted features from FeatureLead.py
# ---------------------------------------------------------------
print("Loading extracted features...")
X, y, scaler, class_mapping, filenames = load_features()

print(f"Loaded feature matrix: {X.shape}")
print(f"Loaded labels: {y.shape}")

# ---------------------------------------------------------------
# second step => train / test Split
# ---------------------------------------------------------------
# scale features
# X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)



print("\nDataset Split:")
print("Training samples:", len(X_train))
print("Validation samples:", len(X_test))

# ---------------------------------------------------------------
# third step => SVM Model Definition
# Note: RBF kernel is chosen because:
#   - handles non-linear patterns
#   - works well with high-dimensional features (HOG + histograms)
#   - robust generalization
# ---------------------------------------------------------------
svm_model = SVC(
    C=10,
    kernel="rbf",
    gamma="scale",
    probability=True,
    class_weight="balanced"
)


# 3. Define SVM + hyperparameter grid (RBF kernel)
# param_grid = {
#     "C": [0.1, 1, 10],
#     "gamma": ["scale", 0.01, 0.001],
#     "kernel": ["rbf"]
# }
# base_svm = SVC(probability=True, class_weight="balanced")  # balanced helps class imbalance

# grid = GridSearchCV(
#     base_svm,
#     param_grid,
#     cv=3,
#     n_jobs=-1,
#     verbose=2
# )

# print("\nTraining SVM (RBF kernel)...")
# grid.fit(X_train, y_train)

# best_svm = grid.best_estimator_
# print("Best params:", grid.best_params_)

# # 4. Evaluate
# y_pred = best_svm.predict(X_test)
# print("Validation accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# configs = [
#     {"C": 1, "kernel": "rbf", "gamma": "scale"},
#     {"C": 10, "kernel": "rbf", "gamma": "scale"},
# ]

# for C in [50]:
#     svm = SVC(C=C, kernel="rbf", gamma= 0.005,
#               probability=True, class_weight="balanced")
#     svm.fit(X_train, y_train)
#     y_pred = svm.predict(X_test)
#     print("C =", C, "acc =", accuracy_score(y_test, y_pred))

# print(X.var(axis=0)[:20])
# print("NaNs:", np.isnan(X).sum())
# print("Infs:", np.isinf(X).sum())

# col_variances = X.var(axis=0)
# print("Zero-variance features:", (col_variances == 0).sum())

# print("Class counts:", np.bincount(y))

# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

# p = PCA(n_components=2)
# X2 = p.fit_transform(X)

# plt.scatter(X2[:,0], X2[:,1], c=y, s=5)
# plt.title("2D PCA projection of features")
# plt.show()

from sklearn.decomposition import PCA

# Keep 85% variance (sweet spot for this dataset)
# pca = PCA(n_components=0.85, random_state=42)

# X_train_pca = pca.fit_transform(X_train)
# X_test_pca = pca.transform(X_test)

# print("Original dim:", X_train.shape[1])
# print("Reduced dim:", X_train_pca.shape[1])
# print("pca components:", pca.n_components_)

print("\nTraining SVM (RBF kernel)...")
svm_model.fit(X_train, y_train)

# # ---------------------------------------------------------------
# # fourth step => evaluate accuracy => must be >= 85%
# # ---------------------------------------------------------------
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nValidation Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# ---------------------------------------------------------------
# fifth step => save the trained model + scaler
# ---------------------------------------------------------------
# os.makedirs("models", exist_ok=True)
# joblib.dump(svm_model, "models/svm_model.pkl")
# joblib.dump(scaler, "models/scaler.pkl")

# print("\nSaved:")
# print(" - models/svm_model.pkl")
# print(" - models/scaler.pkl")

# # ---------------------------------------------------------------
# # sixth step => implement rejection function 
# # ---------------------------------------------------------------
# def predict_with_rejection(model, features, threshold=0.55):
#     """
#     Returns:
#       - predicted class (0 - 5)
#       - OR 6 (Unknown) if confidence < threshold
#     """
#     probs = model.predict_proba(features.reshape(1, -1))[0]
#     best_class = np.argmax(probs)
#     best_prob = probs[best_class]

#     if best_prob < threshold:
#         return 6
#     return best_class

# print("\nModel training complete.")
