import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
from FeatureLead import load_features
from sklearn.metrics import classification_report
import os

# --------------------------------------------------
# 1. Load CNN-extracted features (2048-D from ResNet50)
# --------------------------------------------------
print("Loading CNN features from FeatureLead pipeline...")
X, y, final_scaler, class_mapping, _ = load_features()

# --------------------------------------------------
# 2. Split data: 80% train, 20% validation
# --------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------------------------------
# 3. Train k-NN and find best k
# --------------------------------------------------
print("Searching for the best k...")
best_k, best_acc, best_model = 1, 0, None

for k in range(3, 30, 2):
    model = KNeighborsClassifier(n_neighbors=k, weights='distance')
    model.fit(X_train, y_train)
    acc = model.score(X_val, y_val)
    if acc > best_acc:
        best_acc, best_k, best_model = acc, k, model

print(f"Best k = {best_k} → Validation Accuracy: {best_acc:.4f}")

# --------------------------------------------------
# 4. Compute rejection threshold (90th percentile)
# --------------------------------------------------
distances, _ = best_model.kneighbors(X_val)
threshold = np.percentile(distances[:, 0], 90)
print(f"\n📌 Rejection Threshold: {threshold:.4f}")

# --------------------------------------------------
# 5. Save only what's needed (NO HOG-PCA anymore)
# --------------------------------------------------
print("\n💾 Saving model components...")
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/knn_model.pkl")
joblib.dump(final_scaler, "models/scaler.pkl")
joblib.dump(threshold, "models/threshold.pkl")
joblib.dump(class_mapping, "models/class_mapping.pkl")
print("✅ Saved: knn_model.pkl, scaler.pkl, threshold.pkl, class_mapping.pkl")

# --------------------------------------------------
# 6. Detailed Evaluation (FOR REPORT)
# --------------------------------------------------
y_pred = best_model.predict(X_val)

print("\nClassification Report (k-NN):")
print(classification_report(y_val, y_pred))