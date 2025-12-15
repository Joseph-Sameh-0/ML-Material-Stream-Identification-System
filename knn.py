import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from FeatureLead import load_features
import os
from sklearn.decomposition import PCA
from FeatureLead import extract_features_from_image

# --------------------------------------------------
# 1. Load features and labels
#    => Load pre-extracted features from FeatureLead pipeline
# --------------------------------------------------
print("Loading features and labels from team's feature pipeline...")
X, y, final_scaler, class_mapping, _ = load_features()

# --------------------------------------------------
# 2. Split data: 80% training, 20% validation
#    => Stratified split to maintain class balance
# --------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------------------------------
# 3. Train k-NN and find the best k
#    => k-NN with 'distance' weights gives better importance to closer neighbors
#    => Search odd k values from 3 to 29 to avoid ties
# --------------------------------------------------
print("Searching for the best k value...")
best_k = 1
best_acc = 0
best_model = None

for k in range(3, 30, 2):
    model = KNeighborsClassifier(n_neighbors=k, weights='distance')
    model.fit(X_train, y_train)
    acc = model.score(X_val, y_val)
    if acc > best_acc:
        best_acc = acc
        best_k = k
        best_model = model

print(f"Best k = {best_k} → Validation Accuracy (on classes 0–5): = {best_acc:.4f}")

if best_acc >= 0.85:
    print("🎉 Target accuracy (≥ 0.85) achieved!")
else:
    print("⚠️ Accuracy is below 0.85 — consider improving feature quality.")

# --------------------------------------------------
# 4. Prepare Rejection Mechanism (for real-time deployment)
#    => Reject samples that are too far from known training data
#    => Use 90th percentile of nearest-neighbor distances as threshold
# --------------------------------------------------
distances, _ = best_model.kneighbors(X_val)
min_distances = distances[:, 0]
threshold = np.percentile(min_distances, 90)
print(f"\n📌 Rejection Threshold (for real-time use): {threshold:.4f}")

# Classification function with rejection (Unknown class = 6)
def classify_with_rejection(model, X, thresh):
    dists, _ = model.kneighbors(X)
    preds = model.predict(X)
    preds[dists[:, 0] > thresh] = 6  # Assign Unknown (class 6)
    return preds

# --------------------------------------------------
# 5. Save model and preprocessing objects
#    => Save all required components for standalone inference later
#    => Includes: KNN model, HOG-PCA, scaler, threshold, and class mapping
# --------------------------------------------------
print("\n💾 Saving model and preprocessing objects...")

# Rebuild HOG-PCA from original images (to match training pipeline)
hog_samples = []
count = 0
classes_local = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
for cls in classes_local:
    folder = os.path.join("./augmented_data/", cls)
    if not os.path.exists(folder):
        continue
    for f in os.listdir(folder):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            feats = extract_features_from_image(os.path.join(folder, f))
            if feats:
                hog_samples.append(feats[2])  # HOG features (index 2)
                count += 1
                if count >= 100:
                    break
    if count >= 100:
        break

hog_pca = PCA(n_components=70, whiten=True, random_state=42)
hog_pca.fit(np.array(hog_samples))

# Save all components to 'models/' directory
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/knn_model.pkl")
joblib.dump(hog_pca, "models/hog_pca.pkl")
joblib.dump(final_scaler, "models/scaler.pkl")
joblib.dump(threshold, "models/threshold.pkl")
joblib.dump(class_mapping, "models/class_mapping.pkl")

print("✅ All components saved in 'models/' folder:")
print("   - knn_model.pkl        # Trained KNN classifier")
print("   - hog_pca.pkl          # PCA transformer for HOG features")
print("   - scaler.pkl           # Feature standard scaler")
print("   - threshold.pkl        # Rejection threshold (90th percentile)")
print("   - class_mapping.pkl    # Mapping from class name to ID")
