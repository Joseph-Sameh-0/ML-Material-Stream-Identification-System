import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from FeatureLead import load_features

# --------------------------------------------------
# 2. Load features and labels
# --------------------------------------------------
print("Loading features and labels from team's feature pipeline...")
X, y, final_scaler, class_mapping, _ = load_features()

# --------------------------------------------------
# 3. Split data: 80% training, 20% validation
# --------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------------------------------
# 4. Train k-NN and find the best k
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
# 6. Prepare Rejection Mechanism (for real-time deployment)
# --------------------------------------------------
# Compute rejection threshold from validation set distances
distances, _ = best_model.kneighbors(X_val)
min_distances = distances[:, 0]
threshold = np.percentile(min_distances, 90)
print(f"\n📌 Rejection Threshold (for real-time use): {threshold:.4f}")

# Classification function with rejection (for Unknown class = 6)
def classify_with_rejection(model, X, thresh):
    dists, _ = model.kneighbors(X)
    preds = model.predict(X)
    preds[dists[:, 0] > thresh] = 6  # Assign Unknown (class 6)
    return preds

# --------------------------------------------------
# 7. 🧪 Test Rejection Mechanism using synthetic samples
# --------------------------------------------------
print("\n" + "="*50)
print("🧪 Testing Rejection Mechanism with synthetic data")
print("="*50)

# Real sample (should NOT be rejected)
real_sample = X_val[0:1]

# Fake/Noisy sample (should BE rejected)
np.random.seed(42)
fake_sample = real_sample + np.random.normal(0, 10, size=real_sample.shape)

# Classify both
pred_real = classify_with_rejection(best_model, real_sample, threshold)
pred_fake = classify_with_rejection(best_model, fake_sample, threshold)

print(f"Real sample → Prediction: {pred_real[0]}")
print(f"Noisy sample → Prediction: {pred_fake[0]}")

# Verify behavior
if pred_real[0] != 6:
    print("✅ Real sample was NOT rejected (correct!)")
else:
    print("⚠️ Real sample was incorrectly rejected! Threshold may be too low.")

if pred_fake[0] == 6:
    print("✅ Noisy sample was correctly rejected (rejection mechanism works!)")
else:
    print("⚠️ Noisy sample was NOT rejected! Threshold may be too high.")

# --------------------------------------------------
# 8. Save required files for submission and deployment
# --------------------------------------------------
# joblib.dump(best_model, 'knn_model.pkl')
# joblib.dump(threshold, 'knn_rejection_threshold.pkl')
# print("\n💾 Saved: knn_model.pkl + knn_rejection_threshold.pkl")
#
# print("\n✅ Code is ready for submission and real-time integration!")
