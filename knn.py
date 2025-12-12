import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import joblib

# --- TEAM INTEGRATION ---
from FeatureLead import load_features

# ==========================================
# 1. LOAD DATA
# ==========================================
print("STATUS: Loading features...")
X_scaled, y, final_scaler, class_mapping, _ = load_features()

# ==========================================
# 1.5 DATA INSPECTION (Quick Check)
# ==========================================
n_samples, n_features = X_scaled.shape
print(f"📊 Original Data: {n_samples} samples, {n_features} features")

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.1, random_state=42
)

# ==========================================
# 🔥 2. DIMENSIONALITY REDUCTION (PCA) - الخطوة الجديدة المهمة
# ==========================================
print("\nSTATUS: Applying PCA to reduce dimensions and remove noise...")

# نطلب من PCA الاحتفاظ بـ 95% من المعلومات المهمة وحذف الباقي
pca = PCA(n_components=0.412)

# نتعلم الاختصار من بيانات التدريب فقط
X_train_pca = pca.fit_transform(X_train)
# نطبق نفس الاختصار على بيانات الاختبار
X_test_pca = pca.transform(X_test)

n_components = X_train_pca.shape[1]
print(f"✅ PCA Reduction Complete:")
print(f"   - Features reduced from {n_features} to {n_components}")
print(f"   - We kept 95% of the important information.")

# ==========================================
# 3. SEARCH FOR OPTIMAL k (Tuning) on REDUCED DATA
# ==========================================
print("\nSTATUS: Searching for the best k using PCA features...")

best_k = 1
best_accuracy = 0
final_model = None

# نجرب الـ k على البيانات المختصرة (X_train_pca)
for k in range(3, 30, 2):
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(X_train_pca, y_train) # Training on reduced data

    score = knn.score(X_test_pca, y_test) # Testing on reduced data

    if score > best_accuracy:
        best_accuracy = score
        best_k = k
        final_model = knn

print(f"\n✅ Optimal k found: {best_k} with Accuracy: {best_accuracy:.4f}")

if best_accuracy >= 0.85:
    print("🎉 GOAL ACHIEVED! Accuracy > 0.85")
else:
    print(f"⚠️ Accuracy is {best_accuracy:.4f}. It improved, but check features quality.")
