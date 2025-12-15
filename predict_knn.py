import cv2
import numpy as np
import joblib
from sklearn.preprocessing import normalize
from FeatureLead import extract_features_from_image
import os

# --------------------------------------------------
# 1. Load all saved components from training
#    => Model, HOG-PCA, scaler, and rejection threshold
# --------------------------------------------------
if not os.path.exists("models/knn_model.pkl"):
    print("❌ Error: The 'models/' folder does not exist. Please run 'knn.py' first.")
    exit(1)

knn_model = joblib.load("models/knn_model.pkl")      # Trained KNN classifier
hog_pca = joblib.load("models/hog_pca.pkl")          # PCA for HOG features
scaler = joblib.load("models/scaler.pkl")            # Feature standard scaler
threshold = joblib.load("models/threshold.pkl")      # Rejection threshold (90th percentile)

# Class names in the same order as during training (IDs 0–5)
classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# --------------------------------------------------
# 2. Feature extraction and classification function
#    => Applies the exact same preprocessing pipeline used during training
#    => Returns class name or "unknown" if sample is rejected
# --------------------------------------------------
def extract_features(image_path):
    """
    Extract features from image and predict its class using the trained KNN model.
    Returns: class name (str) or "unknown" if rejected.
    """
    extracted = extract_features_from_image(image_path)
    if extracted is None:
        return "unknown"

    # Unpack the 7 feature groups (same order as in FeatureLead)
    color_features, color_moments, hog_features, lbp_features, \
        texture_features, shape_features, material_features = extracted

    # Apply the saved HOG-PCA transformer
    hog_pca_feat = hog_pca.transform(hog_features.reshape(1, -1))

    # L2-normalize each feature group separately (same as training)
    features = np.hstack([
        normalize(hog_pca_feat, 'l2'),
        normalize(color_features.reshape(1, -1), 'l2'),
        normalize(texture_features.reshape(1, -1), 'l2'),
        normalize(shape_features.reshape(1, -1), 'l2'),
        normalize(color_moments.reshape(1, -1), 'l2'),
        normalize(material_features.reshape(1, -1), 'l2'),
        normalize(lbp_features.reshape(1, -1), 'l2')
    ])

    # Apply global feature scaling
    features_scaled = scaler.transform(features)

    # KNN prediction with rejection
    dists, _ = knn_model.kneighbors(features_scaled)
    pred = knn_model.predict(features_scaled)[0]

    # Reject if nearest neighbor distance exceeds threshold
    if dists[0, 0] > threshold:
        return "unknown"
    return classes[pred]

# --------------------------------------------------
# 3. Run prediction on a single image
#    => Change image_path to test different images
# --------------------------------------------------
# image_path = "boy.jpeg"
# image_path = "cardboard.jpg"
# image_path = "glass.jpg"
# image_path = "metal.jpg"
# image_path = "paper.jpg"
# image_path = "plastic.jpg"
# image_path = "trash.jpg"
image_path = "woman.jpeg"  # ← Set your test image here

if not os.path.exists(image_path):
    print(f"❌ Error: Image '{image_path}' not found in the current directory!")
    exit(1)

result = extract_features(image_path)
print(f"Predicted Class: {result}")