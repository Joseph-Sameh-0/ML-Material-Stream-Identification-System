# predict_knn_simple.py
import numpy as np
import joblib
import os

# --------------------------------------------------
# Import CNN extractor from FeatureLead (without modifying it)
# --------------------------------------------------
from FeatureLead import CNNFeatureExtractor

# --------------------------------------------------
# Load saved components
# --------------------------------------------------
if not os.path.exists("models/knn_model.pkl"):
    print("❌ Error: Run 'knn.py' first to generate 'models/' folder.")
    exit(1)

knn_model = joblib.load("models/knn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
threshold = joblib.load("models/threshold.pkl")
classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


# --------------------------------------------------
# Prediction function
# --------------------------------------------------
def predict_image(image_path):
    """
    Extract CNN features from image and classify using trained KNN.
    """
    # Use the same CNN extractor as in FeatureLead
    extractor = CNNFeatureExtractor()
    features = extractor.extract_features(image_path)

    if features is None:
        return "unknown"

    # Preprocess exactly like during training
    features = features.reshape(1, -1)
    features_scaled = scaler.transform(features)

    # Predict with rejection
    dists, _ = knn_model.kneighbors(features_scaled)
    pred = knn_model.predict(features_scaled)[0]

    if dists[0, 0] > threshold:
        return "unknown"
    return classes[pred]


# --------------------------------------------------
# Run prediction
# --------------------------------------------------
# image_path = "woman.jpeg"  # ← Change this to your image
# image_path = "glass.jpg"
# image_path = "cardboard.jpg"
# image_path = "paper.jpg"
# image_path = "plastic.jpg"
# image_path = "metal.jpg"
image_path = "boy.jpeg"
if not os.path.exists(image_path):
    print(f"❌ Error: Image '{image_path}' not found!")
    exit(1)

result = predict_image(image_path)
print(f"Predicted Class: {result}")