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

# Use the same CNN extractor as in FeatureLead
extractor = CNNFeatureExtractor()


# --------------------------------------------------
# Prediction function
# --------------------------------------------------
def predict_with_rejection(image_path):
    """
    Extract CNN features from image and classify using trained KNN.
    Returns (class_id, confidence)
    """
    features = extractor.extract_features(image_path)

    if features is None:
        return 6, 0.0

    # Preprocess exactly like during training
    features = features.reshape(1, -1)
    features_scaled = scaler.transform(features)

    # Predict with rejection
    dists, _ = knn_model.kneighbors(features_scaled)
    pred = knn_model.predict(features_scaled)[0]

    if dists[0, 0] > threshold:
        return 6, dists[0, 0]
    return int(pred), dists[0, 0]


# --------------------------------------------------
# Run prediction
# --------------------------------------------------
image_paths = [
    "test_images/woman.jpeg",
    "test_images/glass.jpg",
    "test_images/glass2.png",
    "test_images/cardboard.jpg",
    "test_images/paper.jpg",
    "test_images/plastic.jpg",
    "test_images/metal.jpg",
    "test_images/metal2.png",
    "test_images/trash.jpg",
    "test_images/boy.jpeg"
    ]
for image_path in image_paths:
    if not os.path.exists(image_path):
        print(f"❌ Error: Image '{image_path}' not found!")
        continue

    class_id, confidence = predict_with_rejection(image_path)
    predicted_class = "unknown" if class_id == 6 else classes[class_id]
    print(f"Image: {image_path}, Predicted Class: {predicted_class}, Confidence: {confidence:.4f}")
