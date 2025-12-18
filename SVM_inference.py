import os
import numpy as np
import joblib
from FeatureLead import CNNFeatureExtractor



classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash", "unknown"]


feature_extractor = CNNFeatureExtractor()
svm_model = joblib.load("models/svm_model.pkl")
scaler = joblib.load("models/scaler.pkl")

def classify_image(image_path):

    extracted = feature_extractor.extract_features(image_path)
    if extracted is None:
        return "unknown", 0.0

    extracted = extracted.reshape(1, -1)

    scaled_features = scaler.transform(extracted)

    class_id, confidence = predict_with_rejection(svm_model, scaled_features, threshold=0.55)
    predicted_label = "unknown" if class_id == 6 else classes[class_id]
    return predicted_label, confidence

def predict_with_rejection(model, features, threshold=0.55):
    probs = model.predict_proba(features)[0]
    best_class = np.argmax(probs)
    best_prob = probs[best_class]
    if best_prob < threshold:
        return 6, best_prob
    return best_class, best_prob



if __name__ == "__main__":
    test_image_paths = [
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
    for test_image_path in test_image_paths:
        if not os.path.exists(test_image_path):
            print(f"❌ Error: Image '{test_image_path}' not found!")
            continue

        predicted_class, conf = classify_image(test_image_path)
        print(f"Image: {test_image_path}, Predicted Class: {predicted_class}, Confidence: {conf:.4f}")