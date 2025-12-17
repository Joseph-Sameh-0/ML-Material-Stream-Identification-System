import numpy as np
import joblib
from FeatureLead import CNNFeatureExtractor
from SVM_Classifier import predict_with_rejection

# load the trained SVM model and scaler
svm_model = joblib.load("models/svm_model.pkl")
scaler = joblib.load("models/scaler.pkl")

classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash", "unknown"]

feature_extractor = CNNFeatureExtractor()

def classify_image(image_path):
    extracted = feature_extractor.extract_features(image_path)
    if extracted is None:
        return "unknown", 0.0

    extracted = extracted.reshape(1, -1)

    scaled_features = scaler.transform(extracted)

    
    class_id, confidence = predict_with_rejection(svm_model, scaled_features, threshold=0.55)
    predicted_label = "unknown" if class_id == 6 else classes[class_id]
    return predicted_label, confidence


if __name__ == "__main__":  
    test_image_path = "test_images/glass2.png"  
    predicted_class, conf = classify_image(test_image_path)
    print(f"Predicted Class: {predicted_class}, Confidence: {conf:.4f}")