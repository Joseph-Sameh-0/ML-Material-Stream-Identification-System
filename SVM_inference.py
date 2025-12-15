import cv2
import numpy as np
import joblib
from sklearn.preprocessing import normalize
from FeatureLead import extract_features_from_image
from SVM_Classifier import predict_with_rejection
from sklearn.decomposition import PCA

# load the trained SVM model
svm_model = joblib.load("models/svm_model.pkl")
scaler = joblib.load("models/scaler.pkl")
pca = joblib.load("models/hog_pca.pkl")

classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash", "unknown"]


def extract_features(image_path):
    extracted = extract_features_from_image(image_path)
    if extracted is None:
        return "unknown", 0.0

    color_features, color_moments, hog_features, lbp_features, texture_features, shape_features, material_features = extracted

    # 1) PCA on HOG (match training)
    hog_pca = pca.transform(hog_features.reshape(1, -1))

    # 2) L2-normalize each block (same as training)
    hog_norm = normalize(hog_pca, norm='l2')
    color_norm = normalize(color_features.reshape(1, -1), norm='l2')
    texture_norm = normalize(texture_features.reshape(1, -1), norm='l2')
    shape_norm = normalize(shape_features.reshape(1, -1), norm='l2')
    color_moments_norm = normalize(color_moments.reshape(1, -1), norm='l2')
    material_norm = normalize(material_features.reshape(1, -1), norm='l2')
    lbp_norm = normalize(lbp_features.reshape(1, -1), norm='l2')

    # 3) Concatenate in the same order as X
    features = np.hstack((
        hog_norm, color_norm, texture_norm,
        shape_norm, color_moments_norm,
        material_norm, lbp_norm
    ))

    # 4) Apply the saved StandardScaler
    features_scaled = scaler.transform(features)

    # 5) Predict with rejection
    class_id, confidence = predict_with_rejection(svm_model, features_scaled, threshold=0.55)
    predicted_label = "unknown" if class_id == 6 else classes[class_id]
    return predicted_label, confidence


if __name__ == "__main__":  
    test_image_path = "test_images/woman.jpeg"  
    predicted_class, conf = extract_features(test_image_path)
    print(f"Predicted Class: {predicted_class}, Confidence: {conf:.4f}")