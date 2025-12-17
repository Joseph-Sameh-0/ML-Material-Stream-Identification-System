# test.py

import os
import numpy as np
import joblib

from FeatureLead import CNNFeatureExtractor
from SVM_Classifier import predict_with_rejection


# Class index mapping (must match training)
classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash", "unknown"]


def predict(dataFilePath, bestModelPath):

    svm_model = joblib.load(bestModelPath)
    scaler = joblib.load("models/scaler.pkl")


    feature_extractor = CNNFeatureExtractor()

    predictions = []

    image_files = sorted(os.listdir(dataFilePath))

    for img_name in image_files:
        img_path = os.path.join(dataFilePath, img_name)


        extracted = feature_extractor.extract_features(img_path)

        if extracted is None:
            predictions.append(6)  # Unknown class
            continue

        extracted = extracted.reshape(1, -1)


        scaled_features = scaler.transform(extracted)

        class_id, confidence = predict_with_rejection(
            svm_model,
            scaled_features,
            threshold=0.55
        )

        predictions.append(int(class_id))

    return predictions
