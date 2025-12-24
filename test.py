# test.py

import os
import numpy as np
import joblib
import csv

from FeatureLead import CNNFeatureExtractor
from SVM_inference import predict_with_rejection


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
            # Unknown class (index 6) and no confidence
            predictions.append({"image": img_name, "class_id": 6, "confidence": 0.0})
            continue

        extracted = extracted.reshape(1, -1)


        scaled_features = scaler.transform(extracted)

        class_id, confidence = predict_with_rejection(
            svm_model,
            scaled_features,
            threshold=0.55
        )

        predictions.append({"image": img_name, "class_id": int(class_id), "confidence": float(confidence)})

    return predictions


if __name__ == "__main__":
    dataFilePath = "test_images/"
    bestModelPath = "models/svm_model.pkl"

    predictions = predict(dataFilePath, bestModelPath)

    # Print results and export to CSV
    out_csv = "predictions.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "class_id", "class_name", "confidence"])
        writer.writeheader()
        for p in predictions:
            class_name = classes[p["class_id"]]
            print(f"{p['image']}: {class_name} (ID: {p['class_id']}) - conf: {p.get('confidence')}")
            writer.writerow({
                "image": p["image"],
                "class_id": p["class_id"],
                "class_name": class_name,
                "confidence": p.get("confidence", "")
            })

    print(f"Wrote {len(predictions)} rows to {out_csv}")