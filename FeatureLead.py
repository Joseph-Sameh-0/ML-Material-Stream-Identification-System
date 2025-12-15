import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pandas as pd
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# Base path to augmented data
augmented_base_path = "./augmented_data/"
output_path = "./features/"
os.makedirs(output_path, exist_ok=True)

classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
class_to_id = {cls: idx for idx, cls in enumerate(classes)}

print("=" * 60)
print("Feature Extraction Pipeline")
print("=" * 60)

device = torch.device("cpu")

class CNNFeatureExtractor:
    def __init__(self):
        print(f"\nLoading pre-trained ResNet50 model...")

        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.feature_dim = 2048

        self.model = self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        print(f"Model loaded successfully. Feature dimension: {self.feature_dim}")

    def extract_features(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0)
            image_tensor = image_tensor.to(device)

            with torch.no_grad():
                features = self.model(image_tensor)

            features = features.squeeze().cpu().numpy()
            return features

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None


def process_dataset():
    feature_extractor = CNNFeatureExtractor()

    features_list = []
    labels_list = []
    filenames_list = []

    print("\nProcessing images and extracting features...\n")

    for class_name in classes:
        class_folder = os.path.join(augmented_base_path, class_name)

        if not os.path.exists(class_folder):
            print(f"Warning: Folder {class_folder} not found. Skipping.")
            continue

        image_files = [
            filename
            for filename in os.listdir(class_folder)
            if filename.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        print(f"Processing {class_name}: {len(image_files)} images")

        for image_filename in tqdm(image_files, desc=f"  {class_name}", ncols=80):
            image_path = os.path.join(class_folder, image_filename)

            extracted_features = feature_extractor.extract_features(image_path)

            if extracted_features is not None:
                features_list.append(extracted_features)
                labels_list.append(class_to_id[class_name])
                filenames_list.append(image_path)

    features_array = np.array(features_list)
    labels_array = np.array(labels_list)

    print(f"\n{'='*60}")
    print(f"Feature Extraction Complete!")
    print(f"{'='*60}")
    print(f"Total samples: {len(features_array)}")

    if len(features_array) == 0:
        print("ERROR: No features were extracted! Please check for errors above.")
        return None, None, None

    print(f"Feature vector size: {features_array.shape[1]}")
    print(f"Class distribution:")
    for class_name, class_id in class_to_id.items():
        sample_count = np.sum(labels_array == class_id)
        print(f"  {class_name} (ID {class_id}): {sample_count} samples")

    print(f"\nNormalizing features...")
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features_array)

    print(f"\nSaving extracted features...")

    feature_column_names = [f"feature_{i}" for i in range(normalized_features.shape[1])]
    features_dataframe = pd.DataFrame(normalized_features, columns=feature_column_names)
    features_dataframe["label"] = labels_array
    features_dataframe["class_name"] = [classes[label] for label in labels_array]
    features_dataframe["filename"] = filenames_list

    features_dataframe.to_csv(os.path.join(output_path, "features.csv"), index=False)

    with open(os.path.join(output_path, "class_mapping.json"), "w") as json_file:
        json.dump(class_to_id, json_file, indent=2)

    scaler_dataframe = pd.DataFrame(
        {"mean": scaler.mean_, "scale": scaler.scale_, "var": scaler.var_}
    )
    scaler_dataframe.to_csv(os.path.join(output_path, "scaler_params.csv"), index=False)

    print(f"\nFeatures saved to: {os.path.abspath(output_path)}")
    print(
        f"   - features.csv: All features, labels, class names, and filenames ({features_dataframe.shape})"
    )
    print(f"   - class_mapping.json: Class name to ID mapping")
    print(f"   - scaler_params.csv: Feature scaler parameters (mean, scale, var)")

    return normalized_features, labels_array, scaler


def load_features():
    features_dataframe = pd.read_csv(os.path.join(output_path, "features.csv"))

    feature_column_names = [
        col for col in features_dataframe.columns if col.startswith("feature_")
    ]
    features_array = features_dataframe[feature_column_names].values
    labels_array = features_dataframe["label"].values
    filenames_list = features_dataframe["filename"].tolist()

    with open(os.path.join(output_path, "class_mapping.json"), "r") as json_file:
        class_mapping = json.load(json_file)

    scaler_dataframe = pd.read_csv(os.path.join(output_path, "scaler_params.csv"))

    scaler = StandardScaler()
    scaler.mean_ = scaler_dataframe["mean"].values
    scaler.scale_ = scaler_dataframe["scale"].values
    scaler.var_ = scaler_dataframe["var"].values

    return features_array, labels_array, scaler, class_mapping, filenames_list


if __name__ == "__main__":
    if not os.path.exists(augmented_base_path):
        print(f"Error: Augmented data folder not found at {augmented_base_path}")
        print("Please run DataLead.py first to generate augmented data.")
    else:
        normalized_features, labels, scaler = process_dataset()

        print(f"\n{'='*60}")
        print("Example: Loading Extracted Features")
        print(f"{'='*60}\n")

        print("Loading features from disk...")
        (
            loaded_features,
            loaded_labels,
            loaded_scaler,
            loaded_class_mapping,
            loaded_filenames,
        ) = load_features()

        print(f"\nFeatures successfully loaded:")
        print(f"   Features shape: {loaded_features.shape}")
        print(f"   Labels shape: {loaded_labels.shape}")
        print(f"   Feature mean: {loaded_features.mean():.6f}")
        print(f"   Feature std: {loaded_features.std():.6f}")
        print(f"   Filenames loaded: {len(loaded_filenames)}")

        print(f"\nClass mapping:")
        for class_name, class_id in sorted(
            loaded_class_mapping.items(), key=lambda x: x[1]
        ):
            sample_count = np.sum(loaded_labels == class_id)
            print(f"   {class_name} (ID {class_id}): {sample_count} samples")

        print(f"\nExample: First sample")
        print(f"   Feature vector shape: {loaded_features[0].shape}")
        print(f"   Feature vector (first 10 values): {loaded_features[0][:10]}")
        print(f"   Label: {loaded_labels[0]} ({classes[loaded_labels[0]]})")

        print(f"\n{'='*60}\n")
