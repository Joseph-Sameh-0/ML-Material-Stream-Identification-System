import os
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler, normalize
from tqdm import tqdm
import pandas as pd
import json
from sklearn.decomposition import PCA

# Base path to augmented data
augmented_base_path = "./augmented_data/"
output_path = "./features/"
os.makedirs(output_path, exist_ok=True)

classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
class_to_id = {cls: idx for idx, cls in enumerate(classes)}

print("=" * 60)
print("Feature Extraction Pipeline")
print("=" * 60)


def extract_color_histogram(image, bins=16):
    histogram_features = []
    for channel in range(3):
        channel_hist = cv2.calcHist([image], [channel], None, [bins], [0, 256])
        normalized_hist = cv2.normalize(channel_hist, channel_hist).flatten()
        histogram_features.extend(normalized_hist)
    return np.array(histogram_features)


def extract_hog_features(grayscale_image):
    resized_image = cv2.resize(grayscale_image, (32, 32))
    hog_descriptor = cv2.HOGDescriptor(
        _winSize=(32, 32),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    return hog_descriptor.compute(resized_image).flatten()


def extract_texture_features(grayscale_image):
    resized_image = cv2.resize(grayscale_image, (128, 128))
    gradient_x = cv2.Sobel(resized_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(resized_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    statistical_features = [
        np.mean(gradient_magnitude),
        np.std(gradient_magnitude),
        np.percentile(gradient_magnitude, 25),
        np.percentile(gradient_magnitude, 50),
        np.percentile(gradient_magnitude, 75),
        np.max(gradient_magnitude),
        np.min(gradient_magnitude),
    ]
    return np.array(statistical_features)


def extract_shape_features(grayscale_image):
    resized_image = cv2.resize(grayscale_image, (64, 64))
    edges = cv2.Canny(resized_image, 50, 150)
    image_moments = cv2.moments(edges)
    hu_moments = cv2.HuMoments(image_moments).flatten()
    log_hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

    intensity_mean = np.mean(resized_image)
    intensity_std = np.std(resized_image)

    return np.concatenate([log_hu_moments, [intensity_mean, intensity_std]])


def extract_features_from_image(image_path):
    try:
        bgr_image = cv2.imread(image_path)
        if bgr_image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        grayscale_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        color_features = extract_color_histogram(rgb_image, bins=16)
        hog_features = extract_hog_features(grayscale_image)
        texture_features = extract_texture_features(grayscale_image)
        shape_features = extract_shape_features(grayscale_image)

        return color_features, hog_features, texture_features, shape_features
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def process_dataset():
    hog_list = []
    color_list = []
    texture_list = []
    shape_list = []
    
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

            extracted_for_pca = extract_features_from_image(image_path)
            color_features, hog_features, texture_features, shape_features = extracted_for_pca
            full_features = np.concatenate(
                [hog_features, color_features, texture_features, shape_features]
            )
            extracted_features = full_features

            if extracted_features is not None:
                color_list.append(extracted_for_pca[0])
                # print(extracted_features[0])
                hog_list.append(extracted_for_pca[1])
                texture_list.append(extracted_for_pca[2])
                shape_list.append(extracted_for_pca[3])

                features_list.append(np.concatenate([color_list[-1], hog_list[-1], texture_list[-1], shape_list[-1]]))
                labels_list.append(class_to_id[class_name])
                filenames_list.append(image_path)

    hog = np.array(hog_list)
    color = np.array(color_list)    
    texture = np.array(texture_list)
    shape = np.array(shape_list)
    features_array = np.array(features_list)
    labels_array = np.array(labels_list)

    pca = PCA(n_components=70, whiten=True, random_state=42)
    hog_pca = pca.fit_transform(hog)

    hog_normalized = normalize(hog_pca, norm='l2')
    color_normalized = normalize(color, norm='l2')
    texture_normalized = normalize(texture, norm='l2')
    shape_normalized = normalize(shape, norm='l2')

    X = np.hstack((hog_normalized, color_normalized, texture_normalized, shape_normalized))



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
    normalized_features = scaler.fit_transform(X)

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
