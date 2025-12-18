# ML Material Stream Identification System

A machine learning-based system for automatically identifying and classifying recyclable materials using computer vision. The system uses deep learning feature extraction (ResNet50) combined with classical ML classifiers (SVM/KNN) to categorize materials into six classes: cardboard, glass, metal, paper, plastic, and trash.

## 🎯 Features

- **Automated Material Classification**: Identifies materials from images with high accuracy
- **Deep Learning Feature Extraction**: Uses pre-trained ResNet50 CNN for robust feature representation
- **Multiple Classifier Options**: Supports both SVM (primary) and KNN classifiers
- **Real-time Camera Integration**: Live classification using webcam feed
- **Data Augmentation Pipeline**: Balances dataset through intelligent augmentation
- **Rejection Mechanism**: Classifies uncertain predictions as "unknown" to improve reliability
- **Confidence Scoring**: Provides probability estimates for predictions

## 📋 Table of Contents

- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Pipeline Overview](#pipeline-overview)
- [Usage](#usage)
  - [1. Data Augmentation](#1-data-augmentation)
  - [2. Feature Extraction](#2-feature-extraction)
  - [3. Model Training](#3-model-training)
  - [4. Testing](#4-testing)
  - [5. Live Classification](#5-live-classification)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Configuration](#configuration)
- [Results](#results)

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for faster feature extraction)
- Webcam (for live classification)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/ML-Material-Stream-Identification-System.git
cd ML-Material-Stream-Identification-System
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Additional dependencies (if not included):

```bash
pip install scikit-learn opencv-python pillow
```

## 📁 Dataset Structure

Place your training images in the following structure:

```
data/
├── cardboard/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

The system will automatically:

- Count images per class
- Balance the dataset through augmentation
- Generate augmented data in `./augmented_data/`

## 🔄 Pipeline Overview

The system follows a four-stage pipeline:

```
Raw Images → Data Augmentation → Feature Extraction → Model Training → Inference
```

1. **Data Augmentation** (`DataLead.py`): Balances class distribution
2. **Feature Extraction** (`FeatureLead.py`): Extracts 2048-D features using ResNet50
3. **Model Training** (`SVM_Classifier.py` or `knn.py`): Trains classifier
4. **Inference** (`SVM_inference.py`, `predict_knn.py`, or `LiveApp.py`): Classifies new images

## 📖 Usage

### 1. Data Augmentation

Balance your dataset by augmenting underrepresented classes:

```bash
python DataLead.py
```

**What it does:**

- Counts images per class
- Identifies the maximum class count as target
- Applies augmentation transformations:
  - Random rotation (±10°)
  - Random horizontal/vertical flips
  - Color jittering (brightness, contrast, saturation, hue)
  - Resizing to 224×224
- Saves augmented data to `./augmented_data/`

**Output:**

```
cardboard: 450 images
glass: 380 images
...
Max class count: 450
Target count for all classes: 450
```

### 2. Feature Extraction

Extract deep learning features from augmented images:

```bash
python FeatureLead.py
```

**What it does:**

- Loads pre-trained ResNet50 (ImageNet weights)
- Processes all augmented images
- Extracts 2048-dimensional feature vectors
- Normalizes features using StandardScaler
- Saves to `./features/`:
  - `features.csv`: Feature vectors with labels
  - `class_mapping.json`: Class name to ID mapping
  - `scaler_params.csv`: Normalization parameters

**Output:**

```
Processing cardboard: 450 images
Processing glass: 450 images
...
Total samples: 2700
Feature vector size: 2048
```

### 3. Model Training

#### Option A: SVM Classifier (Recommended)

```bash
python SVM_Classifier.py
```

**Configuration:**

- Kernel: RBF (Radial Basis Function)
- C: 10 (regularization parameter)
- Gamma: scale
- Train/Test split: 80/20
- Rejection threshold: 0.55 (confidence)

**Output:**

```
Training samples: 2160
Validation samples: 540
Validation Accuracy: 0.9537
```

#### Option B: KNN Classifier

```bash
python knn.py
```

**Configuration:**

- Searches for optimal k (3-29, odd numbers)
- Uses distance-weighted voting
- Computes rejection threshold (90th percentile)
- Train/Test split: 80/20

**Output:**

```
Best k = 7 → Validation Accuracy: 0.9213
Rejection Threshold: 1.2345
```

### 4. Testing

Test the model on unseen images:

```bash
python test.py
```

Place test images in `./test_images/` directory. The script will:

- Load the trained SVM model
- Extract features for each test image
- Apply rejection mechanism
- Output predictions with class IDs

**Example Output:**

```
glass1.png: glass (ID: 1)
cardboard3.jpg: cardboard (ID: 0)
mixed_item.png: unknown (ID: 6)
```

### 5. Live Classification

Run real-time classification using your webcam:

```bash
python LiveApp.py
```

**Controls:**

- Press `q` to quit
- Classification updates every 30 frames for efficiency

**Features:**

- Real-time video feed
- On-screen class prediction and confidence
- Automatic frame processing and inference

## 🗂️ Project Structure

```
ML-Material-Stream-Identification-System/
├── DataLead.py              # Data augmentation pipeline
├── FeatureLead.py           # CNN feature extraction
├── SVM_Classifier.py        # SVM model training
├── SVM_inference.py         # SVM prediction utilities
├── knn.py                   # KNN model training
├── predict_knn.py           # KNN prediction utilities
├── LiveApp.py               # Real-time webcam classification
├── test.py                  # Batch testing script
├── requirements.txt         # Python dependencies
├── README.md                # Documentation
├── data/                    # Original training images
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
├── augmented_data/          # Augmented training images (generated)
├── features/                # Extracted features (generated)
│   ├── features.csv
│   ├── class_mapping.json
│   └── scaler_params.csv
├── models/                  # Trained models (generated)
│   ├── svm_model.pkl
│   ├── knn_model.pkl
│   ├── scaler.pkl
│   ├── threshold.pkl
│   └── class_mapping.pkl
└── test_images/             # Test images for evaluation
```

## 🧠 Model Details

### Feature Extraction

- **Architecture**: ResNet50 pre-trained on ImageNet
- **Feature Dimension**: 2048
- **Preprocessing**:
  - Resize to 224×224
  - Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- **Device**: Automatic GPU/CPU selection

### SVM Classifier

- **Algorithm**: Support Vector Machine with RBF kernel
- **Hyperparameters**:
  - C=10: Less regularization for complex decision boundaries
  - gamma='scale': Automatic gamma calculation
  - probability=True: Enables confidence scores
- **Rejection**: Predictions with confidence < 0.55 → "unknown"

### KNN Classifier

- **Algorithm**: k-Nearest Neighbors with distance weighting
- **Hyperparameters**:
  - k: Auto-selected (typically 5-9)
  - weights='distance': Closer neighbors have more influence
- **Rejection**: Predictions with distance > threshold → "unknown"

## ⚙️ Configuration

### Modifying Augmentation

Edit `DataLead.py`:

```python
augmentation_transforms = transforms.Compose([
    transforms.RandomRotation(degrees=10),      # Adjust rotation range
    transforms.RandomHorizontalFlip(p=0.5),     # Flip probability
    transforms.ColorJitter(brightness=0.2, ...),# Color adjustments
])
```

### Adjusting Rejection Threshold

**For SVM** (edit `SVM_inference.py`):

```python
def predict_with_rejection(model, features, threshold=0.55):  # Change threshold
```

**For KNN** (edit `knn.py`):

```python
threshold = np.percentile(distances[:, 0], 90)  # Change percentile
```

### Camera Settings

Edit `LiveApp.py`:

```python
camera = cv2.VideoCapture(0)  # Change camera index
skip_frames = 30              # Adjust classification frequency
```

## 📊 Results

### Expected Performance

| Metric              | SVM  | KNN       |
| ------------------- | ---- | --------- |
| Validation Accuracy | ~95% | ~92%      |
| Feature Dimension   | 2048 | 2048      |
| Inference Speed     | Fast | Very Fast |
| Memory Usage        | Low  | Medium    |

### Classification Classes

1. **Cardboard** (ID: 0)
2. **Glass** (ID: 1)
3. **Metal** (ID: 2)
4. **Paper** (ID: 3)
5. **Plastic** (ID: 4)
6. **Trash** (ID: 5)
7. **Unknown** (ID: 6) - Low confidence predictions

## 🔧 Troubleshooting

### Issue: CUDA out of memory

**Solution**: Feature extraction uses GPU. Disable CUDA:

```python
device = torch.device("cpu")
```

### Issue: Low accuracy

**Solutions**:

- Increase training data per class
- Adjust augmentation parameters
- Try different SVM hyperparameters (C, gamma)
- Increase rejection threshold for higher precision

### Issue: Camera not detected

**Solution**: Change camera index in `LiveApp.py`:

```python
camera = cv2.VideoCapture(1)  # Try different indices
```

## 📝 Notes

- The system requires augmented data before feature extraction
- Feature extraction is the most time-consuming step (~5-10 seconds per 100 images on CPU)
- SVM generally outperforms KNN for this task
- The rejection mechanism improves real-world reliability by avoiding false positives

## 👥 Authors

- Youssef Joseph - 20220389 - DataLead.py
- Joseph Sameh - 20220099 - FeatureLead.py
- Salma Mohamed - 20220152 - SVM_Classifier.py, SVM_inference.py
- Rana Ibrahim - 20220130 - knn.py, predict_knn.py
- Jonathan Mokhles - 20220100 - LiveApp.py, test.py, documentation
