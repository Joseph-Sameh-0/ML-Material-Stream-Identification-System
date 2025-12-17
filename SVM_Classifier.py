import os
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from FeatureLead import load_features


# first step => load extracted features
print("Loading extracted features...")
X, y, scaler, class_mapping, filenames = load_features()

print(f"Loaded feature matrix: {X.shape}")
print(f"Loaded labels: {y.shape}")

# second step => split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)



print("\nDataset Split:")
print("Training samples:", len(X_train))
print("Validation samples:", len(X_test))

# third step => train SVM model
svm_model = SVC(
    C=10, # regularization parameter => higher C means less regularization => 10 got highest accuracy
    kernel="rbf", # rbf is better for non-linear data 
    gamma="scale", # default value for gamma and got best accuracy
    probability=True, # enable probability estimates => needed for rejection
)



print("\nTraining SVM (RBF kernel)...")
svm_model.fit(X_train, y_train)

# fourth step => evaluate on test set
print("\nEvaluating on test set...")
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nValidation Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# fifth step => save the trained model and scaler
os.makedirs("models", exist_ok=True)
joblib.dump(svm_model, "models/svm_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nSaved:")
print(" - models/svm_model.pkl")
print(" - models/scaler.pkl")

# sixth step => define predict_with_rejection function
def predict_with_rejection(model, features, threshold=0.55):
    probs = model.predict_proba(features)[0]
    best_class = np.argmax(probs)
    best_prob = probs[best_class]
    if best_prob < threshold:
        return 6, best_prob
    return best_class, best_prob

print("\nModel training complete.")
