import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

def load_data():
    with open("data/rps_landmarks.json", "r") as f:
        data = json.load(f)
    return np.array(data["X"], dtype=np.float32), np.array(data["y"], dtype=np.int64)


def train_classifier(X, y):
    print(f"Training on {len(X)} samples...")
    
    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    class_names = ["rock", "paper", "scissors"]
    print("Class distribution:")
    for i, count in zip(unique, counts):
        print(f"  {class_names[i]}: {count} samples")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Create and train pipeline
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=200, 
            multi_class="multinomial",
            random_state=42
        )
    )
    
    classifier.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = classifier.predict(X_test)
    
    print(classification_report(
        y_test, y_pred, 
        target_names=class_names
    ))
    
    cm = confusion_matrix(y_test, y_pred)
    print("        ", " ".join(f"{name:>8}" for name in class_names))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>8}", " ".join(f"{val:>8}" for val in row))
    
    accuracy = np.mean(y_pred == y_test)
    print(f"\nAccuracy: {accuracy:.3f}")
    
    return classifier

def save_model(classifier):
    os.makedirs("models", exist_ok=True)
    model_path = "models/rps_landmark_clf.joblib"
    joblib.dump(classifier, model_path)

def main():    
    # Load data
    X, y = load_data()
    if X is None:
        return
    # Train classifier
    classifier = train_classifier(X, y)
    
    # Save model
    save_model(classifier)


if __name__ == "__main__":
    main()
