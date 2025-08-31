import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
import time

# Define the categories and paths
CATEGORIES = [
    'healthy',
    'frozen_teats',
    'mastitis',
    'teat_lesions',
    'low_udder_score',
    'medium_udder_score',
    'high_udder_score'
]
DATASET_DIR = 'DS_COW'
IMG_SIZE = 224  # For consistency with our app

def load_training_data():
    """Load training data from the organized dataset structure"""
    X_train = []
    y_train = []
    X_val = []
    y_val = []

    print("Loading training and validation data...")

    # Load training data
    for idx, category in enumerate(CATEGORIES):
        train_dir = os.path.join(DATASET_DIR, 'train', category)
        valid_dir = os.path.join(DATASET_DIR, 'valid', category)

        print(f"Loading {category} training images...")

        # Load training images
        for img_file in os.listdir(train_dir):
            if not img_file.endswith('.jpg'):
                continue

            try:
                img_path = os.path.join(train_dir, img_file)
                img = cv2.imread(img_path)

                if img is None:
                    print(f"Warning: Could not read {img_path}")
                    continue

                # Preprocess the image (same preprocessing as in the app)
                img = preprocess_for_training(img)

                X_train.append(img)
                y_train.append(idx)
            except Exception as e:
                print(f"Error processing {img_file}: {e}")

        print(f"Loading {category} validation images...")

        # Load validation images
        for img_file in os.listdir(valid_dir):
            if not img_file.endswith('.jpg'):
                continue

            try:
                img_path = os.path.join(valid_dir, img_file)
                img = cv2.imread(img_path)

                if img is None:
                    print(f"Warning: Could not read {img_path}")
                    continue

                # Preprocess the image
                img = preprocess_for_training(img)

                X_val.append(img)
                y_val.append(idx)
            except Exception as e:
                print(f"Error processing {img_file}: {e}")

    # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)

    # Shuffle the data
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_val, y_val = shuffle(X_val, y_val, random_state=42)

    print(f"Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")
    print(f"Validation data shape: {X_val.shape}, Labels shape: {y_val.shape}")

    return X_train, y_train, X_val, y_val

def preprocess_for_training(image):
    """Preprocess an image for model training"""
    # Convert to RGB if needed
    if image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Resize to target dimension
    resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))

    # Normalize pixel values
    normalized = resized / 255.0

    return normalized

def build_model():
    """Build a model for cow udder condition classification"""
    # We'll use scikit-learn models instead of TensorFlow
    from sklearn.ensemble import RandomForestClassifier

    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,  # Use all available cores
        verbose=1
    )

    return rf_model

def extract_features(images):
    """Extract features from images for traditional ML models"""
    print("Extracting features from images...")
    features = []

    for img in images:
        # Resize for consistency
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Convert to uint8 for OpenCV processing
        img_uint8 = (img_resized * 255).astype(np.uint8)

        # Convert to grayscale for some features
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

        # Extract histogram features
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        # Simple color statistics
        color_means = np.mean(img_resized, axis=(0, 1))
        color_stds = np.std(img_resized, axis=(0, 1))

        # Simple texture features - using Sobel filter which is more robust
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        texture_stats = [
            np.mean(np.abs(sobel_x)),
            np.std(np.abs(sobel_x)),
            np.mean(np.abs(sobel_y)),
            np.std(np.abs(sobel_y))
        ]

        # Combine all features
        img_features = np.concatenate([
            hist,
            color_means,
            color_stds,
            texture_stats
        ])

        features.append(img_features)

    return np.array(features)

def train_model():
    """Train the cow udder health detection model"""
    # Load the data
    X_train, y_train, X_val, y_val = load_training_data()

    # Flatten images and extract features for ML model
    print("Preparing features for training...")
    X_train_features = extract_features(X_train)
    X_val_features = extract_features(X_val)

    print(f"Feature vector shape: {X_train_features.shape}")

    # Build the model
    model = build_model()

    # Train the model
    print("Training the model...")
    start_time = time.time()
    model.fit(X_train_features, y_train)
    training_time = time.time() - start_time
    print(f"Model training completed in {training_time:.2f} seconds")

    # Evaluate on validation set
    val_accuracy = model.score(X_val_features, y_val)
    print(f"Validation accuracy: {val_accuracy:.4f}")

    # Evaluate on training set
    train_accuracy = model.score(X_train_features, y_train)
    print(f"Training accuracy: {train_accuracy:.4f}")

    # Save the model
    print("Saving the model...")
    model_data = {
        'model': model,
        'classes': CATEGORIES,
        'training_accuracy': train_accuracy,
        'validation_accuracy': val_accuracy,
        'img_size': IMG_SIZE
    }

    with open('udder_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    return model, train_accuracy, val_accuracy

if __name__ == "__main__":
    print("Starting model training...")
    model, train_acc, val_acc = train_model()
    print("Model training complete!")
    print(f"Final training accuracy: {train_acc:.4f}")
    print(f"Final validation accuracy: {val_acc:.4f}")