import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from model import load_or_create_model, extract_features, classify_image

# Load the model
print("Loading model...")
model = load_or_create_model()

# Test with sample image from each category
def test_prediction(image_path, expected_category=None):
    """Test prediction on a sample image"""
    print(f"\nTesting with image: {image_path}")
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Get RGB version for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Extract features
    features = extract_features(img_rgb, model)
    
    # Make prediction
    condition_probs, predicted_condition = classify_image(features, model)
    
    # Print results
    print(f"Predicted condition: {predicted_condition}")
    print(f"Probabilities:")
    for condition, prob in sorted(condition_probs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {condition}: {prob:.4f}")
    
    # Verify against expected category if provided
    if expected_category:
        expected_display = expected_category.replace('_', ' ').title()
        if predicted_condition == expected_display:
            print(f"✓ CORRECT: Prediction matches expected category '{expected_category}'")
        else:
            print(f"✗ INCORRECT: Prediction '{predicted_condition}' does not match expected '{expected_category}'")

# Test with one image from each category folder
def run_tests():
    # Categories to test
    categories = [
        'healthy',
        'frozen_teats',
        'mastitis',
        'teat_lesions',
        'low_udder_score',
        'medium_udder_score',
        'high_udder_score'
    ]
    
    for category in categories:
        # Get an image from the category folder
        category_dir = os.path.join('DS_COW', 'train', category)
        if not os.path.exists(category_dir):
            print(f"Warning: Category folder {category} does not exist")
            continue
        
        images = [f for f in os.listdir(category_dir) if f.endswith('.jpg')]
        if not images:
            print(f"Warning: No images found in category {category}")
            continue
        
        # Test with the first image in the folder
        test_image = os.path.join(category_dir, images[0])
        test_prediction(test_image, category)

if __name__ == "__main__":
    run_tests()