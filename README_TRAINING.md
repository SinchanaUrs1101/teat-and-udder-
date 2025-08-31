# Cow Udder Health Detection - Training Guide

This guide explains how to train the Cow Udder Health Detection model using the enhanced training script.

## Enhanced Training Features

The training script has been enhanced with the following features:

1. **Epoch-based Training**: Train the model for a specified number of epochs
2. **Comprehensive Metrics**: Calculate and display accuracy, precision, recall, and F1 score
3. **Visualization**: Generate graphs showing model performance over epochs
4. **Confusion Matrix**: Display a confusion matrix to analyze model predictions
5. **Test Set Evaluation**: Evaluate the model on a separate test set

## How to Run the Training

To train the model, use the following command:

```bash
python train_softmax_model.py --epochs 10
```

### Command-line Arguments

- `--epochs`: Number of training epochs (default: 10)
- `--no-vgg16`: Disable VGG16 feature extraction (by default, VGG16 features are enabled)

Examples:

```bash
# Train for 20 epochs with VGG16 features
python train_softmax_model.py --epochs 20

# Train for 10 epochs without VGG16 features
python train_softmax_model.py --no-vgg16
```

## Training Process

The training process follows these steps:

1. **Data Loading**: Load and preprocess images from the DS_COW directory
2. **Feature Extraction**: Extract features from images (with or without VGG16)
3. **Model Training**: Train the RandomForestClassifier for the specified number of epochs
4. **Metrics Calculation**: Calculate performance metrics after each epoch
5. **Visualization**: Generate and save performance graphs
6. **Model Saving**: Save the trained model to 'udder_model.pkl'

## Output

The training script will output:

1. **Console Output**: Detailed metrics for each epoch and final performance
2. **Saved Model**: The trained model saved to 'udder_model.pkl'
3. **Performance Graphs**: Four graphs saved to the 'training_plots' directory:
   - `accuracy.png`: Training and validation accuracy over epochs
   - `prf.png`: Precision, recall, and F1 score over epochs
   - `confusion_matrix.png`: Normalized confusion matrix
   - `final_metrics.png`: Bar chart of final performance metrics

## Notes on Epoch Implementation

Since RandomForestClassifier doesn't natively support epoch-based training, epochs are simulated by:

1. Dividing the total number of trees (n_estimators) by the number of epochs
2. Incrementally increasing the number of trees in each epoch
3. Retraining the model with the updated number of trees

This approach allows for monitoring the model's performance as it grows in complexity.

## Model Metrics

The training script calculates and reports the following metrics:

- **Accuracy**: Proportion of correctly classified samples
- **Precision**: Ability of the model to avoid false positives
- **Recall**: Ability of the model to find all positive samples
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions by class

These metrics are calculated for both the validation set (during training) and the test set (after training).

## Saved Model Data

The trained model is saved with the following information:

```python
model_data = {
    'model': model,                          # The trained RandomForestClassifier
    'classes': CATEGORIES,                   # Class names
    'training_accuracy': train_accuracies[-1],
    'validation_accuracy': val_accuracies[-1],
    'test_accuracy': test_accuracy,
    'precision': test_precision,
    'recall': test_recall,
    'f1_score': test_f1,
    'img_size': IMG_SIZE,                    # Image size used for training
    'use_vgg16': use_vgg16,                  # Whether VGG16 features were used
    'vgg16_features': {},                    # VGG16 feature information
    'scaler': scaler,                        # Feature scaler
    'confusion_matrix': cm.tolist()          # Confusion matrix
}
```

This comprehensive model data allows for detailed analysis and reporting in the application.
