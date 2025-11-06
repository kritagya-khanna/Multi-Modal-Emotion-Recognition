import tensorflow as tf
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ReduceLROnPlateau, 
    ModelCheckpoint, 
    TensorBoard
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import time
from pathlib import Path
import datetime
import argparse
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your modules
from models.audio.cnn_lstm import build_cnn_lstm_attention
from utils.data_generator import AudioDataGenerator
from utils.spectrogram_generator import load_and_prepare_data
from utils.metrics import calculate_metrics, plot_confusion_matrix, plot_metrics, print_classification_report
from sklearn.utils.class_weight import compute_class_weight

def train_audio_model(config_path):
    """
    Train audio emotion recognition model
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        model: Trained model
        history: Training history
        experiment_dir: Path to experiment directory
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create experiment directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_dir = Path(f"experiments/{config['model']['name']}_{timestamp}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config to experiment directory
    with open(experiment_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=4)
    
    # Load and prepare data
    print("\n=== Loading and preparing data ===")
    X_train, X_val, X_test, y_train, y_val, y_test, label_mapping = load_and_prepare_data(config)
    
    # Save label mapping to experiment directory
    with open(experiment_dir / "label_mapping.json", 'w') as f:
        json.dump({str(v): k for k, v in label_mapping.items()}, f, indent=4)
    
    # Create data generators
    train_generator = AudioDataGenerator(
        X_train, y_train,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        augment=config['augmentation']['enabled'],
        config=config
    )
    
    val_generator = AudioDataGenerator(
        X_val, y_val,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    test_generator = AudioDataGenerator(
        X_test, y_test,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )

    # Calculate class weights to handle imbalance
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\nClass weights: {class_weight_dict}")
    
    # Create model
    print("\n=== Building model ===")
    model = build_cnn_lstm_attention(config)
    
    # Print model summary
    model.summary()
    
    # Save model architecture visualization
    try:
        from tensorflow.keras.utils import plot_model
        plot_model(model, to_file=experiment_dir / "model_architecture.png", show_shapes=True)
    except ImportError:
        print("Could not generate model visualization. Make sure pydot and graphviz are installed.")
    
    # Compile model
    optimizer = AdamW(
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config['training']['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=config['training']['reduce_lr_factor'],
            patience=config['training']['reduce_lr_patience'],
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=experiment_dir / "best_model.h5",
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        TensorBoard(
            log_dir=experiment_dir / "logs",
            histogram_freq=1,
            update_freq='epoch'
        )
    ]
    
    # Train model
    print("\n=== Training model ===")
    start_time = time.time()
    
    history = model.fit(
        train_generator,
        epochs=config['training']['epochs'],
        validation_data=val_generator,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Save final model
    model.save(experiment_dir / "final_model.h5")
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(experiment_dir / "training_history.csv", index=False)
    
    # Plot training history
    plot_metrics(history, output_path=experiment_dir / "training_history.png")
    
    # Evaluate on test set
    print("\n=== Evaluating model on test set ===")
    evaluate_model(model, X_test, y_test, label_mapping, experiment_dir)
    
    # Analyze predictions to check if model is stuck
    print("\n=== Analyzing predictions ===")
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Check prediction distribution
    unique, counts = np.unique(y_pred, return_counts=True)
    print("\nPrediction distribution on test set:")
    for emotion_idx, count in zip(unique, counts):
        emotion_name = label_mapping[emotion_idx]
        print(f"  {emotion_name}: {count} ({count/len(y_pred)*100:.1f}%)")
    
    # Check if it's predicting mostly one class
    if max(counts) > len(y_pred) * 0.5:
        print(f"\n⚠️  WARNING: Model is predicting mostly ONE class!")
        print(f"   Predicted '{label_mapping[unique[np.argmax(counts)]]}' for {max(counts)/len(y_pred)*100:.1f}% of samples")
    
    # Check actual distribution
    unique_true, counts_true = np.unique(y_test, return_counts=True)
    print("\nActual distribution on test set:")
    for emotion_idx, count in zip(unique_true, counts_true):
        emotion_name = label_mapping[emotion_idx]
        print(f"  {emotion_name}: {count} ({count/len(y_test)*100:.1f}%)")
    
    return model, history, experiment_dir

def evaluate_model(model, X_test, y_test, label_mapping, output_dir):
    """
    Evaluate model on test set
    
    Args:
        model: Trained model
        X_test: Test spectrogram data
        y_test: Test labels
        label_mapping: Dictionary mapping label indices to names
        output_dir: Directory to save evaluation results
    """
    # Get predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    class_names = [label_mapping[i] for i in range(len(label_mapping))]
    metrics = calculate_metrics(y_test, y_pred, class_names)
    
    # Print classification report
    print_classification_report(metrics)
    
    # Plot confusion matrix
    plot_confusion_matrix(metrics, output_path=output_dir / "confusion_matrix.png")
    
    # Save metrics to file
    results = {
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'per_class': metrics['per_class']
    }
    
    with open(output_dir / "test_results.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nEvaluation results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train audio emotion recognition model")
    parser.add_argument(
        "--config", 
        type=str,
        default="config/audio_config.json",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    train_audio_model(args.config)


