import tensorflow as tf
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import time
from pathlib import Path
import datetime
import argparse
import sys
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fusion.multimodal_fusion import build_early_fusion_model, build_late_fusion_model, build_hybrid_fusion_model
from utils.multimodal_data_generator import MultimodalDataGenerator, load_and_prepare_multimodal_data
from utils.metrics import calculate_metrics, plot_confusion_matrix, print_classification_report


def train_fusion_model(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    fusion_type = config['model']['fusion_type']
    experiment_dir = Path(f"experiments/fusion_{fusion_type}_{timestamp}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    with open(experiment_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=4)
    
    print("\n=== Loading and preparing multimodal data ===")
    (audio_train, audio_val, audio_test,
     video_train, video_val, video_test,
     y_train, y_val, y_test, label_mapping) = load_and_prepare_multimodal_data(config)
    
    with open(experiment_dir / "label_mapping.json", 'w') as f:
        json.dump({str(k): v for k, v in label_mapping.items()}, f, indent=4)
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\nClass weights: {class_weight_dict}")
    
    train_generator = MultimodalDataGenerator(
        audio_train, video_train, y_train,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        augment=config['augmentation']['enabled'],
        config=config
    )
    
    val_generator = MultimodalDataGenerator(
        audio_val, video_val, y_val,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    print("\n=== Building fusion model ===")
    fusion_type = config['model']['fusion_type']
    
    if fusion_type == 'early':
        model = build_early_fusion_model(config)
    elif fusion_type == 'late':
        model = build_late_fusion_model(config)
    elif fusion_type == 'hybrid':
        model = build_hybrid_fusion_model(config)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    optimizer = AdamW(
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=config['training']['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=config['training']['reduce_lr_factor'],
            patience=config['training']['reduce_lr_patience'],
            verbose=1,
            min_lr=1e-7
        ),
        ModelCheckpoint(
            filepath=str(experiment_dir / 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        TensorBoard(
            log_dir=str(experiment_dir / 'logs'),
            histogram_freq=1
        )
    ]
    
    print("\n=== Training fusion model ===")
    start_time = time.time()
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=config['training']['epochs'],
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    model.save(experiment_dir / 'final_model.h5')
    
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(experiment_dir / "training_history.csv", index=False)
    
    plot_training_history(history, experiment_dir / "training_history.png")
    
    print("\n=== Evaluating model on test set ===")
    evaluate_fusion_model(model, audio_test, video_test, y_test, label_mapping, experiment_dir)
    
    return model, history, experiment_dir


def evaluate_fusion_model(model, audio_test, video_test, y_test, label_mapping, output_dir):
    y_pred_proba = model.predict([audio_test, video_test])
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    class_names = [label_mapping[i] for i in range(len(label_mapping))]
    metrics = calculate_metrics(y_test, y_pred, class_names)
    
    print_classification_report(metrics)
    
    cm = metrics['confusion_matrix']
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    import seaborn as sns
    sns.heatmap(
        cm_norm, 
        annot=True, 
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Fusion Model - Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1': float(metrics['f1']),
        'per_class': {k: {kk: float(vv) for kk, vv in v.items()} 
                     for k, v in metrics['per_class'].items()}
    }
    
    with open(output_dir / "test_results.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nEvaluation results saved to {output_dir}")


def plot_training_history(history, output_path):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multimodal fusion model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/fusion_config.json",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    train_fusion_model(args.config)