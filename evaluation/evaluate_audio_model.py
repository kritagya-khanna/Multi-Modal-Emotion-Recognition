import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pathlib import Path
import argparse
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.spectrogram_generator import load_and_prepare_data
from utils.metrics import calculate_metrics, plot_confusion_matrix, print_classification_report
from models.audio.attention import SelfAttention

def evaluate_audio_model(model_path, config_path, output_dir=None):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if output_dir is None:
        output_dir = Path(f"evaluation/results/{Path(model_path).stem}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    _, _, X_test, _, _, y_test, label_mapping = load_and_prepare_data(config)
    
    custom_objects = {'SelfAttention': SelfAttention}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    with open(output_dir / "label_mapping.json", 'w') as f:
        json.dump({str(v): k for k, v in label_mapping.items()}, f, indent=4)
    
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    class_names = [label_mapping[i] for i in range(len(label_mapping))]
    metrics = calculate_metrics(y_test, y_pred, class_names)
    
    print_classification_report(metrics)
    
    plot_confusion_matrix(metrics, output_path=output_dir / "confusion_matrix.png")
    
    plot_examples(X_test, y_test, y_pred, class_names, output_dir)
    
    results = {
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'per_class': metrics['per_class']
    }
    
    with open(output_dir / "evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    plot_per_class_metrics(metrics, output_path=output_dir / "per_class_metrics.png")
    
    print(f"\nEvaluation results saved to {output_dir}")
    
    return results

def plot_examples(X, y_true, y_pred, class_names, output_dir, n_examples=5):
    correct_idx = np.where(y_true == y_pred)[0]
    incorrect_idx = np.where(y_true != y_pred)[0]
    
    if len(correct_idx) > 0:
        plt.figure(figsize=(15, 12))
        for i in range(min(n_examples, len(correct_idx))):
            idx = correct_idx[i]
            plt.subplot(n_examples, 2, i*2+1)
            plt.imshow(X[idx, :, :, 0], aspect='auto', cmap='viridis')
            plt.title(f"True: {class_names[y_true[idx]]}")
            plt.colorbar()
            
            plt.subplot(n_examples, 2, i*2+2)
            plt.imshow(X[idx, :, :, 0], aspect='auto', cmap='viridis')
            plt.title(f"Predicted: {class_names[y_pred[idx]]}")
            plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(output_dir / "correct_examples.png")
        plt.close()
    
    if len(incorrect_idx) > 0:
        plt.figure(figsize=(15, 12))
        for i in range(min(n_examples, len(incorrect_idx))):
            idx = incorrect_idx[i]
            plt.subplot(n_examples, 2, i*2+1)
            plt.imshow(X[idx, :, :, 0], aspect='auto', cmap='viridis')
            plt.title(f"True: {class_names[y_true[idx]]}")
            plt.colorbar()
            
            plt.subplot(n_examples, 2, i*2+2)
            plt.imshow(X[idx, :, :, 0], aspect='auto', cmap='viridis')
            plt.title(f"Predicted: {class_names[y_pred[idx]]}")
            plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(output_dir / "incorrect_examples.png")
        plt.close()

def plot_per_class_metrics(metrics, output_path=None, figsize=(12, 6)):
    per_class = metrics['per_class']
    class_names = metrics['class_names']
    
    df = pd.DataFrame({
        'Precision': [per_class['precision'][c] for c in class_names],
        'Recall': [per_class['recall'][c] for c in class_names],
        'F1-Score': [per_class['f1'][c] for c in class_names]
    }, index=class_names)
    
    plt.figure(figsize=figsize)
    df.plot(kind='bar', figsize=figsize)
    plt.title('Per-Class Metrics')
    plt.xlabel('Emotion')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend(loc='lower right')
    
    if output_path:
        plt.savefig(output_path)
    
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate audio emotion recognition model")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model")
    parser.add_argument("--config", type=str, default="config/audio_config.json", help="Path to configuration file")
    parser.add_argument("--output", type=str, default=None, help="Directory to save evaluation results")
    
    args = parser.parse_args()
    evaluate_audio_model(args.model, args.config, args.output)
