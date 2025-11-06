import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

def calculate_metrics(y_true, y_pred, class_names=None):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        class_names = [str(i) for i in range(len(np.unique(y_true)))]
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'per_class': {
            'precision': dict(zip(class_names, per_class_precision)),
            'recall': dict(zip(class_names, per_class_recall)),
            'f1': dict(zip(class_names, per_class_f1)),
            'support': dict(zip(class_names, per_class_support))
        },
        'confusion_matrix': cm,
        'class_names': class_names
    }
    
    return metrics

def plot_confusion_matrix(metrics, output_path=None, figsize=(10, 8)):
    cm = metrics['confusion_matrix']
    class_names = metrics['class_names']
    
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=figsize)
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
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    
    plt.show()

def plot_metrics(history, output_path=None, figsize=(12, 5)):
    plt.figure(figsize=figsize)
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    
    plt.show()

def print_classification_report(metrics):
    per_class = metrics['per_class']
    class_names = metrics['class_names']
    
    report = pd.DataFrame({
        'Precision': [per_class['precision'][c] for c in class_names],
        'Recall': [per_class['recall'][c] for c in class_names],
        'F1-Score': [per_class['f1'][c] for c in class_names],
        'Support': [per_class['support'][c] for c in class_names]
    }, index=class_names)
    
    report.loc['weighted avg'] = [
        metrics['precision'],
        metrics['recall'],
        metrics['f1'],
        sum(per_class['support'].values())
    ]
    
    print("\nClassification Report:")
    print("======================")
    print(report.round(3))
    print(f"\nOverall Accuracy: {metrics['accuracy']:.3f}")