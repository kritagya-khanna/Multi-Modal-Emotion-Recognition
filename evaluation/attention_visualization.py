import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import os
from pathlib import Path
import json
import argparse
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your modules
from utils.spectrogram_generator import feature_to_spectrogram
from models.audio.attention import SelfAttention

def create_attention_model(model_path):
    """
    Create model that outputs attention weights
    
    Args:
        model_path: Path to saved model
    
    Returns:
        attention_model: Model that outputs attention weights
    """
    # Custom objects for loading model
    custom_objects = {
        'SelfAttention': SelfAttention
    }
    
    # Load model
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    # Find self-attention layer
    attention_layer_name = None
    for layer in model.layers:
        if isinstance(layer, SelfAttention):
            attention_layer_name = layer.name
            break
    
    if attention_layer_name is None:
        raise ValueError("Could not find SelfAttention layer in model")
    
    # Create model that outputs intermediate activations
    attention_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(attention_layer_name).output, model.output]
    )
    
    return attention_model

def visualize_attention(model_path, feature_file, config_path, output_dir=None):
    """
    Visualize attention weights for audio features
    
    Args:
        model_path: Path to saved model
        feature_file: Path to feature file
        config_path: Path to configuration file
        output_dir: Directory to save visualizations
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create output directory if not provided
    if output_dir is None:
        output_dir = Path(f"evaluation/attention_visualizations/{Path(model_path).stem}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load features
    import pandas as pd
    features_df = pd.read_csv(feature_file)
    
    # Get label mapping
    model_dir = Path(model_path).parent
    with open(model_dir / "label_mapping.json", 'r') as f:
        label_mapping = json.load(f)
    
    # Load model for attention visualization
    attention_model = create_attention_model(model_path)
    
    # Select a few examples for visualization
    n_examples = min(8, len(features_df))
    examples_idx = np.random.choice(len(features_df), n_examples, replace=False)
    
    # Visualize attention for each example
    plt.figure(figsize=(15, 12))
    
    for i, idx in enumerate(examples_idx):
        # Get features and label
        row = features_df.iloc[idx]
        emotion = row['emotion']
        
        # Prepare feature vector
        drop_cols = ['emotion', 'actor_id', 'gender', 'intensity', 'filename']
        feature_cols = [col for col in features_df.columns if col not in drop_cols]
        feature_vector = row[feature_cols].values
        
        # Convert to spectrogram
        n_time_steps = config['model']['input_shape'][0]
        n_features = config['model']['input_shape'][1]
        spectrogram = feature_to_spectrogram(feature_vector, n_time_steps, n_features)
        
        # Add batch and channel dimensions
        spectrogram = spectrogram.reshape(1, n_time_steps, n_features, 1)
        
        # Get attention weights and prediction
        attention_output, prediction = attention_model.predict(spectrogram)
        
        # Get predicted class
        pred_class = np.argmax(prediction[0])
        pred_emotion = label_mapping[str(pred_class)]
        
        # Plot spectrogram and attention
        plt.subplot(n_examples, 2, i*2+1)
        plt.imshow(spectrogram[0, :, :, 0], aspect='auto', cmap='viridis')
        plt.title(f"True: {emotion}, Pred: {pred_emotion}")
        plt.colorbar()
        
        # Plot attention weights
        plt.subplot(n_examples, 2, i*2+2)
        attention_weights = np.mean(attention_output, axis=2)
        plt.imshow(attention_weights, aspect='auto', cmap='hot')
        plt.title("Attention Heatmap")
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(output_dir / "attention_visualizations.png")
    plt.show()
    
    print(f"Attention visualizations saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize attention weights")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model")
    parser.add_argument("--features", type=str, required=True, help="Path to feature file")
    parser.add_argument("--config", type=str, default="config/audio_config.json", help="Path to configuration file")
    parser.add_argument("--output", type=str, default=None, help="Directory to save visualizations")
    
    args = parser.parse_args()
    visualize_attention(args.model, args.features, args.config, args.output)