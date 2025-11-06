import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import os
from pathlib import Path
import json
import argparse
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.spectrogram_generator import feature_to_spectrogram
from models.audio.attention import SelfAttention

def create_attention_model(model_path):
    custom_objects = {
        'SelfAttention': SelfAttention
    }
    
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    attention_layer_name = None
    for layer in model.layers:
        if isinstance(layer, SelfAttention):
            attention_layer_name = layer.name
            break
    
    if attention_layer_name is None:
        raise ValueError("Could not find SelfAttention layer in model")
    
    attention_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(attention_layer_name).output, model.output]
    )
    
    return attention_model

def visualize_attention(model_path, feature_file, config_path, output_dir=None):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if output_dir is None:
        output_dir = Path(f"evaluation/attention_visualizations/{Path(model_path).stem}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import pandas as pd
    features_df = pd.read_csv(feature_file)
    
    model_dir = Path(model_path).parent
    with open(model_dir / "label_mapping.json", 'r') as f:
        label_mapping = json.load(f)
    
    attention_model = create_attention_model(model_path)
    
    n_examples = min(8, len(features_df))
    examples_idx = np.random.choice(len(features_df), n_examples, replace=False)
    
    plt.figure(figsize=(15, 12))
    
    for i, idx in enumerate(examples_idx):
        row = features_df.iloc[idx]
        emotion = row['emotion']
        
        drop_cols = ['emotion', 'actor_id', 'gender', 'intensity', 'filename']
        feature_cols = [col for col in features_df.columns if col not in drop_cols]
        feature_vector = row[feature_cols].values
        
        n_time_steps = config['model']['input_shape'][0]
        n_features = config['model']['input_shape'][1]
        spectrogram = feature_to_spectrogram(feature_vector, n_time_steps, n_features)
        
        spectrogram = spectrogram.reshape(1, n_time_steps, n_features, 1)
        
        attention_output, prediction = attention_model.predict(spectrogram)
        
        pred_class = np.argmax(prediction[0])
        pred_emotion = label_mapping[str(pred_class)]
        
        plt.subplot(n_examples, 2, i*2+1)
        plt.imshow(spectrogram[0, :, :, 0], aspect='auto', cmap='viridis')
        plt.title(f"True: {emotion}, Pred: {pred_emotion}")
        plt.colorbar()
        
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
