import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import json
import os
from pathlib import Path

# Import attention mechanisms
from models.audio.attention import SelfAttention, BahdanauAttention

def build_cnn_lstm_attention(config):
    """
    Very simple CNN for small datasets
    Only ~200K parameters instead of 5M
    """
    input_shape = tuple(config['model']['input_shape'])
    num_classes = config['model']['num_classes']
    
    inputs = layers.Input(shape=input_shape)
    
    # Simple CNN - only 3 layers
    x = layers.Conv2D(32, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    
    # Simple dense layers
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='simple_cnn')
    return model


def build_audio_model(config_path=None):
    """
    Build audio model from config or with default parameters
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Compiled model
    """
    # Load configuration if provided
    if config_path is not None:
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            "model": {
                "name": "cnn_lstm_attention",
                "input_shape": [128, 128, 1],
                "cnn_filters": [32, 64, 128, 256],
                "cnn_kernel_size": 3,
                "cnn_pool_size": 2,
                "lstm_units": 128,
                "lstm_layers": 2,
                "dense_units": [256, 128],
                "dropout_rate": 0.5,
                "attention_units": 64,
                "num_classes": 8
            }
        }
    
    model = build_cnn_lstm_attention(config)
    
    return model


if __name__ == "__main__":
    # Test model building
    import numpy as np
    from tensorflow.keras.utils import plot_model
    
    # Load config
    with open('config/audio_config.json', 'r') as f:
        config = json.load(f)
    
    # Build model
    model = build_cnn_lstm_attention(config)
    
    # Print model summary
    model.summary()
    
    # Visualize model architecture
    plot_model(model, to_file='audio_model_architecture.png', show_shapes=True, show_layer_names=True)
    
    # Test with random input
    input_shape = config['model']['input_shape']
    batch_size = 32
    test_input = np.random.random((batch_size, *input_shape))
    test_output = model.predict(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")