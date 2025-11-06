import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import json
import os
from pathlib import Path

# Import attention mechanisms
from models.audio.attention import SelfAttention, BahdanauAttention

def build_cnn_lstm_attention(config):
    """
    Build CNN-LSTM model with attention for audio emotion recognition
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Compiled model
    """
    # Get parameters from config
    input_shape = tuple(config['model']['input_shape'])
    cnn_filters = config['model']['cnn_filters']
    kernel_size = config['model']['cnn_kernel_size']
    pool_size = config['model']['cnn_pool_size']
    lstm_units = config['model']['lstm_units']
    lstm_layers = config['model']['lstm_layers']
    dense_units = config['model']['dense_units']
    dropout_rate = config['model']['dropout_rate']
    attention_units = config['model']['attention_units']
    num_classes = config['model']['num_classes']
    
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # CNN layers
    x = inputs
    for i, filters in enumerate(cnn_filters):
        # Conv2D + BatchNorm + ReLU
        x = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            kernel_regularizer=regularizers.l2(1e-5),
            name=f'conv2d_{i+1}'
        )(x)
        x = layers.BatchNormalization(name=f'bn_{i+1}')(x)
        x = layers.Activation('relu', name=f'relu_{i+1}')(x)
        
        # Apply pooling to reduce dimensions except in the last layer
        # to preserve temporal information for LSTM
        if i < len(cnn_filters) - 1:
            x = layers.MaxPooling2D(
                pool_size=pool_size,
                name=f'pool_{i+1}'
            )(x)
        
        # Add dropout with increasing rate for deeper layers
        dropout_factor = (i + 1) / len(cnn_filters)
        x = layers.Dropout(
            rate=dropout_rate * dropout_factor,
            name=f'dropout_{i+1}'
        )(x)
    
    # Reshape for LSTM (preserve time dimension)
    # Assuming input is (batch, time, frequency, channel)
    # Need to reshape to (batch, time, frequency * channel)
    shape = x.get_shape().as_list()
    x = layers.Reshape(
        (shape[1], shape[2] * shape[3]),
        name='reshape_for_lstm'
    )(x)
    
    # Bidirectional LSTM layers
    for i in range(lstm_layers):
        return_sequences = True  # Always return sequences for attention
        x = layers.Bidirectional(
            layers.LSTM(
                units=lstm_units,
                return_sequences=return_sequences,
                dropout=0.3,
                recurrent_dropout=0.3,
                kernel_regularizer=regularizers.l2(1e-5),
                name=f'lstm_{i+1}'
            ),
            name=f'bidirectional_{i+1}'
        )(x)
    
    # Self-attention mechanism
    x = SelfAttention(
        attention_units=attention_units,
        return_sequences=False,
        name='self_attention'
    )(x)
    
    # Dense layers
    for i, units in enumerate(dense_units):
        x = layers.Dense(
            units=units,
            activation='relu',
            kernel_regularizer=regularizers.l2(1e-5),
            name=f'dense_{i+1}'
        )(x)
        x = layers.BatchNormalization(name=f'dense_bn_{i+1}')(x)
        x = layers.Dropout(rate=dropout_rate, name=f'dense_dropout_{i+1}')(x)
    
    # Output layer
    outputs = layers.Dense(
        units=num_classes,
        activation='softmax',
        name='output'
    )(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name='cnn_lstm_attention')
    
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