import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import json
import os
from pathlib import Path
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.audio.attention import SelfAttention

def build_video_cnn_lstm_attention(config):
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
    
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Reshape((input_shape[0], input_shape[1], 1))(inputs)
    
    for i, filters in enumerate(cnn_filters):
        x = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            kernel_regularizer=regularizers.l2(1e-5),
            name=f'conv2d_{i+1}'
        )(x)
        x = layers.BatchNormalization(name=f'bn_{i+1}')(x)
        x = layers.Activation('relu', name=f'relu_{i+1}')(x)
        
        if i < len(cnn_filters) - 1:
            x = layers.MaxPooling2D(
                pool_size=pool_size,
                name=f'pool_{i+1}'
            )(x)
        
        dropout_factor = (i + 1) / len(cnn_filters)
        x = layers.Dropout(
            rate=dropout_rate * dropout_factor,
            name=f'dropout_{i+1}'
        )(x)
    
    shape = x.shape
    x = layers.Reshape(
        (shape[1], shape[2] * shape[3]),
        name='reshape_for_lstm'
    )(x)
    
    for i in range(lstm_layers):
        return_sequences = True
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
    
    x = SelfAttention(
        attention_units=attention_units,
        return_sequences=False,
        name='self_attention'
    )(x)
    
    for i, units in enumerate(dense_units):
        x = layers.Dense(
            units=units,
            activation='relu',
            kernel_regularizer=regularizers.l2(1e-5),
            name=f'dense_{i+1}'
        )(x)
        x = layers.BatchNormalization(name=f'dense_bn_{i+1}')(x)
        x = layers.Dropout(rate=dropout_rate, name=f'dense_dropout_{i+1}')(x)
    
    outputs = layers.Dense(
        units=num_classes,
        activation='softmax',
        name='output'
    )(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='video_cnn_lstm_attention')
    
    return model


def build_video_model(config_path=None):
    if config_path is not None:
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {
            "model": {
                "name": "video_cnn_lstm_attention",
                "input_shape": [100, 20],
                "cnn_filters": [32, 64, 128],
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
    
    model = build_video_cnn_lstm_attention(config)
    
    return model


if __name__ == "__main__":
    import numpy as np
    from tensorflow.keras.utils import plot_model
    
    model = build_video_model()
    
    model.summary()
    
    batch_size = 32
    test_input = np.random.random((batch_size, 100, 20))
    test_output = model.predict(test_input)
    
    print(f"\nInput shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")
