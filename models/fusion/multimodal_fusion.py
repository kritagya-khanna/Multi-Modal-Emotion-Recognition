import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.audio.attention import SelfAttention


def build_early_fusion_model(config):
    """
    Early Fusion: Concatenate audio and video features at input level
    
    Simple approach: Combine features early, single model processes both
    """
    audio_shape = tuple(config['audio']['input_shape'])  # (128, 128, 1)
    video_shape = tuple(config['video']['input_shape'])  # (100, 20)
    num_classes = config['model']['num_classes']
    
    # Audio input branch
    audio_input = layers.Input(shape=audio_shape, name='audio_input')
    
    # Audio CNN
    x_audio = layers.Conv2D(32, 3, padding='same', activation='relu')(audio_input)
    x_audio = layers.BatchNormalization()(x_audio)
    x_audio = layers.MaxPooling2D(2)(x_audio)
    x_audio = layers.Dropout(0.3)(x_audio)
    
    x_audio = layers.Conv2D(64, 3, padding='same', activation='relu')(x_audio)
    x_audio = layers.BatchNormalization()(x_audio)
    x_audio = layers.MaxPooling2D(2)(x_audio)
    x_audio = layers.Dropout(0.3)(x_audio)
    
    x_audio = layers.GlobalAveragePooling2D()(x_audio)
    
    # Video input branch
    video_input = layers.Input(shape=video_shape, name='video_input')
    
    # Video processing
    x_video = layers.Reshape((video_shape[0], video_shape[1], 1))(video_input)
    
    x_video = layers.Conv2D(32, 3, padding='same', activation='relu')(x_video)
    x_video = layers.BatchNormalization()(x_video)
    x_video = layers.MaxPooling2D(2)(x_video)
    x_video = layers.Dropout(0.3)(x_video)
    
    x_video = layers.GlobalAveragePooling2D()(x_video)
    
    # Early fusion: Concatenate features
    fused = layers.concatenate([x_audio, x_video], name='early_fusion')
    
    # Shared classification layers
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(fused)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = models.Model(inputs=[audio_input, video_input], outputs=outputs, name='early_fusion')
    return model


def build_late_fusion_model(config):
    """
    Late Fusion: Train separate models, combine predictions
    
    Each modality has its own deep network, combine at decision level
    """
    audio_shape = tuple(config['audio']['input_shape'])
    video_shape = tuple(config['video']['input_shape'])
    num_classes = config['model']['num_classes']
    
    # Audio branch (deeper)
    audio_input = layers.Input(shape=audio_shape, name='audio_input')
    
    x_audio = layers.Conv2D(32, 3, padding='same', activation='relu')(audio_input)
    x_audio = layers.BatchNormalization()(x_audio)
    x_audio = layers.MaxPooling2D(2)(x_audio)
    x_audio = layers.Dropout(0.3)(x_audio)
    
    x_audio = layers.Conv2D(64, 3, padding='same', activation='relu')(x_audio)
    x_audio = layers.BatchNormalization()(x_audio)
    x_audio = layers.MaxPooling2D(2)(x_audio)
    x_audio = layers.Dropout(0.3)(x_audio)
    
    x_audio = layers.Conv2D(128, 3, padding='same', activation='relu')(x_audio)
    x_audio = layers.BatchNormalization()(x_audio)
    x_audio = layers.GlobalAveragePooling2D()(x_audio)
    
    # Audio-specific dense layers
    x_audio = layers.Dense(256, activation='relu')(x_audio)
    x_audio = layers.Dropout(0.5)(x_audio)
    audio_logits = layers.Dense(num_classes, activation='softmax', name='audio_logits')(x_audio)
    
    # Video branch (deeper)
    video_input = layers.Input(shape=video_shape, name='video_input')
    
    x_video = layers.Reshape((video_shape[0], video_shape[1], 1))(video_input)
    
    x_video = layers.Conv2D(32, 3, padding='same', activation='relu')(x_video)
    x_video = layers.BatchNormalization()(x_video)
    x_video = layers.MaxPooling2D(2)(x_video)
    x_video = layers.Dropout(0.3)(x_video)
    
    x_video = layers.Conv2D(64, 3, padding='same', activation='relu')(x_video)
    x_video = layers.BatchNormalization()(x_video)
    x_video = layers.GlobalAveragePooling2D()(x_video)
    
    # Video-specific dense layers
    x_video = layers.Dense(128, activation='relu')(x_video)
    x_video = layers.Dropout(0.5)(x_video)
    video_logits = layers.Dense(num_classes, activation='softmax', name='video_logits')(x_video)
    
    # Late fusion: Average predictions
    fused = layers.Average(name='late_fusion')([audio_logits, video_logits])
    
    model = models.Model(inputs=[audio_input, video_input], outputs=fused, name='late_fusion')
    return model


def build_hybrid_fusion_model(config):
    """
    Hybrid Fusion: Combine features with cross-modal attention
    
    Best approach: Feature-level fusion with attention mechanism
    """
    audio_shape = tuple(config['audio']['input_shape'])
    video_shape = tuple(config['video']['input_shape'])
    num_classes = config['model']['num_classes']
    
    # Audio branch
    audio_input = layers.Input(shape=audio_shape, name='audio_input')
    
    x_audio = layers.Conv2D(32, 3, padding='same', activation='relu')(audio_input)
    x_audio = layers.BatchNormalization()(x_audio)
    x_audio = layers.MaxPooling2D(2)(x_audio)
    x_audio = layers.Dropout(0.3)(x_audio)
    
    x_audio = layers.Conv2D(64, 3, padding='same', activation='relu')(x_audio)
    x_audio = layers.BatchNormalization()(x_audio)
    x_audio = layers.MaxPooling2D(2)(x_audio)
    x_audio = layers.Dropout(0.3)(x_audio)
    
    x_audio = layers.Conv2D(128, 3, padding='same', activation='relu')(x_audio)
    x_audio = layers.BatchNormalization()(x_audio)
    x_audio = layers.GlobalAveragePooling2D()(x_audio)
    audio_features = layers.Dense(256, activation='relu', name='audio_features')(x_audio)
    
    # Video branch
    video_input = layers.Input(shape=video_shape, name='video_input')
    
    x_video = layers.Reshape((video_shape[0], video_shape[1], 1))(video_input)
    
    x_video = layers.Conv2D(32, 3, padding='same', activation='relu')(x_video)
    x_video = layers.BatchNormalization()(x_video)
    x_video = layers.MaxPooling2D(2)(x_video)
    x_video = layers.Dropout(0.3)(x_video)
    
    x_video = layers.Conv2D(64, 3, padding='same', activation='relu')(x_video)
    x_video = layers.BatchNormalization()(x_video)
    x_video = layers.GlobalAveragePooling2D()(x_video)
    video_features = layers.Dense(256, activation='relu', name='video_features')(x_video)
    
    # Cross-modal attention
    # Audio attending to video
    audio_expanded = layers.Reshape((1, 256))(audio_features)
    video_expanded = layers.Reshape((1, 256))(video_features)
    
    # Concatenate for attention
    combined = layers.concatenate([audio_expanded, video_expanded], axis=1)
    
    # Apply self-attention
    attended = SelfAttention(attention_units=128, return_sequences=False)(combined)
    
    # Fusion with residual connection
    fused = layers.concatenate([audio_features, video_features, attended], name='hybrid_fusion')
    
    # Classification layers
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(fused)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.4)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = models.Model(inputs=[audio_input, video_input], outputs=outputs, name='hybrid_fusion')
    return model


def build_fusion_model(config, fusion_type='hybrid'):
    """
    Build fusion model based on type
    
    Args:
        config: Configuration dictionary
        fusion_type: 'early', 'late', or 'hybrid'
    
    Returns:
        Compiled model
    """
    if fusion_type == 'early':
        return build_early_fusion_model(config)
    elif fusion_type == 'late':
        return build_late_fusion_model(config)
    elif fusion_type == 'hybrid':
        return build_hybrid_fusion_model(config)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")


if __name__ == "__main__":
    # Test model building
    import json
    
    config = {
        'audio': {
            'input_shape': [128, 128, 1]
        },
        'video': {
            'input_shape': [100, 20]
        },
        'model': {
            'num_classes': 8
        }
    }
    
    print("Building Early Fusion Model...")
    early_model = build_early_fusion_model(config)
    early_model.summary()
    
    print("\n" + "="*60 + "\n")
    print("Building Late Fusion Model...")
    late_model = build_late_fusion_model(config)
    late_model.summary()
    
    print("\n" + "="*60 + "\n")
    print("Building Hybrid Fusion Model...")
    hybrid_model = build_hybrid_fusion_model(config)
    hybrid_model.summary()