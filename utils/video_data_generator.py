import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
import random


class VideoDataGenerator(tf.keras.utils.Sequence):
    """Generator for video feature sequences with augmentation"""
    
    def __init__(self, X, y, batch_size=32, shuffle=True, augment=False, config=None):
        """
        Initialize generator
        
        Args:
            X: Array of video sequences (N, time_steps, features)
            y: Array of labels (N,)
            batch_size: Batch size
            shuffle: Whether to shuffle data after each epoch
            augment: Whether to apply data augmentation
            config: Configuration dictionary
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.config = config
        self.n_classes = len(np.unique(y))
        self.indices = np.arange(len(self.X))
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, idx):
        """Get batch at index idx"""
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_x = self.X[batch_indices]
        batch_y = self.y[batch_indices]
        
        # Apply augmentation if enabled
        if self.augment and self.config and self.config['augmentation']['enabled']:
            batch_x = self._augment_batch(batch_x)
        
        # Convert labels to one-hot encoding if not already
        if len(batch_y.shape) == 1:
            batch_y = to_categorical(batch_y, num_classes=self.n_classes)
        
        return batch_x, batch_y
    
    def on_epoch_end(self):
        """Shuffle indices after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def _augment_batch(self, batch):
        """Apply augmentation to batch of sequences"""
        augmented_batch = np.copy(batch)
        
        for i in range(len(batch)):
            if random.random() < 0.5:
                augmented_batch[i] = self._augment_sequence(batch[i])
        
        return augmented_batch
    
    def _augment_sequence(self, sequence):
        """Apply augmentation to a single sequence"""
        aug_sequence = np.copy(sequence)
        
        # Time masking
        if self.config['augmentation']['time_masking'] and random.random() < 0.5:
            time_mask_size = int(sequence.shape[0] * 0.1)
            t0 = np.random.randint(0, sequence.shape[0] - time_mask_size)
            aug_sequence[t0:t0 + time_mask_size, :] = 0
        
        # Feature masking
        if self.config['augmentation']['feature_masking'] and random.random() < 0.5:
            feat_mask_size = int(sequence.shape[1] * 0.1)
            f0 = np.random.randint(0, sequence.shape[1] - feat_mask_size)
            aug_sequence[:, f0:f0 + feat_mask_size] = 0
        
        # Gaussian noise
        if self.config['augmentation']['gaussian_noise'] > 0 and random.random() < 0.5:
            noise = np.random.normal(0, self.config['augmentation']['gaussian_noise'], 
                                    aug_sequence.shape)
            aug_sequence = aug_sequence + noise
        
        return aug_sequence


def load_and_prepare_video_data(config):
    """
    Load video features and prepare sequences
    
    Args:
        config: Configuration dictionary
    
    Returns:
        X_train, X_val, X_test: Training, validation, and test sequences
        y_train, y_val, y_test: Training, validation, and test labels
        label_mapping: Dictionary mapping label indices to names
    """
    # Load speech features
    speech_path = Path(config['data']['speech_features_path'])
    
    if not speech_path.exists():
        raise FileNotFoundError(f"Speech features not found at {speech_path}")
    
    speech_df = pd.read_csv(speech_path)
    
    # Optionally load song features
    if config['data']['use_song_data']:
        song_path = Path(config['data']['song_features_path'])
        if song_path.exists():
            song_df = pd.read_csv(song_path)
            features_df = pd.concat([speech_df, song_df], ignore_index=True)
        else:
            print(f"Warning: Song features not found at {song_path}. Using only speech features.")
            features_df = speech_df
    else:
        features_df = speech_df
    
    # Get feature columns
    feature_cols = config['data']['feature_columns']
    available_features = [col for col in feature_cols if col in features_df.columns]
    
    print(f"Using {len(available_features)} features: {available_features}")
    
    # Group by video_id to create sequences
    sequences = []
    labels = []
    
    for video_id, group in features_df.groupby('video_id'):
        # Get features and label
        features = group[available_features].values
        emotion = group['emotion'].iloc[0]
        
        # Create fixed-length sequences
        seq_length = config['data']['sequence_length']
        
        if len(features) >= seq_length:
            # If longer, take first seq_length frames
            sequences.append(features[:seq_length])
            labels.append(emotion)
        elif len(features) > seq_length // 2:
            # If shorter but reasonable, pad with zeros
            padded = np.zeros((seq_length, len(available_features)))
            padded[:len(features)] = features
            sequences.append(padded)
            labels.append(emotion)
        # Skip sequences that are too short
    
    X = np.array(sequences)
    y = np.array(labels)
    
    print(f"Created {len(X)} sequences of shape {X.shape}")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Save label mapping
    label_mapping = dict(zip(range(len(le.classes_)), le.classes_))
    print(f"Label mapping: {label_mapping}")
    
    # Split into train, validation, and test sets
    test_size = config['data']['test_size']
    val_size = config['data']['validation_size']
    random_state = config['data']['random_state']
    
    # First split off test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )
    
    # Then split remaining data into train and validation
    val_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_adjusted,
        random_state=random_state,
        stratify=y_train_val
    )
    
    print(f"Training set shape: {X_train.shape}, {len(np.unique(y_train))} classes")
    print(f"Validation set shape: {X_val.shape}, {len(np.unique(y_val))} classes")
    print(f"Test set shape: {X_test.shape}, {len(np.unique(y_test))} classes")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_mapping