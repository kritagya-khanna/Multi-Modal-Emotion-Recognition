import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import random

class AudioDataGenerator(tf.keras.utils.Sequence):
    """Generator that provides batches of spectrogram data with augmentation"""
    
    def __init__(self, X, y, batch_size=32, shuffle=True, augment=False, config=None):
        """
        Initialize generator
        
        Args:
            X: Array of spectrogram data (N, height, width, channels)
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
        # Get batch indices
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Get batch data
        batch_x = self.X[batch_indices]
        batch_y = self.y[batch_indices]
        
        # Apply augmentation if enabled
        if self.augment and self.config and self.config['augmentation']['enabled']:
            batch_x = self._augment_batch(batch_x)
            
            # Apply mixup if enabled
            if self.config['augmentation']['mixup_alpha'] > 0:
                batch_x, batch_y = self._mixup(batch_x, batch_y)
        
        # Convert labels to one-hot encoding if not already
        if len(batch_y.shape) == 1:
            batch_y = to_categorical(batch_y, num_classes=self.n_classes)
        
        return batch_x, batch_y
    
    def on_epoch_end(self):
        """Shuffle indices after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def _augment_batch(self, batch):
        """Apply augmentation to batch of spectrograms"""
        augmented_batch = np.copy(batch)
        
        for i in range(len(batch)):
            # Apply augmentation with random chance
            if random.random() < 0.5:
                augmented_batch[i] = self._augment_spectrogram(batch[i])
        
        return augmented_batch
    
    def _augment_spectrogram(self, spectrogram):
        """Apply augmentation to a single spectrogram"""
        # Get a copy to avoid modifying the original
        aug_spectrogram = np.copy(spectrogram)
        
        # SpecAugment: time masking
        if self.config['augmentation']['spec_augment'] and random.random() < 0.5:
            time_mask_size = int(spectrogram.shape[0] * 0.1)  # 10% of time dimension
            t0 = np.random.randint(0, spectrogram.shape[0] - time_mask_size)
            aug_spectrogram[t0:t0 + time_mask_size, :, :] = 0
        
        # SpecAugment: frequency masking
        if self.config['augmentation']['spec_augment'] and random.random() < 0.5:
            freq_mask_size = int(spectrogram.shape[1] * 0.1)  # 10% of frequency dimension
            f0 = np.random.randint(0, spectrogram.shape[1] - freq_mask_size)
            aug_spectrogram[:, f0:f0 + freq_mask_size, :] = 0
        
        return aug_spectrogram
    
    def _mixup(self, x, y):
        """
        Apply mixup augmentation
        
        Args:
            x: Batch of images (batch_size, height, width, channels)
            y: Batch of labels (batch_size, num_classes) in one-hot encoding
        
        Returns:
            Mixed up batch of images and labels
        """
        # Convert labels to one-hot if not already
        if len(y.shape) == 1:
            y = to_categorical(y, num_classes=self.n_classes)
        
        # Sample beta distribution
        alpha = self.config['augmentation']['mixup_alpha']
        lam = np.random.beta(alpha, alpha)
        
        # Get batch size
        batch_size = x.shape[0]
        
        # Generate shuffled indices
        indices = np.random.permutation(batch_size)
        
        # Mix data
        mixed_x = lam * x + (1 - lam) * x[indices]
        mixed_y = lam * y + (1 - lam) * y[indices]
        
        return mixed_x, mixed_y