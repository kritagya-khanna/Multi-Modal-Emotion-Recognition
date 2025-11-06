import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import random

class AudioDataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, X, y, batch_size=32, shuffle=True, augment=False, config=None):
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
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_x = self.X[batch_indices]
        batch_y = self.y[batch_indices]
        
        if self.augment and self.config and self.config['augmentation']['enabled']:
            batch_x = self._augment_batch(batch_x)
            
            if self.config['augmentation']['mixup_alpha'] > 0:
                batch_x, batch_y = self._mixup(batch_x, batch_y)
        
        if len(batch_y.shape) == 1:
            batch_y = to_categorical(batch_y, num_classes=self.n_classes)
        
        return batch_x, batch_y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def _augment_batch(self, batch):
        augmented_batch = np.copy(batch)
        
        for i in range(len(batch)):
            if random.random() < 0.5:
                augmented_batch[i] = self._augment_spectrogram(batch[i])
        
        return augmented_batch
    
    def _augment_spectrogram(self, spectrogram):
        aug_spectrogram = np.copy(spectrogram)
        
        if self.config['augmentation']['spec_augment'] and random.random() < 0.5:
            time_mask_size = int(spectrogram.shape[0] * 0.1)
            t0 = np.random.randint(0, spectrogram.shape[0] - time_mask_size)
            aug_spectrogram[t0:t0 + time_mask_size, :, :] = 0
        
        if self.config['augmentation']['spec_augment'] and random.random() < 0.5:
            freq_mask_size = int(spectrogram.shape[1] * 0.1)
            f0 = np.random.randint(0, spectrogram.shape[1] - freq_mask_size)
            aug_spectrogram[:, f0:f0 + freq_mask_size, :] = 0
        
        return aug_spectrogram
    
    def _mixup(self, x, y):
        if len(y.shape) == 1:
            y = to_categorical(y, num_classes=self.n_classes)
        
        alpha = self.config['augmentation']['mixup_alpha']
        lam = np.random.beta(alpha, alpha)
        
        batch_size = x.shape[0]
        
        indices = np.random.permutation(batch_size)
        
        mixed_x = lam * x + (1 - lam) * x[indices]
        mixed_y = lam * y + (1 - lam) * y[indices]
        
        return mixed_x, mixed_y
