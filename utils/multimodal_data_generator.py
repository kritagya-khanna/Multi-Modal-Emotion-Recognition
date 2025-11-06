import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
import random
import os


class MultimodalDataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, audio_data, video_data, labels, batch_size=32, 
                 shuffle=True, augment=False, config=None):
        self.audio_data = audio_data
        self.video_data = video_data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.config = config
        self.n_classes = len(np.unique(labels))
        self.indices = np.arange(len(self.labels))
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.labels) / self.batch_size))
        
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_audio = self.audio_data[batch_indices]
        batch_video = self.video_data[batch_indices]
        batch_labels = self.labels[batch_indices]
        
        if self.augment and self.config and self.config['augmentation']['enabled']:
            if self.config['augmentation']['audio_augment']:
                batch_audio = self._augment_audio_batch(batch_audio)
            if self.config['augmentation']['video_augment']:
                batch_video = self._augment_video_batch(batch_video)
        
        if len(batch_labels.shape) == 1:
            batch_labels = to_categorical(batch_labels, num_classes=self.n_classes)
        
        return (batch_audio, batch_video), batch_labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def _augment_audio_batch(self, batch):
        augmented_batch = np.copy(batch)
        
        for i in range(len(batch)):
            if random.random() < 0.5:
                augmented_batch[i] = self._augment_audio(batch[i])
        
        return augmented_batch
    
    def _augment_audio(self, spectrogram):
        aug_spec = np.copy(spectrogram)
        
        if random.random() < 0.5:
            time_mask_size = int(aug_spec.shape[0] * 0.1)
            t0 = np.random.randint(0, max(1, aug_spec.shape[0] - time_mask_size))
            aug_spec[t0:t0 + time_mask_size, :, :] = 0
        
        if random.random() < 0.5:
            freq_mask_size = int(aug_spec.shape[1] * 0.1)
            f0 = np.random.randint(0, max(1, aug_spec.shape[1] - freq_mask_size))
            aug_spec[:, f0:f0 + freq_mask_size, :] = 0
        
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.01, aug_spec.shape)
            aug_spec = aug_spec + noise
        
        return aug_spec
    
    def _augment_video_batch(self, batch):
        augmented_batch = np.copy(batch)
        
        for i in range(len(batch)):
            if random.random() < 0.5:
                augmented_batch[i] = self._augment_video(batch[i])
        
        return augmented_batch
    
    def _augment_video(self, sequence):
        aug_seq = np.copy(sequence)
        
        if random.random() < 0.5:
            time_mask_size = int(sequence.shape[0] * 0.1)
            t0 = np.random.randint(0, max(1, sequence.shape[0] - time_mask_size))
            aug_seq[t0:t0 + time_mask_size, :] = 0
        
        if random.random() < 0.5:
            feat_mask_size = int(sequence.shape[1] * 0.1)
            f0 = np.random.randint(0, max(1, sequence.shape[1] - feat_mask_size))
            aug_seq[:, f0:f0 + feat_mask_size] = 0
        
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.01, aug_seq.shape)
            aug_seq = aug_seq + noise
        
        return aug_seq


def load_and_prepare_multimodal_data(config):
    print("\n=== Loading Multimodal Data ===")
    
    print("Loading audio spectrograms...")
    audio_data, audio_labels = load_audio_spectrograms(config)
    
    print("Loading video features...")
    video_data, video_labels = load_video_features(config)
    
    print("\nAligning audio and video samples...")
    audio_data, video_data, labels = align_multimodal_data(
        audio_data, audio_labels, video_data, video_labels
    )
    
    print(f"Total aligned samples: {len(labels)}")
    print(f"Audio shape: {audio_data.shape}")
    print(f"Video shape: {video_data.shape}")
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    label_mapping = dict(zip(range(len(le.classes_)), le.classes_))
    print(f"Label mapping: {label_mapping}")
    
    test_size = config['data']['test_size']
    val_size = config['data']['validation_size']
    random_state = config['data']['random_state']
    
    audio_train_val, audio_test, video_train_val, video_test, y_train_val, y_test = train_test_split(
        audio_data, video_data, y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )
    
    val_adjusted = val_size / (1 - test_size)
    audio_train, audio_val, video_train, video_val, y_train, y_val = train_test_split(
        audio_train_val, video_train_val, y_train_val,
        test_size=val_adjusted,
        random_state=random_state,
        stratify=y_train_val
    )
    
    print(f"\nTrain: {len(y_train)} samples")
    print(f"Val: {len(y_val)} samples")
    print(f"Test: {len(y_test)} samples")
    
    return (audio_train, audio_val, audio_test,
            video_train, video_val, video_test,
            y_train, y_val, y_test, label_mapping)


def load_audio_spectrograms(config):
    speech_dir = Path(config['data']['audio_speech_path'])
    
    all_spectrograms = []
    all_labels = []
    
    if speech_dir.exists():
        speech_spec_file = speech_dir / 'spectrograms.npy'
        speech_labels_file = speech_dir / 'labels.npy'
        
        if speech_spec_file.exists() and speech_labels_file.exists():
            speech_specs = np.load(speech_spec_file)
            speech_labels = np.load(speech_labels_file)
            all_spectrograms.append(speech_specs)
            all_labels.append(speech_labels)
            print(f"Loaded {len(speech_specs)} speech spectrograms")
        else:
            raise FileNotFoundError(f"Spectrogram files not found in {speech_dir}")
    
    if config['data'].get('use_song_data', False):
        song_dir = Path(config['data']['audio_song_path'])
        if song_dir.exists():
            song_spec_file = song_dir / 'spectrograms.npy'
            song_labels_file = song_dir / 'labels.npy'
            
            if song_spec_file.exists() and song_labels_file.exists():
                song_specs = np.load(song_spec_file)
                song_labels = np.load(song_labels_file)
                all_spectrograms.append(song_specs)
                all_labels.append(song_labels)
                print(f"Loaded {len(song_specs)} song spectrograms")
    
    spectrograms = np.concatenate(all_spectrograms, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    print(f"Total audio spectrograms: {len(spectrograms)}, shape: {spectrograms.shape}")
    return spectrograms, labels

def load_video_features(config):
    speech_path = Path(config['data']['video_speech_path'])
    
    speech_df = pd.read_csv(speech_path)
    
    if config['data'].get('use_song_data', False):
        song_path = Path(config['data']['video_song_path'])
        if song_path.exists():
            song_df = pd.read_csv(song_path)
            features_df = pd.concat([speech_df, song_df], ignore_index=True)
        else:
            features_df = speech_df
    else:
        features_df = speech_df
    
    feature_cols = [col for col in features_df.columns 
                   if col not in ['emotion', 'actor_id', 'gender', 'intensity', 'video_id']]
    
    sequences = []
    labels = []
    
    seq_length = config['data']['video_sequence_length']
    
    for video_id, group in features_df.groupby('video_id'):
        features = group[feature_cols].values
        emotion = group['emotion'].iloc[0]
        
        if len(features) >= seq_length:
            sequences.append(features[:seq_length])
            labels.append(emotion)
        elif len(features) > seq_length // 2:
            padded = np.zeros((seq_length, len(feature_cols)))
            padded[:len(features)] = features
            sequences.append(padded)
            labels.append(emotion)
    
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    print(f"Loaded {len(sequences)} video sequences")
    return sequences, labels


def align_multimodal_data(audio_data, audio_labels, video_data, video_labels):
    common_indices = []
    
    min_len = min(len(audio_labels), len(video_labels))
    
    for i in range(min_len):
        if audio_labels[i] == video_labels[i]:
            common_indices.append(i)
    
    if len(common_indices) == 0:
        raise ValueError("No matching samples found between audio and video data!")
    
    audio_aligned = audio_data[common_indices]
    video_aligned = video_data[common_indices]
    labels_aligned = audio_labels[common_indices]
    
    print(f"Aligned {len(common_indices)} samples out of {min_len} total")
    
    return audio_aligned, video_aligned, labels_aligned