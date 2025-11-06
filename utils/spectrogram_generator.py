import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def feature_to_spectrogram(features, n_time_steps=128, n_features=128):

    #Convert tabular features to a 2D representation that can be used as a spectrogram-like input
    
    # Args:
    #    features: 1D array of audio features
    #    n_time_steps: Number of time steps in the output
    #    n_features: Number of features in the output
    

    # Determine the feature dimension
    feature_dim = len(features)
    
    # If we have more features than needed, truncate
    if feature_dim >= n_features * n_time_steps:
        # Truncate and reshape
        features = features[:n_features * n_time_steps]
        return features.reshape(n_time_steps, n_features)
    
    # If we have fewer features than needed, pad with zeros
    padding_needed = n_features * n_time_steps - feature_dim
    padded_features = np.pad(features, (0, padding_needed), 'constant')
    return padded_features.reshape(n_time_steps, n_features)

def generate_spectrograms_from_features(features_df, label_column='emotion', drop_columns=None, config=None):
    # Convert tabular audio features to spectrogram-like 2D arrays suitable for CNN
    
    # Args:
    #    features_df: DataFrame containing audio features
    #    label_column: Column name for the label
    #    drop_columns: Columns to drop before conversion (metadata, etc.)
    #    config: Configuration dictionary with parameters

    if drop_columns is None:
        # Default columns to drop - metadata and label columns
        drop_columns = [
            'emotion', 'actor', 'actor_id', 'gender', 'intensity', 
            'filename', 'file_path', 'modality', 'voice_channel', 
            'statement', 'repetition'
        ]
    
    # Remove columns that don't exist in the DataFrame
    drop_columns = [col for col in drop_columns if col in features_df.columns]
    
    # Get labels
    y = features_df[label_column].values
    
    # Drop non-feature columns
    feature_df = features_df.drop(columns=drop_columns, errors='ignore')
    
    # Convert each row of features to a spectrogram-like representation
    n_time_steps = config['model']['input_shape'][0] if config else 128
    n_features = config['model']['input_shape'][1] if config else 128
    
    X = np.zeros((len(feature_df), n_time_steps, n_features))
    
    print("Converting features to spectrogram-like representations...")
    for i, row in tqdm(enumerate(feature_df.values), total=len(feature_df)):
        X[i] = feature_to_spectrogram(row, n_time_steps, n_features)
    
    # Add channel dimension for CNN input (n_samples, height, width, channels)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    
    return X, y


def load_and_prepare_data(config):
    """
    Load real mel-spectrograms and prepare for training
    
    Args:
        config: Configuration dictionary
    
    Returns:
        X_train, X_val, X_test: Training, validation, and test spectrograms
        y_train, y_val, y_test: Training, validation, and test labels
        label_mapping: Dictionary mapping label indices to names
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    
    print("Loading real mel-spectrograms...")
    
    # Load speech spectrograms
    speech_spec_path = Path("processed/audio_spectrograms/speech")
    X_speech = np.load(speech_spec_path / "spectrograms.npy")
    y_speech = np.load(speech_spec_path / "labels.npy")
    
    print(f"Loaded speech spectrograms: {X_speech.shape}")
    
    # Optionally load song spectrograms
    if config['data']['use_song_data']:
        song_spec_path = Path("processed/audio_spectrograms/song")
        if song_spec_path.exists() and (song_spec_path / "spectrograms.npy").exists():
            X_song = np.load(song_spec_path / "spectrograms.npy")
            y_song = np.load(song_spec_path / "labels.npy")
            print(f"Loaded song spectrograms: {X_song.shape}")
            
            # Combine speech and song
            X = np.concatenate([X_speech, X_song], axis=0)
            y = np.concatenate([y_speech, y_song], axis=0)
            print(f"Combined spectrograms: {X.shape}")
        else:
            print(f"Warning: Song spectrograms not found. Using only speech spectrograms.")
            X = X_speech
            y = y_speech
    else:
        X = X_speech
        y = y_speech
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Create label mapping (index -> emotion name)
    label_mapping = dict(zip(range(len(le.classes_)), le.classes_))
    print(f"Label mapping: {label_mapping}")
    print(f"Class distribution: {dict(zip(*np.unique(y_encoded, return_counts=True)))}")
    
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
    
    print(f"\nData split:")
    print(f"  Training set: {X_train.shape}, {len(np.unique(y_train))} classes")
    print(f"  Validation set: {X_val.shape}, {len(np.unique(y_val))} classes")
    print(f"  Test set: {X_test.shape}, {len(np.unique(y_test))} classes")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_mapping

if __name__ == "__main__":
    # Test the spectrogram generator
    import json
    
    with open('config/audio_config.json', 'r') as f:
        config = json.load(f)
    
    X_train, X_val, X_test, y_train, y_val, y_test, label_mapping = load_and_prepare_data(config)
    
    # Visualize a few examples
    plt.figure(figsize=(12, 8))
    for i in range(min(8, X_train.shape[0])):
        plt.subplot(2, 4, i+1)
        plt.imshow(X_train[i, :, :, 0], aspect='auto', cmap='viridis')
        plt.title(f"Class: {label_mapping[y_train[i]]}")
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('spectrogram_examples.png')
    plt.show()