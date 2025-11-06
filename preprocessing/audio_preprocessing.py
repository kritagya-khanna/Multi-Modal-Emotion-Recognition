import os
import numpy as np
import pandas as pd
import librosa 
import soundfile as sf 
import matplotlib.pyplot as plt 
from scipy.stats import skew, kurtosis
from pathlib import Path
from tqdm import tqdm
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')

def load_audio_file(file_path):
    try:
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        return audio_data, sample_rate
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None, None
    
def remove_silence(audio_data, sample_rate, threshold = 0.01):
    trimmed_audio, _ = librosa.effects.trim(audio_data, top_db = 20, frame_length=512, hop_length=64)
    return trimmed_audio

def normalize_volume(audio_data):
    return librosa.util.normalize(audio_data)

def extract_features(audio_data ,sample_rate, n_mfcc=13, n_mels=128, fmax=8000):
    features = {}

    hop_length = int(sample_rate * 0.010)
    win_length = int(sample_rate * 0.025)

    zrc = librosa.feature.zero_crossing_rate(audio_data, hop_length = hop_length)[0]
    features["zrc_mean"] = np.mean(zrc)
    features['zrc_std'] = np.std(zrc)
    features['zrc-max'] = np.max(zrc)

    rms  = librosa.feature.rms(y = audio_data, hop_length = hop_length)[0]
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)
    features['rms_max'] = np.max(rms)

    mfcc_coeff = librosa.feature.mfcc(
        y = audio_data,
        sr = sample_rate,
        n_mfcc = n_mfcc,
        hop_length = hop_length,
        win_length = win_length,
        n_mels = n_mels,
        fmax = fmax,
    )

    for i in range(n_mfcc):
        features[f'mfcc{i+1}_mean'] = np.mean(mfcc_coeff[i])
        features[f'mfcc{i+1}_std'] = np.std(mfcc_coeff[i])
        features[f'mfcc{i+1}_skew'] = skew(mfcc_coeff[i])
        features[f'mfcc{i+1}_kurt'] = kurtosis(mfcc_coeff[i])

    mfcc_delta = librosa.feature.delta(mfcc_coeff)
    mfcc_delta_delta = librosa.feature.delta(mfcc_coeff, order=2)

    for i in range(n_mfcc):
        features[f'mfcc{i+1}_delta_mean'] = np.mean(mfcc_delta[i])
        features[f'mfcc{i+1}_delta_std'] = np.std(mfcc_delta[i])
        features[f'mfcc{i+1}_delta2_mean'] = np.mean(mfcc_delta_delta[i])
        features[f'mfcc{i+1}_delta2_std'] = np.std(mfcc_delta_delta[i])

    cent = librosa.feature.spectral_centroid(y = audio_data, sr = sample_rate, hop_length = hop_length)[0]
    features['spectral_centroid_mean'] = np.mean(cent)
    features['spectral_centroid_std'] = np.std(cent)

    bandwidth = librosa.feature.spectral_bandwidth(y= audio_data, sr = sample_rate, hop_length = hop_length)[0]
    features['spectral_bandwidth_mean'] = np.mean(bandwidth)
    features['spectral_bandwidth_std'] = np.std(bandwidth)

    contrast = librosa.feature.spectral_contrast(y = audio_data, sr = sample_rate, hop_length = hop_length)[0]
    features['spectral_contrast_mean'] = np.mean(contrast)
    features['spectral_contrast_std'] = np.std(contrast)

    rolloff = librosa.feature.spectral_rolloff(y = audio_data, sr = sample_rate, hop_length = hop_length)[0]
    features['rolloff_mean'] = np.mean(rolloff)
    features['rolloff_std'] = np.std(rolloff)

    if len(audio_data) > 0:
        try:
            pitches, magnitude = librosa.piptrack(y = audio_data, sr = sample_rate, hop_length = hop_length, fmin= 50, fmax = 1600)

            pitches_values = []
            for i in range(magnitude.shape[1]):
                index = magnitude[:, i].argmax()
                pitch = pitches[index, i]
                if pitch > 0:
                    pitches_values.append(pitch)

            if pitches_values:
                features['pitch_mean'] = np.mean(pitches_values)
                features['pitch_std'] = np.std(pitches_values)
                features['pitch_max'] = np.max(pitches_values)
                features['pitch_min'] = np.min(pitches_values)

        except:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_max'] = 0
            features['pitch_min'] = 0

    onset_env = librosa.onset.onset_strength(y = audio_data, sr = sample_rate, hop_length = hop_length)
    tempo = librosa.beat.tempo(onset_envelope = onset_env, sr = sample_rate, hop_length = hop_length)[0]
    features['tempo'] = tempo

    chroma= librosa.feature.chroma_stft(y = audio_data, sr = sample_rate, hop_length = hop_length)
    for i in range(12):
        features[f"chroma{i+1}_mean"] = np.mean(chroma[i])
        features[f"chroma{i+1}_std"] = np.std(chroma[i])

    return features


def process_audio_file(file_path, visualize = False):
    file_name = os.path.basename(file_path)
    file_parts = file_name.split("-")
    
    emotion_labels = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }

    metadata = {}

    if len(file_parts) >= 7:
        metadata['file_path'] = file_path
        metadata['modality'] = file_parts[0]
        metadata['voice_channel'] = file_parts[1]
        metadata['emotion'] = emotion_labels.get(file_parts[2], 'unknown')
        metadata['intensity'] = file_parts[3]
        metadata['statement'] = file_parts[4]
        metadata['repetition'] = file_parts[5]
        metadata['actor'] = file_parts[6].split('.')[0]
        metadata['gender'] = 'female' if int(metadata['actor']) % 2 == 0 else 'male'

    else:
        print(f"Invalid file name format: {file_name}")
        return None, None

    audio_data, sample_rate = load_audio_file(file_path)
    if audio_data is None:
        return None, metadata

    audio_data = remove_silence(audio_data, sample_rate)
    audio_data = normalize_volume(audio_data)

    features = extract_features(audio_data, sample_rate)


    if visualize: 

        visualization_dir = Path("d:/multimodal emotion recognition system/processed/audio_visualizations") 
        visualization_dir.mkdir(parents=True, exist_ok=True)

        actor_emotion_dir = visualization_dir / f"actor_{metadata['actor']}_{metadata['emotion']}"
        actor_emotion_dir.mkdir(exist_ok=True)
        
        base_filename = file_name.replace('.wav', '')

        plt.figure(figsize=(12, 4))
        plt.subplot(2, 1, 1)
        librosa.display.waveshow(audio_data, sr=sample_rate)
        plt.title(f"Waveform of - {metadata['emotion']}({metadata['intensity']})")

        plt.subplot(2, 1, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data, hop_length=512, n_fft=2048)), ref=np.max)
        librosa.display.specshow(D, sr = sample_rate, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')

        plt.tight_layout()
        plt.savefig(actor_emotion_dir / f"{base_filename}_waveform.png")
        plt.close()

        plt.figure(figsize=(12, 6))
        mfcc_coeff = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)

        librosa.display.specshow(mfcc_coeff, sr = sample_rate, x_axis='time')
        plt.colorbar()
        plt.title('MFCC')
        plt.tight_layout()
        plt.savefig(actor_emotion_dir / f"{base_filename}_mfcc.png")
        plt.close()

    return features, metadata


def process_audio_directory(directory_path, output_path, max_files = None, visualize_sample = False):
    all_features = []
    all_metadata = []
    File_count = 0

    wav_files = []

    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))

    wav_files.sort()

    if max_files is not None:
        wav_files = wav_files[:max_files]

    for file_path in tqdm(wav_files, desc="Processing audio files"):
        vis_this_file = visualize_sample and File_count % 50 == 0
        features, metadata = process_audio_file(file_path, visualize=vis_this_file)

        if features is not None and metadata is not None:
            all_features.append(features)
            all_metadata.append(metadata)
        File_count += 1

    print(f"Successfully processed {len(all_features)} out of {len(wav_files)} files")

    features_df = pd.DataFrame(all_features)
    metadata_df = pd.DataFrame(all_metadata)

    os.makedirs(output_path, exist_ok=True)
    features_df.to_csv(os.path.join(output_path, "audio_features.csv"), index = False)
    metadata_df.to_csv(os.path.join(output_path, "audio_metadata.csv"), index = False)

    combine_df = pd.concat([features_df, metadata_df], axis=1)
    combine_df.to_csv(os.path.join(output_path, "audio_combined.csv"), index = False)

    print(f"Features and metadata saved to {output_path}")


    return features_df, metadata_df

def analyze_features(features_df, metadata_df, output_path):
    analysis_dir = Path(output_path) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    combined_df = pd.concat([features_df, metadata_df], axis=1)

    plt.figure(figsize=(12, 6))
    emotion_counts = combined_df['emotion'].value_counts()
    sns.barplot(x=emotion_counts.index, y=emotion_counts.values)
    plt.title('Distribution of Emotions in Dataset')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(analysis_dir / "emotion_distribution.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    gender_counts = combined_df['gender'].value_counts()
    sns.barplot(x=gender_counts.index, y=gender_counts.values)
    plt.title('Gender Distribution in Dataset')
    plt.tight_layout()
    plt.savefig(analysis_dir / "gender_distribution.png")
    plt.close()

    key_features = ['mfcc1_mean', 'mfcc2_mean', 'spectral_centroid_mean', 'zrc_mean', 'rms_mean']

    for feature in key_features:
        if feature in features_df.columns:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='emotion', y=feature, data=combined_df)
            plt.title(f'Distribution of {feature} by Emotion')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(analysis_dir / f"{feature}_by_emotion.png")
            plt.close()
    
    plt.figure(figsize = (20, 16))
    numeric_features = features_df.select_dtypes(include = [np.number])
    correlation_matrix = numeric_features.corr()
    sns.heatmap(correlation_matrix, cmap = 'coolwarm', annot = False, square = True)
    plt.title("Features correlation Matrix")
    plt.tight_layout()
    plt.savefig(analysis_dir / "features_correlation_matrix.png")
    plt.close()

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(numeric_features)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)

    pca_df = pd.DataFrame(data = principal_components, columns = ['PC1', 'PC2'])
    pca_df['emotion'] = metadata_df['emotion']

    plt.figure(figsize = (12, 8))
    sns.scatterplot(x = 'PC1', y = 'PC2', hue = 'emotion', data = pca_df, palette = 'viridis')
    plt.title("PCA of Audio Features by emotion")
    plt.tight_layout()
    plt.savefig(analysis_dir / "PCA_visualization.png")
    plt.close()

    with open(analysis_dir / 'feature_analysis.txt', 'w') as f:
        f.write("Audio Feature Analysis Summary\n")
        f.write("============================\n\n")
        f.write(f"Total samples: {len(combined_df)}\n")
        f.write(f"Emotion distribution:\n{emotion_counts.to_string()}\n\n")
        f.write(f"Gender distribution:\n{gender_counts.to_string()}\n\n")
        f.write(f"Feature statistics:\n{features_df.describe().to_string()}\n\n")
        f.write(f"PCA explained variance ratio: {pca.explained_variance_ratio_}\n")


def generate_mel_spectrogram(audio_data, sample_rate, n_mels=128, max_len=128):
    try:
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=sample_rate,
            n_mels=n_mels,
            n_fft=2048,
            hop_length=512
        )
        
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        if mel_spec_norm.shape[1] < max_len:
            pad_width = max_len - mel_spec_norm.shape[1]
            mel_spec_norm = np.pad(mel_spec_norm, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_norm = mel_spec_norm[:, :max_len]
        
        return mel_spec_norm
    
    except Exception as e:
        print(f"Error generating spectrogram: {e}")
        return np.zeros((n_mels, max_len))


def process_audio_file_with_spectrogram(file_path, n_mels=128, max_len=128):
    file_name = os.path.basename(file_path)
    file_parts = file_name.split("-")
    
    emotion_labels = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    
    metadata = {}
    
    if len(file_parts) >= 7:
        metadata['file_path'] = file_path
        metadata['modality'] = file_parts[0]
        metadata['voice_channel'] = file_parts[1]
        metadata['emotion'] = emotion_labels.get(file_parts[2], 'unknown')
        metadata['intensity'] = file_parts[3]
        metadata['statement'] = file_parts[4]
        metadata['repetition'] = file_parts[5]
        metadata['actor'] = file_parts[6].split('.')[0]
        metadata['gender'] = 'female' if int(metadata['actor']) % 2 == 0 else 'male'
    else:
        print(f"Invalid file name format: {file_name}")
        return None, None
    
    audio_data, sample_rate = load_audio_file(file_path)
    if audio_data is None:
        return None, metadata
    
    audio_data = remove_silence(audio_data, sample_rate)
    audio_data = normalize_volume(audio_data)
    
    spectrogram = generate_mel_spectrogram(audio_data, sample_rate, n_mels, max_len)
    
    return spectrogram, metadata


def process_audio_directory_spectrograms(directory_path, output_path, n_mels=128, max_len=128, max_files=None):
    all_spectrograms = []
    all_labels = []
    all_metadata = []
    
    wav_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    
    wav_files.sort()
    
    if max_files is not None:
        wav_files = wav_files[:max_files]
    
    print(f"Processing {len(wav_files)} audio files for spectrograms...")
    
    for file_path in tqdm(wav_files, desc="Generating spectrograms"):
        spectrogram, metadata = process_audio_file_with_spectrogram(file_path, n_mels, max_len)
        
        if spectrogram is not None and metadata is not None:
            all_spectrograms.append(spectrogram)
            all_labels.append(metadata['emotion'])
            all_metadata.append(metadata)
    
    print(f"Successfully processed {len(all_spectrograms)} spectrograms")
    
    X = np.array(all_spectrograms)
    y = np.array(all_labels)
    metadata_df = pd.DataFrame(all_metadata)
    
    X = X[..., np.newaxis]
    
    os.makedirs(output_path, exist_ok=True)
    np.save(os.path.join(output_path, "spectrograms.npy"), X)
    np.save(os.path.join(output_path, "labels.npy"), y)
    metadata_df.to_csv(os.path.join(output_path, "spectrogram_metadata.csv"), index=False)
    
    print(f"Spectrograms saved to {output_path}")
    print(f"Spectrogram shape: {X.shape}")
    
    return X, y, metadata_df


if __name__ == "__main__":
    from data_setup import AUDIO_SONG_PATH, AUDIO_SPEECH_PATH, PROCESSED_PATH
    import argparse
    
    parser = argparse.ArgumentParser(description="Audio preprocessing")
    parser.add_argument("--mode", type=str, choices=["features", "spectrograms", "both"], 
                       default="both", help="What to generate")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    
    args = parser.parse_args()
    
    speech_features_output = PROCESSED_PATH / "audio_features" / "speech"
    song_features_output = PROCESSED_PATH / "audio_features" / "song"
    speech_spec_output = PROCESSED_PATH / "audio_spectrograms" / "speech"
    song_spec_output = PROCESSED_PATH / "audio_spectrograms" / "song"
    
    if args.mode in ["features", "both"]:
        print("\n" + "="*60)
        print("PROCESSING AUDIO FEATURES")
        print("="*60 + "\n")
        
        print("Processing speech audio files...")
        speech_features, speech_metadata = process_audio_directory(
            AUDIO_SPEECH_PATH, speech_features_output, visualize_sample=args.visualize
        )
        
        print("Processing song audio files...")
        song_features, song_metadata = process_audio_directory(
            AUDIO_SONG_PATH, song_features_output, visualize_sample=args.visualize
        )
        
        print("Analyzing speech features...")
        analyze_features(speech_features, speech_metadata, speech_features_output)
        
        print("Analyzing song features...")
        analyze_features(song_features, song_metadata, song_features_output)
    
    if args.mode in ["spectrograms", "both"]:
        print("\n" + "="*60)
        print("GENERATING MEL-SPECTROGRAMS")
        print("="*60 + "\n")
        
        print("Generating spectrograms for speech audio...")
        X_speech, y_speech, meta_speech = process_audio_directory_spectrograms(
            AUDIO_SPEECH_PATH, speech_spec_output, n_mels=128, max_len=128
        )
        
        print("Generating spectrograms for song audio...")
        X_song, y_song, meta_song = process_audio_directory_spectrograms(
            AUDIO_SONG_PATH, song_spec_output, n_mels=128, max_len=128
        )
        
        if args.visualize:
            plt.figure(figsize=(15, 10))
            for i in range(min(8, len(X_speech))):
                plt.subplot(2, 4, i+1)
                plt.imshow(X_speech[i, :, :, 0], aspect='auto', cmap='viridis', origin='lower')
                plt.title(f"{y_speech[i]}")
                plt.colorbar()
            plt.tight_layout()
            plt.savefig(speech_spec_output / "spectrogram_samples.png")
            plt.close()
            print(f"Spectrogram samples saved to {speech_spec_output}")
    
    print("\n" + "="*60)
    print("AUDIO PREPROCESSING COMPLETE!")
    print("="*60)
