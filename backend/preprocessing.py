
# backend/preprocessing.py

"""
EEG Signal Preprocessing Module
Handles loading, filtering, artifact removal, and epoching of EEG signals
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch, welch
from sklearn.decomposition import FastICA
import warnings
warnings.filterwarnings('ignore')


def load_eeg(file_path):
    """
    Load EEG data from CSV file
    Returns: raw signal (channels × samples), metadata dict
    """
    try:
        df = pd.read_csv(file_path)
        
        # Identify signal columns (exclude label/metadata columns)
        exclude_cols = ['label', 'subject_id', 'timestamp', 'emotion', 'depression_stage']
        signal_cols = [col for col in df.columns if col not in exclude_cols]
        
        raw_signal = df[signal_cols].values.T  # Shape: (n_channels, n_samples)
        
        metadata = {
            'fs': 128,  # Default sampling frequency (adjust based on your data)
            'n_channels': len(signal_cols),
            'n_samples': raw_signal.shape[1],
            'channel_names': signal_cols,
            'labels': df['label'].values if 'label' in df.columns else None
        }
        
        return raw_signal, metadata
    
    except Exception as e:
        raise ValueError(f"Error loading EEG file: {str(e)}")


def bandpass_filter(sig, fs, low=0.5, high=45, order=4):
    """
    Apply bandpass filter to remove noise outside frequency range
    """
    nyq = fs / 2
    low_norm = low / nyq
    high_norm = high / nyq
    
    b, a = butter(order, [low_norm, high_norm], btype='band')
    filtered = np.zeros_like(sig)
    
    for i in range(sig.shape[0]):
        filtered[i] = filtfilt(b, a, sig[i])
    
    return filtered


def notch_filter(sig, fs, freq=50, quality=30):
    """
    Apply notch filter to remove power line interference
    """
    nyq = fs / 2
    freq_norm = freq / nyq
    
    b, a = iirnotch(freq_norm, quality)
    filtered = np.zeros_like(sig)
    
    for i in range(sig.shape[0]):
        filtered[i] = filtfilt(b, a, sig[i])
    
    return filtered


def run_ica(raw_signal, n_components=None):
    """
    Apply Independent Component Analysis to remove artifacts
    """
    if n_components is None:
        n_components = min(raw_signal.shape[0], 10)
    
    ica = FastICA(n_components=n_components, random_state=42, max_iter=500)
    
    # Transpose for ICA (samples × channels)
    signals_transposed = raw_signal.T
    components = ica.fit_transform(signals_transposed)
    
    # Reconstruct (remove first component which often contains artifacts)
    components[:, 0] = 0  # Zero out first component
    cleaned = ica.inverse_transform(components)
    
    return cleaned.T  # Back to (channels × samples)


def epoch_signal(sig, fs, epoch_len=2.0, overlap=0.5):
    """
    Segment signal into overlapping epochs
    Returns: array of shape (n_epochs, n_channels, n_samples_per_epoch)
    """
    n_channels, n_samples = sig.shape
    samples_per_epoch = int(epoch_len * fs)
    step = int(samples_per_epoch * (1 - overlap))
    
    epochs = []
    start = 0
    
    while start + samples_per_epoch <= n_samples:
        epoch = sig[:, start:start + samples_per_epoch]
        epochs.append(epoch)
        start += step
    
    return np.array(epochs)


def preprocess_pipeline(file_path, apply_ica=True):
    """
    Complete preprocessing pipeline
    Returns: epochs, metadata, filtered_signal
    """
    # Load
    raw, meta = load_eeg(file_path)
    print(f"Loaded: {meta['n_channels']} channels, {meta['n_samples']} samples")
    
    # Filter
    filtered = bandpass_filter(raw, meta['fs'], low=0.5, high=45)
    filtered = notch_filter(filtered, meta['fs'], freq=50)
    print("Filtering complete")
    
    # ICA (optional)
    if apply_ica:
        cleaned = run_ica(filtered)
        print("ICA complete")
    else:
        cleaned = filtered
    
    # Epoch
    epochs = epoch_signal(cleaned, meta['fs'], epoch_len=2.0, overlap=0.5)
    print(f"Created {len(epochs)} epochs")
    
    return epochs, meta, filtered


def get_sample_for_plotting(file_path, n_samples=1000):
    """
    Get a sample of raw and filtered data for visualization
    """
    raw, meta = load_eeg(file_path)
    filtered = bandpass_filter(raw, meta['fs'])
    
    return raw[:, :n_samples], filtered[:, :n_samples], meta
