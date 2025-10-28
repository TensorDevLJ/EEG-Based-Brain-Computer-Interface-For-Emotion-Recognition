"""
EEG Feature Extraction Module
Extracts frequency, time-domain, connectivity, and complexity features
"""

import numpy as np
import pywt
from scipy import signal
from scipy.signal import welch, coherence, hilbert
from scipy.stats import skew, kurtosis, entropy


def compute_welch(epoch_channel, fs, nperseg=None):
    """
    Compute Power Spectral Density using Welch's method
    """
    if nperseg is None:
        nperseg = min(256, len(epoch_channel))
    
    freqs, psd = welch(epoch_channel, fs=fs, nperseg=nperseg)
    return freqs, psd

def fix_feature_shape(X, expected_dim):
    """
    Pads or trims features matrix to ensure correct feature shape.
    """
    current_dim = X.shape[1]
    if current_dim < expected_dim:
        pad_width = expected_dim - current_dim
        X = np.pad(X, ((0,0), (0, pad_width)), 'constant')
    elif current_dim > expected_dim:
        X = X[:, :expected_dim]
    return X


def bandpower(freqs, psd, band):
    """
    Calculate power in a specific frequency band
    """
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.trapz(psd[idx], freqs[idx])


def relative_power(band_powers):
    """
    Calculate relative band powers (normalized by total power)
    """
    total = sum(band_powers.values())
    return {k: v / total for k, v in band_powers.items()}


def peak_alpha_frequency(freqs, psd, band=(8, 13)):
    """
    Find the peak frequency in alpha band (Individual Alpha Frequency)
    """
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    alpha_freqs = freqs[idx]
    alpha_psd = psd[idx]
    
    if len(alpha_psd) > 0:
        peak_idx = np.argmax(alpha_psd)
        return alpha_freqs[peak_idx]
    return 10.0  # Default


def frontal_alpha_asymmetry(alpha_left, alpha_right, eps=1e-10):
    """
    Calculate frontal alpha asymmetry (FAA)
    Positive = right dominant (withdrawal), Negative = left dominant (approach)
    """
    return np.log(alpha_right + eps) - np.log(alpha_left + eps)


def spectral_entropy(psd, normalize=True):
    """
    Calculate spectral entropy (signal complexity)
    """
    psd_norm = psd / np.sum(psd)
    ent = entropy(psd_norm + 1e-10)
    
    if normalize:
        max_ent = np.log2(len(psd))
        ent = ent / max_ent if max_ent > 0 else 0
    
    return ent


def hjorth_parameters(signal):
    """
    Calculate Hjorth parameters: Activity, Mobility, Complexity
    """
    # Activity (variance)
    activity = np.var(signal)
    
    # First derivative
    dsignal = np.diff(signal)
    
    # Mobility
    mobility = np.sqrt(np.var(dsignal) / activity) if activity > 0 else 0
    
    # Second derivative
    ddsignal = np.diff(dsignal)
    
    # Complexity
    complexity = (np.sqrt(np.var(ddsignal) / np.var(dsignal)) / mobility) if mobility > 0 else 0
    
    return activity, mobility, complexity


def statistical_moments(signal):
    """
    Calculate statistical features: mean, variance, skewness, kurtosis
    """
    return {
        'mean': np.mean(signal),
        'var': np.var(signal),
        'skew': skew(signal),
        'kurt': kurtosis(signal)
    }


def compute_coherence_band(sig1, sig2, fs, band, nperseg=128):
    """
    Compute coherence between two signals in a specific frequency band
    """
    f, Cxy = coherence(sig1, sig2, fs=fs, nperseg=nperseg)
    
    idx = np.logical_and(f >= band[0], f <= band[1])
    return np.mean(Cxy[idx])


def compute_plv(sig1, sig2, fs, band):
    """
    Compute Phase Locking Value between two signals in a frequency band
    """
    # Bandpass filter
    nyq = fs / 2
    low, high = band
    b, a = signal.butter(4, [low/nyq, high/nyq], btype='band')
    
    sig1_filt = signal.filtfilt(b, a, sig1)
    sig2_filt = signal.filtfilt(b, a, sig2)
    
    # Hilbert transform to get phases
    phase1 = np.angle(hilbert(sig1_filt))
    phase2 = np.angle(hilbert(sig2_filt))
    
    # PLV
    phase_diff = phase1 - phase2
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
    
    return plv


def cwt_features(signal, fs):
    """
    Compute Continuous Wavelet Transform features
    Returns scalogram and energy in different scales
    """
    scales = np.arange(1, 128)
    coeffs, freqs = pywt.cwt(signal, scales, 'morl', sampling_period=1/fs)
    
    # Energy at different frequency bands
    scalogram = np.abs(coeffs)
    
    return scalogram, np.mean(scalogram, axis=1)


def extract_channel_features(epoch_channel, fs):
    """
    Extract all features from a single channel
    """
    features = {}
    
    # PSD
    freqs, psd = compute_welch(epoch_channel, fs)
    
    # Band definitions
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    
    # Band powers
    band_powers = {}
    for band_name, band_range in bands.items():
        bp = bandpower(freqs, psd, band_range)
        features[f'{band_name}_power'] = bp
        band_powers[band_name] = bp
    
    # Relative powers
    rel_powers = relative_power(band_powers)
    for band_name, rel_power in rel_powers.items():
        features[f'{band_name}_rel_power'] = rel_power
    
    # Peak alpha frequency
    features['peak_alpha_freq'] = peak_alpha_frequency(freqs, psd)
    
    # Band ratios
    features['theta_alpha_ratio'] = band_powers['theta'] / (band_powers['alpha'] + 1e-10)
    features['alpha_beta_ratio'] = band_powers['alpha'] / (band_powers['beta'] + 1e-10)
    features['theta_beta_ratio'] = band_powers['theta'] / (band_powers['beta'] + 1e-10)
    
    # Spectral entropy
    features['spectral_entropy'] = spectral_entropy(psd)
    
    # Hjorth parameters
    activity, mobility, complexity = hjorth_parameters(epoch_channel)
    features['hjorth_activity'] = activity
    features['hjorth_mobility'] = mobility
    features['hjorth_complexity'] = complexity
    
    # Statistical moments
    stats = statistical_moments(epoch_channel)
    features.update(stats)
    
    return features


def build_feature_vector(epoch, metadata):
    """
    Build complete feature vector from an epoch (all channels)
    Returns: 1D feature array
    """
    fs = metadata['fs']
    n_channels = epoch.shape[0]
    all_features = []
    
    # Per-channel features
    for ch in range(n_channels):
        ch_features = extract_channel_features(epoch[ch], fs)
        all_features.extend(ch_features.values())
    
    # Frontal alpha asymmetry (if frontal channels exist)
    # Assuming first few channels are frontal
    if n_channels >= 2:
        freqs_0, psd_0 = compute_welch(epoch[0], fs)
        freqs_1, psd_1 = compute_welch(epoch[1], fs)
        
        alpha_left = bandpower(freqs_0, psd_0, (8, 13))
        alpha_right = bandpower(freqs_1, psd_1, (8, 13))
        faa = frontal_alpha_asymmetry(alpha_left, alpha_right)
        all_features.append(faa)
    
    # Connectivity features (sample: first two channels)
    if n_channels >= 2:
        bands = {'alpha': (8, 13), 'beta': (13, 30)}
        for band_name, band_range in bands.items():
            coh = compute_coherence_band(epoch[0], epoch[1], fs, band_range)
            plv = compute_plv(epoch[0], epoch[1], fs, band_range)
            all_features.extend([coh, plv])
    
    # Global features (mean across channels)
    mean_power = np.mean([np.var(epoch[ch]) for ch in range(n_channels)])
    all_features.append(mean_power)
    
    return np.array(all_features, dtype=np.float32)


def extract_all_features(epochs, metadata):
    """
    Extract features from all epochs
    Returns: 2D array (n_epochs, n_features)
    """
    features_list = []
    
    for epoch in epochs:
        feat_vec = build_feature_vector(epoch, metadata)
        features_list.append(feat_vec)
    
    return np.array(features_list)


def preprocess_features(df, scaler=None):
    """
    Preprocess pre-extracted features from CSV
    Args:
        df: pandas DataFrame with features
        scaler: fitted StandardScaler (optional)
    Returns:
        X: numpy array of features (n_samples, n_features)
        feature_names: list of feature names
    """
    # Remove label column if exists
    exclude_cols = ['Label', 'label', 'subject_id', 'timestamp']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].values.astype(np.float32)
    
    # Replace inf/nan
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale if scaler provided
    if scaler is not None:
        X = scaler.transform(X)
    else:
        # Simple standardization
        mean = np.mean(X, axis=0, keepdims=True)
        std = np.std(X, axis=0, keepdims=True) + 1e-8
        X = (X - mean) / std
    
    return X, feature_cols


def compute_depression_index(features, probs):
    """
    Compute continuous depression index from features and probabilities
    Args:
        features: feature array (n_samples, n_features)
        probs: class probabilities [Not Depressed, Mild, Moderate, Mod Severe, Severe]
    Returns:
        depression_index: float [0-1]
    """
    # Weighted average based on class severity
    weights = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    di = np.sum(probs * weights)
    
    # Additional feature-based adjustment (if features indicate high theta/alpha ratio)
    # Assuming first few features contain relevant band power info
    if features.shape[1] > 10:
        # Extract relevant features (example indices, adjust based on your feature order)
        theta_alpha_proxy = np.mean(features[:, 2:4]) / (np.mean(features[:, 4:6]) + 1e-8)
        if theta_alpha_proxy > 1.5:
            di = min(1.0, di + 0.1)
    
    return np.clip(di, 0.0, 1.0)
