# backend/plot_extended.py


"""
Extended Visualization Module
Additional plots for comprehensive EEG analysis
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import io
import base64
from typing import Dict, List, Optional
import pandas as pd


class ExtendedEEGPlotter:
    """Generate additional EEG analysis plots"""
    
    def __init__(self, fs: int = 256):
        self.fs = fs
        sns.set_style('whitegrid')
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 150
        
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
    
    def plot_line_graph(self, features: np.ndarray, feature_names: List[str]) -> str:
        """Line graph of mean features across channels"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        mean_features = [f for f in feature_names if f.startswith('mean_') and f[-1].isdigit()]
        indices = [feature_names.index(f) for f in mean_features]
        values = features[0, indices]
        
        ax.plot(mean_features, values, marker='o', linewidth=2, markersize=8, color='#2E86AB')
        ax.set_xlabel('EEG Channels', fontsize=12)
        ax.set_ylabel('Mean Activity (µV)', fontsize=12)
        ax.set_title('Time Series of Average Brain Activity', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        return self._fig_to_base64(fig)
    
    def plot_bar_chart(self, features: np.ndarray, feature_names: List[str]) -> str:
        """Bar chart of standard deviation (energy) per channel"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        std_features = [f for f in feature_names if f.startswith('std_') and f[-1].isdigit() and 'lag' not in f]
        indices = [feature_names.index(f) for f in std_features if f in feature_names]
        values = features[0, indices]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(values)))
        bars = ax.bar(range(len(values)), values, color=colors, edgecolor='black')
        
        ax.set_xlabel('EEG Channel', fontsize=12)
        ax.set_ylabel('Standard Deviation (Energy)', fontsize=12)
        ax.set_title('Variation (Energy) per EEG Channel', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([f'Ch {i}' for i in range(len(values))])
        ax.grid(True, axis='y', alpha=0.3)
        
        return self._fig_to_base64(fig)
    
    def plot_radar_chart(self, features: np.ndarray, feature_names: List[str]) -> str:
        """Radar chart of skewness and kurtosis"""
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        skew_features = [f for f in feature_names if f.startswith('skew_') and 'lag' not in f]
        kurt_features = [f for f in feature_names if f.startswith('kurt_') and 'lag' not in f]
        
        skew_indices = [feature_names.index(f) for f in skew_features if f in feature_names]
        kurt_indices = [feature_names.index(f) for f in kurt_features if f in feature_names]
        
        skew_values = features[0, skew_indices]
        kurt_values = features[0, kurt_indices]
        
        angles = np.linspace(0, 2 * np.pi, len(skew_values), endpoint=False).tolist()
        skew_values = np.concatenate((skew_values, [skew_values[0]]))
        kurt_values = np.concatenate((kurt_values, [kurt_values[0]]))
        angles += angles[:1]
        
        ax.plot(angles, skew_values, 'o-', linewidth=2, label='Skewness', color='#2E86AB')
        ax.fill(angles, skew_values, alpha=0.25, color='#2E86AB')
        ax.plot(angles, kurt_values, 's-', linewidth=2, label='Kurtosis', color='#BC4B51')
        ax.fill(angles, kurt_values, alpha=0.25, color='#BC4B51')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f'Ch {i}' for i in range(len(skew_values)-1)])
        ax.set_title('Distribution of Asymmetry & Peakedness', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        return self._fig_to_base64(fig)
    
    def plot_connectivity_heatmap(self, features: np.ndarray, feature_names: List[str]) -> str:
        """Heatmap of covariance matrix or eigenvalues"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract covariance matrix elements
        cov_features = [f for f in feature_names if f.startswith('covM_') and 'lag' not in f]
        
        if len(cov_features) >= 10:  # At least 4x4 matrix elements
            # Reconstruct covariance matrix
            n_channels = 4
            cov_matrix = np.zeros((n_channels, n_channels))
            
            for feat in cov_features:
                if feat in feature_names:
                    idx = feature_names.index(feat)
                    parts = feat.replace('covM_', '').split('_')
                    i, j = int(parts[0]), int(parts[1])
                    cov_matrix[i, j] = features[0, idx]
                    cov_matrix[j, i] = features[0, idx]  # Symmetric
            
            sns.heatmap(cov_matrix, annot=True, fmt='.2e', cmap='RdYlBu_r',
                       xticklabels=[f'Ch{i}' for i in range(n_channels)],
                       yticklabels=[f'Ch{i}' for i in range(n_channels)],
                       square=True, ax=ax, cbar_kws={'label': 'Covariance'})
        else:
            # Use eigenvalues as fallback
            eigen_features = [f for f in feature_names if f.startswith('eigenval_') and 'lag' not in f]
            indices = [feature_names.index(f) for f in eigen_features if f in feature_names]
            values = features[0, indices]
            
            # Create a diagonal matrix for visualization
            n = len(values)
            matrix = np.diag(values)
            sns.heatmap(matrix, annot=True, fmt='.2f', cmap='viridis',
                       square=True, ax=ax, cbar_kws={'label': 'Eigenvalue'})
        
        ax.set_title('Brain Connectivity Map', fontsize=14, fontweight='bold')
        
        return self._fig_to_base64(fig)
    
    def plot_fft_spectrum(self, features: np.ndarray, feature_names: List[str]) -> str:
        """FFT Spectrum plot"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Extract frequency features
        freq_features = [f for f in feature_names if f.startswith('freq_') and f.endswith('_0') and 'lag' not in f]
        freq_features = sorted(freq_features, key=lambda x: int(x.split('_')[1]))
        
        indices = [feature_names.index(f) for f in freq_features if f in feature_names]
        powers = features[0, indices]
        
        # Extract frequency bins from feature names
        freqs = [int(f.split('_')[1]) / 10.0 for f in freq_features]
        
        ax.plot(freqs, powers, linewidth=2, color='#2E86AB')
        ax.fill_between(freqs, powers, alpha=0.3, color='#2E86AB')
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('Power', fontsize=12)
        ax.set_title('FFT Spectrum - Frequency vs Power', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 75])
        
        return self._fig_to_base64(fig)
    
    def plot_quadrant_scatter(self, features: np.ndarray, feature_names: List[str]) -> str:
        """Scatter plot of quadrant differences"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract quadrant difference features
        q1q2_features = [f for f in feature_names if 'mean_d_q1q2_' in f and 'lag' not in f]
        q2q3_features = [f for f in feature_names if 'mean_d_q2q3_' in f and 'lag' not in f]
        
        q1q2_indices = [feature_names.index(f) for f in q1q2_features if f in feature_names]
        q2q3_indices = [feature_names.index(f) for f in q2q3_features if f in feature_names]
        
        x_vals = features[0, q1q2_indices] if len(q1q2_indices) > 0 else [0]
        y_vals = features[0, q2q3_indices] if len(q2q3_indices) > 0 else [0]
        
        scatter = ax.scatter(x_vals, y_vals, c=range(len(x_vals)), 
                            cmap='viridis', s=100, alpha=0.7, edgecolors='black')
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Q1-Q2 Difference (Emotion Intensity)', fontsize=12)
        ax.set_ylabel('Q2-Q3 Difference (Emotion Intensity)', fontsize=12)
        ax.set_title('Quadrant Differences (Emotion Intensity)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Channel Index')
        
        return self._fig_to_base64(fig)
    
    def plot_emotion_distribution(self, predictions: Dict[str, float]) -> str:
        """Pie chart of emotion distribution"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        labels = list(predictions.keys())
        sizes = list(predictions.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                           colors=colors, startangle=90, 
                                           textprops={'fontsize': 12, 'fontweight': 'bold'})
        
        ax.set_title('Emotion Distribution (Predicted Probabilities)', 
                    fontsize=14, fontweight='bold')
        
        return self._fig_to_base64(fig)
    
    def plot_pca_variance(self, pca_obj) -> str:
        """PCA cumulative explained variance curve"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        cumsum = np.cumsum(pca_obj.explained_variance_ratio_)
        
        ax.plot(range(1, len(cumsum) + 1), cumsum, marker='o', 
               linewidth=2, markersize=6, color='#2E86AB')
        ax.axhline(y=0.95, color='red', linestyle='--', label='95% Variance')
        ax.axhline(y=0.99, color='orange', linestyle='--', label='99% Variance')
        
        ax.set_xlabel('Number of Components', fontsize=12)
        ax.set_ylabel('Cumulative Explained Variance', fontsize=12)
        ax.set_title('PCA - Cumulative Explained Variance Curve', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim([0, 1.05])
        
        return self._fig_to_base64(fig)
    
    def plot_band_power_per_emotion(self, band_powers: Dict[str, np.ndarray], 
                                    emotion_labels: List[str]) -> str:
        """Band power distribution per emotion"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bands = list(band_powers.keys())
        x = np.arange(len(bands))
        width = 0.35
        
        # Assuming 2 emotions for binary classification
        if len(emotion_labels) == 2:
            powers1 = [band_powers[band][0] for band in bands]
            powers2 = [band_powers[band][1] for band in bands]
            
            ax.bar(x - width/2, powers1, width, label=emotion_labels[0], 
                  color='#2E86AB', edgecolor='black')
            ax.bar(x + width/2, powers2, width, label=emotion_labels[1], 
                  color='#BC4B51', edgecolor='black')
        
        ax.set_xlabel('Frequency Bands', fontsize=12)
        ax.set_ylabel('Average Power (µV²)', fontsize=12)
        ax.set_title('Band Power Distribution per Emotion', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(bands, rotation=15)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        return self._fig_to_base64(fig)
    
    def plot_attention_map(self, attention_weights: np.ndarray) -> str:
        """Transformer attention map"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(attention_weights, cmap='YlOrRd', ax=ax, 
                   cbar_kws={'label': 'Attention Weight'})
        
        ax.set_xlabel('Key Position', fontsize=12)
        ax.set_ylabel('Query Position', fontsize=12)
        ax.set_title('Model Attention Map (Interpretability)', fontsize=14, fontweight='bold')
        
        return self._fig_to_base64(fig)
    
    def plot_filtered_signal(self, raw_signal: np.ndarray, 
                            filtered_signal: np.ndarray, 
                            channel_idx: int = 0) -> str:
        """Compare raw and filtered signals"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        time = np.arange(raw_signal.shape[1]) / self.fs
        
        ax1.plot(time, raw_signal[channel_idx], linewidth=0.5, color='gray', alpha=0.7)
        ax1.set_ylabel('Amplitude (µV)', fontsize=11)
        ax1.set_title(f'Raw EEG Signal - Channel {channel_idx}', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(time, filtered_signal[channel_idx], linewidth=0.8, color='#2E86AB')
        ax2.set_xlabel('Time (s)', fontsize=11)
        ax2.set_ylabel('Amplitude (µV)', fontsize=11)
        ax2.set_title(f'Filtered EEG Signal (0.5-45 Hz) - Channel {channel_idx}', 
                     fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def plot_correlation_heatmap(self, features: np.ndarray, 
                                feature_names: List[str],
                                max_features: int = 30) -> str:
        """Feature correlation heatmap"""
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Select subset of features if too many
        if features.shape[1] > max_features:
            # Select important feature groups
            selected_features = []
            for prefix in ['mean_', 'std_', 'skew_', 'kurt_']:
                selected_features += [f for f in feature_names if f.startswith(prefix) and 'lag' not in f][:8]
            
            indices = [feature_names.index(f) for f in selected_features if f in feature_names]
            features_subset = features[:, indices]
            names_subset = [feature_names[i] for i in indices]
        else:
            features_subset = features
            names_subset = feature_names
        
        # Compute correlation
        corr = np.corrcoef(features_subset.T)
        
        sns.heatmap(corr, cmap='coolwarm', center=0, 
                   xticklabels=names_subset, yticklabels=names_subset,
                   square=True, ax=ax, cbar_kws={'label': 'Correlation'})
        
        ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
