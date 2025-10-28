#embedding.py
"""
Embedding Module
Converts feature sequences into transformer-ready input with positional encoding
"""

import torch
import torch.nn as nn
import numpy as np
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequence data
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor shape (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def build_sequence(features, device='cpu'):
    """
    Convert feature array to transformer input tensor
    
    Args:
        features: numpy array (n_epochs, n_features)
        device: torch device
    
    Returns:
        tensor: shape (seq_len, batch_size=1, feature_dim)
    """
    # Normalize features
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-8
    features_norm = (features - mean) / std
    
    # Convert to tensor
    tensor = torch.FloatTensor(features_norm).to(device)
    
    # Reshape: (seq_len, batch_size=1, feature_dim)
    tensor = tensor.unsqueeze(1)
    
    return tensor, mean, std


def apply_positional_encoding(X, d_model, max_len=5000, dropout=0.1):
    """
    Apply positional encoding to sequence
    
    Args:
        X: tensor (seq_len, batch_size, feature_dim)
        d_model: model dimension
    
    Returns:
        encoded tensor
    """
    pos_encoder = PositionalEncoding(d_model, max_len, dropout)
    return pos_encoder(X)


class FeatureProjection(nn.Module):
    """
    Projects input features to model dimension
    """
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.projection = nn.Linear(input_dim, d_model)
    
    def forward(self, x):
        return self.projection(x)


def prepare_transformer_input(features, d_model, device='cpu'):
    """
    Complete pipeline: features -> normalized -> projected -> positional encoding
    
    Args:
        features: numpy array (n_epochs, n_features)
        d_model: transformer model dimension
        device: torch device
    
    Returns:
        tensor ready for transformer input
    """
    # Build sequence
    seq_tensor, mean, std = build_sequence(features, device)
    
    # Project to d_model dimension
    input_dim = seq_tensor.shape[-1]
    projector = FeatureProjection(input_dim, d_model).to(device)
    projected = projector(seq_tensor)
    
    # Add positional encoding
    encoded = apply_positional_encoding(projected, d_model)
    
    return encoded, projector, mean, std
