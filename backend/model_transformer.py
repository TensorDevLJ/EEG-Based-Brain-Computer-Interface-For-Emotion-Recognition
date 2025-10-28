# backend/model_transformer.py

"""
Transformer Model for EEG Emotion & Depression Classification
"""

import torch
import torch.nn as nn
import math


class EEGTransformer(nn.Module):
    """
    Transformer model for EEG sequence classification
    """
    def __init__(
        self,
        input_dim,
        d_model=128,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        num_classes=5,
        max_seq_len=100
    ):
        super().__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=False
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, src_mask=None):
        """
        Args:
            src: (seq_len, batch_size, input_dim)
        
        Returns:
            logits: (batch_size, num_classes)
            memory: transformer outputs for interpretability
        """
        # Project input
        src = self.input_projection(src) * math.sqrt(self.d_model)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Transformer encoding
        memory = self.transformer_encoder(src, src_mask)
        
        # Global pooling (mean over sequence)
        pooled = memory.mean(dim=0)  # (batch_size, d_model)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits, memory
    
    def get_attention_weights(self):
        """Extract attention weights for visualization"""
        attention_weights = []
        for layer in self.transformer_encoder.layers:
            attention_weights.append(layer.self_attn)
        return attention_weights


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Precompute positional encodings up to max_len, but allow dynamic extension
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # If incoming sequence is longer than our precomputed pe, dynamically extend it
        seq_len = x.size(0)
        if seq_len > self.pe.size(0):
            device = x.device
            d_model = x.size(2)
            extra_len = seq_len - self.pe.size(0)
            start_pos = self.pe.size(0)
            position = torch.arange(start_pos, start_pos + extra_len, device=device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))

            extra_pe = torch.zeros(extra_len, 1, d_model, device=device)
            extra_pe[:, 0, 0::2] = torch.sin(position * div_term)
            extra_pe[:, 0, 1::2] = torch.cos(position * div_term)

            pe = torch.cat([self.pe.to(device), extra_pe], dim=0)
        else:
            pe = self.pe.to(x.device)

        x = x + pe[:seq_len]
        return self.dropout(x)


def save_model(model, path, metadata=None):
    """Save model checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': model.input_dim,
            'd_model': model.d_model,
        },
        'metadata': metadata
    }
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(path, device='cpu'):
    """Load model from checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    
    config = checkpoint['model_config']
    model = EEGTransformer(
        input_dim=config['input_dim'],
        d_model=config.get('d_model', 128)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint.get('metadata', {})
