

"""
CNN + LSTM Hybrid Model for EEG Emotion Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import os
import json


class CNN_LSTM_Model(nn.Module):
    """
    Hybrid CNN + LSTM model for EEG feature classification
    """
    
    def __init__(self, 
                 input_dim: int = 50,
                 num_classes: int = 2,
                 cnn_channels: list = [64, 128, 256],
                 lstm_hidden: int = 128,
                 lstm_layers: int = 2,
                 dropout: float = 0.5):
        """
        Args:
            input_dim: Number of input features (after PCA)
            num_classes: Number of emotion classes
            cnn_channels: List of CNN channel sizes
            lstm_hidden: LSTM hidden size
            lstm_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(CNN_LSTM_Model, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Reshape layer to add channel dimension
        # Input: (batch, input_dim) -> (batch, 1, input_dim)
        
        # CNN layers
        self.conv1 = nn.Conv1d(1, cnn_channels[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_channels[0])
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_channels[1])
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(cnn_channels[1], cnn_channels[2], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(cnn_channels[2])
        self.pool3 = nn.MaxPool1d(2)
        
        # Calculate size after conv layers
        self.cnn_output_size = cnn_channels[2]
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Fully connected layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_hidden * 2, 256)  # *2 for bidirectional
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: (batch, input_dim) or (batch, seq_len, input_dim)
        Returns:
            (batch, num_classes)
        """
        # Handle both 2D and 3D inputs
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
        
        batch_size = x.size(0)
        
        # CNN feature extraction
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len) for Conv1d
        
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Prepare for LSTM: (batch, seq_len, features)
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take last output
        x = lstm_out[:, -1, :]
        
        # Fully connected layers
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def predict(self, x):
        """
        Predict with softmax
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
        return probs


def save_model(model: CNN_LSTM_Model, path: str, metadata: dict = None):
    """Save model and metadata"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': model.input_dim,
            'num_classes': model.num_classes
        },
        'metadata': metadata or {}
    }, path)
    
    # Save metadata as JSON
    if metadata:
        meta_path = path.replace('.pth', '_metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"Model saved to {path}")


def load_model(path: str, device: str = 'cpu') -> tuple:
    """
    Load model from file
    Returns: (model, metadata)
    """
    checkpoint = torch.load(path, map_location=device)
    
    config = checkpoint['model_config']
    model = CNN_LSTM_Model(
        input_dim=config['input_dim'],
        num_classes=config['num_classes']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    metadata = checkpoint.get('metadata', {})
    
    return model, metadata
