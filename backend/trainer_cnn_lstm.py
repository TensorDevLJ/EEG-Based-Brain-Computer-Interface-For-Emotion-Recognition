"""
Training Pipeline for CNN+LSTM Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import copy


class EEGDataset(Dataset):
    """PyTorch Dataset for EEG features"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class EarlyStopping:
    """Early stopping callback"""
    
    def __init__(self, patience: int = 10, delta: float = 0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None
        
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0


class CNNLSTMTrainer:
    """Training pipeline for CNN+LSTM model"""
    
    def __init__(self, model, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': []
        }
        
    def train(self,
              train_dataset: EEGDataset,
              val_dataset: Optional[EEGDataset] = None,
              epochs: int = 50,
              batch_size: int = 32,
              lr: float = 0.001,
              weight_decay: float = 1e-5,
              patience: int = 10) -> Dict:
        """
        Train the model
        """
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                         factor=0.5, patience=5)
        
        early_stopping = EarlyStopping(patience=patience)
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_preds, train_targets = [], []
            
            for features, labels in train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                train_preds.extend(preds.cpu().numpy())
                train_targets.extend(labels.cpu().numpy())
            
            train_loss /= len(train_loader)
            train_acc = accuracy_score(train_targets, train_preds)
            train_f1 = f1_score(train_targets, train_preds, average='macro')
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_f1'].append(train_f1)
            
            # Validation phase
            if val_loader:
                val_loss, val_acc, val_f1 = self.evaluate(val_dataset)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                self.history['val_f1'].append(val_f1)
                
                scheduler.step(val_loss)
                early_stopping(val_loss, self.model)
                
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
                
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    self.model.load_state_dict(early_stopping.best_model)
                    break
            else:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        
        return self.history
    
    def evaluate(self, dataset: EEGDataset) -> Tuple[float, float, float]:
        """
        Evaluate model on dataset
        Returns: (loss, accuracy, f1_score)
        """
        self.model.eval()
        loader = DataLoader(dataset, batch_size=32)
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for features, labels in loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(loader)
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='macro')
        
        return avg_loss, accuracy, f1
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions
        Returns: (predicted_classes, probabilities)
        """
        self.model.eval()
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
        
        return preds, probs
