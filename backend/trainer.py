# backend/trainer.py



"""
Training Pipeline for Transformer Model
Supports standard train/val/test split and Leave-One-Subject-Out (LOSO)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
import time
from tqdm import tqdm


class EEGDataset(Dataset):
    """PyTorch Dataset for EEG features"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def collate_fn(batch):
    """Custom collate function for variable length sequences"""
    features, labels = zip(*batch)
    features = torch.stack(features)
    labels = torch.stack(labels)
    return features, labels


class Trainer:
    """Trainer class for EEG Transformer"""
    def __init__(
        self,
        model,
        device='cpu',
        lr=1e-4,
        weight_decay=1e-5,
        patience=10
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=patience//2, factor=0.5
        )
        self.patience = patience
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': []
        }
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for features, labels in train_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            
            # Reshape for transformer: (seq_len=1, batch, features)
            features = features.unsqueeze(0)
            logits, _ = self.model(features)
            
            loss = self.criterion(logits, labels)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        return avg_loss, acc, f1
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                features = features.unsqueeze(0)
                logits, _ = self.model(features)
                
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        return avg_loss, acc, f1, all_preds, all_labels
    
    def train(self, train_loader, val_loader, epochs=50, save_path='saved_models/best_model.pth'):
        """Complete training loop"""
        best_val_f1 = 0
        patience_counter = 0
        
        print(f"Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, val_f1, _, _ = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            
            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(self.model.state_dict(), save_path)
                patience_counter = 0
            else:
                patience_counter += 1
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.2f}s) - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            
            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load(save_path))
        print(f"Training complete. Best val F1: {best_val_f1:.4f}")
        
        return self.history


def evaluate(model, test_loader, device='cpu'):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            features = features.unsqueeze(0)
            logits, _ = model(features)
            probs = torch.softmax(logits, dim=1)
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    precision, recall, f1_per_class, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }
    
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1 (macro): {f1_macro:.4f}")
    
    return metrics


def prepare_data_loaders(features, labels, batch_size=32, test_size=0.2, val_size=0.1):
    """Prepare train/val/test data loaders"""
    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, labels, test_size=test_size+val_size, random_state=42, stratify=labels
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size/(test_size+val_size), random_state=42, stratify=y_temp
    )
    
    # Datasets
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    test_dataset = EEGDataset(X_test, y_test)
    
    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
