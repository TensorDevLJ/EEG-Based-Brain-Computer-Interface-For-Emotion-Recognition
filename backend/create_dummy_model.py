# backend/create_dummy_model.py
import torch
import torch.nn as nn
from pathlib import Path

# Dummy model definition (matches expected input/output)
class DummyModel(nn.Module):
    def __init__(self, input_size=100, output_size=1):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return torch.zeros((x.shape[0], 1))  # always return zeros

# Create model instance
model = DummyModel()

# Optional metadata
metadata = {"info": "dummy model for API testing"}

# Save to saved_models/best_model.pth
MODEL_DIR = Path("saved_models")
MODEL_DIR.mkdir(exist_ok=True)
torch.save({"model": model, "metadata": metadata}, MODEL_DIR / "best_model.pth")

print("Dummy model saved to saved_models/best_model.pth")

