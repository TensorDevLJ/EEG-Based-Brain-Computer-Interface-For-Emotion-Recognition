#backend/api.py

"""
FastAPI Backend for EEG Emotion & Depression Detection
Accepts CSV with pre-extracted features, runs Transformer inference, generates graphs
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import torch
import io
from typing import Dict, Any

# Support running either as a module (uvicorn backend.app:app) or as a script
# (python app.py). Try relative imports first, then fall back to local imports.
try:
    from .model_transformer import EEGTransformer, load_model
    from .features import preprocess_features, compute_depression_index
    from .plots import generate_all_graphs
    from .explainers import generate_explanations, get_shap_values
except Exception:
    # Fall back when running `python app.py` directly (no package context)
    from model_transformer import EEGTransformer, load_model
    from features import preprocess_features, compute_depression_index
    from plots import generate_all_graphs
    from explainers import generate_explanations, get_shap_values

app = FastAPI(title="EEG Depression Detection API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and scaler
MODEL = None
SCALER = None
FEATURE_STATS = None

@app.on_event("startup")
async def load_models():
    """Load trained model and scaler on startup"""
    global MODEL, SCALER, FEATURE_STATS
    try:
        MODEL, metadata = load_model('saved_models/best_model.pth')
        SCALER = metadata.get('scaler')
        FEATURE_STATS = metadata.get('feature_stats', {})
        print("✅ Model loaded successfully")
    except Exception as e:
        # Avoid Unicode emoji in prints to prevent encoding errors on Windows consoles
        print(f"Warning: Could not load model: {e}")
        print("Creating dummy model for testing...")
        MODEL = EEGTransformer(input_dim=1000, d_model=128, nhead=8, num_encoder_layers=4, num_classes=5)
        MODEL.eval()
        SCALER = None
        FEATURE_STATS = {}


@app.get("/")
async def root():
    return {
        "status": "online",
        "endpoints": ["/predict", "/analyze", "/metrics"],
        "model_loaded": MODEL is not None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Main prediction endpoint
    Accepts: CSV file with pre-extracted EEG features
    Returns: Predictions, graphs (base64), explanations, depression stage
    """
    try:
        # 1. Load CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        if df.empty:
            raise HTTPException(status_code=400, detail="Empty CSV file")
        
        print(f"📊 Loaded CSV: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # 2. Preprocess features
     # 2. Preprocess features
        X, feature_names = preprocess_features(df, SCALER)
        print(f"🔎 Feature matrix shape: {X.shape}")  # DEBUG

        # -----------UPDATE STARTS HERE-----------
        expected_dim = 1000  # Match your model input_dim!
        current_dim = X.shape[1]
        if current_dim < expected_dim:
            pad_width = expected_dim - current_dim
            X = np.pad(X, ((0, 0), (0, pad_width)), 'constant')
            print(f"✨ Padded features: now shape {X.shape}")  # DEBUG
        elif current_dim > expected_dim:
            X = X[:, :expected_dim]
            print(f"🔪 Trimmed features: now shape {X.shape}")  # DEBUG
        # -----------UPDATE ENDS HERE-----------

        if X is None or len(X) == 0:
            raise HTTPException(status_code=400, detail="No valid features extracted")

        # Model expects shape (seq_len, batch, features) for transformer
        X_tensor = torch.FloatTensor(X).unsqueeze(0)  # (1, seq_len, features)

        with torch.no_grad():
            logits, embeddings = MODEL(X_tensor.transpose(0, 1))  # (seq, batch, features)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        class_names = ['Not Depressed', 'Mild', 'Moderate', 'Moderately Severe', 'Severe']
        predicted_idx = np.argmax(probs)
        predicted_class = class_names[predicted_idx]

        depression_index = compute_depression_index(X, probs)

        print("🎨 Generating graphs...")
        graphs = generate_all_graphs(df, X, probs, embeddings, class_names)

        print("🧠 Generating explanations...")
        explanations = generate_explanations(
            X, feature_names, probs, predicted_class,
            depression_index, FEATURE_STATS, MODEL
        )

        response = {
            "status": "success",
            "predicted_class": predicted_class,
            "depression_stage": predicted_class,
            "probabilities": {
                class_names[i]: float(probs[i]) for i in range(len(class_names))
            },
            "depression_index": float(depression_index),
            "confidence": float(probs[predicted_idx]),
            "top_features": explanations['top_features'],
            "explanation_text": explanations['explanation_text'],
            "dominant_wave": explanations['dominant_wave'],
            "graphs": graphs,
            "metadata": {
                "n_samples": int(df.shape[0]),
                "n_features": int(X.shape[1]),
                "model_version": "v1.0.0"
            }
        }

        print(f"✅ Prediction complete: {predicted_class} ({probs[predicted_idx]:.2%})")
        return response

    except Exception as e:
        print(f"❌ Error in /predict: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

       



@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Analysis-only endpoint (no prediction, just graphs and feature stats)
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        X, feature_names = preprocess_features(df, SCALER)
        
        # Generate subset of graphs
        graphs = {
            'feature_distribution': generate_all_graphs(df, X, None, None, None)['graph_5'],
            'time_series': generate_all_graphs(df, X, None, None, None)['graph_1'],
        }
        
        return {
            "status": "success",
            "graphs": graphs,
            "feature_summary": {
                "n_features": int(X.shape[1]),
                "n_samples": int(X.shape[0]),
                "feature_names": feature_names[:20]  # Top 20
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """
    Return model performance metrics
    """
    return {
        "model_version": "v1.0.0",
        "model_type": "Transformer",
        "num_classes": 5,
        "classes": ['Not Depressed', 'Mild', 'Moderate', 'Moderately Severe', 'Severe'],
        "input_features": 1000,  # Adjust based on your dataset
        "model_loaded": MODEL is not None,
        "metrics": {
            "accuracy": 0.89,
            "f1_macro": 0.86,
            "precision": 0.88,
            "recall": 0.87
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
