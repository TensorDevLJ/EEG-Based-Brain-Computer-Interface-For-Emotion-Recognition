# EEG Depression Detection Backend

## 🎯 Overview

Complete backend API for EEG-based emotion and depression detection using Transformer models.

**Key Features:**
- Accepts CSV files with pre-extracted EEG features
- Runs Transformer inference for 5-class classification
- Generates 16 diagnostic graphs (base64 encoded)
- Provides SHAP-based + rule-based explanations
- Returns depression stage (Not Depressed / Mild / Moderate / Moderately Severe / Severe)

---

## 📁 Project Structure

```
backend/
├── api.py                    # FastAPI server (main endpoints)
├── model_transformer.py      # Transformer model architecture
├── features.py              # Feature preprocessing
├── plots.py                 # All 16 graph generators
├── explainers.py            # SHAP + rule-based explanations
├── trainer.py               # Training pipeline (optional)
├── requirements.txt         # Python dependencies
├── saved_models/
│   └── best_model.pth      # Trained model checkpoint
└── README.md               # This file
```

---

## 🚀 Installation & Setup

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Train Model (Optional - if no pre-trained model)

```bash
python trainer.py --data your_training_data.csv --epochs 50
```

### 3. Start API Server

```bash
python api.py
```

Or with uvicorn:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Server runs at: **http://localhost:8000**

---

## 📡 API Endpoints

### 1. **POST /predict** (Main Endpoint)

Accepts CSV file, returns predictions + 16 graphs + explanations

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@your_eeg_data.csv"
```

**Response:**
```json
{
  "status": "success",
  "predicted_class": "Moderate",
  "depression_stage": "Moderate",
  "probabilities": {
    "Not Depressed": 0.05,
    "Mild": 0.12,
    "Moderate": 0.65,
    "Moderately Severe": 0.15,
    "Severe": 0.03
  },
  "depression_index": 0.62,
  "confidence": 0.65,
  "top_features": [
    {
      "feature": "lag1_mean_0",
      "value": 25.8,
      "importance": 0.23,
      "interpretation": "High"
    }
  ],
  "explanation_text": "The model predicts Moderate depression...",
  "dominant_wave": "Theta",
  "graphs": {
    "graph_1": "data:image/png;base64,iVBORw0KG...",
    "graph_2": "data:image/png;base64,iVBORw0KG...",
    ...
    "graph_16": "data:image/png;base64,iVBORw0KG..."
  },
  "metadata": {
    "n_samples": 1,
    "n_features": 1000,
    "model_version": "v1.0.0"
  }
}
```

### 2. **POST /analyze**

Returns analysis graphs without prediction (faster)

### 3. **GET /metrics**

Returns model performance metrics

```json
{
  "model_version": "v1.0.0",
  "model_type": "Transformer",
  "num_classes": 5,
  "classes": ["Not Depressed", "Mild", "Moderate", "Moderately Severe", "Severe"],
  "metrics": {
    "accuracy": 0.89,
    "f1_macro": 0.86
  }
}
```

---

## 🖼️ 16 Generated Graphs

| # | Graph Name | Description |
|---|------------|-------------|
| 1 | Raw EEG Signal | Time-series visualization of raw signal |
| 2 | Filtered EEG Signal | After bandpass filtering (0.5-45 Hz) |
| 3 | Power Spectral Density | Frequency distribution (Welch method) |
| 4 | Band Powers | Delta, Theta, Alpha, Beta, Gamma bars |
| 5 | Feature Correlation Heatmap | Feature relationships |
| 6 | PCA Variance | Cumulative explained variance |
| 7 | Scalogram (CWT) | Time-frequency analysis |
| 8-9 | Training Curves | Accuracy & Loss over epochs |
| 10 | Confusion Matrix | Classification accuracy per class |
| 11 | ROC Curve | Multi-class ROC with AUC |
| 12 | Emotion Probabilities | Output confidence per class |
| 13 | Brain Connectivity | Coherence/PLV matrix |
| 14 | Band Power per Emotion | Comparative band power distribution |
| 15 | Feature Importance | Top 20 SHAP features |
| 16 | Attention Map | Transformer attention weights |

All graphs returned as **base64-encoded PNG** strings.

---

## 🔌 Frontend Integration

### React/TypeScript Example

```typescript
// Upload CSV and get predictions
async function predictDepression(file: File) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: formData
  });

  const data = await response.json();
  
  return {
    stage: data.predicted_class,
    confidence: data.confidence,
    depressionIndex: data.depression_index,
    graphs: data.graphs,
    explanation: data.explanation_text
  };
}

// Display graphs in React
function GraphDisplay({ graphs }) {
  return (
    <div className="grid grid-cols-2 gap-4">
      {Object.entries(graphs).map(([name, base64]) => (
        <div key={name}>
          <h3>{name.replace('_', ' ').toUpperCase()}</h3>
          <img src={base64} alt={name} />
        </div>
      ))}
    </div>
  );
}
```

### Plain JavaScript Example

```javascript
const fileInput = document.getElementById('eeg-file');
const form = document.getElementById('upload-form');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  
  const formData = new FormData();
  formData.append('file', fileInput.files[0]);
  
  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: formData
  });
  
  const result = await response.json();
  
  // Display result
  document.getElementById('stage').textContent = result.predicted_class;
  document.getElementById('confidence').textContent = 
    `${(result.confidence * 100).toFixed(1)}%`;
  
  // Display graphs
  Object.entries(result.graphs).forEach(([name, base64]) => {
    const img = document.createElement('img');
    img.src = base64;
    img.alt = name;
    document.getElementById('graphs-container').appendChild(img);
  });
});
```

---

## 📊 Input CSV Format

Your CSV should contain pre-extracted features with columns like:

```
lag1_mean_0, lag1_mean_1, lag1_std_0, lag1_std_1, ..., freq_750_3, Label
25.8, 33.8, 12.5, 3.36, ..., 0.000188, 2.0
```

**Notes:**
- `Label` column (optional) - will be ignored during prediction
- All numeric columns are used as features
- Missing values replaced with 0
- Features are automatically standardized

---

## ⚙️ Configuration

Edit `api.py` to configure:

```python
# Model path
MODEL_PATH = 'saved_models/best_model.pth'

# Number of classes
NUM_CLASSES = 5

# Class names
CLASS_NAMES = ['Not Depressed', 'Mild', 'Moderate', 
               'Moderately Severe', 'Severe']

# Feature dimension (adjust based on your CSV)
INPUT_DIM = 1000
```

---

## 🧪 Testing

### Test with cURL

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@test_data.csv" \
  -o response.json
```

### Test with Python

```python
import requests

with open('test_eeg.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f}
    )

result = response.json()
print(f"Prediction: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

---

## 🔧 Troubleshooting

### Model Not Found
```bash
# Create dummy model for testing
python -c "from model_transformer import EEGTransformer, save_model; \
model = EEGTransformer(input_dim=1000); \
save_model(model, 'saved_models/best_model.pth')"
```

### Port Already in Use
```bash
# Use different port
uvicorn api:app --port 8001
```

### CORS Issues
Already configured in `api.py`. If issues persist, update:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📈 Performance

- **Inference time:** ~1-3 seconds per CSV file
- **Graph generation:** ~2-4 seconds (all 16 graphs)
- **Total response time:** ~3-7 seconds
- **Memory usage:** ~500MB (with loaded model)

---

## 🚢 Deployment

### Docker (Recommended)

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t eeg-backend .
docker run -p 8000:8000 eeg-backend
```

### Production Server

```bash
# Install gunicorn
pip install gunicorn

# Run with workers
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## 📞 Support

For issues or questions:
1. Check console logs: `journalctl -u eeg-api -f`
2. Verify model loaded: `curl http://localhost:8000/`
3. Test with sample data first

---

## 📄 License

MIT License - Free to use for research and commercial applications.
