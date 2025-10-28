# EEG Depression Detection (FastAPI backend)

This repository contains a FastAPI backend for EEG-based emotion and depression detection. The backend accepts CSV files with pre-extracted features, runs inference with a Transformer model, generates diagnostic graphs, and returns explanations.

This README explains how to set up a local development environment (Windows / PowerShell), install dependencies, run the FastAPI server, and test the `/predict` endpoint.

## Prerequisites

- Python 3.11 installed and available on PATH (or provide the path to a Python executable)
- pip (comes with Python)
- (Optional) GPU and CUDA for PyTorch if you plan to run model inference on GPU

## Recommended: create and use a virtual environment

Open PowerShell and run the following from the project root (`C:\Users\Tarun\Desktop\project`):

```powershell
# create venv
python -m venv .venv

# activate venv
.\.venv\Scripts\Activate.ps1

# upgrade pip
python -m pip install --upgrade pip
```

If your system blocks scripts, you can enable running local scripts for the session:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

## Install Python dependencies

Install the backend requirements (file located at `backend/requirements.txt`):

```powershell
cd C:\Users\Tarun\Desktop\project\backend
pip install -r requirements.txt
```

Notes:
- The `requirements.txt` file includes packages such as `fastapi`, `uvicorn`, `numpy`, `pandas`, `torch`, `scikit-learn`, and more.
- If you need a GPU-enabled PyTorch, install the matching `torch` wheel from https://pytorch.org/ instead of the CPU-only wheel.

## Running the FastAPI server (development)

From the `backend` directory run uvicorn and serve the `app` module (this ensures local imports resolve correctly):

```powershell
cd C:\Users\Tarun\Desktop\project\backend
# Run in foreground (use Ctrl+C to stop)
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload

# Or start detached (background)
Start-Process -NoNewWindow -WorkingDirectory 'C:\Users\Tarun\Desktop\project\backend' -FilePath python -ArgumentList '-m','uvicorn','app:app','--host','127.0.0.1','--port','8000'
```

Open the API docs in your browser: http://127.0.0.1:8000/docs

If you need the server to be accessible from other machines, bind to `0.0.0.0` instead of `127.0.0.1` and ensure firewall rules permit incoming traffic. On Windows you may need to run PowerShell as Administrator to bind to 0.0.0.0 without restrictions.

## Using the `/predict` endpoint

- The `/predict` endpoint expects a CSV file upload containing pre-extracted features. The server will attempt to load `backend/saved_models/best_model.pth` on startup. If this checkpoint is missing or invalid, the app falls back to a dummy model for testing.
- Example using curl (from PowerShell or another shell):

```powershell
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@path\to\your\features.csv"
```

Or use the interactive Swagger UI at `/docs` to upload a file and test.

Common error seen during development:
- "Prediction failed: The size of tensor a (2479) must match the size of tensor b (100) at non-singleton dimension 0"
	- This indicates a mismatch between expected model input dimensions and the features provided. The `app.py` includes padding/trim logic to adapt feature length to `input_dim=1000`, but other tensors (e.g., positional encodings or attention buffers) may need adjustment. If you hit this error, ensure your CSV produces the correct-sized feature vector expected by the model, or contact the maintainer to adjust the model's expected input length.

## Model checkpoint

- The server looks for a checkpoint at `backend/saved_models/best_model.pth`.
- If you have a trained PyTorch checkpoint, place it there. The checkpoint should contain the model state dict and, optionally, a metadata dict with `scaler` and `feature_stats` used by preprocessing.

## Development notes

- Main FastAPI entrypoint: `backend/app.py` (module name `app` when run from `backend/` directory).
- Model and preprocessing code: `backend/model_transformer.py`, `backend/features.py`.
- Plotting and explanation helpers: `backend/plots.py`, `backend/explainers.py`.
- If you modify Python code, the `--reload` flag for uvicorn will auto-restart the server.

## Troubleshooting

- If uvicorn fails to bind to a port, try using `127.0.0.1` instead of `0.0.0.0`. A WinError 10013 indicates a socket permission issue.
- If local imports fail (ModuleNotFoundError), ensure you run uvicorn with the working directory set to `backend` or use the full module path from repo root: `python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000`.

## Optional: start/stop helper scripts

You can add a small PowerShell script `backend\run_server.ps1` to encapsulate the Start-Process command and a `stop_server.ps1` to kill the PID. If you'd like, I can add these in a follow-up.

## Contact / Next steps

- If you want, I can:
	- Add start/stop scripts to `backend/`.
	- Validate and repair the model input shape handling inside `model_transformer.py` and `app.py` (I already drafted a plan to adjust positional encodings dynamically).
	- Add a small test CSV and a lightweight unit test for the `/predict` endpoint to avoid shape errors.

---

Generated on 2025-10-26.

## Run frontend and backend together (one command)

To make development simpler you can run both the backend and frontend at once from the project root using the included `package.json` script. This uses `npx concurrently` so you don't need to install `concurrently` globally.

From the project root run (PowerShell):

```powershell
# install frontend deps first (only required once)
cd C:\Users\Tarun\Desktop\project\frontend
npm install

# from project root, run both servers together
cd C:\Users\Tarun\Desktop\project
npm run dev
```

What this does:
- Starts the FastAPI backend: `python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000`
- Starts the Vite dev server for the React frontend at `http://localhost:8080` and proxies API calls under `/api` to the backend

If you prefer to run servers separately, use the instructions in the sections above.

