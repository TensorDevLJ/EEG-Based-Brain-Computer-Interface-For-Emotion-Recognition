"""Backend package initializer.

This file makes the `backend` folder a Python package so relative imports
in `api.py` work when the app is started via `uvicorn backend.api:app`.
"""

__all__ = ["api", "model_transformer", "features", "plots", "explainers"]
