"""Canonical API entry point; run with: uvicorn src.api:app --reload."""
from app.api import app, Customer, predict, health

__all__ = ["app", "Customer", "predict", "health"]
