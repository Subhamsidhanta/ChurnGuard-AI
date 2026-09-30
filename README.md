# ChurnGuard AI

Customer Churn Prediction and Retention Recommendation System

End-to-end reference implementation with synthetic data, feature engineering, a scikit-learn churn pipeline, optional SHAP explanations, business retention recommendations, FastAPI inference, and a five-page Streamlit dashboard.

## Quick start

```bash
pip install -r requirements.txt
python -m src.data_generation
python -m src.train
python -m src.evaluate
uvicorn src.api:app --reload
# in another terminal:
streamlit run app/app.py
```

POST a JSON customer (the fields in `src/api.py`) to `http://localhost:8000/predict`. The response contains churn probability, risk level, estimated value, recommended actions, and top feature impacts. `/docs` provides interactive OpenAPI documentation. Set `CHURN_MODEL=xgboost` or `CHURN_MODEL=lightgbm` to use an installed optional booster; unavailable optional packages automatically fall back to scikit-learn.

## Project layout

`data/churn_data.csv` is the generated source dataset. `src/data_generation.py`, `src/train.py`, `src/evaluate.py`, `src/explain.py`, and `src/api.py` are the requested canonical entry points; existing modules provide reusable internals. Training writes both `models/churn_model.pkl` and `models/churn_pipeline.joblib`, plus metrics. `app/app.py` is the requested Streamlit entry point and `app/dashboard.py` remains available. The workflow is deterministic (seed 42), handles missing values and unseen categories, and includes optional XGBoost/LightGBM/imbalanced-learn dependencies.

The generated CSV has exactly 21 columns: customer identity and signup/tenure, plan and billing fields, engagement/support/experience signals, retention signals, and `churn`. It contains 20,000 rows by default, a seeded 15–25% churn rate, seasonal variation, missing values, and a small number of billing outliers.
