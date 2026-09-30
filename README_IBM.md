# ChurnGuard AI

Customer Churn Prediction and Retention Recommendation System

This repository contains an IBM project submission notebook and an end-to-end churn prediction application.

## Main submission file

Open and run:

```text
main.ipynb
```

The notebook is self-contained and covers:

- Synthetic CRM data generation
- Exploratory data analysis
- Missing-value and outlier checks
- Feature engineering
- Logistic Regression and Random Forest
- Stratified cross-validation
- Accuracy, precision, recall, F1, ROC-AUC, and PR-AUC
- Confusion matrix, ROC curve, PR curve, decile lift
- Feature importance and churn reason codes
- Personalized retention offers
- Revenue-at-risk and ROI estimation

## Local application setup

```bash
pip install -r requirements.txt
python -m src.data_generation
python -m src.train
python -m src.evaluate
```

Run the API:

```bash
uvicorn src.api:app --reload
```

Run the dashboard:

```bash
streamlit run app/app.py
```

## API endpoints

- `GET /health`
- `POST /predict`
- `GET /docs`

## Project artifacts

- `main.ipynb`: IBM submission notebook
- `PROJECT_REPORT.pdf`: project report for IBM submission
- `requirements.txt`: Python dependencies
- `data/churn_data.csv`: generated customer dataset
- `models/churn_model.pkl`: persisted model artifact
