"""Evaluate a persisted churn model on a labelled CSV."""
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from .feature_engineering import add_features


def evaluate(model_path="models/churn_model.pkl", data_path="data/churn_data.csv"):
    bundle = joblib.load(model_path)
    frame = add_features(pd.read_csv(data_path))
    x, y = frame[bundle["features"]], frame["churn"]
    probability = bundle["pipeline"].predict_proba(x)[:, 1]
    result = {"roc_auc": float(roc_auc_score(y, probability)),
              "report": classification_report(y, probability >= .5, output_dict=True)}
    Path(model_path).with_suffix(".metrics.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
