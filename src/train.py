"""Train, evaluate, and persist the churn model."""
from pathlib import Path
import json
import os
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from .feature_engineering import add_features
from .preprocessing import build_preprocessor


def _classifier():
    """Prefer an optional boosted tree when explicitly requested; always has a fallback."""
    requested = os.getenv("CHURN_MODEL", "hist").lower()
    if requested == "xgboost":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=180, max_depth=4, learning_rate=.06,
                                 subsample=.85, colsample_bytree=.85, eval_metric="logloss",
                                 random_state=42)
        except ImportError:
            pass
    if requested == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(n_estimators=180, learning_rate=.06, num_leaves=31,
                                  random_state=42, verbosity=-1)
        except ImportError:
            pass
    return HistGradientBoostingClassifier(max_iter=150, random_state=42)


def train_model(data_path="data/churn_data.csv", model_path="models/churn_pipeline.joblib",
                legacy_model_path="models/churn_model.pkl"):
    df = pd.read_csv(data_path) if Path(data_path).exists() else __import__(
        "src.data_generation", fromlist=["generate_and_save"]).generate_and_save(data_path)
    df = add_features(df)
    target = "churn"
    drop = ["customer_id", target]
    features = [c for c in df.columns if c not in drop]
    categorical = df[features].select_dtypes(include="object").columns.tolist()
    numeric = [c for c in features if c not in categorical]
    x_train, x_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=.2, random_state=42, stratify=df[target])
    pipeline = Pipeline([("preprocessor", build_preprocessor(categorical, numeric)),
                         ("model", _classifier())])
    pipeline.fit(x_train, y_train)
    proba = pipeline.predict_proba(x_test)[:, 1]
    metrics = {"roc_auc": float(roc_auc_score(y_test, proba)),
               "report": classification_report(y_test, proba >= .5, output_dict=True)}
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "features": features, "metrics": metrics}, model_path)
    if legacy_model_path:
        joblib.dump({"pipeline": pipeline, "features": features, "metrics": metrics}, legacy_model_path)
    Path(model_path).with_suffix(".json").write_text(json.dumps(metrics, indent=2))
    return pipeline, metrics


if __name__ == "__main__":
    train_model()
