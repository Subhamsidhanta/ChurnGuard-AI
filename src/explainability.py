"""Optional SHAP explanations with a model-agnostic fallback."""
import numpy as np


def explain_prediction(bundle, customer: dict, top_n: int = 5) -> list[dict]:
    pipeline, features = bundle["pipeline"], bundle["features"]
    import pandas as pd
    row = pd.DataFrame([customer]).reindex(columns=features)
    probability = float(pipeline.predict_proba(row)[0, 1])
    try:
        import shap
        transformed = pipeline.named_steps["preprocessor"].transform(row)
        model = pipeline.named_steps["model"]
        values = shap.TreeExplainer(model)(transformed).values
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[0, :, 1]
        else:
            values = values[0]
        names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        order = np.argsort(np.abs(values))[::-1][:top_n]
        return [{"feature": str(names[i]), "impact": float(values[i])} for i in order]
    except Exception:
        return [{"feature": "predicted_probability", "impact": probability}]

