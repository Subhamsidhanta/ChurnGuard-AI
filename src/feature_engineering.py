"""Feature engineering shared by training and inference."""
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "signup_date" in result:
        dates = pd.to_datetime(result["signup_date"], errors="coerce")
        result["signup_year"] = dates.dt.year
        result["signup_month"] = dates.dt.month
        result = result.drop(columns=["signup_date"])
    result["avg_monthly_value"] = result["total_charges"] / result["tenure_months"].clip(lower=1)
    result["support_rate_30d"] = result["num_support_tickets_30d"] / result["tenure_months"].clip(lower=1)
    result["payment_failure_rate"] = result["num_payment_failures_6m"] / 6
    result["tenure_bucket"] = pd.cut(
        result["tenure_months"], bins=[-1, 6, 12, 24, 48, float("inf")],
        labels=["0-6", "7-12", "13-24", "25-48", "49+"],
    ).astype("object")
    result["charges_per_month"] = result["total_charges"] / result["tenure_months"].clip(lower=1)
    result["usage_trend"] = (
        result["login_count_30d"] * 0.4
        + result["session_minutes_30d"] * 0.01
        + result["feature_usage_score"] * 0.6
    )
    result["engagement_score"] = (
        result["login_count_30d"].clip(upper=30) / 30 * 40
        + result["session_minutes_30d"].clip(upper=1200) / 1200 * 30
        + result["feature_usage_score"] * 0.3
    )
    result["support_load"] = result["num_support_tickets_30d"] * result["avg_resolution_time_hours"]
    result["payment_reliability"] = 1 / (1 + result["num_payment_failures_6m"])
    result["recency_score"] = 1 / (1 + result["days_since_last_login"])
    result["loyalty_score"] = (
        result["tenure_months"].clip(upper=60) / 60 * 0.7
        + result["referral_count"].clip(upper=10) / 10 * 0.3
    )
    result["is_new_customer"] = (result["tenure_months"] <= 6).astype(int)
    result["high_value"] = (result["monthly_charges"] >= 90).astype(int)
    result["low_satisfaction"] = (result["satisfaction_score"] <= 2.5).astype(int)
    return result
