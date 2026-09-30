"""Generate the canonical 20,000-row churn dataset."""
from pathlib import Path
import numpy as np
import pandas as pd

SCHEMA = [
    "customer_id", "signup_date", "tenure_months", "plan_type",
    "monthly_charges", "total_charges", "contract_type", "payment_method",
    "auto_pay", "num_support_tickets_30d", "avg_resolution_time_hours",
    "satisfaction_score", "login_count_30d", "session_minutes_30d",
    "feature_usage_score", "days_since_last_login", "discount_applied",
    "price_increase_last_3m", "num_payment_failures_6m", "referral_count", "churn",
]


def generate_customers(n: int = 20_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    signup = pd.Timestamp("2018-01-01") + pd.to_timedelta(rng.integers(0, 2920, n), unit="D")
    tenure = np.maximum(1, ((pd.Timestamp("2026-01-01") - signup).days // 30
                             + rng.integers(-2, 3, n))).astype(int)
    plan = rng.choice(["Basic", "Standard", "Premium"], n, p=[.35, .45, .20])
    contract = rng.choice(["Monthly", "Yearly"], n, p=[.62, .38])
    payment = rng.choice(["Card", "UPI", "Bank", "Wallet"], n, p=[.42, .25, .20, .13])
    monthly = np.clip(rng.normal({"Basic": 35, "Standard": 65, "Premium": 105}.get("Basic"), 15, n), 15, 180)
    monthly += (plan == "Standard") * 30 + (plan == "Premium") * 70
    season = 1 + .08 * np.sin(2 * np.pi * signup.month.to_numpy() / 12)
    monthly = np.clip(monthly * season, 15, 220).round(2)
    total = np.maximum(0, monthly * tenure * rng.uniform(.88, 1.12, n))
    auto_pay = rng.binomial(1, .68, n)
    tickets = rng.poisson(1.4, n)
    resolution = np.clip(rng.gamma(2.2, 8, n), 1, 96).round(1)
    satisfaction = np.clip(rng.normal(3.6, .75, n), 1, 5).round(1)
    login = np.clip(rng.poisson(13 * season, n), 0, 60)
    minutes = np.clip(login * rng.normal(22, 7, n), 0, 1800).round(1)
    usage = np.clip(rng.normal(62, 20, n), 0, 100).round(1)
    last_login = np.clip(rng.exponential(12, n), 0, 120).round().astype(int)
    discount = rng.binomial(1, .24, n)
    increase = rng.binomial(1, .12, n)
    failures = rng.poisson(.35 + .25 * (1 - auto_pay), n)
    referrals = rng.poisson(1.1, n)
    logit = (-1.25 + .95 * (contract == "Monthly") + .28 * (payment == "Wallet")
             + .26 * tickets + .012 * resolution - .75 * (satisfaction - 3)
             - .035 * login - .018 * usage + .028 * last_login + .65 * increase
             + .30 * failures - .18 * auto_pay - .12 * discount - .08 * referrals
             + .22 * (plan == "Premium"))
    probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, probability)
    frame = pd.DataFrame({
        "customer_id": [f"C{i:07d}" for i in range(n)],
        "signup_date": signup.strftime("%Y-%m-%d"),
        "tenure_months": tenure, "plan_type": plan,
        "monthly_charges": monthly, "total_charges": total.round(2),
        "contract_type": contract, "payment_method": payment, "auto_pay": auto_pay,
        "num_support_tickets_30d": tickets, "avg_resolution_time_hours": resolution,
        "satisfaction_score": satisfaction, "login_count_30d": login,
        "session_minutes_30d": minutes, "feature_usage_score": usage,
        "days_since_last_login": last_login, "discount_applied": discount,
        "price_increase_last_3m": increase, "num_payment_failures_6m": failures,
        "referral_count": referrals, "churn": churn,
    })
    # Realistic data quality issues, while retaining the exact column schema.
    for col in ["monthly_charges", "avg_resolution_time_hours", "satisfaction_score", "feature_usage_score"]:
        frame.loc[rng.random(n) < .01, col] = np.nan
    frame.loc[rng.choice(n, size=max(1, n // 200), replace=False), "monthly_charges"] *= 3
    return frame[SCHEMA]


if __name__ == "__main__":
    output = Path(__file__).with_name("churn_data.csv")
    output.parent.mkdir(exist_ok=True)
    generate_customers().to_csv(output, index=False)
    print(f"Wrote {output}")
