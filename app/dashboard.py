"""Streamlit five-page dashboard. Run: streamlit run app/dashboard.py."""
from pathlib import Path
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from src.feature_engineering import add_features
from src.retention_engine import recommend_retention

st.set_page_config(page_title="ChurnGuard AI", page_icon="📉", layout="wide")
st.sidebar.title("ChurnGuard AI")
page = st.sidebar.radio("Navigate", ["Overview", "Customer Scoring", "Retention", "Explainability", "Model Performance"])

@st.cache_data
def data():
    path = Path("data/churn_data.csv")
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

@st.cache_resource
def bundle():
    path = Path("models/churn_model.pkl")
    if not path.exists():
        from src.train import train_model
        train_model(model_path=str(path))
    return joblib.load(path)

df = data()
if page == "Overview":
    st.title("ChurnGuard AI - Customer Churn Overview")
    if df.empty: st.warning("Run data/generate_data.py to populate the dataset.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Customers", len(df)); c2.metric("Churn rate", f"{df.churn.mean():.1%}")
        c3.metric("Monthly revenue at risk", f"${df.loc[df.churn == 1, 'monthly_charges'].sum():,.0f}")
        contract_summary = (
            df.groupby("contract_type", as_index=False)["churn"]
            .mean()
            .rename(columns={"churn": "churn_rate"})
        )
        figure = px.bar(
            contract_summary,
            x="contract_type",
            y="churn_rate",
            title="Churn rate by contract type",
            labels={"churn_rate": "Churn rate", "contract_type": "Contract"},
        )
        figure.update_yaxes(tickformat=".0%")
        st.plotly_chart(figure, use_container_width=True)
elif page == "Customer Scoring":
    st.title("Score a Customer")
    source = df.drop(columns=["churn"], errors="ignore")
    idx = st.number_input("Customer row", 0, max(0, len(source)-1), 0)
    if len(source):
        customer = source.iloc[int(idx)].to_dict()
        probability = float(bundle()["pipeline"].predict_proba(add_features(pd.DataFrame([customer]))[bundle()["features"]])[0, 1])
        st.metric("Churn probability", f"{probability:.1%}")
        st.json(recommend_retention(customer, probability))
elif page == "Retention":
    st.title("Retention Actions")
    st.info("Prioritize high-risk customers for proactive service recovery and contract migration.")
    if not df.empty: st.dataframe(df.sort_values("monthly_charges", ascending=False).head(25), use_container_width=True)
elif page == "Explainability":
    st.title("Model Explainability")
    st.write("The API returns top SHAP feature impacts for every prediction. Use this view to inspect feature distributions.")
    if not df.empty:
        figure = px.scatter(
            df,
            x="tenure_months",
            y="monthly_charges",
            color="churn",
            title="Tenure and monthly charges",
            hover_data=["customer_id", "contract_type", "satisfaction_score"],
        )
        st.plotly_chart(figure, use_container_width=True)
else:
    st.title("Model Performance")
    st.json(bundle().get("metrics", {}))
