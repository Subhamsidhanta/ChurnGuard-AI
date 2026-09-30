# ChurnGuard AI

## Customer Churn Prediction and Retention Recommendation System

## 1. Executive Summary

Subscription businesses lose revenue when customers leave without warning. This project builds an early-warning churn system that predicts customer churn risk, explains the main risk drivers, and recommends a practical retention action for each customer.

The solution combines synthetic CRM data generation, exploratory analysis, feature engineering, supervised machine learning, model evaluation, customer-level reason codes, retention recommendations, an API, and a Streamlit dashboard.

## 2. Business Problem

Customer Success teams need to answer three questions:

1. Which customers are most likely to churn?
2. Why is each customer at risk?
3. Which retention intervention should be offered?

The project treats churn prediction as a ranking and prioritization problem. A probability score helps teams focus limited outreach capacity on customers with high churn risk and high revenue exposure.

## 3. Dataset

The notebook generates 20,000 synthetic customers with the following CRM fields:

- Customer identity and signup date
- Tenure, plan, contract, and payment method
- Monthly and cumulative charges
- Auto-pay and payment failures
- Support tickets and resolution time
- Satisfaction and product engagement
- Login recency and usage behavior
- Discounts, price increases, and referrals
- Binary churn target

The generated data includes missing values, billing outliers, moderate class imbalance, and seasonal churn variation near billing-cycle boundaries. The expected churn rate is approximately 15% to 25%.

## 4. Analytical Approach

### Exploratory Data Analysis

The notebook examines:

- Churn class balance
- Missing values
- Churn by contract type
- Churn by plan
- Churn by payment method
- Customer support and engagement patterns

### Feature Engineering

Business-oriented variables are created:

- `tenure_bucket`
- `charges_per_month`
- `usage_trend`
- `engagement_score`
- `support_load`
- `payment_reliability`
- `recency_score`
- `loyalty_score`

These features make the model easier to interpret and connect predictions to customer-success actions.

### Preprocessing

Numeric variables are median-imputed and standardized. Categorical variables are most-frequent-imputed and one-hot encoded. The preprocessing is implemented inside a scikit-learn pipeline to avoid training-serving inconsistencies.

### Models

The notebook trains and compares:

- Logistic Regression with balanced class weights
- Random Forest with balanced class weights

The production project also supports optional XGBoost and LightGBM models with a scikit-learn fallback.

Stratified cross-validation is used to preserve the churn ratio in each fold. ROC-AUC is the main ranking metric, while recall is monitored because failing to identify a likely churner can be expensive.

## 5. Evaluation

The notebook reports:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Classification report
- Confusion matrix
- ROC curve
- Precision-recall curve
- Decile analysis
- Lift chart

Decile analysis is especially useful for operations. It shows whether the highest-ranked 10% of customers contains a substantially higher churn concentration than the overall customer population.

## 6. Explainability

Logistic-regression coefficients are converted into directional reason codes. Positive contributions indicate increased churn likelihood; negative contributions indicate reduced churn likelihood.

Example churn drivers include:

- Monthly contract
- Long time since last login
- Low satisfaction
- High support burden
- Payment failures
- Recent price increase
- Low feature usage

The production application also includes SHAP-compatible explanations for tree-based models, with a safe fallback when SHAP is unavailable.

## 7. Retention Recommendation Engine

The recommendation logic maps risk and customer profile to an intervention:

- Low risk: product education and nurture
- Medium risk: limited-time discount and plan review
- High risk: proactive outreach and loyalty discount
- Low satisfaction or high ticket volume: priority support callback
- Monthly contract: discounted annual-plan migration
- Payment failures: payment-method assistance

The recommendation is intended to support a human Customer Success decision rather than replace it.

## 8. Business Value and ROI

The notebook estimates campaign economics by:

1. Selecting the highest-risk 10% of customers.
2. Estimating expected churners from predicted probabilities.
3. Applying a configurable retention success rate.
4. Estimating revenue saved over six months.
5. Subtracting campaign cost.
6. Calculating net value and ROI multiple.

The assumptions are explicit and should be replaced with historical campaign conversion and offer-cost data in production.

## 9. Application Architecture

The project includes:

- Data generation module
- Feature engineering module
- Preprocessing module
- Model training and evaluation
- Explainability module
- Retention recommendation engine
- FastAPI prediction endpoint
- Streamlit early-warning dashboard

FastAPI endpoints:

- `GET /health`
- `POST /predict`
- `GET /docs`

The Streamlit dashboard includes Overview, Customer Scoring, Retention, Explainability, and Model Performance sections.

## 10. Limitations and Future Work

- The dataset is synthetic and should be replaced or calibrated with real historical CRM data.
- Probability calibration should be checked before financial decisions.
- Retention offers should be evaluated with controlled experiments.
- Fairness and segment-level performance should be monitored.
- Data drift and campaign feedback should be tracked over time.
- A causal uplift model could identify customers who are persuadable, not only customers who are likely to churn.

## 11. Conclusion

This project demonstrates a complete, business-oriented machine-learning workflow. It does not stop at a churn probability: it connects risk scoring to explanations, prioritized alerts, retention actions, and measurable revenue impact.
