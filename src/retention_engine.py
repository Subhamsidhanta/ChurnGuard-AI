"""Business rules turning churn risk into actionable interventions."""
def recommend_retention(customer: dict, probability: float) -> dict:
    actions = []
    if probability >= .70:
        priority = "high"
        actions.append({"action": "proactive outreach", "offer": "20% loyalty discount", "channel": "phone"})
    elif probability >= .40:
        priority = "medium"
        actions.append({"action": "send retention offer", "offer": "10% discount for 6 months", "channel": "email"})
    else:
        priority = "low"
        actions.append({"action": "relationship nurture", "offer": "service tips and upgrade options", "channel": "email"})
    if customer.get("satisfaction_score", 5) <= 2.5 or customer.get("num_support_tickets_30d", 0) >= 4:
        actions.insert(0, {"action": "service recovery", "offer": "priority support callback", "channel": "customer success"})
    if customer.get("contract_type") == "Monthly":
        actions.append({"action": "contract migration", "offer": "discounted annual plan", "channel": "account manager"})
    return {"risk_level": priority, "actions": actions, "estimated_value": round(
        float(customer.get("monthly_charges", 0)) * 6 * probability, 2)}
