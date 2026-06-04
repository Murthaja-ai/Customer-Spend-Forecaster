import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Layout and Title
st.set_page_config(page_title="AI Financial Forecaster", layout="wide", page_icon="🏦")

# --- THE BUSINESS-FRIENDLY SIDEBAR ---
st.sidebar.title("📊 Project Info")
st.sidebar.write("**Architecture:** K-Means + XGBoost")
st.sidebar.write("**Dataset:** Behavioral Credit Card Transactions")
st.sidebar.write("**Target:** Forecast Customer Spend Change") # Fixed jargon!
st.sidebar.divider()
st.sidebar.caption("Built as an end-to-end machine learning forecasting system combining behavioral segmentation (K-Means) and future spend prediction (XGBoost).")

@st.cache_resource
def load_models():
    scaler = joblib.load('models/clv_scaler.joblib')
    kmeans = joblib.load('models/clv_kmeans_model.joblib')
    persona_map = joblib.load('models/persona_map.joblib')
    xgb = joblib.load('models/future_spend_xgboost_model.joblib')
    features = joblib.load('models/predictive_feature_columns.joblib')
    return scaler, kmeans, persona_map, xgb, features

try:
    scaler, kmeans_model, persona_map, xgb_model, expected_features = load_models()
except Exception as e:
    st.error(f"❌ Model initialization failed. Error: {e}")
    st.stop()

st.title("🏦 AI Customer Spend Forecaster")
st.write("Enter a customer's behavioral data below to predict their 2020 spending and segment them into a business persona.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Time & Engagement")
    duration = st.number_input("Account Age (Days)", min_value=1, value=360)
    avg_gap = st.number_input("Avg Days Between Swipes", min_value=0.0, value=0.3)
    freq = st.number_input("Total Transactions (Frequency)", min_value=1, value=1200)

with col2:
    st.subheader("Financial Behavior")
    monetary_sum = st.number_input("Total Spend ($)", min_value=0.0, value=75000.0)
    monetary_mean = st.number_input("Average Transaction ($)", min_value=0.0, value=62.5)
    monetary_max = st.number_input("Max Single Transaction ($)", min_value=0.0, value=2500.0)
    monetary_std = st.number_input("Spend Volatility ($ Std Dev)", min_value=0.0, value=150.0)

persona_descriptions = {
    "High-Value Loyal": "Elite spenders with high loyalty and massive transaction volumes.",
    "High Variability": "Erratic spenders with massive spikes in transaction amounts.",
    "Engaged Regulars": "Highly frequent, stable, and reliable daily users.",
    "Moderate Casual": "Average users with occasional, moderate-value transactions.",
    "Low Engagement": "Infrequent users with lower overall spend and engagement."
}

# --- NEUTRAL BUTTON COLOR ---
if st.button("🔮 Predict Future Spend", type="primary",use_container_width=True): # Removed type="primary"
    
    # Validation Checks
    if monetary_mean > monetary_sum:
        st.error("❌ Data Error: Average transaction cannot exceed Total Spend.")
        st.stop()
    if monetary_max < monetary_mean:
        st.error("❌ Data Error: Max transaction must be greater than or equal to Average transaction.")
        st.stop()
    if freq * monetary_mean > monetary_sum * 10:
        st.warning("⚠️ Data Consistency Warning: The combination of Frequency and Average Transaction seems highly unusual compared to Total Spend. Please verify.")

    new_customer = pd.DataFrame([{
        'active_duration_days': duration,
        'avg_days_between_swipes': avg_gap,
        'frequency': freq,
        'monetary_sum': monetary_sum,
        'monetary_mean': monetary_mean,
        'monetary_max': monetary_max,
        'monetary_std': monetary_std
    }])

    scaled_data = scaler.transform(new_customer)
    cluster_id = kmeans_model.predict(scaled_data)[0]
    persona_name = persona_map[cluster_id]

    final_data = pd.DataFrame(0.0, index=[0], columns=expected_features)
    for col in new_customer.columns:
        if col in expected_features:
            final_data.at[0, col] = new_customer.at[0, col]
            
    persona_col = f'Seg_{persona_name}'
    if persona_col in expected_features:
        final_data.at[0, persona_col] = 1.0

    log_ratio = xgb_model.predict(final_data)[0]
    real_ratio = np.exp(log_ratio)
    growth_pct = (real_ratio - 1.0) * 100
    future_spend = real_ratio * monetary_sum

    st.divider()
    st.subheader("🧠 AI Insights")
    
    if freq > 2500 or monetary_sum > 250000:
        st.warning("⚠️ **Note:** The entered data exceeds the historical training limits. The model is extrapolating, so predictions may carry higher variance.")

    st.info(f"**Identified Persona: [{persona_name}]** — {persona_descriptions.get(persona_name, '')}")
    
    met1, met2 = st.columns(2)
    met1.metric("Predicted Growth Ratio", f"{real_ratio:.2f}x", f"{growth_pct:+.1f}%")
    met2.metric("Forecasted 2020 Spend", f"${future_spend:,.2f}")

    if real_ratio < 0.95:
        st.error("🚨 **DECISION:** This customer is predicted to drop their spending. Trigger retention marketing immediately!")
    elif real_ratio > 1.05:
        st.success("📈 **DECISION:** This customer is predicted to grow their spend. Offer premium upgrades!")
    else:
        st.warning("⚖️ **DECISION:** This customer's spending is predicted to remain stable. Monitor engagement.")

    st.caption("ℹ️ *Note: Predictions are directional estimates based on historical behavioral data. Not financial advice.*")