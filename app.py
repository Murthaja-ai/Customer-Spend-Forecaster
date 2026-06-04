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
st.sidebar.write("**Target:** Forecast Customer Spend Change")
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

# --- CREATE TWO TABS FOR THE UI ---
tab1, tab2 = st.tabs(["👤 Single Customer Analysis", "📁 Bulk Data Upload"])

with tab1:
    st.write("Enter a customer's behavioral data below to predict their future spending and segment them into a business persona.")

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

    if st.button("🔮 Predict Future Spend", use_container_width=True): 
        
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

        # --- THE BUSINESS FIX: ANNUALIZED MATH ---
        log_ratio = xgb_model.predict(final_data)[0]
        real_ratio = np.exp(log_ratio)
        future_spend_6_months = real_ratio * monetary_sum
        
        estimated_annual_spend = future_spend_6_months * 2
        annualized_ratio = estimated_annual_spend / monetary_sum
        annualized_growth_pct = (annualized_ratio - 1.0) * 100

        st.divider()
        st.subheader("🧠 AI Insights & Financial Forecast")
        
        if freq > 2500 or monetary_sum > 250000:
            st.warning("⚠️ **Note:** The entered data exceeds the historical training limits. The model is extrapolating, so predictions may carry higher variance.")

        st.info(f"**Identified Persona: [{persona_name}]** — {persona_descriptions.get(persona_name, '')}")
        
        # Updated to 3 columns to show the full financial breakdown
        met1, met2, met3 = st.columns(3)
        met1.metric("Predicted 6-Month Ratio", f"{real_ratio:.2f}x")
        met2.metric("Forecasted 6-Month Spend", f"${future_spend_6_months:,.2f}")
        met3.metric("Estimated Annual Spend", f"${estimated_annual_spend:,.2f}", f"{annualized_growth_pct:+.1f}%")
        st.caption(
            "Annualized estimate assumes the predicted next 6-month spending pattern continues for a full year."
        )

        # Business rules now perfectly aligned with the 1.0 baseline
        if annualized_ratio < 0.95:
            st.error("🚨 **DECISION:** Annualized spend is dropping. Trigger retention marketing immediately!")
        elif annualized_ratio > 1.05:
            st.success("📈 **DECISION:** Annualized spend is growing. Offer premium upgrades!")
        else:
            st.warning("⚖️ **DECISION:** Annualized spending is predicted to remain stable. Monitor engagement.")

        st.caption("ℹ️ *Note: Predictions are directional estimates based on historical behavioral data. Not financial advice.*")

with tab2:
    st.write("Upload a CSV of customer transaction data to generate batch predictions.")
    uploaded_file = st.file_uploader("Upload Customer CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(df)} customers.")
            
            if st.button("Run Batch Prediction"):
                with st.spinner("Processing data through ML Pipeline..."):
                    st.write("Preview of uploaded data:")
                    st.dataframe(df.head())
                    st.info("Future enhancement: uploaded customer records will be processed through the same Scaler → K-Means → XGBoost pipeline, producing downloadable customer forecasts and retention recommendations.")
        except Exception as e:
            st.error(f"Error reading file: {e}")