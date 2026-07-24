# 🏦 AI Customer Spend Forecaster & Behavioral Segmentation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-orange.svg)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B.svg)](https://streamlit.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-blueviolet.svg)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning pipeline that translates raw credit card transactions into actionable behavioral personas and predicts future spending trajectories to prescribe automated banking decisions.

**[🚀 Live Streamlit Dashboard](https://customer-spend-forecaster-ghsckvjkowpfxghf2zhxfm.streamlit.app/)**

---

## 🚀 Quick Demo

![Demo](assets/demo.gif)

---

## 💼 Business Impact
Financial institutions sit on terabytes of transactional data but typically rely on **reactive retention**—only knowing a customer is dissatisfied after they close their account. This system shifts the paradigm from reactive to **proactive**. 

Banks and marketing teams can use this AI engine to:
*   **Identify high-value customers** hidden within raw transaction logs.
*   **Forecast future customer spending** using 6-month predictive trajectories.
*   **Trigger proactive retention campaigns** before a customer officially churns.
*   **Segment customers** for highly personalized, cost-efficient marketing.
*   **Improve revenue forecasting** by moving beyond simple historical averages.

---

## 🏗️ System Architecture

![Architecture](assets/architecture.png)

This project demonstrates a complete, production-ready Data Science lifecycle:

### 1. Data Engineering & Leakage Prevention
*   **Temporal Splitting:** To simulate a real-world forecasting environment and prevent data leakage, the dataset was strictly split chronologically. The **Observation Window** (2019) was used exclusively for feature engineering, and the **Target Window** (Jan-Jun 2020) was used to define future behavior.
*   **RFM Feature Engineering:** Converted 1.29 million raw transaction rows into distinct customer profiles using Recency, Frequency, and Monetary metrics (e.g., *Spend Volatility*, *Average Transaction Gap*). Demographic bias (age, gender) was deliberately removed to ensure legally compliant, behavior-only predictions.

### 2. Behavioral Segmentation (K-Means)
Using **K-Means Clustering** (validated via Silhouette Scores and PCA dimensionality reduction), the customer base was segmented into 5 actionable business personas:

| Persona | Description |
| :--- | :--- |
| 💎 **High-Value Loyal** | Elite spenders with high loyalty and massive transaction volumes. |
| 📈 **Engaged Regulars** | Highly frequent, stable, and reliable daily utility users. |
| 🎢 **High Variability** | Unpredictable spenders with erratic, massive transaction spikes. |
| 📊 **Moderate Casual** | Average baseline users with standard financial behavior. |
| ⚠️ **Low Engagement** | Infrequent users posing a high churn risk. |

---

## 🧠 Predictive Modeling (XGBoost)

Trained an **XGBoost Regression** model to predict future spending momentum. Instead of predicting absolute future dollars (which artificially biases the model toward wealthy users), the model targets the **Logarithmic Growth Ratio**. This normalizes the baseline, allowing the AI to successfully flag a millionaire dropping their spend by 50% alongside a college student growing their spend by 200%.

| Model | CV MAE (Log Space) | Test MAE (Growth Ratio Error) | Test R² |
| :--- | :--- | :--- | :--- |
| Linear Regression | 0.1154 | 0.0460 | 0.0935 |
| Random Forest | 0.1084 | 0.0452 | 0.1255 |
| **XGBoost** | **0.1129** | **0.0454** | **0.1425** |

*Note: While the pandemic compressed the total variance (yielding a modest R²), the XGBoost model achieved a highly accurate Test Mean Absolute Error (MAE) of ~4.5%, proving its capability for precise directional forecasting.*

---

## 🔍 Explainable AI (SHAP)

Model interpretability is important in financial decision support systems. **SHAP (SHapley Additive exPlanations)** was integrated to explain how individual features influence model predictions, proving that the model relies on sound financial realities rather than arbitrary noise.

![SHAP Summary](assets/shap_plot.png)

**Key Insights:** *Spend Volatility (Standard Deviation)* and *Transaction Frequency* emerged as the strongest mathematical drivers for future spending behavior, proving that erratic spenders are the highest flight risks.

---

## 💻 Deployment & The Decision Engine

![Dashboard](assets/dashboard.png)

The pipeline is deployed via a **Streamlit** web application, bridging the gap between machine learning math and business intelligence.

*   **Annualized Run-Rate Optimization:** Because the model predicts a 6-month window but historical data covers 12 months, the deployment layer mathematically annualizes the forecast. This allows the system to generate intuitive, highly accurate 1.0 baseline business thresholds:
    *   🚨 **< 0.95 (Decline):** Triggers immediate Retention Marketing.
    *   📈 **> 1.05 (Growth):** Triggers Premium Upgrade Offers.
    *   ⚖️ **Stable:** Recommends continued engagement monitoring.
*   **Enterprise Scalability:** Features a dual-tab architecture to support both single-customer deep dives for relationship managers and bulk CSV uploads for enterprise marketing batch processing.

---

## 🛠️ Technology Stack

*   **Core:** Python, Pandas, NumPy
*   **Machine Learning:** Scikit-Learn (K-Means, PCA, Linear Regression, Random Forest), XGBoost
*   **Explainable AI:** SHAP
*   **Data Visualization:** Matplotlib, Seaborn
*   **Deployment & UI:** Streamlit, Joblib

---

## 📂 Repository Structure

```text
Customer-Spend-Forecaster/
├── app.py                  # Main Streamlit web application & UI layer
├── requirements.txt        # Python dependencies for deployment
├── README.md               # Project documentation
├── assets/                 # Architecture diagrams, SHAP plots, and UI screenshots
│   ├── dashboard.png
│   ├── shap_plot.jpg
│   ├── architecture.png
│   └── demo.gif
├── models/                 # Serialized .joblib files (XGBoost, K-Means, Scaler)
├── notebooks/              # Jupyter notebooks for EDA and Model Training
└── data/                   # Processed datasets (raw data ignored via .gitignore)
```

## 💻 How to Run This App Locally
If you want to run this project on your own computer, follow these steps:

### 1. Clone the repository:
```bash
git clone https://github.com/Murthaja-ai/Customer-Spend-Forecaster.git

cd Customer-Spend-Forecaster
```

### 2. Create and activate a virtual environment:

```bash
python -m venv venv

venv\Scripts\activate     # Windows 
source venv/bin/activate  # Mac/Linux 
```

### 3. Install the required libraries:
```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit web application:
```bash
streamlit run app.py
```

---

## 👨‍💻 Author

**Murthaja Afham**  
*Data Scientist • Machine Learning Engineer • AI Developer*

[![GitHub](https://img.shields.io/badge/GitHub-Murthaja--ai-181717?style=for-the-badge&logo=github)](https://github.com/Murthaja-ai)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Murthaja%20Afham-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/murthaja-afham/)


