# 🏦 AI Customer Spend Forecaster & Segmentation Engine

An interactive Machine Learning web application designed to forecast customer credit card spending and group users into actionable business personas.

## 📊 The Business Problem
Banks and financial institutions sit on terabytes of transactional data. The challenge is moving from looking at the past to predicting the future. This project solves this by:
1. **Identifying Personas:** Grouping customers based on their engagement and financial behavior.
2. **Forecasting Revenue:** Accurately predicting a customer's spending growth for the upcoming year.
3. **Automating Decisions:** Triggering automated retention marketing alerts when a high-value customer is predicted to drop their spending.

## 🏗️ Technical Architecture
This project demonstrates an end-to-end Machine Learning pipeline:
* **Data & Features:** Engineered RFM (Recency, Frequency, Monetary) features from raw transactions without data leakage.
* **Unsupervised Learning (K-Means):** Clustered users into distinct business personas (e.g., "High-Value Loyal", "Engaged Regulars").
* **Supervised Learning (XGBoost):** Trained a gradient-boosted regression tree to predict future spend ratios.
* **Web Deployment (Streamlit):** Wrapped the models into a live, interactive web application with mathematical validation guardrails.

## 🧠 Dynamic Business Personas
The unsupervised clustering engine dynamically maps customers into one of five categories:
* **High-Value Loyal** - Elite spenders with high loyalty and massive transaction volumes.
* **High Variability** - Erratic spenders with massive spikes in transaction amounts.
* **Engaged Regulars** - Highly frequent, stable, and reliable daily users.
* **Moderate Casual** - Average users with occasional, moderate-value transactions.
* **Low Engagement** - Infrequent users with lower overall spend and engagement.

## 💻 How to Run This App Locally
If you want to run this project on your own computer, follow these steps:

1. **Clone the repository:**
   git clone https://github.com/YOUR-GITHUB-USERNAME/Customer-Spend-Forecaster.git
   cd Customer-Spend-Forecaster

2. **Install the required libraries:**
   pip install -r requirements.txt

3. **Launch the web app:**
   streamlit run app.py

---
*Developed by [MURTHAJA AFHAM] to demonstrate a complete, production-ready Machine Learning workflow.*