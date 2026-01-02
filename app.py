import streamlit as st
import pandas as pd
import joblib

# Load artifacts
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

CLUSTER_INFO = {
    0: ("Budget-Conscious Browsers",
        "Low-spending, price-sensitive users.",
        "Discounts & retargeting"),
    1: ("Elite Omnichannel Spenders",
        "High-income, cross-channel buyers.",
        "VIP & premium offers"),
    2: ("Window Shoppers",
        "High traffic, low conversion.",
        "UX & nudges"),
    3: ("Loyal High-Value Customers",
        "Consistent spenders, strong loyalty.",
        "Retention & cross-sell"),
    4: ("Traditional Premium Buyers",
        "Prefer high-value in-store purchases.",
        "In-store personalization"),
    5: ("Occasional Low-Spend Customers",
        "Infrequent, low-value buyers.",
        "Reactivation campaigns")
}

# Page config
st.set_page_config(page_title="Customer Segmentation", layout="centered")

st.title("PersonaIQ")
st.caption("Predict customer persona using behavioral data")

st.divider()

# ---------------------------
# Compact Input Layout
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 30)
    income = st.number_input("Income", 0, 200000, 50000)
    web_purchases = st.number_input("Web Purchases", 0, 100, 10)
    web_visits = st.number_input("Web Visits / Month", 0, 100, 5)

with col2:
    spending = st.number_input("Total Spending", 0, 5000, 1000)
    store_purchases = st.number_input("Store Purchases", 0, 100, 10)
    recency = st.number_input("Recency (Days)", 0, 365, 30)

# Input dataframe
input_df = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "Total_Spending": [spending],
    "NumWebPurchases": [web_purchases],
    "NumStorePurchases": [store_purchases],
    "NumWebVisitsMonth": [web_visits],
    "Recency": [recency]
})

st.divider()

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Segment"):
    scaled_input = scaler.transform(input_df)
    cluster_id = kmeans.predict(scaled_input)[0]

    name, desc, action = CLUSTER_INFO[cluster_id]

    st.success(f"Segment: {name}")

    st.markdown(
        f"""
        **Overview:** {desc}  
        **Action:** {action}
        """
    )
