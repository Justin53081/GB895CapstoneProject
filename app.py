#Modified from the example used in class
import streamlit as st
import numpy as np
import pandas as pd
import pickle

#page setup
st.set_page_config(
    page_title="Churn Model",
    page_icon="🌀",
    initial_sidebar_state="expanded"
)

# Load model and encoder once at startup (cached so they don't reload on every interaction)
@st.cache_resource
def load_artifacts():
    with open("xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, encoder

model, encoder = load_artifacts()

# ── UI ────────────────────────────────────────────────────────────────────────

st.title("Customer Renewal Probability Predictor")
st.write("Enter customer attributes to predict the likelihood of subscription renewal.")

age               = st.number_input("Age", min_value=18, max_value=100, value=35)
income_level      = st.radio("Income Level",  ["Low", "Medium", "High", "Very High"])
education         = st.radio("Education",     ["Graduate", "High School", "Other", "Post-Graduate"])
device_type       = st.radio("Device Type",   ["Desktop-only", "Mobile-only", "Multi-device"])
tech_comfort_score = st.slider("Tech Comfort Score", min_value=1, max_value=5, value=3)
num_active_days = st.number_input("Number of Active Days", min_value=0, max_value=365, value=0)
num_active_qtrs = st.number_input("Number of Active Quarters", min_value=0, max_value=4, value=2)
total_session_length = st.number_input("Total Session Length", min_value=0, max_value=15601, value=1800)
total_num_sessions = st.number_input("Total Number of Sessions", min_value=0, max_value=262, value=40)
avg_sessions_per_qtr = st.number_input("Average Sessions per Active Quarter", min_value=0, max_value=500, value=150) 
num_products_owned = st.slider("Number of Products Owned", min_value=0, max_value=5, value=3)
num_active_products_owned = st.slider("Number of Active Products Owned", min_value=0, max_value=5, value=3)
has_healthy_meals = int(st.toggle("Has Healthy Meals Subscription"))
has_daily_fitness = int(st.toggle("Has Daily Fitness Subscription"))
has_wellness_tracker = int(st.toggle("Has Wellness Tracker Subscription"))
has_mindful_living = int(st.toggle("Has Mindful Living Subscription"))
has_premium_health = int(st.toggle("Has Premium Health Subscription"))


if st.button("Predict", type="primary"):

    # Build categorical DataFrame — column names and must match encoder exactly
    raw = pd.DataFrame([{
        'INCOME_LEVEL': income_level,
        'EDUCATION':    education,
        'DEVICE_TYPE':  device_type,
    }])

    # Apply the saved encoder (transform only — never fit_transform)
    encoded = encoder.transform(raw)
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

    # Numeric features first, then encoded dummies — must match training column order
    numeric_df = pd.DataFrame([{
        'TOTAL_NUM_SESSIONS': total_num_sessions,
        'TOTAL_SESSION_LENGTH': total_session_length,
        'ACTIVE_DAYS': num_active_days,
        'ACTIVE_PRODUCTS': num_active_products_owned,
        'ACTIVE_QUARTERS': num_active_qtrs,
        'AVG_SESSIONS_PER_ACTIVE_QUARTER': avg_sessions_per_qtr,
        'AGE': age,
        'TECH_COMFORT_SCORE': tech_comfort_score,
        'PRODUCTS_OWNED': num_products_owned,
        'HAS_HEALTHY_MEALS': has_healthy_meals,
        'HAS_DAILY_FITNESS': has_daily_fitness,
        'HAS_WELLNESS_TRACKER': has_wellness_tracker,
        'HAS_MINDFUL_LIVING': has_mindful_living,
        'HAS_PREMIUM_HEALTH': has_premium_health,
        
    }])

    input_df = pd.concat([numeric_df, encoded_df], axis=1)

    # Column 1 = P(renewed), column 0 = P(churned)
    probability = model.predict_proba(input_df)[0][1]
    risk = "Low" if probability >= 0.6 else "Medium" if probability >= 0.4 else "High"

    st.metric("Renewal Probability", f"{probability:.2f}")
    if risk == "High":
        st.error(f"Churn Risk: {risk}", icon="🔴")
    elif risk == "Medium":
        st.warning(f"Churn Risk: {risk}", icon="🟡")
    else:
        st.success(f"Churn Risk: {risk}", icon="🟢")
