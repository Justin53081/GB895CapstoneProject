#Modified from the example used in class
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns # Added for sns.histplot

#page setup
st.set_page_config(
    page_title="Churn Model",
    page_icon="🌀",
    initial_sidebar_state="expanded"
)

# Define the overlay_input_on_figure function (UPDATED)
def overlay_input_on_figure(histograms_data, field_name, input_value):
  """
  Generates a new plot for a given field from stored data and overlays an input value.

  Args:
    histograms_data (dict): The dictionary containing plot data and metadata.
    field_name (str): The name of the field to plot.
    input_value: The value to plot as an overlay.

  Returns:
    matplotlib.figure.Figure: A newly generated figure object with the overlay.
  """
  # Retrieve data and plot type from the loaded histograms_data
  plot_info = histograms_data[field_name]
  data_series = plot_info['data']
  plot_type = plot_info['plot_type'] # This will now consistently be 'histplot'

  fig, ax = plt.subplots(figsize=(8, 4)) # Create a new figure and axes each time

  # Determine appropriate bins for the histogram
  if data_series.nunique() <= 2 and data_series.min() >= 0 and data_series.max() <= 1:
      # Likely a binary feature (0 or 1), set bins to 2 for clear representation
      bins_param = 2
  else:
      # For other numeric/categorical features, let seaborn decide or use 'auto'
      bins_param = 'auto'

  # Always use histplot as per user's instruction
  sns.histplot(data_series, ax=ax, kde=False, bins=bins_param) # kde=False to remove the density line, added bins
  ax.set_xlabel(field_name)
  ax.set_title(f'Distribution of {field_name} (Churned Customers)')

  # Add vertical line for the input value
  ax.axvline(x=input_value, color='red', linestyle='--', linewidth=2, label=f'Current Customer: {input_value}')

  ax.legend()
  fig.tight_layout() # Adjust layout to prevent labels from overlapping
  return fig

# Load model and encoder once at startup (cached so they don't reload on every interaction)
@st.cache_resource
def load_artifacts():
    with open("xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, encoder

model, encoder = load_artifacts()

@st.cache_resource
def load_histograms_data():
    try:
        # The pickled file now contains data series, stats, and plot_type, not Figure objects
        with open("histograms_with_figs.pkl", "rb") as f:
            histograms_dict = pickle.load(f)
        return histograms_dict
    except FileNotFoundError:
        st.error("Error: histograms_with_figs.pkl not found. Please ensure it's in the correct path.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading histograms: {e}")
        st.stop()

histograms_data = load_histograms_data()

# ── UI ────────────────────────────────────────────────────────────────────────

st.title("Customer Churn Probability Predictor")
st.write("Enter customer attributes to predict the likelihood of subscription renewal.")

st.header("Customer Demographics")

age               = st.number_input("Age", min_value=18, max_value=100, value=35)
income_level      = st.radio("Income Level",  ["Low", "Medium", "High", "Very High"])
education         = st.radio("Education",     ["Graduate", "High School", "Other", "Post-Graduate"])
device_type       = st.radio("Device Type",   ["Desktop-only", "Mobile-only", "Multi-device"])
tech_comfort_score = st.slider("Tech Comfort Score", min_value=1, max_value=5, value=3)

st.header("Product Engagement Metrics")

num_active_days = st.number_input("Number of Active Days", min_value=0, max_value=365, value=0)
num_active_qtrs = st.number_input("Number of Active Quarters", min_value=0, max_value=4, value=2)
total_session_length = st.number_input("Total Session Length", min_value=0, max_value=15601, value=1800)
total_num_sessions = st.number_input("Total Number of Sessions", min_value=0, max_value=262, value=40)
avg_sessions_per_qtr = st.number_input("Average Sessions per Active Quarter", min_value=0, max_value=500, value=150)

st.header("Product Ownership")

num_products_owned = st.slider("Number of Products Owned", min_value=0, max_value=5, value=3)
num_active_products_owned = st.slider("Number of Active Products Owned", min_value=0, max_value=num_products_owned, value=3)
has_healthy_meals = int(st.toggle("Has Healthy Meals Subscription"))
has_daily_fitness = int(st.toggle("Has Daily Fitness Subscription"))
has_wellness_tracker = int(st.toggle("Has Wellness Tracker Subscription"))
has_mindful_living = int(st.toggle("Has Mindful Living Subscription"))
has_premium_health = int(st.toggle("Has Premium Health Subscription"))

# --- Create input_df BEFORE the Predict button so it's always available ---
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


# --- Prediction Logic (only runs when button is clicked) ---
if st.button("Predict", type="primary"):
    # Column 1 = P(renewed), column 0 = P(churned)
    probability = 1-model.predict_proba(input_df)[0][1]
    risk = "Low" if probability <= 0.33 else "Medium" if probability <= 0.66 else "High"

    st.metric("Churn Probability", f"{probability:.2%}")
    if risk == "High":
        st.error(f"**Churn Risk: {risk}**\n\nTop Three Reasons for High Churn Risk: \n\n  * Device Type is Multi Device \n\n  * Low Tech Comfort Score \n\n  * Low Number of Products Owned", icon="🔴")

    elif risk == "Medium":
        st.warning(f"**Churn Risk: {risk}**\n\nTop Three Reasons for High Churn Risk:\n\n  * Device Type is Multi Device\n\n  * Low Tech Comfort Score\n\n  * Low Number of Products Owned", icon="🟡")
    else:
        st.success(f"**Churn Risk: {risk}**", icon="🟢")

# --- Visualization Logic (always runs) ---
st.header("Customer Explorer")
selected_viz_feature = st.selectbox(
"Select Visual",
('TOTAL_NUM_SESSIONS', 'TOTAL_SESSION_LENGTH', 'ACTIVE_DAYS', 'ACTIVE_PRODUCTS', 'ACTIVE_QUARTERS', 'AVG_SESSIONS_PER_ACTIVE_QUARTER','AGE', 'TECH_COMFORT_SCORE', 'PRODUCTS_OWNED', 'HAS_HEALTHY_MEALS', 'HAS_DAILY_FITNESS', 'HAS_WELLNESS_TRACKER', 'HAS_MINDFUL_LIVING', 'HAS_PREMIUM_HEALTH'),
key='visual_feature_selector' # Added a unique key to preserve state
)
st.subheader(f"Distribution of '{selected_viz_feature}' for Churned Customers with Your Input")

user_input_for_viz = input_df[selected_viz_feature].iloc[0]
if user_input_for_viz is not None:
    # Call the overlay function, which now generates a new figure each time
    # Pass histograms_data directly as the first argument
    modified_plot_fig = overlay_input_on_figure(histograms_data, selected_viz_feature, user_input_for_viz)
    st.pyplot(modified_plot_fig)
    plt.close(modified_plot_fig) # Close the figure to free up memory after displaying
else:
    st.warning(f"Could not find an input value for '{selected_viz_feature}' to overlay.")
