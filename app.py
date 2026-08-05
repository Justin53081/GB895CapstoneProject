#Modified from the example used in class
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

#page setup
st.set_page_config(
    page_title="Churn Model",
    page_icon="🌀",
    initial_sidebar_state="expanded"
)

# Define the overlay_input_on_figure function
def overlay_input_on_figure(fig_obj, field_name, input_value):
  """
  Overlays an input value as a vertical line on an existing Matplotlib Figure.

  Args:
    fig_obj (matplotlib.figure.Figure): The existing figure object to modify.
    field_name (str): The name of the field, used for context in labels/title.
    input_value: The value to plot as an overlay.

  Returns:
    matplotlib.figure.Figure: The modified figure object.
  """
  ax = fig_obj.get_axes()[0]

  # Remove ALL existing vertical lines or scatter points from the axes to ensure only one input is shown.
  for artist in list(ax.collections) + list(ax.lines):
      # Check if it's a scatter plot (PathCollection) or a line (Line2D), and remove it
      if isinstance(artist, plt.matplotlib.collections.PathCollection) or isinstance(artist, plt.Line2D):
          artist.remove()

  # Plot the input_value as a vertical line.
  ax.axvline(x=input_value, color='red', linestyle='--', linewidth=2, label=f'Current Input: {input_value}')

  # Update the legend to include the new input marker
  ax.legend()

  return fig_obj # Return the modified figure object directly
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
        with open("pickled_artifacts/histograms_with_figs.pkl", "rb") as f:
            histograms_dict = pickle.load(f)
        return histograms_dict
    except FileNotFoundError:
        st.error("Error: pickled_artifacts/histograms_with_figs.pkl not found. Please ensure it's in the correct path.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading histograms: {e}")
        st.stop()

histograms_data = load_histograms_data()
