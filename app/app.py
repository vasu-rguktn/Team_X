import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys
import os
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from utils.feature_generator import generate_single_features, PRIMARY_USE_MAP, CITY_MAP, METER_MAP

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    with open("../models/xgboost_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Building Energy Predictor",
    page_icon="⚡",
    layout="centered"
)

st.title("⚡ Building Energy Consumption Predictor")
st.write("Predict energy usage based on building characteristics.")

st.divider()

# ----------------------------
# User Inputs
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    building_type = st.selectbox("Building Type", list(PRIMARY_USE_MAP.keys()))

with col2:
    city = st.selectbox("City", list(CITY_MAP.keys()))

area = st.number_input(
    "Building Area (sq ft)",
    min_value=500,
    value=50000,
    step=500
)

meter_type = st.selectbox("Meter Type", list(METER_MAP.keys()))

now = datetime.now()

st.caption(
    f"Using current time automatically: {now.strftime('%d %B %Y')} at {now.hour}:00"
)

# ----------------------------
# Prediction
# ----------------------------
if st.button("⚡ Predict Energy Consumption"):

    with st.spinner("Generating features and predicting..."):

        input_df = generate_single_features(
            building_type,
            area,
            city,
            now.date(),
            now.hour,
            meter_type
        )

        pred_log = model.predict(input_df)[0]
        pred_kwh = np.expm1(pred_log)

    st.success("Prediction Complete")

    st.metric("Predicted Energy Consumption", f"{pred_kwh:,.2f} kWh")

    with st.expander("See Model Features"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}))