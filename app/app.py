# app/app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import sys
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date

# ── Path Setup ────────────────────────────────────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "utils"))

from utils.feature_generator import (
    generate_single_features,
    predict_monthly,
    get_coordinates
)
from utils.feature_labels import (
    PRIMARY_USE_MAP, METER_MAP,
    PRIMARY_USE_LABELS, METER_LABELS,
    get_feature_label, get_value_label
)
from utils.recommendations import generate_recommendations
from utils.summary import generate_full_report

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title = "Building Energy Predictor",
    page_icon  = "⚡",
    layout     = "wide"
)

# ── Styling ───────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0f1117; }
.main-title {
    font-size: 2.5rem;
    font-weight: 800;
    color: #00d4aa;
    text-align: center;
    margin-bottom: 0.2rem;
}
.sub-title {
    font-size: 1rem;
    color: #888;
    text-align: center;
    margin-bottom: 2rem;
}
.section-header {
    font-size: 1.2rem;
    font-weight: 700;
    color: #00d4aa;
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
}
.metric-card {
    background: #1e2130;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    border: 1px solid #2d3148;
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #00d4aa;
}
.metric-label {
    font-size: 0.85rem;
    color: #888;
    margin-top: 0.2rem;
}
.rec-card {
    background: #1e2130;
    border-left: 4px solid #00d4aa;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.8rem;
}
.rec-title {
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.3rem;
}
.rec-saving {
    color: #00d4aa;
    font-size: 0.9rem;
    font-weight: 600;
}
.summary-box {
    background: #1e2130;
    border-radius: 12px;
    padding: 1.2rem;
    color: #cccccc;
    font-size: 0.95rem;
    line-height: 1.7;
    border: 1px solid #2d3148;
}
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = os.path.join(project_root, "models", "xgb_time_model.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

model     = load_model()
explainer = load_explainer(model)

# ── Header ────────────────────────────────────────────────────
st.markdown('<div class="main-title">⚡ Building Energy Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Explainable AI-powered energy audit system for buildings worldwide</div>', unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🏢 Quick Predict", "📂 CSV Upload"])

# ════════════════════════════════════════════════════════════════
# TAB 1 — QUICK PREDICT
# ════════════════════════════════════════════════════════════════
with tab1:

    st.markdown('<div class="section-header">Building Information</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        building_type = st.selectbox(
            "Building Type",
            list(PRIMARY_USE_MAP.keys()),
            index = 6
        )
        area = st.number_input(
            "Building Area (sq ft)",
            min_value = 500,
            max_value = 5000000,
            value     = 50000,
            step      = 500
        )

    with col2:
        city_name = st.text_input(
            "City",
            value       = "Hyderabad",
            placeholder = "Type any city in the world..."
        )
        meter_type = st.selectbox(
            "Energy Meter Type",
            list(METER_MAP.keys())
        )

    st.markdown('<div class="section-header">Prediction Mode</div>', unsafe_allow_html=True)

    mode = st.radio(
        "Select prediction period",
        ["Current Hour", "Monthly Report"],
        horizontal = True
    )

    if mode == "Monthly Report":
        col3, col4 = st.columns(2)
        with col3:
            pred_month = st.selectbox(
                "Month",
                list(range(1, 13)),
                format_func = lambda x: datetime(2000, x, 1).strftime("%B"),
                index       = datetime.now().month - 1
            )
        with col4:
            pred_year = st.selectbox(
                "Year",
                list(range(2023, 2027)),
                index = 2
            )
    else:
        pred_month = datetime.now().month
        pred_year  = datetime.now().year
        now        = datetime.now()
        st.caption(f"Using current time: {now.strftime('%d %B %Y')} at {now.hour}:00")

    st.divider()

    predict_btn = st.button("⚡ Predict Energy Consumption", width="stretch")

    # ── Run Prediction ────────────────────────────────────────
    if predict_btn:

        if not city_name.strip():
            st.error("Please enter a city name.")
            st.stop()

        with st.spinner("Looking up city coordinates..."):
            coords = get_coordinates(city_name)

        if not coords:
            st.error(f"City '{city_name}' not found. Please check the spelling.")
            st.stop()

        lat, lon, display_name = coords
        st.caption(f"📍 Location found: {display_name.split(',')[0]}")

        with st.spinner("Fetching weather and generating prediction..."):

            if mode == "Current Hour":
                now      = datetime.now()
                input_df = generate_single_features(
                    building_type = building_type,
                    area          = area,
                    city_name     = city_name,
                    target_date   = now.date(),
                    hour          = now.hour,
                    meter_type    = meter_type
                )
                pred_log   = model.predict(input_df)[0]
                pred_kwh   = float(np.expm1(pred_log))
                shap_vals  = explainer.shap_values(input_df)[0]
                peak_row   = input_df
                site_id    = int(input_df["site_id"].iloc[0])
                result_obj = None

            else:
                result_obj = predict_monthly(
                    building_type = building_type,
                    area          = area,
                    city_name     = city_name,
                    year          = pred_year,
                    month         = pred_month,
                    meter_type    = meter_type,
                    model         = model
                )
                pred_kwh  = result_obj["monthly_total_kwh"]
                peak_row  = result_obj["peak_row"]
                shap_vals = explainer.shap_values(peak_row)[0]
                site_id   = result_obj["site_id"]

        feature_names  = peak_row.columns.tolist()
        feature_values = peak_row.iloc[0].tolist()

        recs = generate_recommendations(
            shap_vals, feature_names, feature_values
        )

        report = generate_full_report(
            prediction_kwh     = pred_kwh,
            building_type_code = PRIMARY_USE_MAP[building_type],
            area               = area,
            site_id            = site_id,
            meter_code         = METER_MAP[meter_type],
            recommendations    = recs,
            period             = "monthly" if mode == "Monthly Report" else "hourly",
            month              = pred_month,
            year               = pred_year
        )

        # ── Store in session state ────────────────────────────
        st.session_state["prediction_done"]  = True
        st.session_state["result"]           = result_obj
        st.session_state["report"]           = report
        st.session_state["recs"]             = recs
        st.session_state["shap_vals"]        = shap_vals
        st.session_state["feature_names"]    = feature_names
        st.session_state["feature_values"]   = feature_values
        st.session_state["sim_done"]         = False
        st.session_state["inputs"] = {
            "building_type": building_type,
            "area":          area,
            "city_name":     city_name,
            "meter_type":    meter_type,
            "mode":          mode,
            "month":         pred_month,
            "year":          pred_year
        }

    # ── Show Results — runs every rerun if prediction exists ──
    if st.session_state.get("prediction_done"):

        report     = st.session_state["report"]
        recs       = st.session_state["recs"]
        result_obj = st.session_state["result"]
        inputs     = st.session_state["inputs"]
        mode_saved = inputs["mode"]
        pred_kwh   = report["prediction_kwh"]

        total_saving_min = sum(r["saving_min"] for r in recs)
        total_saving_max = sum(r["saving_max"] for r in recs)
        saving_kwh       = pred_kwh * (total_saving_max / 100)

        # ── Section 1 — Metric Cards ──────────────────────────
        st.markdown('<div class="section-header">📊 Prediction Result</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{pred_kwh:,.0f}</div>
                <div class="metric-label">kWh {"this month" if mode_saved == "Monthly Report" else "this hour"}</div>
            </div>""", unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_saving_min}–{total_saving_max}%</div>
                <div class="metric-label">Potential Saving</div>
            </div>""", unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{saving_kwh:,.0f}</div>
                <div class="metric-label">kWh Saveable</div>
            </div>""", unsafe_allow_html=True)

        # ── Section 2 — Daily Chart ───────────────────────────
        if mode_saved == "Monthly Report" and result_obj:

            st.markdown('<div class="section-header">📈 Daily Energy Breakdown</div>', unsafe_allow_html=True)

            daily        = result_obj["daily_breakdown"]
            peak_day_num = int(result_obj["peak_day"]["day"])
            avg_kwh      = daily["total_kwh"].mean()
            colors       = ["#ff4b4b" if d == peak_day_num else "#00d4aa" for d in daily["day"]]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x            = daily["day"],
                y            = daily["total_kwh"],
                marker_color = colors,
                name         = "Daily kWh"
            ))
            fig.add_hline(
                y               = avg_kwh,
                line_dash       = "dash",
                line_color      = "#ffaa00",
                annotation_text = f"Avg: {avg_kwh:,.0f} kWh"
            )
            fig.update_layout(
                plot_bgcolor  = "#1e2130",
                paper_bgcolor = "#1e2130",
                font_color    = "#cccccc",
                xaxis_title   = "Day of Month",
                yaxis_title   = "Energy (kWh)",
                showlegend    = False,
                height        = 350
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(f"🔴 Peak day: Day {peak_day_num} — {result_obj['peak_day']['total_kwh']:,.0f} kWh")

        # ── Section 3 — Recommendations ──────────────────────
        st.markdown('<div class="section-header">💡 Top Energy Saving Recommendations</div>', unsafe_allow_html=True)

        for i, rec in enumerate(recs, 1):
            st.markdown(f"""
            <div class="rec-card">
                <div class="rec-title">{i}. {rec['feature_label']}: {rec['feature_value']}</div>
                <div style="color:#cccccc; font-size:0.9rem; margin:0.3rem 0;">
                    {rec['advice']}
                </div>
                <div class="rec-saving">
                    💰 Estimated saving: {rec['saving_min']}–{rec['saving_max']}%
                </div>
            </div>""", unsafe_allow_html=True)

        # ── Section 4 — Summary ───────────────────────────────
        st.markdown('<div class="section-header">📝 Summary</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="summary-box">
            {report["summary_text"]}
        </div>""", unsafe_allow_html=True)

        # ── Analytics Link ────────────────────────────────────
        st.divider()
        st.markdown("### 🔍 Want deeper insights?")
        st.markdown("See before vs after simulation, SHAP waterfall and full energy audit report.")

        if st.button("📊 View Detailed Analytics →", width="stretch"):
            st.switch_page("pages/analytics.py")


# ════════════════════════════════════════════════════════════════
# TAB 2 — CSV UPLOAD
# ════════════════════════════════════════════════════════════════
with tab2:

    st.markdown('<div class="section-header">📂 Batch Building Prediction</div>', unsafe_allow_html=True)
    st.write("Upload a CSV to predict energy consumption for multiple buildings at once.")

    st.markdown("**Required CSV columns:** `building_type, area_sqft, city, meter_type, year, month`")

    sample_data = pd.DataFrame({
        "building_type": ["Office", "Education", "Healthcare"],
        "area_sqft":     [50000, 30000, 80000],
        "city":          ["Hyderabad", "Mumbai", "Delhi"],
        "meter_type":    ["Electricity", "Electricity", "Chilled Water"],
        "year":          [2025, 2025, 2025],
        "month":         [6, 6, 6]
    })

    st.download_button(
        "⬇ Download Sample CSV",
        sample_data.to_csv(index=False).encode("utf-8"),
        "sample_buildings.csv",
        "text/csv"
    )

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:

        df_upload = pd.read_csv(uploaded_file)

        st.subheader("Preview")
        st.dataframe(df_upload.head(), width="stretch")

        if st.button("⚡ Run Batch Prediction", width="stretch"):

            results  = []
            errors   = []
            progress = st.progress(0)
            total    = len(df_upload)

            with st.spinner("Running batch predictions..."):
                for i, row in df_upload.iterrows():
                    try:
                        result = predict_monthly(
                            building_type = row["building_type"],
                            area          = row["area_sqft"],
                            city_name     = row["city"],
                            year          = int(row["year"]),
                            month         = int(row["month"]),
                            meter_type    = row["meter_type"],
                            model         = model
                        )
                        results.append(result["monthly_total_kwh"])
                    except Exception as e:
                        results.append(None)
                        errors.append(f"Row {i+1}: {str(e)}")

                    progress.progress((i + 1) / total)

            df_upload["predicted_kwh"] = pd.to_numeric(
                pd.Series(results), errors="coerce"
            ).fillna(0)

            st.success(f"✅ Predictions complete for {total} buildings!")

            if errors:
                with st.expander(f"⚠ {len(errors)} errors"):
                    for err in errors:
                        st.write(err)

            st.subheader("Results")
            st.dataframe(
                df_upload.style.format({"predicted_kwh": "{:,.0f}"}),
                width="stretch"
            )

            fig2 = px.bar(
                df_upload,
                x     = "city",
                y     = "predicted_kwh",
                color = "building_type",
                title = "Monthly Energy Consumption by Building"
            )
            fig2.update_layout(
                plot_bgcolor  = "#1e2130",
                paper_bgcolor = "#1e2130",
                font_color    = "#cccccc"
            )
            st.plotly_chart(fig2, width="stretch")

            csv_out = df_upload.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download Results",
                csv_out,
                "energy_predictions.csv",
                "text/csv"
            )
