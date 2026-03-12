# app/pages/analytics.py

import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle
import sys
import os
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

# ── Path Setup ────────────────────────────────────────────────
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "utils"))

from utils.feature_generator import predict_monthly, generate_single_features
from utils.feature_labels    import (
    PRIMARY_USE_MAP, METER_MAP,
    get_feature_label, get_value_label
)
from utils.recommendations   import generate_recommendations
from utils.summary           import generate_full_report

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title = "Energy Analytics",
    page_icon  = "📊",
    layout     = "wide"
)

st.markdown("""
<style>
.stApp { background-color: #0f1117; }
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
.before-value {
    font-size: 2rem;
    font-weight: 800;
    color: #ff4b4b;
}
.after-value {
    font-size: 2rem;
    font-weight: 800;
    color: #00d4aa;
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #ffaa00;
}
.metric-label {
    font-size: 0.85rem;
    color: #888;
    margin-top: 0.2rem;
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

# ── Load Model + Explainer ────────────────────────────────────
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

# ── Guard ─────────────────────────────────────────────────────
if not st.session_state.get("prediction_done"):
    st.warning("No prediction found. Please run a prediction first.")
    if st.button("← Go Back to Predictor"):
        st.switch_page("app.py")
    st.stop()

# ── Read session state ────────────────────────────────────────
result         = st.session_state["result"]
report         = st.session_state["report"]
recs           = st.session_state["recs"]
shap_vals      = st.session_state["shap_vals"]
feature_names  = st.session_state["feature_names"]
feature_values = st.session_state["feature_values"]
inputs         = st.session_state["inputs"]

before_kwh = report["prediction_kwh"]
mode       = inputs["mode"]

# ── Header ────────────────────────────────────────────────────
st.markdown("# 📊 Energy Analytics")
st.markdown(
    f"**{inputs['building_type']}** · "
    f"{inputs['city_name']} · "
    f"{inputs['area']:,} sq ft · "
    f"{inputs['meter_type']}"
)
st.divider()

# ════════════════════════════════════════════════════════════════
# SECTION 1 — SHAP Waterfall
# ════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🔍 SHAP Explanation — What Drove This Prediction?</div>', unsafe_allow_html=True)
st.caption("Red bars push energy consumption higher. Blue bars push it lower.")

shap.plots.waterfall(
    shap.Explanation(
        values        = shap_vals,
        base_values   = explainer.expected_value,
        data          = [get_value_label(f, v) for f, v in zip(feature_names, feature_values)],
        feature_names = [get_feature_label(f) for f in feature_names]
    ),
    show = False
)

fig_shap = plt.gcf()
fig_shap.set_size_inches(12, 6)
fig_shap.patch.set_facecolor("#1e2130")

ax = fig_shap.axes[0]
ax.set_facecolor("#1e2130")
ax.tick_params(colors="#cccccc", labelsize=11)
ax.xaxis.label.set_color("#cccccc")
ax.yaxis.label.set_color("#cccccc")
ax.title.set_color("#cccccc")

for spine in ax.spines.values():
    spine.set_edgecolor("#2d3148")

for text in ax.texts:
    text.set_color("#ffffff")

plt.tight_layout()
st.pyplot(fig_shap, width="stretch")
plt.close()

# ════════════════════════════════════════════════════════════════
# SECTION 2 — What-If Simulator
# ════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🎛️ What-If Simulator — Adjust & Re-Predict</div>', unsafe_allow_html=True)
st.caption("Modify the values below based on recommendations and see the impact instantly.")

sim_inputs = {}

with st.form("simulator_form"):

    st.markdown("**Adjust values based on recommendations then click Run Simulation:**")

    cols = st.columns(3)

    for idx, rec in enumerate(recs):
        fname = rec["feature_name"]
        fval  = feature_values[feature_names.index(fname)]
        col   = cols[idx % 3]

        with col:
            st.markdown(f"**{rec['feature_label']}**")
            st.caption(f"💡 {rec['advice'][:80]}...")

            if fname == "square_feet":
                sim_inputs[fname] = st.number_input(
                    "Area (sq ft)",
                    min_value = 500,
                    max_value = 5000000,
                    value     = int(fval),
                    step      = 1000,
                    key       = f"sim_{fname}"
                )

            elif fname == "air_temperature":
                sim_inputs[fname] = st.slider(
                    "Temperature (C)",
                    min_value = -10.0,
                    max_value = 50.0,
                    value     = float(round(fval, 1)),
                    step      = 0.5,
                    key       = f"sim_{fname}"
                )

            elif fname == "dew_temperature":
                sim_inputs[fname] = st.slider(
                    "Dew Point (C)",
                    min_value = -10.0,
                    max_value = 40.0,
                    value     = float(round(fval, 1)),
                    step      = 0.5,
                    key       = f"sim_{fname}"
                )

            elif fname == "cloud_coverage":
                sim_inputs[fname] = st.slider(
                    "Cloud Coverage",
                    min_value = 0.0,
                    max_value = 10.0,
                    value     = float(round(fval, 1)),
                    step      = 0.5,
                    key       = f"sim_{fname}"
                )

            elif fname == "wind_speed":
                sim_inputs[fname] = st.slider(
                    "Wind Speed (m/s)",
                    min_value = 0.0,
                    max_value = 50.0,
                    value     = float(round(fval, 1)),
                    step      = 0.5,
                    key       = f"sim_{fname}"
                )

            elif fname == "meter":
                meter_options = list(METER_MAP.keys())
                current_label = get_value_label("meter", fval)
                default_idx   = meter_options.index(current_label) if current_label in meter_options else 0
                selected      = st.selectbox(
                    "Meter Type",
                    meter_options,
                    index = default_idx,
                    key   = f"sim_{fname}"
                )
                sim_inputs[fname] = METER_MAP[selected]

            elif fname == "primary_use":
                use_options   = list(PRIMARY_USE_MAP.keys())
                current_label = get_value_label("primary_use", fval)
                default_idx   = use_options.index(current_label) if current_label in use_options else 0
                selected      = st.selectbox(
                    "Building Type",
                    use_options,
                    index = default_idx,
                    key   = f"sim_{fname}"
                )
                sim_inputs[fname] = PRIMARY_USE_MAP[selected]

            elif fname == "hour":
                sim_inputs[fname] = st.slider(
                    "Hour of Day",
                    min_value = 0,
                    max_value = 23,
                    value     = int(fval),
                    key       = f"sim_{fname}"
                )

            elif fname == "site_id":
                sim_inputs[fname] = st.slider(
                    "Site ID",
                    min_value = 0,
                    max_value = 15,
                    value     = int(fval),
                    key       = f"sim_{fname}"
                )

            else:
                sim_inputs[fname] = st.number_input(
                    f"{rec['feature_label']}",
                    value = float(fval),
                    key   = f"sim_{fname}"
                )

    run_sim = st.form_submit_button(
        "⚡ Run Simulation",
        width="stretch"
    )

# ── Run Simulation ────────────────────────────────────────────
if run_sim:

    sim_row = dict(zip(feature_names, feature_values))

    for fname, new_val in sim_inputs.items():
        sim_row[fname] = new_val

    sim_df    = pd.DataFrame([sim_row])[feature_names]
    sim_log   = model.predict(sim_df)[0]
    after_kwh = float(np.expm1(sim_log))

    # Scale to monthly if needed
    if mode == "Monthly Report" and result:
        num_days  = result["daily_breakdown"].shape[0]
        orig_row  = pd.DataFrame(
            [dict(zip(feature_names, feature_values))]
        )[feature_names]
        orig_hour = float(np.expm1(model.predict(orig_row)[0]))
        if orig_hour > 0:
            scale     = before_kwh / (orig_hour * num_days * 24)
            after_kwh = after_kwh * num_days * 24 * scale

    saved_kwh  = before_kwh - after_kwh
    saving_pct = (saved_kwh / before_kwh * 100) if before_kwh > 0 else 0

    st.session_state["sim_after_kwh"]  = after_kwh
    st.session_state["sim_saved_kwh"]  = saved_kwh
    st.session_state["sim_saving_pct"] = saving_pct
    st.session_state["sim_done"]       = True

# ── Show Simulation Results ───────────────────────────────────
if st.session_state.get("sim_done"):

    after_kwh  = st.session_state["sim_after_kwh"]
    saved_kwh  = st.session_state["sim_saved_kwh"]
    saving_pct = st.session_state["sim_saving_pct"]

    st.markdown('<div class="section-header">⚡ Before vs After Simulation</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="before-value">{before_kwh:,.0f}</div>
            <div class="metric-label">kWh Before</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="after-value">{after_kwh:,.0f}</div>
            <div class="metric-label">kWh After</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{saved_kwh:,.0f}</div>
            <div class="metric-label">kWh Saved</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        color = "after-value" if saving_pct > 0 else "before-value"
        st.markdown(f"""
        <div class="metric-card">
            <div class="{color}">{saving_pct:.1f}%</div>
            <div class="metric-label">Reduction</div>
        </div>""", unsafe_allow_html=True)

    fig_compare = go.Figure()

    fig_compare.add_trace(go.Bar(
        name         = "Before",
        x            = ["Energy Consumption"],
        y            = [before_kwh],
        marker_color = "#ff4b4b",
        width        = 0.3
    ))

    fig_compare.add_trace(go.Bar(
        name         = "After",
        x            = ["Energy Consumption"],
        y            = [after_kwh],
        marker_color = "#00d4aa",
        width        = 0.3
    ))

    fig_compare.update_layout(
        barmode       = "group",
        plot_bgcolor  = "#1e2130",
        paper_bgcolor = "#1e2130",
        font_color    = "#cccccc",
        title         = f"Before vs After — {saving_pct:.1f}% Reduction",
        yaxis_title   = "Energy (kWh)",
        height        = 350,
        legend        = dict(orientation="h", y=1.1)
    )

    st.plotly_chart(fig_compare, width="stretch")

# ════════════════════════════════════════════════════════════════
# SECTION 3 — Daily Breakdown
# ════════════════════════════════════════════════════════════════
if mode == "Monthly Report" and result:

    st.markdown('<div class="section-header">📅 Daily Energy Breakdown</div>', unsafe_allow_html=True)

    daily        = result["daily_breakdown"]
    peak_day_num = int(result["peak_day"]["day"])
    avg_kwh      = daily["total_kwh"].mean()
    colors       = ["#ff4b4b" if d == peak_day_num else "#00d4aa" for d in daily["day"]]

    fig_daily = go.Figure()
    fig_daily.add_trace(go.Bar(
        x            = daily["day"],
        y            = daily["total_kwh"],
        marker_color = colors,
        name         = "Daily kWh"
    ))
    fig_daily.add_hline(
        y               = avg_kwh,
        line_dash       = "dash",
        line_color      = "#ffaa00",
        annotation_text = f"Avg: {avg_kwh:,.0f} kWh"
    )
    fig_daily.update_layout(
        plot_bgcolor  = "#1e2130",
        paper_bgcolor = "#1e2130",
        font_color    = "#cccccc",
        xaxis_title   = "Day of Month",
        yaxis_title   = "Energy (kWh)",
        showlegend    = False,
        height        = 350
    )
    st.plotly_chart(fig_daily, width="stretch")
    st.caption(f"🔴 Peak day: Day {peak_day_num} — {result['peak_day']['total_kwh']:,.0f} kWh")

# ════════════════════════════════════════════════════════════════
# SECTION 4 — Recommendations Detail
# ════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">💡 Recommendations Applied</div>', unsafe_allow_html=True)

for i, rec in enumerate(recs, 1):
    with st.expander(
        f"{i}. {rec['feature_label']}: {rec['feature_value']} "
        f"— Save {rec['saving_min']}–{rec['saving_max']}%"
    ):
        st.write(rec["advice"])
        st.markdown(f"**SHAP Impact:** +{rec['shap_impact']:.3f} on log scale")
        st.markdown(f"**Estimated Saving:** {rec['saving_min']}–{rec['saving_max']}%")

# ════════════════════════════════════════════════════════════════
# SECTION 5 — Full Summary
# ════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📝 Full Energy Audit Summary</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="summary-box">
    {report["summary_text"]}
</div>""", unsafe_allow_html=True)

# ── Back Button ───────────────────────────────────────────────
st.divider()
if st.button("← Back to Predictor", width="stretch"):
    st.switch_page("app.py")
