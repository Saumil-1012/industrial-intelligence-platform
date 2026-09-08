"""
Industrial Intelligence Platform — Streamlit Dashboard
Tab 1: Supply Chain Demand Forecasting
Tab 2: Airline Delay & Disruption Intelligence
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
from datetime import datetime, timedelta

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Industrial Intelligence Platform",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("🏭 Industrial Intelligence")
st.sidebar.markdown("**Supply Chain + Airline Operations**")
st.sidebar.divider()

module = st.sidebar.radio(
    "Select Module",
    ["📦 Supply Chain", "✈️ Airline Operations"],
)

st.sidebar.divider()
st.sidebar.markdown("**API Status**")
try:
    r = requests.get(f"{API_BASE}/health", timeout=2)
    st.sidebar.success("API Online ✅") if r.status_code == 200 else st.sidebar.error("API Error ❌")
except Exception:
    st.sidebar.warning("API Offline — Demo Mode")


# SUPPLY CHAIN MODULE

if module == "📦 Supply Chain":
    st.title("📦 Supply Chain Demand Forecasting")
    st.caption("LightGBM · Quantile Regression P10/P50/P90 · SHAP · Anomaly Detection")

    tab1, tab2, tab3 = st.tabs(["🔮 Forecast", "🚨 Anomaly Detection", "📊 Model Performance"])

    with tab1:
        st.subheader("4-Week Demand Forecast")
        col1, col2, col3 = st.columns(3)
        with col1:
            product_id  = st.text_input("Product ID", value="HOBBIES_1_001")
            location_id = st.text_input("Location ID", value="CA_1")
        with col2:
            lag_7  = st.number_input("Last 7-day avg sales",  value=45.0, step=0.5)
            lag_28 = st.number_input("Last 28-day avg sales", value=43.0, step=0.5)
        with col3:
            is_holiday = st.selectbox("German Holiday?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            has_event  = st.selectbox("Has Event?",      [0, 1], format_func=lambda x: "Yes" if x else "No")

        if st.button("🔮 Generate Forecast", type="primary"):
            with st.spinner("Running LightGBM forecast..."):
                try:
                    resp = requests.post(f"{API_BASE}/supply/forecast", json={
                        "product_id": product_id, "location_id": location_id,
                        "horizon_days": 28, "lag_7": lag_7, "lag_28": lag_28,
                        "is_german_holiday": is_holiday, "has_event": has_event,
                    }, timeout=10)
                    data = resp.json()
                except Exception:
                    data = {
                        "product_id": product_id,
                        "forecast": {"p10": 38.2, "p50": 45.7, "p90": 54.1},
                        "shap_top5": {"lag_7": 3.4, "rolling_mean_28": 2.1,
                                      "is_german_holiday": -1.8, "has_event": 1.2,
                                      "rolling_std_28": -0.9},
                        "mode": "demo",
                    }

            forecast = data.get("forecast", {})
            shap     = data.get("shap_top5", {})

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("P10 (Pessimistic)", f"{forecast.get('p10', 38.2):.1f}")
            m2.metric("P50 (Forecast)",    f"{forecast.get('p50', 45.7):.1f}", delta="Point estimate")
            m3.metric("P90 (Optimistic)",  f"{forecast.get('p90', 54.1):.1f}")
            m4.metric("Mode", data.get("mode", "demo").upper())

            st.subheader("📈 28-Day Forecast with Confidence Bands")
            dates    = [datetime.today() + timedelta(days=i) for i in range(28)]
            np.random.seed(42)
            p50_base = forecast.get("p50", 45.7)
            p10_base = forecast.get("p10", 38.2)
            p90_base = forecast.get("p90", 54.1)
            noise    = np.random.normal(0, 2, 28)
            p50_s    = p50_base + noise
            p10_s    = p10_base + noise * 0.7
            p90_s    = p90_base + noise * 1.3

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates + dates[::-1], y=list(p90_s) + list(p10_s[::-1]),
                fill="toself", fillcolor="rgba(99,110,250,0.15)",
                line=dict(color="rgba(255,255,255,0)"), name="P10–P90 Band",
            ))
            fig.add_trace(go.Scatter(x=dates, y=p50_s, mode="lines",
                line=dict(color="#636EFA", width=2.5), name="P50 Forecast"))
            fig.add_trace(go.Scatter(x=dates, y=p10_s, mode="lines",
                line=dict(color="#EF553B", width=1, dash="dot"), name="P10"))
            fig.add_trace(go.Scatter(x=dates, y=p90_s, mode="lines",
                line=dict(color="#00CC96", width=1, dash="dot"), name="P90"))
            fig.update_layout(xaxis_title="Date", yaxis_title="Demand (units)",
                              legend=dict(orientation="h", y=-0.2),
                              margin=dict(l=0, r=0, t=20, b=0), height=350)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("🔍 SHAP Explanation — What's Driving This Forecast?")
            if shap:
                shap_df = pd.DataFrame({
                    "Feature": list(shap.keys()),
                    "SHAP Value": list(shap.values()),
                }).sort_values("SHAP Value")
                colors = ["#EF553B" if v < 0 else "#00CC96" for v in shap_df["SHAP Value"]]
                fig2 = go.Figure(go.Bar(
                    x=shap_df["SHAP Value"], y=shap_df["Feature"],
                    orientation="h", marker_color=colors,
                ))
                fig2.update_layout(xaxis_title="SHAP Value (impact on forecast)",
                                   margin=dict(l=0, r=0, t=10, b=0), height=250)
                st.plotly_chart(fig2, use_container_width=True)
                top_pos = max(shap.items(), key=lambda x: x[1])
                top_neg = min(shap.items(), key=lambda x: x[1])
                st.info(f"**Forecast driven by:** `{top_pos[0]}` adds **+{top_pos[1]:.1f} units**, "
                        f"`{top_neg[0]}` reduces by **{top_neg[1]:.1f} units**")

    with tab2:
        st.subheader("🚨 Real-Time Demand Anomaly Detection")
        st.caption("Isolation Forest + Statistical Process Control (SPC)")
        col1, col2 = st.columns(2)
        with col1:
            a_product      = st.text_input("Product ID", value="HOBBIES_1_001", key="a_prod")
            current_sales  = st.number_input("Current Sales (today)", value=120.0, step=1.0)
            a_mean_28      = st.number_input("28-day rolling mean", value=43.0, step=0.5)
            a_std_28       = st.number_input("28-day rolling std",  value=7.0,  step=0.5)

        if st.button("🔍 Check for Anomaly", type="primary"):
            with st.spinner("Running anomaly detection..."):
                try:
                    resp = requests.post(f"{API_BASE}/supply/anomaly", json={
                        "product_id": a_product, "current_sales": current_sales,
                        "rolling_mean_28": a_mean_28, "rolling_std_28": a_std_28,
                    }, timeout=10)
                    result = resp.json()
                except Exception:
                    z = abs(current_sales - a_mean_28) / (a_std_28 + 1e-6)
                    result = {"is_anomaly": z > 3.0, "z_score": round(z, 3),
                              "detection_layer": "spc_only", "mode": "demo",
                              "root_cause_features": {"deviation": round(current_sales - a_mean_28, 2)}}

            is_anomaly = result.get("is_anomaly", False)
            if is_anomaly:
                st.error(f"⚠️ **ANOMALY DETECTED** — Z-score: {result.get('z_score', 'N/A')}")
            else:
                st.success(f"✅ **Normal** — Z-score: {result.get('z_score', 'N/A')}")

            c1, c2, c3 = st.columns(3)
            c1.metric("Z-Score", result.get("z_score", "N/A"))
            c2.metric("Detection Layer", result.get("detection_layer", "N/A"))
            c3.metric("Anomaly Score", result.get("anomaly_score", "N/A"))

    with tab3:
        st.subheader("📊 Model Performance Over Time")
        weeks      = [f"W{i}" for i in range(1, 13)]
        mae_vals   = [8.2, 7.9, 7.5, 7.8, 7.3, 7.1, 6.9, 7.0, 6.8, 6.7, 6.6, 6.5]
        rmse_vals  = [12.1, 11.8, 11.3, 11.6, 11.0, 10.7, 10.4, 10.5, 10.2, 10.0, 9.9, 9.7]
        drift_scores = [0.04, 0.05, 0.06, 0.08, 0.09, 0.07, 0.11, 0.13, 0.10, 0.08, 0.09, 0.07]

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=weeks, y=mae_vals,  name="MAE",  mode="lines+markers", line=dict(color="#636EFA")))
        fig3.add_trace(go.Scatter(x=weeks, y=rmse_vals, name="RMSE", mode="lines+markers", line=dict(color="#EF553B")))
        fig3.update_layout(xaxis_title="Week", yaxis_title="Error",
                           height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Drift Score Timeline")
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(x=weeks, y=drift_scores,
            marker_color=["#EF553B" if v > 0.15 else "#00CC96" for v in drift_scores]))
        fig4.add_hline(y=0.15, line_dash="dot", line_color="red", annotation_text="Retrain threshold")
        fig4.update_layout(yaxis_title="Drift Score (PSI)",
                           height=250, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig4, use_container_width=True)


# AIRLINE MODULE

elif module == "✈️ Airline Operations":
    st.title("✈️ Airline Delay & Disruption Intelligence")
    st.caption("XGBoost · Platt Calibration · SHAP · NetworkX Cascade Detection")

    tab1, tab2, tab3 = st.tabs(["🛫 Delay Prediction", "🔍 Root Cause", "🌊 Cascade Detection"])

    # ── DELAY PREDICTION ──
    with tab1:
        st.subheader("Flight Delay Prediction (24h ahead)")
        col1, col2, col3 = st.columns(3)
        with col1:
            flight_id    = st.text_input("Flight ID", value="AA101")
            origin       = st.text_input("Origin",    value="JFK")
            dest         = st.text_input("Dest",      value="LAX")
        with col2:
            sched_dep    = st.number_input("Scheduled Dep (HHMM)", value=800, step=100)
            month        = st.slider("Month", 1, 12, 12)
            is_holiday   = st.selectbox("Holiday Week?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        with col3:
            rot_time     = st.number_input("Rotation time (min)", value=35.0, step=5.0)
            weather_sev  = st.slider("Weather severity (0–1)", 0.0, 1.0, 0.3, step=0.05)
            congestion   = st.selectbox("Airport congestion", [0.0, 1.0, 2.0],
                                         format_func=lambda x: {0: "Low", 1: "Medium", 2: "High"}[x])

        if st.button("🛫 Predict Delay", type="primary"):
            with st.spinner("Running XGBoost + Platt calibration..."):
                try:
                    resp = requests.post(f"{API_BASE}/airline/delay", json={
                        "flight_id": flight_id, "origin": origin, "dest": dest,
                        "carrier": "AA", "scheduled_dep": float(sched_dep),
                        "hour_of_day": sched_dep // 100, "month": month,
                        "is_holiday_week": is_holiday,
                        "rotation_time_min": rot_time,
                        "rotation_risk": int(rot_time < 45),
                        "weather_severity": weather_sev,
                        "congestion_tier": congestion,
                    }, timeout=10)
                    data = resp.json()
                except Exception:
                    data = {
                        "delay_probability": 0.72, "delay_minutes_pred": 34.5,
                        "risk_level": "HIGH", "mode": "demo",
                        "shap_top5": {
                            "rotation_risk": 0.31, "weather_severity": 0.22,
                            "is_holiday_week": 0.18, "congestion_tier": 0.14,
                            "carrier_delay_rate": 0.11,
                        },
                    }

            prob      = data.get("delay_probability", 0.72)
            risk      = data.get("risk_level", "HIGH")
            delay_min = data.get("delay_minutes_pred", 34.5)
            shap      = data.get("shap_top5", {})

            risk_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(risk, "⚪")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Delay Probability", f"{prob:.0%}")
            m2.metric("Expected Delay",    f"{delay_min:.0f} min")
            m3.metric("Risk Level",        f"{risk_color} {risk}")
            m4.metric("Mode", data.get("mode", "demo").upper())

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Delay Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#EF553B" if prob > 0.6 else "#FFA500" if prob > 0.3 else "#00CC96"},
                    "steps": [
                        {"range": [0, 30],   "color": "rgba(0,204,150,0.15)"},
                        {"range": [30, 60],  "color": "rgba(255,165,0,0.15)"},
                        {"range": [60, 100], "color": "rgba(239,85,59,0.15)"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 3}, "value": 60},
                },
                number={"suffix": "%", "font": {"size": 32}},
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            if shap:
                st.subheader("🔍 SHAP — What's driving this prediction?")
                shap_df = pd.DataFrame({
                    "Feature": list(shap.keys()),
                    "SHAP Value": list(shap.values()),
                }).sort_values("SHAP Value")
                colors = ["#EF553B" if v > 0 else "#00CC96" for v in shap_df["SHAP Value"]]
                fig_shap = go.Figure(go.Bar(
                    x=shap_df["SHAP Value"], y=shap_df["Feature"],
                    orientation="h", marker_color=colors,
                ))
                fig_shap.update_layout(xaxis_title="SHAP Value (↑ increases delay risk)",
                                       height=250, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_shap, use_container_width=True)

    # ── ROOT CAUSE ──
    with tab2:
        st.subheader("Root Cause Attribution")
        st.caption("5 DOT delay categories: Carrier · Weather · NAS/ATC · Security · Late Aircraft")

        col1, col2 = st.columns(2)
        with col1:
            rc_flight_id    = st.text_input("Flight ID", value="AA101", key="rc_fid")
            delay_observed  = st.number_input("Observed delay (min)", value=45.0, step=5.0)
            rc_weather      = st.slider("Weather severity", 0.0, 1.0, 0.3, step=0.05, key="rc_w")
            rc_rotation     = st.selectbox("Rotation risk?", [0, 1],
                                            format_func=lambda x: "Yes (tight)" if x else "No", key="rc_r")

        if st.button("🔍 Attribute Root Cause", type="primary"):
            with st.spinner("Running multi-label root cause classifier..."):
                try:
                    resp = requests.post(f"{API_BASE}/airline/rootcause", json={
                        "flight_id": rc_flight_id, "delay_observed": delay_observed,
                        "weather_severity": rc_weather, "rotation_risk": rc_rotation,
                    }, timeout=10)
                    data = resp.json()
                except Exception:
                    data = {
                        "cause_probabilities_norm": {
                            "Late Aircraft": 0.45, "Carrier": 0.25,
                            "NAS / ATC": 0.20, "Weather": 0.08, "Security": 0.02,
                        },
                        "primary_cause": "Late Aircraft",
                        "mode": "demo",
                    }

            causes = data.get("cause_probabilities_norm", {})
            primary = data.get("primary_cause", "N/A")

            st.success(f"**Primary cause: {primary}**")

            fig_pie = go.Figure(go.Pie(
                labels=list(causes.keys()),
                values=list(causes.values()),
                hole=0.4,
                marker_colors=["#EF553B", "#636EFA", "#FFA15A", "#00CC96", "#AB63FA"],
            ))
            fig_pie.update_layout(height=320, margin=dict(l=0, r=0, t=20, b=0),
                                   legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig_pie, use_container_width=True)

            cause_df = pd.DataFrame({
                "Cause": list(causes.keys()),
                "Contribution": [f"{v:.0%}" for v in causes.values()],
            })
            st.dataframe(cause_df, use_container_width=True, hide_index=True)

    # ── CASCADE DETECTION ──
    with tab3:
        st.subheader("Cascade Delay Propagation")
        st.caption("NetworkX directed graph — aircraft rotation connections")

        col1, col2 = st.columns(2)
        with col1:
            cascade_fid   = st.text_input("Delayed Flight ID", value="AA101", key="cas_fid")
            cascade_delay = st.number_input("Delay (minutes)", value=75.0, step=5.0)

        if st.button("🌊 Detect Cascade", type="primary"):
            with st.spinner("Propagating through flight network graph..."):
                try:
                    resp = requests.post(f"{API_BASE}/airline/cascade", json={
                        "delayed_flight_id": cascade_fid,
                        "delay_minutes": cascade_delay,
                    }, timeout=10)
                    data = resp.json()
                except Exception:
                    # Demo cascade result
                    data = {
                        "source_flight_id": cascade_fid,
                        "source_delay_min": cascade_delay,
                        "total_affected": 3,
                        "max_propagated_delay": 45.0,
                        "cascade_depth_reached": 2,
                        "affected_flights": [
                            {"flight_id": "AA102", "origin": "LAX", "dest": "SFO",
                             "original_dep_hour": 11.5, "estimated_new_dep_hour": 12.75,
                             "propagated_delay_min": 45.0, "cascade_depth": 1},
                            {"flight_id": "AA103", "origin": "SFO", "dest": "ORD",
                             "original_dep_hour": 13.5, "estimated_new_dep_hour": 14.25,
                             "propagated_delay_min": 15.0, "cascade_depth": 2},
                        ],
                        "graph_summary": {"total_flights": 7, "total_connections": 5,
                                          "buffer_minutes": 30},
                    }

            total_aff = data.get("total_affected", 0)
            max_prop  = data.get("max_propagated_delay", 0)
            depth     = data.get("cascade_depth_reached", 0)
            graph_sum = data.get("graph_summary", {})

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Flights Affected",     total_aff)
            m2.metric("Max Propagated Delay", f"{max_prop:.0f} min")
            m3.metric("Cascade Depth",        depth)
            m4.metric("Network Size",         f"{graph_sum.get('total_flights', '?')} flights")

            affected = data.get("affected_flights", [])
            if affected:
                st.subheader("Affected Downstream Flights")
                aff_df = pd.DataFrame(affected)[[
                    "flight_id", "origin", "dest",
                    "original_dep_hour", "estimated_new_dep_hour",
                    "propagated_delay_min", "cascade_depth",
                ]].rename(columns={
                    "flight_id": "Flight", "origin": "From", "dest": "To",
                    "original_dep_hour": "Orig Dep (h)",
                    "estimated_new_dep_hour": "New Dep (h)",
                    "propagated_delay_min": "Propagated Delay (min)",
                    "cascade_depth": "Depth",
                })

                # Color-code by delay severity
                def color_delay(val):
                    if isinstance(val, (int, float)):
                        if val >= 45:
                            return "background-color: rgba(239,85,59,0.2)"
                        elif val >= 15:
                            return "background-color: rgba(255,165,0,0.2)"
                    return ""

                st.dataframe(
                    aff_df.style.applymap(color_delay, subset=["Propagated Delay (min)"]),
                    use_container_width=True,
                    hide_index=True,
                )

                # Simple visual cascade tree
                st.subheader("Cascade Chain")
                st.markdown(f"**{cascade_fid}** (source, {cascade_delay:.0f} min delay)")
                for f in affected:
                    indent = "　" * f["cascade_depth"]
                    st.markdown(
                        f"{indent}└─ **{f['flight_id']}** "
                        f"({f['origin']} → {f['dest']}) — "
                        f"+{f['propagated_delay_min']:.0f} min delay "
                        f"[depth {f['cascade_depth']}]"
                    )
