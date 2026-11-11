
import streamlit as st
import plotly.express as px

from member3_agent_app.services.data_loader import (
    load_diagnoses,
    get_session_data
)

st.set_page_config(
    page_title="BMS Intelligence",
    page_icon="🔋",
    layout="wide"
)

st.title("🔋 BMS Intelligent Diagnosis")
st.caption("Machine Learning • Fault Detection • Evidence Analysis")

# 1. Charger les diagnostics de M2
diagnoses = load_diagnoses()

session_ids = sorted([
    d["session_id"] for d in diagnoses
])

# 2. Choisir une session
default_index = (
    session_ids.index("F02_R002")
    if "F02_R002" in session_ids
    else 0
)

selected_session = st.selectbox(
    "Select BMS session",
    session_ids,
    index=default_index
)

# 3. Récupérer diagnostic + mesures
diagnosis, signals = get_session_data(selected_session)

if diagnosis is None or signals.empty:
    st.error("No data available for this session.")
    st.stop()

# 4. Afficher les résultats
st.subheader("Diagnostic Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Session", selected_session)
col2.metric("Functional Verdict", diagnosis["verdict"])
col3.metric(
    "Measurement Limits",
    diagnosis.get("measurement_limits_status", "UNKNOWN")
)
col4.metric(
    "Detected Fault",
    diagnosis.get("detected_fault") or "None"
)

if diagnosis["verdict"] == "NOT_EVALUATED":
    st.info(
        "Functional PASS/FAIL has not been evaluated. "
        "The available signals do not include ECU "
        "protective-response validation."
    )

# 5. Graphiques des signaux de M1
st.subheader("BMS Signal Monitoring")

signal_units = {
    "pack_voltage_v": "V",
    "pack_current_a": "A",
    "temperature_c": "°C",
    "soc_pct": "%"
}

selected_signal = st.selectbox(
    "Choose a signal",
    list(signal_units.keys())
)

fig = px.line(
    signals.sort_values("timestamp_s"),
    x="timestamp_s",
    y=selected_signal,
    title=f"{selected_signal} over time",
    labels={
        "timestamp_s": "Time (seconds)",
        selected_signal: signal_units[selected_signal]
    }
)

fig.update_layout(
    template="plotly_dark",
    height=420
)

st.plotly_chart(fig, use_container_width=True)

# 6. Résultats de la détection d'anomalies
st.subheader("Anomaly Detection")

anomaly = diagnosis.get("anomaly_detection", {})

c1, c2, c3 = st.columns(3)

c1.metric(
    "Anomaly detected",
    str(anomaly.get("anomaly_detected", False))
)

c2.metric(
    "Anomalous windows",
    anomaly.get("anomalous_windows", 0)
)

c3.metric(
    "Total windows",
    anomaly.get("total_windows", 0)
)

# 7. Preuves du diagnostic
st.subheader("Diagnostic Evidence")

evidence = diagnosis.get("evidence", [])

if evidence:
    st.dataframe(
        evidence,
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No diagnostic evidence reported.")

# 8. Causes probables
st.subheader("Probable Causes")

causes = diagnosis.get("probable_causes", [])

if causes:
    for cause in causes:
        st.write("**Cause:**", cause.get("code"))
        st.write("**Method:**", cause.get("method"))
        st.write("**Model score:**", cause.get("score"))
else:
    st.info("No probable cause reported.")

# 9. Vérification recommandée
st.subheader("Recommended Test")

st.write(
    diagnosis.get("recommended_test", "Not available")
)

st.write(
    "Test performed:",
    diagnosis.get("recommended_test_done", False)
)

st.caption(
    "Synthetic BMS prototype — no physical root-cause "
    "confirmation or validated ECU functional verdict."
)
