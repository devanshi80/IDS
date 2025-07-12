# streamlit_app.py
import streamlit as st
import pandas as pd
import os
from src.port_scanner import scan_ports
from src.predictor import load_model, predict
from src.logger import log_prediction
from src.explainability import explain_prediction

st.set_page_config(page_title="Smart Threat Detection", layout="wide", initial_sidebar_state="expanded")

# --- Sidebar ---
st.sidebar.title("🔧 Settings")
scan_option = st.sidebar.radio("Scan Type", ["Live Scan", "Upload CSV"])
min_risk = st.sidebar.slider("Minimum Risk Score (%)", 0, 100, 30)
port_range = st.sidebar.slider("Port Range", 0, 65535, (0, 10000))

# --- Load Model ---
MODEL_PATH = "models/best_model.pkl"
model = load_model(MODEL_PATH)

# --- Title ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #cfcfcf; }
    h1, h2, h3 { color: #f39c12; }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ Smart Threat Detection")
st.markdown("Real-time bot detection via port classification with model explainability.")

# --- Main Section ---
if scan_option == "Live Scan":
    if st.button("🔍 Scan My Open Ports"):
        ports = scan_ports()
        filtered_ports = [p for p in ports if port_range[0] <= p <= port_range[1]]
        st.write(f"🧪 Found {len(filtered_ports)} port(s) in range {port_range}: {filtered_ports}")

        for port in filtered_ports:
            features = {"port": port}
            label = predict(model, features)
            prob = model.predict_proba([features])[0][1]  # Probability of bot (class=1)

            if prob * 100 >= min_risk:
                log_prediction(port, prob, label)

                st.markdown(f"### Port `{port}`")
                st.write(f"🧠 Prediction: **{'BOT' if label else 'BENIGN'}**")
                st.write(f"⚠️ Risk Score: `{prob * 100:.2f}%`")
                st.pyplot(fig=explain_prediction(model, features))

elif scan_option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV with a `port` column", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "port" not in df.columns:
            st.error("❌ CSV must contain a column named `port`")
        else:
            for _, row in df.iterrows():
                port = row['port']
                if not (port_range[0] <= port <= port_range[1]):
                    continue
                features = {"port": port}
                label = predict(model, features)
                prob = model.predict_proba([features])[0][1]
                if prob * 100 >= min_risk:
                    log_prediction(port, prob, label)
                    st.markdown(f"### Port `{port}`")
                    st.write(f"Prediction: **{'BOT' if label else 'BENIGN'}**")
                    st.write(f"Risk Score: `{prob * 100:.2f}%`")
                    st.pyplot(fig=explain_prediction(model, features))

# --- Risk Logs ---
st.markdown("---")
st.subheader("📈 Risk Score Logs")
if st.checkbox("Show all logged predictions"):
    logs = pd.read_csv("logs/risk_scores.csv") if os.path.exists("logs/risk_scores.csv") else pd.DataFrame()
    st.dataframe(logs)
