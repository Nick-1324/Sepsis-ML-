import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ---------------- CONFIG ----------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
MODEL_PATH = "sepsis_model.pkl"

# ---------------- MEDICAL LOGIC ----------------
def calculate_qsofa(rr, sbp, mental_status):
    print(int(rr >= 22) + int(sbp <= 100) + int(mental_status == "Altered"))
    return int(rr >= 22) + int(sbp <= 100) + int(mental_status == "Altered")


def calculate_sirs(temp, hr, rr, wbc):
    return (
        int(temp < 36 or temp > 38) +
        int(hr > 90) +
        int(rr > 20) +
        int(wbc < 4 or wbc > 12)
    )

# ---------------- ML MODEL ----------------
def train_model():
    np.random.seed(42)
    n = 1200
    data = pd.DataFrame({
        "heart_rate": np.random.normal(95, 20, n),
        "resp_rate": np.random.normal(22, 6, n),
        "sbp": np.random.normal(105, 20, n),
        "temperature": np.random.normal(38, 1.2, n),
        "wbc": np.random.normal(11, 4, n),
        "age": np.random.normal(60, 15, n),
    })

    data["sepsis"] = (
        (data["resp_rate"] > 22).astype(int) +
        (data["sbp"] < 100).astype(int) +
        (data["temperature"] > 38).astype(int) +
        (data["heart_rate"] > 100).astype(int)
    ) >= 2

    X = data.drop(columns=["sepsis"])
    y = data["sepsis"].astype(int)

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    return model


def load_model():
    if not os.path.exists(MODEL_PATH):
        return train_model()
    return joblib.load(MODEL_PATH)


model = load_model()

# ---------------- GROQ CHAT ----------------
def groq_chat(messages):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.2
    }

    r = requests.post(GROQ_ENDPOINT, headers=headers, json=payload, timeout=30)
    data = r.json()

    if "choices" not in data:
        return f"⚠️ Groq error:\n{data}"

    return data["choices"][0]["message"]["content"]

# ---------------- SESSION STATE ----------------
if "chat" not in st.session_state:
    st.session_state.chat = []

if "context" not in st.session_state:
    st.session_state.context = ""

# ---------------- UI ----------------
st.set_page_config(page_title="Sepsis Early Warning + AI", layout="wide")
st.title("🩺 Sepsis Early Warning System + AI Assistant")

tab1, tab2 = st.tabs(["📊 Risk Assessment", "💬 AI Chat"])

# ================= TAB 1 =================
with tab1:
    st.warning(
        "⚠️ Clinical decision support only. "
        "Not a diagnosis or replacement for medical judgment."
    )

    col1, col2 = st.columns(2)

    with col1:
        hr = st.number_input("Heart Rate", 30, 200, 95)
        rr = st.number_input("Respiratory Rate", 10, 60, 22)
        sbp = st.number_input("Systolic BP", 60, 200, 105)

    with col2:
        temp = st.number_input("Temperature (°C)", 34.0, 42.0, 38.0)
        wbc = st.number_input("WBC (×10⁹/L)", 1.0, 30.0, 11.0)
        age = st.number_input("Age", 0, 120, 60)

    mental = st.selectbox("Mental Status", ["Normal", "Altered"])

    if st.button("🚨 Evaluate Risk"):
        qsofa = calculate_qsofa(rr, sbp, mental)
        sirs = calculate_sirs(temp, hr, rr, wbc)

        features = np.array([[hr, rr, sbp, temp, wbc, age]])
        risk = model.predict_proba(features)[0][1]

        st.metric("qSOFA", qsofa)
        st.metric("SIRS", sirs)
        st.metric("ML Risk", f"{risk:.2f}")

        if risk >= 0.7 or qsofa >= 2:
            status = "HIGH RISK"
            st.error("🚨 HIGH SEPSIS RISK")
        elif risk >= 0.4:
            status = "MODERATE RISK"
            st.warning("⚠️ MODERATE RISK")
        else:
            status = "LOW RISK"
            st.success("🟢 LOW RISK")

        # ---- SAVE CONTEXT FOR CHAT ----
        st.session_state.context = f"""
Patient Summary:
- Age: {age}
- HR: {hr}
- RR: {rr}
- SBP: {sbp}
- Temp: {temp}
- WBC: {wbc}
- Mental Status: {mental}

Scores:
- qSOFA: {qsofa}
- SIRS: {sirs}
- ML Risk Probability: {risk:.2f}
- Risk Category: {status}

You are a clinical AI assistant.
Answer cautiously. Never diagnose.
"""

        st.success("Context saved. Ask questions in the AI Chat tab →")

# ================= TAB 2 =================
with tab2:
    st.subheader("💬 AI Clinical Assistant")
    st.caption("Ask questions about the risk, scores, next steps, or interpretation.")

    # Chat history display (WhatsApp-like)
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about the result...")

    if user_input:
        st.session_state.chat.append({"role": "user", "content": user_input})

        messages = [
            {"role": "system", "content": st.session_state.context},
            *st.session_state.chat
        ]

        with st.spinner("Thinking..."):
            reply = groq_chat(messages)

        st.session_state.chat.append({"role": "assistant", "content": reply})

        with st.chat_message("assistant"):
            st.markdown(reply)

