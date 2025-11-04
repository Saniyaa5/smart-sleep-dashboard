# app.py
# 🩺 Smart Sleep Dashboard — Robust, Secure, Cloud-Ready Version

import streamlit as st
st.set_page_config(page_title="Sleep Apnea Dashboard", layout="wide", page_icon="🩺")

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from io import BytesIO
from dotenv import load_dotenv
import os
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from matplotlib.backends.backend_pdf import PdfPages
import re
import hashlib

# --------------------------
# Environment & Model
# --------------------------
load_dotenv()

# Load email credentials (works for both local and Streamlit Cloud)
EMAIL_USER = None
EMAIL_PASS = None

if "secrets" in dir(st) and "EMAIL_USER" in st.secrets:
    EMAIL_USER = st.secrets["EMAIL_USER"]
    EMAIL_PASS = st.secrets["EMAIL_PASS"]
else:
    EMAIL_USER = os.getenv("EMAIL_USER")
    EMAIL_PASS = os.getenv("EMAIL_PASS")

# Auto-detect model path (works locally + on Streamlit Cloud)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "sleep_apnea_model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = None
try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.error("❌ Model file not found. Please ensure 'sleep_apnea_model.pkl' is in the same folder as app.py.")
except Exception as e:
    st.warning(f"⚠️ Model couldn't be loaded automatically. Error: {e}")

TRAIN_FEATURES = getattr(model, "feature_names_in_", []) if model else []

# --------------------------
# Utility functions
# --------------------------
def hash_password(password: str) -> str:
    """Simple SHA256 hashing for local storage (better than plaintext)."""
    return hashlib.sha256(password.encode()).hexdigest()

EMAIL_REGEX = re.compile(r"^[\w\.-]+@[\w\.-]+\.\w+$")

def is_valid_email(email: str) -> bool:
    return bool(email and EMAIL_REGEX.match(email))

# --------------------------
# Database (persistent sqlite file)
# --------------------------
DB_PATH = "app_db.sqlite"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT,
            name TEXT PRIMARY KEY,
            age INTEGER,
            gender TEXT,
            weight REAL,
            guardian TEXT
        )
    """)
    conn.commit()
    conn.close()

def register_user(username, password, role):
    conn = get_db_connection()
    c = conn.cursor()
    hashed = hash_password(password)
    c.execute("SELECT 1 FROM users WHERE username = ?", (username,))
    if c.fetchone():
        conn.close()
        return False, "Username already exists"
    c.execute("INSERT INTO users (username,password,role) VALUES (?,?,?)", (username, hashed, role))
    conn.commit()
    conn.close()
    return True, "Registered successfully"

def login_user(username, password):
    conn = get_db_connection()
    c = conn.cursor()
    hashed = hash_password(password)
    c.execute("SELECT role FROM users WHERE username=? AND password=?", (username, hashed))
    res = c.fetchone()
    conn.close()
    return res[0] if res else None

def store_patient_info(patient_info):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO patients (patient_id, name, age, gender, weight, guardian)
        VALUES (?,?,?,?,?,?)
    """, (patient_info['patient_id'], patient_info['name'], patient_info['age'],
          patient_info['gender'], patient_info['weight'], patient_info['guardian']))
    conn.commit()
    conn.close()

# --------------------------
# Feature Engineering
# --------------------------
def feature_engineer(df, window=5):
    df = df.copy()
    df["hr_mean"] = df["heart_rate"].rolling(window).mean()
    df["spo2_mean"] = df["spo2"].rolling(window).mean()
    df["hr_std"] = df["heart_rate"].rolling(window).std()
    df["spo2_std"] = df["spo2"].rolling(window).std()
    df["hr_diff"] = df["heart_rate"].diff()
    df["spo2_diff"] = df["spo2"].diff()
    return df.dropna().reset_index(drop=True)

# --------------------------
# Email Alert
# --------------------------
def send_email_alert(to_email, patient_name):
    if not EMAIL_USER or not EMAIL_PASS:
        st.warning("Email alerts are disabled (missing credentials).")
        return False
    if not is_valid_email(to_email):
        st.error("Guardian email is invalid.")
        return False

    try:
        subject = "⚠️ Sleep Apnea Alert!"
        body = f"Patient {patient_name} shows abnormal sleep patterns. Please check immediately."
        msg = MIMEMultipart()
        msg["From"] = EMAIL_USER
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)
        server.quit()
        st.success(f"📩 Alert email sent to {to_email}")
        return True
    except Exception as e:
        st.error(f"⚠️ Error sending email: {e}")
        return False

# --------------------------
# PDF Report Generation
# --------------------------
def generate_pdf_report(patient_info, df_proc):
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        fig, ax = plt.subplots(figsize=(8,6))
        ax.axis('off')
        info_text = (
            f"Patient ID: {patient_info['patient_id']}\n"
            f"Patient Name: {patient_info['name']}\n"
            f"Age: {patient_info['age']}\n"
            f"Gender: {patient_info['gender']}\n"
            f"Weight: {patient_info['weight']} kg\n"
            f"Guardian: {patient_info['guardian']}\n\n"
            f"Total Apnea Events: {int(df_proc['predicted_apnea'].sum())}"
        )
        ax.text(0.1, 0.5, info_text, fontsize=14)
        pdf.savefig(fig)
        plt.close()

        if "time_sec" in df_proc.columns:
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(df_proc["time_sec"], df_proc["heart_rate"], label="Heart Rate")
            ax.plot(df_proc["time_sec"], df_proc["spo2"], label="SpO₂")
            ax.set_title(f"{patient_info['name']} - HR & SpO₂ Trends")
            ax.legend()
            pdf.savefig(fig)
            plt.close()
    buffer.seek(0)
    return buffer

# --------------------------
# UI & Layout
# --------------------------
def local_css():
    st.markdown("""
        <style>
        .stApp { background: linear-gradient(180deg, #f7fbff 0%, #ffffff 100%); }
        .title { font-size:32px; font-weight:700; color:#0b486b; }
        </style>
    """, unsafe_allow_html=True)

local_css()

init_db()
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.user = None

# --------------------------
# Sidebar Navigation
# --------------------------
st.sidebar.title("Menu")
menu = ["Doctor Register", "Doctor Login", "Admin Register", "Admin Login"]
choice = st.sidebar.selectbox("", menu)

st.markdown('<div class="title">🩺 Smart Sleep Dashboard</div>', unsafe_allow_html=True)
st.markdown("---")

# --------------------------
# Auth System
# --------------------------
if choice in ("Doctor Register", "Admin Register"):
    role = "doctor" if choice == "Doctor Register" else "admin"
    st.subheader(f"{role.capitalize()} Registration")
    with st.form(f"reg_form_{role}"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Register")
    if submit:
        if not username or not password:
            st.error("Please fill all fields.")
        else:
            ok, msg = register_user(username.strip(), password, role)
            if ok:
                st.success(msg + " You are now logged in.")
                st.session_state.logged_in = True
                st.session_state.role = role
                st.session_state.user = username.strip()
            else:
                st.warning(msg)

elif choice in ("Doctor Login", "Admin Login"):
    role = "doctor" if choice == "Doctor Login" else "admin"
    st.subheader(f"{role.capitalize()} Login")
    with st.form(f"login_form_{role}"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
    if submit:
        role_found = login_user(username.strip(), password)
        if role_found == role:
            st.success(f"Logged in as {role.capitalize()} ✅")
            st.session_state.logged_in = True
            st.session_state.role = role
            st.session_state.user = username.strip()
        else:
            st.error("Invalid credentials ❌")

# --------------------------
# Doctor Dashboard
# --------------------------
if st.session_state.get("logged_in") and st.session_state.get("role") == "doctor":
    st.subheader("Patient Info & CSV Upload")
    with st.expander("Enter patient details"):
        with st.form("patient_form"):
            col1, col2 = st.columns([1,2])
            with col1:
                patient_id = st.text_input("Patient ID")
                patient_name = st.text_input("Patient Name")
                age = st.number_input("Age", min_value=0, max_value=120, value=25)
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=60.0)
                guardian = st.text_input("Guardian Email")
            with col2:
                uploaded_file = st.file_uploader("Upload CSV (heart_rate, spo2, time_sec)", type=["csv"])
            submitted = st.form_submit_button("Process CSV")

    if submitted:
        if not patient_name or not patient_id:
            st.error("Please provide Patient ID and Name.")
        elif guardian and not is_valid_email(guardian):
            st.error("Invalid guardian email.")
        elif not uploaded_file:
            st.error("Please upload a CSV file.")
        else:
            try:
                df = pd.read_csv(uploaded_file)
                if not {"heart_rate", "spo2"}.issubset(df.columns):
                    st.error("CSV must contain 'heart_rate' and 'spo2' columns.")
                else:
                    df_proc = feature_engineer(df)
                    X_test = (
                        df_proc.reindex(columns=TRAIN_FEATURES, fill_value=0)
                        if TRAIN_FEATURES else
                        df_proc.drop(columns=['apnea_label','patient_id','time_sec'], errors='ignore')
                    )
                    df_proc["predicted_apnea"] = model.predict(X_test) if model else 0
                    apnea_events = int(df_proc["predicted_apnea"].sum())

                    patient_info = {
                        "patient_id": patient_id,
                        "name": patient_name,
                        "age": int(age),
                        "gender": gender,
                        "weight": float(weight),
                        "guardian": guardian
                    }
                    store_patient_info(patient_info)
                    st.success(f"✅ Total Apnea Events Detected: {apnea_events}")

                    if "time_sec" in df_proc.columns:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df_proc["time_sec"], y=df_proc["heart_rate"],
                            mode="lines", name="❤️ Heart Rate", line=dict(width=2)
                        ))
                        fig.add_trace(go.Scatter(
                            x=df_proc["time_sec"], y=df_proc["spo2"],
                            mode="lines", name="🩸 SpO₂", line=dict(width=2, dash="dot")
                        ))
                        fig.update_layout(
                            title=f"{patient_name} - HR & SpO₂ Trends",
                            xaxis=dict(title="Time (sec)", rangeslider=dict(visible=True)),
                            yaxis_title="Values",
                            hovermode="x unified",
                            template="plotly_white",
                            dragmode="zoom"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    pdf_buffer = generate_pdf_report(patient_info, df_proc)
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"{patient_id}_{patient_name}_report.pdf",
                        mime="application/pdf"
                    )

                    if apnea_events > 0 and guardian:
                        send_email_alert(guardian, patient_name)
            except Exception as e:
                st.error(f"Error processing CSV: {e}")

# --------------------------
# Admin Dashboard
# --------------------------
elif st.session_state.get("logged_in") and st.session_state.get("role") == "admin":
    st.subheader("Admin Dashboard")
    conn = get_db_connection()
    df_users = pd.read_sql_query("SELECT username, role FROM users", conn)
    df_patients = pd.read_sql_query("SELECT * FROM patients", conn)
    conn.close()
    st.markdown("**Registered Users**")
    st.dataframe(df_users)
    st.markdown("**Stored Patients**")
    st.dataframe(df_patients)
