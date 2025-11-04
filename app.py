# app.py
# Smart Sleep Dashboard — improved persistence, email validation, nicer UI

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
EMAIL_USER = os.getenv("EMAIL_USER") or st.secrets.get("EMAIL_USER") if "secrets" in dir(st) else None
EMAIL_PASS = os.getenv("EMAIL_PASS") or st.secrets.get("EMAIL_PASS") if "secrets" in dir(st) else None

MODEL_PATH = r"C:\Users\SANIYA SULTHANA\FYPROJECT\sleep_apnea_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = None
try:
    model = load_model()
except Exception as e:
    st.warning("Model couldn't be loaded automatically. If you are in development, ignore this. Error: " + str(e))

TRAIN_FEATURES = getattr(model, "feature_names_in_", None)

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
    # Users table
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        )
    """)
    # Patients table
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
    c.execute("INSERT INTO users (username,password,role) VALUES (?,?,?)",
              (username, hashed, role))
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
        st.error("Email credentials are not configured (EMAIL_USER / EMAIL_PASS). Alerts disabled.")
        return False

    if not is_valid_email(to_email):
        st.error("Guardian email is not valid; can't send alert.")
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
        st.error(f"⚠️ Error sending alert: {e}")
        return False

# --------------------------
# PDF Report Generation
# --------------------------
def generate_pdf_report(patient_info, df_proc):
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        # Page 1: Patient Info
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

        # Page 2: HR & SpO2 Trends
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
# UI helpers & CSS
# --------------------------
def local_css():
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(180deg, #f7fbff 0%, #ffffff 100%); }
        .title { font-size:32px; font-weight:700; color:#0b486b; }
        .card { padding: 12px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.06); background: white; }
        </style>
        """, unsafe_allow_html=True
    )

local_css()

# --------------------------
# Init DB & Session
# --------------------------
init_db()
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.user = None

# --------------------------
# Sidebar menu
# --------------------------
st.sidebar.title("Menu")
menu = ["Doctor Register", "Doctor Login", "Admin Register", "Admin Login"]
choice = st.sidebar.selectbox("", menu)

st.markdown('<div class="title">🩺 Smart Sleep Dashboard</div>', unsafe_allow_html=True)
st.markdown("---")

# --------------------------
# Register / Login forms
# --------------------------
if choice in ("Doctor Register", "Admin Register"):
    role = "doctor" if choice == "Doctor Register" else "admin"
    st.subheader(f"{role.capitalize()} Registration")
    with st.form(key=f"reg_form_{role}", clear_on_submit=False):
        username = st.text_input("Username", key=f"{role}_reg_user")
        password = st.text_input("Password", type="password", key=f"{role}_reg_pass")
        submit = st.form_submit_button("Register")
    if submit:
        if not username or not password:
            st.error("Please fill username and password.")
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
    with st.form(key=f"login_form_{role}"):
        username = st.text_input("Username", key=f"{role}_login_user")
        password = st.text_input("Password", type="password", key=f"{role}_login_pass")
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
# Doctor Section (after login)
# --------------------------
if st.session_state.get("logged_in") and st.session_state.get("role") == "doctor":
    st.subheader("Patient Info & CSV Upload")
    st.markdown("Use the form below to upload patient CSV and run apnea detection.")

    with st.expander("Enter patient details"):
        with st.form("patient_form", clear_on_submit=False):
            col1, col2 = st.columns([1,2])
            with col1:
                patient_id = st.text_input("Patient ID", key="patient_id")
                patient_name = st.text_input("Patient Name", key="patient_name")
                age = st.number_input(
                    "Age", min_value=0, max_value=120, value=25, step=1, key="age"
                )
                gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="gender")
                weight = st.number_input(
                    "Weight (kg)", min_value=20.0, max_value=200.0, value=60.0, step=0.1, key="weight"
                )
                guardian = st.text_input("Guardian Email", key="guardian")
            with col2:
                uploaded_file = st.file_uploader("Upload CSV (heart_rate, spo2, time_sec)", type=["csv"], key="csv")
                st.write("CSV must include `heart_rate` and `spo2`. Optional `time_sec` for plotting.")
            submitted = st.form_submit_button("Process CSV")

    if submitted:
        if not patient_name or not patient_id:
            st.error("Please provide Patient ID and Name.")
        elif guardian and not is_valid_email(guardian):
            st.error("Guardian email looks invalid. Please correct.")
        elif not uploaded_file:
            st.error("Please upload a CSV file.")
        else:
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error("Error reading CSV: " + str(e))
                df = None

            if df is not None:
                if not {"heart_rate", "spo2"}.issubset(df.columns):
                    st.error("CSV must contain 'heart_rate' and 'spo2' columns.")
                else:
                    df_proc = feature_engineer(df)
                    if TRAIN_FEATURES is not None:
                        X_test = df_proc.reindex(columns=TRAIN_FEATURES, fill_value=0)
                    else:
                        X_test = df_proc.drop(columns=['apnea_label','patient_id','time_sec'], errors='ignore')

                    if model is None:
                        st.error("Model not available — prediction skipped.")
                        df_proc["predicted_apnea"] = 0
                        apnea_events = 0
                    else:
                        df_proc["predicted_apnea"] = model.predict(X_test)
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

                    # Updated Interactive Plot
                    if "time_sec" in df_proc.columns:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df_proc["time_sec"], y=df_proc["heart_rate"],
                            mode="lines", name="❤️ Heart Rate (bpm)",
                            line=dict(width=2)
                        ))
                        fig.add_trace(go.Scatter(
                            x=df_proc["time_sec"], y=df_proc["spo2"],
                            mode="lines", name="🩸 SpO₂ (%)",
                            line=dict(width=2, dash="dot")
                        ))
                        fig.update_layout(
                            title=dict(
                                text=f"{patient_name} - HR & SpO₂ Interactive Trends",
                                x=0.5, xanchor="center", font=dict(size=20, color="#1c4e80")
                            ),
                            xaxis=dict(
                                title="⏱️ Time (seconds)",
                                showgrid=True,
                                rangeslider=dict(visible=True)
                            ),
                            yaxis=dict(title="📊 Measured Values", showgrid=True),
                            hovermode="x unified",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            template="plotly_white",
                            height=500,
                            dragmode="zoom",
                            hoverlabel=dict(bgcolor="white", font_size=13),
                            margin=dict(l=40, r=20, t=60, b=40)
                        )
                        st.plotly_chart(fig, use_container_width=True, config={
                            "displayModeBar": True,
                            "displaylogo": False,
                            "modeBarButtonsToAdd": ["drawline", "eraseshape"]
                        })

                    pdf_buffer = generate_pdf_report(patient_info, df_proc)
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"{patient_id}_{patient_name}_apnea_report.pdf",
                        mime="application/pdf"
                    )

                    if apnea_events > 0 and guardian:
                        send_email_alert(guardian, patient_name)

# --------------------------
# Admin Section
# --------------------------
elif st.session_state.get("logged_in") and st.session_state.get("role") == "admin":
    st.subheader("Admin Dashboard")
    st.markdown("Manage the system - view registered users & patients.")
    conn = get_db_connection()
    df_users = pd.read_sql_query("SELECT username, role FROM users", conn)
    df_patients = pd.read_sql_query("SELECT * FROM patients", conn)
    conn.close()

    st.markdown("**Registered Users**")
    st.dataframe(df_users)

    st.markdown("**Stored Patients**")
    st.dataframe(df_patients)
