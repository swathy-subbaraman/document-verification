# Enhanced real-time Streamlit dashboard for MongoDB logs
import streamlit as st
from pymongo import MongoClient
import pandas as pd
import plotly.express as px
import time
from uuid import uuid4

# MongoDB connection
MONGO_URI = "mongodb+srv://vijaisuria:vijai1234@vijai.v30aeb2.mongodb.net/?retryWrites=true&w=majority&appName=vijai"
DB_NAME = "insureshield-ai-dev"
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# Streamlit page config
st.set_page_config(layout="wide")
st.title("🟢 Real-Time InsureShield Log Analytics Dashboard")

# Sidebar controls
log_type = st.sidebar.selectbox("Select Log Type", ["info_log", "events_log", "error_log"])
refresh_rate = st.sidebar.slider("Refresh every (sec)", 2, 30, 5)

# Function to safely fetch logs
def fetch_logs(coll_name, limit=500):
    data = list(db[coll_name].find().sort("timestamp", -1).limit(limit))
    df = pd.json_normalize(data)
    if 'timestamp' not in df.columns:
        return pd.DataFrame()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df = df.sort_values(by='timestamp')
    return df

# Main UI container
placeholder = st.empty()

while True:
    with placeholder.container():
        df = fetch_logs(log_type)
        if df.empty:
            st.warning("⚠️ No data available for selected log type.")
            time.sleep(refresh_rate)
            continue

        uid = str(uuid4())
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.floor('min')

        # Graphs layout
        st.subheader(f"📊 {log_type.upper()} Analytics")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Events per Hour**")
            fig = px.histogram(df, x="hour", nbins=24)
            st.plotly_chart(fig, use_container_width=True, key=f"hour_{uid}")

        with col2:
            st.markdown("**Messages Over Time**")
            freq_df = df.groupby("minute").size().reset_index(name="count")
            fig = px.line(freq_df, x='minute', y='count')
            st.plotly_chart(fig, use_container_width=True, key=f"timeline_{uid}")

        with col3:
            st.markdown("**Top Services**")
            if 'service' in df.columns:
                fig = px.histogram(df, x='service', color='service')
                st.plotly_chart(fig, use_container_width=True, key=f"service_{uid}")

        if log_type == "events_log":
            doc_df = df[df['metadata.action'] == 'document_status_updated']
            if 'metadata.document_type' in doc_df.columns:
                st.subheader("📄 Document Types Uploaded")
                fig = px.pie(doc_df, names='metadata.document_type', title="Uploaded Document Type Share")
                st.plotly_chart(fig, use_container_width=True, key=f"doctype_{uid}")

            ekyc_df = df[df['metadata.action'] == 'ekyc_verified']
            if 'metadata.documents' in ekyc_df.columns:
                ekyc_df['document_count'] = ekyc_df['metadata.documents'].apply(lambda d: len(d) if isinstance(d, list) else 0)
                fig = px.bar(ekyc_df, x='timestamp', y='document_count', title="Document Count per eKYC Verification")
                st.plotly_chart(fig, use_container_width=True, key=f"ekyc_docs_{uid}")

        if log_type == "error_log":
            st.subheader("❌ Login Failures & Lockouts")
            login_df = df[df['message'].str.contains("Login failed", case=False)]
            if 'metadata.attempts_left' in login_df.columns:
                login_df = login_df.dropna(subset=['metadata.attempts_left'])
                fig = px.pie(login_df, names='metadata.attempts_left', title="Distribution of Login Attempts Left")
                st.plotly_chart(fig, use_container_width=True, key=f"login_attempts_{uid}")

            locked_df = df[df['message'].str.contains("Account is locked", case=False)]
            if not locked_df.empty:
                locked_df['hour'] = locked_df['timestamp'].dt.hour
                fig = px.bar(locked_df, x='hour', title="Account Lockouts by Hour")
                st.plotly_chart(fig, use_container_width=True, key=f"lockouts_{uid}")

        if log_type == "info_log":
            st.subheader("📘 Info Log Insights")
            if 'metadata.policy_count' in df.columns:
                df['policy_count'] = df['metadata.policy_count']
                fig = px.area(df.dropna(subset=['policy_count']), x='timestamp', y='policy_count', title="Policy Count Over Time")
                st.plotly_chart(fig, use_container_width=True, key=f"policy_count_{uid}")

            if 'metadata.document_count' in df.columns:
                df['document_count'] = df['metadata.document_count']
                fig = px.area(df.dropna(subset=['document_count']), x='timestamp', y='document_count', title="Document Count Over Time")
                st.plotly_chart(fig, use_container_width=True, key=f"doc_count_{uid}")

        with st.expander("🔍 View Latest Logs"):
            st.dataframe(df.tail(20), use_container_width=True)

    time.sleep(refresh_rate)
