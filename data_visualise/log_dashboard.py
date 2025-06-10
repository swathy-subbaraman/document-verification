import streamlit as st
from pymongo import MongoClient
import pandas as pd
import plotly.express as px
import time

# MongoDB connection
MONGO_URI = "mongodb+srv://vijaisuria:vijai1234@vijai.v30aeb2.mongodb.net/?retryWrites=true&w=majority&appName=vijai"
DB_NAME = "insureshield-ai-dev"
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# UI
st.set_page_config(layout="wide")
st.title("🟢 Real-Time Log Dashboard")

log_type = st.sidebar.selectbox("Select Log Type", ["info_log", "events_log", "error_log"])
refresh_rate = st.sidebar.slider("Refresh every (sec)", 2, 30, 5)

# Function to fetch data
def fetch_logs(coll_name, limit=200):
    data = list(db[coll_name].find().sort("timestamp", -1).limit(limit))
    df = pd.json_normalize(data)

    # Handle missing or malformed timestamps
    if 'timestamp' not in df.columns:
        return pd.DataFrame()

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])  # remove rows with invalid timestamps
    df = df.sort_values(by='timestamp')
    return df

# Live updater
placeholder = st.empty()

from uuid import uuid4  # Place this import at the top

# Inside the loop:
while True:
    with placeholder.container():
        df = fetch_logs(log_type)

        if df.empty:
            st.warning("⚠️ No data found or timestamp missing in selected collection.")
            time.sleep(refresh_rate)
            continue

        # Generate unique keys for every refresh
        unique_id = str(uuid4())

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"{log_type.upper()} - Count by Hour")
            df['hour'] = df['timestamp'].dt.hour
            fig1 = px.histogram(df, x="hour", nbins=24, title="Events per Hour")
            st.plotly_chart(fig1, use_container_width=True, key=f"{log_type}_plot_hour_{unique_id}")

        with col2:
            st.subheader("Messages Over Time")
            df['minute'] = df['timestamp'].dt.floor('min')
            msg_df = df.groupby("minute").size().reset_index(name="count")
            fig2 = px.line(msg_df, x='minute', y='count', title="Log Frequency Over Time")
            st.plotly_chart(fig2, use_container_width=True, key=f"{log_type}_plot_time_{unique_id}")

        with st.expander("🔍 Raw Logs"):
            st.dataframe(df.tail(20), use_container_width=True)

    time.sleep(refresh_rate)
