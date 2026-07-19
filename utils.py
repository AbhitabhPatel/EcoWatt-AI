import pandas as pd
import streamlit as st
from config import DEFAULT_DATASET

@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(DEFAULT_DATASET)
    return df


def prepare_dashboard_data(df):

    df = df.copy()

    # Create datetime if only Date & Time exist
    if "datetime" not in df.columns:
        if "Date" in df.columns and "Time" in df.columns:
            df["datetime"] = pd.to_datetime(
                df["Date"] + " " + df["Time"],
                dayfirst=True,
                errors="coerce"
            )

    # Ensure datetime is datetime type
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(
            df["datetime"],
            errors="coerce"
        )

    # Standardize anomaly column
    if "anomaly" not in df.columns:

        if "final_predicted_anomaly" in df.columns:
            df["anomaly"] = df["final_predicted_anomaly"]

        elif "predicted_anomaly" in df.columns:
            df["anomaly"] = df["predicted_anomaly"]

    # Standardize anomaly score
    if "anomaly_score" not in df.columns:

        if "final_anomaly_score" in df.columns:
            df["anomaly_score"] = df["final_anomaly_score"]

    return df