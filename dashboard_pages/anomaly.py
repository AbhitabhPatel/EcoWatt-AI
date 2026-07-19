import streamlit as st 
import pandas as pd 
import plotly.express as px 


def show(df,feature_columns):   
    st.header("🚨 AI Anomaly Detection")

    total_records = len(df)
    anomaly_count = (df["anomaly"] == 1).sum()
    normal_count = (df["anomaly"] == 0).sum()
    anomaly_percent = (anomaly_count / total_records) * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", f"{total_records:,}")
    col2.metric("Normal Records", f"{normal_count:,}")
    col3.metric("Anomalies", f"{anomaly_count:,}")
    col4.metric("Anomaly %", f"{anomaly_percent:.2f}%")

    fig = px.pie(
    names=["Normal", "Anomaly"],
    values=[normal_count, anomaly_count],
    title="Normal vs Anomaly Distribution")

    st.plotly_chart(fig, width="stretch")

    st.subheader("Filter by Date")

    df["datetime"] = pd.to_datetime(df["datetime"])

    start_date = st.date_input(
    "Start Date",
    df["datetime"].min().date())

    end_date = st.date_input(
    "End Date",
    df["datetime"].max().date())

    filtered_df = df[
    (df["datetime"].dt.date >= start_date) &
    (df["datetime"].dt.date <= end_date)]

    st.subheader("Detected Anomalies")

    anomaly_df = filtered_df[
    filtered_df["anomaly"] == 1]

    st.dataframe(anomaly_df)
    csv = anomaly_df.to_csv(index=False)

    st.download_button(
    "📥 Download Anomalies",
    csv,
    "detected_anomalies.csv",
    "text/csv")
    feature = st.selectbox(
    "Select Feature",feature_columns)

    fig = px.scatter(
    filtered_df,
    x="datetime",
    y=feature,
    color=filtered_df["anomaly"].astype(str),
    title=f"Anomaly Detection for {feature}")

    st.plotly_chart(fig, width="stretch")
