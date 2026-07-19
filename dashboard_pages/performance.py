import streamlit as st 
import plotly.express as px 
import matplotlib.pyplot as plt 
import pandas as pd


def show(df,feature_columns):
    st.header("📊 Model Performance")

    st.write("""
    This page summarizes the trained Isolation Forest model,
    dataset statistics, anomaly detection results and model
    configuration.
    """)

    # ======================================
    # Dataset Summary
    # ======================================

    total_records = len(df)
    anomaly_count = (df["anomaly"] == 1).sum()
    normal_count = (df["anomaly"] == 0).sum()
    anomaly_percent = anomaly_count / total_records * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", f"{total_records:,}")
    col2.metric("Normal", f"{normal_count:,}")
    col3.metric("Anomalies", f"{anomaly_count:,}")
    col4.metric("Anomaly %", f"{anomaly_percent:.2f}%")

    st.divider()

    # ======================================
    # Model Information
    # ======================================

    st.subheader("🤖 Model Information")

    model_info = pd.DataFrame({

        "Property":[
            "Algorithm",
            "Learning Type",
            "Feature Scaling",
            "Input Features",
            "Prediction Labels",
            "Output"
        ],

        "Value":[
            "Isolation Forest",
            "Unsupervised Learning",
            "StandardScaler",
            len(feature_columns),
            "-1 = Anomaly, 1 = Normal",
            "Anomaly Score + Prediction"
        ]

    })

    st.dataframe(model_info,
                 width="stretch")

    st.divider()

    # ======================================
    # Prediction Distribution
    # ======================================

    st.subheader("Prediction Distribution")

    fig = px.pie(

        names=["Normal","Anomaly"],

        values=[normal_count, anomaly_count],

        title="Normal vs Anomaly"

    )

    st.plotly_chart(fig, width="stretch")

    st.divider()

    # ================================
    # SHAP Feature Importance
    # ================================

    st.subheader("Feature Importance")

    if "importance" in st.session_state:

        importance = st.session_state["importance"]

        fig = px.bar(
        importance.head(10),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 10 SHAP Features")

        st.plotly_chart(fig, width="stretch")
    else:

        st.info("Open the SHAP Explainability page once to generate feature importance.")

    st.subheader("Anomaly Score Distribution")

    fig = px.histogram(
    df,
    x="anomaly_score",
    nbins=50,
    title="Distribution of Anomaly Scores")

    st.plotly_chart(fig, width="stretch")

    st.subheader("Top 10 Most Anomalous Records")

    top10 = df.nsmallest(10,"anomaly_score")

    st.dataframe(top10)

    st.subheader("Model Health")

    col1, col2 = st.columns(2)

    with col1:
       st.metric("Features Used", len(feature_columns))

    with col2:
        st.metric("Algorithm", "Isolation Forest")



    # ======================================
    # Model Explanation
    # ======================================

    st.subheader("How Isolation Forest Works")

    st.info("""

    Isolation Forest is an unsupervised anomaly detection algorithm.

    Instead of learning normal and abnormal labels,
    it randomly partitions the feature space.

    Records requiring fewer splits to isolate are
    considered anomalies.

    Prediction Labels

    • 0  → Normal

    • 1 → Anomaly

    The anomaly score indicates how strongly the
    model believes a record is anomalous.

    """)

    st.divider()

    # ======================================
    # Advantages
    # ======================================

    st.subheader("Advantages of Isolation Forest")

    advantages = pd.DataFrame({

        "Advantages":[

            "No labelled data required",

            "Fast prediction",

            "Works well on high-dimensional data",

            "Suitable for rare event detection",

            "Scalable for large datasets"

        ]

    })

    st.table(advantages)

    st.divider()

    # ======================================
    # Project Summary
    # ======================================

    st.success("""

    ✅ Data Cleaning Completed

    ✅ Feature Engineering Completed

    ✅ StandardScaler Applied

    ✅ Isolation Forest Trained

    ✅ Dynamic Prediction Enabled

    ✅ SHAP Explainability Implemented

    ✅ Interactive Dashboard Completed

    """)
