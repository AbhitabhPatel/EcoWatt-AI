import streamlit as st 
import pandas as pd 
import plotly.express as px 
import matplotlib.pyplot as plt 
from shap_utils import (load_explainer, calculate_shap_values, get_feature_importance)
import shap


def show(df,feature_columns,scaler):
    st.header("🧠 SHAP Explainability")

    st.write("""
    SHAP explains why the Isolation Forest predicted a
    particular record as normal or anomalous.
    """)

    record_index = st.slider(
    "Select Record",
    0,
    len(df)-1,
    0)

    (explainer,shap_values,sample,sample_features,sample_scaled) = calculate_shap_values(df,feature_columns,scaler,record_index)

    st.subheader("Selected Record")

    st.dataframe(sample_features)
    
    st.subheader("Prediction Summary")

    prediction = sample["anomaly"].iloc[0]

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
           st.error("🚨 Anomaly Detected")
        else:
           st.success("✅ Normal Behaviour")

    with col2:
        if "anomaly_score" in sample.columns:
            st.metric(
            "Decision Score",
            f"{sample['anomaly_score'].iloc[0]:.4f}"
        )
    
    fig = plt.figure(figsize=(10,6))

    explanation = shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=sample_scaled[0],
    feature_names=feature_columns)

    shap.plots.waterfall(
    explanation,
    show=False)

    st.pyplot(fig)

    plt.close(fig)

    importance = get_feature_importance(feature_columns, shap_values)
    
    st.session_state["importance"] = importance

    st.subheader("Feature Importance")

    st.dataframe(importance)

    fig = px.bar(

    importance.head(10),

    x="SHAP Value",

    y="Feature",

    orientation="h",

    title="Top 10 SHAP Feature Contributions")

    st.plotly_chart(fig, width="stretch")

    top3 = importance.head(3)["Feature"].tolist()

    st.info(f"""
    ### AI Explanation

    The Isolation Forest classified this record using all engineered features.

    The strongest contributors were:

    • {top3[0]}
    • {top3[1]}
    • {top3[2]}

    These variables deviated most from the learned normal energy consumption pattern, resulting in the current prediction.
    """)
