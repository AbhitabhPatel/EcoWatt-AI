import shap
import pandas as pd
import streamlit as st

from prediction import (load_model,get_scaled_dataset)

@st.cache_data
def get_background(df):

    scaled_dataset = get_scaled_dataset(df)

    return shap.sample(
        scaled_dataset,
        30,
        random_state=42
    )

@st.cache_resource
def load_explainer(background):
    model = load_model()

    return shap.KernelExplainer(
        model.decision_function,
        background
    )

def calculate_shap_values(df, feature_columns, scaler, record_index):

    sample = df.iloc[[record_index]]

    sample_features = sample[feature_columns]

    sample_scaled = scaler.transform(sample_features)

    background = get_background(df)

    explainer = load_explainer(background)

    shap_values = explainer.shap_values(sample_scaled)

    return (
        explainer,
        shap_values,
        sample,
        sample_features,
        sample_scaled
    )


def get_feature_importance(feature_columns, shap_values):

    importance = pd.DataFrame({
        "Feature": feature_columns,
        "SHAP Value": shap_values[0]
    })

    importance["Importance"] = importance["SHAP Value"].abs()

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    return importance
