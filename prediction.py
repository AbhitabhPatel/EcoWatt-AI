import joblib
import streamlit as st

from feature_engineering import create_features
from config import (feature_columns, MODEL_PATH, SCALER_PATH)

# ===========================
# Load Model
# ===========================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ===========================
# Load Scaler
# ===========================

@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)


# ===========================
# Predict
# ===========================

@st.cache_data
def prepare_dataset(df):


    df = create_features(df)

    X = df[feature_columns]

    X_scaled = get_scaled_dataset(df)
    model = load_model()

    df["anomaly"] = model.predict(X_scaled)

    df["anomaly_score"] = model.decision_function(X_scaled)

    return df

@st.cache_data
def get_scaled_dataset(df):

    scaler = load_scaler()

    X = df[feature_columns]

    X_scaled = scaler.transform(X)

    return X_scaled

@st.cache_data
def prepare_features(df):
    return create_features(df)