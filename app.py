import streamlit as st

# -----------------------------
# Config
# -----------------------------

from config import (
    feature_columns,
    required_columns
)

# -----------------------------
# Utilities
# -----------------------------

from utils import (
    load_data,
    prepare_dashboard_data
)

# -----------------------------
# Prediction
# -----------------------------

from prediction import (
    prepare_dataset,
    load_model,
    load_scaler,
    prepare_features
)

# -----------------------------
# Pages
# -----------------------------

from dashboard_pages import (
    overview,
    dataset,
    energy,
    anomaly,
    shap_page,
    performance
)

# -----------------------------
# Page Config
# -----------------------------

st.set_page_config(
    page_title="EcoWatt AI",
    page_icon="⚡",
    layout="wide"
)

# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("EcoWatt AI")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"]
)

# -----------------------------
# Load Data
# -----------------------------

df = load_data(uploaded_file)

df = prepare_dashboard_data(df)

# -----------------------------
# Load Model
# -----------------------------

model = load_model()

scaler = load_scaler()

# -----------------------------
# Prediction
# -----------------------------

if uploaded_file is not None:
    df = prepare_dataset(df)
    st.sidebar.success("Custom Dataset Loaded")

else:
    df = prepare_features(df)
    st.sidebar.info("Using Default Dataset")

# -----------------------------
# Validation
# -----------------------------

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error("Dataset Missing Required Columns")

    st.write(missing_columns)

    st.stop()


# -----------------------------
# Header
# -----------------------------

st.title("⚡ EcoWatt AI")

st.subheader(
    "AI-Based Energy Consumption Optimization and Anomaly Detection"
)

# -----------------------------
# Navigation
# -----------------------------

page = st.sidebar.radio(

    "Navigation",

    [

        "Project Overview",

        "Dataset Explorer",

        "Energy Analysis",

        "Anomaly Detection",

        "SHAP Explainability",

        "Model Performance"

    ]

)

# -----------------------------
# Routing
# -----------------------------

if page == "Project Overview":

    overview.show()

elif page == "Dataset Explorer":

    dataset.show(df)

elif page == "Energy Analysis":

    energy.show(df)

elif page == "Anomaly Detection":

    anomaly.show(df, feature_columns)

elif page == "SHAP Explainability":

    shap_page.show(
        df,
        feature_columns,
        scaler
    )

elif page == "Model Performance":

    performance.show(
        df,
        feature_columns
    )

# -----------------------------
# Footer
# -----------------------------

st.markdown("---")

st.caption(
    "EcoWatt AI • Isolation Forest • SHAP • Streamlit • Plotly"
)