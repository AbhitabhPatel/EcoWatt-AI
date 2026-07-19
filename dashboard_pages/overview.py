import streamlit as st


def show():

    st.header("Project Overview")

    st.markdown("""
### Objective

Detect abnormal energy consumption using Isolation Forest
and explain predictions using SHAP.

### Workflow

- Data Cleaning
- Feature Engineering
- Isolation Forest
- Hyperparameter Tuning
- SHAP Explainability
- Interactive Dashboard
""")