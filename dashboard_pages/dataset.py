import streamlit as st
import pandas as pd

def show(df) :
    st.header("📋 Dataset Explorer")

    st.write(f"Total Records: {len(df):,}")
    st.write(f"Total Features: {df.shape[1]}")

    st.subheader("Dataset Preview")

    columns = st.multiselect(
    "Select Columns",
    df.columns.tolist(),
    default=df.columns.tolist()
    )
    st.dataframe(df[columns])

    if "datetime" in df.columns:

       st.subheader("Search by Datetime")

       search_date = st.text_input(
        "Enter Date (Example: 2007-02-15)")

    if search_date:
        filtered_df = df[
            df["datetime"].astype(str).str.contains(search_date)
        ]

        st.dataframe(filtered_df)

    st.subheader("Dataset Statistics")
    st.dataframe(df.describe())

    st.subheader("Missing Values")
    missing = pd.DataFrame({"Column": df.columns,"Missing Values": df.isnull().sum().values})
    st.dataframe(missing)
