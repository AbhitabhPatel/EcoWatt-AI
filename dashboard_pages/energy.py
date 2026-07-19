import streamlit as st 
import pandas as pd 
import plotly.express as px 

def show(df):
    st.header("📈 Energy Consumption Analysis")
    energy_columns = [
    col for col in df.columns
    if df[col].dtype != "object"]

    selected_feature = st.selectbox("Select Energy Feature",energy_columns)
    df["datetime"] = pd.to_datetime(df["datetime"])
    fig = px.line(df,x="datetime",y=selected_feature,title=f"{selected_feature} Over Time")
    st.plotly_chart(fig, width="stretch")
    
    st.subheader("Feature Statistics")
    st.write(df[selected_feature].describe())
    fig = px.histogram(
    df,
    x=selected_feature,
    nbins=50,
    title=f"Distribution of {selected_feature}")
    st.plotly_chart(fig, width="stretch")

    fig = px.box(
    df,
    y=selected_feature,
    title=f"Box Plot of {selected_feature}")
    st.plotly_chart(fig, width="stretch")

    numeric_df = df.select_dtypes(include="number")

    corr = numeric_df.corr()
    fig = px.imshow(corr,text_auto=".2f",aspect="auto",title="Correlation Heatmap")
    st.plotly_chart(fig, width="stretch")
