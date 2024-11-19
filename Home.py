
import streamlit as st

# Main Page Title and Description
st.title("Customer Behavioral Analysis and Sentiment Insights")
st.subheader("Welcome to the Data Analysis Platform!")
st.write("""
This application offers interactive visualizations and machine learning models for:
- Fraud Detection
- Customer Behavioral Analysis
- Clustering and Segmentation
- Sentiment Analysis

Explore detailed insights by navigating to respective pages using the sidebar.
""")

# Provide Dataset Download Option
st.header("Download Dataset")
with open("merged_main_df.csv", "rb") as file:
    st.download_button("Download Dataset", file, "merged_main_df.csv")

# Provide Project Files Download as a ZIP
st.header("Download Project Files")
if st.button("Download Complete Project"):
    zip_path = "project.zip"
    with open(zip_path, "rb") as zf:
        st.download_button("Download Project ZIP", zf, "project.zip")
