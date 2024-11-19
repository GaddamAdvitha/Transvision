import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, MeanShift
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Streamlit app title
st.title("Customer Behavioral Analysis and Segmentation")

# File upload
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file into a dataframe
    main_df = pd.read_csv(uploaded_file)

    # Display the first few rows of the dataset
    st.subheader("Data Preview")
    st.write(main_df.head())

    # Check columns to ensure 'Transaction_DateTime' exists
    st.write("Columns in the dataset:")
    st.write(main_df.columns)

    # Ensure 'Transaction_DateTime' is correctly parsed as datetime
    if 'Transaction_DateTime' in main_df.columns:
        main_df['Transaction_DateTime'] = pd.to_datetime(main_df['Transaction_DateTime'], errors='coerce')
    else:
        st.write("Column 'Transaction_DateTime' not found!")

    # Transaction Pattern Analysis
    st.subheader("Transaction Pattern Analysis")
    if 'Transaction_DateTime' in main_df.columns:
        main_df['Transaction_Hour'] = main_df['Transaction_DateTime'].dt.hour
        peak_hours = main_df['Transaction_Hour'].value_counts().sort_index()
        st.write("Peak transaction hours:\n", peak_hours)
    else:
        st.write("Unable to perform Transaction Pattern Analysis due to missing 'Transaction_DateTime' column.")
    
    category_preferences = main_df['Category'].value_counts()
    st.write("Category preferences:\n", category_preferences)

    # Customer Loyalty Analysis
    st.subheader("Customer Loyalty Analysis")
    main_df['Loyalty_Level'] = pd.cut(main_df['Customer_Loyalty_Score'], bins=[0, 1, 2, 3, 4, 5], labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    avg_income_by_loyalty = main_df.groupby('Loyalty_Level')['Household_Income'].mean()
    st.write("Average income by loyalty level:\n", avg_income_by_loyalty)
    category_loyalty = main_df.groupby('Loyalty_Level')['Category'].value_counts(normalize=True).unstack().fillna(0)
    st.write("Category preference by loyalty level:\n", category_loyalty)

    # Financial and Spending Analysis
    st.subheader("Financial and Spending Analysis")
    spending_behavior = main_df.groupby('Customer_ID')['Amount'].agg(['mean', 'sum']).rename(columns={'mean': 'Avg_Spend', 'sum': 'Total_Spend'})
    st.write("Spending behavior per customer:\n", spending_behavior.head())

    # Cohort Analysis
    st.subheader("Cohort Analysis")
    main_df['Signup_Date'] = pd.to_datetime(main_df['Signup_Date'], errors='coerce')
    main_df['Signup_YearMonth'] = main_df['Signup_Date'].dt.to_period('M')
    cohort_spending = main_df.groupby('Signup_YearMonth')['Amount'].sum()
    st.write("Total spending by signup cohort:\n", cohort_spending)

    # Customer Lifetime Value (CLTV) Prediction
    st.subheader("Customer Lifetime Value (CLTV) Prediction")
    cltv_data = main_df.groupby('Customer_ID').agg({
        'Amount': 'mean',
        'Transaction_ID': 'count',
        'Signup_Date': 'min'
    }).rename(columns={'Amount': 'Avg_Transaction_Value', 'Transaction_ID': 'Transaction_Frequency'})
    main_df['Last_Transaction_Date'] = main_df.groupby('Customer_ID')['Transaction_DateTime'].transform('max')
    main_df['Customer_Lifetime'] = (main_df['Last_Transaction_Date'] - main_df['Signup_Date']).dt.days / 30
    cltv_data['Customer_Lifetime'] = main_df.groupby('Customer_ID')['Customer_Lifetime'].first()
    cltv_data['CLTV'] = cltv_data['Avg_Transaction_Value'] * cltv_data['Transaction_Frequency'] / cltv_data['Customer_Lifetime']
    st.write("Customer Lifetime Value (CLTV):\n", cltv_data[['CLTV']].head())

    # Time-Series Analysis of Transactions
    st.subheader("Time-Series Analysis of Transactions")
    if 'Transaction_DateTime' in main_df.columns:
        main_df.set_index('Transaction_DateTime', inplace=True)
        monthly_transactions = main_df['Amount'].resample('M').sum()
        monthly_avg_transaction = main_df['Amount'].resample('M').mean()
        fig, ax = plt.subplots(figsize=(12, 6))
        monthly_transactions.plot(label='Monthly Total Transactions', ax=ax)
        monthly_avg_transaction.plot(label='Monthly Average Transaction', linestyle='--', ax=ax)
        ax.set_title('Monthly Transaction Trends')
        ax.set_xlabel('Month')
        ax.set_ylabel('Transaction Amount')
        ax.legend()
        st.pyplot(fig)
    else:
        st.write("Unable to perform Time-Series Analysis due to missing 'Transaction_DateTime' column.")
    


    # Outliers Detection using Boxplot
    st.subheader("Outliers Detection")
    fig = plt.figure(figsize=(10, 6))
    sns.boxplot(x=main_df['Amount'])
    plt.title('Boxplot of Transaction Amounts')
    st.pyplot(fig)

    # Bivariate Analysis: Spending Behavior by Age Group
    st.subheader("Bivariate Analysis (Exploring Relationships)")
        # Ensure 'Age' column is numeric and create age groups
    main_df['Age'] = pd.to_numeric(main_df['Age'], errors='coerce')
    main_df['Age_Group'] = pd.cut(main_df['Age'], bins=[18, 30, 40, 50, 60, 100], labels=['18-30', '30-40', '40-50', '50-60', '60+'])
        
        # Drop any rows where 'Age_Group' or 'Amount' is NaN
    main_df = main_df.dropna(subset=['Age_Group', 'Amount'])
        
    fig = plt.figure(figsize=(10, 6))
    sns.boxplot(x='Age_Group', y='Amount', data=main_df)
    plt.title('Spending Behavior by Age Group')
    st.pyplot(fig)

        # Distributions and Skewness of Numerical Variables
    st.subheader("Distributions and Skewness of Numerical Variables")
    numerical_columns = ['Amount', 'Age', 'Old_Balance', 'New_Balance', 'Customer_Loyalty_Score']
    for column in numerical_columns:
        fig = plt.figure(figsize=(10, 6))
        sns.histplot(main_df[column], kde=True)
        plt.title(f'Distribution of {column}')
        st.pyplot(fig)



    
    
    

   
