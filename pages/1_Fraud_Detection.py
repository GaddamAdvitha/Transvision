import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler

# Title and Introduction
st.title("Fraud Detection Using Machine Learning Models")
st.write("""
This application demonstrates fraud detection in transactions using various machine learning models. 
You can explore data preprocessing, model training, and evaluation through this interface.
""")

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose Mode", ["Upload Data", "Data Preprocessing", "Model Training", "Visualization"])

# Initialize session state for the DataFrame
if 'main_df' not in st.session_state:
    st.session_state.main_df = None
if 'df' not in st.session_state:
    st.session_state.df = None

if app_mode == "Upload Data":
    # Upload CSV file
    st.subheader("Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        st.session_state.main_df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Dataset Preview")
        st.write(st.session_state.main_df.head())

if app_mode == "Data Preprocessing":
    if st.session_state.main_df is None:
        st.warning("Please upload a dataset first.")
    else:
        # Data info and preprocessing steps
        st.subheader("Data Preprocessing")
        main_df = st.session_state.main_df
        st.write("### Raw Data Information")
        st.write(main_df.info())

        # Display initial data
        st.write("### First few rows of the dataset")
        st.write(main_df.head())

        st.write("### Handling Missing Data")
        st.write(main_df.isna().sum())

        # Data cleaning and feature encoding
        df = main_df.copy()
        df.drop(['Transaction_ID', 'Customer_ID', 'Transaction_DateTime', 'Merchant_ID', 'Merchant_Location', 
                 'Payment_Type', 'Age', 'Gender', 'Marital_Status', 'Location_Type', 'Location_City', 
                 'Occupation', 'Household_Income', 'Life_Stage', 'Preferred_Language', 'Interests_Hobbies', 
                 'Customer_Loyalty_Score', 'Signup_Date', 'Review_ID', 'Category_Review', 'Review_Text', 
                 'Review_Rating'], axis=1, inplace=True)
        st.write("### Cleaned Data")
        st.write(df.head())

        # One Hot Encoding
        df['Category'] = df['Category'].astype(str)
        df['Transaction_Type'] = df['Transaction_Type'].astype(str)
        df = pd.get_dummies(df, columns=['Category', 'Transaction_Type'], prefix=['Category', 'Transaction_Type'], drop_first=False)
        st.write("### Data after One Hot Encoding")
        st.write(df.head())

        # Normalizing Amount, Old_Balance, New_Balance
        standard = MinMaxScaler()
        df['Amount'] = standard.fit_transform(df['Amount'].values.reshape(-1, 1))
        df['Old_Balance'] = standard.fit_transform(df['Old_Balance'].values.reshape(-1, 1))
        df['New_Balance'] = standard.fit_transform(df['New_Balance'].values.reshape(-1, 1))
        st.write("### Data after Normalization")
        st.write(df.head())

        # Save processed data to session state for later use
        st.session_state.df = df

if app_mode == "Model Training":
    # Model Training & Evaluation
    st.subheader("Model Training & Evaluation")

    if st.session_state.df is None:
        st.warning("Data preprocessing must be completed first.")
    else:
        # Data cleaning step: Drop non-numeric columns or convert them
        df = st.session_state.df.copy()

        # Drop columns that are identifiers or non-numeric
        non_numeric_columns = ['Transaction_ID', 'Customer_ID', 'Transaction_DateTime', 'Merchant_ID', 'Merchant_Location', 
                               'Payment_Type', 'Age', 'Gender', 'Marital_Status', 'Location_Type', 'Location_City', 
                               'Occupation', 'Household_Income', 'Life_Stage', 'Preferred_Language', 'Interests_Hobbies', 
                               'Customer_Loyalty_Score', 'Signup_Date', 'Review_ID', 'Category_Review', 'Review_Text', 
                               'Review_Rating']
        
        df = df.drop(columns=non_numeric_columns, errors='ignore')  # Drop non-numeric columns

        # Make sure all remaining columns are numeric
        df = df.select_dtypes(include=[np.number])

        # Ensure there are no non-numeric columns in X
        X = df.drop('IsFraud', axis=1)
        y = df['IsFraud']

        # SMOTE Handling for Imbalanced Dataset
        st.write("### Handling Imbalanced Dataset using SMOTE")
        sm = SMOTE(sampling_strategy='minority', random_state=42)
        X_smote, y_smote = sm.fit_resample(X, y)

        # Model Training Functions
        def train_model(model, X, y):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            st.write("### Model Performance")
            y_preds = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_preds) * 100
            st.write(f"Accuracy: {accuracy:.2f}%")
            st.write("### Classification Report")
            st.text(classification_report(y_test, y_preds))
            st.write("### Confusion Matrix")
            cf_matrix = confusion_matrix(y_test, y_preds)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cf_matrix, annot=True, cmap='coolwarm', fmt='g', ax=ax)
            ax.set(xlabel="Predicted Label", ylabel="Actual Label")
            st.pyplot(fig)

        # Models to train
        models = {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "XGBoost": XGBClassifier(),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB()
        }

        # Button to train all models
        if st.button("Train All Models"):
            for model_name, model in models.items():
                st.write(f"### Training {model_name}...")
                train_model(model, X_smote, y_smote)

if app_mode == "Visualization":
    # Visualizations
    st.subheader("Data Visualizations")
    if st.session_state.main_df is None:
        st.warning("Please upload a dataset first.")
    else:
        main_df = st.session_state.main_df
        st.write("### Class Distribution")
        class_count_df = pd.DataFrame(main_df['IsFraud'].value_counts().rename_axis('Class').reset_index(name='Counts'))
        class_count_df['Class'].replace({0: 'Normal', 1: 'Fraudulent'}, inplace=True)
        fig = plt.figure()
        ax = sns.barplot(x=class_count_df['Class'], y=class_count_df['Counts'])
        ax.bar_label(ax.containers[0], color='red')
        ax.set_xticklabels(labels=list(class_count_df['Class']), c='blue', rotation=0, fontsize=10, fontweight='bold')
        plt.xlabel('Type of Transactions', fontsize=14, fontweight='bold').set_color('purple')
        plt.ylabel('Frequency', fontsize=14, fontweight='bold').set_color('purple')
        plt.title('Count Values of Normal vs Fraud Class', fontsize=24, fontweight='bold').set_color('purple')
        st.pyplot(fig)

        st.write("### Transaction Amount Distribution by Class")
        fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax0.hist(main_df[main_df['IsFraud'] == 1]['Amount'], bins=50, color='red')
        ax0.set_title('Fraud')
        ax1.hist(main_df[main_df['IsFraud'] == 0]['Amount'], bins=50, color='blue')
        ax1.set_title('Normal')
        ax0.set_ylabel('No. of Transactions')
        ax1.set_ylabel('No. of Transactions')
        plt.xlabel('Amount ($)')
        plt.yscale('log')
        st.pyplot(fig)
