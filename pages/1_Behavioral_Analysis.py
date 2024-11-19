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
    
    # Clustering - KMeans, GaussianMixture, DBSCAN, Agglomerative
    st.subheader("Clustering Analysis")

    # Feature Engineering for clustering
    main_df['Transaction_DateTime'] = pd.to_datetime(main_df['Transaction_DateTime'], errors='coerce')
    main_df['Recency'] = (pd.to_datetime('now') - main_df.groupby('Customer_ID')['Transaction_DateTime'].transform('max')).dt.days
    main_df['Frequency'] = main_df.groupby('Customer_ID')['Transaction_ID'].transform('count')
    main_df['Monetary'] = main_df.groupby('Customer_ID')['Amount'].transform('sum')
    features = ['Recency', 'Frequency', 'Monetary']

    # KMeans Clustering
    st.subheader("KMeans Clustering")
    kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
    kmeans_labels = kmeans.fit_predict(main_df[features])
    main_df['KMeans_Cluster'] = kmeans_labels
    st.write(f"Silhouette Score for KMeans: {silhouette_score(main_df[features], kmeans_labels):.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(main_df['Recency'], main_df['Frequency'], c=kmeans_labels, cmap='viridis')
    ax.set_title('KMeans Clustering')
    ax.set_xlabel('Recency')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Gaussian Mixture Model
    st.subheader("Gaussian Mixture Model (GMM) Clustering")
    gmm = GaussianMixture(n_components=4, random_state=42)
    gmm_labels = gmm.fit_predict(main_df[features])
    main_df['GMM_Cluster'] = gmm_labels
    st.write(f"Silhouette Score for GMM: {silhouette_score(main_df[features], gmm_labels):.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(main_df['Recency'], main_df['Frequency'], c=gmm_labels, cmap='plasma')
    ax.set_title('GMM Clustering')
    ax.set_xlabel('Recency')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Hierarchical Clustering
    st.subheader("Hierarchical Clustering")
    hierarchical = AgglomerativeClustering(n_clusters=4, linkage='ward')
    hierarchical_labels = hierarchical.fit_predict(main_df[features])
    main_df['Hierarchical_Cluster'] = hierarchical_labels
    import scipy.cluster.hierarchy as sch
    fig, ax = plt.subplots(figsize=(10, 7))
    sch.dendrogram(sch.linkage(main_df[features], method='ward'))
    ax.set_title('Dendrogram for Hierarchical Clustering')
    st.pyplot(fig)

    # DBSCAN
    st.subheader("DBSCAN Clustering")
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(main_df[features])
    main_df['DBSCAN_Cluster'] = dbscan_labels
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(main_df['Recency'], main_df['Frequency'], c=dbscan_labels, cmap='plasma')
    ax.set_title('DBSCAN Clustering')
    ax.set_xlabel('Recency')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Mean Shift
    st.subheader("Mean Shift Clustering")
    mean_shift = MeanShift()
    mean_shift_labels = mean_shift.fit_predict(main_df[features])
    main_df['MeanShift_Cluster'] = mean_shift_labels
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(main_df['Recency'], main_df['Frequency'], c=mean_shift_labels, cmap='coolwarm')
    ax.set_title('Mean Shift Clustering')
    ax.set_xlabel('Recency')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # t-SNE Visualization
    # t-SNE Visualization
    st.subheader("t-SNE Visualization of Clusters")
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(main_df[features])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=main_df['KMeans_Cluster'], cmap='viridis')
    ax.set_title('t-SNE Visualization of Clusters')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    st.pyplot(fig)

    # Model Training and Prediction (Optional)
    st.subheader("Model Training - Predicting Customer Segments")
    
    # Prepare features for model training
    model_features = ['Recency', 'Frequency', 'Monetary']
    X = main_df[model_features]
    y = main_df['KMeans_Cluster']  # Use the KMeans clusters as labels for supervised learning
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Random Forest Classifier for Prediction
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    
    # Model Accuracy
    accuracy = rf_classifier.score(X_test, y_test)
    st.write(f"Random Forest Classifier Accuracy: {accuracy:.4f}")

    # Feature Importances
    feature_importances = pd.Series(rf_classifier.feature_importances_, index=model_features).sort_values(ascending=False)
    st.write("Feature Importances:\n", feature_importances)

    # Predict on New Data (Optional)
    st.subheader("Predicting New Customer Segments")
    new_data = st.text_input("Enter new customer data (Recency, Frequency, Monetary) separated by commas:")
    if new_data:
        new_data = np.array([list(map(float, new_data.split(',')))])
        predicted_cluster = rf_classifier.predict(new_data)
        st.write(f"The predicted customer segment is: Cluster {predicted_cluster[0]}")

    # Additional visualizations or insights can be added here


   
