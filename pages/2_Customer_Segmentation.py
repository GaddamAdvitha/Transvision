import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch

# Streamlit app title
st.title("Clustering Analysis for Customer Segmentation")

# File upload
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    main_df = pd.read_csv(uploaded_file)

    # Ensure 'Transaction_DateTime' exists and is parsed correctly
    if 'Transaction_DateTime' in main_df.columns:
        main_df['Transaction_DateTime'] = pd.to_datetime(main_df['Transaction_DateTime'], errors='coerce')
        main_df['Recency'] = (pd.to_datetime('now') - main_df.groupby('Customer_ID')['Transaction_DateTime'].transform('max')).dt.days
        main_df['Frequency'] = main_df.groupby('Customer_ID')['Transaction_ID'].transform('count')
        main_df['Monetary'] = main_df.groupby('Customer_ID')['Amount'].transform('sum')
        features = ['Recency', 'Frequency', 'Monetary']

        # Check for missing values
        st.write("Missing values in clustering features:")
        st.write(main_df[features].isnull().sum())

        # Handle missing values (Impute with mean)
        main_df[features] = main_df[features].fillna(main_df[features].mean())
        st.write("After imputing missing values:")
        st.write(main_df[features].isnull().sum())

        # Clustering Model Selection
        clustering_model = st.sidebar.selectbox(
            "Choose a clustering model:",
            ["KMeans", "Gaussian Mixture (GMM)", "Hierarchical", "DBSCAN", "Mean Shift"]
        )

        # KMeans Clustering
        if clustering_model == "KMeans":
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

        # Gaussian Mixture Model (GMM)
        elif clustering_model == "Gaussian Mixture (GMM)":
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
        elif clustering_model == "Hierarchical":
            st.subheader("Hierarchical Clustering")
            hierarchical = AgglomerativeClustering(n_clusters=4, linkage='ward')
            hierarchical_labels = hierarchical.fit_predict(main_df[features])
            main_df['Hierarchical_Cluster'] = hierarchical_labels

            fig, ax = plt.subplots(figsize=(10, 7))
            sch.dendrogram(sch.linkage(main_df[features], method='ward'), ax=ax)
            ax.set_title('Dendrogram for Hierarchical Clustering')
            st.pyplot(fig)

        # DBSCAN Clustering
        elif clustering_model == "DBSCAN":
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

        # Mean Shift Clustering
        elif clustering_model == "Mean Shift":
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

        #t-SNE Visualization
        # st.subheader("t-SNE Visualization of Clusters")
        # tsne = TSNE(n_components=2, random_state=42)
        # tsne_results = tsne.fit_transform(main_df[features])
        # fig, ax = plt.subplots(figsize=(10, 6))
        # ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=kmeans_labels, cmap='viridis')
        # ax.set_title('t-SNE Visualization of Clusters')
        # ax.set_xlabel('t-SNE Component 1')
        # ax.set_ylabel('t-SNE Component 2')
        # st.pyplot(fig)

    else:
        st.write("Error: 'Transaction_DateTime' column is missing from the dataset.")
