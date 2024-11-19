import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm  # Simple progress bar

# Load your data into Streamlit (assuming you have a CSV file to upload)
# You can upload the CSV using the file uploader in Streamlit
st.title("Sentiment Analysis using VADER")

uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

if uploaded_file is not None:
    # Load the dataframe
    main_df = pd.read_csv(uploaded_file)

    # Display the first few rows
    st.subheader("Data Preview")
    st.write(main_df.head())

    # Show data shape
    st.write(f"Data Shape: {main_df.shape}")
    
    # Plot the count of reviews by ratings
    st.subheader("Count of Reviews by Ratings")
    fig, ax = plt.subplots(figsize=(10, 5))
    main_df['Review_Rating'].value_counts().sort_index().plot(kind='bar', ax=ax, title='Count of Reviews by Ratings')
    ax.set_xlabel('Review Rating')
    st.pyplot(fig)

    # VADER Sentiment Analysis
    st.subheader("VADER Sentiment Scoring")
    
    # Download VADER lexicon for sentiment analysis
    nltk.download('vader_lexicon')

    # Initialize SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    # Run polarity score on the entire dataset
    st.write("Running polarity score on the dataset...")
    res = {}
    for i, row in tqdm(main_df.iterrows(), total=len(main_df), desc="Processing Reviews"):
        text = row['Review_Text']
        myid = row['Transaction_ID']
        res[myid] = sia.polarity_scores(text)

    # Create a DataFrame with the VADER results
    vaders = pd.DataFrame(res).T
    vaders = vaders.reset_index().rename(columns={'index': 'Transaction_ID'})
    vaders = vaders.merge(main_df, how='left')

    # Display the first few rows of VADER sentiment scores
    st.subheader("VADER Sentiment Scores")
    st.write(vaders.head())

    # Plot Compound Score of Reviews
    st.subheader("Compound Score of Reviews")
    fig, ax = plt.subplots()
    sns.barplot(data=vaders, x='Review_Rating', y='compound', ax=ax)
    ax.set_title('Compound Score of Reviews')
    st.pyplot(fig)

    # Plot Positive, Neutral, and Negative Sentiments
    st.subheader("Positive, Neutral, Negative Sentiments")
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    sns.barplot(data=vaders, x='Review_Rating', y='pos', ax=axs[0])
    sns.barplot(data=vaders, x='Review_Rating', y='neu', ax=axs[1])
    sns.barplot(data=vaders, x='Review_Rating', y='neg', ax=axs[2])
    axs[0].set_title('Positive')
    axs[1].set_title('Neutral')
    axs[2].set_title('Negative')
    plt.tight_layout()
    st.pyplot(fig)

    # Group by category and calculate the mean for each sentiment score
    category_sentiment = vaders.groupby('Category')[['neg', 'neu', 'pos', 'compound']].mean().reset_index()
    st.subheader("Average Sentiment Scores by Category")
    st.write(category_sentiment)

    # Bar chart for each Category
    st.subheader("Average Sentiment Scores by Category (Bar Chart)")
    fig, ax = plt.subplots(figsize=(10, 6))
    category_sentiment.set_index('Category')[['neg', 'neu', 'pos', 'compound']].plot(kind='bar', ax=ax)
    ax.set_title('Average Sentiment Scores by Category')
    ax.set_ylabel('Sentiment Score')
    ax.set_xlabel('Category')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Pie charts for each category
    for category in category_sentiment['Category']:
        data = category_sentiment[category_sentiment['Category'] == category][['neg', 'neu', 'pos']]
        sentiment_labels = ['Negative', 'Neutral', 'Positive']
        sentiment_data = [data['neg'].values[0], data['neu'].values[0], data['pos'].values[0]]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(sentiment_data, labels=sentiment_labels, autopct='%1.1f%%', startangle=90)
        ax.set_title(f'Sentiment Distribution for {category}')
        ax.axis('equal')
        st.pyplot(fig)

    # Stacked bar chart for sentiment proportions (neg, neu, pos)
    st.subheader("Sentiment Proportions by Category (Stacked Bar Chart)")
    fig, ax = plt.subplots(figsize=(10, 6))
    category_sentiment.set_index('Category')[['neg', 'neu', 'pos']].plot(kind='bar', stacked=True, ax=ax, color=['red', 'gray', 'green'])
    ax.set_title('Sentiment Proportions by Category')
    ax.set_ylabel('Sentiment Proportion')
    ax.set_xlabel('Category')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Bar plot for compound score across categories
    st.subheader("Compound Sentiment Score by Category (Bar Plot)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Category', y='compound', data=category_sentiment, palette='cool', ax=ax)
    ax.set_title('Compound Sentiment Score by Category')
    ax.set_ylabel('Compound Sentiment Score')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Boxplot of sentiment scores
    st.subheader("Distribution of Compound Sentiment Scores by Category (Boxplot)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Category', y='compound', data=vaders, palette='Set2', ax=ax)
    ax.set_title('Distribution of Compound Sentiment Scores by Category')
    ax.set_xlabel('Category')
    ax.set_ylabel('Compound Sentiment Score')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Histogram for compound sentiment score by category
    st.subheader("Histogram of Compound Sentiment Scores by Category")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=vaders, x='compound', hue='Category', multiple="stack", bins=20, kde=True, ax=ax)
    ax.set_title('Histogram of Compound Sentiment Scores by Category')
    ax.set_xlabel('Compound Sentiment Score')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Heatmap of sentiment scores
    st.subheader("Heatmap of Sentiment Scores by Category")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(category_sentiment.set_index('Category')[['neg', 'neu', 'pos', 'compound']], annot=True, cmap='coolwarm', cbar=True, ax=ax)
    ax.set_title('Heatmap of Sentiment Scores by Category')
    ax.set_ylabel('Category')
    st.pyplot(fig)

    # Boxplot for Customer Loyalty Score vs Compound Sentiment
    st.subheader("Customer Loyalty Score vs Compound Sentiment (Boxplot)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Customer_Loyalty_Score', y='compound', data=vaders, palette='cool', ax=ax)
    ax.set_title('Customer Loyalty Score vs Compound Sentiment')
    ax.set_xlabel('Customer Loyalty Score')
    ax.set_ylabel('Compound Sentiment Score')
    st.pyplot(fig)

    # Boxplot for Sentiment by Review Rating
    st.subheader("Sentiment by Review Rating (Box Plot)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Review_Rating', y='compound', data=vaders, palette='Set1', ax=ax)
    ax.set_title('Sentiment by Review Rating')
    ax.set_xlabel('Review Rating')
    ax.set_ylabel('Compound Sentiment Score')
    st.pyplot(fig)

    # Bar Plot Sentiment by Customer Gender
    st.subheader("Sentiment by Customer Gender (Bar Plot)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Gender', y='compound', data=vaders, palette='cool', ax=ax)
    ax.set_title('Sentiment by Customer Gender')
    ax.set_xlabel('Gender')
    ax.set_ylabel('Compound Sentiment Score')
    st.pyplot(fig)

    # Box Plot for Sentiment by Customer Age
    st.subheader("Sentiment by Customer Age (Box Plot)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Age', y='compound', data=vaders, palette='coolwarm', ax=ax)
    ax.set_title('Sentiment by Customer Age')
    ax.set_xlabel('Age')
    ax.set_ylabel('Compound Sentiment Score')
    st.pyplot(fig)

