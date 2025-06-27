import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Step 1: Load sample tweets
tweets = pd.read_csv('sample/mental_health_tweets.csv')  # Download dataset from Kaggle

# Step 2: Clean text
def clean_text(text):
    import re
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^A-Za-z\s]", "", text)  # remove non-letters
    return text.lower()

tweets['cleaned'] = tweets['text'].apply(clean_text)

# Step 3: Sentiment analysis
def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

tweets['sentiment'] = tweets['cleaned'].apply(get_sentiment)

# Step 4: Visualization
tweets['sentiment'].value_counts().plot(kind='bar', title='Sentiment Analysis of Mental Health Tweets')
plt.show()