import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import download

 
download('vader_lexicon')
data = pd.read_csv('twitter_training_cleaned.csv', encoding='latin1')
print(data.head())
print("Columns in dataset:", data.columns)
text_column = 'im getting on borderlands and i will murder you all ,'


if text_column not in data.columns:
    raise ValueError(f"The dataset must contain a '{text_column}' column. Available columns: {data.columns}")

data[text_column] = data[text_column].astype(str)
sid = SentimentIntensityAnalyzer()
data['sentiment_score'] = data[text_column].apply(lambda x: sid.polarity_scores(x)['compound'])
data['sentiment'] = data['sentiment_score'].apply(lambda score: 'positive' if score > 0.05 else ('negative' if score < -0.05 else 'neutral'))
plt.figure(figsize=(8, 6))
sns.countplot(x='sentiment', data=data, palette='viridis', hue='sentiment', legend=False)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Number of Posts')
plt.show()
if 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'])
    sentiment_over_time = data.groupby(data['date'].dt.date)['sentiment_score'].mean()
    
    plt.figure(figsize=(10, 6))
    sentiment_over_time.plot()
    plt.title('Sentiment Over Time')
    plt.xlabel('Date')
    plt.ylabel('Average Sentiment Score')
    plt.show()
else:
    print("No 'date' column found. Skipping the sentiment over time analysis.")
