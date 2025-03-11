import pandas as pd
import plotly.express as px 
from scipy.stats import pearsonr, chi2_contingency
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, date

#Add sentiment columns to the news dataset
df = pd.read_csv('cleanedFinNews.csv', nrows=100, dtype=str).fillna('')

def pos_score(sentence: str):
    sentence_analyzer = SentimentIntensityAnalyzer()
    temp_sentence = sentence
    sentiment_result = sentence_analyzer.polarity_scores(temp_sentence)
    return (sentiment_result['pos'])

# returns the negative sentiment intensity for a sentence
def neg_score(sentence: str):
    sentence_analyzer = SentimentIntensityAnalyzer()
    temp_sentence = sentence
    sentiment_result = sentence_analyzer.polarity_scores(temp_sentence)
    return (sentiment_result['neg'])

# returns the neutral sentiment intensity for a sentence
def neu_score(sentence: str):
    sentence_analyzer = SentimentIntensityAnalyzer()
    temp_sentence = sentence
    sentiment_result = sentence_analyzer.polarity_scores(temp_sentence)
    return (sentiment_result['neu'])

columns_to_iterate = [df.columns[2]] + list(df.columns[6:])
df['pos_score'] = df.apply(lambda row: sum(pos_score(row[col]) for col in columns_to_iterate)/6, axis=1)
df['neg_score'] = df.apply(lambda row: sum(neg_score(row[col]) for col in columns_to_iterate)/6, axis=1)
df['neu_score'] = df.apply(lambda row: sum(neu_score(row[col]) for col in columns_to_iterate)/6, axis=1)


# print(df[['pos_score', 'neg_score', 'neu_score']])

