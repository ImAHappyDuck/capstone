import pandas as pd


df = pd.read_csv('cleanedFinNews.csv')
print(df.count())
print(df['pos_score'].max())
