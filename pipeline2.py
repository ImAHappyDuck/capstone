import pandas as pd
import plotly.express as px 
from scipy.stats import pearsonr, chi2_contingency
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, date

#Add sentiment columns to the news dataset
df = pd.read_csv('cleanedFinNews.csv', dtype=str).fillna('')

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
df['pos_score'] = df.apply(lambda row: sum(pos_score(row[col]) for col in columns_to_iterate)/5, axis=1)
df['neg_score'] = df.apply(lambda row: sum(neg_score(row[col]) for col in columns_to_iterate)/5, axis=1)
df['neu_score'] = df.apply(lambda row: sum(neu_score(row[col]) for col in columns_to_iterate)/5, axis=1)


print(df[['pos_score', 'neg_score', 'neu_score']])

df.to_csv('cleanedFinNews.csv', index=False)

# # add stock prices and option profitablilty to optData
# df2 = pd.read_csv('cleaned_optData.csv')
# import pandas as pd
# import zipfile

# # Assume df2 is your original DataFrame
# df2['expiration'] = pd.to_datetime(df2['expiration'])

# # Path to the ZIP archive containing the full_history folder
# zip_file_path = 'full_history.zip'

# # Dictionary to store loaded stock price data for tickers
# price_cache = {}

# # List to hold the fetched stock price data
# price_data = []

# # Open the ZIP file
# with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
#     # List of all files inside the 'full_history' folder in the zip
#     zip_file_list = zip_file.namelist()

#     # Iterate over each unique ticker and expiration date in df2
#     for ticker, expiration in df2[['act_symbol', 'expiration']].drop_duplicates().values:
#         # Check if the ticker's stock data has already been loaded
#         if ticker not in price_cache:
#             # Construct the name of the ticker's CSV file inside the full_history folder
#             ticker_file = f"full_history/{ticker}.csv"
            
#             # Check if the CSV file exists in the ZIP archive
#             if ticker_file in zip_file_list:
#                 try:
#                     # Load the CSV file for the ticker directly from the ZIP archive
#                     with zip_file.open(ticker_file) as file:
#                         stock_data = pd.read_csv(file, parse_dates=['date'])
                    
#                     # Cache the stock data
#                     price_cache[ticker] = stock_data
#                 except Exception as e:
#                     print(f"Error reading {ticker_file}: {e}")
#                     price_cache[ticker] = None
#             else:
#                 print(f"File {ticker_file} not found in the ZIP archive.")
#                 price_cache[ticker] = None
        
#         # If stock_data is None, skip processing this ticker
#         stock_data = price_cache.get(ticker)
#         if stock_data is not None:
#             # Filter the stock data for the given expiration date
#             expiration_data = stock_data[stock_data['date'] == expiration]
            
#             if not expiration_data.empty:
#                 close_price = expiration_data['close'].iloc[0]
                
#                 # Append the price data for each row in the corresponding group
#                 for _ in range(len(df2[(df2['act_symbol'] == ticker) & (df2['expiration'] == expiration)])):
#                     price_data.append({'act_symbol': ticker, 'expiration': expiration, 'close_price': close_price})
#         else:
#             print(f"No stock data available for ticker {ticker}, skipping.")

# # Convert the price data list to a DataFrame
# price_data_df = pd.DataFrame(price_data)

# # Merge price data with the original DataFrame (df2)
# df2 = df2.merge(price_data_df, on=['act_symbol', 'expiration'], how='left')

# # Calculate moneyness (In the money or out of the money)
# df2['moneyness'] = df2['close_price'] - df2['strike']

# print(df2.head())
# df2.to_csv('cleaned_optData.csv', index=False)