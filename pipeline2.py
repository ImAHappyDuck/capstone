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


# print(df[['pos_score', 'neg_score', 'neu_score']])

df.to_csv('cleanedFinNews.csv', index=False)




import pandas as pd
import zipfile
from datetime import timedelta

df2 = pd.read_csv('cleaned_optData.csv')
df2['expiration'] = pd.to_datetime(df2['expiration'])
zip_file_path = 'full_history.zip'
price_cache = {}

# Create a new DataFrame to collect all ticker/expiration pairs
all_pairs = df2[['act_symbol', 'expiration']].drop_duplicates().reset_index(drop=True)
all_pairs['close_price'] = None  # Initialize with None

# Open the ZIP file
with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
    # Get list of all files in the ZIP
    zip_file_list = zip_file.namelist()
    zip_tickers = [f.split('/')[-1].split('.')[0] for f in zip_file_list if f.startswith('full_history/') and f.endswith('.csv')]
    
    # Log how many tickers are available
    print(f"Found {len(zip_tickers)} ticker files in ZIP archive")
    
    # Process each ticker only once
    for ticker in all_pairs['act_symbol'].unique():
        # Construct the name of the ticker's CSV file inside the full_history folder
        ticker_file = f"full_history/{ticker}.csv"
        
        # Check if the CSV file exists in the ZIP archive
        if ticker_file in zip_file_list:
            try:
                # Load the CSV file for the ticker directly from the ZIP archive
                with zip_file.open(ticker_file) as file:
                    stock_data = pd.read_csv(file)
                    # Ensure date column is in datetime format
                    stock_data['date'] = pd.to_datetime(stock_data['date'])
                    
                # Cache the stock data
                price_cache[ticker] = stock_data
                print(f"Loaded price data for {ticker}: {len(stock_data)} records")
            except Exception as e:
                print(f"Error reading {ticker_file}: {e}")
        else:
            print(f"File for ticker {ticker} not found in the ZIP archive")
    
    # Now process each ticker/expiration pair
    for index, row in all_pairs.iterrows():
        ticker = row['act_symbol']
        expiration_date = row['expiration']
        
        # Skip if we don't have data for this ticker
        if ticker not in price_cache:
            continue
            
        stock_data = price_cache[ticker]
        
        # Try to find an exact match for the expiration date
        expiration_data = stock_data[stock_data['date'] == expiration_date]
        
        # If no exact match, try one day before (some options expire on weekends)
        if expiration_data.empty:
            # Try looking for the trading day before expiration
            for days_back in range(1, 5):  # Check up to 4 days back (handles weekends and holidays)
                check_date = expiration_date - timedelta(days=days_back)
                expiration_data = stock_data[stock_data['date'] == check_date]
                if not expiration_data.empty:
                    print(f"Found price for {ticker} using date {check_date} instead of {expiration_date}")
                    break
        
        # If we found data, update the close price
        if not expiration_data.empty:
            close_price = expiration_data['close'].iloc[0]
            all_pairs.loc[index, 'close_price'] = close_price
        else:
            print(f"No price data found for {ticker} on or near {expiration_date}")

# Check how many price points we found
print(f"Found prices for {all_pairs['close_price'].notna().sum()} out of {len(all_pairs)} ticker/expiration pairs")

# Merge the price data back to the original dataframe
result = pd.merge(df2, all_pairs, on=['act_symbol', 'expiration'], how='left')

# Calculate moneyness and other option profitability metrics
result['moneyness'] = None  # Initialize column

# Apply for calls
call_mask = result['call_put'] == 'Call'
result.loc[call_mask, 'moneyness'] = result.loc[call_mask, 'close_price'] - result.loc[call_mask, 'strike']

# Apply for puts (reversed calculation)
put_mask = result['call_put'] == 'Put'
result.loc[put_mask, 'moneyness'] = result.loc[put_mask, 'strike'] - result.loc[put_mask, 'close_price']

# Classify as ITM/OTM/ATM
result['position'] = 'Unknown'
result.loc[result['moneyness'] > 0, 'position'] = 'ITM'  # In the money
result.loc[result['moneyness'] < 0, 'position'] = 'OTM'  # Out of the money
result.loc[result['moneyness'].abs() < 0.01, 'position'] = 'ATM'  # At the money (within $0.01)

# Print summary
print(result[['act_symbol', 'expiration', 'strike', 'call_put', 'close_price', 'moneyness', 'position']].head(10))

# Save to CSV
result.to_csv('cleaned_optData.csv', index=False)
print(f"Saved data with {result['close_price'].notna().sum()} price points to cleaned_optData_with_prices.csv")