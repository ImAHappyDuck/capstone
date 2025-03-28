# import pandas as pd
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# import time

# # Start timing
# start_time = time.time()

# # Load the data
# print("Loading dataset...")
# df = pd.read_csv('cleanedFinNews.csv', dtype=str).fillna('')

# # Create a single analyzer instance to reuse
# analyzer = SentimentIntensityAnalyzer()

# # Define columns to analyze
# columns_to_iterate = [df.columns[2]] + list(df.columns[6:])
# print(f"Analyzing sentiment for columns: {columns_to_iterate}")

# # Define more efficient functions that use the shared analyzer
# def calculate_sentiment_scores(text):
#     return analyzer.polarity_scores(text)

# # Pre-calculate all sentiment scores at once
# print(f"Processing {len(df)} rows...")

# # Initialize result columns
# df['pos_score']= 0.0
# df['neg_score']= 0.0
# df['neu_score']= 0.0

# # Process in batches for better performance feedback
# batch_size = 1000
# total_rows = len(df)

# for start_idx in range(0, total_rows, batch_size):
#     end_idx = min(start_idx + batch_size, total_rows)
#     print(f"Processing rows {start_idx} to {end_idx} of {total_rows}...")
    
#     # Process each row in the current batch
#     for idx in range(start_idx, end_idx):
#         pos_sum = 0
#         neg_sum = 0
#         neu_sum = 0
        
#         # Process each text column for this row
#         for col in columns_to_iterate:
#             # Get sentiment scores once for this text
#             sentiment = calculate_sentiment_scores(df.iloc[idx][col])
#             pos_sum += sentiment['pos']
#             neg_sum += sentiment['neg']
#             neu_sum += sentiment['neu']
        
#         # Store average scores
#         df.loc[idx,'pos_score']= pos_sum / len(columns_to_iterate)
#         df.loc[idx,'neg_score']= neg_sum / len(columns_to_iterate)
#         df.loc[idx,'neu_score']= neu_sum / len(columns_to_iterate)

# # Save results
# print("Saving results...")
# df.to_csv('cleanedFinNews.csv', index=False)

# # Calculate and display execution time
# end_time = time.time()
# execution_time = end_time - start_time
# print(f"Completed in {execution_time:.2f} seconds")

#
import pandas as pd
import zipfile
from datetime import timedelta

def get_stock_price(stock_data, target_date):
    exact_match = stock_data[stock_data['date'] == target_date]
    if not exact_match.empty:
        return exact_match['close'].iloc[0]
    
    for days_back in range(1, 6):
        check_date = target_date - timedelta(days=days_back)
        date_match = stock_data[stock_data['date'] == check_date]
        if not date_match.empty:
            return date_match['close'].iloc[0]
    
    return None
def calculate_stock_delta(stock_data, current_date):
    """Calculate 60-day stock delta, handling weekend/non-trading days."""
    stock_data = stock_data.sort_values('date')
    sixty_days_ago = current_date - timedelta(days=60)
    past_dates = stock_data[stock_data['date'] <= sixty_days_ago]
    if past_dates.empty:
        return None
    

    closest_past_date = past_dates.loc[past_dates['date'].idxmax()]
    past_price = closest_past_date['close']
    current_price = get_stock_price(stock_data, current_date)
    
    if current_price is None or past_price is None:
        return None
    delta = ((current_price - past_price) / past_price) * 100
    return delta

df2 = pd.read_csv('cleaned_optData.csv')
df2['expiration']= pd.to_datetime(df2['expiration'])
zip_file_path = 'full_history.zip'
price_cache = {}
all_pairs = df2[['act_symbol','expiration']].drop_duplicates().reset_index(drop=True)
all_pairs['close_price']= None  # Initialize with None

with zipfile.ZipFile(zip_file_path,'r') as zip_file:
    zip_file_list = zip_file.namelist()
    zip_tickers = [f.split('/')[-1].split('.')[0] for f in zip_file_list if f.startswith('full_history/') and f.endswith('.csv')]
       
    # Process each ticker only once
    for ticker in all_pairs['act_symbol'].unique():
        ticker_file = f"full_history/{ticker}.csv"
        
        # Check if the CSV file exists in the ZIP archive
        if ticker_file in zip_file_list:
            try:
                # Load the CSV file for the ticker directly from the ZIP archive
                with zip_file.open(ticker_file) as file:
                    stock_data = pd.read_csv(file)
                    # Ensure date column is in datetime format
                    stock_data['date']= pd.to_datetime(stock_data['date'])
                    
                # Cache the stock data
                price_cache[ticker]= stock_data
                print(f"Loaded price data for {ticker}: {len(stock_data)} records")
            except Exception as e:
                print(f"Error reading {ticker_file}: {e}")
        else:
            print(f"File for ticker {ticker} not found in the ZIP archive")
    for index, row in all_pairs.iterrows():
        ticker = row['act_symbol']
        expiration_date = row['expiration']
        
        # Skip if we don't have data for ticker
        if ticker not in price_cache:
            continue
            
        stock_data = price_cache[ticker]
        expiration_data = stock_data[stock_data['date']== expiration_date]
        
        # If no exact match, try one day before (some options expire on weekends)
        if expiration_data.empty:
            # Try looking for the trading day before expiration
            for days_back in range(1, 5):  # Check up to 4 days back (handles weekends and holidays)
                check_date = expiration_date - timedelta(days=days_back)
                expiration_data = stock_data[stock_data['date']== check_date]
                if not expiration_data.empty:
                    print(f"Found price for {ticker} using date {check_date} instead of {expiration_date}")
                    break
        if not expiration_data.empty:
            close_price = expiration_data['close'].iloc[0]
            all_pairs.loc[index,'close_price']= close_price
        else:
            print(f"No price data found for {ticker} on or near {expiration_date}")

# After finding prices
print(f"Found prices for {all_pairs['close_price'].notna().sum()} out of {len(all_pairs)} ticker/expiration pairs")

# Check columns in all_pairs before merge
print(f"Columns in all_pairs: {all_pairs.columns.tolist()}")

# Merge once and do it correctly
result = pd.merge(df2, all_pairs, on=['act_symbol','expiration'], how='left')

# Verify the merge worked and check for the close_price column
print(f"Columns in result after merge: {result.columns.tolist()}")
print(f"Number of rows with close_price not null: {result['close_price'].notna().sum()}")

#Initialize columns for synthetic features
result['moneyness']= None
result['current_stock_price'] = None
result['stock_delta_60days'] = None
result['date'] = pd.to_datetime(result['date'])

# Ensure we only calculate for rows that have close_price values
valid_data_mask = result['close_price'].notna()
print(f"Valid rows with close_price data: {valid_data_mask.sum()}")

# Calculate for calls where we have valid close_price
call_mask = (result['call_put']== 'Call') & valid_data_mask
if call_mask.sum() > 0:
    result.loc[call_mask,'moneyness']= result.loc[call_mask,'close_price'] - result.loc[call_mask,'strike']
    print(f"Calculated moneyness for {call_mask.sum()} call options")

# Calculate for puts where we have valid close_price
put_mask = (result['call_put']== 'Put') & valid_data_mask
if put_mask.sum() > 0:
    result.loc[put_mask,'moneyness']= result.loc[put_mask,'strike'] - result.loc[put_mask,'close_price']
    print(f"Calculated moneyness for {put_mask.sum()} put options")

## This adds the current stock price, and delta columns
for index, row in result.iterrows():
    ticker = row['act_symbol']
    current_date = row['date']
    
    if ticker in price_cache:
        stock_data = price_cache[ticker]
        
        result.loc[index, 'current_stock_price'] = get_stock_price(stock_data, current_date)
        result.loc[index, 'stock_delta_60days'] = calculate_stock_delta(stock_data, current_date)

# Classify as ITM/OTM/ATM only for rows with calculated moneyness
result['position']= 'Unknown'
moneyness_mask = result['moneyness'].notna()
result.loc[moneyness_mask & (result['moneyness'] > 0),'position']= 'ITM'
result.loc[moneyness_mask & (result['moneyness'] < 0),'position']= 'OTM'
result.loc[moneyness_mask & (result['moneyness'].fillna(float('nan')).abs() < 0.01),'position']= 'ATM'
result['opt_price']= (result['bid'] + result['ask']) / 2
profit_mask = result['moneyness'].notna() & result['opt_price'].notna()


result.loc[profit_mask,'profit']= result.loc[profit_mask,'moneyness'] - result.loc[profit_mask,'opt_price']
print(result[['act_symbol','expiration','strike','call_put','close_price','moneyness','position','opt_price','profit']].head(10))
result.to_csv('cleaned_optData_with_prices.csv', index=False)
print(f"Saved data with {result['close_price'].notna().sum()} price points to cleaned_optData_with_prices.csv")