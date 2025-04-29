import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
import pandas as pd
from datetime import timedelta
import zipfile
import tqdm
import os


start_time = time.time()


df = pd.read_csv('cleanedFinNews.csv', dtype=str).fillna('')

analyzer = SentimentIntensityAnalyzer()

columns_to_iterate = [df.columns[2]] + list(df.columns[6:])
print(f"Analyzing sentiment for columns: {columns_to_iterate}")

def calculate_sentiment_scores(text):
    return analyzer.polarity_scores(text)

print(f"evaluating {len(df)} rows")
df['pos_score']= 0.0
df['neg_score']= 0.0
df['neu_score']= 0.0




##chunks
batch_size = 1000
total_rows = len(df)



for start_idx in range(0, total_rows, batch_size):
    end_idx = min(start_idx + batch_size, total_rows)
    print(f"Processing rows {start_idx} to {end_idx} of {total_rows}...")
    for idx in range(start_idx, end_idx):
        pos_sum = 0
        neg_sum = 0
        neu_sum = 0
        
        for col in columns_to_iterate:
            # Get sentiment scores only once for this text
            sentiment = calculate_sentiment_scores(df.iloc[idx][col])
            pos_sum += sentiment['pos']
            neg_sum += sentiment['neg']
            neu_sum += sentiment['neu']
        df.loc[idx,'pos_score']= pos_sum / len(columns_to_iterate)
        df.loc[idx,'neg_score']= neg_sum / len(columns_to_iterate)
        df.loc[idx,'neu_score']= neu_sum / len(columns_to_iterate)

# print("Saving results")
df.to_csv('cleanedFinNews.csv', index=False)
print("saved")


##Time for test purposes. ====
end_time = time.time()
execution_time = end_time - start_time
print(f"Completed in {execution_time:.2f} seconds")




def get_stock_price(stock_data_dict, target_date):
    if target_date in stock_data_dict:
        return stock_data_dict[target_date]
    
    for days_back in range(1, 6):
        check_date = target_date - timedelta(days=days_back)
        if check_date in stock_data_dict:
            return stock_data_dict[check_date]
    
    return None

def calculate_stock_delta(stock_data_dict, current_date):
    if not stock_data_dict:
        return None

    # Convert string keys to timestamps for proper comparison
    stock_data_dict = {pd.to_datetime(date): price for date, price in stock_data_dict.items()}
    sixty_days_ago = current_date - timedelta(days=60)
    past_dates = [date for date in stock_data_dict if date <= sixty_days_ago]
    if not past_dates:
        return None

    closest_past_date = max(past_dates)  ##when a date fell on a weekend or market closed day, we find the next available market data. 
    past_price = stock_data_dict[closest_past_date]
    current_price = stock_data_dict.get(current_date, None)

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
                price_cache[ticker] = dict(zip(stock_data['date'], stock_data['close']))
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
        stock_dates = list(stock_data.keys())
        expiration_data = {date: price for date, price in stock_data.items() if pd.to_datetime(date) == expiration_date}        
        # If no exact match, try one day before (some options expire on weekends)
        if not expiration_data:
    # Try looking for the trading day before expiration
             for days_back in range(1, 5):  # Check up to 4 days back (handles weekends and holidays)
                 check_date = expiration_date - timedelta(days=days_back)
                 expiration_data = {date: price for date, price in stock_data.items() if pd.to_datetime(date) == check_date}
                 if expiration_data:
                     print(f"Found price for {ticker} using date {check_date} instead of {expiration_date}")
                     break
        if expiration_data:
         close_price = list(expiration_data.values())[0]
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
from tqdm import tqdm

tqdm.pandas()  # Enables progress bar for apply functions

# Convert price_cache into DataFrame for vectorized operations
stock_prices_df = pd.DataFrame([
    {'act_symbol': ticker, 'date': date, 'close_price': price}
    for ticker, stock_data in price_cache.items()
    for date, price in stock_data.items()
])
stock_prices_df['date'] = pd.to_datetime(stock_prices_df['date'])

# Merge current stock price
result = result.merge(stock_prices_df, on=['act_symbol', 'date'], how='left', suffixes=('', '_current'))
result.rename(columns={'close_price_current': 'current_stock_price'}, inplace=True)
def fast_stock_delta(row):
    stock_data = price_cache.get(row['act_symbol'], {})
    return calculate_stock_delta(stock_data, row['date'])
 
# Convert price_cache into a DataFrame to eliminate slow row-by-row lookups
stock_delta_list = []
for ticker, stock_data in price_cache.items():
    for date in stock_data.keys():
        stock_delta_list.append({
            'act_symbol': ticker,
            'date': date,
            'stock_delta_60days': calculate_stock_delta(stock_data, date)
        })

# Create a new DataFrame with precomputed deltas
stock_delta_df = pd.DataFrame(stock_delta_list)

# Merge precomputed deltas into `result` instead of applying row-by-row
result = result.merge(stock_delta_df, on=['act_symbol','date'],how='left')



result['position']= 'Unknown'
moneyness_mask = result['moneyness'].notna()
result.loc[moneyness_mask & (result['moneyness'] > 0),'position']= 'ITM'
result.loc[moneyness_mask & (result['moneyness'] < 0),'position']= 'OTM'
result.loc[moneyness_mask & (result['moneyness'].fillna(float('nan')).abs() < 0.01),'position']= 'ATM'
result['opt_price']= (result['bid'] + result['ask']) / 2
profit_mask = result['moneyness'].notna() & result['opt_price'].notna()


result.loc[profit_mask,'profit']= result.loc[profit_mask,'moneyness'] - result.loc[profit_mask,'opt_price']
print(result[['act_symbol','expiration','strike','call_put','close_price','moneyness','position','opt_price','profit']].head(10))
result.to_csv('cleaned_optData_with_prices.csv',index=False)
print(f"Saved data with {result['close_price'].notna().sum()} price points to cleaned_optData_with_prices.csv")



# Load the options data
df2 = pd.read_csv('cleaned_optData.csv')
df2['expiration'] = pd.to_datetime(df2['expiration'])
df2['date'] = pd.to_datetime(df2['date'])

# Initialize price cacheaqaaqaa
zip_file_path = 'full_history.zip'
price_cache = {}

# Extract unique tickers and expiration dates
all_pairs = df2[['act_symbol','expiration']].drop_duplicates().reset_index(drop=True)
all_pairs['close_price'] = None

with zipfile.ZipFile(zip_file_path,'r') as zip_file:
    zip_file_list = zip_file.namelist()
    tickers = {f.split('/')[-1].split('.')[0] for f in zip_file_list if f.startswith('full_history/') and f.endswith('.csv')}
    
    for ticker in tqdm.tqdm(df2['act_symbol'].unique(),desc='Loading Stock Data'):
        ticker_file = f"full_history/{ticker}.csv"
        if ticker_file in zip_file_list:
            try:
                with zip_file.open(ticker_file) as file:
                    stock_data = pd.read_csv(file,parse_dates=['date'])
                    price_cache[ticker] = dict(zip(stock_data['date'],stock_data['close']))
            except Exception as e:
                print(f"Error reading {ticker_file}: {e}")
        else:
            print(f"File for ticker {ticker} not found in ZIP archive")

df2['current_stock_price'] = None
df2['stock_price_at_expiration'] =None

for index,row in tqdm.tqdm(df2.iterrows(),total=len(df2),desc='Assigning Stock Prices'):
    ticker,option_date,exp_date = row['act_symbol'],row['date'],row['expiration']
    
    if ticker in price_cache:
        stock_data = price_cache[ticker]

        # Get stock price on the option's date
        current_price = stock_data.get(option_date,None)
        if current_price is None:
            for days_back in range(1,5):
                current_price = stock_data.get(option_date - timedelta(days=days_back), None)
                if current_price:
                    break
        df2.at[index, 'current_stock_price'] = current_price

    ## print(ticker,option_date,exp_date)



        expiration_price = stock_data.get(exp_date, None)
        if expiration_price is None:
            for days_back in range(1, 5):
                expiration_price = stock_data.get(exp_date - timedelta(days=days_back), None)
                if expiration_price:
                    break
        df2.at[index, 'stock_price_at_expiration'] = expiration_price

df2['moneyness'] = None
call_mask = (df2['call_put'] == 'Call') & df2['stock_price_at_expiration'].notna()
put_mask = (df2['call_put'] == 'Put') & df2['stock_price_at_expiration'].notna()
df2.loc[call_mask, 'moneyness'] = df2.loc[call_mask, 'stock_price_at_expiration'] - df2.loc[call_mask, 'strike']
df2.loc[put_mask, 'moneyness'] = df2.loc[put_mask, 'strike'] - df2.loc[put_mask, 'stock_price_at_expiration']

##Possibly useful, but never used. 
# df2['position'] = 'Unknown'
# moneyness_mask = df2['moneyness'].notna()
# df2.loc[moneyness_mask & (df2['moneyness'] > 0), 'position'] = 'ITM'
# df2.loc[moneyness_mask & (df2['moneyness'] < 0), 'position'] = 'OTM'
# df2.loc[moneyness_mask & (df2['moneyness'].fillna(0).abs() < 0.01), 'position'] = 'ATM'

# Compute option price
df2['opt_price'] = (df2['bid'] + df2['ask']) / 2

# Compute profit
profit_mask = df2['moneyness'].notna() & df2['opt_price'].notna()
df2.loc[profit_mask, 'profit'] = df2.loc[profit_mask, 'moneyness'] - df2.loc[profit_mask, 'opt_price']

import os

# Load existing delta progress if available
delta_progress_file = 'stock_deltas_progress.csv'
if os.path.exists(delta_progress_file):
    existing_stock_deltas = pd.read_csv(delta_progress_file)
    existing_stock_deltas['date'] = pd.to_datetime(existing_stock_deltas['date'])
else:
    existing_stock_deltas = pd.DataFrame(columns=['act_symbol', 'date', 'stock_delta_60days'])


##speed optimization. 
existing_pairs = set(zip(existing_stock_deltas['act_symbol'], existing_stock_deltas['date']))

print("Calculating stock delts")
stock_delta_list = []

for ticker, stock_data in tqdm.tqdm(price_cache.items(), desc='Processing Stock Deltas'):
    dates = sorted(stock_data.keys())


    for i, date in enumerate(dates):


        if (ticker, date) in existing_pairs:  
            continue

        past_date = date - timedelta(days=60)
        past_prices = [stock_data[d] for d in dates[:i] if d <= past_date]

        if past_prices and past_prices[-1] != 0: 
            delta = ((stock_data[date] - past_prices[-1]) / past_prices[-1]) * 100
        else:
            delta = None  

        stock_delta_list.append({'act_symbol': ticker, 'date': date, 'stock_delta_60days': delta})

        
        if len(stock_delta_list) >= 500:
            temp_df = pd.DataFrame(stock_delta_list)
            temp_df.to_csv(delta_progress_file, mode='a', header=not os.path.exists(delta_progress_file), index=False)
            stock_delta_list.clear()  





if stock_delta_list:
    temp_df = pd.DataFrame(stock_delta_list)
    temp_df.to_csv(delta_progress_file, mode='a', header=not os.path.exists(delta_progress_file), index=False)





# Load final delta results and merge with df2
stock_delta_df = pd.read_csv(delta_progress_file)
stock_delta_df['date'] = pd.to_datetime(stock_delta_df['date'])
df2 = df2.merge(stock_delta_df, on=['act_symbol', 'date'], how='left')
df2['opt_price'] = (df2['bid'] + df2['ask'])/2

df2.to_csv('cleaned_optData_with_prices.csv', index=False)
print(f"Saved {df2.shape[0]} rows to cleaned_optData_with_prices.csv")
print("now adding sentiment score to the cleaned_optData_with_prices.csv")

