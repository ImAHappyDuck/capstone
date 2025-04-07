
# ## This loops through the dataset day by day, and gives the model the rows for that day
# ## The model predicts profit, and the top 5% predicted profit trades are selected and added to a dataframe
# #We then sum and print the predicted profit, and compare to the actual profit for that day
# # We also print the percentage of trades that are profitable for that day
# #The baseline randomly selects trades. If abs(delta) >.8, it has a 10% chance of selecting that trade. 
# ## This dataframe is then written to a csv called selected_trades.csv


import pandas as pd
import numpy as np
df = pd.read_csv('test.csv')
df = df[df['call_put'] == "Call"]
import joblib
model = joblib.load('linear_regression_model.pkl')

features = model.feature_names_in_
df = df.dropna(subset=features)
df = df.sort_values('date')
selected_trades_list = []
daily_profits = []

baseline_daily = []

for date, group in df.groupby('date'):
    ######Comment this out if you want longer term trades
    group = group[pd.to_datetime(group['expiration']) <= pd.to_datetime(date) + pd.Timedelta(days=30)]

    X_day = group[features]
    y_true = group['profit']
    
    yPred = model.predict(X_day)
    group['predictedProfit'] = yPred

    threshold = np.percentile(yPred, 97)
    top_trades = group[group['predictedProfit'] >= 35]
    
    actualProfit = top_trades['profit'].sum()
    predictedProfit = top_trades['predictedProfit'].sum()
    # win_rate = (top_trades['profit'] > 0).count() / len(top_trades)
    
    print(f"Date: {date}")
    print(f"Top predicted trades: {len(top_trades)}")
    print(f"Predicted profit: {predictedProfit:.2f}")
    print(f"Actual profit: {actualProfit:.2f}")
    print(f"cost of contracts: {top_trades['opt_price'].sum():.2f}")
    print(f'actual percent return: {(predictedProfit  / top_trades["opt_price"].sum()) * 100:.2f}%')
    print('\n'*2)
    # print(f"Win rate: {win_rate}\n")
    
    selected_trades_list.append(top_trades)

    baseline_trades = group[group['delta'].abs() > 0.6].sample(frac=0.05, random_state=2)
    baseline_profit = baseline_trades['profit'].sum()
    
    daily_profits.append({
        'date': date,
        'predicted_profit': predictedProfit,
        'actual_profit': actualProfit,
        'baseline_profit': baseline_profit
    })

daily_profits_df = pd.DataFrame(daily_profits)
daily_profits_df.to_csv('daily_profits_summary.csv', index=False)

pd.options.display.float_format = '{:,.2f}'.format

print("=== DAILY PROFIT SUMMARY ===")
print(daily_profits_df.sum(numeric_only=True))

