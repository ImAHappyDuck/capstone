
# ## This loops through the dataset day by day, and gives the model the rows for that day
# ## The model predicts profit, and the top 5% predicted profit trades are selected and added to a dataframe
# #We then sum and print the predicted profit, and compare to the actual profit for that day
# # We also print the percentage of trades that are profitable for that day
# #The baseline randomly selects trades. If abs(delta) >.8, it has a 10% chance of selecting that trade. 
# ## This dataframe is then written to a csv called selected_trades.csv

import pandas as pd
import numpy as np
df = pd.read_csv('test.csv') 
# df = df.drop(columns=['Unnamed: 0'])
##Model is trained on call options only
df = df[df['call_put'] == "Call"]
##comment out for big budgets 
# df= df[df['opt_price'] < 30]
import joblib
# model = joblib.load('linear_regression_model.pkl')
model = joblib.load('callModel.pkl')

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

    threshold = np.percentile(yPred, 99.5)  # Top 2.5% of predicted profits
    top_trades = group[group['predictedProfit'] >= threshold] 
    
    actualProfit = top_trades['profit'].sum()
    if not top_trades.empty:
        total_cost = top_trades['opt_price'].sum()
        predictedProfit = top_trades['predictedProfit'].sum()
        actualProfit = top_trades['profit'].sum()
    else:
        total_cost = 0
        predictedProfit = 0
        actualProfit = 0

    # win_rate = (top_trades['profit'] > 0).count() / len(top_trades)
 
    # print(f"Date: {date}")
    # print(f"Top predicted trades: {len(top_trades)}")
    # print(f"Predicted profit: {predictedProfit:.2f}")
    # print(f"Actual profit: {actualProfit:.2f}")
    # print(f"cost of contracts: {top_trades['opt_price'].sum():.2f}")
    # print(f'actual percent return: {(actualProfit  / top_trades["opt_price"].sum()) * 100:.2f}%')
    # print(f'average predicted profit: {top_trades["predictedProfit"].mean():.2f}')
    # print(f'average actual profit: {top_trades["profit"].mean():.2f}')
  
   
    selected_trades_list.append(top_trades)

    baseline_trades = group[(group['delta'].abs() > 0.85) & (group['vol'] > group['vol'].median()) ]
    baseline_profit = baseline_trades['profit'].sum()

    
    daily_profits.append({
        'date': date,
        'predicted_profit': predictedProfit,
        'actual_profit': actualProfit,
        'baseline_profit': baseline_profit,
        'cost':total_cost})

daily_profits_df = pd.DataFrame(daily_profits)
daily_profits_df.to_csv('daily_profits_summary.csv', index=False)

pd.options.display.float_format = '{:,.2f}'.format

print("=== DAILY PROFIT SUMMARY ===")
print(daily_profits_df.sum(numeric_only=True))

total_cost = daily_profits_df['cost'].sum()
total_profit = daily_profits_df['actual_profit'].sum()
average_daily_profit = daily_profits_df['actual_profit'].mean()
annualized_return = total_profit / total_cost

print("\n=== RETURN METRICS ===")
print(f"Total Cost of All Trades: ${total_cost:,.2f}")
print(f"Total Profit: ${total_profit:,.2f}")
print(f"Annualized Return: {annualized_return * 100:.2f}%")
print(f"Average Daily Profit: ${average_daily_profit:,.2f}")


import matplotlib.pyplot as plt

daily_profits_df['date'] = pd.to_datetime(daily_profits_df['date'])
daily_profits_df = daily_profits_df.sort_values('date')

daily_profits_df['cumulative_actual_profit'] = daily_profits_df['actual_profit'].cumsum()
daily_profits_df['cumulative_baseline_profit'] = daily_profits_df['baseline_profit'].cumsum()
daily_profits_df['cost'] = daily_profits_df.get('cost', 0)  # fallback in case cost wasn't added yet
if 'cost' not in daily_profits_df.columns or daily_profits_df['cost'].sum() == 0:
    daily_profits_df['cost'] = [trades['opt_price'].sum() for trades in selected_trades_list]

daily_profits_df['cumulative_cost'] = daily_profits_df['cost'].cumsum()
daily_profits_df['cumulative_return_percent'] = (daily_profits_df['cumulative_actual_profit'] / daily_profits_df['cumulative_cost']) * 100


def plot_cumulative_profits(daily_profits_df):
    daily_profits_df['date'] = pd.to_datetime(daily_profits_df['date'])
    daily_profits_df = daily_profits_df.sort_values('date')
    daily_profits_df['cumulative_actual_profit'] = daily_profits_df['actual_profit'].cumsum()
    daily_profits_df['cumulative_baseline_profit'] = daily_profits_df['baseline_profit'].cumsum()

    plt.figure(figsize=(12, 6))
    plt.plot(daily_profits_df['date'], daily_profits_df['cumulative_actual_profit'], label='Agent Cumulative Profit')
    plt.plot(daily_profits_df['date'], daily_profits_df['cumulative_baseline_profit'], label='Baseline Cumulative Profit', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Profit')
    plt.title('Cumulative Profit Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_cumulative_profits(daily_profits_df)

import seaborn as sns

## make a histogram of actual profit of selected trades
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

daily_profits_df["profit_cat"] = daily_profits_df["actual_profit"].apply(lambda x: "pos" if x > 0 else "neg")


#profit_col = daily_profits_df[(daily_profits_df["actual_profit"] > -100) & (daily_profits_df["actual_profit"] < 100)]["actual_profit"]
profit_col = daily_profits_df["actual_profit"]
sns.histplot(profit_col, color="green", binrange=(-15000,15000), binwidth=3000)
plt.title("Post-Model Profit Distribution")
plt.xlabel("Profit ($)")
plt.ylabel("Count")
plt.show()

sns.countplot(x="profit_cat", data=daily_profits_df, order=['pos','neg'],palette="Set2")
plt.title("Post-Model Categorical Profit Distribution")
plt.xlabel("Profit Category")
plt.ylabel("Count")
plt.show()

# plot_cumulative_profits(daily_profits_df)