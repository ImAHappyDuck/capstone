import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import random

df =pd.read_csv('test.csv')
df=df[df['call_put'] == 'Call']
df=df.dropna()
CallModel = joblib.load('callModel.pkl')

features = CallModel.feature_names_in_
df = df.dropna(subset=features)
df['date'] = pd.to_datetime(df['date'])
df['expiration'] = pd.to_datetime(df['expiration'])
df = df.sort_values('date')
##initialize portfolio
starting_balance = 10000
cashReserveRate = 0.5
portfolio_value = starting_balance
cash_reserve = portfolio_value * cashReserveRate
available_investment = portfolio_value -cash_reserve
portfolio_over_time = []
open_trades = []
finalPVal = 0
total_profit = 0
## run 1k simulations of portfolios and averages performance


n = 1000
final_portfolio_values = []
for _ in range(n):
    portfolio_value = starting_balance
    cash_reserve = portfolio_value * cashReserveRate
    available_investment = portfolio_value - cash_reserve
    open_trades = []
    portfolio_over_time = []

    for date, group in df.groupby('date'):
        #reinvest profits from trades that expire today
        for trade in list(open_trades):
            if trade['expiration'] <= date:
                profit = trade['profit']
                if profit <0:
                    profit = -trade['cost']
                trade_total = profit+ trade['cost']
                available_investment += trade_total
                open_trades.remove(trade)

        # Recalculate total portfolio value including tied-up capital
        tied_up = sum(t['cost'] for t in open_trades)
        portfolio_value = available_investment + cash_reserve + tied_up

        # Update dynamic cash reserve and available investment
        cash_reserve = portfolio_value * cashReserveRate
        available_investment = portfolio_value - cash_reserve - tied_up


        group = group[group['expiration'] <= date + pd.Timedelta(days=30)]

        # Predict profits
        X_day = group[features]
        group['predicted_profit'] = CallModel.predict(X_day)



        # Top 1% trades
        threshold = np.percentile(group['predicted_profit'], 99)
        candidates = group[group['predicted_profit'] >= threshold]

        for _, trade in candidates.iterrows():
            if random.random() > 0.5:
                continue
            cost = trade['opt_price']*100
            profit = trade['profit']*100

            if available_investment >= cost:
                available_investment -= cost
                open_trades.append({
                    'expiration': trade['expiration'],
                    'profit': profit,
                    'cost': cost
                })

        # Record portfolio value
        portfolio_over_time.append({'date': date, 'portfolio_value': portfolio_value})
    final_portfolio_values.append(portfolio_value)
    finalPVal += portfolio_value
    total_profit += portfolio_value - starting_balance
avg_final_port_value = finalPVal/n
avg_return = total_profit/n/starting_balance
print("Average Final Portfolio Value: ", avg_final_port_value)
print("Average Return: ", avg_return)
# Plot performance
portfolio_df = pd.DataFrame(portfolio_over_time)
print(portfolio_df.head())
portfolio_df = portfolio_df.sort_values('date')

plt.figure(figsize=(12, 6))
plt.plot(portfolio_df['date'], portfolio_df['portfolio_value'], label='Portfolio Value', color='blue')
plt.title('Simulated Portfolio Performance ($25,000 Starting Value, with 70% Cash Reserve)')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
