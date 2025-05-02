
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import random

# Load data
df = pd.read_csv('test.csv')
df['call_put'] = df['call_put'].map({'Call': 1, 'Put': 0})  # 1 = Call, 0 = Put
df = df[df['vol'] >0]
df['opt_price'] = df['opt_price'] *100
df['profit'] = df['profit'] * 100


CallModel = joblib.load('callModel.pkl')
PutModel = joblib.load('putModel.pkl')

features = CallModel.feature_names_in_
df = df.dropna(subset=features)
df['date'] = pd.to_datetime(df['date'])
df['expiration'] = pd.to_datetime(df['expiration'])
df = df.sort_values('date')

##Simulation settings
starting_balance = 10000
cashReserveRate = 0.3
n_simulations = 1

finalPVal = 0
total_profit = 0
final_portfolio_values = []

for _ in range(n_simulations):
    portfolio_value = starting_balance
    cash_reserve = portfolio_value * cashReserveRate
    available_investment = portfolio_value - cash_reserve
    open_trades = []
    portfolio_over_time = []

    for date, group in df.groupby('date'):
        ##Closing trades as they expire and justifing account balance. 
        for trade in list(open_trades):
            if trade['expiration'] <= date:
                profit = trade['profit']
                if profit < 0:
                    profit = -trade['cost']  
                trade_total = profit + trade['cost'] 
                available_investment += trade_total
                open_trades.remove(trade)
            

        tied_up = sum(t['cost'] for t in open_trades)
        portfolio_value = available_investment + cash_reserve + tied_up
        cash_reserve = portfolio_value * cashReserveRate
        available_investment = portfolio_value - cash_reserve - tied_up

        group = group[group['expiration'] <= date + pd.Timedelta(days=32)]

        if group.empty:
            continue

        call_group = group[group['call_put'] == 1]
        put_group = group[group['call_put'] == 0]

        if not call_group.empty:
            X_call = call_group[features]
            call_group['predicted_profit'] = CallModel.predict(X_call)
        if not put_group.empty:
            X_put = put_group[features]
            put_group['predicted_profit'] = PutModel.predict(X_put)

        group = pd.concat([call_group, put_group])

        # Select top % trades
        threshold = np.percentile(group['predicted_profit'], 98)
        candidates = group[group['predicted_profit'] >= threshold]

        for _, trade in candidates.iterrows():
            if random.random() > 0.8:
                continue
            cost = trade['opt_price'] 
            profit = trade['profit']
            # print(trade)

            if available_investment >= cost:
                available_investment -= cost
                open_trades.append({
                    'expiration': trade['expiration'],
                    'profit': profit,
                    'cost': cost
                })

        portfolio_over_time.append({'date': date, 'portfolio_value': portfolio_value})

    final_portfolio_values.append(portfolio_value)
    finalPVal += portfolio_value
    total_profit += portfolio_value - starting_balance
    portfolio_df = pd.DataFrame(portfolio_over_time).sort_values('date')

    baseline_trades = group[(group['delta'].abs() > 0.85) & (group['vol'] > group['vol'].median()) ]
    baseline_profit = baseline_trades['profit'].sum()
    # plt.figure(figsize=(12, 6))
    # plt.plot(portfolio_df['date'], portfolio_df['portfolio_value'], label='Portfolio Value', color='green')
    # plt.title('Simulated Portfolio Performance ($10,000 Starting Value, and maintaining a 50% Cash Reserve)')
    # plt.xlabel('Date')
    # plt.ylabel('Portfolio Value')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()



# Final metrics ## 
avg_final_port_value = finalPVal / n_simulations
avg_return = total_profit / n_simulations / starting_balance
print("Average Final Portfolio Value: ", round(avg_final_port_value, 2))
print("Average Return: ", round(avg_return * 100, 2), "%")

#Load SPY data and normalize to starting value
spy_df = pd.read_csv('SPY.csv')
spy_df['date'] = pd.to_datetime(spy_df['date'])
spy_df = spy_df.sort_values('date')

spy_df = spy_df[['date', 'close']]
spy_df = spy_df.set_index('date').reindex(portfolio_df['date']).interpolate(method='time').reset_index()

spy_df['spy_value'] = (spy_df['close'] / spy_df['close'].iloc[0]) * starting_balance

comparison_df = portfolio_df.copy()
comparison_df['spy_value'] = spy_df['spy_value'].values

comparison_df['portfolio_return'] = (comparison_df['portfolio_value'] - starting_balance) / starting_balance * 100
comparison_df['spy_return'] = (comparison_df['spy_value'] - starting_balance) / starting_balance * 1000

plt.figure(figsize=(12, 6))
plt.plot(portfolio_df['date'], portfolio_df['portfolio_value'], label='Portfolio Value', color='green')
# plt.plot(spy_df['date'], spy_df['spy_value'], label='SPY Value', color='blue', linestyle='--')
plt.title('Simulated Portfolio Performance ($10,000 Starting Value, and maintaining a 50% Cash Reserve)')
plt.xlabel('Date')
plt.ylabel('Value ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Percentage return comparison
plt.figure(figsize=(12, 6))
plt.plot(comparison_df['date'], comparison_df['portfolio_return'], label='Portfolio % Return', color='blue')
plt.plot(comparison_df['date'], comparison_df['spy_return'], label='SPY % Return', color='green', linestyle='--')
plt.title('Portfolio vs 10x Weighted SPY - Percentage Return')
plt.xlabel('Date')
plt.ylabel('% Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()









