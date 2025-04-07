## scatter plot of delta vs stock delta 60days for profitable trades 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df = pd.read_csv('cleaned_optData_with_prices.csv')
## Randomly sample 50000 rows for faster plotting
df = df.sample(n=50000, random_state=23) 
dfp= df[df['profit'].notna() & (df['profit'] > 0)]



# dfp.groupby('date')['profit'].mean().plot(kind='line', figsize=(10,5), title="Profit Trend Over Time (Of Profitable Trades)")
# plt.ylabel("Average Profit")
# plt.show()

# dfp.groupby('date')['priceDelta'].mean().plot(kind='line', figsize=(10,5), title="Profit Delta Trend Over Time")
# plt.ylabel("Average Profit Delta")
# plt.show()

##plot profit vs rho 
df.plot.scatter(x='rho', y='profit', alpha=0.5)
plt.title('Scatter Plot of Rho vs Profit')
plt.xlabel('Rho')
plt.ylabel('Profit')
plt.show()

## plot of pos_sentiment vs profit
# df.plot.scatter(x='pos_sentiment', y='profit', alpha=0.5)
# plt.xlabel('Positive Sentiment Score')
# plt.ylabel('Profit')
# plt.title('Scatter Plot of Positive Sentiment vs Profit')
# plt.show()

## calculate what percentage of trades are profitable
total_trades = len(df)
profitable_trades = len(dfp)
profit_percentage = (profitable_trades / total_trades) * 100
print(f"Percentage of profitable trades: {profit_percentage:.2f}%")

##calculate what percentage of trades with negative rho are profitable
negative_rho_trades = df[df['rho'] < 0]
profitable_negative_rho_trades = negative_rho_trades[negative_rho_trades['profit'] > 0]
negative_rho_total = len(negative_rho_trades)
print("percentage of negative rho trades that are profitable:")
if negative_rho_total > 0:
    negative_rho_profitable_percentage = (len(profitable_negative_rho_trades) / negative_rho_total) * 100
    print(f"{negative_rho_profitable_percentage:.2f}%")


## Plot of profit vs delta for calls 
# df1 = df[df['call_put'] == "Call"]
# df1= df1[df1['profit'].notna() & (df1['profit'] > 0)]
# plt.scatter(y = df1['delta'], x = df1['profit'], alpha=0.5)
# plt.title('Scatter Plot of Delta vs profit for Profitable Calls')
# plt.xlabel('Stock Delta 60 Days')
# plt.ylabel('Delta')
# plt.show()

# # df1 = df[df['call_put'] == "Put"]
# df1= df1[df1['profit'].notna() & (df1['profit'] > 0)]
# plt.scatter(y = df1['delta'], x = df1['profit'], alpha=0.5)
# plt.title('Scatter Plot of Delta vs profit for Profitable Puts')
# plt.xlabel('Stock Delta 60 Days')
# plt.ylabel('Delta')
# plt.show()

