## scatter plot of delta vs stock delta 60days for profitable trades 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df = pd.read_csv('cleaned_optData_with_prices.csv')
dfp= df[df['profit'].notna() & (df['profit'] > 0)]



# dfp.groupby('date')['profit'].mean().plot(kind='line', figsize=(10,5), title="Profit Trend Over Time (Of Profitable Trades)")
# plt.ylabel("Average Profit")
# plt.show()

dfp.groupby('date')['priceDelta'].mean().plot(kind='line', figsize=(10,5), title="Profit Delta Trend Over Time")
plt.ylabel("Average Profit Delta")
plt.show()




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

