import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import numpy as np
warnings.filterwarnings("ignore")


data = pd.read_csv("NewestDataset.csv").dropna()

X = data.select_dtypes(include=[np.number]) 
X = X.drop(columns=['profit', 'moneyness', 'stock_price_at_expiration'], errors='ignore')  
features = X.columns.tolist()
import joblib
callModel = joblib.load("callModel.pkl")


importances = callModel.feature_importances_
feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_df = feat_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feat_df)
plt.title("Feature Importances from Random Forest")
plt.show()




data = pd.read_csv("NewestDataset.csv").dropna()
data["profit_cat"] = data["profit"].apply(lambda x: "pos" if x > 0 else "neg")

# interesting scatterplot of profit vs rho
sns.scatterplot(x="profit", y="rho", data=data)
plt.show()
data = data[data["call_put"] == "Call"]

# categorical profit distribution
sns.countplot(x="profit_cat", data=data, palette="Set2")
plt.title("Pre-Model Categorical Profit Distribution")
plt.xlabel("Profit Category")
plt.ylabel("Count")
plt.ticklabel_format(style="plain", axis='y')
plt.show()

most_data = data[(data["profit"] > -100) & (data["profit"] < 100)] # range of -100 to 100 for profit

# profit distribution
profit_col = most_data["profit"]
sns.histplot(profit_col, bins=8, color="red")
plt.ticklabel_format(style=None, axis='y')
plt.title("Pre-Model Profit Distribution")
plt.xlabel("Profit ($)")
plt.show()

# brief information relevant to the visuals
print(f"min profit: {data["profit"].min()}")
print(f"max profit: {data["profit"].max()}")
print(f"mean profit: {data["profit"].mean()}")
print(f"median profit: {data["profit"].median()}")

data_tot = data.shape[0] # complete number of entries after na were dropped
loss_out = data[data["profit"] <= -100]["profit"].count() # getting loss outliers
gain_out = data[data["profit"] >= 100]["profit"].count() # getting gain outliers
most_data_tot = most_data.shape[0]
print(f"total entries: {data_tot}")
print(f"total entries in range -100 to 100: {most_data_tot}")
print(f"total entries out of range -100 to 100: {loss_out + gain_out}")
print(f"percentage of outliers to total: {(loss_out + gain_out) / data_tot * 100:.2f}%")

pos_profit = data[data["profit"] > 0]["profit"]
neg_profit = data[data["profit"] < 0]["profit"]
pos_profit_count = pos_profit.count()
neg_profit_count = neg_profit.count()
profitable = pos_profit_count / data_tot
print(f"percent profitable: {profitable * 100:.2f}%")