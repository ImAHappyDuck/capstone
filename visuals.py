import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("NewestDataset.csv").dropna()
data["profit_cat"] = data["profit"].apply(lambda x: "pos" if x > 0 else "neg")

sns.scatterplot(x="profit", y="rho", data=data)
plt.show()
data = data[data["call_put"] == "Call"]