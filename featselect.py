from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
df = pd.read_csv('cleaned_optData_with_prices.csv')
df = df.dropna()
#seperate out calls and puts
df = df[df['call_put'] == "Put"]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('moneyness') 
numeric_cols.remove('profit') 
numeric_cols.remove('stock_price_at_expiration') 


X = df[numeric_cols]
y = df['profit']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

selector = SelectKBest(score_func=f_regression, k=1)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
selected_features = X.columns[selector.get_support()].tolist()

print("\nSelected features:", selected_features)
print("Selected feature scores:")
for feature, score in zip(selected_features, selector.scores_[selector.get_support()]):
    print(f"{feature}: {score:.4f}")

print("\nSelected features shape (train):", X_train_selected.shape)
print("Selected features shape (test):", X_test_selected.shape)