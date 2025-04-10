import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report, mean_absolute_error
import joblib
import argparse


df = pd.read_csv('train.csv')

df = df.dropna()
df = df[df['call_put'] == 'Put']
X = df.select_dtypes(include=[np.number]) 
X = X.drop(columns=['profit', 'moneyness', 'stock_price_at_expiration'], errors='ignore')  
y = df['profit']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# model = LinearRegression()
# model.fit(X_train, y_train)
model = RandomForestRegressor(n_estimators=25, max_depth=10, random_state=27)
model.fit(X_train, y_train)
model.feature_names_in_ = X.columns


y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('\nModel Performance Metrics:')
print('-' * 25)
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'R² Score: {r2:.4f}')

# # Feature weights
# feature_names = X.columns  
# weights = model.coef_  
# print("Feature Weights:")
# for feature, weight in zip(feature_names, weights):
#     print(f"{feature}: {weight:.4f}")

print("Comparing to Baseline:")
average = df['profit'].mean()
print(f"Random Baseline Profit: {average}")
y_pred = model.predict(X_test)
average_projected_profit = np.mean(y_pred)
print(f"Projected Profit: {average_projected_profit:.4f}")
joblib.dump(model, 'putModel.pkl')


