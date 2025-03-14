import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load data
df = pd.read_csv('cleaned_optData.csv')
df2 = pd.read_csv('cleanedFinNews.csv')

# Filter and clean data
df = df[df['moneyness'].notna() & (df['moneyness'] != 0)]

# Aggregate sentiment scores
sentiment_by_stock = df2.groupby('Stock_symbol').agg({
    'pos_score': 'mean', 'neg_score': 'mean'}).reset_index()

sentiment_by_stock = sentiment_by_stock.rename(columns={
    'pos_score': 'avg_pos_score',
    'neg_score': 'avg_neg_score'})

df = df.merge(
    sentiment_by_stock,
    left_on='act_symbol',
    right_on='Stock_symbol',
    how='left'
)

# Compute average moneyness
averageMoneyness = df['moneyness'].mean()
print(averageMoneyness)

X = df[['delta', 'gamma', 'theta', 'vega', 'rho', 'vol', 'ask', 'bid', 'avg_pos_score', 'avg_neg_score']]
y = df['moneyness']
X = X.fillna(X.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

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

# Print feature weights
feature_names = X.columns  
weights = model.coef_  
print("Feature Weights:")
for feature, weight in zip(feature_names, weights):
    print(f"{feature}: {weight:.4f}")

