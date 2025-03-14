import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report, mean_absolute_error
import joblib


#Model 1, Logistic regression predicting ITM or OTM
df = pd.read_csv('cleaned_optData.csv')
data = df[df['moneyness'].isna() == False]
data['num_position'] = data['position'].apply(lambda x: 1 if x == 'ITM' else 0)

X = data[['bid', 'delta', 'gamma', 'theta', 'vega', 'rho']]
y = data['num_position']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
joblib.dump(model, 'logistic_regression_model.pkl')

#Model 2, linear regression predicting moneyness, with options data and sentiment data. 

df = pd.read_csv('cleaned_optData.csv')

df2 = pd.read_csv('cleanedFinNews.csv')
df = df[df['moneyness'].notna() & (df['moneyness'] != 0)]

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

# average moneyness
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

# Feature weights
feature_names = X.columns  
weights = model.coef_  
print("Feature Weights:")
for feature, weight in zip(feature_names, weights):
    print(f"{feature}: {weight:.4f}")

print("Comparing to Baseline:")
average = df['moneyness'].mean()
print(f"Random Baseline Moneyness: {average}")
y_pred = model.predict(X_test)
average_projected_moneyness = np.mean(y_pred)
print(f"Projected Moneyness: {average_projected_moneyness:.4f}")
joblib.dump(model, 'linear_regression_model.pkl')

