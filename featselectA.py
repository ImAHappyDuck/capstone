import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression

# Load dataset
data = pd.read_csv('cleaned_optData.csv')
data = data.dropna()
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('moneyness') 
X = data[numeric_cols]
y = data.moneyness

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


# selecting best 3 features with f_regression
select_k_best = SelectKBest(score_func=f_regression, k=3)
X_train_k_best = select_k_best.fit_transform(X_train, y_train)

# selecting the 3 best features to keep and gets the scores for the features
selected_features = X_train.columns[select_k_best.get_support()]
feature_scores = select_k_best.scores_[select_k_best.get_support()]

# print the selected features and then their scores
print("Selected features:", X_train.columns[select_k_best.get_support()])
for feature, score in zip(selected_features, feature_scores):
    print(f"Feature: {feature}, Score: {score}")
