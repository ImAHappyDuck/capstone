import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression

parser = argparse.ArgumentParser()
parser.add_argument("--featSelect", choices=["old", "new", "new2"], required=True, help="models are old, new, or new2")
args = parser.parse_args()

if args.featSelect == "old":
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

elif args.featSelect == "new":
    # Load dataset
    data = pd.read_csv('cleaned_optData_with_prices.csv')
    data = data.dropna()
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('profit')
    # I'm removing this because I think profit is based off of it
    numeric_cols.remove('moneyness')
    X = data[numeric_cols]
    y = data.profit

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
elif args.featSelect == "new2":
    # Load dataset
    data = pd.read_csv('cleaned_optData_with_prices_new.csv')
    data = data.dropna()
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('moneyness')
    # I'm removing this because I think profit is based off of it
    numeric_cols.remove('profit')
    X = data[numeric_cols]
    y = data.profit

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