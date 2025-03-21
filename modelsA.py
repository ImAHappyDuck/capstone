import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report, mean_absolute_error
import joblib
import argparse
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["a", "b", "c","d"], required=True, help="models are a, b, c, or d")
args = parser.parse_args()

if args.model == "a":
    #Model a, Logistic regression predicting ITM or OTM using old columns
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

#Model b, logistic regression using the new columns
elif args.model == "b":

    df = pd.read_csv('cleaned_optData.csv')
    data = df[df['moneyness'].isna() == False]
    data['num_position'] = data['position'].apply(lambda x: 1 if x == 'ITM' else 0)

    X = data[['bid', 'ask','rho']]
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

# model c, random forest classifier using the columns found in feature selection
elif args.model == "c":
    df = pd.read_csv('cleaned_optData.csv')
    data = df[df['moneyness'].isna() == False]
    data['num_position'] = data['position'].apply(lambda x: 1 if x == 'ITM' else 0)

    X = data[['bid', 'ask','rho']]
    y = data['num_position']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
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
    
# model d, random forest classifier using the columns found in feature selection using higher depth
elif args.model == "d":
    df = pd.read_csv('cleaned_optData.csv')
    data = df[df['moneyness'].isna() == False]
    data['num_position'] = data['position'].apply(lambda x: 1 if x == 'ITM' else 0)

    X = data[['bid', 'ask','rho']]
    y = data['num_position']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)
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
    
 

    
