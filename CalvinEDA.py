import requests
import pandas as pd

# API URL
url = "https://datasets-server.huggingface.co/first-rows?dataset=Zihan1004%2FFNSPID&config=default&split=train"

# Fetch data
response = requests.get(url)
data = response.json()

# Extract column names and rows
columns = data["features"]
rows = [row["row"] for row in data["rows"]]

# Create DataFrame
df = pd.DataFrame(rows, columns=[col["name"] for col in columns])

# Display DataFrame
# print(df.head())
# print(df[pd.notna(df['Lexrank_summary'])]['Lexrank_summary'])
print(df[['Date','Article_title']])
