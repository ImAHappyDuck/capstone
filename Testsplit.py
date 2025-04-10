## Reads in NewestDataset.csv, creates a new csv with a random 
# sample of 25% called test.csv, and writes the rest to train.csv
import pandas as pd
df = pd.read_csv('NewestDataset.csv')
# df['opt_price'] = (df['bid'] + df['ask']) / 2

# test_df =df.sample(frac=0.25, random_state=22)  
# train_df = df.drop(test_df.index)
# test_df.to_csv('test.csv', index=False)
# train_df.to_csv('train.csv', index=False)
# print("train and test files created")


## a different version that splits based on date. Train is months 1-9, test is 10-12
# df = pd.read_csv('NewestDataset.csv')
test = df[df['date_month'] > 6]
train = df[df['date_month']<= 6]

test.to_csv('test.csv', index=False)
train.to_csv('train.csv', index=False)
print("train and test files created")
