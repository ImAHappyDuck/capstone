from datasets import load_dataset
import pandas as pd

# Load the full dataset from Hugging Face
dataset = load_dataset("Zihan1004/FNSPID", split="train")

# Convert to Pandas DataFrame
df = pd.DataFrame(dataset)

# Display DataFrame shape to check the number of rows
print(df.shape)  

# Preview first few rows
print(df.head())
