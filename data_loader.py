import pandas as pd
from numpy.typing import DTypeLike

def load_data(path: str, columns: dict[str,DTypeLike], missing: dict[str,set[str]]) -> pd.DataFrame:
    """Loads the raw dataset from files using parameters from the config file.
    
    Positional Arguments:
    path    - The path to the directory or file where your data is located
    columns - The columns from the dataset you plan to use mapped to their data types
    missing - The columns from the dataset mapped to values that indicate data is missing

    returns a DataFrame loaded from the filepath given with the specified columns and types
    """
    df = pd.read_csv(path,dtype=columns,na_values=missing,usecols=columns.keys())
    return df
    

def save_data(df: pd.DataFrame, path: str) -> None:
    """Saves the transformed dataset into the specified output file"""
    df.to_csv(path,index=False)