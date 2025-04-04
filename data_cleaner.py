import pandas as pd
from typing import Iterable, Any
from statistics import mode

def fix_missing(df: pd.DataFrame, col_name: str, strategy: str) -> pd.DataFrame:
    """Fixes the missing values in a single column using the specified strategy function.
    
    Positional Arguments:
    df       - The dataframe on which missing values are to be fixed (may be modified in place)
    col_name - The name of the column whos missing values will be fixed
    strategy - One of the following function names (defined in this file)
                  1. replace_missing_with_mode
                  2. replace_missing_with_mean
                  3. replace_missing_with_median

    returns a reference to the resulting DataFrame (which may be the same as in the input df)
    """
    # identify the correct function to be run on the specified column
    if strategy == 'replace_missing_with_mode': f = replace_missing_with_mode
    elif strategy == 'replace_missing_with_mean': f = replace_missing_with_mean
    elif strategy == 'replace_missing_with_median': f = replace_missing_with_median
    # elif strategy == 'remove_missing': f = remove_missing
    else: raise ValueError(f"Unrecognized missing replacement strategy: {strategy}")
    # call that function and return the resulting DataFrame
    df[col_name] = f(df, col_name)
    return df

def remove_missing(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Removes all rows from the dataframe where the value in the given column is missing.
    
    Positional Arguments:
    df       - The dataframe on which missing values are to be removed (may be modified in place)
    col_name - The name of the column whos missing values will be removed

    returns a reference to the resulting DataFrame (which may be the same as in the input df)
    """
    return df.dropna(subset=[col_name])
    

def replace_missing_with_value(df: pd.DataFrame, col_name: str, value: Any) -> Iterable[Any]:
    """Returns a modified version of the given column where missing values are filled with the given static value"""
    df[col_name] = df[col_name].fillna(value)
    return df[col_name].values 
def replace_missing_with_mode(df: pd.DataFrame, col_name: str) -> Iterable[Any]:
    """Returns a modified version of the given column where missing values are filled with the most common value"""
    # if there is one mode it will be returned, if there are two the first is returned
    value = df[col_name].mode()[0]
    df[col_name] = df[col_name].fillna(value)
    return df[col_name].values

def replace_missing_with_mean(df: pd.DataFrame, col_name: str) -> Iterable[Any]:
    """Returns a modified version of the given column where missing values are filled with the mean value"""
    value = df[col_name].mean()
    df[col_name] = df[col_name].fillna(value)
    return df[col_name].values

def replace_missing_with_median(df: pd.DataFrame, col_name: str) -> Iterable[Any]:
    """Returns a modified version of the given column where missing values are filled with the mean value"""
    value = df[col_name].median()
    df[col_name] = df[col_name].fillna(value)
    return df[col_name].values
