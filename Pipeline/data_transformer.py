import pandas as pd

from statistics import mean, median, stdev
from typing import Sequence, Callable, Any

import os
import sys
import numpy as np
scriptdir = os.path.abspath(os.path.dirname(__file__))
if scriptdir not in sys.path: sys.path.append(scriptdir)
from data_inspector import count_categories

def transform_feature(df: pd.DataFrame, col_name: str, action: str, args: list[Any], kwargs: dict[str,Any]):
    """Transforms a single column of the dataframe using the specified modification

    Positional Arguments:
    df       - The dataframe on which an attribute will be transformed (may be modified in place)
    col_name - The name of the column whos values will be changed
    action   - One of the following function names (defined in this file)
                  1. z_score_norm
                  2. min_max_norm
                  3. make_named_bins
                  4. make_mean_bins
                  5. make_median_bins
                  6. make_min_bins
                  7. make_max_bins
    args     - A list of the positional arguments required by the action function
    kwargs   - A dictionary of the keyword arguments required by the action function
    """
    # identify the correct function to call given the specified action
    if action == 'z_score_norm': func = z_score_norm
    elif action == 'min_max_norm': func = min_max_norm
    elif action == 'merge_uncommon': func = merge_uncommon
    elif action == 'make_named_bins': func = make_named_bins
    elif action == 'make_mean_bins': func = make_mean_bins
    elif action == 'make_median_bins': func = make_median_bins
    elif action == 'make_min_bins': func = make_min_bins
    elif action == 'make_max_bins': func = make_max_bins
    else: raise ValueError(f"Unrecognized transformation action: {action}")
    # apply this function to the specified column
    df[col_name] = func(df[col_name], *args, **kwargs) # type: ignore

def z_score_norm(items: Sequence[int|float]) -> Sequence[float]:
    """Translates all values into standard deviations above and below the mean"""
    z_scores = []
    mean_value = mean(items)
    stdev_value = stdev(items)
    for item in items:
        z_scores.append((item - mean_value)/stdev_value)
    return z_scores

def min_max_norm(items: Sequence[int|float]) -> Sequence[float]:
    """Scales all items into the range [0, 1]"""
    minV =min(items)
    maxV= max(items)
    range_value = maxV - minV
    if range_value == 0:  
        return [0.0] * len(items)
    
    return [(item - minV) / range_value for item in items]


def merge_uncommon(items: Sequence[str], default: str = 'OTHER',
                   max_categories: int|None = None, 
                   min_count: int|None = None, 
                   min_pct: float|None = None) -> Sequence[str]:
    """Merges infrequent categorical labels into a single miscellaneous category
    
    Positional Arguments:
    items   - A sequence if categorical labels to be transformed
    default - The default value with which to replace uncommon labels

    Keyword Arguments:
    max_categories - The maximum number of distinct labels to be kept (keep most common)
    min_count      - The minimum number of examples a label must have to be kept
    min_pct        - The minimum percentage of the dataset a label must represent to be kept

    returns a transformed version of items where uncommon labels are replaced with the default value
    """
    args = sum(arg is not None for arg in [max_categories, min_count, min_pct])
    if args != 1:
        raise ValueError("Exactly one category type (max_categories, min_count, min_pct) must be specified")

    category_counts = count_categories(items)
    total_items = len(items)
    keep = set()

    if max_categories is not None:
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        keep = {cat for cat, _ in sorted_categories[:max_categories]}
        # print(f"Keep Categories (max_categories): {keep}")
    elif min_count is not None:
        keep = {cat for cat, count in category_counts.items() if count >= min_count}
        # print(f"Keep Categories (min_count): {keep}")
    elif min_pct is not None:
        min_required = total_items * min_pct
        keep = {cat for cat, count in category_counts.items() if count >= min_required}
        # print(f"Keep Categories (min_pct): {keep}")
    result = [item if item in keep else default for item in items]
    # print(f"Result: {result}")
    return result

def _find_bins(items: Sequence[int|float], cut: str, bin_count: int) -> Sequence[int]:
    """Bins the items and returns a sequence of bin numbers in the range [0,bin_count)"""
    # identify the bin cutoffs based on strategy
    if cut == 'width':
        boundaries = _get_equal_width_cuts(items, bin_count)
    elif cut == 'freq':
        boundaries = _get_equal_frequency_cuts(items, bin_count)
    else:
        raise ValueError(f"Unrecognized bin cut strategy: {cut}")
    # determine the bin of each item using those cutoffs and return the list of bins
    return [_find_bin(item, boundaries) for item in items]

def make_named_bins(items: Sequence[int|float], cut: str, names: Sequence[str]):
    """Bins items using the specified strategy and represents each with one of the given names"""
    # get bin_count and check if too few
    bin_count = len(names)
    if bin_count == 0:
        raise ValueError("Too few bins.")
    # use find bins to get the bin numbers
    bin_nums = _find_bins(items, cut, bin_count)
    # return the respective names for each bin number
    return [names[bin_num] for bin_num in bin_nums]

def make_mean_bins(items: Sequence[int|float], cut: str, bin_count: int) -> Sequence[int|float]:
    """Bins items using the specified cut strategy and represents each bin with its mean"""
    # get bin numbers for each item
    bin_nums = _find_bins(items, cut, bin_count)
    # create bins and add the respective items to each bin
    bins = [[] for _ in range(bin_count)]
    
    for i, bin_num in enumerate(bin_nums):
        bins[bin_num].append(items[i])
    # get the means of the items in each bin
    bin_means = [mean(bin_items) for bin_items in bins]
    # return the mean of the respective bin for each item
    mean_bins = [bin_means[bin_num] for bin_num in bin_nums]
    return mean_bins







def make_median_bins(items: Sequence[int|float], cut: str, bin_count: int) -> Sequence[int|float]:
    """Bins items using the specified cut strategy and represents each bin with its median"""
    bin_nums = _find_bins(items, cut, bin_count)
    
    # Create and populate bins
    bins = [[] for _ in range(bin_count)]
    for i, bin_num in enumerate(bin_nums):
        bins[bin_num].append(items[i])
    
    # Calculate median for each bin
    bin_medians = []
    for bin_items in bins:
        if bin_items:
            sorted_items = sorted(bin_items)
            n = len(sorted_items)
            if n % 2 == 0:
                median = (sorted_items[n//2-1] + sorted_items[n//2]) / 2
            else:
                median = sorted_items[n//2]
            bin_medians.append(median)
        else:
            bin_medians.append(0.0)
    
    # Return each original item replaced with its bin's median
    return [bin_medians[bin_num] for bin_num in bin_nums]







def make_min_bins(items: Sequence[int|float], cut: str, bin_count: int) -> Sequence[int|float]:
    """Bins items using the specified cut strategy and represents each bin with its minimum value"""
    bin_nums = _find_bins(items, cut, bin_count)
    
    # Create and populate bins
    bins = [[] for _ in range(bin_count)]
    for i, bin_num in enumerate(bin_nums):
        bins[bin_num].append(items[i])
    
    # Calculate minimum for each bin
    bin_mins = [min(bin_items) if bin_items else float('inf') for bin_items in bins]
    
    # Return each original item replaced with its bin's minimum
    return [bin_mins[bin_num] for bin_num in bin_nums]


def make_max_bins(items: Sequence[int|float], cut: str, bin_count: int) -> Sequence[int|float]:
    """Bins items using the specified cut strategy and represents each bin with its maximum value"""
    bin_nums = _find_bins(items, cut, bin_count)
    
    # Create and populate bins
    bins = [[] for _ in range(bin_count)]
    for i, bin_num in enumerate(bin_nums):
        bins[bin_num].append(items[i])
    
    # Calculate maximum for each bin
    bin_maxs = [max(bin_items) if bin_items else float('-inf') for bin_items in bins]
    
    # Return each original item replaced with its bin's maximum
    return [bin_maxs[bin_num] for bin_num in bin_nums]


   
def _find_bin(item: int|float, boundaries: list[tuple[float,float]]) -> int:
    """Assigns a given item to one of the bins defined by the given boundaries bin_min <= x < bin_max"""
    # check edge cases outside the range of the bins
    if item < boundaries[0][0]: return 0
    if item >= boundaries[-1][-1]: return len(boundaries)-1
    # otherwise find the correct bin
    for bin_num,(bin_min,bin_max) in enumerate(boundaries):
        if bin_min <= item and item < bin_max:
            return bin_num
    # this point should never be reached so raise an exception
    raise ValueError(f"Unable to place {item} in any of the bins")

def _get_equal_width_cuts(items: Sequence[int|float], bin_count: int) -> list[tuple[float,float]]:
    """Returns a list of the lower and upper cutoffs for each of the equal width bins"""
    # find the minimum and maximum values in items
    low: float = min(items)
    high: float = max(items)
    # define the bin width as 1/bin_count of the difference between the min and max values
    width: float = (high - low) / bin_count
    # compute the bin boundaries using this width
    boundaries: list[tuple[float,float]] = []
    for bin_num in range(bin_count):
        # identify the boundaries for this bin and add them to the list
        bin_min = low + bin_num * width
        bin_max = low + (bin_num+1) * width
        boundaries.append((bin_min, bin_max))
    return boundaries

def _get_equal_frequency_cuts(items: Sequence[int|float], bin_count: int) -> list[tuple[float,float]]:
    """Returns a list of the lower and upper cutoffs for each of the equal frequency bins"""
    # get a sorted list of the items to help identify where cuts should be made
    sorted_items: list[int|float] = list(sorted(items))
    # use a cursor to track the index of the last cut made
    last_cut: int = 0
    # use a variables to track how many more bins and items are left
    bins_remaining: int = bin_count
    items_remaining: int = len(sorted_items)
    # create a variable to hold the identified boundaries
    boundaries: list[tuple[float,float]] = []
    # loop to find more cuts until finished
    while bins_remaining > 0:
        # determine how many items should be in this next bin
        items_in_bin: int = min(items_remaining, int(round(items_remaining/bins_remaining)))
        # determine the index where the next cut should be made to include that many items
        next_cut = last_cut + items_in_bin
        # get the values at the relevant indices to make bin cuts
        bin_min = sorted_items[max(0,last_cut)]
        bin_max = sorted_items[min(next_cut, len(sorted_items)-1)]
        # add these values to the boundaries to be returned
        boundaries.append((bin_min, bin_max))
        # decrement bins and items remaining
        bins_remaining -= 1
        items_remaining -= items_in_bin
        # mark this cut as the last cut for the next iteration
        last_cut = next_cut
    return boundaries
