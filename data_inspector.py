from __future__ import annotations

import os
from typing import TypeVar, Iterable, Sequence, Hashable, NamedTuple, Any

import pandas as pd
from statistics import mean, median, mode, stdev

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tempfile
import cv2
from PIL import Image

# declare type variables for use in later functions
K = TypeVar('K', bound=Hashable)

def make_plot(df: pd.DataFrame, col_name: str, action: str, args: list[Any], kwargs: dict[str,Any]) -> Image.Image:
    """Makes a plot of the requested type from the given column.

    Positional Arguments:
    df       - The dataframe containing the variable to be plotted
    col_name - The name of the column whos values will be plotted
    action   - One of the following function names (defined in this file)
                  1. make_density_plot
                  2. make_boxplot
                  3. make_barplot
    args     - A list of the positional arguments required by the action function
    kwargs   - A dictionary of the keyword arguments required by the action function
    """
    # identify the correct plotting function for this action
    if action == 'make_density_plot': plot = make_density_plot
    elif action == 'make_boxplot': plot = make_boxplot
    elif action == 'make_barplot': plot = make_barplot
    else: raise ValueError(f"Unrecognized transformation action: {action}")
    # call that function with the provided arguments
    return plot(df[col_name], *args, **kwargs) # type: ignore

def make_density_plot(data: Sequence[int|float]) -> Image.Image:
    return sns.kdeplot(data).get_figure()






def make_boxplot(data: Sequence[int|float]) -> Image.Image:
    """Create a boxplot to show the distribution of a variable."""
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=data, color="skyblue")
    plt.xlabel("Value")
    plt.title("Boxplot")
    return get_image()

def make_barplot(data: Sequence[str], name:str|None=None, order:list[str]|None=None) -> Image.Image:
    category_counts = count_categories(data)
    if order:
        category_counts = {category: category_counts[category] for category in order if category in category_counts}
    else:
        category_counts = dict(sorted(category_counts.items()))  # Sort alphabetically
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(category_counts.keys()), y=list(category_counts.values()), color='skyblue')
    plt.xlabel(name if name else "Category")
    plt.ylabel("Count")
    plt.title(f"Bar Plot of {name if name else 'Categories'}")
    return get_image()

def count_categories(items: Iterable[K]) -> dict[K, int]:
    """Returns a dictionary mapping each unique item in items to the number of times it appears"""
    counts: dict[K, int] = {}
    for item in items: counts[item] = counts.get(item, 0) + 1
    return counts

# declare statistics type for use in summary statistics
class SummaryStats(NamedTuple):
    mean: float
    median: float
    mode: float
    stdev: float

def get_summary_stats(items: Iterable[int|float]) -> SummaryStats:
    """Computes some basic summary statistics for a single numerical variable"""
    items_tuple: tuple[int|float,...] = tuple(items)
    return SummaryStats(
        mean=mean(items_tuple),
        median=median(items_tuple),
        mode=mode(items_tuple),
        stdev=stdev(items_tuple)
    )

def get_image() -> Image.Image:
    """Converts saves the current matplotlib figure to a PIL Image and clears the current figure"""
    # open a temporary file where the current figure will be saved
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, 'figure.png')
        # save the current figure to the temporary file
        plt.savefig(tmp_path)
        # read the contents of the temporary file as an Image using PIL
        img: Image.Image = Image.fromarray(cv2.imread(tmp_path))
    # clear this plot from matplotlib
    plt.clf()
    # return the PIL Image object (temporary file was automatically deleted)
    return img
