from __future__ import annotations

import unittest
import pandas as pd
import pandas.testing as pd_testing
import matplotlib.pyplot as plt

# ensure that the current directory is in the Python path
import os
import sys
scriptdir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(scriptdir)

# load all the necessary functions for this package
# from data_cleaner import remove_missing, replace_missing_with_value
# from data_cleaner import replace_missing_with_mean, replace_missing_with_median, replace_missing_with_mode
# from data_transformer import z_score_norm, min_max_norm
# from data_transformer import make_named_bins, make_mean_bins, make_median_bins
# from data_transformer import make_min_bins, make_max_bins, merge_uncommon


from data_loader import load_data
import numpy as np

col = {
    "sepal-length": np.float64,
    "sepal-width": np.float64,
    "petal-length": np.float64,
    "petal-width": np.float64,
    "species": str  # Assuming 'species' is categorical
}

df = load_data("C:\\Users\BOSSCJ21\Downloads\DSCapstone\capstone\Pipeline\example_iris.csv",columns=col, missing='drop')
# print(df)

from data_inspector import make_plot, make_density_plot, make_boxplot, make_barplot

make_plot(df, 'sepal-length', 'make_barplot', [], {}).show()
# plt.show()
