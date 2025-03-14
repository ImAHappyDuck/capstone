#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd
from PIL import Image
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import DTypeLike

import json
from typing import Any, NamedTuple
from argparse import Namespace, ArgumentParser

# ensure that the current directory is in the Python path
import os
import sys
scriptdir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(scriptdir)

# load all the necessary functions for this package
from data_loader import load_data, save_data
from data_cleaner import remove_missing, fix_missing
from data_transformer import transform_feature
from data_inspector import make_plot

def filter_csv_by_year(file_path, year_column, years, chunksize=10000):
    filtered_chunks = []
    chunk_count = 0
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        chunk_count += 1
        print(f"Processing chunk {chunk_count}")
        chunk[year_column] = pd.to_datetime(chunk[year_column])
        filtered_chunk = chunk[chunk[year_column].dt.year.isin(years)]
        filtered_chunks.append(filtered_chunk)
    print(f"Finished processing {chunk_count} chunks")
    return pd.concat(filtered_chunks)

def main(args: Namespace):
    print("in Main")
    # load the configuration from the JSON config file
    config: Config = load_config(args.config)
    
    # Filter the CSV file by year before loading it into memory
    years_to_filter = [2020]  # Example years to filter by
    df = filter_csv_by_year(config.raw_dataset_path, 'Date', years_to_filter)
    
    # Define data types and missing values
    dtypes: dict[str, DTypeLike] = {attr_name: get_datatype(attr_config.type) 
                                    for attr_name, attr_config in config.attributes.items()}
    missing: dict[str, set[str]] = {attr_name: set(attr_config.missing_values) 
                                    for attr_name, attr_config in config.attributes.items() 
                                    if attr_config.missing_values is not None}
    
    # Rename any attributes with the rename attribute
    col_renames = {attr_name: attr_config.rename 
                   for attr_name, attr_config in config.attributes.items() 
                   if attr_config.rename is not None}
    df = df.rename(columns=col_renames)
    
    # Fix missing values according to the attribute specifications
    for clean_step in config.clean_steps:
        if clean_step.missing_strategy == 'remove':
            df = remove_missing(df, clean_step.attribute)
        else:
            df = fix_missing(df, clean_step.attribute, clean_step.missing_strategy)
    
    # Apply any transformations in order they are specified
    for ts in config.transform_steps:
        transform_feature(df, ts.attribute, ts.action, ts.args, ts.kwargs)
    
    # Save the data at the determined location
    save_data(df, config.clean_dataset_path)
    
    # Make all requested plots saving them in the plot directory
    for plot_step in config.plotting_steps:
        # Create the requested plot image
        img = make_plot(df, plot_step.attribute, plot_step.action, plot_step.args, plot_step.kwargs)
        # Save this image in the plots directory with the requested file name
        save_plot(img, config.plot_directory_path, plot_step.name)

class Config(NamedTuple):
    raw_dataset_path: str
    clean_dataset_path: str
    plot_directory_path: str
    attributes: dict[str, AttributeConfig]
    clean_steps: list[CleanConfig]
    transform_steps: list[TransformConfig]
    plotting_steps: list[PlotConfig]
    
    @staticmethod
    def parse(d: dict[str, Any]) -> Config:
        return Config(
            str(d['raw_dataset_path']),
            str(d['clean_dataset_path']),
            str(d['plot_directory_path']),
            {k: AttributeConfig.parse(v) for k, v in d['attributes'].items()},
            [CleanConfig.parse(e) for e in d['cleaning']],
            [TransformConfig.parse(e) for e in d['transforming']],
            [PlotConfig.parse(e) for e in d['plotting']]
        )

class AttributeConfig(NamedTuple):
    type: str
    rename: str | None
    missing_values: set[str] | None
    
    @staticmethod
    def parse(d: dict[str, Any]) -> AttributeConfig:
        return AttributeConfig(
            d['type'],
            d.get('rename'),
            d.get('missing_values')
        )

class CleanConfig(NamedTuple):
    attribute: str
    missing_strategy: str
    
    @staticmethod
    def parse(d: dict[str, Any]) -> CleanConfig:
        return CleanConfig(
            d['attribute'],
            d['missing_strategy']
        )

class TransformConfig(NamedTuple):
    action: str
    attribute: str
    args: list[Any]
    kwargs: dict[str, Any]
    
    @staticmethod
    def parse(d: dict[str, Any]) -> TransformConfig:
        return TransformConfig(
            d['action'],
            d['attribute'],
            d.get('args', []),
            d.get('kwargs', {})
        )

class PlotConfig(NamedTuple):
    action: str
    attribute: str
    name: str
    args: list[Any]
    kwargs: dict[str, Any]
    
    @staticmethod
    def parse(d: dict[str, Any]) -> PlotConfig:
        return PlotConfig(
            d['action'],
            d['attribute'],
            d['name'],
            d.get('args', []),
            d.get('kwargs', {})
        )

def load_config(path: str) -> Any:
    with open(path, 'rt', encoding='utf-8') as fin:
        return Config.parse(json.load(fin))
    
def save_plot(img: Any, plot_directory_path: str, plot_name: str):
    """Saves the given plot to the specified directory with the given name"""
    # Ensure the directory exists
    os.makedirs(plot_directory_path, exist_ok=True)
    # Save the plot
    if isinstance(img, Figure):
        img.savefig(os.path.join(plot_directory_path, f"{plot_name}.png"))
    elif isinstance(img, Image.Image):
        img.save(os.path.join(plot_directory_path, f"{plot_name}.png"))
    else:
        raise ValueError("Unsupported image type")

def get_datatype(name: str) -> DTypeLike:
    match name:
        case 'real': return np.float32
        case 'nominal': return np.unicode_
        case _: raise ValueError(f"Unrecognized attribute type {name}")

if __name__ == '__main__':
    parser = ArgumentParser(description=(
        "Run a data cleaning and transformation pipeline on the specified dataset "
        "using the procedures defined in the provided configuration file."
    ))
    parser.add_argument('config', type=str, help='path to JSON config file with procedures')
    args = parser.parse_args()
    main(args)