from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_data(df: pd.DataFrame) -> None:
    """
    Validate input data for potential issues.
    
    Args:
        df: Input DataFrame to validate
        
    Raises:
        ValueError: If data validation fails
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    if df.isnull().any().any():
        logger.warning("DataFrame contains missing values")
        
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")

def preprocess_data(df: pd.DataFrame, scaler: Optional[StandardScaler] = None) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Preprocess the input data by handling missing values and normalizing.
    
    Args:
        df: Input DataFrame
        scaler: Optional pre-fitted scaler
        
    Returns:
        Tuple of (preprocessed DataFrame, fitted scaler)
    """
    # Handle missing values
    if df.isnull().any().any():
        logger.info("Filling missing values with forward fill and backward fill")
        df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Normalize data
    if scaler is None:
        scaler = StandardScaler()
        values = scaler.fit_transform(df.values)
    else:
        values = scaler.transform(df.values)
    
    return pd.DataFrame(values, index=df.index, columns=df.columns), scaler

def generate_graph_seq2seq_io_data(
        df: pd.DataFrame,
        x_offsets: np.ndarray,
        y_offsets: np.ndarray,
        add_time_in_day: bool = True,
        add_day_in_week: bool = False,
        add_day_in_month: bool = False,
        add_week_in_year: bool = False,
        scaler: Optional[StandardScaler] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate samples for graph sequence-to-sequence model.
    
    Args:
        df: Input DataFrame with time series data
        x_offsets: Input time offsets
        y_offsets: Output time offsets
        add_time_in_day: Whether to add time of day features
        add_day_in_week: Whether to add day of week features
        add_day_in_month: Whether to add day of month features
        add_week_in_year: Whether to add week of year features
        scaler: Optional pre-fitted scaler
        
    Returns:
        Tuple of (x, y) arrays for model input/output
    """
    try:
        # Validate input data
        validate_data(df)
        
        # Preprocess data
        df, scaler = preprocess_data(df, scaler)
        
        num_samples, num_nodes = df.shape
        data = np.expand_dims(df.values, axis=-1)
        data_list = [data]
        
        # Add time-based features
        if add_time_in_day:
            time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
            data_list.append(time_in_day)
            
        if add_day_in_week:
            day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
            day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
            data_list.append(day_in_week)
            
        if add_day_in_month:
            day_in_month = np.zeros(shape=(num_samples, num_nodes, 31))
            day_in_month[np.arange(num_samples), :, df.index.day - 1] = 1
            data_list.append(day_in_month)
            
        if add_week_in_year:
            week_in_year = np.zeros(shape=(num_samples, num_nodes, 53))
            week_in_year[np.arange(num_samples), :, df.index.isocalendar().week - 1] = 1
            data_list.append(week_in_year)

        # Concatenate all features
        data = np.concatenate(data_list, axis=-1)
        
        # Generate sequences
        x, y = [], []
        min_t = abs(min(x_offsets))
        max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
        
        for t in range(min_t, max_t):
            x_t = data[t + x_offsets, ...]
            y_t = data[t + y_offsets, ...]
            x.append(x_t)
            y.append(y_t)
            
        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)
        
        return x, y
        
    except Exception as e:
        logger.error(f"Error in generate_graph_seq2seq_io_data: {str(e)}")
        raise

def generate_train_val_test(args: argparse.Namespace) -> None:
    """
    Generate training, validation, and test datasets.
    
    Args:
        args: Command line arguments
    """
    try:
        # Load data
        logger.info(f"Loading data from {args.traffic_df_filename}")
        df = pd.read_hdf(args.traffic_df_filename)
        
        # Define offsets
        x_offsets = np.sort(np.concatenate((np.arange(-11, 1, 1),)))
        y_offsets = np.sort(np.arange(1, 13, 1))
        
        # Generate sequences
        x, y = generate_graph_seq2seq_io_data(
            df,
            x_offsets=x_offsets,
            y_offsets=y_offsets,
            add_time_in_day=True,
            add_day_in_week=True,
            add_day_in_month=True,
            add_week_in_year=True
        )

        logger.info(f"Generated sequences - x shape: {x.shape}, y shape: {y.shape}")
        
        # Split data
        num_samples = x.shape[0]
        num_test = round(num_samples * args.test_ratio)
        num_train = round(num_samples * args.train_ratio)
        num_val = num_samples - num_test - num_train

        # Create splits
        splits = {
            "train": (0, num_train),
            "val": (num_train, num_train + num_val),
            "test": (num_train + num_val, num_samples)
        }
        
        # Save splits
        for split_name, (start, end) in splits.items():
            _x, _y = x[start:end], y[start:end]
            logger.info(f"{split_name} x: {_x.shape}, y: {_y.shape}")
            
            output_path = os.path.join(args.output_dir, f"{split_name}.npz")
            np.savez_compressed(
                output_path,
                x=_x,
                y=_y,
                x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
                y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1])
            )
            logger.info(f"Saved {split_name} data to {output_path}")
            
    except Exception as e:
        logger.error(f"Error in generate_train_val_test: {str(e)}")
        raise

def main(args: argparse.Namespace) -> None:
    """
    Main function to generate training data.
    
    Args:
        args: Command line arguments
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        logger.info("Generating training data")
        generate_train_val_test(args)
        logger.info("Data generation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training data for graph neural network")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/",
        help="Output directory for generated data"
    )
    parser.add_argument(
        "--traffic_df_filename",
        type=str,
        default="data/metr-la.h5",
        help="Raw traffic readings file"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Ratio of training data"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Ratio of test data"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    main(args)
