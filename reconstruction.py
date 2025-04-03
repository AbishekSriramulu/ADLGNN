import numpy as np
import networkx as nx
import netrd
import matplotlib.pyplot as plt
import itertools as it
import pandas as pd
import logging
import os
from typing import Dict, List, Optional, Union
from abc import ABC, abstractmethod

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseReconstructor(ABC):
    """Base class for network reconstruction methods."""
    
    def __init__(self, data_ratio: float = 0.1, top_k: int = 5):
        """
        Initialize the reconstructor.
        
        Args:
            data_ratio: Ratio of data to use for reconstruction (default: 0.1)
            top_k: Number of top connections to keep (default: 5)
        """
        self.data_ratio = data_ratio
        self.top_k = top_k
        self.recons: Dict[str, netrd.reconstruction.BaseReconstructor] = {}
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and preprocess data from file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Preprocessed DataFrame
        """
        try:
            df = pd.read_csv(file_path, header=None)
            df = df[-int(len(df) * self.data_ratio):]
            
            # Validate data
            if df.empty:
                raise ValueError("Empty DataFrame")
            if df.isnull().any().any():
                logger.warning("Found missing values, filling with 0")
                df = df.fillna(0)
                
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise
            
    def process_adjacency(self, adj: pd.DataFrame) -> pd.DataFrame:
        """
        Process adjacency matrix.
        
        Args:
            adj: Raw adjacency matrix
            
        Returns:
            Processed adjacency matrix
        """
        try:
            # Handle infinities and NaNs
            adj = adj.replace([np.inf, -np.inf, np.nan], 0.0)
            
            # Keep top k connections
            adj.mask(adj.rank(axis=0, method='min', ascending=False) > self.top_k, 0, inplace=True)
            adj.mask(adj.rank(axis=1, method='min', ascending=False) > self.top_k, 0, inplace=True)
            
            return adj
            
        except Exception as e:
            logger.error(f"Error processing adjacency matrix: {str(e)}")
            raise
            
    def save_adjacency(self, adj: pd.DataFrame, base_path: str, method_name: str) -> None:
        """
        Save adjacency matrix to file.
        
        Args:
            adj: Adjacency matrix to save
            base_path: Base path for saving
            method_name: Name of the reconstruction method
        """
        try:
            output_path = f"{base_path.split('.')[0]}_{method_name}.csv"
            adj.to_csv(output_path, index=False)
            logger.info(f"Saved adjacency matrix to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving adjacency matrix: {str(e)}")
            raise
            
    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """Fit the reconstruction method to the data."""
        pass
        
    def reconstruct(self, file_path: str) -> None:
        """
        Reconstruct network from data file.
        
        Args:
            file_path: Path to the data file
        """
        try:
            df = self.load_data(file_path)
            logger.info(f"Processing dataset: {file_path}")
            
            for method_name, reconstructor in self.recons.items():
                logger.info(f"Starting reconstruction with {method_name}")
                try:
                    reconstructor.fit(np.array(df.T))
                    logger.info(f"Fit successful for {method_name}")
                    
                    try:
                        adj = pd.DataFrame(reconstructor.results['thresholded_matrix']).abs()
                    except KeyError:
                        adj = pd.DataFrame(reconstructor.results['distance_matrix']).abs()
                        
                    adj = self.process_adjacency(adj)
                    self.save_adjacency(adj, file_path, method_name)
                    
                except Exception as e:
                    logger.error(f"Error in {method_name}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in reconstruction process: {str(e)}")
            raise

class StandardReconstructor(BaseReconstructor):
    """Standard reconstruction methods implementation."""
    
    def __init__(self, data_ratio: float = 0.1, top_k: int = 5):
        super().__init__(data_ratio, top_k)
        self.recons = {
            'ConvergentCrossMapping': netrd.reconstruction.ConvergentCrossMapping(),
            'NaiveTransferEntropy': netrd.reconstruction.NaiveTransferEntropy(),
            'FreeEnergyMinimization': netrd.reconstruction.FreeEnergyMinimization(),
            'MarchenkoPastur': netrd.reconstruction.MarchenkoPastur(),
            'MaximumLikelihoodEstimation': netrd.reconstruction.MaximumLikelihoodEstimation(),
            'OUInference': netrd.reconstruction.OUInference(),
            'ThoulessAndersonPalmer': netrd.reconstruction.ThoulessAndersonPalmer(),
        }

def main():
    """Main function to run reconstructions."""
    datasets = [
        "traffic.txt.gz",
        "solar_AL.txt.gz",
        "electricity.txt.gz",
        "exchange_rate.txt.gz"
    ]
    
    reconstructor = StandardReconstructor()
    
    for dataset in datasets:
        try:
            reconstructor.reconstruct(f"data/{dataset}")
        except Exception as e:
            logger.error(f"Failed to process {dataset}: {str(e)}")
            continue

if __name__ == "__main__":
    main()