import numpy as np
import networkx as nx
import netrd
import matplotlib.pyplot as plt
import itertools as it
import pandas as pd
import asyncio
from reconstruction import BaseReconstructor, logger

class OptimalCausationEntropyReconstructor(BaseReconstructor):
    """Optimal causation entropy-based network reconstruction methods."""
    
    def __init__(self, data_ratio: float = 0.3, top_k: int = 5):  # Note: Using 0.3 as per original file
        super().__init__(data_ratio, top_k)
        self.recons = {
            'OptimalCausationEntropy': netrd.reconstruction.OptimalCausationEntropy(),
        }

def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
    return wrapped

@background
def process_dataset(path: str) -> None:
    """Process a single dataset asynchronously."""
    try:
        reconstructor = OptimalCausationEntropyReconstructor()
        reconstructor.reconstruct(f"data/{path}")
    except Exception as e:
        logger.error(f"Failed to process {path}: {str(e)}")

def main():
    """Main function to run optimal causation entropy-based reconstructions."""
    datasets = [
        "traffic.txt.gz",
        "solar_AL.txt.gz",
        "electricity.txt.gz",
        "exchange_rate.txt.gz"
    ]
    
    # Process datasets in parallel
    tasks = [process_dataset(dataset) for dataset in datasets]
    asyncio.get_event_loop().run_until_complete(asyncio.gather(*tasks))

if __name__ == "__main__":
    main()