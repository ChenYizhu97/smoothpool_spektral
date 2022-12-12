"""
This module provides utils functions used in trainning.
"""
from __future__ import annotations 
import numpy as np


def save_data(file_name, results, pool_method_descriptor: str):
    """Saves the metrics to file.
    """ 
    accs, losses, epochs = results
    with open(file_name, "a") as f:
        f.write(pool_method_descriptor+"\n")
        for acc in accs:
            f.write(f"{acc:.4f}, ")            
        f.write("\n")
        for loss in losses:
            f.write(f"{loss:.4f}, ")            
        f.write("\n")
        for epoch in epochs:
            f.write(f"{epoch:.4f}, ")            
        f.write("\n")
        std = np.std(accs)
        avg = np.mean(accs)
        avg_epoch=int(np.mean(epochs))
        f.write(f"mean: {avg:.4f}, stdev: {std:.4f}, average epochs run {avg_epoch}\n")

def generate_random_seeds(filename:str, n:int):
    """Generates random seeds, and writes them into the given file.

    Args:
        filename: The name of file. A string.
        n: Number of seeds. An int.
    """

    seeds = np.random.randint(low=0, high=1000, size=n)
    seeds = seeds.tolist()
    with open(filename,"w") as f:
        for seed in seeds:
            f.write(f"{seed},")

def read_seeds(filename:str) -> list[int]:
    """Reads seeds from the given file.

    Args:
        filename: The name of file. A string.
    
    Returns:
        seeds: List of seeds. A list of int.
    """

    seeds = []

    with open(filename, "r") as f:
        seeds = f.readline()
        seeds = seeds.split(",")[:-1]
        seeds = [int(seed) for seed in seeds]
    return seeds

def generate_experiment_descriptor(pool:str="smoothpool",
                                k:float=0.5, 
                                r:int=30, 
                                batch:int=32, 
                                lr:float=0.01, 
                                epochs:int=400, 
                                p:int=40, 
                                d:str="",
                                edges:bool=False,
                                dataset:str="FRANKENSTEIN"
                                )->str:
    """Generate string that descripts the method and parameters used
    
    Return:
        descriptor: String that descripts the method and parameters used.
    """
    descriptor = f"pooling operator: {pool} ; k: {k}; r: {r}; batch size: {batch}; lr: {lr}; maximum epochs: {epochs}; patience: {p}; fixed seeds;"

    additional_descriptor = d

    if pool == "smoothpool":
        if edges:
            descriptor += "Edge features;"
        #if args.augment:
        #    args.augment = calculate_augment(data)
        #    descriptor += f"Connectivity augment is {args.augment}. "

    if additional_descriptor != "":
        descriptor += additional_descriptor

    print("*"*10)
    print(f"Dataset used is {dataset}...")
    print(descriptor)
    print("*"*10)
    
    return descriptor