"""
This module provides utils functions used in trainning.
"""
from __future__ import annotations
import statistics 
import numpy as np


def save_data(file_name, datas: list[float], pool_method_descriptor: str, epochs: list[int]):
    """Saves the metrics to file.

    Args:
        file_name: The name of file for saving data. A string.
        datas: List of metrics. A list of float.
        pool_method_descriptor: A string describing the pooling method.
        epochs: Epochs that have been done. A list of int.
    """ 
    with open(file_name, "a") as f:
        f.write(pool_method_descriptor+"\n")
        for data in datas:
            f.write(f"{data:.4f}, ")            
        f.write("\n")
        std = statistics.stdev(datas)
        avg = statistics.mean(datas)
        avg_epoch=int(statistics.mean(epochs))
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
    descriptor = f"The pooling method used is {pool}. K is {k}. Running time is {r}. \
    Hyperparameters are set as follow. Batch size: {batch}. Learning rate: {lr}, \
    maximum epochs: {epochs}. Runing time: {r}. Patience: {p}. Using fixed seeds. "

    additional_descriptor = d

    if pool == "smoothpool":
        if edges:
            descriptor += "Edge features are used for pooling. "
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