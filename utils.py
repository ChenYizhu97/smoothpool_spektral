"""
This module provides utils functions used in trainning.
"""
from __future__ import annotations
import statistics 
import numpy as np
import networkx as nx
from spektral.layers import TopKPool, SAGPool, DiffPool
from model import BasicModel
from smoothpool import SmoothPool

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


def ratio_to_number(k: float, dataset) -> int:
    """Calculates fix number of centroids from the ratio of pooling.
    
    Args:
        k: Ratio of pooling. A float.
        dataset: Spektral dataset.

    Returns:
        k_fixed: Fix number of centroids. An int.
    """
    n = 0
    for graph in dataset:
        n += graph.n_nodes
    n /= dataset.n_graphs

    k_fixed = int(k*n)
    return k_fixed

def calculate_augment(dataset) -> int:
    """Calculates the value for connectivity augmentation used by smoothpool.
    
    Args:
        dataset: A Spektral dataset.
    
    Returns:
        connectivity_augment: Value for connectivity augmentation. An int.
    """
    aves = []
    for graph in dataset:
        adj = graph.a
        g = nx.from_scipy_sparse_array(adj)
        ave_c = []
        for C in (g.subgraph(c).copy() for c in nx.connected_components(g)):
            ave = nx.average_shortest_path_length(C)
            ave_c.append(ave)
        ave = sum(ave_c)/len(ave_c)
        aves.append(ave)
    connectivity_augment = int(sum(aves)/len(aves))  

    return connectivity_augment

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

def create_model(out, pooling_method:str, k, use_edge_features=False, activation="softmax") -> BasicModel:
    """Returns the hierarchical gnn model for given pooling methods.
    
    Args:
        out: Number of out channels. A int. 
        pooling_method: Pooling method to be used. A string.
        k: Ratio of pooling. A float between 0 and 1. For diffpool, its a fixed int numbers.
        use_edge_features: Whether to use edge features. A boolean.
        activation: Activation function that keras supports. A string
    
    Returns:
        model: A hierarchical pooling GNN model. A subclass of HpoolGNN.
    """
    pool = lambda x:x
    if pooling_method == "smoothpool":
        pool = SmoothPool(k, use_edge_features=use_edge_features)
    if pooling_method == "topkpool":
        pool = TopKPool(k)
    if pooling_method == "sagpool":
        pool = SAGPool(k)
    if pooling_method == "diffpool":
        pool = DiffPool(k, 256)

    model = BasicModel(out, pool=pool, use_edge_features=use_edge_features, activation=activation)
    return model
