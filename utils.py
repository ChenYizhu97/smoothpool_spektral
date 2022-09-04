"""
This module provides utils functions used in trainning.
"""
from __future__ import annotations
from statistics import stdev, mean
import numpy as np
import networkx as nx


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
            f.write(f"{data}, ")            
        f.write("\n")
        std = stdev(datas)
        avg = mean(datas)
        avg_epoch=mean(epochs)
        f.write(f"mean: {avg}, stdev: {std}, average epochs run {avg_epoch}\n")

def shuffle_and_split(length_dataset: int) -> tuple:    
    """Returns training, validation and test splitting indexs for dataset with length length_dataset.

    Args:
        length_dataset: The length of dataset. A int.
    
    Returns:
        idx_tr: Indexs of training set. A list of numpy array.
        idx_va: Indexs of validation set . A list of numpy array.
        idx_te: Indexs of test set. A list of numpy array.
    """
    idxs = np.random.permutation(length_dataset)
    split_va, split_te = int(0.8 * length_dataset), int(0.9 * length_dataset)
    idx_tr, idx_va, idx_te = np.split(idxs, [split_va, split_te])

    return idx_tr, idx_va, idx_te

def ratio_to_number(k: float, dataset) -> int:
    """Calculates a fix number of centroids from the ratio of pooling.
    
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