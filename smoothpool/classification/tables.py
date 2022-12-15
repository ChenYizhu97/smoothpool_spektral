import numpy as np
from smoothpool.classification.train import load_data
import pandas as pd
import os
from scipy.stats import rankdata
from collections import defaultdict

datasets = ["FRANKENSTEIN", "PROTEINS", "Mutagenicity", "ENZYMES", "COX2", "ogbg-molclintox", "ogbg-molbbbp", "ogbg-molbace",]
datasets_edgeatrr = ["ER_MD", "COX2_MD", "BZR_MD"]
datasets_info_header = ["graphs", "average nodes", "average edges", "classes", "edge attributes"]
pooler_header = ["smoothpool", "topkpool", "sagpool", "diffpool", "smoothpool(edge)"]
datasets_full = datasets + datasets_edgeatrr

def plot_dataset_info(datasets):
    info_dict = {}
    for dataset in datasets:
        data, _ = load_data(dataset)
        nodes_ave = data.map(lambda g: g.n_nodes, reduce=lambda res: np.ceil(np.mean(res)))
        edges_ave = data.map(lambda g: g.n_edges, reduce=lambda res: np.ceil(np.mean(res)))
        edge_attr = True if dataset in datasets_edgeatrr else False
        graphs = data.n_graphs   
        #print(f"{dataset}. graphs: {graphs}; classes: {data.n_labels}; average nodes: {nodes_ave}; average edges: {edges_ave}; edge features: {edge_feature}")
        info_dict[dataset] = [graphs, nodes_ave, edges_ave, data.n_labels, edge_attr]
        info_table = pd.DataFrame.from_dict(info_dict, orient="index", columns=datasets_info_header)
    return info_table.to_latex(escape=False, bold_rows=True)

def split_and_strip(raw_str, seperator=","):
    out_str = raw_str.split(seperator)[:-1]
    out_str = [s.strip() for s in out_str]
    return out_str

def get_value(kv):
    v = kv.split(":")[-1].strip()
    return v

def process_recorder(record):
    method_description = split_and_strip(record[0], seperator=";")
    method = get_value(method_description[0])
    if "edge features" in method_description:
        method += "(edge)"
    avg, std = split_and_strip(record[-1])[:2]
    avg = get_value(avg)
    std = get_value(std)
    return method, [float(avg), float(std)]

def add_rank(records):
    methods = []
    accs = []
    for method, (acc, _) in records.items():
        methods.append(method)
        accs.append(-acc)
    ranks = rankdata(accs, method='dense')
    for i, method in enumerate(methods):
        records[method].append(ranks[i])
    return records

def get_results(datasets):
    results_dict = {}
    for dataset in datasets:
        records = {}
        file_path = os.path.join("results", f"{dataset}.txt")
        with open(file_path, "r") as f:
            results = f.read().split("\n\n")[:-1]
        results = [result.split("\n") for result in results]
        for record in results:
            method, acc = process_recorder(record)
            records[method] = acc
        records = add_rank(records)
        results_dict[dataset] = records
        #print(records)       
    return results_dict

def repr_acc_latex(avg, std):
    acc = rf"{100*avg:.1f} \tiny{{$\pm${100*std:.1f}}}"
    return acc

def plot_results(results_dict, pooler_header):
    ranks = defaultdict(list)
    print(pooler_header)
    for dataset, records in results_dict.items():
        accs = []
        for method in pooler_header:
            avg, std, rank = records[method]
            accs.append(repr_acc_latex(avg, std))       
            ranks[method].append(rank)          
        results_dict[dataset]= accs
    for method, rank in ranks.items():
        ranks[method] = np.mean(rank)
    ranks = [ranks[method] for method in pooler_header]
    ranks = [rf"\textbf\{{{rank}}}" for rank in ranks]
    results_dict["Rank"] = ranks
    print(results_dict)
    results_table = pd.DataFrame.from_dict(results_dict, orient="index", columns=pooler_header)

    return results_table.to_latex(escape=False, bold_rows=True)

if __name__ == "__main__":
    print(plot_results(get_results(datasets), pooler_header[:-1]))
    #print(plot_dataset_info(datasets_full))
    #print(get_results(datasets))

'''
info_dict = get_dataset_info(datasets)

'''