from posixpath import split
from statistics import stdev, mean
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session
from spektral.data import DisjointLoader, BatchLoader
from spektral.datasets import TUDataset
from model import HpoolGNN
import networkx as nx

def save_data(file_name, datas, pool_method, epochs):    
    with open(file_name, "a") as f:
        f.write(f"data for {pool_method}:\n")
        for data in datas:
            f.write(f"{data}, ")            
        f.write("\n")
        std = stdev(datas)
        avg = mean(datas)
        avg_epoch=mean(epochs)
        f.write(f"mean: {avg}, stdev: {std}, average epochs run {avg_epoch}\n")

def shuffle_and_split(data):
    idxs = np.random.permutation(len(data))
    split_va, split_te = int(0.8 * len(data)), int(0.9 * len(data))
    idx_tr, idx_va, idx_te = np.split(idxs, [split_va, split_te])
    data_tr = data[idx_tr]
    data_va = data[idx_va]
    data_te = data[idx_te]
    return data_tr, data_va, data_te



physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


batch_size = 32
learning_rate = 0.01
epochs = 400
k = 0.3
#dataset = "COX2_MD"
#dataset = "PROTEINS"
#dataset = "DD"
dataset = "FRANKENSTEIN"
data = TUDataset(dataset)
pool_method = "diffpool"
connectivity_augment = False
use_edge_features = False

if pool_method == "smoothpool":
    print(f"The k is {k}, The pooling method used is {pool_method}, connectivity augment is {connectivity_augment}, use edge features is {use_edge_features}.")
else:
    print(f"The k is {k}, The pooling method used is {pool_method}.")

if pool_method == "diffpool":
    n = 0
    for graph in data:
        n += graph.n_nodes
    n /= data.n_graphs
    k = int(k*n)

if connectivity_augment:
    aves = []
    for _data in data:
        adj = _data.a
        g = nx.from_scipy_sparse_array(adj)
        ave_c = []
        for C in (g.subgraph(c).copy() for c in nx.connected_components(g)):
            ave = nx.average_shortest_path_length(C)
            ave_c.append(ave)
        ave = sum(ave_c)/len(ave_c)
        aves.append(ave)
    ave = int(sum(aves)/len(aves))  
    connectivity_augment = ave
    print(f"connectivity_augment is {connectivity_augment}.")

accs=[]
epochs_run=[]
for i in range(80):
    model = HpoolGNN(data.n_labels, k=k, activation="softmax", 
                        i_hpool=3, 
                        h_pool=pool_method, 
                        connectivity_augment=connectivity_augment,
                        use_edge_features=use_edge_features)
    optimizer = Adam(learning_rate)
    loss_fn = CategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[CategoricalAccuracy(name="acc"),])
    
    # Train/test split
    data_tr, data_va, data_te = shuffle_and_split(data)

    if pool_method == "diffpool":
        loader_tr = BatchLoader(data_tr, batch_size=batch_size, epochs=epochs)
        loader_va = BatchLoader(data_va, batch_size=batch_size,)
        loader_te = BatchLoader(data_te, batch_size=batch_size,)
    else:
        loader_tr = DisjointLoader(data_tr, batch_size=batch_size, epochs=epochs)
        loader_va = DisjointLoader(data_va, batch_size=batch_size,)
        loader_te = DisjointLoader(data_te, batch_size=batch_size)

    earlystopping_monitor= EarlyStopping(patience=40, restore_best_weights=True, monitor="val_loss", mode="min")
    model.fit(
        loader_tr.load(),
        steps_per_epoch=loader_tr.steps_per_epoch,
        epochs=epochs,
        callbacks=[earlystopping_monitor],
        validation_data=loader_va,
        validation_steps=loader_va.steps_per_epoch,
        verbose=0,
    )
    print(f"The {i+1} run completes, testing model...")
    
    loss, acc = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
    epoch_run = earlystopping_monitor.stopped_epoch
    
    print(f"Done. The {i+1} run. Test loss: {loss}. Test acc: {acc}. Epoch run: {epoch_run}")
    
    accs.append(acc)
    epochs_run.append(epoch_run)
    
    clear_session()

file_name = f"DATA_{dataset}.txt"
save_data(file_name, accs, pool_method, epochs_run)
#print(accs)