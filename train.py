from statistics import stdev, mean
import numpy as np
import networkx as nx
import tensorflow as tf
from spektral.data import DisjointLoader, BatchLoader
from spektral.datasets import TUDataset
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session
from model import HpoolGNN
from utils import save_data, shuffle_and_split, ratio_to_number, calculate_augment 


physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


batch_size = 32
learning_rate = 0.01
epochs = 400
k = 0.5
dataset = "COX2_MD"
#dataset = "PROTEINS"
#dataset = "DD"
#dataset = "FRANKENSTEIN"
data = TUDataset(dataset)
pool_method = "diffpool"
connectivity_augment = False
use_edge_features = False

method_descriptor = f"The pooling method used is {pool_method}. K is {k}. "
additional_descriptor = ""

if pool_method == "smoothpool":
    if use_edge_features:
        method_descriptor += "Edge features are used for pooling. "
    if connectivity_augment:
        connectivity_augment = calculate_augment(data)
        method_descriptor += f"Connectivity augment is {connectivity_augment}. "

if additional_descriptor != "":
    method_descriptor += additional_descriptor

print(method_descriptor)

if pool_method == "diffpool":
    k = ratio_to_number(k, data)

accs=[]
epochs_run=[]
length_dataset = len(data)
for i in range(80):
    model = HpoolGNN(data.n_labels, k=k, activation="softmax", 
                        h_pool=pool_method, 
                        connectivity_augment=connectivity_augment,
                        use_edge_features=use_edge_features)
    optimizer = Adam(learning_rate)
    loss_fn = CategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[CategoricalAccuracy(name="acc"),])
    
    # Train/test split    
    idx_tr, idx_va, idx_te = shuffle_and_split(length_dataset)
    data_tr, data_va, data_te = data[idx_tr], data[idx_va], data[idx_te]

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
save_data(file_name, accs, method_descriptor, epochs_run)
#print(accs)