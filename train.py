import argparse
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

parser = argparse.ArgumentParser(description="Trains and evaluates pooling methods...")
parser.add_argument("--batch", default="32", type=int, help="Batch size.",)
parser.add_argument("--lr", default="0.01", type=float, help="Learning rate.",)
parser.add_argument("--epochs", default="400", type=int, help="Maximum epoches.",)
parser.add_argument("-k", default="0.5", type=float, help="Ratio of pooling.",)
parser.add_argument("--dataset", default="FRANKENSTEIN", help="Dataset for testing.",
                    choices=["FRANKENSTEIN", "PROTEINS", "DD"]
                    )
parser.add_argument("--pool", default="topkpool", help="Pooling method to use.",
                    choices=["smoothpool", "topkpool", "sagpool", "diffpool"]
                    )
parser.add_argument("--edges", action="store_true", help="Use edge features.")
parser.add_argument("--augment", action="store_true", help="Connectivity augmentation.")
parser.add_argument("-r", default="80", type=int, help="Running times of evaluation.")
parser.add_argument("-p", default="40", type=int, help="Patience of early stopping.")
parser.add_argument("-d", default="", type=str, help="Addtional description to be logged.")

args = parser.parse_args()
batch_size = args.batch
learning_rate = args.lr
epochs = args.epochs
k = args.k
dataset = args.dataset
#dataset = "PROTEINS"
#dataset = "DD"
#dataset = "FRANKENSTEIN"
data = TUDataset(dataset)
pool_method = args.pool
connectivity_augment = args.augment
use_edge_features = args.edges
run_times = args.r
patience = args.p

method_descriptor = f"The pooling method used is {pool_method}. K is {k}. "
additional_descriptor = args.d

if pool_method == "smoothpool":
    if use_edge_features:
        method_descriptor += "Edge features are used for pooling. "
    if connectivity_augment:
        connectivity_augment = calculate_augment(data)
        method_descriptor += f"Connectivity augment is {connectivity_augment}. "

if additional_descriptor != "":
    method_descriptor += additional_descriptor

print("*"*10)
print(f"Dataset used is {dataset}...")
print(method_descriptor)
print("*"*10)

if pool_method == "diffpool":
    k = ratio_to_number(k, data)

accs=[]
epochs_run=[]
length_dataset = len(data)
for i in range(run_times):
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

    earlystopping_monitor= EarlyStopping(patience=patience, restore_best_weights=True, monitor="val_loss", mode="min")

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