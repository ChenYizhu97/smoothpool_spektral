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

parser = argparse.ArgumentParser(description="Trains and evaluates pooling methods...")
parser.add_argument("--batch", default="32", type=int, help="Batch size. 32 by default.",)
parser.add_argument("--lr", default="0.01", type=float, help="Learning rate. 0.01 by default.",)
parser.add_argument("--epochs", default="400", type=int, help="Maximum epoches. 400 by default.",)
parser.add_argument("-k", default="0.5", type=float, help="Ratio of pooling. 0.5 by default.",)
parser.add_argument("--dataset", default="FRANKENSTEIN", help="Dataset for testing. FRANKENSTEIN by default.",
                    choices=["FRANKENSTEIN", "PROTEINS", "DD"]
                    )
parser.add_argument("--pool", default="topkpool", help="Pooling method to use. topkpool by default.",
                    choices=["smoothpool", "topkpool", "sagpool", "diffpool"]
                    )
parser.add_argument("--edges", action="store_true", help="Use edge features.")
parser.add_argument("--augment", action="store_true", help="Connectivity augmentation.")
parser.add_argument("-r", default="30", type=int, help="Running times of evaluation. 30 by default.")
parser.add_argument("-p", default="40", type=int, help="Patience of early stopping. 40 by default.")
parser.add_argument("-d", default="", type=str, help="Addtional description to be logged.")

args = parser.parse_args()

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

method_descriptor = f"The pooling method used is {args.pool}. K is {args.k}. \
Hyperparameters are set as follow. Batch size: {args.batch}. Learning rate: {args.lr}, \
maximum epochs: {args.epochs}. Runing time: {args.r}. Patience: {args.p}. "

additional_descriptor = args.d
data = TUDataset(args.dataset)

if args.pool == "smoothpool":
    if args.edges:
        method_descriptor += "Edge features are used for pooling. "
    if args.augment:
        args.augment = calculate_augment(data)
        method_descriptor += f"Connectivity augment is {args.augment}. "

if additional_descriptor != "":
    method_descriptor += additional_descriptor

print("*"*10)
print(f"Dataset used is {args.dataset}...")
print(method_descriptor)
print("*"*10)

if args.pool == "diffpool":
    k = ratio_to_number(args.k, data)
else:
    k = args.k

accs=[]
epochs_run=[]
length_dataset = len(data)
for i in range(args.r):
    # Initialize model, loss function and optimizer.
    model = HpoolGNN(data.n_labels, k=k, activation="softmax", 
                        h_pool=args.pool, 
                        connectivity_augment=args.augment,
                        use_edge_features=args.edges)
    optimizer = Adam(args.lr)
    loss_fn = CategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[CategoricalAccuracy(name="acc"),])
    
    # Split training, validation and testing dataset   
    idx_tr, idx_va, idx_te = shuffle_and_split(length_dataset)
    data_tr, data_va, data_te = data[idx_tr], data[idx_va], data[idx_te]

    if args.pool == "diffpool":
        loader_tr = BatchLoader(data_tr, batch_size=args.batch, epochs=args.epochs)
        loader_va = BatchLoader(data_va, batch_size=args.batch,)
        loader_te = BatchLoader(data_te, batch_size=args.batch,)
    else:
        loader_tr = DisjointLoader(data_tr, batch_size=args.batch, epochs=args.epochs)
        loader_va = DisjointLoader(data_va, batch_size=args.batch,)
        loader_te = DisjointLoader(data_te, batch_size=args.batch)
    
    # Train with early stopping
    earlystopping_monitor= EarlyStopping(patience=args.p, restore_best_weights=True, monitor="val_loss", mode="min")
    model.fit(
        loader_tr.load(),
        steps_per_epoch=loader_tr.steps_per_epoch,
        epochs=args.epochs,
        callbacks=[earlystopping_monitor],
        validation_data=loader_va,
        validation_steps=loader_va.steps_per_epoch,
        verbose=0,
    )

    print(f"The {i+1} run completes, testing model...")
    
    # Evaluate on test set and record metrics
    loss, acc = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
    epoch_run = earlystopping_monitor.stopped_epoch
    
    print(f"Done. The {i+1} run. Test loss: {loss}. Test acc: {acc}. Epoch run: {epoch_run}")
    
    accs.append(acc)
    epochs_run.append(epoch_run)
    
    clear_session()

file_name = f"DATA_{args.dataset}.txt"
save_data(file_name, accs, method_descriptor, epochs_run)
#print(accs)