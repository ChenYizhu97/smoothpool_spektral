import os
import argparse
import tensorflow as tf
from spektral.data import DisjointLoader, BatchLoader
from spektral.datasets import TUDataset
from sklearn.model_selection import KFold
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session
from utils import save_data, get_model, ratio_to_number, shuffle_and_split, read_seeds

parser = argparse.ArgumentParser(description="Trains and evaluates pooling methods...")
parser.add_argument("--batch", default="32", type=int, help="Batch size. 32 by default.",)
parser.add_argument("--lr", default="0.01", type=float, help="Learning rate. 0.01 by default.",)
parser.add_argument("--epochs", default="400", type=int, help="Maximum epoches. 400 by default.",)
parser.add_argument("-k", default="0.5", type=float, help="Ratio of pooling. 0.5 by default.",)
parser.add_argument("--dataset", default="FRANKENSTEIN", help="Dataset for testing. FRANKENSTEIN by default.",
                    choices=["FRANKENSTEIN", "PROTEINS", "AIDS", "ENZYMES"]
                    )
parser.add_argument("--pool", default="topkpool", help="Pooling method to use. topkpool by default.",
                    choices=["smoothpool", "topkpool", "sagpool", "diffpool"]
                    )
parser.add_argument("--edges", action="store_true", help="Use edge features.")
parser.add_argument("--augment", action="store_true", help="Connectivity augmentation.")
parser.add_argument("-r", default="30", type=int, help="Running times of evaluation. 30 by default.")
parser.add_argument("-p", default="40", type=int, help="Patience of early stopping. 40 by default.")
parser.add_argument("-d", default="", type=str, help="Addtional description to be logged.")
parser.add_argument("-fseeds", default="random_seeds", type=str, help="File that holds fixed seeds.")

args = parser.parse_args()
#set GPU
physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
#load spektral dataset
data = TUDataset(args.dataset)
# Ensure GPU reproducibility
#os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
#os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'

def generate_method_description()->str:
    """Generate string that descripts the method and parameters used
    
    Return:
        method_descriptor: String that descripts the method and parameters used.
    """
    method_descriptor = f"The pooling method used is {args.pool}. K is {args.k}. Running time is {args.r}. \
    Hyperparameters are set as follow. Batch size: {args.batch}. Learning rate: {args.lr}, \
    maximum epochs: {args.epochs}. Runing time: {args.r}. Patience: {args.p}. Using fixed seeds. "

    additional_descriptor = args.d

    if args.pool == "smoothpool":
        if args.edges:
            method_descriptor += "Edge features are used for pooling. "
        #if args.augment:
        #    args.augment = calculate_augment(data)
        #    method_descriptor += f"Connectivity augment is {args.augment}. "

    if additional_descriptor != "":
        method_descriptor += additional_descriptor

    print("*"*10)
    print(f"Dataset used is {args.dataset}...")
    print(method_descriptor)
    print("*"*10)
    
    return method_descriptor

#For diffpool, calculate the fixed numbers of pooled nodes.
if args.pool == "diffpool":
    k = ratio_to_number(args.k, data)
else:
    k = args.k


def train_and_eval(loader_tr, loader_te, loader_va=None):
    """sets up model, trains and evaluates"""
    model = get_model(data.n_labels, args.pool, k, use_edge_features=args.edges, activation="softmax")
    optimizer = Adam(args.lr)
    loss_fn = CategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[CategoricalAccuracy(name="acc"),])
    # if there is a validation set, early stoping monitors the validation loss.
    if loader_va is not None:
        earlystopping_monitor= EarlyStopping(patience=args.p, restore_best_weights=True, monitor="val_loss", mode="min")
        model.fit(
            loader_tr,
            steps_per_epoch=loader_tr.steps_per_epoch,
            epochs=args.epochs,
            validation_data=loader_va,
            validation_steps=loader_va.steps_per_epoch,
            callbacks=[earlystopping_monitor],
            verbose=0,
        )
    else:
        earlystopping_monitor= EarlyStopping(patience=args.p, restore_best_weights=True, monitor="loss", mode="min")
        model.fit(
            loader_tr,
            steps_per_epoch=loader_tr.steps_per_epoch,
            epochs=args.epochs,
            callbacks=[earlystopping_monitor],
            verbose=0,
        )
    loss, acc = model.evaluate(loader_te, steps=loader_te.steps_per_epoch)
    epoch_run = earlystopping_monitor.stopped_epoch   
    clear_session()
    return acc, loss, epoch_run

def run_with_10fold(i_run:int, length_dataset:int, seed=None):
    # Set fixed seed for each run.
    tf.keras.utils.set_random_seed(seed)
    #10-fold cross validation
    kf10 = KFold(10, shuffle=True, random_state=seed)
    acc_folds = 0
    epoch_folds = 0
    for i_fold, (idx_tr, idx_te) in enumerate(kf10.split(range(length_dataset))):
        # prepare train and test set for each fold.
        data_tr, data_te = data[idx_tr],  data[idx_te]
        if args.pool == "diffpool":
            loader_tr = BatchLoader(data_tr, batch_size=args.batch, epochs=args.epochs)
            loader_te = BatchLoader(data_te, batch_size=args.batch,)
        else:
            loader_tr = DisjointLoader(data_tr, batch_size=args.batch, epochs=args.epochs,)
            loader_te = DisjointLoader(data_te, batch_size=args.batch,)
        # Initialize model, train and evaluate.
        acc, loss, epoch = train_and_eval(loader_tr, loader_te)
        print(f"Done. The {i_fold+1} fold of {i_run+1} run. Test loss: {loss}. Test acc: {acc}. Epoch run: {epoch}")
        acc_folds += acc
        epoch_folds += epoch
    acc_run = acc_folds/10
    epoch_run = epoch_folds/10
    return acc_run, epoch_run

def run_with_randomsplit(i_run:int, length_dataset:int, seed=None):
    # Set fixed seed for each run.
    tf.keras.utils.set_random_seed(seed)
    idx_tr, idx_va, idx_te = shuffle_and_split(length_dataset)
    data_tr, data_va, data_te = data[idx_tr], data[idx_va], data[idx_te]
    if args.pool == "diffpool":
        loader_tr = BatchLoader(data_tr, batch_size=args.batch, epochs=args.epochs, shuffle=False)
        loader_va = BatchLoader(data_va, batch_size=args.batch, shuffle=False)
        loader_te = BatchLoader(data_te, batch_size=args.batch, shuffle=False)
    else:
        loader_tr = DisjointLoader(data_tr, batch_size=args.batch, epochs=args.epochs, shuffle=False)
        loader_va = DisjointLoader(data_va, batch_size=args.batch, shuffle=False)
        loader_te = DisjointLoader(data_te, batch_size=args.batch, shuffle=False)
    
    acc, loss, epoch = train_and_eval(loader_tr, loader_te, loader_va=loader_va)
    return acc, loss, epoch

# run the experiments for args.r times and save the metrics
accs=[]
epochs=[]
length_dataset = len(data)
#load fixed seeds
seeds = read_seeds(args.fseeds)
method_descriptor = generate_method_description()
for i_run in range(args.r):
    #acc, epoch = run_with_10fold(i_run, length_dataset, seed=seeds[i_run])     
    #print(f"Done. The {i_run+1} run. Test acc: {acc}. Epoch run: {epoch}")
    acc, loss, epoch = run_with_randomsplit(i_run, length_dataset, seed=seeds[i_run])
    print(f"Done. The {i_run+1} run. Test loss: {loss}. Test acc: {acc}. Epoch run: {epoch}")
    accs.append(acc)
    epochs.append(epoch)
    
file_name = f"DATA_{args.dataset}.txt"
save_data(file_name, accs, method_descriptor, epochs)
