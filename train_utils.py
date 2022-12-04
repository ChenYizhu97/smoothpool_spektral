import tensorflow as tf
import numpy as np
from spektral.data.loaders import BatchLoader, DisjointLoader
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session
from utils import create_model


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

def generate_method_description(pool:str="smoothpool",
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
        method_descriptor: String that descripts the method and parameters used.
    """
    method_descriptor = f"The pooling method used is {pool}. K is {k}. Running time is {r}. \
    Hyperparameters are set as follow. Batch size: {batch}. Learning rate: {lr}, \
    maximum epochs: {epochs}. Runing time: {r}. Patience: {p}. Using fixed seeds. "

    additional_descriptor = d

    if pool == "smoothpool":
        if edges:
            method_descriptor += "Edge features are used for pooling. "
        #if args.augment:
        #    args.augment = calculate_augment(data)
        #    method_descriptor += f"Connectivity augment is {args.augment}. "

    if additional_descriptor != "":
        method_descriptor += additional_descriptor

    print("*"*10)
    print(f"Dataset used is {dataset}...")
    print(method_descriptor)
    print("*"*10)
    
    return method_descriptor

def run_with_randomsplit(dataset, pool, batch, epochs, edges, lr, p, k, seed=None):
    length_dataset = len(dataset)
    # Set fixed seed for each run.
    tf.keras.utils.set_random_seed(seed)
    idx_tr, idx_va, idx_te = shuffle_and_split(length_dataset)
    data_tr, data_va, data_te = dataset[idx_tr], dataset[idx_va], dataset[idx_te]
    if pool == "diffpool":
        loader_tr = BatchLoader(data_tr, batch_size=batch, epochs=epochs, shuffle=False)
        loader_va = BatchLoader(data_va, batch_size=batch, shuffle=False)
        loader_te = BatchLoader(data_te, batch_size=batch, shuffle=False)
    else:
        loader_tr = DisjointLoader(data_tr, batch_size=batch, epochs=epochs, shuffle=False)
        loader_va = DisjointLoader(data_va, batch_size=batch, shuffle=False)
        loader_te = DisjointLoader(data_te, batch_size=batch, shuffle=False)
    
    acc, loss, epoch = train_and_eval(
        dataset.n_labels,
        pool,
        edges,
        epochs,
        lr,
        p,
        k,
        loader_tr, loader_te, loader_va=loader_va)
    return acc, loss, epoch

def train_and_eval(n_labels, pool, edges, epochs, lr, p, k, loader_tr, loader_te, loader_va=None):
    """sets up model, trains and evaluates"""
    model = create_model(n_labels, pool, k, use_edge_features=edges, activation="softmax")
    optimizer = Adam(lr)
    loss_fn = CategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[CategoricalAccuracy(name="acc"),])
    # if there is a validation set, early stoping monitors the validation loss.
    if loader_va is not None:
        earlystopping_monitor= EarlyStopping(patience=p, restore_best_weights=True, monitor="val_loss", mode="min")
        model.fit(
            loader_tr,
            steps_per_epoch=loader_tr.steps_per_epoch,
            epochs=epochs,
            validation_data=loader_va,
            validation_steps=loader_va.steps_per_epoch,
            callbacks=[earlystopping_monitor],
            verbose=0,
        )
    else:
        earlystopping_monitor= EarlyStopping(patience=p, restore_best_weights=True, monitor="loss", mode="min")
        model.fit(
            loader_tr,
            steps_per_epoch=loader_tr.steps_per_epoch,
            epochs=epochs,
            callbacks=[earlystopping_monitor],
            verbose=0,
        )
    loss, acc = model.evaluate(loader_te, steps=loader_te.steps_per_epoch)
    epoch_run = earlystopping_monitor.stopped_epoch   
    clear_session()
    return acc, loss, epoch_run

"""
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
"""