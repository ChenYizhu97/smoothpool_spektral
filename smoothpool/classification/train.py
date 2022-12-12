import numpy as np
import tensorflow as tf
from ogb.graphproppred import GraphPropPredDataset
from spektral.datasets import TUDataset, OGB
from spektral.transforms import OneHotLabels
from spektral.layers import TopKPool, SAGPool, DiffPool
from spektral.data.loaders import BatchLoader, DisjointLoader
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session
from smoothpool.models.classifier import BasicModel
from smoothpool.models.smoothpool import SmoothPool


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

def transformer_ogb(graph):
    graph.y = graph.y[0]
    graph.a = graph.a.astype("f4")
    return graph

def load_data(dataset):
    split_idx = None
    if dataset in TUDataset.available_datasets():
        data = TUDataset(dataset)
    elif dataset.startswith("ogbg"):
        data = GraphPropPredDataset(dataset)
        #get split index provided by ogb
        idx = data.get_idx_split()
        idx_tr, idx_va, idx_te = idx["train"], idx["valid"], idx["test"]
        data = OGB(data)
        #one hot encode for ogb dataset.
        labels = data.map(lambda g: g.y, reduce=np.unique)
        data.apply(transformer_ogb)
        data.apply(OneHotLabels(labels=labels))
        split_idx = [idx_tr, idx_va, idx_te]
    else:
        data = None
    return data, split_idx

def run_classifier(dataset, pool, batch, epochs, edges, lr, p, k, split_idx=None, seed=None):
    #sets fixed seed for each run.
    tf.keras.utils.set_random_seed(seed)
    #split data
    if split_idx is not None:
        idx_tr, idx_va, idx_te = split_idx
    else:
        l_data = len(dataset)
        idxs = np.random.permutation(l_data)
        split_va, split_te = int(0.8 * l_data), int(0.9 * l_data)
        idx_tr, idx_va, idx_te = np.split(idxs, [split_va, split_te])
    data_tr, data_va, data_te = dataset[idx_tr], dataset[idx_va], dataset[idx_te]    
    
    if pool == "diffpool":
        loader_class = BatchLoader
        #calculates fixed k for diffpool
        N_avg = dataset.map(lambda g: g.n_nodes, reduce=lambda res: np.ceil(np.mean(res)))
        k = int(k*N_avg)
    else:
        loader_class = DisjointLoader
    #wraps data in spektral loader
    loader_tr = loader_class(data_tr, batch_size=batch, epochs=epochs, shuffle=True)
    loader_va = loader_class(data_va, batch_size=batch, shuffle=True)
    loader_te = loader_class(data_te, batch_size=batch, shuffle=True)
    #creates model, loss function and optimizer
    model = create_model(dataset.n_labels, pool, k, use_edge_features=edges, activation="softmax")
    loss_fn = CategoricalCrossentropy()
    acc_fn = CategoricalAccuracy()
    optimizer = Adam(lr)
    input_signature = loader_tr.tf_signature()
    
    @tf.function(input_signature=input_signature, experimental_relax_shapes=True)
    def train_step(inputs, targets):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss_value = loss_fn(targets, predictions)  # Main loss
            loss_value += sum(model.losses)  # Auxiliary losses of the model
            acc_value = acc_fn(targets, predictions)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss_value, acc_value
    
    @tf.function(input_signature=input_signature, experimental_relax_shapes=True)
    def evaluate_step(inputs, targets):
        predictions = model(inputs, training=False)
        loss_value = loss_fn(targets, predictions)
        acc_value = acc_fn(targets, predictions)
        return loss_value, acc_value
    
    def evaluate(loader):
        results = []
        for _ in range(loader.steps_per_epoch):
            data_batch = next(loader)
            result = evaluate_step(*data_batch)
            results.append(result)
        return np.mean(results, 0)
        
    batchs_per_epoch = loader_tr.steps_per_epoch
    acc_te = epoch_run = wait = 0 
    loss_te = best_val_loss = np.inf
    es_tol = 1e-6
    for epoch in range(epochs):        
        #train
        for batch in range(batchs_per_epoch):
            data_batch = next(loader_tr)
            train_step(*data_batch)
        #test on validation dataset
        loss_val, _= evaluate(loader_va)
        # earlystoping with patience
        if loss_val + es_tol < best_val_loss:
            best_val_loss = loss_val
            loss_te, acc_te = evaluate(loader_te)
        else:
            wait += 1
        if wait >= p:
            epoch_run = epoch + 1
            break
    '''
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[CategoricalAccuracy(name="acc"),])
    #sets earlystoping monitor
    monitor = EarlyStopping(patience=p, restore_best_weights=True, monitor="val_loss", mode="min")
    #trains and evaluates
    model.fit(
            loader_tr,
            steps_per_epoch=loader_tr.steps_per_epoch,
            epochs=epochs,
            validation_data=loader_va,
            validation_steps=loader_va.steps_per_epoch,
            callbacks=[monitor],
            verbose=0,
    )
    loss, acc = model.evaluate(loader_te, steps=loader_te.steps_per_epoch)
    epoch_run = monitor.stopped_epoch
    '''
    clear_session()
    return acc_te, loss_te, epoch_run