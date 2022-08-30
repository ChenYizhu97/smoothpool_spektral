import tensorflow as tf

from spektral.layers.convolutional import GCNConv 
from smoothpool import SmoothPool
from spektral.layers.pooling import (
    global_pool,
    LaPool,
    DiffPool,
    TopKPool,
    SAGPool
)
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
)


class HpoolGNN(Model):

    def __init__(
        self,
        output,
        activation=None,
        hidden=256,
        message_passing=5,
        pre_process=3,
        post_process=2,
        edge_process=2, 
        hidden_activation="tanh",
        gpool="sum",
        i_hpool=None,
        h_pool=None,
        k=None,
        connectivity_augment=None,
        use_edge_features=False
    ):
        super().__init__()
        self.config = {
            "output": output,
            "activation": activation,
            "hidden": hidden,
            "message_passing": message_passing,
            "pre_process": pre_process,
            "post_process": post_process,
            "hidden_activation": hidden_activation,
            "gpool": gpool,
            "i_hpool": i_hpool,
            "h_pool": h_pool,
            "k": k,
            "connectivity_augmen": connectivity_augment,
            "use_edge_features": use_edge_features,
            "edge_process": edge_process
        }

        # Global pooling
        if gpool is not None:
            self.gpool = global_pool.get(gpool)()
        else:
            self.gpool = None

        self.use_edge_features=use_edge_features
        self.gnn = [
            GCNConv(hidden, hidden_activation)
            for _ in range(message_passing)
        ]
         # Neural blocks
        self.pre = MLP(
            hidden,
            hidden,
            pre_process,
            activation=hidden_activation,
            final_activation=hidden_activation,
        )
        self.i_hpool = i_hpool
        
        self.k = k

        self.pool_method = h_pool
        if h_pool == "lapool":
            self.h_pool = LaPool()
        if h_pool == "diffpool" and k is not None:
            self.h_pool = DiffPool(k, hidden, activation=None)
        if h_pool == "topkpool" and k is not None:
            self.h_pool = TopKPool(k)
        if h_pool == "smoothpool" and k is not None:
            mlp = MLP(1, hidden=hidden, layers=3, final_activation="sigmoid")
            self.h_pool = SmoothPool(k, mlp=mlp, connectivity_augment=connectivity_augment, use_edge_features=use_edge_features)
        if h_pool == "sagpool" and k is not None:
            self.h_pool = SAGPool(k)


        self.post = MLP(
            output,
            hidden,
            post_process,
            activation=hidden_activation,
            final_activation=activation,
        )
        if self.use_edge_features:
            self.pre_edge = MLP(
                hidden,
                hidden,
                edge_process,
                activation=hidden_activation,
                final_activation=hidden_activation,
            )

    def call(self, inputs):
        batch_mode = False
        e = None
        if len(inputs) == 2:
            batch_mode = True
            x, a = inputs
            i = None
        elif len(inputs) == 3:
            x, a, i = inputs
            if len(x.shape) == 3:
                batch_mode = True
                e = i
                i = None
        else:
            x, a, e, i = inputs
            
        # Pre-process
        out = self.pre(x)
        if self.use_edge_features:
            e = self.pre_edge(e)
        # Message passing
        i_layer = 0

        for layer in self.gnn:
            i_layer += 1
            out = layer([out, a])
            if self.i_hpool is not None and i_layer == self.i_hpool:
                if batch_mode:
                    out, a = self.h_pool([out, a])
                elif e is not None and self.pool_method == "smoothpool":
                    out, a, i = self.h_pool([out, a, e, i])
                else:
                    out, a, i = self.h_pool([out, a, i])
        # Global pooling
        if self.gpool is not None:
            if batch_mode:
                out = self.gpool(x)
            else:
                out = self.gpool([out] + ([i] if i is not None else []))
        # Post-process
        out = self.post(out)

        return out

    def get_config(self):
        return self.config


class MLP(Model):
    def __init__(
        self,
        output,
        hidden=256,
        layers=2,
        batch_norm=True,
        dropout=0.0,
        activation="tanh",
        final_activation=None,
    ):
        super().__init__()
        self.config = {
            "output": output,
            "hidden": hidden,
            "layers": layers,
            "batch_norm": batch_norm,
            "dropout": dropout,
            "activation": activation,
            "final_activation": final_activation,
        }
        self.batch_norm = batch_norm
        self.dropout_rate = dropout

        self.mlp = Sequential()
        for i in range(layers):
            # Linear
            self.mlp.add(Dense(hidden if i < layers - 1 else output))
            # Batch norm
            if self.batch_norm:
                self.mlp.add(BatchNormalization())
            # Dropout
            self.mlp.add(Dropout(self.dropout_rate))
            # Activation
            self.mlp.add(Activation(activation if i < layers - 1 else final_activation))

    def call(self, inputs):
        return self.mlp(inputs)

    def get_config(self):
        return self.config
