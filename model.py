from re import S
import tensorflow as tf

from spektral.layers.convolutional import GCNConv 
from smoothpool import SmoothPool
from spektral.layers.pooling import (
    global_pool,
    LaPool,
    DiffPool,
    TopKPool,
    SAGPool,
)
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
)

def get_hpool(h_pool, k, hidden, connectivity_augment=None, use_edge_features=False):
    if h_pool == "lapool":
        h_pool = LaPool()
    if h_pool == "diffpool" and k is not None:
        h_pool = DiffPool(k, hidden, activation=None)
    if h_pool == "topkpool" and k is not None:
        h_pool = TopKPool(k)
    if h_pool == "smoothpool" and k is not None:
        mlp = MLP(1, hidden=hidden, layers=3, final_activation="sigmoid")
        h_pool = SmoothPool(k, mlp=mlp, connectivity_augment=connectivity_augment, use_edge_features=use_edge_features)
    if h_pool == "sagpool" and k is not None:
        h_pool = SAGPool(k)

    return h_pool

class HpoolGNN(Model):

    def __init__(
        self,
        output,
        activation=None,
        hidden=256,
        preprocess_layers=3,
        post_layers=2,
        edge_preprocess_layers=2, 
        hidden_activation="tanh",
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
            "preprocess_layers": preprocess_layers,
            "post_layers": post_layers,
            "hidden_activation": hidden_activation,
            "h_pool": h_pool,
            "k": k,
            "connectivity_augmen": connectivity_augment,
            "use_edge_features": use_edge_features,
            "edge_preprocess_layers": edge_preprocess_layers
        }

        self.use_edge_features=use_edge_features

        self.pre_nodes = MLP(
            hidden,
            hidden,
            preprocess_layers,
            activation=hidden_activation,
            final_activation=hidden_activation,
        )

        self.k = k
        self.batch_mode = False
        self.pool_method = h_pool

        self.conv1 = GCNConv(hidden, hidden_activation)
        self.conv2 = GCNConv(hidden, hidden_activation)
        self.pool1 = get_hpool(self.pool_method, self.k, hidden, connectivity_augment=connectivity_augment, use_edge_features=use_edge_features)
        self.conv3 = GCNConv(hidden, hidden_activation)
        self.conv4 = GCNConv(hidden, hidden_activation)
        self.pool2 = get_hpool(self.pool_method, self.k, hidden, connectivity_augment=connectivity_augment, use_edge_features=use_edge_features)
        self.conv5 = GCNConv(hidden, hidden_activation)

        self.readout = Readout()

        self.post_mlp = MLP(
            output,
            hidden,
            post_layers,
            activation=hidden_activation,
            final_activation=activation,
        )

        if self.use_edge_features:
            self.pre_edges = MLP(
                hidden,
                hidden,
                edge_preprocess_layers,
                activation=hidden_activation,
                final_activation=hidden_activation,
            )

    def call(self, inputs):
        
        x, a, e, i = self._get_inputs(inputs)
            
        # Pre-process
        x = self.pre_nodes(x)
        if self.use_edge_features:
            e = self.pre_edge(e)

        z = self.conv1([x, a])
        z = self.conv2([z, a])

        if self.batch_mode:
            x1, a = self.pool1([z,a])
            x1_out = self.readout(x1)
        elif e is not None and self.pool_method == "smoothpool":
                    x1, a, i = self.pool1([z, a, e, i])
                    x1_out = self.readout([x1, i])
        else:
            x1, a, i = self.pool1([z, a, i])
            x1_out = self.readout([x1, i])
        
        z = self.conv3([x1, a])
        z = self.conv4([z, a])
        
        if self.batch_mode:
            x2, a = self.pool1([z,a])
            x2_out = self.readout(x2)
        elif e is not None and self.pool_method == "smoothpool":
                    x2, a, i = self.pool2([z, a, e, i])
                    x2_out = self.readout([x2, i])
        else:
            x2, a, i = self.pool2([z, a, i])
            x2_out = self.readout([x2, i])
        
        x3 = self.conv5([x2, a])    
        
        if self.batch_mode:
            x3_out = self.readout(x3)
        else:
            x3_out = self.readout([x3, i])
      

        x_out = x1_out + x2_out + x3_out
        
        out = self.post_mlp(x_out)
        return out

    def _get_inputs(self, inputs):
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
        
        self.batch_mode = batch_mode
        return x, a, e, i


    def get_config(self):
        return self.config


class Readout(Model):
    def __init__(self) -> None:
        super().__init__()
        self.global_mean = global_pool.get("avg")()
        self.global_sum  = global_pool.get("sum")()

    def call(self, inputs):
        x_mean = self.global_mean(inputs)
        x_sum = self.global_sum(inputs)
        x_readout = tf.concat([x_mean, x_sum], -1)
        return x_readout

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
