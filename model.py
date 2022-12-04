"""
This module provides the hierarchical GNN model for testing pooling layers.
"""
import tensorflow as tf
from spektral.layers.convolutional import GCNConv 
from spektral.layers.pooling import (
    global_pool,
)
from spektral.models.general_gnn import MLP
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Concatenate,
)


class BasicModel(Model):
    """Basic GNN model.

    Arguments:
        output: Dimension of output features. An int.
        activation: Activation function for output features. A string.
        hidden: Width of hidden learning layers. An int.
        preprocess_layers: Number of linear layers that transforms node features before feeding them to GNNs. An int.
        post_layers: Number of linear layers that transforms features learnt by GNNS. An int.
        edge_preprocess_layers: Number of linear layers that transforms edge features before feeding them to GNNs. An int.
        hidden_activation: Activation function for hidden features. A string.
        use_edge_features: Whether to use edge features. A boolean.
    """
    def __init__(
        self,
        output,
        pool = lambda x:x,
        activation=None,
        hidden=256,
        preprocess_layers=3,
        post_layers=2,
        edge_preprocess_layers=2, 
        hidden_activation="tanh",
        use_edge_features=False,
    ):
        super().__init__()
        self.use_edge_features = use_edge_features
        #linear layers that pre-process node features.
        self.pre_nodes = MLP(
            hidden,
            hidden,
            preprocess_layers,
            activation=hidden_activation,
            final_activation=hidden_activation,
        )
        self.pool = pool
        #convolutional GNNs.
        self.conv1 = GCNConv(hidden, hidden_activation)
        self.conv2 = GCNConv(hidden, hidden_activation)
        #Reduces node features to graph representation.
        self.readout = Readout()
        #skip link
        self.skip = Concatenate()
        #Linear layers that post-process learnt graph representations.
        self.post_mlp = MLP(
            output,
            hidden,
            post_layers,
            activation=hidden_activation,
            final_activation=activation,
        )
        #Linear layers that pre-processes the edge features if they are used.
        if self.use_edge_features:
            self.pre_edges = MLP(
                hidden,
                hidden,
                edge_preprocess_layers,
                activation=hidden_activation,
                final_activation=hidden_activation,
            )

    def call(self, inputs):
        """basic model with 2 GNN layers"""
        x, a, e, i = self.pre_process(inputs)
        x = self.skip([self.conv1([x,a]), x])
        #x = self.conv1([x,a])
        inputs = [x, a]

        if e is not None:
            inputs.append(e)
        if i is not None:
            inputs.append(i)
               
        outputs = self.pool(inputs)
        if len(outputs) == 2:
            x, a = outputs
        elif len(outputs) ==3:
            x, a, i = outputs
        else:
            x, a, e, i = outputs

        x = self.skip([self.conv2([x,a]), x])
        #x = self.conv2([x,a])
        if i is not None:
            readout = self.readout([x, i])
        else:
            readout = self.readout(x)
        z = self.post_mlp(readout)
        return z

    def pre_process(self, inputs):
        """Pre-processes input node and edge features."""
        #get inputs
        x, a, e, i = self._get_inputs(inputs)
        #pre-processes node and edge features
        x = self.pre_nodes(x)
        if self.use_edge_features:
            e = self.pre_edges(e)
        else:
            e = None
        return x, a, e, i

    def _get_inputs(self, inputs):
        """Parses x, a, e and i from inputs."""
        e = None
        #batchmode without edge attributes
        if len(inputs) == 2:
            x, a = inputs
            i = None
        elif len(inputs) == 3:
            #disjoint mode without edge attributes.
            x, a, i = inputs
            if len(x.shape) == 3:
                #batch mode with edge attributes.
                e = i
                i = None
        else:
            #disjoint mode with edge attributes.
            x, a, e, i = inputs
        return x, a, e, i

class Readout(Model):
    """A layer that reduces node features of a graph to a single representation."""
    def __init__(self) -> None:
        super().__init__()
        self.global_mean = global_pool.get("avg")()
        self.global_max  = global_pool.get("max")()

    def call(self, inputs):
        x_mean = self.global_mean(inputs)
        x_max = self.global_max(inputs)
        x_readout = tf.concat([x_mean, x_max], -1)
        return x_readout
