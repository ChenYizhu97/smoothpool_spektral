"""
This module provides our custom pooling layer.
"""
import tensorflow as tf
from spektral import layers
from spektral.layers import pooling, ops
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense

class SmoothPool(pooling.TopKPool):
    """Pooling layer that performs local pooling on input graph, resulting a coarsened graph.

    Usage:
    smoothpool = Smoothpool(k) #k is the ratio of pooling.
    x, a, e, i = smoothpool(x, a, e, i) 
    #x, a, e and i represents node feature, adjancency matrix, edge feature and indices repectively.
    

    Arguments:
        ratio: Ratio of pooling. A float between 0 and 1.
        hidden: Hidden unit of the trainable vectors. An Int.
        connectivity_augment: Value of connectivity augmentaion. Nore or an int.
        use_edge_features: Whether to use edge features. A boolean.
        return_selection: Whether to return the selection mask. A boolean.
        return_score: Whether to return the node scoring vector. A boolean.
    """
    def __init__(
        self,
        ratio,
        hidden=3,
        connectivity_augment=None,
        use_edge_features=False,
        return_selection=False,
        return_score=False,
        **kwargs
    ):
        super().__init__(
            ratio,
            return_selection=return_selection,
            return_score=return_score,
            **kwargs
        )

        self.hidden = hidden
        self.connectivity_augment = connectivity_augment
        self.use_edge_features = use_edge_features
    
        self.dense_node_features = Dense(self.hidden, activation="tanh")
        self.dense_diff = Dense(self.hidden, activation="tanh")
        self.dense_score = Dense(1, activation="sigmoid")
        if self.use_edge_features is True:
            self.dense_div = Dense(self.hidden, activation="tanh")
        else:
            self.dense_div = None
        self.smoothmp = SmoothMP(self.dense_node_features, self.dense_diff, self.dense_score, self.dense_div)
  
    def call(self, inputs:list, **kwargs) -> list:
        """Receives input graph and produces coarsened graph
        
        Args:
            inputs: list of value that represents (a batch of) graphs. A list.
        
        returns:
            outputs: list of value that represents (a batch of) coarsened graphs. A list.
        """
        x, a, e, i = self.get_inputs(inputs)

        if self.use_edge_features is False:
            e = None
        smoothness = self.smoothmp([x, a, e])
        outputs = self.pool(x, a, i, y=smoothness)       
        x, a, i = outputs
        e = None

        outputs = [x, a, e, i]

        if self.return_score:
            outputs.append(smoothness)

        return outputs
    
    def connect(self, a, s, **kwargs):
        """See pooling.SRCpool.connect"""
        # Augment graph connectivity with given value.
        if self.connectivity_augment is not None:      
            if not isinstance(self.connectivity_augment, int):
                raise ValueError("Invalid connectivity augment value!")      
            a = ops.matrix_power(a, self.connectivity_augment)
            a_values = tf.clip_by_value(a.values, clip_value_min=0, clip_value_max=1)
            a = tf.SparseTensor(indices=a.indices, values=a_values, dense_shape=a.dense_shape)

        return super().connect(a, s, **kwargs)

    def reduce(self, x, s, y=None):
        """See pooling.SRCpool.reduce"""
        x_pool = tf.gather(x * y, s)

        return x_pool
    
    def get_inputs(self, inputs):
        """Parses x, a, e and i from given inputs"""
        if len(inputs) == 3:
            x, a, i = inputs
            if K.ndim(i) == 2:
                i = i[:, 0]
            assert K.ndim(i) == 1, "i must have rank 1"
            e = None
        elif len(inputs) == 4:
            x, a, e, i = inputs
        else:
            raise ValueError(
                "Expected 3 or 4 inputs tensors (x, a, e, i), got {}.".format(len(inputs))
            )

        self.n_nodes = tf.shape(x)[-2]

        return x, a, e, i

    def build(self, input_shape):
        self.n_nodes = input_shape[0][0]
        super(pooling.TopKPool, self).build(input_shape)

class SmoothMP(layers.MessagePassing):
    """Message passing layer that produces features for local pooling. See layers.MessagePassing.
    
    Arguments:
        dense_node_features: Trainable vector that transforms node features. A dense layer.
        dense_diff: Trainbale vector that transforms differences. A dense layer.
        dense_score: Trainbale vector that produces rank scores. A dense layer.
        dense_dive: Trainbale vector that transforms divergences. A dense layer or None.
    """
    def __init__(self, dense_node_features, dense_diff, dense_score, dense_div=None, **kwargs):
        super().__init__(**kwargs)
        self.dense_node_features = dense_node_features
        self.dense_diff = dense_diff
        self.dense_score = dense_score
        self.dense_div = dense_div
    
    def message(self, x, **kwargs):
        diff = self.get_sources(x)-self.get_targets(x)
        return diff
    
    def aggregate(self, messages, e=None, **kwargs):
        diff_agg = ops.scatter_mean(messages, self.index_targets, self.n_nodes)
        diff_agg = self.dense_diff(diff_agg)
        if e is not None and self.dense_div is not None:
            divergence = ops.scatter_mean(e, self.index_targets, self.n_nodes)
            divergence = self.dense_div(divergence)
            return tf.concat([diff_agg, divergence], -1)
        else:
            return diff_agg
    
    def call(self, inputs, **kwargs):
        x, a, e = self.get_inputs(inputs)
        diff_and_dive = self.propagate(x, a, e)
        x = self.dense_node_features(x)
        z = tf.concat([x, diff_and_dive], -1)
        z = self.dense_score(z)
        return z

    @staticmethod
    def get_inputs(inputs):
        """
        Parses the inputs lists and returns a tuple (x, a, e) with node features,
        adjacency matrix and edge features. In the inputs only contain x and a, then
        e=None is returned.
        """
        if len(inputs) == 3:
            x, a, e = inputs
            if e is not None:
                assert K.ndim(e) in (2, 3), "E must have rank 2 or 3"
        elif len(inputs) == 2:
            x, a = inputs
            e = None
        else:
            raise ValueError(
                "Expected 2 or 3 inputs tensors (X, A, E), got {}.".format(len(inputs))
            )
        assert K.ndim(x) in (2, 3), "X must have rank 2 or 3"
        assert K.is_sparse(a), "A must be a SparseTensor"
        assert K.ndim(a) == 2, "A must have rank 2"

        return x, a, e


if __name__ == "__main__":
    X = tf.convert_to_tensor([[1,2,3,4],
                              [1,3,2,4],
                              [3,1,4,2],
                              [4,1,2,3]],
                              dtype= tf.float32
                            )
    A = tf.SparseTensor(indices=[[0,1],[0,2],[1,0],[1,3],[2,0],[2,3],[3,1],[3,2],],
                        values=[1,1,1,1,1,1,1,1],
                        dense_shape=[4,4])
    E = None
    I = tf.ones((X.shape[0],), tf.int32)
    print(I)
    """
    # test message passing.
    print(tf.sparse.to_dense(A))
    mlp = lambda x:x 
    smoothmp = SmoothMP(mlp=mlp)
    z = smoothmp([X,A])
    print(z)
    """
    # test input shape
    mlp = lambda x:x
    smoothpool = SmoothPool(0.5,mlp)
    x, a, e, i = smoothpool([X,A,E,I])
    