from re import A
from spektral.layers.pooling import TopKPool
from spektral.layers import MessagePassing
from spektral.layers.ops import scatter_mean, scatter_sum, matrix_power
from tensorflow.keras import backend as K
import tensorflow as tf

class SmoothPool(TopKPool):
    def __init__(
        self,
        ratio,
        mlp=None,
        connectivity_augment=None,
        use_edge_features=False,
        return_selection=False,
        return_score=False,
        sigmoid_gating=False,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        **kwargs
    ):
        super().__init__(
            ratio,
            return_selection=return_selection,
            return_score=return_score,
            sigmoid_gating=sigmoid_gating,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            kernel_constraint=kernel_constraint,
            **kwargs
        )
        self.smoothmp = SmoothMP(mlp, use_edge_features=use_edge_features)
        self.connectivity_augment = connectivity_augment

    def call(self, inputs, **kwargs):
        x, a, e, i = self.get_inputs(inputs)
        smoothness = self.smoothmp([x, a, e])
        output = self.pool(x, a, i, y=smoothness)
        if self.return_score:
            output.append(smoothness)
        return output
    
    def connect(self, a, s, **kwargs):
        if self.connectivity_augment is not None:
            assert isinstance(self.connectivity_augment, int), "Invalid connectivity augment parameter!"
            a = matrix_power(a, self.connectivity_augment)
            a_values = tf.clip_by_value(a.values, clip_value_min=0, clip_value_max=1)
            a = tf.SparseTensor(indices=a.indices, values=a_values, dense_shape=a.dense_shape)
        return super().connect(a, s, **kwargs)

    def reduce(self, x, s, y=None):
        x_pool = tf.gather(x * y, s)
        return x_pool
    
    def get_inputs(self, inputs):
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
        super(TopKPool, self).build(input_shape)

class SmoothMP(MessagePassing):
    def __init__(self, mlp, use_edge_features=False, **kwargs):
        super().__init__(**kwargs)
        self.mlp = mlp
        self.use_edge_features = use_edge_features
    
    def message(self, x, **kwargs):
        diff = self.get_sources(x)-self.get_targets(x)
        return diff
    
    def aggregate(self, messages, e=None, **kwargs):
        diff = scatter_mean(messages, self.index_targets, self.n_nodes)
        if e is not None and self.use_edge_features:
            divergence = scatter_mean(e, self.index_targets, self.n_nodes)
            return tf.concat([diff, divergence], -1)
        else:
            return diff
    
    def call(self, inputs, **kwargs):
        x, a, e = self.get_inputs(inputs)
        diff_and_dive = self.propagate(x, a, e)
        z = tf.concat([x, diff_and_dive], -1)
        z = self.mlp(z)
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
    print(tf.sparse.to_dense(A))
    mlp = lambda x:x 
    smoothmp = SmoothMP(mlp=mlp)
    z = smoothmp([X,A])
    print(z)