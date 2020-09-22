import tensorflow as tf
from tensorflow.keras import layers as L
import numpy as np

class DenseBlock(L.Layer):
    """A Dense layer block containing batch normalization and activation.

    Args:
        units (int): number of hidden units
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
    """

    def __init__(self, units, activation=None, batch_normalization=False, name=None):
        super().__init__(name=name)

        operations = []
        operations.append(L.Dense(units, kernel_initializer='he_normal',
                                       kernel_regularizer=tf.keras.regularizers.l2(1e-4)))


        if batch_normalization:
            operations.append(L.BatchNormalization())
        if activation is not None:
            operations.append(L.Activation(activation))

        self.dense_layer = tf.keras.Sequential(operations)

    def call(self, x, training=None):
        return self.dense_layer(x, training=training)

class AverageReduce2D(L.Layer):
    """Average spatial information into a single value."""

    def call(self, x):
        return tf.reduce_mean(x, axis=(1,2))


class TimeInvariant(L.Layer):
    """Time invariant layer (repeats operation on all time slices).
    Done by merging the batch and time dimension together (new batch size) and reversing it back after the operation.

    Deprecated: can be replaced with `tf.keras.layers.TimeDistributed`
    """

    def __init__(self, layer, name=None):
        super().__init__(name=name)

        self.layer = layer

    def build(self, input_shape):
        self.time_dim = input_shape[1]
        self.features_dims = input_shape[2:]


    def call(self, x, training=None):
        # Merge batch and time dimension into samples dimension
        x = tf.reshape(x, tf.concat([[-1], self.features_dims], axis=0))

        # Process
        x = self.layer(x, training=training)

        # Split samples dimension back into batch and time
        merged_out_shape = tf.shape(x)
        out_shape = tf.concat([[-1, self.time_dim], merged_out_shape[1:]], axis=0)
        x = tf.reshape(x, out_shape)

        return x


class LinearCombination(L.Layer):
    """Aggregation as a linear combination of inputs."""

    def __init__(self, axis=-1, name=None):
        super().__init__(name=name)

        self.axis = axis

    def build(self, input_shape):
        num_units = input_shape[self.axis]

        weight_shape = [1 for ax in input_shape]
        weight_shape[self.axis] = num_units

        self.w = self.add_weight(shape=weight_shape, initializer='random_normal', trainable=True, name='w')

    def call(self, x):
        prod = self.w * x
        lin_comb = tf.math.reduce_sum(prod, axis=self.axis)

        return lin_comb

class TemporalLinearCombination(L.Layer):
    """Time-dependent linear combination. Each temporal slice uses separate combination weights."""

    def __init__(self, combination_axis=-1, temporal_axis=-2, name=None):
        super().__init__(name=name)

        self.combination_axis = combination_axis
        self.temporal_axis = temporal_axis

    def build(self, input_shape):
        weight_shape = [1 for ax in input_shape]
        weight_shape[self.combination_axis] = input_shape[self.combination_axis]
        weight_shape[self.temporal_axis] = input_shape[self.temporal_axis]

        self.w = self.add_weight(shape=weight_shape, initializer='random_normal', trainable=True, name='w')

    def call(self, x):
        prod = self.w * x
        lin_comb = tf.math.reduce_sum(prod, axis=self.combination_axis)

        return lin_comb

def LSTMStack(units, name='lstm_stack'):
    """Build a stack of LSTM layers.

    Args:
        units (List[int]): number of units per layer
        name (str): layer name
    """
    layers = [L.LSTM(num_units, return_sequences=True) for num_units in units[:-1]]
    layers.append(L.LSTM(units[-1], return_sequences=False))

    return tf.keras.Sequential(layers, name=name)


class SpatialEncoding(L.Layer):
    """
    Adds spatial encoding to the input image (4D).
    Coordiantes are encoded as relative x and y coordinates (between 0 and 1).
    """

    def build(self, input_shape):
        _, h, w, _ = input_shape
        xs = np.linspace(0, 1, w)
        ys = np.linspace(0, 1, h)
        x_pos, y_pos = np.meshgrid(xs,ys)
        self.pos_features = np.stack([x_pos, y_pos], axis=-1)[np.newaxis, ...]

    def call(self, x):
        shape = tf.shape(x)
        ones = tf.ones((shape[0], shape[1], shape[2], 1))
        pos_features = ones * self.pos_features

        res = tf.concat([x, pos_features], axis=-1)

        return res

class FlattenSpatial(L.Layer):
    """Flattens the x and y spatial dimensions."""
    def build(self, input_shape):
        self.reshape = L.Reshape((input_shape[1], input_shape[2]*input_shape[3], input_shape[4]))

    def call(self, x):
        res = self.reshape(x)

        return res
