""" Implementation of the TCN model. """

import tensorflow as tf
from tensorflow.keras import layers as L


class TCNBlock(L.Layer):
    """Temporal Convolution Network (TCN) block."""
    def __init__(self, units, kernel_size=3, dilation=1, dropout_rate=0.2, name='tcn_block'):
        super().__init__(name=name)

        self.units = units
        self.block = tf.keras.Sequential([
            L.Conv1D(units, kernel_size, padding='causal', dilation_rate=dilation),
            L.BatchNormalization(),
            L.ReLU(),
            L.Dropout(rate=dropout_rate),
            # Repeat again
            L.Conv1D(units, kernel_size, padding='causal', dilation_rate=dilation),
            L.BatchNormalization(),
            L.ReLU(),
            L.Dropout(rate=dropout_rate)
        ], name='tcn_block')

        self.res_conv = None

    def build(self, input_shape):
        if input_shape[-1] == self.units:
            self.res_conv = L.Lambda(lambda x: x)
        else:
            # Add conv layer to residual connection if shapes missmatch
            self.res_conv = L.Conv1D(self.units, 1, activation='relu')

    def call(self, inputs, training=None):
        trans = self.block(inputs, training=training)
        res = self.res_conv(inputs)

        return res + trans

class TCN(tf.keras.Model):
    """Temporal Convolution Network (TCN) composed of several TCN blocks.

    Args:
        units_list (list): list of integers, one per TCN block, specifying the number of units for the block
        kernel_size (int, optional): convolution kernel size. Defaults to 3.
        dropout_rate (float, optional): dropout rate. Defaults to 0.2.
        return_sequences (bool, optional): Whether to return the last output in the output sequence, or the full sequence. Defaults to False.
    """
    def __init__(self, units_list, kernel_size=3, dropout_rate=0.2, return_sequences=False, name='tcn'):
        super().__init__(name=name)

        self.return_sequences = return_sequences
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

        # Create TCN blocks. For each additional block the dilation is doubled
        layers = []
        for i, units in enumerate(units_list):
            dilation = 2 ** i
            block = TCNBlock(units, kernel_size=self.kernel_size,
                             dilation=dilation, dropout_rate=self.dropout_rate)
            layers.append(block)

        self.net = tf.keras.Sequential(layers, name='tcn')

    def call(self, inputs, training=None):
        out = self.net(inputs, training=training)

        if not self.return_sequences:
            out = out[:,-1,:]

        return out
