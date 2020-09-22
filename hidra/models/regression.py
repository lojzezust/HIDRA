import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers as L

from .layers import DenseBlock

def RegressionNetwork(num_predictions, units=[192], dropout_rate=0.5, probabilistic=False, name='regression'):
    """Creates a regression network. To be used after feature extraction.

    num_predictions (int): Number of predictions (= number of units in the last layer)
    units (List[int]): List with number of units for each layer.
    dropout_rate (float): The dropout rate used.
    probabilistic (bool): Use probabilistic regression (mean and std).
    """

    # Hidden layers
    regression_layers = [DenseBlock(num_u, activation='relu') for num_u in units]

    # Add dropout and predictions layer
    out_units = num_predictions
    if probabilistic:
        out_units = num_predictions * 2

    regression_layers += [
        L.Dropout(dropout_rate),
        DenseBlock(out_units)
    ]

    # Optional probability distribution layer
    if probabilistic:
        dist_layer = tfp.layers.DistributionLambda(
            lambda x: tfp.distributions.Normal(
                loc=x[..., :num_predictions],
                scale=1e-6 + tf.math.softplus(0.05 * x[..., num_predictions:])))

        regression_layers.append(dist_layer)

    return tf.keras.Sequential(regression_layers, name=name)
