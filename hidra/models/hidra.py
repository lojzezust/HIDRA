""" HIDRA models. """

import tensorflow as tf
import tensorflow.keras.layers as L

from .resnet import ResNet_v2
from .layers import SpatialEncoding, TimeInvariant, FlattenSpatial, TemporalLinearCombination, LinearCombination, LSTMStack
from .tcn import TCN
from .regression import RegressionNetwork

class HIDRABase(tf.keras.Model):
    """Customizable base of the HIDRA architecture.

    Args:
        weather_cnn (tf.keras.layers.Layer): Spatial atmospheric encoder module.
        weather_pr (tf.keras.layers.Layer): Atmospheric data temporal encoder module.
        ssh_pr (tf.keras.layers.Layer): Tidal data temporal encoder module.
        regression (tf.keras.layers.Layer): Residual regression module.
    """

    def __init__(self, weather_cnn, weather_pr, ssh_pr, regression, name='HIDRA'):
        super().__init__(name=name)

        self.w_spatial = weather_cnn

        self.w_process = weather_pr
        self.ssh_process = ssh_pr

        self.regression = regression

    def call(self, inputs, training=None):
        w, ssh = inputs

        # Weather CNN
        weather_features_t = self.w_spatial(w, training=training)
        height_features_t = ssh

        # Process features
        weather_features = self.w_process(weather_features_t)
        height_features = self.ssh_process(height_features_t)

        # Concatenate
        combined_features = tf.concat([weather_features, height_features], axis=1)

        # Regression
        predicted_h = self.regression(combined_features, training=training)

        return predicted_h

def HIDRA(temporal_encoders='HIDRA', probabilistic=True, num_predictions=72, name='HIDRA'):
    """Build the HIDRA model.

    Args:
        temporal_encoders (str, optional): Which temporal encoders to use (One of: 'HIDRA', 'LSTM', 'TCN'). Defaults to 'HIDRA'.
        probabilistic (bool, optional): Model outputs as probability distributions. Defaults to True.
        num_predictions (int, optional): Number of predicted times.
    """
    # Time invariant atmospheric spatial encoder
    weather_cnn = tf.keras.Sequential([
        SpatialEncoding(),
        ResNet_v2(num_res_blocks=2, reduce_fn=L.AveragePooling2D((2,2))),
    ])

    # Add spatial attention and ReLU
    weather_cnn_full = tf.keras.Sequential([
        TimeInvariant(weather_cnn),
        FlattenSpatial(),
        TemporalLinearCombination(combination_axis=2, temporal_axis=1),
        L.ReLU()
    ])

    # Regression network
    regression = RegressionNetwork(
        num_predictions=num_predictions,
        units=[256, 256, 256],
        dropout_rate=0.5,
        probabilistic=probabilistic)

    # Temporal encoders
    if temporal_encoders == 'HIDRA':
        weather_pr = LinearCombination(axis=1)
        ssh_pr = L.Flatten()
    elif temporal_encoders == 'LSTM':
        weather_pr = LSTMStack([128,128,128])
        ssh_pr = LSTMStack([32,32,32])
    elif temporal_encoders == 'TCN':
        weather_pr = TCN([128,128,128])
        ssh_pr = TCN([32,32,32])

    model = HIDRABase(weather_cnn_full, weather_pr, ssh_pr, regression, name=name)
    return model

def compile_model(model):
    """Prepare model for training."""
    negloglik = lambda y_t, y_p: -y_p.log_prob(y_t)

    model.compile(loss=negloglik, metrics=['mean_absolute_error'], optimizer='adam')

    return model

def add_inference_head(model):
    """Add inference head (outputs mean and std) to the model for a probabilistic HIDRA model."""

    inference_layer = L.Lambda(lambda x: tf.stack([x.mean(), x.stddev()], axis=-1))

    model = tf.keras.Sequential([
        model,
        inference_layer
    ])

    return model
