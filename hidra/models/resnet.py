import tensorflow as tf
from tensorflow.keras import layers

from .layers import AverageReduce2D

class ResNetLayer(layers.Layer):
    """ A building block of ResNet containing convolution, normalization and activation. """

    def __init__(self,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=layers.BatchNormalization,
                 conv_first=True):

        """2D Convolution-Batch Normalization-Activation

        # Arguments
            num_filters (int): Conv2D number of filters
            kernel_size (int): Conv2D square kernel dimensions
            strides (int): Conv2D square stride dimensions
            activation (string): activation name
            batch_normalization (callable): callable that creates a normalization layer.
                                            If None, will be skipped.
            conv_first (bool): conv-bn-activation (True) or
                bn-activation-conv (False)

        """
        super().__init__()

        operations = []
        conv = layers.Conv2D(num_filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))


        if batch_normalization is not None:
            operations.append(batch_normalization())

        if activation is not None:
            operations.append(layers.Activation(activation))

        if conv_first:
            operations.insert(0, conv)
        else:
            operations.append(conv)


        self.resnet_layer = tf.keras.Sequential(operations)

    def call(self, x, training=None):
        return self.resnet_layer(x, training=training)

class ResNet_v2(tf.keras.Model):
    """ ResNet_v2 base (without classification head)

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    """

    def __init__(self, num_res_blocks, reduce_fn=AverageReduce2D(), layer_normalization=layers.BatchNormalization):
        """ Init ResNet_v2.

        # Arguments
            num_res_blocks (int): number of residual blocks
            reduce_fn (callable): reduction method used to reduce output dimensions
        """
        super().__init__()

        self.num_res_blocks = num_res_blocks
        self.reduce_fn = reduce_fn
        self.layer_normalization = layer_normalization

    def build(self, input_shape):

        # Input placeholder
        inputs = tf.keras.layers.Input(input_shape[1:])

        # Start model definition.
        num_filters_in = 16
        num_res_blocks = self.num_res_blocks

        # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
        x = ResNetLayer(num_filters=num_filters_in,
                        conv_first=True,
                        batch_normalization=self.layer_normalization)(inputs)

        # Instantiate the stack of residual units
        for stage in range(3):
            for res_block in range(self.num_res_blocks):
                activation = 'relu'
                batch_normalization = self.layer_normalization
                strides = 1
                if stage == 0:
                    num_filters_out = num_filters_in * 4
                    if res_block == 0:  # first layer and first stage
                        activation = None
                        batch_normalization = None
                else:
                    num_filters_out = num_filters_in * 2
                    if res_block == 0:  # first layer but not first stage
                        strides = 2    # downsample

                # bottleneck residual unit
                y = ResNetLayer(num_filters=num_filters_in,
                                kernel_size=1,
                                strides=strides,
                                activation=activation,
                                batch_normalization=batch_normalization,
                                conv_first=False)(x)

                y = ResNetLayer(num_filters=num_filters_in,
                                conv_first=False)(y)

                y = ResNetLayer(num_filters=num_filters_out,
                                kernel_size=1,
                                conv_first=False)(y)

                if res_block == 0:
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = ResNetLayer(num_filters=num_filters_out,
                                    kernel_size=1,
                                    strides=strides,
                                    activation=None,
                                    batch_normalization=None)(x)

                x = layers.Add()([x, y])

            num_filters_in = num_filters_out

        # Add classifier on top.
        # v2 has BN-ReLU before Pooling
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        # x = layers.AveragePooling2D(pool_size=8)(x)
        # x = layers.Flatten()(x)

        # Spatial reduction
        if self.reduce_fn is not None:
            x = self.reduce_fn(x)

        # Instantiate model.
        self.model = tf.keras.Model(inputs=inputs, outputs=x)

    def call(self, x, training=None):
        return self.model(x, training=training)
