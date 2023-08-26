import tensorflow as tf 
import numpy as np 

class ConvUnit(keras.layers.Layer):
    def __init__(
            self,
            filters,
            activation,
            normalization,
            convolution,
            kernel=3,
            name="conv3d",
            **kwargs
    ):
        super(ConvUnit, self).__init__(name=name, **kwargs)

        self.filters = filters

        if activation == 'PReLU':
            self.activation = tf.keras.layers.PReLU()
        else:
            self.activation = tf.keras.layers.Activation(activation)
        self.normalization = normalization
        self.convolution = convolution
        self.kernel = kernel

        self.conv_layer = self.convolution(self.filters, self.kernel, padding='same')
        self.norm_layer = self.normalization() if self.normalization is not None else None

    def call(self, inputs, **kwargs):
        # CONV_NORM_ACTIVATION
        x = self.conv_layer(inputs, **kwargs)
        if self.norm_layer is not None:
            x = self.norm_layer(x, **kwargs)
        x = self.activation(x, **kwargs)
        return x