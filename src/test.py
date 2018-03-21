import keras.backend as K
from keras.engine.topology import Layer
from keras import initializations
from keras import regularizers
from keras import constraints
import numpy as np
import theano.tensor as T
from keras.layers import Input, Dense
from keras.models import Model


class Average(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Average, self).__init__(**kwargs)

    def call(self, x, mask=None):
        print("average1")
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask)
            x = x * mask
            print("average2")
        print("average3")
        return K.sum(x, axis=-2) / K.sum(mask, axis=-2)


    def get_output_shape_for(self, input_shape):
        return input_shape[0:-2] + input_shape[-1:]

    def compute_mask(self, x, mask=None):
        return None



inputs = Input(shape=(784,10,10))


x = Dense(64, activation='relu')(inputs)

y_z = Average()(inputs)