# from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
# from keras_vggface.vggface import VGGFace
# from . import config
from tensorflow.keras import backend as K
# from source import utils


class TripletLossLayer(Layer):
    def __init__(self, margin, name=None, **kwargs):
        super(TripletLossLayer, self).__init__(name=name)
        self.margin = margin
        super(TripletLossLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super(Layer, self).get_config()
        config.update({"margin": self.margin})
        return config

    def triplet_loss(self, inputs):
        anchor, negative, positive = inputs
        anchor = anchor/K.sqrt(K.maximum(K.sum(K.square(anchor),axis=1,keepdims=True),1e-10))
        negative = negative/K.sqrt(K.maximum(K.sum(K.square(negative),axis=1,keepdims=True),1e-10))
        positive = positive/K.sqrt(K.maximum(K.sum(K.square(positive),axis=1,keepdims=True),1e-10))

        p_dist = K.sqrt(K.sum(K.square(anchor-positive), axis=1))
        n_dist = K.sqrt(K.sum(K.square(anchor-negative), axis=1))

        return K.sum(K.maximum(p_dist - n_dist + self.margin, 0))

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss