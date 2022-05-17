from numpy import dtype
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Dense, Dropout, Lambda, BatchNormalization, AveragePooling2D
from keras_vggface.vggface import VGGFace
from . import config
from tensorflow.keras import backend as K
from source import utils

from tensorflow import keras
from tensorflow.keras import layers



def face_network(input_shape, load_model, embedding_dim=config.EMBEDDING_DIM):
    # base_model = ResNet50(weights='imagenet', include_top=False, pooling='max')
    if load_model == False:
        inputs = keras.Input(shape=input_shape, dtype=tf.float32)
        transformed_inputs = layers.Rescaling(1./255)(inputs)
            
        base_model = VGGFace(model='vgg16', include_top=False, pooling='max', weights = 'vggface', input_tensor=transformed_inputs)

        for layer in base_model.layers:
            layer.trainable = False
            
        x = base_model.get_layer('global_max_pooling2d').output # layer <= 120
        x = BatchNormalization()(x)
        x = Dropout(0.7)(x)
        x = Dense(embedding_dim*2, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.7)(x)
        x = Dense(embedding_dim, activation='relu',\
                     kernel_regularizer=keras.regularizers.l2(1e-1))(x)
        x = BatchNormalization()(x)
        last_x = Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(x)

        embedding_model = Model(inputs=inputs, outputs=last_x, name='embedding_model')

    else: 
        embedding_model = tf.keras.models.load_model(config.SAVED_MODEL_PATH, custom_objects={"K": K})
        last_x = embedding_model.output

    return embedding_model, last_x