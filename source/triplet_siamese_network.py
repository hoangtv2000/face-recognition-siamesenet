import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, BatchNormalization, Input, Conv2D, Dense, Dropout, Lambda, GlobalMaxPooling2D, MaxPooling2D
from keras_vggface.vggface import VGGFace
from . import config
from tensorflow.keras import backend as K
from source import utils, triplet_loss
from .triplet_loss import TripletLossLayer


def triplet_siamese_network(input_shape, embedding_dim=config.EMBEDDING_DIM, continue_training=None):
    if continue_training == False:
        # base_model = ResNet50(weights='imagenet', include_top=False, pooling='max')
        base_model = VGGFace(model='resnet50', include_top=False, pooling='max')
        for layer in base_model.layers:
            layer.trainable = False
        x = base_model.get_layer('global_max_pooling2d').output # layer <= 120
        x = BatchNormalization()(x)
        x = Dropout(0.7)(x)
        x = Dense(embedding_dim*2, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.7)(x)
        x = Dense(embedding_dim, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Lambda(lambda  x: K.l2_normalize(x, axis=1))(x)

        embedding_model = Model(base_model.input, x, name="embedding")
    else:
        embedding_model = tf.keras.models.load_model(config.CONTINUE_TRAINING_DIR)

    anchor_input = Input(input_shape, name='anchor_input')
    positive_input = Input(input_shape, name='positive_input')
    negative_input = Input(input_shape, name='negative_input')

    anchor_embedding = embedding_model(anchor_input)
    positive_embedding = embedding_model(positive_input)
    negative_embedding = embedding_model(negative_input)

    inputs = [anchor_input, negative_input, positive_input]
    outputs = [anchor_embedding, negative_embedding, positive_embedding]

    outputs = TripletLossLayer(margin=config.MARGIN, name='triplet_loss')([anchor_embedding, negative_embedding, positive_embedding])
    triplet_model = Model(inputs, outputs, name='triplet_model')

    # triplet_model.add_loss((triplet_loss.triplet_loss_tensor(outputs)))

    for layer in embedding_model.layers[146:]:
            layer.trainable = True
    for layer in embedding_model.layers[:146]:
            layer.trainable = False

    return embedding_model, triplet_model
