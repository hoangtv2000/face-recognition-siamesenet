import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from source import config, utils, base_network
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input
import os, glob
import json
import numpy as np

from source.cosface_loss import CosFace
from source.arcface_loss import ArcFace
from source.sphereface_loss import SphereFace
from source.adacos_loss import AdaCos

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


# callbacks
early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=20)
callbacks_list = [early_stopping]

# Dataset
train_datapath = 'processed_data/processed_face'
num_classes = len(os.listdir(train_datapath))

# Fix GPU
gpus = tf.config.experimental.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(gpus[0], True)

configGPU = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=configGPU)

def to_one_hot(a):
    b = np.zeros((a.size, num_classes))
    b[np.arange(a.size),a] = 1
    return b


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            'processed_data/processed_face',
            label_mode='categorical',
            seed=123,
            shuffle=True,
            image_size=config.IMG_SHAPE[:2])

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            'processed_data/valid_face',
            seed=123,
            shuffle=False,
            image_size=config.IMG_SHAPE[:2])

# train_ds = train_ds.unbatch()
# val_ds = val_ds.unbatch()

train_images = np.concatenate([x for x, y in train_ds], axis=0)
train_labels = np.concatenate([y for x, y in train_ds], axis=0)

val_images = np.concatenate([x for x, y in val_ds], axis=0)
val_labels = to_one_hot(np.concatenate([y for x, y in val_ds], axis=0))


def custom_generator(images, labels, batch_size=config.BATCH_SIZE):
    data_idx = 0
    while True:
        batch_x = [list(), list()] 
        batch_y = list() 

        for b in range(batch_size):
            if data_idx == images.shape[0]:
                data_idx = 0
            
            batch_x[0].append(images[data_idx])
            batch_x[1].append(labels[data_idx])
            batch_y.append(labels[data_idx])
            data_idx += 1

        batch_x[0] = np.array(batch_x[0])
        batch_x[1] = np.array(batch_x[1])
        batch_y = np.array(batch_y)

        yield batch_x, batch_y




# optimizer
Adam = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')




def get_model(training_mode=True, load_model=True):
    feature_extractor, last_embedding = base_network.face_network(input_shape=config.IMG_SHAPE,\
                                            embedding_dim=config.EMBEDDING_DIM, load_model=load_model)

    onehot_labels = Input(shape=(num_classes,), dtype='int32', name='onehot_label')
    
    if config.METRIC == 'cosface':
        predictions = CosFace(n_classes=num_classes)([last_embedding, onehot_labels])

    elif config.METRIC == 'arcface':
        predictions = ArcFace(n_classes=num_classes)([last_embedding, onehot_labels])

    elif config.METRIC == 'sphereface':
        predictions = SphereFace(n_classes=num_classes)([last_embedding, onehot_labels])

    elif config.METRIC == 'adacos':
        predictions = AdaCos(n_classes=num_classes)([last_embedding, onehot_labels])
        
    model = Model(inputs=[feature_extractor.input, onehot_labels], outputs=predictions)
    model.compile(optimizer=Adam, loss="categorical_crossentropy", metrics=[tf.keras.metrics.categorical_accuracy])

    for layer in model.layers[143:]:
        layer.trainable = True
    for layer in model.layers[:143]:
        layer.trainable = False
    
    return model, feature_extractor




if __name__ == "__main__":
    model, feature_extractor = get_model(load_model=False)
    # print(model.summary()) 

    history = model.fit(custom_generator(train_images, train_labels),
                        validation_data=custom_generator(val_images, val_labels),
                        callbacks=callbacks_list,
                        epochs=config.EPOCHS,
                        shuffle=True,
                        verbose='auto',
                        workers=0,
                        steps_per_epoch=10,
                        validation_steps=30, use_multiprocessing=True)

    print("[INFO] saving embedding model...")
    save_dir = config.MODEL_TRIPLET_PATH
    feature_extractor.save(save_dir)
    # plot the training history
    print("[INFO] plotting training history...")
    utils.plot_training(history, config.PLOT_TRILET_PATH, loss=False)
    history_dict = history.history
    # Save it under the form of a json file
    json.dump(history_dict, open(config.HISTORY_DICT_PATH, 'w+'))