import multiprocessing
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout, add, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

import argparse, json

parser = argparse.ArgumentParser(description = "train classifier of facenet model")

parser.add_argument("--model", default = "20180402-114759", help = "model name")
parser.add_argument("--embeddings", default = "processed_data\embedded_face\embs_class_20180402-114759.csv", help = "path to file index embeddings for train")
parser.add_argument("--size", default = 160, help = 'image size')
args = parser.parse_args()

import sys, os
sys.path.append('C:\\Users\\ASUS\\face-recognition-siamesenet')
from source.adacos_loss import AdaCos
from source import config, utils

INPUT_SHAPE = 512
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 100
NUM_CLASSES = len(os.listdir('processed_data\\embedded_face\\20180402-114759\\processed_face_aligned'))


def build_model():
    model = Sequential()
    model.add(Dense(2048, input_dim = INPUT_SHAPE, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(NUM_CLASSES+1, activation = "relu"))
    model.add(BatchNormalization())

    onehot_labels = Input(shape=(NUM_CLASSES+1,), dtype='int32', name='onehot_label')
    predictions = AdaCos(n_classes=NUM_CLASSES+1)([model.output, onehot_labels])

    model = Model(inputs=[model.input, onehot_labels], outputs=predictions)

    return model


def DataGenerator(labels, classes, batch_size):
    i = 0
    while True:
        embeddings_batch = [np.array([],dtype = np.float32).reshape(0,512), list()]
        class_batch = []
        for b in range(batch_size):
            if i == labels.shape[0]:
                i = 0
            sample = np.load(labels["emb"][i]).reshape(-1,512)
            cl = classes[i]
            embeddings_batch[0] = np.vstack((embeddings_batch[0], sample))
            embeddings_batch[1].append(cl) 
            class_batch.append(cl)
            i += 1

        embeddings_batch[0] = np.array(embeddings_batch[0])
        embeddings_batch[1] = np.array(embeddings_batch[1])
        class_batch = np.array(class_batch)

        yield embeddings_batch, class_batch



def convert2categorical(labels):
    categories = []
    for index, row in labels.iterrows():
        categories.append(to_categorical(row[2], num_classes = NUM_CLASSES+1))
    return np.array(categories)



if __name__ == '__main__':
    model = build_model()
    labels = pd.read_csv(args.embeddings)
    labels.columns = ['index', 'emb', 'class']
    categories = convert2categorical(labels)

    labels_train = labels.sample(frac = 0.8, random_state = 0)
    labels_val = labels.drop(labels_train.index)
    categories_train = convert2categorical(labels_train)
    categories_val = convert2categorical(labels_val)
    
    labels_train.reset_index(inplace = True)
    labels_val.reset_index(inplace = True)

    # print(labels_train.head)
    # print(labels_val.head)
    print(categories_train.shape)
    print(categories_val.shape)

    train_generator = DataGenerator(labels_train, categories_train, BATCH_SIZE)
    val_generator = DataGenerator(labels_val, categories_val, BATCH_SIZE)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
    metric = tf.keras.metrics.CategoricalAccuracy()
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = [metric])

    callbacks_list = [
		tf.keras.callbacks.EarlyStopping(patience=9, monitor='val_categorical_accuracy', restore_best_weights=True),
		# tf.keras.callbacks.ModelCheckpoint(filepath ="./checkpoints/facenet/"+args.model+"/weights-epoch{epoch:02d}-loss{val_loss:.2f}.h5")
		]
        

    history = model.fit_generator(train_generator, 
                        validation_data=val_generator,
                        epochs = EPOCHS,
                        callbacks=callbacks_list,
                        verbose = 'auto',
                        workers=1,
                        use_multiprocessing=False,
                        steps_per_epoch = len(labels_train)/BATCH_SIZE,
                        validation_steps = len(labels_val)/BATCH_SIZE)
    

    print("[INFO] saving embedding model...")
    save_dir = config.MODEL_TRIPLET_PATH
    model.save(save_dir)
    # plot the training history
    print("[INFO] plotting training history...")
    utils.plot_training(history, config.PLOT_TRILET_PATH, loss=False)
    history_dict = history.history
    # Save it under the form of a json file
    json.dump(history_dict, open(config.HISTORY_DICT_PATH, 'w+'))