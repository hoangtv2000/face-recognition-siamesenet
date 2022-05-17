import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout, Lambda, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from source import config, utils
from source import triplet_generator as tg
from source import triplet_siamese_network as sn
import pandas as pd
import json

# callbacks
early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=25)
callbacks_list = [early_stopping]

# Dataset
train_datapath = ''
data = pd.read_csv('processed_data/processed_face_label.csv')

train, val = train_test_split(data, train_size=0.8, random_state=4953)
image_label_mapping_train = {k: v for k, v in zip(train.image.values, train.label.values)}
image_label_mapping_val = {k: v for k, v in zip(val.image.values, val.label.values)}

# Make triplet dataset
train_triplet = tg.make_triplet(image_label_mapping_train)
val_triplet = tg.make_triplet(image_label_mapping_val)

# Keras generator
keras_datagen = ImageDataGenerator(
		rotation_range=20,
		zoom_range=0.1,
		width_shift_range=0.15,
		height_shift_range=0.15,
		shear_range=0.12,
		vertical_flip=True,
		fill_mode="nearest")

train_generator = tg.triplet_generator(train_triplet, path=train_datapath, train_generator_mode=True, aug=keras_datagen)
val_generator = tg.triplet_generator(val_triplet, path=train_datapath, train_generator_mode=False)

# optimizer
Adam = tf.keras.optimizers.Adam(learning_rate=0.00025, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')

if __name__ == "__main__":
	# Model
	embedding_model, triplet_model = sn.triplet_siamese_network(config.IMG_SHAPE, continue_training=True)
	triplet_model.compile(loss=None, optimizer=Adam, )

	history = triplet_model.fit_generator(generator=train_generator,
	                            validation_data=val_generator,
								callbacks=callbacks_list,
	                            epochs=config.EPOCHS,
	                            verbose=1,
	                            workers=0,
	                            steps_per_epoch=10,
	                            validation_steps=30, use_multiprocessing=True)


	print("[INFO] saving embedding model...")
	save_dir = config.MODEL_TRIPLET_PATH
	embedding_model.save(save_dir)
	# plot the training history
	print("[INFO] plotting training history...")
	utils.plot_training(history, config.PLOT_TRILET_PATH, loss=False)
	history_dict = history.history
	# Save it under the form of a json file
	json.dump(history_dict, open(config.HISTORY_DICT_PATH, 'w+'))
