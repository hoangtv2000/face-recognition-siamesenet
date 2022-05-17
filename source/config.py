"""Config variables, hyper-parameters and storage path.
"""
import os

IMG_SHAPE = (224, 224, 3)

# Recognition threshold 
THRESHOLD = 0.01

#===============================================================================#
# SIMILARITY SEARCH
K_NEIGHBORS = 3
#===============================================================================#
# TRAINING
METRIC = 'adacos'

TRAIN_ANNO_DIR = 'processed_data/processed_face_label.csv'
TRAIN_IMG_DIR = 'data/train'

BATCH_SIZE = 16
EPOCHS = 150
MARGIN = 0.95
EMBEDDING_DIM = 512

BASE_OUTPUT = "output"

#===============================================================================#

MODEL_PATH = os.path.join(BASE_OUTPUT, 'siamese_model')
PLOT_PATH = os.path.join(BASE_OUTPUT, "plot.png")

#===============================================================================#

OUTPUT_TRIPLET = f'output/output_facenet_{METRIC}_KEEP_ADACOSLAYER'
MODEL_TRIPLET_PATH = os.path.join(OUTPUT_TRIPLET, 'best_model')
PLOT_TRILET_PATH = os.path.join(OUTPUT_TRIPLET, "model_result.png")
HISTORY_DICT_PATH = os.path.join(OUTPUT_TRIPLET, "histogram.json")

SAVED_MODEL = f'output/output_adacos'
SAVED_MODEL_PATH = 'output\\output_triplet_3\\best_model' # os.path.join(SAVED_MODEL, 'best_model')

#===============================================================================#
CONTINUE_TRAINING_DIR = 'output\\output_adacos\\best_model'
INFERENCE_MODEL = 'output\\output_adacos\\best_model'