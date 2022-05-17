import cv2
from tensorflow.keras.models import load_model
from source import config
import pandas as pd 
import tensorflow as tf
from deepface.deepface.detectors import FaceDetector
from similarity_search import run_faiss
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import backend as K
import numpy as np

# Deep Face
test_database = 'data/test'
backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']

face_detector = FaceDetector.build_model(backends[0])
face_representor = load_model(config.INFERENCE_MODEL, custom_objects={"K": K})

# Load svm and train embedded database
# known_faces_embedded = pd_csv['feature_vector'].values
# known_faces_name = pd_csv['feature_vector'].values



def img_extraction(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face = cv2.resize(img, (config.IMG_SHAPE[0], config.IMG_SHAPE[1]))
    face = preprocess_input(tf.expand_dims(np.array(face), axis=0))
    embedded_face = face_representor.predict(face)
    return embedded_face



def img_face_recognition(val_dataframe, threshold=config.THRESHOLD):
    # Load train database
    index, face_ids, retrie_dataframe = run_faiss.preprocess_index_anno_file('processed_data/train_feature.pkl')
    val_dataframe = pd.read_csv(val_dataframe)
    true_cnter = 0

    for idx, image_dir in enumerate(val_dataframe['image']):
        embedded_face = img_extraction(image_dir)
        score_list, index_list = run_faiss.inference(index, embedded_face)

        if val_dataframe['label'][idx] in index_list[0]:
            true_cnter +=1

    print("OverAll Accuracy: {}".format(float(true_cnter/len(val_dataframe))))



if __name__ == '__main__':
    img_face_recognition(val_dataframe='processed_data/valid_face_label.csv')
