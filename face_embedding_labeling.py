"""This file to localize and encode images contain faces in database.
"""
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import numpy as np
import os
from source import config
import cv2
import pandas as pd


mtcnn_detector = MTCNN()

test_database = 'data/test'
embedd_model = load_model(config.INFERENCE_MODEL, custom_objects={"K": K})


def preprocess_n_detect_face(path, required_size = (config.IMG_SHAPE[0], config.IMG_SHAPE[1])):
    """Detect and encode faces of a image, input is path of image.
    """
    face = []
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img)
    
    # return tf.expand_dims(np.array(img), axis=0)

    result = mtcnn_detector.detect_faces(img)
    try:
        x, y, w, h = result[0]['box']
        x, y = abs(x), abs(y)
        face = img[y: y+h, x: x+w]
        face = cv2.resize(face, (config.IMG_SHAPE[0], config.IMG_SHAPE[1]))
        return tf.expand_dims(np.array(face), axis=0)
    except IndexError:
        return None




def get_name(anno_file):
    result_df = pd.DataFrame(columns=['ID', 'name', 'feature_vector'])
    for i in range(len(anno_file)):
        _, full_img_name, img_lbl = anno_file.iloc[i]
        if "_" in full_img_name.split("\\")[-1]:
            continue
        
        print('Extracting image :{} | {}'.format(img_lbl, full_img_name.split("\\")[-1]))
        face = preprocess_n_detect_face(full_img_name)
        if face == None:
            continue
        
        else:
            feature_vector = embedd_model.predict(face)[0]
            result_dict = {'ID': img_lbl, 'name': full_img_name, 'feature_vector' : np.array(feature_vector, np.float32)}
            result_dict = pd.DataFrame([result_dict])
            result_df = pd.concat([result_df, result_dict], ignore_index=True)  
            
    return result_df


celeb_dict = dict()


if __name__ == "__main__":
    train_dir = config.TRAIN_ANNO_DIR 
    train_file = pd.read_csv(train_dir)
    result_df =  get_name(train_file)
    result_df.to_pickle('processed_data/train_feature.pkl')
