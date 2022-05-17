from turtle import begin_fill
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from scipy.optimize import linear_sum_assignment
from source import config, utils
from time import time
import pandas as pd 
from deepface.deepface.detectors import FaceDetector
from similarity_search import run_faiss

# Deep Face
test_database = 'data/test'
backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']

face_detector = FaceDetector.build_model(backends[0])
face_representor = load_model(config.INFERENCE_MODEL, custom_objects={"K": K})

# Load train database
index, face_ids, dataframe = run_faiss.preprocess_index_anno_file('processed_data/train_feature.pkl')

# Load svm and train embedded database
# known_faces_embedded = pd_csv['feature_vector'].values
# known_faces_name = pd_csv['feature_vector'].values




def img_face_detection(img, input_path=False):
    list_embedded_faces = []
    list_bboxes_faces = []
    if input_path == True:
        img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mtcnn_faces = FaceDetector.detect_faces(face_detector, backends[0], img)
    # img = cv2.resize(img, (config.IMG_SHAPE[:2]), interpolation = cv2.INTER_AREA)

    for face, bbox in mtcnn_faces:
        face = cv2.resize(face, (config.IMG_SHAPE[0], config.IMG_SHAPE[1]))
        face = preprocess_input(tf.expand_dims(np.array(face), axis=0))
        embedded_face = face_representor.predict(face)
        list_embedded_faces.append(embedded_face)
        list_bboxes_faces.append(bbox)
    list_embedded_faces = np.array(list_embedded_faces).reshape(len(list_embedded_faces), config.EMBEDDING_DIM).astype(np.float32)

    return img, list_embedded_faces, np.array(list_bboxes_faces)




def img_face_recognition(path, dataframe, threshold=config.THRESHOLD):
    img, list_embedd_faces, list_bboxes_faces = img_face_detection(path, input_path=True)
    score_list, index_list = run_faiss.inference(index, list_embedd_faces)
    
    img_list = []

    for idx in range(len(index_list)):
        x, y, w, h = list_bboxes_faces[idx]
        clone_img = img.copy()

        if score_list[idx][0] < threshold:
            cv2.rectangle(clone_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(clone_img, str('Hông biết!'), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,255), 3)
            
        else:
            content = str(index_list[idx][0]) + ' : ' + str(round(float(score_list[idx][0]), 4))
            cv2.rectangle(clone_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(clone_img, content, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,255), 3)

            img_list.append(clone_img)

            rele_score = index_list[idx][0]
            img_name = dataframe[dataframe['ID'] == rele_score]['name'].values[0]
            # img_path = os.path.join(config.TRAIN_IMG_DIR, img_name)
            img_list.append(cv2.imread(img_name)[:,:,::-1])
    
    if len(img_list)>0:
        img_list = [img_list[i:i+2] for i in range(0, len(img_list), 2)]
        stacked_img = utils.stack_imgs(image_list=img_list, scale=0.6)
    else:
        stacked_img = img

    winname = 'Face_Recognition'
    cv2.namedWindow(winname)
    cv2.moveWindow(winname, 100, 100)
    cv2.imshow(winname, cv2.cvtColor(stacked_img, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)

if __name__ == '__main__':
    img_face_recognition('C:/Users/ASUS/OneDrive/Desktop/qag-1544686937-1595322851654-640x375.jpg', dataframe)
