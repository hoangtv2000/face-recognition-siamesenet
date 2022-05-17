"""This file's used for real-time camera visualizing results (frames with faces's bounding boxes and coressponding names).
"""

from turtle import begin_fill
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from scipy.optimize import linear_sum_assignment
from source import config, utils
from time import time
import pandas as pd 
from deepface.deepface.detectors import FaceDetector
from similarity_search import run_faiss
from tensorflow.keras import backend as K

# Deep Face
test_database = 'data/test'
backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']

face_detector = FaceDetector.build_model(backends[0])
face_representor = load_model(config.INFERENCE_MODEL, custom_objects={"K": K})

# Load train database
index, face_ids, _ = run_faiss.preprocess_index_anno_file('processed_data/train_feature.pkl')

# Load svm and train embedded database
# known_faces_embedded = pd_csv['feature_vector'].values
# known_faces_name = pd_csv['feature_vector'].values


def img_face_detection(img, input_path=False):
    list_embedded_faces = []
    list_bboxes_faces = []
    if input_path == True:
        img = cv2.imread(img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mtcnn_faces = FaceDetector.detect_faces(face_detector, backends[0], img)

    for face, bbox in mtcnn_faces:
        face = cv2.resize(face, (config.IMG_SHAPE[0], config.IMG_SHAPE[1]))
        face = preprocess_input(tf.expand_dims(np.array(face), axis=0))
        embedded_face = face_representor.predict(face)
        list_embedded_faces.append(embedded_face)
        list_bboxes_faces.append(bbox)
    list_embedded_faces = np.array(list_embedded_faces).reshape(len(list_embedded_faces), config.EMBEDDING_DIM).astype(np.float32)

    return list_embedded_faces, np.array(list_bboxes_faces)


def camera_face_recogintion(threshold=0.015):
    cap = cv2.VideoCapture(0)
    previous = time()
    delta = 0
    memories = ()

    while True:
        current = time()
        delta += current - previous
        previous = current

        timer = cv2.getTickCount()
        _, frame = cap.read()
        list_embedded_faces, list_bboxes_faces = img_face_detection(frame, input_path=False)

        if list_bboxes_faces.shape[0] == 0:
            continue

        if delta > 5 or len(memories) == 0:
            score_list, index_list = run_faiss.inference(index, list_embedded_faces)

        for idx in range(len(index_list)):
            x, y, w, h = list_bboxes_faces[idx]

            if score_list[idx][0] < threshold:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, str('Hok biet!'), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            
            else:
                content = str(index_list[idx][0]) + ' : ' + str(round(float(score_list[idx][0]), 4))
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, content, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        fps = cv2.getTickFrequency()/(cv2.getTickCount() - timer)
        info = 'fps:{}'.format(str(int(fps)))
        cv2.putText(frame, info, (70, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 50, 50), 1)

        cv2.imshow('Face_Recognition', frame)

        if cv2.waitKey(1) == ord('q'):
            break

        print(' + Predict 1 frame costs {}'.format(time() - current))
    cap.release()


if __name__ == "__main__":
    camera_face_recogintion(threshold=config.THRESHOLD)
