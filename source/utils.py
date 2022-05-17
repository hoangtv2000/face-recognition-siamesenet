from keras import backend as K
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from . import config
import os
import cv2
import pandas as pd


def euclidean_distance(featsA, featsB):
    sumSquared = K.sum(K.square(featsA - featsB), keepdims=True)
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))


def plot_training(H, plotPath, loss=True):
    # construct a plot that plots and saves the training history
    plt.style.use("ggplot")
    plt.figure()
    if loss == True:
        plt.plot(H.history["loss"], label="train_loss")
        plt.plot(H.history["val_loss"], label="val_loss")
        plt.plot(H.history["accuracy"], label="train_acc")
        plt.plot(H.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
    else:
        plt.plot(H.history["loss"], label="train_loss")
        plt.plot(H.history["val_loss"], label="val_loss")
        plt.title("Training Loss")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend(loc="lower left")
    plt.savefig(plotPath)


#Stack images horizontally
def stack_imgs(scale, image_list, labels = []):
    """Return stacked image for all images image_list.
    """
    rows = len(image_list)
    cols = len(image_list[0])
    rows_available = isinstance(image_list[0],list)
    wid = image_list[0][0].shape[1]
    hei = image_list[0][0].shape[0]
    #have columns
    if rows_available:
        for x in range(rows):
            for y in range(cols):
                if image_list[x][y].shape[:2] == image_list[0][0].shape[:2]:
                    image_list[x][y] = cv2.resize(image_list[x][y],(0,0),None,scale,scale)
                else:
                    image_list[x][y] = cv2.resize(image_list[x][y],(image_list[0][0].shape[1],image_list[0][0].shape[0]),None,scale,scale)
                if len(image_list[x][y].shape) == 2:
                    image_list[x][y]= cv2.cvtColor(image_list[x][y], cv2.COLOR_GRAY2BGR)
        blank_img = np.zeros((hei,wid,3), np.uint8)
        hor = [blank_img]*rows
        hor_con = [blank_img]*rows
        for x in range(rows):
            hor[x] = np.hstack(image_list[x])
        ver = np.vstack(hor)
    #no column
    else:
        for x in range(rows):
            if image_list[x].shape[:2] == image_list[0].shape[:2]:
                image_list[x] = cv2.resize(image_list[x], (0, 0), None, scale, scale)
            else:
                image_list[x] = cv2.resize(image_list[x], (image_list[0].shape[1], image_list[0].shape[0]), None,scale, scale)
            if len(image_list[x].shape) == 2:
                image_list[x] = cv2.cvtColor(image_list[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(image_list)
        ver = hor

    if len(labels) != 0:
        each_img_w = int(ver.shape[1]/cols)
        each_img_h = int(ver.shape[0]/rows)
        for d in range(rows):
            for c in range(cols):
                cv2.rectangle(ver, (c*each_img_w, each_img_h*d),(c*each_img_w + len(labels[d])*13+27, 30+each_img_h*d),(255,255,255), cv2.FILLED)
                cv2.putText(ver, labels[d], (each_img_w*c+10, each_img_h*d+20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,0,255), 2)
    return ver



def from_data_to_df(processed_face_dir='processed_data\processed_face'):
    result_df = pd.DataFrame(columns=['ID', 'name'])
    for label in os.listdir(processed_face_dir):
        img_label_dirs = os.listdir(os.path.join(processed_face_dir, label))
        if len(img_label_dirs) < 3:
            continue
        img_label_dirs = [str(processed_face_dir + f'/{label}/'+img_label_dir) for img_label_dir in img_label_dirs]
        for img_label_dir in img_label_dirs:
            label_img_dict = {'ID' : label, 'name' : img_label_dir}
            label_img_dict = pd.DataFrame([label_img_dict])
            result_df = pd.concat([result_df, label_img_dict], ignore_index=True)  
    return result_df