import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import sys
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description = "arguments for output model prediction")
# ./models/facenet/....

parser.add_argument("--model", default = "output/output_facenet_adacos_20180402-114759/best_model", help = "path to model")
parser.add_argument("--embeddings", default = "processed_data\embedded_face_val\embs_class_20180402-114759.csv", help = "path to csv index to embedding files")
args = parser.parse_args()

NUM_CLASSES = len(os.listdir('processed_data\\embedded_face\\20180402-114759\\processed_face_aligned'))

if __name__ == "__main__":
    model = load_model(args.model)
    labels = pd.read_csv(args.embeddings)
    labels.columns = ["image", "class"]
    print(labels.head)
    probs = []
    cols = ["image"]

    for i in range(1, NUM_CLASSES+1):
        cols.append("class_{}".format(i))

    for i in range(labels.shape[0]):
        # if not os.path.exists(labels["image"][i]):
        #     p = [0 for i in range(1,NUM_CLASSES)]
        #     probs.append([labels["image"][i], p]) 
            
        embs = np.load(labels["image"][i]).reshape(-1, 512)
        pred = model.predict(embs, verbose = 1)   
        p = []
        p.append(labels["image"][i])     
        for j in range(NUM_CLASSES):
            p.append(pred[0][j])
        probs.append(p)

    probs = pd.DataFrame(probs)

    probs.columns = [cols]
    print(probs.head)
    probs.to_csv("predictions/" + args.model.split("/")[1] + ".csv")


