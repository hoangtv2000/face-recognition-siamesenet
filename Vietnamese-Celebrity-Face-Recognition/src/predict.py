import pandas as pd
import numpy as np
import sys
import os

path1 = "predictions\output_facenet_adacos_20180402-114759.csv"

NUM_CLASSES = len(os.listdir('processed_data\\embedded_face\\20180402-114759\\processed_face_aligned'))

if __name__ == '__main__':
    probs1 = pd.read_csv(path1)
    results = []

    for i in range(probs1.shape[0]):
        print("{}/{}".format(i, probs1.shape[0]))
        avg_probs = np.array([(probs1["class_{}".format(x)][i]) for x in range(1, NUM_CLASSES+1)])
        avg_probs = np.argsort(avg_probs)
        
        if avg_probs[0] == 0:
            res = "1003"

        res = " ".join(avg_probs+1)

        results.append([probs1["image"][i], res])

    results = pd.DataFrame(results)
    results.to_csv("final_result/final.csv")