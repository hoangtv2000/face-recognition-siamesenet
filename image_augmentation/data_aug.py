from PIL import Image
import numpy as np
import cv2
import pandas as pd
import random
import csv
import os 

from auto_augment import SVHNPolicy, SubPolicy

augmentation_policy = SVHNPolicy(fillcolor=(0, 0, 0))
index = 0

for class_dir in os.listdir('processed_data\processed_face'):
    img_list = []

    for images in os.listdir(f'processed_data\processed_face\{class_dir}'):
        img_list.append(images)

        # if 'aug_' in images:
        #     os.remove(f'processed_data\processed_face\{class_dir}\{images}')
        #     print('DELETED : ' + f'processed_data\processed_face\{class_dir}\{images}')

    if len(img_list) <= 14:
        random_img_dir = random.choice(img_list)
        random_img_dir = f'processed_data/processed_face/{class_dir}/{random_img_dir}'
        img = Image.open(random_img_dir)
        aug_img = augmentation_policy(img)
        aug_img_name = str(random_img_dir.split('.')[0] + '_aug_' + str(index) + '.png')
        aug_img.save(aug_img_name)

        print('Saved an augmentated image at: ', aug_img_name)
        index += 1


import glob
import pandas as pd

result = pd.DataFrame(columns=['image', 'label'])

for full_img_dir in glob.glob('processed_data\processed_face\*\*'):
    label_ = full_img_dir.split('\\')[-2]
    dict_ = {'image': full_img_dir, 'label': label_}
    dict_ = pd.DataFrame([dict_])
    result = pd.concat([result, dict_])

result.to_csv('processed_data\processed_face.csv')