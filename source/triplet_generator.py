from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import backend as K
from collections import defaultdict
import numpy as np
import tensorflow as tf
from . import config
import cv2
import os

def read_and_resize(filename):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return tf.image.resize(image, [config.IMG_SHAPE[0], config.IMG_SHAPE[1]])


class make_triplet(object):

    def __init__(self, image_label_mapping):
        self.image_label_mapping = image_label_mapping
        self.label_to_list_images = defaultdict(list)
        self.list_all_images = list(image_label_mapping.keys())
        self.range_all_images = list(range(len(self.list_all_images)))

        for image_name, label in image_label_mapping.items():
            self.label_to_list_images[label].append(image_name)

        self.list_labels = list(set(self.image_label_mapping.values()))
        self.range_list_labels = range(len(self.list_labels))
        self.label_weight = np.array([len(self.label_to_list_images[label]) for label in self.list_labels])
        self.label_weight = self.label_weight/np.sum(self.label_weight)

    def get_sample(self):
        label_id = np.random.choice(self.range_list_labels, 1, p=self.label_weight)[0]
        positive_label_id = np.random.choice(range(len(self.label_to_list_images[self.list_labels[label_id]])), 2)
        positive_example_1, positive_example_2 = \
            self.label_to_list_images[self.list_labels[label_id]][positive_label_id[0]],\
            self.label_to_list_images[self.list_labels[label_id]][positive_label_id[1]]
        negative_example = None
        negative_example_id = None
        negative_label = None
        while negative_example is None or self.image_label_mapping[negative_example] == \
                self.image_label_mapping[positive_example_1]:
            negative_example_id = np.random.choice(self.range_all_images, 1)[0]
            negative_example = self.list_all_images[negative_example_id]
            negative_label = self.image_label_mapping[negative_example]
        return positive_example_1, negative_example, positive_example_2,\
         label_id, negative_label, label_id



def triplet_generator(triplet_gen, path=None, train_generator_mode=False, aug=None):

    while True:
        list_positive_examples_1 = []
        list_negative_examples = []
        list_positive_examples_2 = []
        label_list = []

        for i in range(config.BATCH_SIZE):
            positive_example_1, negative_example, positive_example_2,\
             label_id, negative_example_id, positive_label_id = triplet_gen.get_sample()
             
            path_pos1 = os.path.join(path, positive_example_1)
            path_neg = os.path.join(path, negative_example)
            path_pos2 = os.path.join(path, positive_example_2)

            positive_example_1_img = read_and_resize(path_pos1)
            negative_example_img = read_and_resize(path_neg)
            positive_example_2_img = read_and_resize(path_pos2)

            list_positive_examples_1.append(positive_example_1_img)
            list_negative_examples.append(negative_example_img)
            list_positive_examples_2.append(positive_example_2_img)

            label_list.append([label_id, negative_example_id, positive_label_id])

        LABEL = np.array(label_list)

        if train_generator_mode == False:
            P1 = preprocess_input(np.array(list_positive_examples_1))
            N = preprocess_input(np.array(list_negative_examples))
            P2 = preprocess_input(np.array(list_positive_examples_2))

            yield [P1, N, P2], LABEL
            # yield [P1, N, P2]

        if train_generator_mode == True:
            P1 = aug.flow(np.array(list_positive_examples_1), LABEL, batch_size=config.BATCH_SIZE, shuffle=False)
            N = aug.flow(np.array(list_negative_examples), LABEL, batch_size=config.BATCH_SIZE, shuffle=False)
            P2 = aug.flow(np.array(list_positive_examples_2), LABEL, batch_size=config.BATCH_SIZE, shuffle=False)

            P1i = P1.next()
            Ni = N.next()
            P2i = P2.next()

            yield [P1i[0], Ni[0], P2i[0]], P1i[1]
            # yield [P1i[0], Ni[0], P2i[0]]