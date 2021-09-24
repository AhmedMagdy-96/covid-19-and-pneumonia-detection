import cv2
import os
import numpy as np
from random import shuffle
from tqdm import tqdm
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import pandas as pd
from csv import writer
import random

import tensorflow as tf

Normal_TRAIN_DIR = r'/content/drive/My Drive/DataSet/train/train/NORMAL'
Corona_TRAIN_DIR = r'/content/drive/My Drive/DataSet/train/train/COVID19 AND PNEUMONIA'
TEST_DIR = r'/content/drive/My Drive/DataSet/test/test'
IMG_SIZE = 60
LR = 0.001
MODEL_NAME = 'project_NN_CNN'

'''Helper functions for Augmentation'''

'''Labelling the dataset'''


def label_img(img):
    word_label = img

    if 'IM' in word_label:
        return [1, 0]
    else:
        return [0, 1]


def rotation(img, angle):
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w / 2), int(h / 2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img


'''Creating the training data and Preprocessing it'''


def create_train_data():
    training_data = []
    # Reading and preprocessing normal training dataset
    for img in tqdm(os.listdir(Normal_TRAIN_DIR)):
        # labeling the images
        label = label_img(img)

        path = os.path.join(Normal_TRAIN_DIR, img)

        img = cv2.imread(path, 0)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        normalizedImg = np.zeros((60, 60))
        img = cv2.normalize(img, normalizedImg, 0, 255, cv2.NORM_MINMAX)

        training_data.append([np.array(img), np.array(label)])

        imgaug = cv2.flip(img, 1)
        training_data.append([np.array(imgaug), np.array(label)])

        imgaug2 = rotation(img, 30)
        training_data.append([np.array(imgaug2), np.array(label)])

    # Reading and preprocessing Covid training dataset

    for img2 in tqdm(os.listdir(Corona_TRAIN_DIR)):
        # labeling the images
        label = label_img(img2)

        path = os.path.join(Corona_TRAIN_DIR, img2)

        img2 = cv2.imread(path, 0)
        img2 = cv2.resize(img2, (IMG_SIZE, IMG_SIZE))
        normalizedImg = np.zeros((60, 60))
        img2 = cv2.normalize(img2, normalizedImg, 0, 255, cv2.NORM_MINMAX)

        training_data.append([np.array(img2), np.array(label)])

    shuffle(training_data)

    # saving Trained dataset
    np.save('train_data.npy', training_data)
    return training_data


'''Processing the given test data'''


def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)

        img_num = img
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        normalizedImg = np.zeros((60, 60))
        img = cv2.normalize(img, normalizedImg, 0, 255, cv2.NORM_MINMAX)
        testing_data.append([np.array(img), img_num])

    # saving testing Dataset
    np.save('test_data.npy', testing_data)
    return testing_data


'''Running the training and the testing in the dataset'''
if (os.path.exists('train_data.npy')):
    train_data = np.load('train_data.npy', allow_pickle=True)

else:
    train_data = create_train_data()

if (os.path.exists('test_data.npy')):
    test_data = np.load('test_data.npy', allow_pickle=True)

else:
    test_data = process_test_data()

# Model


tf.reset_default_graph()
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 128, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 256, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 512, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 256, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 128, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = fully_connected(convnet, 2, activation='softmax')

convnet = regression(convnet, optimizer='SGD', learning_rate=LR,
                     loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

train = train_data
test = test_data

# splitting test and train dataset into featues and labels
# X-Features & Y-Labels

X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = [i[1] for i in train]

X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = [i[1] for i in test]

'''Fitting the data into our model'''

if (os.path.exists('model.tfl.meta')):
    model.load('./model.tfl')
else:

    model.fit({'input': X_train}, {'targets': y_train}, n_epoch=20,
              snapshot_step=200, show_metric=True, run_id=MODEL_NAME)
    model.save('model.tfl')

if (os.path.exists('test_data.npy')):
    test_data = np.load('test_data.npy', allow_pickle=True)
else:
    test_data = process_test_data()

for num, data in enumerate(test_data):
    # normal: [1, 0]
    # corona: [0, 1]

    # img_num is image label and image_data is image values
    img_num = data[1]
    img_data = data[0]

    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)

    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 0:
        classy = '0'
    else:
        classy = '1'

    list_of_elem = [img_num, classy]

    # writing Results

    with open('Submit.csv', 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)
    list_of_elem.clear()




