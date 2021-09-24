import cv2
import os
import numpy as np
from random import shuffle
from tqdm import tqdm

import pandas as pd
from csv import writer
import random
import tensorflow as tf

from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.models import load_model

Normal_TRAIN_DIR = r'/content/drive/My Drive/DataSet/train/train/NORMAL'
Corona_TRAIN_DIR = r'/content/drive/My Drive/DataSet/train/train/COVID19 AND PNEUMONIA'
TEST_DIR = r'/content/drive/My Drive/DataSet/test/test'
weights = r'/content/drive/My Drive/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
IMG_SIZE = 224
MODEL_NAME = 'project_NN_CNN'

'''helper fundtions for data augmentation'''
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


'''Creating the training data'''


def create_train_data():
    training_data = []

    # loading the training data
    for img in tqdm(os.listdir(Normal_TRAIN_DIR)):
        # labeling the images
        label = label_img(img)

        path = os.path.join(Normal_TRAIN_DIR, img)

        img = cv2.imread(path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        normalizedImg = np.zeros((IMG_SIZE, IMG_SIZE))
        img = cv2.normalize(img, normalizedImg, 0, 255, cv2.NORM_MINMAX)

        # final step-forming the training data list with numpy array of the images
        training_data.append([np.array(img), np.array(label)])

        imgaug = cv2.flip(img, 1)
        training_data.append([np.array(imgaug), np.array(label)])

        imgaug2 = rotation(img, 30)
        training_data.append([np.array(imgaug2), np.array(label)])

    for img2 in tqdm(os.listdir(Corona_TRAIN_DIR)):
        # labeling the images
        label = label_img(img2)

        path = os.path.join(Corona_TRAIN_DIR, img2)

        img2 = cv2.imread(path)
        img2 = cv2.resize(img2, (IMG_SIZE, IMG_SIZE))
        normalizedImg = np.zeros((IMG_SIZE, IMG_SIZE))
        img2 = cv2.normalize(img2, normalizedImg, 0, 255, cv2.NORM_MINMAX)

        training_data.append([np.array(img2), np.array(label)])

    # shuffling of the training
    shuffle(training_data)

    # saving our trained data
    np.save('train_data.npy', training_data)
    return training_data


'''Processing and augmented the given test data'''


# we dont have to label it.
def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img
        img = cv2.imread(path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        normalizedImg = np.zeros((IMG_SIZE, IMG_SIZE))
        img = cv2.normalize(img, normalizedImg, 0, 255, cv2.NORM_MINMAX)
        testing_data.append([np.array(img), img_num])

    np.save('test_data.npy', testing_data)
    return testing_data


'''Running the training and the testing in the dataset for our model'''
if (os.path.exists('train_data.npy')):  # If you have already created the dataset:
    train_data = np.load('train_data.npy', allow_pickle=True)
    # train_data = create_train_data()
else:  # If dataset is not created:
    train_data = create_train_data()

if (os.path.exists('test_data.npy')):
    test_data = np.load('test_data.npy', allow_pickle=True)
else:
    test_data = process_test_data()

# vgg19 model using keras


model = Sequential()
model.add(
    Conv2D(input_shape=(IMG_SIZE, IMG_SIZE, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(units=4096, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(units=4096, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(units=1000, activation="softmax"))

opt = SGD(lr=0.01)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])

# Splitting the testing data and training data
train = train_data
test = test_data

'''Setting up the features and lables'''
# X-Features & Y-Labels

X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y_train = np.array([i[1] for i in train])

X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y_test = [i[1] for i in test]

'''Fitting the data into our model and saving it'''

if (os.path.exists('/content/model.h5')):
    model = model.load_model('/content/model.h5')
else:

    # transform learning using weights of pretrained vgg19 model with top layer
    model.load_weights(weights)
    model.add(Dense(units=2, activation="softmax"))

    model.fit(X_train, y_train, epochs=5)
    model.save('model.h5')

if (os.path.exists('test_data.npy')):
    test_data = np.load('test_data.npy', allow_pickle=True)
else:
    test_data = process_test_data()

for num, data in enumerate(test_data):
    # normal: [1, 0]
    # corona: [0, 1]

    img_num = data[1]
    img_data = data[0]

    data = img_data.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

    model_out = model.predict([data])

    if np.argmax(model_out) == 0:
        classy = '0'
    else:
        classy = '1'

    list_of_elem = [img_num, classy]

    with open('Submit.csv', 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)
    list_of_elem.clear()




