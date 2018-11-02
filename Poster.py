#!/bin/python3

import ast
import numpy as np
import pandas as pd
import glob
import scipy.misc
import skimage
import imageio
import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim

path = "E:\Project\Posters"

images = glob.glob(path + "/" + "*.jpg")
img_dict = {}

def get_id(filename):
    key = 't' + filename[-12:-4]
    return key

for fn in images:
    try:
        img_dict[get_id(fn)] = imageio.imread(fn)
    except:
        pass

data = pd.read_csv("movies_metadata.csv")

data["genres"] = data["genres"].apply(lambda x: ast.literal_eval(x))

def show_img(id):
    title = data[data["imdb_id"] == id]["original_title"].values[0]
    genre = data[data["imdb_id"] == "tt0029583"]["genres"].values[0][0]["name"]
    plt.imshow(img_dict[id])
    plt.title("{} \n {}".format(title, genre))

img_dict["tt0029583"][0]

def preprocess(img, size = (150, 101)):
    img = scipy.misc.imresize(img, size)
    img = img.astype(np.float64)
    img = (img / 127.5) - 1
    return img

def prepare_data(data, img_dict, size = (150, 101)):
    print("Generate dataset...")
    dataset = []
    y = []
    ids = []
    label_dict = {"word2idx": {}, "idx2word": []}
    idx = 0
    genre_per_movie = data["genres"].apply(lambda x: [x[i]["name"] for i in range(len(x))])
    for l in [g for d in genre_per_movie for g in d]:
        if l in label_dict["idx2word"]:
            pass
        else:
            label_dict["idx2word"].append(l)
            label_dict["word2idx"][l] = idx
            idx += 1
    n_classes = len(label_dict["idx2word"])
    print("identified {} classes".format(n_classes))
    n_samples = len(img_dict)
    print("got {} samples".format(n_samples))
    for k in img_dict:
        g = data[data["imdb_id"] == k]["genres"].values[0]
        img = preprocess(img_dict[k], size)
        if img.shape != (150, 101, 3):
            print(k)
            continue
        l = np.sum([np.eye(n_classes, dtype = np.float64)[label_dict["word2idx"][s["name"]]] for s in g], axis = 0)
        y.append(l)
        dataset.append(img)
        ids.append(k)
    print("Done")
    return dataset, y, label_dict, ids

SIZE = (150, 101)
dataset, y, label_dict, ids = prepare_data(data, img_dict, size = SIZE)

model = Sequential()

model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = (SIZE[0], SIZE[1], 3)))
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = keras.optimizers.Adam(), metrics = ['accuracy'])

n = 1000
y_train = np.zeros((n, 32))
for i in range(n):
    for j in range(y[i].size):    
        y_train[i, j] = y[i].item(j)

type(y)
type(y_train)

model.fit(np.array(dataset[:1000]), y_train, batch_size = 16, epochs = 3, verbose = 1, validation_split = 0.1)

n = 1000
n_test = 1000
X_test = dataset[n:n + n_test]
y_test = y[n:n + n_test]

pred = model.predict(np.array(X_test))

def show_example(idx):
    N_true = int(np.sum(y_test[idx]))
    show_img(ids[n + idx])
    print("Prediction :- ", end = ' ')
    for i in range(32):
        print(label_dict["idx2word"][i], pred[idx][i])
    
show_example(99)
