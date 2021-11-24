from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import cv2, numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from collections import Counter
import time
import numpy
import tensorflow as tf
from tensorflow import keras
from pickle import dump
from pickle import load
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Activation, Bidirectional, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D,TimeDistributed, LSTM
Dataset = "words.txt"
pickle_path = 'features.pkl'

def sample(Dataset, width, height, numbers): 
    file = open(Dataset, mode= 'rt' , encoding= 'utf-8')
    text = file.readlines()
    features = dict()
    for line in text[:numbers]:
        if not line or line[0]=='#':
            continue
        bad_samples_reference = ['a01-117-05-02', 'r06-022-03-05']  # known broken images in IAM dataset
        linesplit = line.strip().split(' ')

        fileNameSplit = linesplit[0].split('-')

        img_path = 'words/'+fileNameSplit[0]+'/'+fileNameSplit[0] + '-' + fileNameSplit[1]+'/'+linesplit[0]+'.png'   
        if linesplit[0] in bad_samples_reference:
            print('Ignoring known broken image:', img_path)
            continue 
        img_word = linesplit[-1]
        img = tf.io.read_file(img_path)
        img = tf.io.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        image = tf.image.resize_with_pad(1-img, height, width)
        img2 = tf.transpose(image, perm=[1, 0, 2]).numpy()
        features[img_word] = img2
    return features

def sample_id(Dataset, numbers): 
    file = open(Dataset, mode= 'rt' , encoding= 'utf-8')
    text = file.readlines()
    dataset = list()
    for line in text[:numbers]:
        if not line or line[0]=='#':
            continue
        bad_samples_reference = ['a01-117-05-02', 'r06-022-03-05']  # known broken images in IAM dataset
        linesplit = line.strip().split(' ')

        fileNameSplit = linesplit[0].split('-')

        img_path = 'words/'+fileNameSplit[0]+'/'+fileNameSplit[0] + '-' + fileNameSplit[1]+'/'+linesplit[0]+'.png'   
        if linesplit[0] in bad_samples_reference:
            print('Ignoring known broken image:', img_path)
            continue 
        img_word = linesplit[-1]
        dataset.append(img_word)
    return dataset

def load_features(filename, dataset):
    # load all features
    x=[]
    y=[]
    all_features = load(open(filename, 'rb'))
    # filter features
    for k in dataset:  
        x.append(all_features[k])
        y.append(k) 
    return x,y


if __name__ == "__main__":
	numbers = 1000
	width = 128
	height = 32  
	features = sample(Dataset, width, height, numbers)
	dump(features, open(pickle_path , 'wb'))
	dataset = sample_id(Dataset, numbers)
	x,y = load_features(pickle_path, dataset)
	xz = np.array(x[1])
	xy= np.reshape(xz,(xz.shape[0],xz.shape[1]))
	print(xy.shape)
	plt.figure()
	plt.imshow(xy.T)
	plt.show()
