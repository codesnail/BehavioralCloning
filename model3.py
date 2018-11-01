# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 18:57:50 2018

@author: ahmed
"""

import csv
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.transform import resize
from keras.utils import Sequence

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, AveragePooling2D, Cropping2D, Lambda
from keras import backend as K

from sklearn.model_selection import train_test_split

# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.

class ImageBatchSequence(Sequence):

    def __init__(self, x_set, y_set, batch_size, shuffle=False, generator_name='train'):
        #print("__init__", shuffle, " - ", generator_name)
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.x))
        self.generator_name = generator_name
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        #print("on_epoch_end(): ", self.generator_name)
        self.indexes = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        #print("called __len__(): ", self.generator_name)
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        # Generate indexes of the batch
        #print("__getitem__(): idx*batch_size, (idx+1)*batch_size = ", idx*self.batch_size, (idx+1)*self.batch_size)
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        images = [mpimg.imread(self.x[ind]) for ind in indexes]
        batch_y = [self.y[ind] for ind in indexes]
        
        #if(idx==0):
            #print("batch_x[0] = ", batch_x[0], " - ", self.generator_name)
            #print("indexes[0] = ", indexes[0], " - ", self.generator_name)
        
        for i in range(self.batch_size):
            image_flipped = np.fliplr(images[i])
            measurement_flipped = -batch_y[i]
            images.append(image_flipped)
            batch_y.append(measurement_flipped)
        
        return np.array(images), np.array(batch_y)
    

def lenet(reset_mode=True):
    model = None
    
    if(reset_model):
        
        model = Sequential()
        model.add(Cropping2D(cropping=((75,10),(10,10)), input_shape=(160,320,3)))
        model.add(Lambda(lambda x: (x/255.0)-0.5))
        model.add(Conv2D(6, (5,5), activation='relu', strides=1, padding='valid'))#, input_shape=(160,320,1))) # old depth: 6
        model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
        model.add(Conv2D(16, (5,5), activation='relu', strides=1, padding='valid'))
        model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
        model.add(Flatten())
        model.add(Dense(120, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(84, activation='relu')) #64
        model.add(Dropout(0.1))
        model.add(Dense(1))
    else:
        from keras.models import load_model
        model = load_model('model3.h5')
    
    model.compile(loss='mse', optimizer='adam')
    
    return model
     
def train(training_size, reset_model=True):
    X, y = loadData(training_size)
    model = lenet(reset_model)
    
    for i in range(0,2):
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=np.random.randint(0, 100))
        trainingGenerator = ImageBatchSequence(X_train, y_train, 100, shuffle=True, generator_name="train")
        validationGenerator = ImageBatchSequence(X_valid, y_valid, 40, shuffle=True, generator_name="validate" )
    
        model.fit_generator(
            generator=trainingGenerator,
            validation_data=validationGenerator,
            shuffle=True,
            epochs=2
            #use_multiprocessing=False,
            #workers=4
        )
        
        model.save('model3.h5')

reset_model = True

#training_sizes = [250, 500, 750, 1000, 1250, 1500, 1750, 2000]
training_sizes = [2000] #, 4000, 6000, 8000, 10000]
augmented_training_sizes = []
durations = []
import time

for size in training_sizes:
    start = time.time()
    train(size, reset_model)
    end = time.time()
    durations.append(end-start)
    
