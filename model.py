# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 18:57:50 2018

@author: ahmed
"""

import csv
import cv2
import numpy as np
import matplotlib.image as mpimg

def loadExtras():
    '''
    Additional driving log to train on failed curves on the track
    '''
    
    lines = []
    folder = 'C:\\Yaser\\Udacity\\CarND-Term1\\BehavioralCloning\\extras\\'
    
    with open(folder + 'driving_log2.csv') as csvFile:
        reader = csv.reader(csvFile)
        for line in reader:
            lines.append(line)
            
    images = []
    measurements = []
    for csv_line in lines:
        source_path = csv_line[0]
        filename = source_path.split('\\')[-1]
        #filename = source_path.split('/')[-1]
        #print(filename)
        filename = folder + 'IMG\\' + filename
        img = mpimg.imread(filename) #/255.
        images.append(img)
        center_correction = 0.00
        steering_center = float(csv_line[3]) + center_correction
        measurements.append(steering_center)
        
        image_flipped = np.fliplr(img)
        measurement_flipped = -steering_center
        images.append(image_flipped)
        measurements.append(measurement_flipped)
    
    return np.array(images), np.array(measurements)

def loadImages():
    lines = []
    #folder = 'C:\\Users\\ahmed\\OneDrive\\CarND\\behavioral_cloning\\'
    #folder = 'C:/Users/ahmed/OneDrive/CarND/spyderws/BehavioralCloning/'
    
    # Credit for the initial training set to ericlavigne, on slack channel...
    folder = 'C:\\Yaser\\Udacity\\CarND-Term1\\BehavioralCloning\\ericlavigne-data\\'
    
    #with open('C:\\Users\\ahmed\\OneDrive\\CarND\\behavioral_cloning\\driving_log.csv') as csvFile:
    with open(folder + 'driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)
        for line in reader:
            lines.append(line)
            
    images = []
    measurements = []
    for csv_line in lines:
        source_path = csv_line[0]
        #filename = source_path.split('\\')[-1]
        filename = source_path.split('/')[-1]
        filename = folder + 'IMG\\' + filename
        img = mpimg.imread(filename) #/255.
        images.append(img)
        center_correction = 0.00
        steering_center = float(csv_line[3]) + center_correction
        measurements.append(steering_center)
        
        # Augment data by horizontal flipping the training set, since the 
        # training track is mostly curving left
        image_flipped = np.fliplr(img)
        measurement_flipped = -steering_center
        images.append(image_flipped)
        measurements.append(measurement_flipped)
    
    images = np.array(images)
    measurements = np.array(measurements)
    
    # Add additional training data for the parts of track on which the car was
    # failing with the above data set
    images2, measurements2 = loadExtras()
    
    images = np.concatenate((images, images2))
    measurements = np.concatenate((measurements, measurements2))
    
    return images, measurements


def train(images, measurements, reset_model=True):
    X_train = images #np.array(images)
    y_train = measurements #np.array(measurements)
    
    from keras.models import Sequential
    from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, AveragePooling2D, Cropping2D, Lambda
    from keras import backend as K
    
    #K.tf.Session(config=K.tf.ConfigProto(log_device_placement=True))
    model = None
    
    if(reset_model): # Create and train model from scratch
        model = Sequential()
        #model.add(Flatten(input_shape=(160,320,3)))
        model.add(Cropping2D(cropping=((75,10),(10,10)), input_shape=(160,320,3)))
        #model.add(Lambda(lambda x: K.tf.image.rgb_to_grayscale(x)))
        model.add(Lambda(lambda x: (x/255.0)-0.5))
        
        # Lenet architecture...
        model.add(Conv2D(6, (5,5), activation='relu', strides=1, padding='valid'))#, input_shape=(160,320,1))) # old depth: 6
        model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
        model.add(Dropout(0.2))
        model.add(Conv2D(16, (5,5), activation='relu', strides=1, padding='valid')) #old depth: 14
        model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(400, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(84, activation='relu')) #64
        model.add(Dropout(0.2))
        model.add(Dense(1))
        
    else: # Load a previously trained saved model
        from keras.models import load_model
        model = load_model('model.h5')
    
    model.compile(loss='mse', optimizer='adam')
    for i in range(0,5):
        model.fit(X_train, y_train, validation_split = 0.2, shuffle=True, epochs=5)
        model.save('model.h5')

# Create and train a model from scratch, or start from a previously trained and saved model in h5 file.
# reset_model = True means start from scratch, otherwise set to False.
reset_model = True

img_arr, meas_arr = loadImages()
train(img_arr, meas_arr, reset_model)