import sys
import os
import numpy as np

import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers import Lambda, Input, Convolution2D,InputLayer
from keras.models import model_from_yaml, Model
import keras.callbacks
from tensorflow.keras.optimizers import RMSprop
try:
    from keras.optimizers import RMSpropGraves
except:
    print('You do not have RMSpropGraves')

import keras.backend as KTF
from keras import backend as K

class From_room_model:
    def __init__(self,rate,rooms):
        
        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(rooms,3)))
        self.model.add(Flatten())
        self.model.add(Dense((rooms*3)/2, activation='relu'))
        self.model.add(Dense(rooms))

        optimizer=RMSprop(lr=rate)
        self.model.compile(loss='mean_squared_error',optimizer=optimizer,metrics=['accuracy'])
        

    def Q_values(self, states):
        res = self.model.predict(np.array([states]))
        return res[0]
        
class To_room_model:
    def __init__(self,rate,rooms):

        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(rooms,3)))
        self.model.add(Flatten())
        self.model.add(Dense((rooms*3)/2, activation='relu'))
        self.model.add(Dense(rooms))

        optimizer=RMSprop(lr=rate)
        self.model.compile(loss='mean_squared_error',optimizer=optimizer,metrics=['accuracy'])
        

    def Q_values(self, states):
        res = self.model.predict(np.array([states]))
        return res[0]

class people_model:
    def __init__(self,rate,rooms):

        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(rooms,3)))
        self.model.add(Flatten())
        self.model.add(Dense((rooms*3)/2, activation='relu'))
        self.model.add(Dense(rooms))

        optimizer=RMSprop(lr=rate)
        self.model.compile(loss='mean_squared_error',optimizer=optimizer,metrics=['accuracy'])
        

    def Q_values(self, states):
        res = self.model.predict(np.array([states]))
        return res[0]
