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

from collections import deque
import os
import sys
import copy
##from util import clone_model

sys.path.append(os.pardir)
#from env import popu
#from models import model as Mdl
import tensorflow as tf

import numpy as np

f_log = 'Scripts/logs'
f_model = 'Scripts/models'

From_model_filename = 'From_model.json'
From_weights_filename = 'From_model_weights.hdf5'

To_model_filename = 'To_model.json'
To_weights_filename = 'To_model_weights.hdf5'

People_model_filename = 'People_model.json'
People_weights_filename = 'People_model_weights.hdf5'



class DQNAgent:
    def __init__(self,env_name,rooms):
        self.env_name=env_name
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.episode=1000
        self.minibatch_size=32
        self.lerning_rate=0.001
        self.discount_factor=0.9
        self.exploration=0.2
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        self.model_name = "{}.ckpt".format(self.env_name)
        

        self.D=deque(maxlen=self.episode)

        self.init_From_model(self.lerning_rate,rooms)
        self.init_To_model(self.lerning_rate,rooms)
        self.init_People_model(self.lerning_rate,rooms)

        self.current_loss=0.0
    
    def init_From_model(self,rate,rooms):
        self.From_model = Sequential()
        self.From_model.add(InputLayer(input_shape=(rooms,3)))
        self.From_model.add(Flatten())
        self.From_model.add(Dense((rooms*3)/2, activation='relu'))
        self.From_model.add(Dense(rooms))
        optimizer=RMSprop(lr=rate)
        self.From_model.compile(optimizer='rmsprop',loss='mean_squared_error',metrics=['accuracy'])

    def init_To_model(self,rate,rooms):
        
        self.To_model = Sequential()
        self.To_model.add(InputLayer(input_shape=(rooms,3)))
        self.To_model.add(Flatten())
        self.To_model.add(Dense((rooms*3)/2, activation='relu'))
        self.To_model.add(Dense(rooms))

        optimizer=RMSprop(lr=rate)
        self.To_model.compile(optimizer=optimizer,loss='mean_squared_error',metrics=['accuracy'])

    def init_People_model(self,rate,rooms):
        
        self.People_model = Sequential()
        self.People_model.add(InputLayer(input_shape=(rooms,3)))
        self.People_model.add(Flatten())
        self.People_model.add(Dense((rooms*3)/2, activation='relu'))
        self.People_model.add(Dense(10))

        optimizer=RMSprop(lr=rate)
        self.People_model.compile(optimizer=optimizer,loss='mean_squared_error',metrics=['accuracy'])

    def From_model_Q_values(self, states):
        res = self.From_model.predict(np.array([states]))
        return res[0]

    def To_model_Q_values(self, states):
        res = self.To_model.predict(np.array([states]))
        return res[0]

    def People_model_Q_values(self, states):
        res = self.People_model.predict(np.array([states]))
        return res[0]

    

    def select_action(self,state,ep):
        a=np.random.rand()
        #print(a)
        #print(ep)
        if a<=ep:
            return np.random.randint(3),np.random.randint(3),np.random.randint(1,10)  #移動元、移動先、人数の割合
        else:
            return np.argmax(self.From_model_Q_values(state)),np.argmax(self.To_model_Q_values(state)),np.argmax(self.People_model_Q_values(state))
            #return np.argmax(self.From_model_Q_values(state)),np.argmax(self.To_model_Q_values(state)),10
                
    def store_experience(self, state, action, reward, state_1, flag):
        self.D.append((state, action, reward, state_1, flag))

    def experience_replay(self):
        state_minibatch = []
        y_f_minibatch = []
        y_t_minibatch = []
        y_p_minibatch = []

        # sample random minibatch
        minibatch_size = min(len(self.D), self.minibatch_size)
        minibatch_indexes = np.random.randint(0, len(self.D), minibatch_size)

        for j in minibatch_indexes:
            state_j, action_j, reward_j, state_j_1, terminal = self.D[j]
            action_j_index = action_j

            y_f_j = self.From_model_Q_values(state_j)
            y_t_j = self.To_model_Q_values(state_j)
            y_p_j = self.People_model_Q_values(state_j)

            if terminal:
                y_f_j[action_j_index[0]] = reward_j
                y_t_j[action_j_index[1]] = reward_j
                y_p_j[action_j_index[2]] = reward_j
            else:
                # reward_j + gamma * max_action' Q(state', action')
                print("reward:{}   des:{}   f:{}  t:{}  p:{}".format(reward_j,self.discount_factor,np.max(self.From_model_Q_values(state_j_1)),np.max(self.To_model_Q_values(state_j_1)) ,np.max(self.People_model_Q_values(state_j_1)) ))
                y_f_j[action_j_index[0]] = reward_j + self.discount_factor * np.max(self.From_model_Q_values(state_j_1))  # NOQA
                y_t_j[action_j_index[1]] = reward_j + self.discount_factor * np.max(self.To_model_Q_values(state_j_1))  # NOQA
                y_p_j[action_j_index[2]] = reward_j + self.discount_factor * np.max(self.People_model_Q_values(state_j_1))  # NOQA

            state_minibatch.append(state_j)
            y_f_minibatch.append(y_f_j)
            y_t_minibatch.append(y_t_j)
            y_p_minibatch.append(y_p_j)

        #print(type(self.From_model))
        callbacks_From=tf.keras.callbacks.TensorBoard(log_dir=f_log)
        callbacks_To=tf.keras.callbacks.TensorBoard(log_dir=f_log)
        callbacks_People=tf.keras.callbacks.TensorBoard(log_dir=f_log)
        
        self.From_model.fit(np.array(state_minibatch), np.array(y_f_minibatch),epochs=1, batch_size=minibatch_size,verbose=0,callbacks=callbacks_From)
        self.To_model.fit(np.array(state_minibatch), np.array(y_t_minibatch),epochs=1, batch_size=minibatch_size,verbose=0,callbacks=callbacks_To)
        self.People_model.fit(np.array(state_minibatch), np.array(y_p_minibatch),epochs=1, batch_size=minibatch_size,verbose=0,callbacks=callbacks_People)
        
    def load_model(self, model_path=None):

        json_string = open(os.path.join(f_model, From_model_filename)).read()
        self.From_model = tf.keras.models.model_from_json(json_string)
        self.From_model.load_weights(os.path.join(f_model, From_weights_filename))

        self.From_model.compile(optimizer='rmsprop',loss='mean_squared_error',metrics=['accuracy'])
        
        json_string = open(os.path.join(f_model, To_model_filename)).read()
        self.To_model = tf.keras.models.model_from_json(json_string)
        self.To_model.load_weights(os.path.join(f_model, To_weights_filename))

        self.To_model.compile(optimizer='rmsprop',loss='mean_squared_error',metrics=['accuracy'])

        json_string = open(os.path.join(f_model, People_model_filename)).read()
        self.People_model = tf.keras.models.model_from_json(json_string)
        self.People_model.load_weights(os.path.join(f_model, People_weights_filename))

        self.People_model.compile(optimizer='rmsprop',loss='mean_squared_error',metrics=['accuracy'])


    def save_model(self, num=None):
        json_string = self.From_model.to_json()
        model_name = 'From_model{0}.json'.format((str(num) if num else ''))
        weight_name = 'From_model_weights{0}.hdf5'.format((str(num) if num else ''))
        open(os.path.join(f_model, model_name), 'w').write(json_string)
        self.From_model.save_weights(os.path.join(f_model, weight_name))

        json_string = self.To_model.to_json()
        model_name = 'To_model{0}.json'.format((str(num) if num else ''))
        weight_name = 'To_model_weights{0}.hdf5'.format((str(num) if num else ''))
        open(os.path.join(f_model, model_name), 'w').write(json_string)
        self.To_model.save_weights(os.path.join(f_model, weight_name))

        json_string = self.People_model.to_json()
        model_name = 'People_model{0}.json'.format((str(num) if num else ''))
        weight_name = 'People_model_weights{0}.hdf5'.format((str(num) if num else ''))
        open(os.path.join(f_model, model_name), 'w').write(json_string)
        self.People_model.save_weights(os.path.join(f_model, weight_name))

    
