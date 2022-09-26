from keras.layers.core import Dense, Flatten
from keras.layers import Lambda, Input, Convolution2D
from keras.models import model_from_yaml, Model
import keras.callbacks
from keras.optimizers import RMSprop
try:
    from keras.optimizers import RMSpropGraves
except:
    print('You do not have RMSpropGraves')

import keras.backend.tensorflow_backend as KTF
from keras import backend as K

from collections import deque
import os
import sys
import copy
from util import clone_model

sys.path.append(os.pardir)
#from env import popu
from models import model as Mdl
import tensorflow as tf
import numpy as np

f_log = './log'
f_model = './models'

model_filename = 'dqn_model.yaml'
weights_filename = 'dqn_model_weights.hdf5'



class DQNAgent:
    def __init__(self,env_name,rooms):
        self.env_name=env_name
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.episord=1000
        self.minibatch_size=32
        self.lerning_rate=0.001
        self.discount_factor=0.9
        self.exploration=0.1
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        self.model_name = "{}.ckpt".format(self.environment_name)

        self.D=deque(maxlen=self.episode)

        self.From_model=Mdl.From_room_model(self.lerning_rate)
        self.To_model=Mdl.To_room_model(self.lerning_rate)
        self.People_model=Mdl.people_model(self.lerning_rate)

        self.current_loss=0.0
    
    
    def select_action(self,state,ep):
        if np.random.rand()<=ep:
            return np.random.randint(len(state)),np.random.randint(len(state)),np.random.rand()  #移動元、移動先、人数の割合
        else:
            return np.argmax(self.From_model.Q_values(state)),np.argmax(self.To_model.Q_values(state)),np.argmax(self.People_model.Q_values(state))
    
    def store_experience(self, state, action, reward, state_1, flag):
        self.D.append((state, action, reward, state_1, flag))

    def experience_replay(self):
        state_minibatch = []
        y_minibatch = []

        # sample random minibatch
        minibatch_size = min(len(self.D), self.minibatch_size)
        minibatch_indexes = np.random.randint(0, len(self.D), minibatch_size)

        for j in minibatch_indexes:
            state_j, action_j, reward_j, state_j_1, terminal = self.D[j]
            action_j_index = self.enable_actions.index(action_j)

            y_j = self.Q_values(state_j)

            if terminal:
                y_j[action_j_index] = reward_j
            else:
                # reward_j + gamma * max_action' Q(state', action')
                y_j[action_j_index] = reward_j + self.discount_factor * np.max(self.Q_values(state_j_1))  # NOQA

            state_minibatch.append(state_j)
            y_minibatch.append(y_j)
        
        self.model.fit(np.array(state_minibatch), np.array(y_minibatch), batch_size=minibatch_size,nb_epoch=1,verbose=0)
    def load_model(self, model_path=None):

        yaml_string = open(os.path.join(f_model, model_filename)).read()
        self.model = model_from_yaml(yaml_string)
        self.model.load_weights(os.path.join(f_model, weights_filename))

        self.model.compile(loss='mean_squared_error',optimizer=RMSProp(lr=self.learning_rate),metrics=['accuracy'])

    def save_model(self, num=None):
        yaml_string = self.model.to_yaml()
        model_name = 'dqn_model{0}.yaml'.format((str(num) if num else ''))
        weight_name = 'dqn_model_weights{0}.hdf5'.format((str(num) if num else ''))
        open(os.path.join(f_model, model_name), 'w').write(yaml_string)
        self.model.save_weights(os.path.join(f_model, weight_name))

    
