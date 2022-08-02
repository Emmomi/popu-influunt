from collections import deque
import os
import sys

sys.path.append(os.pardir)
#from env import popu
import tensorflow as tf
import numpy as np

class DQNAgent:
    def __init__(self,env_name):
        self.env_name=env_name

        self.episord=1000
        self.minibatch_size=32
        self.lerning_rate=0.001
        self.discount_factor=0.9
        self.exploration=0.1
        self.D=deque(maxlen=self.episode)

        self.init_model()

        self.current_loss=0.0
    def init_model(self):
        

    def Q_values(self, state):
        # Q(state, action) of all actions
        return self.sess.run(self.y, feed_dict={self.x: [state]})[0]

    def select_action(self,state,ep):
        if np.random.rand()<=ep:
            return np.random.randint(len(state)),np.random.randint(len(state)),np.random.rand()  #移動元、移動先、人数の割合
        else:
            return 