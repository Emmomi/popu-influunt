import sys
import os
sys.path.append(os.pardir)
from Simulator import Simulator
import tensorflow as tf

class popu:
    def __init__(self):
        self.name = os.path.splitext(os.path.basename(__file__))[0]

        self.reset()
    def get_state(self):
        state=tf.Variable(tf.zeros((self.Simu.rooms(),3),tf.int32))
        for i in range(self.Simu.rooms()):
            state=state[i,0].assign(self.Simu.people(i,'c'))
            state=state[i,1].assign(self.Simu.people(i,'l'))
            state=state[i,2].assign(self.Simu.people(i,'e'))

        #print(state)
        return state
    def check_ex_rooms(self,simu):
        x=0
        for i in range(self.Simu.rooms()):
            if self.Simu.people(i,'e'):
                x+=1
        return x
    def exe_action(self,x_r,y_r,x_p):
        ex_rooms_befor=self.check_ex_rooms(self.Simu)
        self.Simu.transfer(x_r,x_p,y_r)
        ex_rooms_after=self.check_ex_rooms(self.Simu)
        
        if ex_rooms_after<ex_rooms_befor:
            self.reward=1
        else:
            self.reward=-1
        
        if ex_rooms_after==0:
            self.flag=True
            
    def observe(self):
        return self.get_state(),self.reward,self.flag
    def reset(self):
        self.flag=False
        self.reward=0
        self.Simu=Simulator.Simulator('Scripts/Simulator/rooms.json')
