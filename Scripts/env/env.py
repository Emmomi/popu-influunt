import sys
import os
sys.path.append(os.pardir)
from Simulator import Simulator

class env:
    def __init__(self):
        self.Simu=Simulator("../Simulator/rooms.json")
        self.reward=0
        self.flag=False
    def check_ex_rooms(self,simu):
        x=0
        for i in range(self.Simu.rooms()):
            if self.Simu.people(i,'e')==True:
                x=x+1
        return x
    def exe_action(self,x_r,y_r,x_p):
        ex_rooms_befor=self.check_ex_rooms(self.Simu)
        self.Simu.transfer(x_r,x_p,y_r)
        ex_rooms_after=self.check_ex_rooms(self.Simu)
        
        if ex_rooms_after<ex_rooms_befor:
            self.reward=self.reward+1
        else:
            self.reward=self.reward-1
    def observe():
        return 