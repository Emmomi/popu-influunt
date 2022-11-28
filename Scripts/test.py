

from Agent import Agent
from env import env
import numpy as np


if __name__ == "__main__":

    env=env.popu()
    rooms=rooms=env.get_state().get_shape().as_list()[0]
    agent=Agent.DQNAgent(env.name,rooms)
    agent.load_model()
    people_action=np.arange(0.1,1.1,0.1)

    
    env.reset()
    flag_t=env.flag
    while not flag_t:
        state_t,reward_t,flag_t=env.observe()
        action_t=agent.select_action(state_t,agent.exploration)
        env.exe_action(action_t[0],action_t[1],int(state_t[action_t[0],0]*people_action[action_t[2]]))
        
        print(action_t)
        print(env.observe())
        #print(state_t.get_shape().as_list())
        state_t,reward_t,flag_t=env.observe()

    print("finish!")
        
