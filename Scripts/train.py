import numpy as np


from Agent import Agent
from env import env


if __name__ == "__main__":

    env=env.popu()
    #print(type(env.get_state()))
    #print(env.get_state())
    rooms=3
    agent=Agent.DQNAgent(env.name,rooms)

    e=0
    epoch=1000

    while e<epoch:
        env.reset()
        flag_t=env.flag
        while not flag_t:
            state_t,reward_t,flag_t=env.observe()
            action_t=agent.select_action(state_t,agent.exploration)
            print(action_t)
            env.exe_action(action_t[0],action_t[1],action_t[2])
            state_t_1,reward_t,flag_t=env.observe()
            agent.store_experience(state_t,action_t,reward_t,state_t_1,flag_t)
            agent.experience_replay()


    agent.save_model()
    print("finish!")