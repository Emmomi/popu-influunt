import numpy as np

from Agent/Agent import DQNAgent
from env/env import popu


if __name__ == "__main__":

    env=popu()
    rooms=len(env.get_state())
    agent=DQNAgent(env.name,rooms)

    e=0
    epoch=1000

    while e<epoch:
        env.reset()
        flag_t=env.flag
        while not flag_t:
            state_t,reward_t,flag_t=env.observe()
            action_t=agent.select_action(stte_t,agent.exploration)
            env.exe_action(action_t)
            state_t_1,reward_t,flag_t=env.observe()