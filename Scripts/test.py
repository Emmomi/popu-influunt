

from Agent import Agent
from env import env


if __name__ == "__main__":

    env=env.popu()
    rooms=len(env.get_state())
    agent=Agent.DQNAgent(env.name,rooms)
    agent.load_model()

    
    env.reset()
    flag_t=env.flag
    while not flag_t:
        state_t,reward_t,flag_t=env.observe()
        action_t=agent.select_action(state_t,agent.exploration)
        env.exe_action(action_t[0],action_t[1],action_t[2])
        
        print(action_t)
        print(env.observe())

    print("finish!")
        
