import numpy as np


from Agent import Agent
from env import env


if __name__ == "__main__":

    env=env.popu()
    #print(type(env.get_state()))
    #print(env.get_state())
    rooms=env.get_state().get_shape().as_list()[0]
    agent=Agent.DQNAgent(env.name,rooms)
    people_late=np.arange(1,11)

    e=0
    epoch=100

    print(epoch)

    while e<epoch:
        env.reset()
        i=0
        flag_t=env.flag
        while not flag_t:
            state_t,reward_t,flag_t=env.observe()
            action_t=agent.select_action(state_t,agent.exploration)
            print("e:{}  i:{}".format(e,i))
            print("action:{}   state:{}".format(action_t,state_t))
            env.exe_action(action_t[0],action_t[1],int(state_t[action_t[0],0]*people_late[action_t[2]]/10))
            print(int(state_t[action_t[0],0]*people_late[action_t[2]]/10))
            state_t_1,reward_t,flag_t=env.observe()
            agent.store_experience(state_t,action_t,reward_t,state_t_1,flag_t)
            agent.experience_replay()
            i+=1
            if i>15:
                break
        e+=1


    agent.save_model()
    print("finish!")