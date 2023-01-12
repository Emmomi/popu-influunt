import numpy as np
import matplotlib.pyplot as plt

from Agent import Agent
from env import env


if __name__ == "__main__":

    e=0
    epoch=500

    env=env.popu()
    #print(type(env.get_state()))
    #print(env.get_state())
    rooms=env.get_state().get_shape().as_list()[0]
    agent=Agent.DQNAgent(env.name,rooms,epoch)
    people_late=np.arange(1,11)

    t=np.arange(epoch)
    loss_f=np.arange(epoch)
    acc_f=np.arange(epoch)
    loss_t=np.arange(epoch)
    acc_t=np.arange(epoch)
    loss_p=np.arange(epoch)
    acc_p=np.arange(epoch)

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
            acc_f[e],loss_f[e],acc_t[e],loss_t[e],acc_p[e],loss_p[e]=agent.experience_replay()
            i+=1
            if i>15:
                break
        e+=1


    agent.save_model()

    fig=plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax1.plot(t, acc_f, 'bo' ,label = 'from_model acc')
    ax1.plot(t, acc_t, 'b+' , label= 'to_model acc')
    ax1.plot(t, acc_p, 'b*' , label= 'people_model acc')
    ax1.legend()

    ax2.plot(t, loss_f, 'bo' ,label = 'from_model loss',)
    ax2.plot(t, loss_t, 'b+' , label= 'to_model loss',)
    ax2.plot(t, loss_p, 'b*' , label= 'people_model loss',)
    ax2.legend()


    #plt.title('models loss and acc')
    fig.show()
    fig.savefig('graph.png')

    print("finish!")