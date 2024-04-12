import os
from multiagent.environment import MultiAgentEnv
from DDPG import *


 
num_episodes = 50000   #训练的总轮数，即循环的次数。
episode_length = 25   #每个训练轮次的步数，即每个轮次中与环境交互的最大步数。
buffer_size = 100000
hidden_dim = 64    #演员和评论家网络的隐藏层的维度。
# actor_lr = 1e-2
actor_lr = 3e-3
critic_lr = 1e-2
g_actor_lr = 1e-3
g_critic_lr = 1e-2
gamma = 0.95  #折扣因子，用于计算未来奖励的折扣累计。
tau = 1e-2     #更新目标网络参数的权重，用于平滑地更新目标网络。
batch_size = 1024      #每次从经验回放缓冲区中抽样的样本数量，用于训练演员和评论家网络。
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
update_interval = 100   #定义了多少步进行一次网络参数更新。
minimal_size = 4000      #定义了经验回放缓冲区中需要的最小样本数量，达到该数量后才开始进行网络参数更新。

initial_eps = 0.5
final_eps = 0.01
decay_steps = 10000

g_replay_buffer = rl_utils.ReplayBuffer(buffer_size)
t_replay_buffer = rl_utils.ReplayBuffer(buffer_size)
env_id = "simple_tag1v1"
env = make_env(env_id)


state_dims = []
action_dims = []
for action_space in env.action_space:
    action_dims.append(action_space.n) #获取每个智能体的动作空间，并将动作空间的维度（action_space.n）添加到action_dims列表中。
for state_space in env.observation_space:
    state_dims.append(state_space.shape[0]) #获取每个智能体的观测空间，并将观测空间的维度（state_space.shape[0]）添加到state_dims列表中。
g_critic_input_dim = state_dims[-1] + action_dims[-1]
t_critic_input_dim = state_dims[0] + action_dims[0]
print("state_dims:",state_dims)
print("action_dims:",action_dims)
 
# t_state_dim = state_dims[0]
# g_state_dim = state_dims[-1]
t_state_dim = 12
g_state_dim = 10
t_action_dim = action_dims[0]
g_action_dim = action_dims[-1]


# for i in range(len(state_dims)):
#     t_state_dim = state_dims[0]
#     g_state_dim = state_dims[-1]

# for j in range(len(action_dims)):
#     t_action_dim = action_dims[0]
#     g_action_dim = action_dims[-1]
#critic_input_dim是评论家网络输入的维度  #tau是软更新的目标网络参数的权重

tracker = DDPG(t_state_dim, t_action_dim, t_critic_input_dim, hidden_dim, actor_lr, critic_lr, device, gamma, tau)
goal = DDPG(g_state_dim, g_action_dim, g_critic_input_dim, hidden_dim, g_actor_lr, g_critic_lr, device, gamma, tau)

t_return_list = []  # 记录tracker每一轮的回报（return）
g_return_list = []
total_step = 0
for i_episode in range(num_episodes):  #迭代num_episodes次进行训练。
    state = env.reset()                #将环境重置为初始状态，并将状态赋值给state变量。
    # print("reset后,state = ",state)
    # ep_returns = np.zeros(len(env.agents))
    n_eps = linear_decay(initial_eps,final_eps,decay_steps,i_episode+1)

    for e_i in range(episode_length):  
        g_actions= goal.take_action([state[-1]], n_eps,explore=True)     
        t_actions = tracker.take_action([state[0]],n_eps, explore=True)
        all_action = [t_actions, g_actions]
        # print("g_actions: ",g_actions,"t_actions: ",t_actions)
        all_next_state, reward, done, _ = env.step(all_action)    #将动作应用于环境
        # print("all_action:",all_action)
        # print(reward)

        g_next_state = all_next_state[-1]
        t_next_state = all_next_state[0]
        
        t_replay_buffer.add(state[0], t_actions, reward[0], t_next_state, done[0])
        g_replay_buffer.add(state[-1], g_actions, reward[-1], g_next_state, done[-1])  #用于后续的经验回放。
        state = all_next_state

        total_step += 1
        # t_return_list.append(reward[0])
        # g_return_list.append(reward[-1])
        if g_replay_buffer.size( #每次执行动作后，增加total_step计数器的值，并检查重放缓冲区是否达到了最小大小（minimal_size）
                                  #并且total_step是否达到了更新间隔（update_interval）.
        ) >= minimal_size and t_replay_buffer.size() >=minimal_size and total_step % update_interval == 0:
            g_sample = g_replay_buffer.sample(batch_size)
            t_sample = t_replay_buffer.sample(batch_size)
                                                    #满足条件，则从重放缓冲区中抽样一批数据，并将其处理为适合训练的形式。
            def stack_array(x):   
                rearranged = [x[i] for i in range(len(x))]
                return [
                    torch.FloatTensor(rearranged).to(device)
                    #for aa in rearranged
                ]

            g_sample = [stack_array(x) for x in g_sample]
            t_sample = [stack_array(x) for x in t_sample]
            # for a_i in range(len(env.agents)):
            #     maddpg.update(sample, a_i)         #对于每个智能体，更新演员和评论家网络的参数。
            # maddpg.update_all_targets()   #更新所有目标网络的参数.
            goal.update(g_sample,n_eps)
            tracker.update(t_sample,n_eps)
            
    if (i_episode + 1) % 100 == 0:   #当前训练周期是100的倍数，则调用evaluate函数对当前策略进行评估，并将评估结果添加到return_list中.
        ep_returns = evaluate(env_id, tracker,goal,n_eps, n_episode=100,episode_length=25)
        t_return_list.append(ep_returns[0])
        # 打印当前训练周期的编号和评估结果
        g_return_list.append(ep_returns[-1])
        print(f"Episode: {i_episode+1},Tracker:{ep_returns[0]},Goal:{ep_returns[-1]}") 


#print("Save critics and actors net parameters!")
role1 = 'tracker1-25(50000)'
role2 = 'goal1-25(50000)'
tracker.save_statedict(role1)
goal.save_statedict(role2)

   

# return_list = [t_return_list ,g_return_list]



print("开始绘制奖励曲线...")
plt.figure()
#np.arange(len(t_return_list)) * 100 生成一个 x 轴上的时间步数组，每个时间步的间隔为100，t_return_list 是存储智能体每个回合奖励的列表。
plt.plot(
       np.arange(len(t_return_list)) * 100,    
       t_return_list)
plt.xlabel("Episodes")
plt.ylabel("Returns")
plt.title(f"tracker_1-25 by 50000-DDPG")
plt.savefig(f"tracker_1-25_plot-50000.png")


plt.figure()
plt.plot(
       np.arange(len(g_return_list)) * 100,
       g_return_list)
plt.xlabel("Episodes")
plt.ylabel("Returns")
plt.title(f"goal_1-25 by 50000-DDPG")
plt.savefig(f"goal__1-25_plot-50000.png")
# return_array = np.array(return_list)
# for i, agent_name in enumerate(["agent_track1-18","agent_good1-18"]):
#    plt.figure()
#    plt.plot(
#        np.arange(return_array.shape[0]) * 100,
#        rl_utils.moving_average(return_array[:, i], 9))
#    plt.xlabel("Episodes")
#    plt.ylabel("Returns")
#    plt.title(f"{agent_name} by MADDPG")
#    plt.savefig(f"{agent_name}_plot.png")
print("奖励曲线绘制完毕！！！")