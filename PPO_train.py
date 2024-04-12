from tqdm import tqdm
from multiagent.environment import MultiAgentEnv
from PPO import *
import numpy as np


num_episodes = 100000  
# actor_lr = 3e-4
# # critic_lr = 1e-2
# critic_lr = 1e-3
actor_lr = 3e-4
critic_lr = 1e-3
hidden_dim = 64
gamma = 0.98
lmbda = 0.95
epochs = 25
eps = 0.2
device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device(
    "cpu")
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
env_id = "simple_tag1v1"
env = make_env(env_id)


state_dims = []
action_dims = []
for action_space in env.action_space:
    action_dims.append(action_space.n) #获取每个智能体的动作空间，并将动作空间的维度（action_space.n）添加到action_dims列表中。
for state_space in env.observation_space:
    state_dims.append(state_space.shape[0]) #获取每个智能体的观测空间，并将观测空间的维度（state_space.shape[0]）添加到state_dims列表中。
print("state_dims:",state_dims)
print("action_dims:",action_dims)


t_state_dim = []
g_state_dim = []
for i in range(len(state_dims)):
    t_state_dim = state_dims[0]
    g_state_dim = state_dims[-1]

action_dim = 5

t_agent = PPO(t_state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
            epochs, eps, gamma, device)

g_agent = PPO(g_state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
            epochs, eps, gamma, device)

# t_return_list = rl_utils.train_on_policy_agent(env, t_agent, num_episodes)
# g_return_list = rl_utils.train_on_policy_agent(env, g_agent, num_episodes)

return_list = []
t_return_list = []
g_return_list = []
# for i in range(10):
#         with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
#             for i_episode in range(int(num_episodes/10)):
#                 t_episode_return = 0
#                 g_episode_return = 0
#                 t_transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
#                 g_transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
#                 state = env.reset()
#                 # print("reset state = ",state)
#                 count = 0
#                 while count < epochs:
#                     # action = agent.take_action(state)
#                     t_action = t_agent.take_action(state[0])
#                     g_action = g_agent.take_action(state[-1])
#                     #print("t_action:",t_action,"g_action:",g_action)
#                     all_action = [t_action,g_action]
#                     next_state, reward, done, _ = env.step(all_action)

#                     t_transition_dict['states'].append(state[0])
#                     t_transition_dict['actions'].append(all_action[0])
#                     t_transition_dict['next_states'].append(next_state[0])
#                     t_transition_dict['rewards'].append(reward[0])
#                     t_transition_dict['dones'].append(done[0])

#                     g_transition_dict['states'].append(state[-1])
#                     g_transition_dict['actions'].append(all_action[-1])
#                     g_transition_dict['next_states'].append(next_state[-1])
#                     g_transition_dict['rewards'].append(reward[-1])
#                     g_transition_dict['dones'].append(done[-1])
#                     state = next_state
#                     t_episode_return += reward[0]
#                     g_episode_return += reward[-1]
#                     terminal = all(done)
#                     count+=1
#                     #print("total_step:",count)
#                 t_return_list.append(t_episode_return)
#                 g_return_list.append(g_episode_return)
#                 t_agent.update(t_transition_dict)
#                 g_agent.update(g_transition_dict)

#                 return_list = [t_return_list, g_return_list]
#                 if (i_episode+1) % 10 == 0:
#                     pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'target_return': '%.3f' % np.mean(t_return_list[-10:]),
#                                     'goal_return': '%.3f' % np.mean(g_return_list[-10:])})
#                 pbar.update(1)

glist = []
tlist = []
for i_episode in range(num_episodes):

    t_episode_return = 0
    g_episode_return = 0
    t_transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
    g_transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
    state = env.reset()
    #print("reset state = ",state)
    # episode_length = num_episodes / epochs   没有这个等量关系
    # for i in range(epochs):
    count = 0
    while count < epochs:
        t_action = t_agent.take_action(state[0])
        g_action = g_agent.take_action(state[-1])
        #print("t_action:",t_action,"g_action:",g_action)
        all_action = [t_action,g_action]
        next_state, reward, done, _ = env.step(all_action)

        t_transition_dict['states'].append(state[0])
        t_transition_dict['actions'].append(all_action[0])
        t_transition_dict['next_states'].append(next_state[0])
        t_transition_dict['rewards'].append(reward[0])
        t_transition_dict['dones'].append(done[0])

        g_transition_dict['states'].append(state[-1])
        g_transition_dict['actions'].append(all_action[-1])
        g_transition_dict['next_states'].append(next_state[-1])
        g_transition_dict['rewards'].append(reward[-1])
        g_transition_dict['dones'].append(done[-1])
        state = next_state
        # t_episode_return += reward[0]
        # g_episode_return += reward[-1]
        # terminal = all(done)
        count+=1
        #print("total_step:",count)
    t_agent.update(t_transition_dict)
    g_agent.update(g_transition_dict)
    # t_return_list.append(t_episode_return)
    # g_return_list.append(g_episode_return)
# 
    # return_list = [t_return_list, g_return_list]
   
    if (i_episode + 1) % 100 == 0:   #当前训练周期是100的倍数，则调用evaluate函数对当前策略进行评估，并将评估结果添加到return_list中.
        ep_returns = evaluate(env_id, t_agent, g_agent, n_episode=100, episode_length=25)
        tlist.append(ep_returns[0])
        glist.append(ep_returns[-1])
        print(f"Episode: {i_episode+1},Tracker:{ep_returns[0]},Goal:{ep_returns[-1]}") 



role1 = 'tracker-1-26'
role2 = 'goal-1-26'
t_agent.save_statedict(role1)
g_agent.save_statedict(role2)


print("开始绘制奖励曲线...")
# print("tlist = ",tlist)
# print("glist = ",glist)
t_episodes_list = np.arange(len(tlist)) * 100  #追踪器的奖励曲线
# print("t_episodes_list = ",t_episodes_list)
plt.figure()
plt.plot(t_episodes_list, tlist)
# plt.plot(np.arange(len(tlist)) * 100, tlist)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO_tracker-1-26 on {}'.format(env_id)) 
plt.savefig(f"tracker_ppo-1-26.png")


g_episodes_list = np.arange(len(glist)) * 100
# print("g_episodes_list = ",g_episodes_list)
plt.figure()
plt.plot(g_episodes_list, glist)
# plt.plot(np.arange(len(glist)) * 100,  glist)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO_goal-1-26 on {}'.format(env_id)) 
plt.savefig(f"goal_ppo-1-26.png")
print("奖励曲线绘制完毕！！！")
# mv_return = rl_utils.moving_average(t_return_list, 9)
# plt.plot(episodes_list, mv_return)
# plt.xlabel('Episodes')
# plt.ylabel('Returns')
# plt.title('PPO on {}'.format(env_id))
# plt.show()