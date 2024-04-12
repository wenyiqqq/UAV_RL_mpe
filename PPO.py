import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils


from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from PPO_net import *

def make_env(scenario_name):
    # 从环境文件脚本中创建环境
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                        scenario.observation)
    return env


def evaluate(env_id, agt1, agt2, n_episode, episode_length):
                                                            #n_episode表示评估过程中执行的轮数，默认为10，episode_length表示每一轮评估的最大步数，默认为25。
    # 对学习的策略进行评估,此时不会进行探索
    env = make_env(env_id)
    returns = np.zeros(len(env.agents)) 
    for _ in range(n_episode):  #执行n_episode次评估循环进行策略评估
        obs = env.reset()       #将环境重置为初始状态，并将观测结果赋值给obs。
        for t_i in range(episode_length):
            actions1 = agt1.take_action([obs[0]])
            actions2 = agt2.take_action([obs[-1]])
            all_actions = [actions1, actions2]
            # print("all_actions :",all_actions)
            obs, rew, done, info = env.step(all_actions) #将动作应用于环境，得到下一个观测结果obs、奖励rew、完成状态done和其他信息info。
            rew = np.array(rew)    #将奖励转换为NumPy数组形式.
            returns += rew / n_episode   #平均回报,将其累加到returns数组中.
    return returns.tolist()   #将returns数组转换为Python列表形式，并将其作为评估结果返回.



class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    # def take_action(self, state):
    #     state = torch.tensor([state], dtype=torch.float).to(self.device) #将输入的状态转换为torch.tensor对象，并将其发送到指定的设备（例如GPU）上。
    #     probs = self.actor(state)  #使用策略网络（actor）对当前状态进行前向传播，得到动作的概率分布
    #     action_dist = torch.distributions.Categorical(probs) #创建一个离散分布对应的概率分布对象。probs是动作的概率分布，它表示每个可能动作的概率。
    #     action = action_dist.sample() #从动作的概率分布中采样一个动作。
    #     return action.item()  #返回采样的动作值。通过.item()方法获取action作为Python标量的值
    
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        #print("probs:",probs)
        action_dist = torch.distributions.Categorical(probs)
        #print("action_dist:",action_dist)
        action = action_dist.sample()
        action = action.item()
        #print("action:",action)
        # return action
        one_hot_action = torch.eye(5)[action].squeeze()  # 将动作转换为one-hot编码
        # action_np = one_hot_action.numpy()  # 转换为NumPy数组
        action_np = one_hot_action.detach().cpu().numpy()
        #print("action_np = ",action_np)
        return action_np  # 返回动作的NumPy表示

    def update(self, transition_dict):
        #print("action_len",len(transition_dict['actions']))
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).to(
            self.device)
        #print("-------------------actions = ",actions,"actions.size:",actions.size())
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones) #TD 目标，使用值函数估计的下一个状态的值和折扣因子 gamma 进行加权
        td_delta = td_target - self.critic(states)  #TD 误差，即 TD 目标与当前状态值函数估计之间的差异。
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        
        acition_index=np.array([[actions[i].argmax().item()] for i in range(actions.shape[0])])
        #print("acition_index:",acition_index)
        tensor_index = torch.tensor(acition_index).to(self.device)
        #print("tensor_index:",tensor_index)
        old_log_probs = torch.log(self.actor(states).gather(1,tensor_index)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, tensor_index)) 
            ratio = torch.exp(log_probs - old_log_probs)   #计算动作概率的比值，即新动作概率与旧动作概率的比值。
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断   #计算第二项 surrogate loss，即比值在一个范围内进行截断后乘以优势函数。
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数   #计算 actor 的损失函数，即 surrogate loss 的最小值的负平均值。
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach())) #计算 critic 的损失函数，使用均方误差损失函数（F.mse_loss）来比较值函数估计和 TD 目标的差异。
            self.actor_optimizer.zero_grad()  #清零 actor 模型的梯度。
            self.critic_optimizer.zero_grad()
            actor_loss.backward()   #对 actor 损失进行反向传播，计算梯度。
            critic_loss.backward()
            self.actor_optimizer.step()  #更新actor模型的参数，根据梯度和优化算法进行参数更新。
            self.critic_optimizer.step()

    def save_statedict(self, role):
        # print("len(self.agents) = ",len(self.agents))
        torch.save(self.critic.state_dict(), 'PPO_critic' + role)
        torch.save(self.actor.state_dict(), 'PPO_actor' + role)

    def load_actor(self, actor_path):
        self.actor.load_state_dict(torch.load(actor_path))

    def load_actor_cpu(self, actor_path):
        self.actor.load_state_dict(torch.load(actor_path,map_location=torch.device('cpu')))

    def load_critic(self, critic_path):
        self.critic.load_state_dict(torch.load(critic_path))



