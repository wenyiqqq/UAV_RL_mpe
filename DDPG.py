import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import rl_utils

from datetime import datetime
import gym
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from Network import *

def make_env(scenario_name):
    # 从环境文件脚本中创建环境
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                        scenario.observation)
    return env

#线性退火算法 用于探索率衰减
def linear_decay(initial_eps , final_eps, decay_steps, current_step):
    # initial_eps = 0.5
    # final_eps = 0.01
    # decay_steps = 10000
    decay_rate = (initial_eps - final_eps) / decay_steps #0.49/10000
    current_eps = max(initial_eps - decay_rate * current_step, final_eps)
    return current_eps


#以下四个函数实现了一些与采样和离散化相关的函数，用于在强化学习中生成离散动作

#接收一个logits张量作为输入，该张量表示动作的概率分布。通过比较概率最大值的索引位置来生成最优动作的独热（one-hot）形式。
def onehot_from_logits(logits, eps = 0.2):  #eps：一个小的浮点数，表示epsilon-贪婪算法中的epsilon值。默认为0.01.
    ''' 生成最优动作的独热（one-hot）形式 '''
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    # 生成随机动作,转换成独热形式
    rand_acs = torch.autograd.Variable(torch.eye(logits.shape[1])[[
        np.random.choice(range(logits.shape[1]), size=logits.shape[0])
    ]],
                                       requires_grad=False).to(logits.device)
    # print("onehot eps = ",eps)
    # 通过epsilon-贪婪算法来选择用哪个动作
    return torch.stack([
        argmax_acs[i] if r > eps else rand_acs[i]  ##通过循环遍历每个样本，并在每次迭代中生成一个随机数r（取值范围为0到1）。
        for i, r in enumerate(torch.rand(logits.shape[0]))   
    ])
#def onehot_from_logits(logits, eps=0.01): 
#    ''' 生成最优动作的独热(one-hot)形式 '''  
#    # argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float() 
#    argmax_acs = (logits == logits.max()).float() 
#    # 生成随机动作,转换成独热形式
#    rand_acs = torch.autograd.Variable(torch.eye(1)[[  #使用torch.eye生成一个单位矩阵，形状为(num_actions, #num_actions)，
#                                                                      #并通过索引选择出随机动作对应的行。
#        np.random.choice(range(1), size=logits.shape[0])
#    ]], requires_grad=False).to(logits.device)
#    # 通过epsilon-贪婪算法来选择用哪个动作
#    return torch.stack([  #通过循环遍历每个样本，并在每次迭代中生成一个随机数r（取值范围为0到1）。
#        argmax_acs if r > eps else rand_acs  
#        for  r in enumerate(torch.rand(logits.shape[0]))
#    ])  #函数返回一个形状为(batch_size, num_actions)的张量，其中每行表示一个样本对应的最优动作的独热形式。

def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor): #shape：采样结果的形状。eps：一个小的浮点数，用于数值稳定性。
    """从Gumbel(0,1)分布中采样"""
    U = torch.autograd.Variable(tens_type(*shape).uniform_(),   #生成一个形状为shape的均匀分布样本U
                                requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)           #生成噪声

def gumbel_softmax_sample(logits, temperature): #logits:一个包含未经过softmax处理的概率分布的张量。temperature：一个正数，用于控制采样的平滑程度。
    """ 从Gumbel-Softmax分布中采样"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(
        logits.device)                      #在logits上加上从Gumbel分布中采样得到的噪声，生成一个新的张量y。
    #print("gumbel_softmax_sample:",y)
    # return F.softmax(y / temperature, dim=1)  #获得Gumbel-Softmax分布的采样结果。
    return F.softmax(y / temperature)


def gumbel_softmax(logits, eps,temperature=1.0):
    """从Gumbel-Softmax分布中采样,并进行离散化"""
    y = gumbel_softmax_sample(logits, temperature)  #从Gumbel-Softmax分布中采样得到一个样本y。
    #print("gumbel_softmax:",y)
    y_hard = onehot_from_logits(y,eps)   #将采样结果离散化为独热形式y_hard
    y = (y_hard.to(logits.device) - y).detach() + y #通过将y_hard转换为与y相同的设备，并将其与y之间的差异（即y_hard - y）与y分离，得到离散化的结果。
    return y  #返回离散动作   可以同时获得与环境交互的离散动作以及正确的梯度反向传播。

                                    

class DDPG(nn.Module):
    ''' DDPG算法 '''
    def __init__(self, state_dim, action_dim, critic_input_dim, hidden_dim,
                 actor_lr, critic_lr, device, gamma, tau):
        super(DDPG,self).__init__()
        # print("actor:",state_dim, action_dim, hidden_dim)
        self.actor = TwoLayerFC(state_dim, action_dim, hidden_dim).to(device)#演员（Actor）网络
        self.target_actor = TwoLayerFC(state_dim, action_dim,#目标演员网络
                                       hidden_dim).to(device)
        # print("critic:",critic_input_dim, hidden_dim)
        self.critic = TwoLayerFC(critic_input_dim, 1, hidden_dim).to(device)#评论家（Critic）网络
        self.target_critic = TwoLayerFC(critic_input_dim, 1,       #目标评论家网络
                                        hidden_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())  #加载目标评论家网络和目标演员网络的初始参数。这是为了保持目标网络与原始网络的参数一致。
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),  #创建了两个优化器对象，分别用于演员网络和评论家网络的参数优化。
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.device = device
        self.gamma = gamma
        self.tau = tau
        
        self.critic_criterion = torch.nn.MSELoss()

    def take_action(self, state, eps, explore=False):  #根据当前状态选择动作。explore：一个布尔值，指示是否进行探索性动作选择。
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        #print("DDPG take_action state:",state) 
        action = self.actor(state)  #通过演员网络self.actor对当前状态进行前向传播，得到动作的概率分布。
        if explore:  #explore为True，使用gumbel_softmax函数对概率分布进行Gumbel-Softmax采样，得到一个离散动作。
            action = gumbel_softmax(action,eps)
        else:        #explore为False，使用onehot_from_logits函数将概率分布转换为独热编码的离散动作。
            action = onehot_from_logits(action,eps)
        # print("take_action -------action:",action)
        # print("action last =",action.detach().cpu().numpy()[0])
        return action.detach().cpu().numpy()[0]  #将离散动作转换为NumPy数组，并返回动作的NumPy表示。

    def soft_update(self, net, target_net, tau):   #软更新目标网络的参数。net：原始网络的对象。target_net：目标网络的对象。tau：软更新的权重。
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()): #通过迭代遍历原始网络和目标网络的参数，并使用指定的权重tau对目标网络的参数进行软更新。
            param_target.data.copy_(param_target.data * (1.0 - tau) +
                                    param.data * tau)
            
    def update(self, sample,eps):
        obs, act, rew, next_obs, done = sample
        #print("obs:",obs[0],"act:",act[0],"rew:",rew[0],"done:",done[0])
        #print("obs:",len(obs),"act:",len(act),"rew:",len(rew),"done:",len(done))
        #obs = torch.tensor(obs, dtype=torch.float).to(self.device)
        ##print(obs.dtype)
        #act = torch.tensor(act, dtype=torch.float).to(self.device)
        ##print(act.dtype)
        #rew = torch.tensor(rew, dtype=torch.float).view(-1, 1).to(self.device)
        #next_obs = torch.tensor(next_obs, dtype=torch.float).to(self.device)
        #done = torch.tensor(done, dtype=torch.float).view(-1, 1).to(self.device)

        target_act = [onehot_from_logits(self.target_actor(_next_obs)) for _next_obs in next_obs]
        #print(next_obs.shape,target_act.shape)
        target_critic_input = torch.cat((*next_obs,*target_act),dim=1)
        target_critic_value = rew[0].view(-1,1) + self.gamma * self.target_critic(target_critic_input) * (1 - done[0].view(-1,1))
        critic_input = torch.cat((*obs, *act), dim=1)
        critic_value = self.critic(critic_input)
        critic_loss = self.critic_criterion(critic_value, target_critic_value.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_out = self.actor(obs[0]) #计算当前演员网络对当前观测值（obs[i_agent]）的输出。
        act_vf_in = [gumbel_softmax(actor_out,eps)] #根据当前代理的演员网络输出构建输入张量vf_in
        # actor_acs = []
        # actor_acs.append(act_vf_in)
        # vf_in = torch.cat((*obs,*actor_acs, dim=1))
        vf_in = torch.cat((*obs,*act_vf_in),dim=1)
        actor_loss = -self.critic(vf_in).mean()
        actor_loss += (actor_out**2).mean() * 1e-3
        actor_loss.backward()
        self.actor_optimizer.step() 

        self.soft_update(self.actor, self.target_actor, self.tau)  
        self.soft_update(self.critic, self.target_critic, self.tau) 

    def save_statedict(self, role):
        current_time = datetime.now().strftime("%Y-%m%d-%H%M")   ## 获取当前时间并格式化为字符串
        critic_filename = f"critic_{role}_{current_time}"
        actor_filename = f"actor_{role}_{current_time}"
        torch.save(self.critic.state_dict(), critic_filename)
        torch.save(self.actor.state_dict(), actor_filename)
        # torch.save(self.critic.state_dict(), 'critic' + '_' + role)
        # torch.save(self.actor.state_dict(), 'actor' + '_' + role)

def evaluate(env_id, agt1, agt2, eps,n_episode=10, episode_length=25):#maddpg表示训练好的MADDPG对象，
                                                            #n_episode表示评估过程中执行的轮数，默认为10，episode_length表示每一轮评估的最大步数，默认为25。
    # 对学习的策略进行评估,此时不会进行探索
    env = make_env(env_id)
    returns = np.zeros(len(env.agents)) 
    for _ in range(n_episode):  #执行n_episode次评估循环进行策略评估
        obs = env.reset()       #将环境重置为初始状态，并将观测结果赋值给obs。
        for t_i in range(episode_length):
            actions1 = agt1.take_action([obs[0]], eps,explore=False)
            actions2 = agt2.take_action([obs[-1]],eps, explore=False)
            all_actions = [actions1, actions2]
            obs, rew, done, info = env.step(all_actions) #将动作应用于环境，得到下一个观测结果obs、奖励rew、完成状态done和其他信息info。
            rew = np.array(rew)    #将奖励转换为NumPy数组形式.
            returns += rew / n_episode   #平均回报,将其累加到returns数组中.
    return returns.tolist()   #将returns数组转换为Python列表形式，并将其作为评估结果返回.





class MADDPG:
    def __init__(self, env, device, actor_lr, critic_lr, hidden_dim,
                 state_dims, action_dims, critic_input_dim, gamma, tau):
        self.agents = []
        # print(len(env.agents))
        for i in range(len(env.agents) - 1): #创建了len(env.agents)个DDPG代理对象，并将其存储在self.agents列表中。每个代理都使用DDPG类进行初始化。
            self.agents.append(
                DDPG(state_dims[i], action_dims[i], critic_input_dim,
                     hidden_dim, actor_lr, critic_lr, device, gamma, tau))
        
        self.gamma = gamma
        self.tau = tau
        self.critic_criterion = torch.nn.MSELoss() #初始化了MSELoss损失函数self.critic_criterion
        self.device = device

    @property
    def policies(self):
        return [agt.actor for agt in self.agents]

    @property
    def target_policies(self):
        return [agt.target_actor for agt in self.agents]   #policies和target_policies是属性方法，用于返回所有代理的演员网络和目标演员网络。

    def take_action(self, states, eps, explore):
        states = [
            torch.tensor([states[i]], dtype=torch.float, device=self.device)  #将状态转换为张量，并存储在states列表中
            for i in range(len(env.agents)-1)
        ]
        return [  #通过迭代遍历代理对象和状态列表，调用每个代理的take_action方法来选择动作，并将动作存储在列表中返回。
            agent.take_action(state, eps, explore)
            for agent, state in zip(self.agents, states)
        ]

    def update(self, sample, i_agent, eps):  #更新代理的演员网络和评论家网络
        obs, act, rew, next_obs, done = sample #一个包含样本的元组，包括观测值（obs）、动作（act）、奖励（rew）、下一个观测值（next_obs）和完成状态（done）。
        # print("obs = ",obs[0].size()," act = ",act[0].size()," rew = ",rew[0].size()," next_obs = ",next_obs[0].size()," done = ",done[0].size())
        cur_agent = self.agents[i_agent] #i_agent：当前代理的索引。根据当前代理的索引获取当前代理的对象cur_agent

        cur_agent.critic_optimizer.zero_grad() #评论家网络梯度清零
        all_target_act = [
            onehot_from_logits(pi(_next_obs))
            for pi, _next_obs in zip(self.target_policies, next_obs)#根据next_obs使用目标策略（target_policies）计算目标动作，将输出转换为独热编码的目标动作
        ]
        target_critic_input = torch.cat((*next_obs, *all_target_act), dim=1) #通过将next_obs和all_target_act拼接成1维向量构建目标评论家网络的输入
        target_critic_value = rew[i_agent].view(  #计算目标评论家网络的值函数估计
            -1, 1) + self.gamma * cur_agent.target_critic(
                target_critic_input) * (1 - done[i_agent].view(-1, 1))
        critic_input = torch.cat((*obs, *act), dim=1)  #将obs和act在维度1上进行拼接来构建当前评论家网络的输入。
        critic_value = cur_agent.critic(critic_input)  #计算当前评论家网络对当前输入的值函数估计。
        critic_loss = self.critic_criterion(critic_value, #计算评论家损失，计算当前评论家网络的值函数估计与目标评论家网络的值函数估计之间的差异。
                                            target_critic_value.detach())
        critic_loss.backward() #反向传播，计算评论家网络的梯度。
        cur_agent.critic_optimizer.step() #根据梯度更新评论家网络的参数。


        cur_agent.actor_optimizer.zero_grad()
        cur_actor_out = cur_agent.actor(obs[i_agent]) #计算当前演员网络对当前观测值（obs[i_agent]）的输出。
        cur_act_vf_in = gumbel_softmax(cur_actor_out, eps) #根据当前代理和其他代理的演员网络输出构建输入张量vf_in
        all_actor_acs = []  #一个空列表，用于存储所有代理的动作
        for i, (pi, _obs) in enumerate(zip(self.policies, obs)):  #遍历self.policies和obs，其中pi表示策略函数，_obs表示观测值。
            if i == i_agent:
                all_actor_acs.append(cur_act_vf_in)
            else:
                all_actor_acs.append(onehot_from_logits(pi(_obs)))#使用对应的策略函数pi和观测值_obs计算动作，并将其转换为独热编码的形式，
                                                                    #然后将结果添加到all_actor_acs中。
        vf_in = torch.cat((*obs, *all_actor_acs), dim=1) #将obs和all_actor_acs在维度1上进行拼接来构建演员网络的输入张量。
        actor_loss = -cur_agent.critic(vf_in).mean()  #计算基于评论家网络的值函数估计，并取均值。
        actor_loss += (cur_actor_out**2).mean() * 1e-3 #添加一个正则化项，该项惩罚演员网络输出的平方的均值乘以一个小的常数（1e-3）。
        actor_loss.backward()
        cur_agent.actor_optimizer.step()


    def save_statedict(self, role):
        current_time = datetime.now().strftime("%Y-%m%d-%H%M")   ## 获取当前时间并格式化为字符串
        critic_filename = f"critic_{role}_{current_time}"
        actor_filename = f"actor_{role}_{current_time}"
        torch.save(self.critic.state_dict(), critic_filename)
        torch.save(self.actor.state_dict(), actor_filename)

    def update_all_targets(self):
        for agt in self.agents[:-1]:  #更新所有目标演员网络和目标评论家网络
            agt.soft_update(agt.actor, agt.target_actor, self.tau)
            agt.soft_update(agt.critic, agt.target_critic, self.tau)


def evaluate1(env_id, agt1,agt2,eps, n_episode=10, episode_length=25):#maddpg表示训练好的MADDPG对象，
                                                            #n_episode表示评估过程中执行的轮数，默认为10，episode_length表示每一轮评估的最大步数，默认为25。
    # 对学习的策略进行评估,此时不会进行探索
    env = make_env(env_id)
    returns = np.zeros(len(env.agents)) #始化一个全零数组returns，长度为智能体的数量（len(env.agents)），用于记录每个智能体在评估过程中的累计回报。
    for _ in range(n_episode):  #执行n_episode次评估循环进行策略评估
        obs = env.reset()       #将环境重置为初始状态，并将观测结果赋值给obs。
        for t_i in range(episode_length):
            action1 = agt1.take_action(obs[:-1], eps, explore=False)
            # print("action1 = ",action1)
            action2 = agt2.take_action([obs[-1]], eps, explore=False)
            # print("action2 = ",action2)
            actions = action1 + [action2]
            # print("evaluate1 actions = ",actions)
            obs, rew, done, info = env.step(actions) #将动作应用于环境，得到下一个观测结果obs、奖励rew、完成状态done和其他信息info。
            rew = np.array(rew)    #将奖励转换为NumPy数组形式.
            returns += rew / n_episode   #平均回报,将其累加到returns数组中.
    return returns.tolist()   #将returns数组转换为Python列表形式，并将其作为评估结果返回.

env_id = "simple_tag"
env = make_env(env_id)
