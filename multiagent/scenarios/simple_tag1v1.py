import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def __init__(self):
        self.radius = 0.075
    
    def up_radius(self):
        #self.radius = radius
        if self.radius < 0.5:
            self.radius += 1e-5

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_p = 2
        num_good_agents = 1
        num_adversaries = 1
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 2
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 4.0
            #agent.accel = 20.0 if agent.adversary else 25.0
            # agent.max_speed = 1.0 if agent.adversary else 1.3
            agent.max_speed = 1.3 if agent.adversary else 1.0
            # 观测半径
            agent.obs_r = 0.5
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world


    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        self.up_radius()        #生成半径自增
        for agent in world.agents:
            # if agent.adversary != True:
            #     agent.state.p_pos = np.random.uniform(-self.radius, +self.radius, world.dim_p)      #目标
            # else:
            #     agent.state.p_pos = np.random.uniform(-self.radius, +self.radius, world.dim_p)
            agent.state.p_pos = np.random.uniform(-self.radius, self.radius, world.dim_p)
            # agent.state.p_pos = np.random.uniform(-0.95, +0.95, world.dim_p) #设置智能体初始位置，避免智能体初始位置位于边界之外
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                # landmark.state.p_pos = np.array([-0.5 + i, 0.5 - i])
                landmark.state.p_vel = np.zeros(world.dim_p)


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist <= dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]
    
    def obstacles(self,world):   #障碍物
        return [landmark for landmark in world.landmarks ]


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    # def agent_reward(self, agent, world):
    #     # Agents are negatively rewarded if caught by adversaries
    #     rew = 0
    #     shape = False
    #     adversaries = self.adversaries(world)
    #     if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
    #         for adv in adversaries:
    #             rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
    #     if agent.collide:
    #         for a in adversaries:
    #             if self.is_collision(a, agent):
    #                 rew -= 10

    #     # agents are penalized for exiting the screen, so that they can be caught by the adversaries
    #     def bound(x):
    #         if x < 0.9:
    #             return 0
    #         if x < 1.0:
    #             return (x - 0.9) * 10
    #         return min(np.exp(2 * x - 2), 10)
    #     for p in range(world.dim_p):
    #         x = abs(agent.state.p_pos[p])
    #         rew -= bound(x)

    #     return rew

    # def adversary_reward(self, agent, world):
    #     # Adversaries are rewarded for collisions with agents
    #     rew = 0
    #     shape = False
    #     agents = self.good_agents(world)
    #     adversaries = self.adversaries(world)
    #     if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
    #         for adv in adversaries:
    #             rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
    #             # rew -= 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
    #     if agent.collide:
    #         for ag in agents:
    #             for adv in adversaries:
    #                 if self.is_collision(ag, adv):
    #                     rew += 10
    #     return rew



    def agent_reward(self, agent, world):   # 被跟踪的目标的奖励。 
        rew = 0  
        shape = True
        agents = self.good_agents(world)
        adversaries = self.adversaries(world) 
        obstacles = self.obstacles(world)
        if shape:
            for adv in adversaries:
                for a in agents:
                    dis = np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))
                    g_radius = a.size
                    t_radius = adv.size
                    max_dis = 2*(g_radius + t_radius)
                    if dis < max_dis:
                        # rew -= 1    #离散负奖励
                        rew -= 10 * (max_dis - dis)   #连续负奖励                极限每轮最大-2.5
                    else:
                        continue
        if agent.collide:
            for l in obstacles:
                if self.is_collision(l, agent):
                    rew -= 10
         # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):  
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)
        return rew     

    def adversary_reward(self, agent, world):   #追踪器
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = True
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        obstacles = self.obstacles(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                for a in agents:
                    distance = np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))
                    g_radius = a.size
                    t_radius = adv.size
                    max_dis = 2*(g_radius + t_radius)
                    min_dis = 2*g_radius + t_radius
                    if distance > max_dis:
                        rew -= 2 * (distance - max_dis)  # Penalty for exceeding maximum distance   最大每轮-3.32
                        # rew -= 1
                    elif distance < min_dis:
                        rew -= 10 * (min_dis - distance)  # Penalty for being too close   极限最大每轮 -1.75
                        # rew -= 1
                    else:
                        rew += 100 * (max_dis - distance)  #最大每轮  +7.5
                        # rew += 1
        if agent.collide:
            for l in obstacles:
                for adv in adversaries:
                    if self.is_collision(l, adv):
                        rew -= 10
                #检查追踪器和目标连线是否经过障碍物
                    for a in agents:
                        line_dir = adv.state.p_pos - a.state.p_pos
                        line_dir /= np.linalg.norm(line_dir)
                        line_distance = np.abs(np.cross(line_dir, a.state.p_pos - l.state.p_pos))
                        if line_distance < l.size:
                            rew -= 2
        def bound(x):  
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                if np.linalg.norm(entity.state.p_pos - agent.state.p_pos) <= agent.obs_r:
                    entity_pos.append(entity.state.p_pos - agent.state.p_pos)
                else:
                    entity_pos.append([0,0])
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            oth_pos = []
            if np.linalg.norm(other.state.p_pos - agent.state.p_pos) <= agent.obs_r:
                oth_pos = other.state.p_pos - agent.state.p_pos
            else:
                oth_pos = [0,0]
            other_pos.append(oth_pos)
            
            if not other.adversary:
                if np.linalg.norm(other.state.p_pos - agent.state.p_pos) <= agent.obs_r:
                    other_vel.append(other.state.p_vel)
                else:
                    other_vel.append([0,0])
        # print("agent.state.p_vel",agent.state.p_vel,"agent.state.p_pos",agent.state.p_pos)
        # print("entity_pos",entity_pos,"other_pos",other_pos)
        # print("other_vel",other_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)
