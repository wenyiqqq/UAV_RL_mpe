# from pettingzoo.mpe import simple_tag_v3
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from MADDPG import *
import torch


env_id = "simple_tag"
env = make_env(env_id)
model1 = torch.load('actor0.pth')
model2 = torch.load('actor1.pth')


observations = env.reset()
env.render()

while env.agents:
    # this is where you would insert your policy
    # actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    actions = {
        "agent-0": model1.load(observations["agent-0"]),
        "agent-1": model2.load(observations["agent-1"])
    }

    observations, rewards, done, infos = env.step(actions)
    env.render()
env.close()


