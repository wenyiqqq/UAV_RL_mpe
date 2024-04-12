import torch
import numpy as np
from multiagent.environment import MultiAgentEnv
from DDPG import *

# tensor1 = torch.tensor([1.123151,0.15,0.654423,0.15346])
# tensor2 = torch.tensor([[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1]])
# #print(tensor1.gather(0,torch.tensor([3])))
# 
# acition_index=np.array([[tensor2[i].argmax().item()] for i in range(tensor2.shape[0])])
# tensor_index = torch.tensor(acition_index)
# 
# print(tensor2.gather(1,tensor_index))    #size[1,5]



# env_id = "simple_tag"
# env = make_env(env_id)
# for i in range(100):
#         env.reset()
#         for i in range(25):
#             _,reward,_,_ = env.step([[0,0,0,0,1],[1,0,0,0,0]])
#             print(reward)
        

# min_distance = 0.1
# max_distance = 1.3

# prev_coord2 = None

# for i in range(10):  # 假设需要生成 10 对坐标
#     coord1 = np.random.uniform(-0.5, 0.5, 2)

#     while True:
#         distance = min_distance + (max_distance - min_distance) * i / 9  # 根据迭代次数计算距离
#         angle = np.random.uniform(0, 2 * np.pi)
#         x_offset = distance * np.cos(angle)
#         y_offset = distance * np.sin(angle)
#         coord2 = coord1 + np.array([x_offset, y_offset])

#         if np.all(np.abs(coord2) <= 0.5) and (prev_coord2 is None or np.linalg.norm(coord2 - coord1) >= np.linalg.norm(prev_coord2 - coord1)):
#             break
#     print("Iteration:", i+1," pos1 = ",coord1," pos2 = ",coord2,"Dis = ",np.linalg.norm(coord2 - coord1))
#     # print("Coordinate 1:", coord1)
#     # print("Coordinate 2:", coord2)
#     # print("Distance:", np.linalg.norm(coord2 - coord1))


# list = [1,2,3,4,5,6,7,8,9,0]
# for a in list[:-1]:
#     print("a = ",a)
import numpy as np

t_actions = [np.array([0., 0., 0., 0., 1.], dtype=np.float32),
             np.array([0., 1., 0., 0., 0.], dtype=np.float32),
             np.array([0., 0., 0., 0., 1.], dtype=np.float32)]
g_actions = np.array([0., 0., 0., 1., 0.], dtype=np.float32)

# 将g_actions转换为数组
g_actions = np.array(g_actions)

# 拼接t_actions和g_actions
all_action = t_actions + [g_actions]

print(all_action)