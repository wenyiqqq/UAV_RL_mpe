import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoLayerFC(torch.nn.Module):  #两层的全连接神经网络，其中有一个隐藏层。
    def __init__(self, num_in, num_out, hidden_dim): #num_in：输入特征的数量。num_out：输出特征的数量。hidden_dim：隐藏层的维度。
        super().__init__()
        self.fc1 = torch.nn.Linear(num_in, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_out)

    def forward(self, x):  #前向传播过程：它接受一个输入张量x作为参数，并返回经过模型前向传播后的输出张量。
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
