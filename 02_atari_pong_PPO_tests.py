import gym
import datetime
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.ppo import AgentPPO, Worker
from utils.init import weights_init_xavier
from utils.breakout_wrapper import make_env

class Net256(nn.Module):
    def __init__(self):
        super(Net256, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.fc1 = nn.Linear(6 * 6 * 64, 256)
        self.fc2 = nn.Linear(256, 6)
        self.fc3 = nn.Linear(6 * 6 * 64, 256)
        self.fc4 = nn.Linear(256, 1)

        self.apply(weights_init_xavier)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = x.view(-1, 2304)

        x_logit = F.relu(self.fc1(x))
        logit = self.fc2(x_logit)
        x_value = F.relu(self.fc3(x))
        value = self.fc4(x_value)
        return logit, value
class Net512(nn.Module):
    def __init__(self):
        super(Net512, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.fc1 = nn.Linear(6 * 6 * 64, 512)
        self.fc2 = nn.Linear(512, 6)
        self.fc3 = nn.Linear(6 * 6 * 64, 512)
        self.fc4 = nn.Linear(512, 1)

        self.apply(weights_init_xavier)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = x.view(-1, 2304)

        x_logit = F.relu(self.fc1(x))
        logit = self.fc2(x_logit)
        x_value = F.relu(self.fc3(x))
        value = self.fc4(x_value)
        return logit, value
class Net768(nn.Module):
    def __init__(self):
        super(Net768, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.fc1 = nn.Linear(6 * 6 * 64, 768)
        self.fc2 = nn.Linear(768, 6)
        self.fc3 = nn.Linear(6 * 6 * 64, 768)
        self.fc4 = nn.Linear(768, 1)

        self.apply(weights_init_xavier)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = x.view(-1, 2304)

        x_logit = F.relu(self.fc1(x))
        logit = self.fc2(x_logit)
        x_value = F.relu(self.fc3(x))
        value = self.fc4(x_value)
        return logit, value
class Net1024(nn.Module):
    def __init__(self):
        super(Net1024, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.fc1 = nn.Linear(6 * 6 * 64, 1024)
        self.fc2 = nn.Linear(1024, 6)
        self.fc3 = nn.Linear(6 * 6 * 64, 1024)
        self.fc4 = nn.Linear(1024, 1)

        self.apply(weights_init_xavier)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = x.view(-1, 2304)

        x_logit = F.relu(self.fc1(x))
        logit = self.fc2(x_logit)
        x_value = F.relu(self.fc3(x))
        value = self.fc4(x_value)
        return logit, value
class Net256_2(nn.Module):
    def __init__(self):
        super(Net256_2, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.conv1_v = nn.Conv2d(4, 32, 3, stride=1, padding=1)
        self.conv2_v = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv3_v = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv4_v = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.fc1 = nn.Linear(6 * 6 * 64, 256)
        self.fc2 = nn.Linear(256, 6)
        self.fc1_v = nn.Linear(6 * 6 * 64, 256)
        self.fc2_v = nn.Linear(256, 1)

        self.apply(weights_init_xavier)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = x.view(-1, 2304)

        x_logit = F.relu(self.fc1(x))
        logit = self.fc2(x_logit)

        x_v = F.relu(self.conv1_v(state))
        x_v = F.max_pool2d(x_v, kernel_size=2, stride=2, padding=0)
        x_v = F.relu(self.conv2_v(x_v))
        x_v = F.max_pool2d(x_v, kernel_size=2, stride=2, padding=0)
        x_v = F.relu(self.conv3_v(x_v))
        x_v = F.max_pool2d(x_v, kernel_size=2, stride=2, padding=0)
        x_v = F.relu(self.conv4_v(x_v))
        x_v = F.max_pool2d(x_v, kernel_size=2, stride=2, padding=0)
        x_v = x.view(-1, 2304)

        x_value = F.relu(self.fc1_v(x_v))
        value = self.fc2_v(x_value)
        return logit, value

def learning(num, i):
    actions = 6

    if i == 0:
        net = Net256()
        name = '256'
    elif i == 1:
        net = Net512()
        name = '512'
    elif i == 2:
        net = Net768()
        name = '768'
    elif i == 3:
        net = Net1024()
        name = '1024'
    else:
        net = Net256_2()
        name = '256_2'

    print(name)
    agent = AgentPPO(0.99, actions, net, 0.001, beta_entropy = 0.001, id=num, name='pong/pong' + name)

    workers = []
    for id in range(16):
        env = make_env('PongNoFrameskip-v4')
        env.seed(id)
        w = Worker(id, env, agent)
        workers.append(w)

    agent.learn(workers, 16, 1501, 12)

if __name__ == '__main__':

    for i in range(4, 5):
        learning(i, 4)
    '''
    for i in range(5):
        learning(3, i)
    '''
