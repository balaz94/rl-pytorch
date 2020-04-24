import gym
import datetime
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.a2c import AgentA2C, Worker
from utils.init import weights_init_xavier
from utils.pong_wrapper import make_env

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.fc1 = nn.Linear(5 * 5 * 64, 512)
        self.fc2 = nn.Linear(512, 4)
        self.fc3 = nn.Linear(5 * 5 * 64, 512)
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
        x = x.view(-1, 5 * 5 * 64)

        x_logit = F.relu(self.fc1(x))
        logit = self.fc2(x_logit)
        x_value = F.relu(self.fc3(x))
        value = self.fc4(x_value)
        return logit, value

def reward_function(r):
    r = r / 10.0

    if r < -1.0:
        r = -1.0

    if r > 1.0:
        r = 1.0
    return r


def learning(num = 1):
    actions = 4

    agent = AgentA2C(0.99, actions, Net(), 0.001, beta_entropy = 0.001, id=num, name='breakout/breakout')

    workers = []
    for id in range(8):
        env = make_env('BreakoutNoFrameskip-v4')
        env.seed(id)
        w = Worker(id, env, agent, reward_function = reward_function)
        workers.append(w)

    agent.learn(workers, 100, 100001)

if __name__ == '__main__':
    animation()
