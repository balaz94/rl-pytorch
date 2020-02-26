import gym
import datetime
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.a2c import AgentA2C, Worker
from utils.init import weights_init_xavier

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(8, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 4)
        self.fc4 = nn.Linear(256, 1)

        self.apply(weights_init_xavier)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logit = self.fc3(x)
        value = self.fc4(x)
        return logit, value

def reward_function(reward):
    r = reward / 10.0

    if r < -1.0:
        r = -1.0

    if r > 1.0:
        r = 1.0
    return r

def learning():
    actions = 4
    state_dim = (8, )

    agent = AgentA2C(0.99, actions, Net(), 0.001)

    workers = []
    for id in range(8):
        env = gym.make('LunarLander-v2')
        env.seed(id)
        w = Worker(id, env, agent, reward_function)
        workers.append(w)

    agent.learn(workers, 8, 100000)

if __name__ == '__main__':
    learning()
