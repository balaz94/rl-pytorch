import gym
import datetime
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.a2c_lstm import AgentA2C, Worker
from utils.init import weights_init_xavier
from utils.pong_wrapper import make_env

'''
This code is not working. Editing in progress.
'''

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.lstm = nn.LSTMCell(32 * 5 * 5, 256)

        self.fc1 = nn.Linear(256, 6)
        self.fc2 = nn.Linear(256, 1)

        self.apply(weights_init_xavier)

    def forward(self, x, hx, cx):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        x = x.view(-1, 32 * 5 * 5)

        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        logit = self.fc1(x)
        value = self.fc2(x)
        return logit, value, hx, cx

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def learning(num):
    actions = 6

    agent = AgentA2C(0.99, actions, Net(), 0.001, beta_entropy = 0.001, id=num, name='pong/pong_lstm')

    workers = []
    for id in range(1):
        env = make_env('PongNoFrameskip-v4')
        env.seed(id)
        w = Worker(id, env, agent)
        workers.append(w)

    agent.learn(workers, 16, 20000)

if __name__ == '__main__':
    for i in range(1):
        learning(i)
