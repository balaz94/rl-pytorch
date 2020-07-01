import gym
import datetime
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.a2c import AgentA2C, Worker
from utils.init import weights_init_xavier
from agents.muzero.atari.pong_wrapper import make_env

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1_representation = nn.Conv2d(4, 64, 3, stride=2, padding=1)
        self.conv2_representation = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv3_representation = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv4_representation = nn.Conv2d(64, 4, 3, stride=2, padding=1)

        self.fc1_prediction_a = nn.Linear(6 * 6 * 4, 256)
        self.fc2_prediction_a = nn.Linear(256, 6)
        self.fc3_prediction_v = nn.Linear(6 * 6 * 4, 256)
        self.fc4_prediction_v = nn.Linear(256, 1)

        self.apply(weights_init_xavier)

    def forward(self, x):
        x = F.relu(self.conv1_representation(x))
        x = F.relu(self.conv2_representation(x))
        x = F.relu(self.conv3_representation(x))
        x = F.relu(self.conv4_representation(x))

        x = x.view(-1, 6 * 6 * 4)
        x_logit = F.relu(self.fc1_prediction_a(x))
        logit = self.fc2_prediction_a(x_logit)
        x_value = F.relu(self.fc3_prediction_v(x))
        value = self.fc4_prediction_v(x_value)
        return logit, value

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def learning(num):
    actions = 6

    agent = AgentA2C(0.99, actions, Net(), 0.001, beta_entropy = 0.001, id=num)

    workers = []
    for id in range(16):
        env = make_env('PongNoFrameskip-v4')
        env.seed(id)
        w = Worker(id, env, agent)
        workers.append(w)

    agent.learn(workers, 16, 200000)

if __name__ == '__main__':
    for i in range(21, 22):
        learning(i)
