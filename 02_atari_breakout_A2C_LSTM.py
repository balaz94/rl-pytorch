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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.lstm = nn.LSTMCell(32 * 3 * 3, 256)

        self.fc1 = nn.Linear(256, 4)
        self.fc2 = nn.Linear(256, 1)

        self.apply(weights_init_xavier)

    def forward(self, inputs):
        inputs, hx, cx = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        x = x.view(-1, 32 * 3 * 3)

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
    actions = 4

    agent = AgentA2C(0.99, actions, Net(), 0.001, beta_entropy = 0.001, id=num, name='breakout/breakout')
    agent.load_model()

    workers = []
    for id in range(16):
        env = make_env('BreakoutNoFrameskip-v4')
        env.seed(id)
        w = Worker(id, env, agent)
        workers.append(w)

    agent.learn(workers, 16, 300001)

def animation(num):
    actions = 4

    agent = AgentA2C(0.99, actions, Net(), 0.001, beta_entropy = 0.001, id=num, name='breakout/breakout')
    env = make_env('BreakoutNoFrameskip-v4')
    agent.load_model()

    while True:
        terminal = False
        observation = env.reset()

        while not terminal:
            env.render()
            time.sleep(0.01)
            observation = torch.from_numpy(observation)
            action = agent.choose_action(observation)
            observation, _, terminal, _ = env.step(action)

    env.close()

if __name__ == '__main__':
    learning(1)
