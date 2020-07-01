import gym
import datetime
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.a2c import AgentA2C, Worker
from utils.init import weights_init_xavier
from utils.breakout_wrapper import make_env

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 32, 3, stride=1, padding=1)

        self.fc1 = nn.Linear(6 * 6 * 32, 256)
        self.fc2 = nn.Linear(256, 6)
        self.fc3 = nn.Linear(6 * 6 * 32, 256)
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
        x = x.view(-1, 1152)

        x_logit = F.relu(self.fc1(x))
        logit = self.fc2(x_logit)
        x_value = F.relu(self.fc3(x))
        value = self.fc4(x_value)
        return logit, value


def learning(num = 1):
    actions = 6

    agent = AgentA2C(0.99, actions, Net(), 0.001, beta_entropy = 0.001, id=num, name='pong/pong_a2c_2')

    workers = []
    for id in range(16):
        env = make_env('PongNoFrameskip-v4')
        env.seed(id)
        w = Worker(id, env, agent)
        workers.append(w)

    agent.learn(workers, 32, 100001)

def animation():
    actions = 4

    agent = AgentPPO(0.99, actions, Net(), 0.001, beta_entropy = 0.001, id=0, name='breakout/breakout_ppo.pt') #breakout/breakout_1_20000_a2c.pt'
    env = make_env('PongNoFrameskip-v4')
    agent.load_model()

    while True:
        terminal = False
        observation = env.reset()

        while not terminal:
            env.render()
            time.sleep(0.02)
            observation = torch.from_numpy(observation)
            action = agent.choose_action(observation)
            observation, _, terminal, _ = env.step(action)

    env.close()

if __name__ == '__main__':
    learning()
