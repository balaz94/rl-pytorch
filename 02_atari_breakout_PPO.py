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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.fc1_p = nn.Linear(6 * 6 * 64, 768)
        self.fc2_p = nn.Linear(768, 4)
        self.fc1_v = nn.Linear(6 * 6 * 64, 768)
        self.fc2_v = nn.Linear(768, 1)

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

        x_logit = F.relu(self.fc1_p(x))
        logit = self.fc2_p(x_logit)

        x_value = F.relu(self.fc1_v(x))
        value = self.fc2_v(x_value)

        return logit, value

def learning(num = 1):
    actions = 4

    agent = AgentPPO(0.99, actions, Net(), lr = 2.5e-4, beta_entropy = 0.01, id=num, name='breakout/rewards_in_interval_batch_epochs')

    workers = []
    for id in range(16):
        env = make_env('BreakoutNoFrameskip-v4')
        env.seed(id)
        w = Worker(id, env, agent)
        workers.append(w)

    agent.learn(workers, 32, 10001, 4)

def animation():
    actions = 4

    agent = AgentPPO(0.99, actions, Net(), 0.001, beta_entropy = 0.001, id=0, name='breakout/rewards_in_interval_batch_epochs_1_ppo.pt') #breakout/breakout_1_20000_a2c.pt'
    env = make_env('BreakoutNoFrameskip-v4')
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
    animation()
