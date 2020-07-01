import gym
import datetime
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.muzero.atari.muzero2 import AgentMuZero, ExperienceReplay
from agents.muzero.atari.pong_wrapper import make_env
from utils.init import weights_init_xavier

class Net(nn.Module):
    def __init__(self, actions):
        super(Net, self).__init__()

        self.conv1_representation = nn.Conv2d(4, 64, 3, stride=2, padding=1)
        self.conv2_representation = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv3_representation = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv4_representation = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.fc1_prediction_a = nn.Linear(6 * 6 * 64, 512)
        self.fc2_prediction_a = nn.Linear(512, actions)
        self.fc3_prediction_v = nn.Linear(6 * 6 * 64, 512)
        self.fc4_prediction_v = nn.Linear(512, 1)

        self.conv1_dynamics = nn.Conv2d(70, 64, 3, stride=1, padding=1)
        self.conv2_dynamics = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv3_dynamics = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv4_dynamics = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.fc1_dynamics_r = nn.Linear(6 * 6 * 64, 512)
        self.fc2_dynamics_r = nn.Linear(512, 1)

        self.apply(weights_init_xavier)

    def representation(self, state):
        out = F.relu(self.conv1_representation(state))
        out = F.relu(self.conv2_representation(out))

        residual = out
        out = self.conv3_representation(out)
        out += residual
        out = F.relu(out)
        out = F.avg_pool2d(out, kernel_size=2, stride=2, padding=0)

        residual = out
        out = self.conv4_representation(out)
        out += residual
        out = F.relu(out)
        out = F.avg_pool2d(out, kernel_size=2, stride=2, padding=0)

        return out

    def prediction(self, state):
        x = state.view(-1, 6 * 6 * 64)
        x_logit = F.relu(self.fc1_prediction_a(x))
        logit = self.fc2_prediction_a(x_logit)
        x_value = F.relu(self.fc3_prediction_v(x))
        value = self.fc4_prediction_v(x_value)
        return logit, value

    def dynamics(self, state):
        out = F.relu(self.conv1_dynamics(state))

        residual = out
        out = self.conv2_dynamics(out)
        out += residual
        out = F.relu(out)

        residual = out
        out = self.conv3_dynamics(out)
        out += residual
        out = F.relu(out)

        residual = out
        out = self.conv4_dynamics(out)
        out += residual
        out = F.relu(out)

        x_reward = out.view(-1, 6 * 6 * 64)
        x_reward = F.relu(self.fc1_dynamics_r(x_reward))
        reward = self.fc2_dynamics_r(x_reward)
        return out, reward

def reward_function(r):
    r = r / 10.0

    if r < -1.0:
        r = -1.0

    if r > 1.0:
        r = 1.0
    return r


def learning():
    actions = 6
    buffer = ExperienceReplay(1000)

    agent = AgentMuZero(Net(actions), actions, buffer)
    env = make_env('PongNoFrameskip-v4')
    agent.play(env)

if __name__ == '__main__':
    learning()
