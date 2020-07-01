import gym
import datetime
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.alphazero.alphazero import AZAgent, ReplayBuffer
from envs.tictactoe import TicTacToe
from utils.init import weights_init_xavier

class Net(nn.Module):
    def __init__(self, inputs_channels, actions):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(inputs_channels, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.batch_norm4 = nn.BatchNorm2d(64)
        self.features_vector_size = actions # / 2

        self.fc1_a = nn.Linear(self.features_vector_size * 64, 512)
        self.fc2_a = nn.Linear(512, actions)
        self.fc3_v = nn.Linear(self.features_vector_size * 64, 512)
        self.fc4_v = nn.Linear(512, 1)
        self.apply(weights_init_xavier)

    def forward(self, state):
        out = F.relu(self.conv1(state))

        residual = out
        out = self.conv2(out)
        out = self.batch_norm2(out)
        out += residual
        out = F.relu(out)

        residual = out
        out = self.conv3(out)
        out = self.batch_norm3(out)
        out += residual
        out = F.relu(out)

        residual = out
        out = self.conv4(out)
        out = self.batch_norm4(out)
        out += residual
        out = F.relu(out)

        #out = F.avg_pool2d(out, kernel_size=2, stride=2, padding=0)

        x = out.view(-1, self.features_vector_size * 64)
        x_logit = F.relu(self.fc1_a(x))
        logit = self.fc2_a(x_logit)
        x_value = F.relu(self.fc3_v(x))
        value = torch.tanh(self.fc4_v(x_value))
        return logit, value

if __name__ == '__main__':
    width = 6
    height = 6
    frames = 4
    wins_count = 5

    actions = width * height
    env = TicTacToe(width = width, height = height, frames = frames, wins_count = wins_count)
    buffer = ReplayBuffer(5000)
    model = Net(frames * 2, actions)

    agent = AZAgent(env, model, buffer, actions, training_steps = 25000, name = 'az/az_tictactoe', lr = 5e-3)
    agent.run()
