import copy
import numpy as np
from random import randrange
import torch
import torch.nn as nn
import torch.optim as optim

class AgentDQN:
    def __init__(self, gamma, actions_count, model, experience_replay, lr,
                 update_steps = 1000, batch_size = 64, path = 'path',
                 epsilon=1.0, epsilon_dec = 1e-4, epsilon_min = 0.01):

        self.gamma = gamma
        self.actions_count = actions_count
        self.online_model = model
        self.target_model = copy.deepcopy(model)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device: ', self.device)
        self.online_model.to(self.device)
        self.target_model.to(self.device)
        for param in self.target_model.parameters():
            param.requires_grad = False
        self.mse = nn.MSELoss()
        self.experience_replay = experience_replay
        self.optimizer = optim.Adam(self.online_model.parameters(), lr=lr)
        self.update_steps = update_steps
        self.current_steps = 0
        self.batch_size = batch_size
        self.path = path
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec

    def choose_action(self, state):
        r = np.random.random()
        with torch.no_grad():
            state = state.unsqueeze(0).to(self.device).float()
            actions = self.online_model(state).cpu()

        if r < self.epsilon:
            action = randrange(self.actions_count)
        else:
            action = torch.argmax(actions).item()
        return action, actions

    def store(self, priority, state, action, reward, state_, terminal):
        self.experience_replay.store(priority, state, action, reward, state_, terminal)

    def learn(self):
        if self.experience_replay.size < 10:
            return

        self.optimizer.zero_grad()
        states, actions, rewards, states_, terminals, indexes, is_weight = self.experience_replay.sample(self.batch_size)

        q_y = self.online_model(states.to(self.device)).cpu()
        q_next = self.target_model(states_.to(self.device)).cpu()
        loss = torch.tensor(0.0).float()
        #print(is_weight)

        for i in range(0, len(states)):
            target = rewards[i] + self.gamma * torch.max(q_next[i]) * (1 - terminals[i])
            error = target - q_y[i, actions[i]]
            self.experience_replay.update(indexes[i], error.detach().item())
            error = is_weight[i] * error**2
            loss += error

        print('loss ', loss.detach().item(), 'max priority', self.experience_replay.tree.max_priority(), 'size', self.experience_replay.size)
        loss.backward()
        self.optimizer.step()

        self.current_steps += 1
        if self.current_steps == self.update_steps:
            self.target_model.load_state_dict(self.online_model.state_dict())
            self.current_steps = 0

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon - self.epsilon_dec, self.epsilon_min)
