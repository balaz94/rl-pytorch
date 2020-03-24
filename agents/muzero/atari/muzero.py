import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mcts import MCTS, Node

class GameData:
    def __init__(self, max_size):
        self.observations = []
        self.actions = torch.zeros(max_size, dtype=torch.int32)
        self.target_rewards = torch.zeros(max_size)
        self.curr_index = 0

    def add(self, observation, action, reward):
        self.observations.append(observation)
        self.actions[self.curr_index] = action
        self.target_rewards[self.curr_index] = reward
        self.curr_index += 1

    def sample(self, history_size):
        if history_size <= self.curr_index:
            self.observations[0], self.actions[0:self.curr_index], self.target_rewards[0:self.curr_index]

class AgentMuZero:
    def __init__(self, model_h, model_g, model_f, actions, replay_buffer, gamma = 0.997, simulations = 50, training = True, c1 = 19625, c2 = 1.25):
        self.model_h = model_h
        self.model_g = model_g
        self.model_f = model_f

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device: ', self.device)

        self.model_h.to(self.device)
        self.model_g.to(self.device)
        self.model_f.to(self.device)

        self.replay_buffer = replay_buffer

        self.gamma = gamma
        self.simulations = simulations
        self.actions = actions
        selt.training = training

        self.c1 = c1
        self.c2 = c2

    def choose_action(self, observation):
        observation = observation.observation(0).to(self.device).float()
        with torch.no_grad():
            state = self.model_h(observation)
            probabilities, value = self.model_f(state)
            root = Node(state, probabilities, value)

            mcts = MCTS(root, self.c1, self.c2)

            for simulation in range(self.simulations):
                edge, edges = mcts.selection()
                state, reward = self.model_g(edge.node1.state)
                probabilities, value = self.model_f(new_state)
                mcts.expansion(edge, state, reward[0].item(), probabilities.cpu(), value[0].item())
                mcts.backup(edges)

            counts = np.zeros(self.actions)
            sum = 0.0
            for edge in rande(mcts.root.edges):
                counts[i] = mcts.root.edges[i].N
                sum += mcts.root.edges[i].N

            if self.training:
                probs = counts / sum
                probs = torch.from_numpy(probs)
                action = probs.multinomial(num_samples=1).detach()
                return action[0].item()
            else:
                return torch.argmax(actions).item()

    def play(self, env, episodes = 1000, memory_sequence_size = 200, batch_size = 256, learn_step = 25, print_episode = 25, history_size = 5):
        game_data = GameData()
        scores = []

        for episode in range(episodes):
            score = 0
            observation = env.reset()
            terminal = False

            while not terminal:
                for step in range(memory_sequence_size):
                    action = self.choose_action(observation)
                    new_observation, reward, terminal = env.step(action)
                    score += reward
                    game_data.add(observation, action, reward)
                    observation = new_observation

                    if step % learn_step == 0:
                        self.learn(batch_size)

                    if terminal:
                        break

                self.replay_buffer.add(self.replay_buffer.max_priority(), game_data)

            scores.append(score)
            score = 0

            if episode % print_episode == 0:
                avg = np.average(self.average_score[-100:])
                print('episodes: ', episode, '\taverage score: ', avg)

    def learn(self, batch_size):
        if self.replay_buffer.size > 10:
            batch, indexes, is_weight = self.replay_buffer.sample(batch_size)
