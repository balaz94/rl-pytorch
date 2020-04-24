import numpy as np
from random import randrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mcts import MCTS, Node

class GameData:
    def __init__(self, max_size):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.size = 0
        self.terminal = False

    def add(self, observation, action, reward):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.size += 1

    def get(self, index):
        return self.observations[index], self.actions[index], self.rewards[index]

    def sample(self, history_size, agent):
        start_index = 0
        if self.size > history_size:
            start_index = randrange(self.size - history_size)
        end_index = start_index + history_size
        if self.terminal and self.size == end_index:
            value = 0
        else:
            with torch.no_grad():
                _, value, _ = agent.predict_from_observation(self.observations[end_index])

        values = torch.zeros(history_size)
        for i in reversed(range(0, end_index - start_index)):
            value = self.rewards[i] + agent.gamma * value
            values[i] = value

        return self.observations[start_index], self.actions[start_index:end_index], self.rewards[start_index:end_index], values


class AgentMuZero:
    def __init__(self, model, actions, replay_buffer, gamma = 0.997, simulations = 50, training = True, c1 = 19625, c2 = 1.25, state_size = (6, 6), lr = 0.001):
        self.model = model

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device: ', self.device)

        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.replay_buffer = replay_buffer

        self.gamma = gamma
        self.simulations = simulations
        self.actions = actions
        self.state_size = state_size
        selt.training = training

        self.c1 = c1
        self.c2 = c2

    def create_hidden_state(self, state, action):
        tensor_actions = torch.zeros([self.actions, *state_size], dtype=torch.float64)
        tensor_actions[action] = torch.ones([*state_size], dtype=torch.float64)
        hidden_state = torch.cat(state, tensor_actions)
        return hidden_state

    def choose_action(self, observation):
        with torch.no_grad():
            observation = observation.to(self.device).float()
            logits, value, state = self.predict_from_observation(observation)
            probabilities = F.log_softmax(logits.cpu(), dim=-1)
            root = Node(state.cpu(), probabilities, value.cpu()[0].item())

            mcts = MCTS(root, self.c1, self.c2)

            for simulation in range(self.simulations):
                edge, edges = mcts.selection()
                hidden_state = self.create_hidden_state(edge.node1.state, edge.action)
                logits, value, state, reward = self.predic_from_state(hidden_state.to(self.device).float())
                probabilities = F.log_softmax(logits.cpu(), dim=-1)
                reward, value = reward.cpu(), value.cpu()
                mcts.expansion(edge, state.cpu(), reward[0].item(), probabilities, value[0].item())
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

    def predict_from_observation(self, observation):
        state = self.model.representation(observation)
        logits, value = self.model.prediction(state)
        return logits, value, state

    def predic_from_state(self, state):
        new_state, reward = self.model.dynamics(state)
        logits, value = self.model.prediction(new_state)
        return logits, value, new_state, reward

    def play(self, env, episodes = 1000, memory_sequence_size = 50 , batch_size = 256, learn_step = 25, print_episode = 25, history_size = 5):
        scores = []

        for episode in range(episodes):
            game_data = GameData()
            game_data2 = GameData()

            step = 0
            score = 0
            observation = env.reset()
            terminal = False

            while not terminal:
                action = self.choose_action(observation)
                new_observation, reward, terminal = env.step(action)
                score += reward

                if game_data.size < memory_sequence_size:
                    game_data.add(observation, action, reward)
                else:
                    game_data2.add(observation, action, reward)
                    if game_data2.size == history_size:
                        self.replay_buffer.add(self.replay_buffer.max_priority(), game_data)
                        game_data = game_data2
                        game_data2 = GameData()

                if terminal:
                    if game_data2.size > 0:
                        for i in range(game_data2.size):
                            o, a, r = game_data2.get(i)
                            game_data.add(o, a, r)
                    game_data.terminal = True
                    self.replay_buffer.add(self.replay_buffer.max_priority(), game_data)

                observation = new_observation

                if step % learn_step == 0:
                    self.learn(batch_size, history_size)
                step += 1

            scores.append(score)
            score = 0

            if episode % print_episode == 0:
                avg = np.average(self.average_score[-100:])
                print('episodes: ', episode, '\taverage score: ', avg)

    def learn(self, batch_size, history_size):
        if self.replay_buffer.size < batch_size:
            return

        batch, indexes, is_weight = self.replay_buffer.sample(batch_size)
        is_weight = torch.from_numpy(is_weight)
        observations = []
        actions = torch.zeros([history_size, batch_size], dtype=torch.int32)
        rewards = torch.zeros(history_size, batch_size)
        values = torch.zeros(history_size, batch_size)

        set_indexes = set()
        for i in range(batch_size):
            o, a, r, v = batch.sample(history_size, self)
            observations.append(o)
            set_indexes.add(indexes[i])

            for j in range(history_size):
                actions[j, i] = a[j]
                rewards[j, i] = r[j]
                values[j, i] = v[j]

        observations = torch.stack(observations).to(self.device)

        logits, value, hidden_state = self.predict_from_observation(observations)
        logits, values, hidden_state = logits.cpu(), values.cpu(), hidden_state.cpu()
        states = []
        for state, action in zip(hidden_state, actions[0]):
            states.append(create_hidden_state(state, action))
        states = torch.stack(states).to(self.device)

        log_probs = F.log_softmax(logits, dim=-1)
        log_probs_policy = log_probs.gather(1, actions[0])

        advantage = value - values[0]
        loss = (is_weight * (advantage**2 + log_probs_policy * advantage.detach())).sum()

        priorities = advantage.detach()

        for index in set_indexes:
            sum = 0.0
            count = 0
            for i in range(batch_size):
                if indexes[i] == index:
                    sum += priorities[i]
                    count += 1
            self.replay_buffer.update(index, sum / count)

        for i in range(1, history_size):
            logits, value, hidden_state, reward = self.predic_from_state(states)
            logits, values, hidden_state, reward = logits.cpu(), values.cpu(), hidden_state.cpu(), reward.cpu()

            states = []
            for state, action in zip(hidden_state, actions[i]):
                states.append(create_hidden_state(state, action))
            states = torch.stack(states).to(self.device)

            log_probs = F.log_softmax(logits, dim=-1)
            log_probs_policy = log_probs.gather(1, actions[i])

            advantage = value - values[i]
            reward_loss = reward - rewards[i-1]
            loss += (is_weight * (advangate**2 + log_probs_policy * advantage.detach() + reward_loss**2)).sum()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
        self.optimizer.step()
