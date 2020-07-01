import numpy as np
from random import randrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from agents.muzero.atari.mcts import MCTS, Node

class GameData:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.probs = []
        self.q = []
        self.rewards = []
        self.size = 0
        self.terminal = False

    def add(self, observation, actions, probs, reward, q):
        self.observations.append(observation)
        self.actions.append(actions)
        self.probs.append(probs)
        self.rewards.append(reward)
        self.q.append(q)
        self.size += 1

    def get(self, index):
        return self.observations[index], self.actions[index], self.probs[index], self.rewards[index], self.q[index]

    def sample(self, history_size, agent):
        start_index = 0
        if self.size > history_size:
            start_index = randrange(self.size - history_size - 1)
        end_index = start_index + history_size

        if self.terminal and self.size == end_index + 1:
            value = 0
        else:
            value = self.q[end_index]

        values = torch.zeros(history_size)
        for i in reversed(range(0, end_index - start_index)):
            value = self.rewards[i] + agent.gamma * value
            values[i] = value

        return self.observations[start_index], self.actions[start_index:end_index], self.probs[start_index:end_index], self.rewards[start_index:end_index], values

class ExperienceReplay:
    def __init__(self, size):
        self.size = size
        self.data = self.data = np.zeros(size, dtype=object)
        self.index = 0

    def add(self, data):
        self.data[self.index % self.size] = data
        self.index += 1
        #print(self.index)

    def sample(self, batch_size):
        length = min(self.size, self.index)
        batch = np.random.choice(length, batch_size)
        return self.data[batch]

def reward_func(r):
    return r

class AgentMuZero:
    def __init__(self, model, actions, replay_buffer, gamma = 0.997, simulations = 50, training = True, c1 = 19625, c2 = 1.25, lr = 0.001):
        self.model = model

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device: ', self.device)

        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1)

        self.replay_buffer = replay_buffer

        self.gamma = gamma
        self.simulations = simulations
        self.actions = actions
        self.training = training

        self.c1 = c1
        self.c2 = c2

    def create_hidden_state(self, state, action):
        tensor_actions = torch.zeros([self.actions, 6, 6], dtype=torch.float32)
        tensor_actions[action] = torch.ones([6, 6], dtype=torch.float32)
        hidden_state = torch.cat((state, tensor_actions))
        return hidden_state

    def choose_action(self, observation):
        with torch.no_grad():
            observation = observation.unsqueeze(0).to(self.device).float()
            logits, value, state = self.predict_from_observation(observation)
            probabilities = F.softmax(logits.cpu(), dim=-1)
            value = value.cpu()
            state = state.cpu()
            root = Node(state[0], probabilities[0], value.item())

            mcts = MCTS(root, self.c1, self.c2, self.gamma)

            for simulation in range(self.simulations):
                edge, edges = mcts.selection()
                hidden_state = self.create_hidden_state(edge.node1.state, edge.action)
                logits, value, state, reward = self.predic_from_state(hidden_state.unsqueeze(0).to(self.device).float())
                probabilities = F.softmax(logits.cpu(), dim=-1)
                state, reward, value = state.cpu(), reward.cpu(), value.cpu()
                mcts.expansion(edge, state[0], reward.item(), probabilities[0], value.item())
                mcts.backup(edges)

            counts = torch.zeros(self.actions)
            sum = 0.0
            max_q = -10000
            for i in range(len(mcts.root.edges)):
                counts[i] = mcts.root.edges[i].N
                sum += mcts.root.edges[i].N
                max_q = max(max_q, mcts.root.edges[i].Q)

            probs = counts / sum
            if self.training:
                print(probs, root.V, root.P)
                action = probs.multinomial(num_samples=1).detach()
                return action[0].item(), probs, max_q
            else:
                return torch.argmax(actions).item(), probs, max_q

    def predict_from_observation(self, observation):
        state = self.model.representation(observation)
        logits, value = self.model.prediction(state)
        return logits, value, state

    def predic_from_state(self, state):
        new_state, reward = self.model.dynamics(state)
        logits, value = self.model.prediction(new_state)
        return logits, value, new_state, reward

    def play(self, env, episodes = 10000, memory_sequence_size = 50 , batch_size = 256, learn_step = 10, print_episode = 1, history_size = 5, reward_function = reward_func):
        scores = []

        for episode in range(episodes):
            game_data = GameData()
            game_data2 = GameData()

            step = 0
            score = 0
            observation = torch.from_numpy(env.reset())
            terminal = False

            while not terminal:
                action, probs, max_q = self.choose_action(observation)
                new_observation, reward, terminal, _ = env.step(action)
                #print(new_observation[0])
                new_observation = torch.from_numpy(new_observation)
                score += reward
                reward = reward_function(reward)

                if game_data.size < memory_sequence_size:
                    game_data.add(observation, action, probs, reward, max_q)
                else:
                    game_data2.add(observation, action, probs, reward, max_q)
                    if game_data2.size == history_size * 2:
                        self.replay_buffer.add(game_data)
                        game_data = game_data2
                        game_data2 = GameData()

                if terminal:
                    if game_data2.size > 0:
                        for i in range(game_data2.size):
                            o, a, p, r, q = game_data2.get(i)
                            game_data.add(o, a, p, r, q)
                    game_data.terminal = True
                    self.replay_buffer.add(game_data)

                observation = new_observation
                if step % learn_step == 0:
                    self.learn(batch_size, history_size)
                step += 1

            scores.append(score)

            if episode % print_episode == 0:
                avg = np.average(scores[-100:])
                #for a in range(10):
                #    print('***')
                print('episodes: ', episode, '\taverage score: ', avg)
                #for a in range(10):
                #    print('***')

    def learn(self, batch_size, history_size):
        if self.replay_buffer.index < batch_size:
            return

        batch = self.replay_buffer.sample(batch_size)
        observations = []
        actions = torch.zeros([history_size, batch_size, 1], dtype=torch.int64)
        probs = torch.zeros([history_size, batch_size, self.actions])
        target_rewards = torch.zeros(history_size, batch_size)
        target_values = torch.zeros(history_size, batch_size)

        ratio = 1.0 / history_size

        for i in range(batch_size):
            o, a, p, r, v = batch[i].sample(history_size, self)
            observations.append(o)

            for j in range(history_size):
                actions[j, i] = a[j]
                probs[j, i] = p[j]
                target_rewards[j, i] = r[j]
                target_values[j, i] = v[j]

        observations = torch.stack(observations).to(self.device)

        logits, values, hidden_states = self.predict_from_observation(observations)
        logits, values, hidden_states = logits.cpu(), values.cpu(), hidden_states.cpu()
        states = []
        for state, action in zip(hidden_states, actions[0]):
            states.append(self.create_hidden_state(state, action))
        states = torch.stack(states).to(self.device)

        log_probs = F.log_softmax(logits, dim=-1)
        advantage = target_values[0] - values

        value_loss = (advantage**2).mean()
        policy_loss = (- log_probs * probs[0]).mean()
        reward_loss = torch.zeros([1])

        #loss = (advantage**2).mean() - (log_probs_policy * advantage.detach()).mean()

        for i in range(1, history_size):
            logits, values, hidden_states, rewards = self.predic_from_state(states)
            logits, values, hidden_states, rewards = logits.cpu(), values.cpu(), hidden_states.cpu(), rewards.cpu()

            states = []
            for state, action in zip(hidden_states, actions[i]):
                hidden_state = self.scale_gradient(state, 0.5)
                states.append(self.create_hidden_state(hidden_state, action))
            states = torch.stack(states).to(self.device)

            log_probs = F.log_softmax(logits, dim=-1)

            advantage = target_values[i] - values

            value_loss += self.scale_gradient((advantage**2).mean(), ratio)
            policy_loss += self.scale_gradient((- log_probs * probs[i]).mean(), ratio)
            reward_loss += self.scale_gradient(((target_rewards[i-1] - rewards)**2).mean(), ratio)
            #l = (advantage**2).mean() - (log_probs_policy * advantage.detach()).mean() + (reward_loss**2).mean()
            #loss += self.scale_gradient(l, ratio)

        loss = value_loss + policy_loss + reward_loss
        print('value', value_loss, 'policy', policy_loss, 'reward', reward_loss)
        #print('loss', loss)
        self.optimizer.zero_grad()
        loss.backward()
        #print(self.model.show())
        self.optimizer.step()

    def scale_gradient(self, tensor, scale):
        return tensor * scale + tensor.detach() * (1 - scale)
