import math
import copy
import numpy as np
from random import randrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Node:
    def __init__(self, probs, state, probs_mask, reward, terminal, parent = None):
        self.state = state
        self.P = probs * probs_mask
        probs_sum = self.P.sum().item()
        length = len(self.P)

        if probs_sum == 0.0:
            mask_sum = probs_mask.sum().item()
            if mask_sum == 0.0:
                self.P = torch.ones(length, dtype = torch.float32)
            else:
                self.P = probs_mask * (1.0 / mask_sum)
        else:
            self.P = self.P / probs_sum

        self.N = torch.zeros(length, dtype = torch.int16)
        self.Q = torch.zeros(length, dtype = torch.float32)
        self.R = reward
        self.T = terminal

        self.children = {}
        self.parent = parent
class MCTS:
    def __init__(self, root, env, c1, c2, gamma):
        self.root = root
        self.env = env

        self.c1 = c1
        self.c2 = c2
        self.gamma = gamma

    def selection(self):
        node = self.root
        actions = []

        while True:
            if node.T == True:
                return node, -1, actions

            sum = node.N.sum().item()
            if sum > 0:
                c = self.c1 + math.log((sum + self.c2 + 1.0) / self.c2)
                u = node.Q + node.P * c * (math.sqrt(sum) / (1.0 + node.N))

                u_max = -10000
                a = 0
                for i in range(len(u)):
                    if node.P[i] > 0.0:
                        if u[i] > u_max:
                            a = i
                            u_max = u[i]
            else:
                a = torch.argmax(node.P).item()
            actions.append(a)

            if a in node.children:
                node = node.children[a]
            else:
                return node, a, actions
    def expansion(self, node, action, probs, state, probs_mask, reward, terminal):
        new_node = Node(probs, state, probs_mask, reward, terminal, node)
        node.children[action] = new_node
        return new_node
    def backup(self, node, value, actions):
        v = node.R + self.gamma * value * (1 - int(node.T))
        actions.reverse()

        for a in actions:
            node = node.parent
            v = node.R - self.gamma * v
            node.Q[a] = (node.N[a] * node.Q[a] + v) / (node.N[a] + 1)
            node.N[a] += 1
class Game:
    def __init__(self, states, values, policy):
        self.states = states
        self.target_values = values
        self.target_policy = policy
class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []

    def store(self, game):
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample(self, batch_size):
        length = len(self.buffer)
        game_indexes = np.random.choice(length, batch_size)

        states, target_values, target_policy = [], [], []
        for index in game_indexes:
            game = self.buffer[index]
            action_index = randrange(len(game.states))
            states.append(game.states[action_index])
            target_values.append(game.target_values[action_index])
            target_policy.append(game.target_policy[action_index])

        return torch.stack(states), torch.stack(target_values), torch.stack(target_policy)

class AZAgent:
    def __init__(self, env, model, replay_buffer, actions, simulation_count = 800, gamma = 0.997, training_steps = 5000, c1 = 1.25, c2 = 19652.0, temperature_update_steps = 100000,
                 dirichlet_alpha = 0.1, exploration_fraction = 0.25, weight_decay = 0.0001, lr = 2e-2, arena_count = 25, self_play_steps = 100, batch_size = 512, name = 'az',
                 run_iteration = 1000, steps = 0):
        self.env = env
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device: ', self.device)
        self.model.to(self.device)
        self.weight_decay = weight_decay
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay = weight_decay)

        self.replay_buffer = replay_buffer
        self.actions = actions

        self.dirichlet_alpha = dirichlet_alpha
        self.exploration_fraction = exploration_fraction

        self.simulation_count = simulation_count
        self.arena_count = arena_count
        self.self_play_steps = self_play_steps
        self.training_steps = training_steps
        self.steps = steps
        self.batch_size = batch_size
        self.run_iteration = run_iteration + 1

        self.name = name

        self.c1 = c1
        self.c2 = c2
        self.gamma = gamma
        self.lr = lr

        self.T = 1
        self.temperature_update_steps = temperature_update_steps
        self.update_parameters()

    def update_parameters(self):
        if self.steps == 3 * self.temperature_update_steps:
            self.T = 0.25
            self.lr = 1e-5
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay = self.weight_decay)
        elif self.steps == 2 * self.temperature_update_steps:
            self.T = 0.5
            self.lr = 1e-4
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay = self.weight_decay)
        elif self.steps == self.temperature_update_steps:
            self.lr = 1e-3
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay = self.weight_decay)

    def run_mcts(self, state, moves, env, model, training = False):
        logit, value = model.forward(state.unsqueeze(0).to(self.device).float())

        probs = F.softmax(logit.cpu(), dim=-1)
        if training == True:
            noise = torch.tensor(np.random.gamma(self.dirichlet_alpha, 1, self.actions))
            probs = probs[0] * (1 - self.exploration_fraction) + noise * self.exploration_fraction
        else:
            probs = probs[0]

        root = Node(probs, state.cpu(), moves, 0, False)
        mcts = MCTS(root, env, self.c1, self.c2, self.gamma)

        for simulation in range(self.simulation_count):
            node, a, actions = mcts.selection()
            if a > -1:
                new_state, reward, terminal, new_moves = env.step(a, node.state)
                if terminal:
                    probs, value = torch.tensor(self.actions).float(), 0
                else:
                    logit, value = model.forward(new_state.unsqueeze(0).to(self.device).float())
                    probs = F.softmax(logit.cpu(), dim=-1)[0]
                    value = value.cpu().item()

                new_node = mcts.expansion(node, a, probs, new_state, new_moves, reward, terminal)
                mcts.backup(new_node, value, actions)
            else:
                mcts.backup(node, 0, actions)

        print('Q', mcts.root.Q)
        print('p', mcts.root.P)
        return mcts.root.N

    def play(self, model1, model2):
        terminal = False
        state, moves = self.env.start()
        player = 1

        states, rewards, target_policy = [], [], []

        logs = ''
        while not terminal:
            with torch.no_grad():
                if player == 1:
                    visit_childs = self.run_mcts(state, moves, self.env, model1)
                else:
                    visit_childs = self.run_mcts(state, moves, self.env, model2)

            temperature_coef = 1.0/self.T
            visit_childs = visit_childs.float()**temperature_coef
            sum = visit_childs.sum().item()
            probs = visit_childs / sum
            action = torch.argmax(probs).item()

            logs += str(action) + ';' + str(probs) + '\n'

            state, reward, terminal, moves = self.env.step(action, state)
            player = - player

            states.append(state.float())
            rewards.append(reward)
            target_policy.append(probs)

        length = len(rewards)
        target_values = torch.zeros(length)

        value = 0
        for i in reversed(range(length)):
            value = rewards[i] - self.gamma * value
            target_values[i] = value

        self.replay_buffer.store(Game(states, target_values, target_policy))

        return reward, logs

    def self_play(self):
        terminal = False
        states, rewards, target_policy = [], [], []
        state, moves = self.env.start()
        while not terminal:
            with torch.no_grad():
                visit_childs = self.run_mcts(state, moves, self.env, self.model, True)
            temperature_coef = 1.0/self.T
            visit_childs = visit_childs.float()**temperature_coef
            sum = visit_childs.sum().item()
            probs = visit_childs / sum
            action = probs.multinomial(num_samples=1).detach().item()

            new_state, reward, terminal, moves = self.env.step(action, state)

            states.append(state.float())
            rewards.append(reward)
            target_policy.append(probs)
            state = new_state

        length = len(rewards)
        target_values = torch.zeros(length)

        value = 0
        for i in reversed(range(length)):
            value = rewards[i] - self.gamma * value
            target_values[i] = value

        self.replay_buffer.store(Game(states, target_values, target_policy))
        return rewards[length-1]

    def learn(self, batch_size):
        states, target_values, target_policy = self.replay_buffer.sample(batch_size)

        logits, values = self.model.forward(states.to(self.device))
        log_probs = F.log_softmax(logits.cpu(), dim=-1)[0]
        values = values.cpu()[0]

        loss_value = (values - target_values)**2
        loss_policy = - (target_policy * log_probs)
        loss = loss_value.mean() + loss_policy.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def arena(self, model1, model2, iteration):
        wins1, wins2, draft = 0, 0, 0
        text = ''
        min_iteration = max(0, iteration - 10)
        game = 1

        for i in range(min_iteration, iteration):
            print(game)
            model2.load_state_dict(torch.load('models/' + self.name + str(i) + '.pt'))

            text += 'arena game ' + str(game) + '\n'
            result, logs = self.play(model1, model2)
            text += logs

            if result == 1:
                wins1 += 1
            elif result == 0:
                draft += 1
            else:
                wins2 += 1

            game += 1
            text += 'arena game ' + str(game) + '\n'
            result, logs = self.play(model2, model1)
            text += logs

            if result == 1:
                wins2 += 1
            elif result == 0:
                draft += 1
            else:
                wins1 += 1

            game += 1

        print('win 1: ', wins1, 'win 2:', wins2, 'draft: ', draft)
        return wins1 / float(wins1 + wins2 + 1e-10), text

    def run(self):
        for iteration in range(1, self.run_iteration):
            print('iteration: ', iteration)

            for i in range(self.self_play_steps):
                if i % 5 == 0:
                    print('self play ', i)
                self.self_play()

            torch.save(self.model.state_dict(), 'models/' + self.name + str(iteration - 1) + '.pt')

            for i in range(self.training_steps):
                if i % 250 == 0:
                    print('training ', i)
                self.learn(self.batch_size)

            self.steps += self.training_steps
            self.update_parameters()

            model2 = copy.deepcopy(self.model)
            model2.to(self.device)

            print('arena')
            ratio, text = self.arena(self.model, model2, iteration)
            print('arena ratio: ', ratio)

            f = open('logs/az/arena' + str(iteration) + '.txt', "w")
            f.write(text)
            f.close()

            del model2

    def run2(self):
        terminal = False
        rewards, target_policy = [], []
        state, moves = self.env.start2()
        for y in range(6):
            line = ' '
            for x in range(6):
                if state[0, y, x] == 1:
                    line += ' X '
                elif state[1, y, x] == 1:
                    line += ' O '
                else:
                    line += ' _ '
            print(line)

        print('m', moves.reshape(6, 6))
        while not terminal:
            with torch.no_grad():
                visit_childs = self.run_mcts(state, moves, self.env, self.model, False)

            temperature_coef = 1.0/self.T
            print('visit', visit_childs)
            visit_childs = visit_childs.float()**temperature_coef
            sum = visit_childs.sum().item()
            probs = visit_childs / sum
            print('visits probs', probs)
            action = probs.multinomial(num_samples=1).detach().item()

            new_state, reward, terminal, moves = self.env.step(action, state)

            rewards.append(reward)
            state = new_state

            for y in range(6):
                line = ' '
                for x in range(6):
                    if state[0, y, x] == 1:
                        line += ' X '
                    elif state[1, y, x] == 1:
                        line += ' O '
                    else:
                        line += ' _ '
                print(line)

            return -1

        length = len(rewards)
        target_values = torch.zeros(length)

        value = 0
        for i in reversed(range(length)):
            value = rewards[i] - self.gamma * value
            target_values[i] = value

        return rewards[length-1]
