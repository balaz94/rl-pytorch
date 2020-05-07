import random
import numpy as np
import torch

class SumTree:
    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.state_dim = state_dim
        dim = (capacity, ) + state_dim

        self.states = torch.zeros(dim)
        self.actions = torch.zeros((self.capacity), dtype=torch.int8)
        self.rewards = torch.zeros(self.capacity)
        self.states_ = torch.zeros(dim)
        self.terminals = torch.zeros(self.capacity)
        self.curr_index = 0

    def add(self, priority, state, action, reward, state_, terminal):
        tree_index = self.curr_index + self.capacity - 1

        self.states[self.curr_index] = state
        self.actions[self.curr_index] = action
        self.rewards[self.curr_index] = reward
        self.states_[self.curr_index] = state_
        self.terminals[self.curr_index] = int(terminal)

        self.update(tree_index, priority)
        self.curr_index += 1
        if self.curr_index >= self.capacity:
            self.curr_index = 0

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get(self, value):
        parent_index = 0

        while True:
            child_left = 2 * parent_index + 1
            child_right = child_left + 1

            if child_left >= len(self.tree):
                tree_index = parent_index
                break
            else:
                if value <= self.tree[child_left]:
                    parent_index = child_left
                else:
                    value -= self.tree[child_left]
                    parent_index = child_right

        data_index = tree_index - self.capacity + 1
        return tree_index, self.tree[tree_index], data_index

    def get_batch(self, batch):
        states =  self.states[batch]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        states_ = self.states_[batch]
        terminal = self.terminals[batch]

        return states, actions, rewards, states_, terminal


    def max_priority(self):
        return self.tree[0]

class PER:
    #muzero has parameters alpha and beta = 1
    def __init__(self, capacity, state_dim, epsilon = 0.001):
        self.tree = SumTree(capacity, state_dim)
        self.epsilon = epsilon
        self.size = 0

    def store(self, priority, state, action, reward, state_, terminal):
        self.tree.add(np.abs(priority) + self.epsilon, state, action, reward, state_, terminal)
        self.size += 1
        self.size = min(self.size, self.tree.capacity)

    def sample(self, batch_size):
        data_indexes = []
        indexes = []
        priorities = np.zeros(batch_size)
        segment = self.tree.max_priority() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = random.uniform(a, b)
            index, priority, data_index = self.tree.get(value)

            data_indexes.append(data_index)
            indexes.append(index)
            priorities[i] = priority

        sampling_priorities = priorities / self.tree.max_priority()
        #print(self.tree.max_priority())
        #print(sampling_priorities)
        is_weight = 1.0 / (self.tree.capacity * sampling_priorities) #beta is 1
        is_weight = is_weight / is_weight.max()

        states, actions, rewards, states_, terminals = self.tree.get_batch(np.asarray(data_indexes))

        return states, actions, rewards, states_, terminals, indexes, is_weight

    def update(self, index, error):
        priority = np.abs(error) + self.epsilon
        self.tree.update(index, priority)
