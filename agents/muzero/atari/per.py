import random
import numpy as np

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.curr_index = 0

    def add(self, priority, data):
        tree_index = self.curr_index + self.capacity - 1
        self.data[self.curr_index] = data
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
        return tree_index, self.tree[tree_index], self.data[data_index]

    def max_priority(self):
        return self.tree[0]

class PER:
    #muzero has parameters alpha and beta = 1
    def __init__(self, capacity, epsilon = 0.001):
        self.tree = SumTree(capacity)
        self.epsilon = epsilon
        self.size = 0

    def add(self, error, data):
        priority = np.abs(error) + self.epsilon #alpha is 1
        self.tree.add(priority, data)
        self.size += 1
        self.size = min(self.size, self.tree.capacity)

    def sample(self, batch_size):
        batch = []
        indexes = []
        priorities = np.zeros(batch_size)
        segment = self.tree.max_priority() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = random.uniform(a, b)
            index, priority, data = self.tree.get(value)

            batch.append(data)
            indexes.append(index)
            priorities[i] = priority

        sampling_priorities = priorities / self.tree.max_priority()
        is_weight = 1.0 / (self.tree.capacity * sampling_priorities) #beta is 1
        is_weight = is_weight / is_weight.max()

        return batch, indexes, is_weight

    def update(self, index, error):
        priority = np.abs(error) + self.epsilon
        self.tree.update(index, priority0)
