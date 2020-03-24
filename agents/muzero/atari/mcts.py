import numpy as np

class Node:
    def __init__(self, state, probabilities, value):
        self.state = state

        self.edges = []
        for i in range(probabilities):
            e = Edge(self, i, probabilities[i])
            self.edges.append(e)
        self.N = 0
        self.V = value

class Edge:
    def __init__(self, node1, action, prob):
        self.node1 = node1
        self.node2 = None
        self.action = action

        self.N = 0
        self.Q = 0
        self.P = prob
        self.R = 0

class MCTS:
    def __init__(self, root, c1, c2):
        self.root = root
        self.c1 = c1
        self.c2 = c2

    def selection(self):
        current = self.root
        edges = []

        while True:
            maxActionResult = 0
            maxActionIndex = 0

            for e in current.edges:
                actionResult = e.Q + e.P * (math.sqrt(current.N) / (1 + e.N)) * (self.c1 + math.log((current.N + c2 + 1) / 2))
                if maxActionResult < actionResult:
                    maxActionResult = actionResult
                    maxActionIndex = e.action

            selectedEdge = current.edges[maxActionIndex]
            edges.append(selectedEdge)

            if selectedEdge.node2 is None:
                return selectedEdge, edges

            current = selectedEdge.node2


    def expansion(self, edge, state, reward, probabilities, value):
        edge.R = reward
        edge.node2 = Node(state, probabilities, value)

    def backup(self, edges, gamma):
        edges.reverse()

        G = edges[0].node2.V

        for e in edges:
            G = e.R + gamma * G
            e.Q = (e.N * e.Q + G) / (e.N + 1)
            e.N += 1
            e.node1.N += 1
