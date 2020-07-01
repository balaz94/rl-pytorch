import torch

def init_board_state(history):
    depth = (1 + history) * 13 + 6
    state = torch.zeros((depth, 8, 8))

    for i in range(8):
        state[0, 6, i] = state[6, 1, i] = 1                                 #pawns

    state[1, 7, 1] = state[1, 7, 6] = state[7, 0, 1] = state[7, 0, 6] = 1   #knights
    state[2, 7, 2] = state[2, 7, 5] = state[8, 0, 2] = state[8, 0, 5] = 1   #bishops
    state[3, 7, 0] = state[3, 7, 7] = state[9, 0, 0] = state[9, 0, 7] = 1   #rocks
    state[4, 7, 3] = state[10, 0, 3] = 1                                     #queens
    state[5, 7, 4] = state[11, 0, 4] = 1                                     #kings

    state[depth - 6] = 1    #white left castling
    state[depth - 5] = 1    #white right castling
    state[depth - 4] = 1    #black left castling
    state[depth - 3] = 1    #black right castling

    return state

class Chess:
    def __init__(self, history = 7, state = None):
        self.history = history
        if state is None:
            self.state = init_board_state(history)
        else:
            self.state = state
