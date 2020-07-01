import torch

class TicTacToe:
    def __init__(self, width = 8, height = 8, wins_count = 5, frames = 4):
        self.width = width
        self.height = height
        self.wins_count = wins_count
        self.max_moves = width * height
        self.frames = frames

    def start(self):
        return torch.zeros((self.frames * 2, self.height, self.width), dtype=torch.int16), torch.ones(self.max_moves, dtype=torch.int16)
    def step(self, action, state):
        y = action // self.width
        x = action % self.width

        new_state = torch.zeros((self.frames * 2, self.height, self.width), dtype=torch.int16)
        for i in reversed(range(self.frames * 2 - 2)):
            if i % 2 == 0:
                new_state[i+3] = state[i].clone()
            else:
                new_state[i+1] = state[i].clone()

        new_state[0] = state[1].clone()
        new_state[1] = state[0].clone()
        new_state[1, y, x] = 1

        if self.check_win(new_state[1], x, y):
            return new_state, 1, True, torch.zeros(self.max_moves, dtype=torch.int16)

        moves, sum = self.possible_moves(new_state)
        if (new_state[0].sum() + new_state[1].sum()).item() + sum != 36:
            print((new_state[0].sum() + new_state[1].sum()).item(), sum)
            print(new_state)

        if sum == 0:
            return new_state, 0, True, torch.zeros(self.max_moves, dtype=torch.int16)
        else:
            return new_state, 0, False, moves
    def check_win(self, state, x, y):
        x_start = max(x - self.wins_count, 0)
        x_end = min(x + self.wins_count, self.width)

        y_start = max(y - self.wins_count, 0)
        y_end = min(y + self.wins_count, self.height)

        sum = 0
        for i in range(x_start, x_end):
            if state[y, i] == 1:
                sum += 1
                if sum == self.wins_count:
                    #print('horizontal')
                    return True
            else:
                sum = 0

        sum = 0
        for i in range(y_start, y_end):
            if state[i, x] == 1:
                sum += 1
                if sum == self.wins_count:
                    #print('vertical')
                    return True
            else:
                sum = 0

        d_start = min(x - x_start, y - y_start)
        d_end = min(x_end - x, y_end - y)
        d_x, d_y = x - d_start, y - d_start
        d_range = x + d_end - d_x

        sum = 0
        for i in range(d_range):
            if state[d_y + i, d_x + i] == 1:
                sum += 1
                if sum == self.wins_count:
                    #print('diagonal1')
                    return True
            else:
                sum = 0

        d_start = min(x_end - x - 1, y - y_start)
        d_end = min(x - x_start + 1, y_end - y)
        d_x, d_y = x + d_start, y - d_start
        d_range = y + d_end - d_y

        sum = 0
        for i in range(d_range):
            if state[d_y + i, d_x - i] == 1:
                sum += 1
                if sum == self.wins_count:
                    #print('diagonal 2')
                    return True
            else:
                sum = 0

        return False
    def possible_moves(self, state):
        mask = torch.zeros(self.max_moves, dtype=torch.int16)
        index, sum = 0, 0
        for y in range(self.height):
            for x in range(self.width):
                if state[0, y, x] == 0 and state[1, y, x] == 0:
                    mask[index] = 1
                    sum += 1
                index += 1
        return mask, sum

    def start2(self):
        state = torch.tensor([[[1, 0, 1, 0, 0, 0],
                             [0, 1, 0, 0, 1, 0],
                             [1, 0, 1, 0, 0, 0],
                             [0, 1, 1, 0, 0, 1],
                             [0, 1, 1, 0, 1, 0],
                             [0, 0, 0, 0, 0, 0]],

                            [[0, 0, 0, 0, 0, 0],
                             [1, 0, 0, 1, 0, 0],
                             [0, 1, 0, 0, 1, 0],
                             [1, 0, 0, 1, 0, 0],
                             [1, 0, 0, 1, 0, 0],
                             [1, 0, 1, 1, 1, 1]],

                            [[1, 0, 1, 0, 0, 0],
                             [0, 1, 0, 0, 1, 0],
                             [1, 0, 1, 0, 0, 0],
                             [0, 1, 1, 0, 0, 1],
                             [0, 1, 1, 0, 1, 0],
                             [0, 0, 0, 0, 0, 0]],

                            [[0, 0, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 1, 0],
                             [1, 0, 0, 1, 0, 0],
                             [1, 0, 0, 1, 0, 0],
                             [1, 0, 1, 1, 1, 1]],

                            [[1, 0, 1, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [1, 0, 1, 0, 0, 0],
                             [0, 1, 1, 0, 0, 1],
                             [0, 1, 1, 0, 1, 0],
                             [0, 0, 0, 0, 0, 0]],

                            [[0, 0, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 1, 0],
                             [1, 0, 0, 1, 0, 0],
                             [1, 0, 0, 1, 0, 0],
                             [1, 0, 1, 1, 1, 1]],

                            [[1, 0, 1, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [1, 0, 1, 0, 0, 0],
                             [0, 1, 1, 0, 0, 1],
                             [0, 1, 1, 0, 1, 0],
                             [0, 0, 0, 0, 0, 0]],

                            [[0, 0, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 1, 0],
                             [1, 0, 0, 1, 0, 0],
                             [0, 0, 0, 1, 0, 0],
                             [1, 0, 1, 1, 1, 1]]], dtype=torch.int16)

        moves, sum = self.possible_moves(state)
        return state, moves
