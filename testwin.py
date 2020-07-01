import time
import numpy as np
from envs.tictactoe import TicTacToe

def game():
    filename = 'arena7.txt'

    arena_number = 'arena game '
    game_number = 'game '

    f = open('logs/az/' + filename, "r")
    lines = f.read().split('\n')
    first = True
    player = 1

    width = 6
    height = 6
    frames = 4
    wins_count = 5

    actions = width * height
    env = TicTacToe(width = width, height = height, frames = frames, wins_count = wins_count)

    step = 0
    state = np.array([[0, 0, 0, 0, 1, 1],
                      [0, 0, 0, 1, 1, 1],
                      [0, 0, 1, 1, 1, 0],
                      [0, 1, 1, 0, 0, 0],
                      [0, 1, 1, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0]], dtype=np.int8)

    sum = 0
    for action in range(36):
        y = action // width
        x = action % width

        if env.check_win(state, x, y):
            print(y, x)
            sum += 1

    print(sum)

    '''
      X  O  _  X  X  O
      X  X  O  _  O  X
      O  _  O  O  X  _
      X  _  X  O  O  X
      X  O  O  X  O  O
      X  O  X  X  O  O
    '''


if __name__ == '__main__':
    game()
