import time
import numpy as np
from envs.tictactoe import TicTacToe

def game():
    filename = 'testgame.txt'

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
    state = None
    env = TicTacToe(width = width, height = height, frames = frames, wins_count = wins_count)

    step = 0
    array = np.zeros((6, 6), dtype=np.int8)
    for line in lines:
        if line.find(arena_number) > -1:
            if first:
                first = False
            else:
                print('end game')
                player = 1
                array = np.zeros((6, 6), dtype=np.int8)
                step = 0
            state, m = env.start()

            print('game', line[len(arena_number):len(line)])
        elif line.find(game_number) > -1:
            print('end game')
            player = 1
            array = np.zeros((6, 6), dtype=np.int8)
            step = 0
            state, m = env.start()

            print('game', line[len(game_number):len(line)])
        elif line.find(';') > -1:
            action = int(line[0:line.find(';')])
            y = action // 6
            x = action % 6
            state, r, t, m = env.step(action, state)

            if player == 1:
                array[y, x] = 1
            else:
                array[y, x] = 2

            player = - player

            for y in range(6):
                text = ''
                for x in range(6):
                    text += ' '
                    if array[y, x] == 0:
                        text += ' _'
                    elif array[y, x] == 1:
                        text += ' X'
                    elif array[y, x] == 2:
                        text += ' O'
                print(text)
            print()
            step += 1
            print(step)
            if step == 31:
                print(state)
                print(r, t)
                return 1
            time.sleep(0.5)

if __name__ == '__main__':
    game()
