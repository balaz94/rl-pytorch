import time
import numpy as np

if __name__ == '__main__':
    filename = 'arena19.txt'

    arena_number = 'arena game '
    game_number = 'game '

    f = open('logs/az/' + filename, "r")
    lines = f.read().split('\n')
    first = True
    player = 1
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

            number = int(line[len(arena_number):len(line)])
            if number % 2 == 0:
                player = -1
            print('game', number)
        elif line.find(';') > -1:
            action = int(line[0:line.find(';')])
            y = action // 6
            x = action % 6

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
            time.sleep(0.5)
