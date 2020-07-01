import numpy as np

if __name__ == '__main__':
    filename = 'arena8.txt'

    arena_number = 'arena game '

    f = open('logs/az/arena' +  + filename, "r")
    lines = f.read().split('\n')
    first = True


    for line in lines:
        if line.find(arena_number) > -1:
            if first:
                first = False
            else:
                print('end game')

            print('game', line[len(arena_number):len(line)])
        elif line.find(';') > -1:
            action = int(line[0:line.find(';')])
            print(action)
