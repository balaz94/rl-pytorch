import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

data = []
colors = ['blue', 'red', 'green', 'black', 'pink', 'brown', 'yellow', 'indianred', 'fuchsia', 'orange', 'royalblue', 'aqua', 'grey', 'chartreuse']
names = ['pong/pong256_1_700_ppo.txt', 'pong/pong512_1_700_ppo.txt', 'pong/pong768_1_700_ppo.txt', 'pong/pong1024_1_700_ppo.txt', 'pong/pong256_2_1_700_ppo.txt',
         'pong/pong256_2_700_ppo.txt', 'pong/pong512_2_700_ppo.txt', 'pong/pong768_2_700_ppo.txt', 'pong/pong1024_2_700_ppo.txt', 'pong/pong256_2_2_700_ppo.txt']

names2 = ['pong/pong256_2_700_ppo.txt', 'pong/pong512_2_700_ppo.txt', 'pong/pong768_2_700_ppo.txt', 'pong/pong1024_2_700_ppo.txt', 'pong/pong256_2_2_700_ppo.txt']


for i in range(len(names)):
    d = pd.read_csv(names[i])
    plt.plot(d['step'], d['score'], color=colors[i], label=names[i])

plt.legend(loc='upper left')
plt.show()
