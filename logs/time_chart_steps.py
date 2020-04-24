import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

data = []
colors = ['blue', 'red', 'green', 'black', 'pink', 'brown', 'yellow', 'indianred', 'fuchsia', 'orange', 'royalblue', 'aqua', 'grey', 'chartreuse']
names = ['breakout/breakout_2_70000_a2c.txt', 'breakout/breakout_1_70000_ppo.txt', 'breakout/breakout_1_70000_a2c.txt']


for i in range(len(names)):
    d = pd.read_csv(names[i])
    plt.plot(d['step'], d['score'], color=colors[i], label=names[i])

plt.legend(loc='upper left')
plt.show()
