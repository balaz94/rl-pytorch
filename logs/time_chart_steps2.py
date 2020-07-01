import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

data = []
colors = ['blue', 'red', 'green', 'black', 'pink', 'brown', 'yellow', 'indianred', 'fuchsia', 'orange', 'royalblue', 'aqua', 'grey', 'chartreuse']
names = ['breakout/rewards_in_interval_1_6200_ppo.txt', 'breakout/rewards_in_interval_batch_epochs_1_8000_ppo.txt']

for i in range(len(names)):
    d = pd.read_csv(names[i])
    plt.plot(d['step'], d['score'], color=colors[i], label=names[i])

plt.legend(loc='upper left')
plt.show()
