import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy import interpolate

# with open("sti.pickle", 'rb') as f:
#     s = np.array(pickle.load(f))

# fig, ax = plt.subplots()

# for a in s:
#     ax.plot(a[:,1], a[:,2])

# plt.show()

def get_y(seed):
    np.random.seed(seed)
    y = [np.random.randint(15, 85) for i in range(20)]
    inter = interpolate.interp1d(np.arange(0,20), y, kind = 'cubic')
    return inter(np.arange(1, 15, 0.1))
    
y = get_y(23)
ys = [get_y(3), get_y(5), get_y(8), get_y(13)]# figure and axis

fig, ax = plt.subplots(1, figsize=(14,10), facecolor='#F3F0E0')
ax.set_facecolor('#F3F0E0')

# annotation arrow

arrowprops = dict(arrowstyle="->", color="green",connectionstyle="angle3,angleA=0,angleB=-90")
max_idx = np.argmax(y, axis=0)
plt.annotate('This line is special', 
             xy=(max_idx, max(y)), 
             xytext=(max_idx+5, max(y)+10), 
             arrowprops=arrowprops,
             size = 18)
             
             # annotation text

line_y = np.quantile(y, 0.75)
plt.plot([line_y]*len(y), linestyle='--', color='#FB5A14', alpha=0.8)
plt.annotate('This value is important', xy=(2, line_y), 
             size=12, ha='left', va="bottom")

# plots

for i in ys:
    plt.plot(i, color='#888888', alpha=0.3)
    plt.plot(y, color='#16264c')
    
# limits
plt.xlim(0,141)
plt.ylim(0,101)

# remove spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.title('Meaningful title', pad=15, loc='left', 
          fontsize = 26, alpha=0.9)# ticks
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.show()
# plt.savefig('mychart.png', facecolor='#F3F0E0', dpi=100)