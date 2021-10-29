import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import pickle

with open("sti.pickle", 'rb') as f:
    s = np.array(pickle.load(f))

fig, ax = plt.subplots()

for a in s:
    ax.plot(a[:,1], a[:,2])

plt.show()