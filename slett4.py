#%%
import numpy as np
import h5py
from datagenerering import generate_U_txy
from pathlib import Path
import matplotlib.pyplot as plt

filnamn_inn = Path("../TONSTAD_TWO_Q100_TWO.hdf5")

with h5py.File(filnamn_inn, 'r') as f:
    Umx = np.asarray(f['Umx'])#[int(t_min*fps):int(t_max*fps),:]
    Vmx = np.asarray(f['Vmx'])#[int(t_min*fps):int(t_max*fps),:]
    (I,J) = (int(np.asarray(f['I'])),int(np.asarray(f['J'])))
    x = np.asarray(f['x']).reshape(J,I)
    y = np.asarray(f['y']).reshape(J,I)
    ribs = np.asarray(f['ribs'])
    x_lang = np.asarray(f['x']) 
    y_lang = np.asarray(f['y']) 

U, txy = generate_U_txy((0,1), Umx,Vmx,x,y,I,J,ribs, experiment="TWO", kutt=False)

txy = txy[txy[:,0]==0]
# %%

myDPI = 150
fig, axes = plt.subplots(figsize=(2050/myDPI,1450/myDPI),dpi=myDPI)
axes.quiver(x_lang,y_lang,Umx[0,:],Vmx[0,:],scale=10000)
axes.plot(ribs.T[0],ribs.T[1])
axes.set_xlim((-40,80))
axes.set_ylim((-25,10))
plt.show()

#%%
filnamn = f"ribs_before_correction.png"
fig.savefig(filnamn)

#%%

from scipy.spatial import Voronoi, voronoi_plot_2d
nonan = np.invert(np.isnan(Umx[0]))
x_y = np.stack((x_lang[nonan],y_lang[nonan])).T

vor = Voronoi(x_y)
import matplotlib
matplotlib.rcParams["figure.dpi"] = 100
fig = voronoi_plot_2d(vor, show_vertices=False,show_points=False)
plt.axis('equal')
plt.xlim((20,60))
plt.ylim((-25,10))
plt.plot(ribs.T[0],ribs.T[1])
plt.show()
# %%
