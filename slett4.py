#%%

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

if not isnotebook():
    import matplotlib
    matplotlib.use("Agg")

import numpy as np
import h5py
from datagenerering import generate_U_txy
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.spatial.qhull as qhull
from hjelpefunksjonar import draw_rect
from rib import Rib

def interpolate(xyz, uvw, values):
    tri = qhull.Delaunay(xyz)
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)

    delta = uvw - temp[:,2]
    bary = np.einsum('njk,nk->nj', temp[:,:2,:], delta)
    wts = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
    return np.einsum('nji,nj->ni', np.take(values,vertices, axis=0), wts)

filnamn_inn = Path("../TONSTAD_TWO_Q100_TWO.hdf5")

with h5py.File(filnamn_inn, 'r') as f:
    Umx = np.asarray(f['Umx'])#[int(t_min*fps):int(t_max*fps),:]
    Vmx = np.asarray(f['Vmx'])#[int(t_min*fps):int(t_max*fps),:]
    (I,J) = (int(np.asarray(f['I'])),int(np.asarray(f['J'])))
    ribs = np.asarray(f['ribs'])
    x_lang = np.asarray(f['x']) 
    y_lang = np.asarray(f['y']) 
    x = x_lang.reshape(J,I)
    y = y_lang.reshape(J,I)

nonan = np.invert(np.isnan(Umx[0]))

xy = np.asarray([x_lang[nonan],y_lang[nonan]]).T
U = np.asarray([Umx[0][nonan],Vmx[0][nonan]]).T

x_min = 20
x_max = 60
y_min = -25
y_max = 10

utval = (xy[:,0] > 20) & (xy[:,0] < 60) & (xy[:,1] > -25) & (xy[:,1] < 10)
xy_utval = xy[utval]
U_utval = U[utval]

U_korrigert, txy_korrigert = generate_U_txy((0,1), Umx,Vmx,x,y,I,J,ribs, experiment="TWO", kutt=False)

utval_korrigert = (txy_korrigert[:,0]==0) & (txy_korrigert[:,1] >20) & (txy_korrigert[:,1] < 60) & (txy_korrigert[:,2] > -25) & (txy_korrigert[:,2] < 10)
txy_korrigert = txy_korrigert[utval_korrigert]
U_korrigert = U_korrigert.T[utval_korrigert]

width, height = 1200, 720

nytt_rutenett = np.meshgrid(np.linspace(20,60,width), np.linspace(-25,10,height))
nytt_rutenett = np.asarray([nytt_rutenett[0].ravel(), nytt_rutenett[1].ravel()]).T

ny_U = interpolate(xy_utval, nytt_rutenett, U_utval)
ny_U_korrigert = interpolate(txy_korrigert[:,1:], nytt_rutenett, U_korrigert)




myDPI = 300
fig, axes = plt.subplots(figsize=(width/myDPI,height/myDPI),dpi=myDPI)
# axes.quiver(x_lang,y_lang,Umx[0,:],Vmx[0,:],scale=10000)
# axes.plot(ribs.T[0],ribs.T[1])
field = axes.imshow(((ny_U_korrigert.T[0]**2+ny_U.T[1]**2)**0.5).reshape(height,width), extent=[x_min,x_max, y_min, y_max], interpolation='none')
# axes.set_xlim((20,60))
# axes.set_ylim((-25,10))
with h5py.File(filnamn_inn.with_name(f"{filnamn_inn.stem}_ribs.hdf5"),'r') as f:
    ribs = [Rib(rib) for rib in np.asarray(f['ribs'])]
draw_rect(axes, ribs ,fill=False)
plt.show()


#%%

filnamn = f"ribs_before_correction.png"
fig.savefig(filnamn)

#%%

# from scipy.spatial import Voronoi, voronoi_plot_2d
# nonan = np.invert(np.isnan(Umx[0]))
# x_y = np.stack((x_lang[nonan],y_lang[nonan])).T

# vor = Voronoi(x_y)
# # matplotlib.rcParams["figure.dpi"] = 100
# # fig = voronoi_plot_2d(vor, show_vertices=False,show_points=False)
# plt.axis('equal')
# plt.xlim((20,60))
# plt.ylim((-25,10))
# plt.plot(ribs.T[0],ribs.T[1])
# plt.show()
