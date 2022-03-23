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
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams['mathtext.fontset'] = 'stix'
import matplotlib.colors as colors
import scipy.spatial.qhull as qhull
from hjelpefunksjonar import draw_rect
from rib import Rib
from math import isclose


def interpolate(xyz, uvw, values):
    tri = qhull.Delaunay(xyz)
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)

    delta = uvw - temp[:,2]
    bary = np.einsum('njk,nk->nj', temp[:,:2,:], delta)
    wts = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
    return np.einsum('nji,nj->ni', np.take(values,vertices, axis=0), wts)

def plott(xy, U_gml,filnamn):
    U_ny = interpolate(xy, nytt_rutenett, U_gml)
    myDPI = 300

    fig, axes = plt.subplots(figsize=figureSize,dpi=myDPI) #(width/myDPI,height/myDPI)
    # axes.quiver(x_lang,y_lang,Umx[0,:],Vmx[0,:],scale=10000)
    # axes.plot(ribs.T[0],ribs.T[1])
    field = axes.imshow(((U_ny.T[0]**2+U_ny.T[1]**2)**0.5).reshape(height,width), extent=[x_min,x_max, y_min, y_max], 
        interpolation='none', origin='lower', zorder=1, norm=colors.Normalize(vmin=0, vmax=260))
    # datapunkt = axes.plot(xy_utval[:,0], xy_utval[:,1],'.',markersize=1)
    with h5py.File(filnamn_inn.with_name(f"{filnamn_inn.stem}_ribs.hdf5"),'r') as f:
        ribs = [Rib(rib) for rib in np.asarray(f['ribs'])]

    from matplotlib.patches import Polygon
    for i,rib in enumerate(ribs):
        rib_indicator = axes.add_patch(Polygon(rib.vertices, facecolor='none', edgecolor = 'white', linewidth = .7, zorder=i+5))

    # draw_rect(axes, ribs ,color='white',fill=False)
    datapunkt = axes.scatter(xy[:,0], xy[:,1],c='red',s=((U_gml[:,0]**2+U_gml[:,1]**2)**0.5)*0.03+.1, zorder=10)
    
    # axes.quiver(xy[:,0], xy[:,1],U_gml[:,0],U_gml[:,1],scale=2000, zorder=10)
    axes.set_xlim((20,60))
    axes.set_ylim((-25,10))


    cb = fig.colorbar(field, ax=axes)
    axes.set_xlabel(r'$x$ [mm]')
    axes.set_ylabel(r'$y$ [mm]')
    cb.set_label(r"$\vec{U}$ [mm/s]")#, fontsize=18)
    axes.legend((datapunkt, rib_indicator), ('data used for interpolation', 'rib outline'), loc='lower left').set_zorder(15)
    plt.show()

    fig.savefig(filnamn)

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

utval = (xy[:,0] > 10) & (xy[:,0] < 70) & (xy[:,1] > -35) & (xy[:,1] < 20)
xy_utval = xy[utval]
U_utval = U[utval]

U_korrigert, txy_korrigert = generate_U_txy((0,1), Umx,Vmx,x,y,I,J,ribs, experiment="TWO", kutt=False)

utval_korrigert = (txy_korrigert[:,0]==0) & (txy_korrigert[:,1] >10) & (txy_korrigert[:,1] < 70) & (txy_korrigert[:,2] > -35) & (txy_korrigert[:,2] < 20)
txy_korrigert = txy_korrigert[utval_korrigert]
U_korrigert = U_korrigert.T[utval_korrigert]

figureSize = (6,4) # i inches
myDPI = 300
width, height = figureSize[0]*myDPI, figureSize[1]*myDPI

assert isclose(width ,int(width)) and isclose(height, int(height))

width, height = int(width), int(height)

nytt_rutenett = np.meshgrid(np.linspace(20,60,width), np.linspace(-25,10,height))
nytt_rutenett = np.asarray([nytt_rutenett[0].ravel(), nytt_rutenett[1].ravel()]).T

# ny_U = interpolate(xy_utval, nytt_rutenett, U_utval)
# ny_U_korrigert = interpolate(txy_korrigert[:,1:], nytt_rutenett, U_korrigert)

plott(txy_korrigert[:,1:], U_korrigert, f"ribs_U_korrigert.png")
plott(xy_utval, U_utval, f"ribs_U_original.png")

# myDPI = 200

fig, axes = plt.subplots(figsize=figureSize,dpi=myDPI) #(width/myDPI,height/myDPI)
# # axes.plot(ribs.T[0],ribs.T[1])
rensk = (x_lang > 10) & (x_lang < 70) & (y_lang > -35) & (y_lang < 20)
rensk_dim = len(x[0,(x[0,:] > 10) & (x[0,:] < 70)]), len(y[(y[:,0] > -35) & (y[:,0] < 20), 0])
U_renska = Umx[0][rensk]
V_renska = Vmx[0][rensk]
x_renska = x_lang[rensk]
y_renska = y_lang[rensk]
field = axes.imshow(((U_renska**2+V_renska**2)**0.5).reshape(rensk_dim[1],rensk_dim[0]), extent=[np.min(x_renska),np.max(x_renska), np.min(y_renska), np.max(y_renska) ], 
    interpolation='none', origin='upper', zorder=1, norm=colors.Normalize(vmin=0, vmax=260))
# # datapunkt = axes.plot(xy_utval[:,0], xy_utval[:,1],'.',markersize=1)
with h5py.File(filnamn_inn.with_name(f"{filnamn_inn.stem}_ribs.hdf5"),'r') as f:
    ribs = [Rib(rib) for rib in np.asarray(f['ribs'])]

from matplotlib.patches import Polygon, Rectangle, FancyBboxPatch
for i,rib in enumerate(ribs):
    rib_indicator = axes.add_patch(Polygon(rib.vertices, facecolor='none', edgecolor = 'red', linewidth = .7, zorder=i+5))

piler = axes.quiver(x_lang,y_lang,Umx[0,:],Vmx[0,:],scale=2000, zorder=10)
# # draw_rect(axes, ribs ,color='white',fill=False)
# datapunkt = axes.scatter(txy_korrigert[:,1], txy_korrigert[:,2],c='red',s=((U_korrigert[:,0]**2+U_korrigert[:,1]**2)**0.5)*0.03+.1, zorder=10)
axes.set_xlim((20,60))
axes.set_ylim((-25,10))
cb = fig.colorbar(field, ax=axes)
axes.set_xlabel(r'$x$ [mm]')
axes.set_ylabel(r'$y$ [mm]')
cb.set_label(r"$\vec{U}$ [mm/s]")#, fontsize=18)

axes.add_patch(FancyBboxPatch((21,-24.5), 27,2.8, facecolor='white', alpha=0.8, boxstyle='Round',edgecolor='none', zorder=14))
axes.quiverkey(piler, label='PIV data magnitude and direction', X=0.1, Y=.05, U=100, labelpos='E').set_zorder(15)
plt.show()

filnamn = f"ribs_original_data.png"
fig.savefig(filnamn)


# fig, axes = plt.subplots(figsize=figureSize,dpi=myDPI) #(width/myDPI,height/myDPI)
# # axes.quiver(x_lang,y_lang,Umx[0,:],Vmx[0,:],scale=10000)
# # axes.plot(ribs.T[0],ribs.T[1])
# field = axes.imshow(((ny_U.T[0]**2+ny_U.T[1]**2)**0.5).reshape(height,width), extent=[x_min,x_max, y_min, y_max], interpolation='none', origin='lower')
# datapunkt = axes.plot(xy_utval[:,0], xy_utval[:,1],'y.',markersize=1)
# axes.set_xlim((20,60))
# axes.set_ylim((-25,10))
# # datapunkt = axes.plot(xy_utval[:,0], xy_utval[:,1],'.',markersize=1)
# with h5py.File(filnamn_inn.with_name(f"{filnamn_inn.stem}_ribs.hdf5"),'r') as f:
#     ribs = [Rib(rib) for rib in np.asarray(f['ribs'])]
# draw_rect(axes, ribs ,color='white',fill=False)
# fig.colorbar(field, ax=axes)
# plt.show()

# from scipy.spatial import Voronoi
# nonan = np.invert(np.isnan(Umx[0]))
# x_y = np.stack((x_lang[nonan],y_lang[nonan])).T

# vor = Voronoi(xy_utval)
# # matplotlib.rcParams["figure.dpi"] = 100
# # fig = voronoi_plot_2d(vor, show_vertices=False,show_points=False)
# plt.axis('equal')
# plt.xlim((20,60))
# plt.ylim((-25,10))
# plt.plot(ribs.T[0],ribs.T[1])
# plt.show()

#  np.savetxt("xy_t3.csv",np.stack((x_lang,y_lang,Umx[0],Vmx[0])).T ,delimiter=",") 
# %%
