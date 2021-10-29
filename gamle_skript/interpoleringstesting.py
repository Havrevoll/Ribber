# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 14:50:10 2021

For å testa ut interpolering og gradientar

@author: havrevol
"""

import numpy as np

from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import matplotlib as mpl
import random


def interpoler(x, tri, values):
    simplex = tri.find_simplex(x)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    
    d=2
    delta = x - temp[:,d]
    bary = np.einsum('njk,nk->nj', temp[:,:d, :], delta)
    wts = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
    
    interpolerte = np.einsum('ij,ij->i', np.take(values, vertices), wts)
    
    return interpolerte


def finn_gradient(x, tri, values, dx = 0.1):
    x_retning1 = (interpoler(x+np.array([dx,0]),tri,values) - interpoler(x,tri,values))/dx
    y_retning1 = (interpoler(x+np.array([0,dx]),tri,values) - interpoler(x,tri,values))/dx
    
    x_retning2 = (interpoler(x,tri,values) - interpoler(x-np.array([dx,0]),tri,values))/dx
    y_retning2 = (interpoler(x,tri,values) - interpoler(x-np.array([0,dx]),tri,values))/dx
    
    
    
    return (x_retning1+x_retning2) * 0.5, (y_retning1+y_retning2) * 0.5

u = np.array([[1,2,3,4],[3,4,5,6],[8,9,10,11],[4,5,6,7]])

x_coords, y_coords = np.meshgrid(np.linspace(1,4,4),np.linspace(1,4,4))

# fig, ax = plt.subplots(1, 1, figsize=(8, 6), subplot_kw={'projection': '3d'})

# norm = mpl.colors.Normalize(-abs(u).max(), abs(u).max())
# p = ax.plot_surface(x_coords, y_coords, u, rstride=1, cstride=1, linewidth=0, antialiased=False, norm=norm, cmap=mpl.cm.Blues)

points = np.column_stack((x_coords.ravel(),y_coords.ravel()))

fart = u.ravel()

tri = Delaunay(points)

# plt.triplot(points[:,0], points[:,1], tri.simplices)

x = np.array([(1.4, 2.0), (2.5, 3.05)])



mange = np.array([[random.uniform(1,4),random.uniform(1,4)],[random.uniform(1,4),random.uniform(1,4)],[random.uniform(1,4),random.uniform(1,4)],[random.uniform(1,4),random.uniform(1,4)],[random.uniform(1,4),random.uniform(1,4)],[random.uniform(1,4),random.uniform(1,4)],[random.uniform(1,4),random.uniform(1,4)],[random.uniform(1,4),random.uniform(1,4)],[random.uniform(1,4),random.uniform(1,4)],[random.uniform(1,4),random.uniform(1,4)]])


interpolerte = interpoler(x,tri,fart)

# dx = 0.1

# delta_x = x + np.array([dx,0])

# interpolerte_x = interpoler(delta_x,tri,fart)

# dydx = (interpolerte_x - interpolerte) / dx

gradient = finn_gradient(x,tri,fart)
