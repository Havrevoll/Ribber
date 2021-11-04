# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 21:24:39 2021

@author: havrevol
"""
import random
import numpy as np
from scipy import interpolate
from math import log2, pi

from scipy.linalg.special_matrices import kron
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams['mathtext.fontset'] = 'stix'


# Skal laga kode for å utvida tabellen med 1000 element mellom kvar verdi.
def get_PSD_part(tal=1, PSD=None, rnd_seed=1):

    if PSD is None:
        PSD = np.array([[0.05,  0.00], #Det er ukjend kor 0 % er, altså kva som er minste storleik. Må anta at det er ganske lite.
                    [0.149, 0.02],	
                    [0.297, 0.095],	
                    [0.594, 0.28],	
                    [1.188, 0.55],	
                    [2.376, 0.79],	
                    [4.752, 0.91],	
                    [9.504, 0.98],	
                    [12.0,  1.00]])
    
    element = 1000

    rho = 2.650e-6 # 2.11.2021: Her stod det 2650.0/1e6. Men det er jo feil. Det skal vera 2.65e-6 i kg/mm³. 
    PSD_full = np.zeros((element*(len(PSD)-1)+1, 4))

    for i in range(len(PSD)-1):
        diff = PSD[i+1,1]- PSD[i,1]
        
        for j in range(element):
            PSD_full[i*element + j, 1] = PSD[i,1]+ diff * j / element
            
        PSD_full[i*element, 0] = PSD[i, 0]
        
        for j in range(element):
            PSD_full[i*element + j, 0] = 2**(log2(PSD[i,0]) + (log2(PSD[i+1,0]) - log2(PSD[i,0])) * (PSD_full[i*element + j, 1] - PSD[i,1]) / (PSD[i+1,1] - PSD[i,1]))
            
    PSD_full[-1,0] = PSD[-1,0]
    PSD_full[-1,1] = PSD[-1,1]
    PSD_full[:,2] = np.concatenate(([0],np.diff(PSD_full[:,1])/(pi/6 * rho * PSD_full[1:,0]**3)))
    sum_partiklar = np.sum(PSD_full[:,2])
    PSD_full[:,3] = np.cumsum(PSD_full[:,2])/sum_partiklar
    PSD_full = PSD_full.T
    # PSD_full er ein tabell med desse kolonnene: 0: diameter. 1: kumulert sannsynsfordeling, oppdelt i 1000 element for kvar av dei opphavlege inndelingane i PSD.
    # 2: Tal på partiklar i eitt kilo av partiklar av den gjevne diameteren med den gjevne fordelinga som er i PSD. 3: kumulert prosentvis tal på partiklar delt på samla tal på partiklar.
    PSD = PSD.T

    f = interpolate.interp1d(PSD_full[3], PSD_full[0])

    np.random.seed(rnd_seed)
    a = np.random.uniform(0.0, 1.0, size = tal)

    korndiameter = f(a)

    return korndiameter

def PSD_plot(part_array):
    korndiameter = part_array

    masse = korndiameter**3 /6 * pi * 2650 * 1e-9 # masse i kg,tettleik i kg/mm³.

    bins = 2.0**(np.arange(-4,5))
    massebins = np.concatenate(([0], bins**3 /6 * pi * 2650 * 1e-9))
    # print( "bins: ", bins)

    x_diameter = np.histogram(korndiameter, np.concatenate(([0], bins)))
    x_masse = np.histogram(masse, massebins)
    sum_masse = np.sum(masse)
    masse_per_gradering = np.array([np.sum(masse[(masse > massebins[i]) & (masse < massebins[i+1])]) for i in range(0,len(massebins)-1)])/sum_masse

    x1 = np.cumsum(masse_per_gradering)

    myDPI = 300

    fig, ax = plt.subplots(figsize=(1190/myDPI,800/myDPI),dpi=myDPI)

    ax.semilogx(bins, x1, color="blue", label="kornfordeling")

    # ax.semilogx(PSD_full[0], PSD_full[3], color="red", label="original")
    ax.set_xticks(bins)
    ax.set_xticklabels(bins)
    # ax.plot(x, y2, color="red", label="y'(x)")
    # ax.plot(x, y3, color="green", label="y”(x)")
    ax.set_xlabel("d [mm]")
    ax.set_ylabel("% tal partiklar passert")
    # ax.legend()

    # logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))

    # n, bins, patches = ax.hist(korndiameter, cumulative=True, logx=True, bins=bins)