# -*- coding: utf-8 -*-
'''køyr funksjonar som plottingar(fil['vassføringar'])'''

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams['mathtext.fontset'] = 'stix'
from matplotlib import animation, colors
import matplotlib as mpl
from matplotlib.patches import Rectangle

import numpy as np
from scipy import interpolate
from scipy.integrate import solve_ivp  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#r179348322575-1
from scipy.optimize import fsolve
import h5py
import re
import scipy.stats as stats

# from IPython.display import clear_output


from math import ceil, floor, log, sqrt
# import os.path.join as pjoin

# fil = h5py.File("D:/Tonstad/alle.hdf5", 'a')
# vass = fil['vassføringar']


discharges = [20,40,60,80,100,120,140]

h= -5.9

def reshape(dataset):
    '''
    Ein metode som tek inn eit datasett og gjer alle reshapings-tinga for x og y, u og v og Re.

    '''
    (I,J)=(int(np.array(dataset['I'])),int(np.array(dataset['J'])))
    
    Umx = np.array(dataset['Umx'])
    Umx_reshape = Umx.reshape((len(Umx),J,I))[:,1:114,1:125]
    
    Vmx = np.array(dataset['Vmx'])
    Vmx_reshape = Vmx.reshape((len(Vmx),J,I))[:,1:114,1:125]
    
    x = np.array(dataset['x'])
    y = np.array(dataset['y'])
    
    x_reshape = x.reshape(J,I)[1:114,1:125]
    y_reshape = y.reshape(J,I)[1:114,1:125]
    
    nonanUmx = np.invert(np.isnan(Umx))
    nonanUmx_reshape = np.invert(np.isnan(Umx_reshape))
    nonanVmx = np.invert(np.isnan(Vmx))
    nonanVmx_reshape = np.invert(np.isnan(Vmx_reshape))
    
    return

def sjekktull():
    for i in range(113):
        for j in range(124):
            if np.any(nonanUmx_reshape[:,i,j]):
                if np.all(nonanUmx_reshape[:,i,j]):
                    print("ingen false i ",i,j)
                else:
                    print( "nokon false i ",i,j)
    
    return

# def lag_mindredatasett(case):
#     '''
#     Ein metode som tek inn eit case frå den store alle.hdf5-fila og tek ut berre dei viktige delane og lagrar i ei ny fil med maks kompresjon. Må laga ein tilsvarande metode for å henta fram data og laga alle dei bearbeida versjonane, på eit per case-basis.
#     '''
#     utfilnamn = "D:/Tonstad/utvalde/Q{}.hdf5".format(case)
#     print(utfilnamn)
        
    
#     with h5py.File(utfilnamn, 'a') as utfil:
#         for sett in ['nonanindex', 'x', 'y', 'nonanu', 'nonanv', 'nonancoords', 'vort', 'Umx', 'Vmx']:
#             utfil.create_dataset(sett, data=vass[case][sett], compression="gzip", compression_opts=9)
            
#         for sett in ['I', 'J', 'd_l', 'filnamn', 'flow_case', 'i', 'j']:
#             utfil.create_dataset(sett, data=vass[case][sett])
#     print("hei")

def vegglov(u_star, y, v):
    # nu = 1 # 1 mm²/s
    y = y - h
    ks = .0025
    return 1/0.4 * log(30 * y / ks) - v/u_star

def finn_u(y,v):
    u = np.zeros(127)
    
    for i in np.arange(0,67):
        u[i]= fsolve(vegglov, 2, args=(y[i],v[i]))
     
    return u

def draw_rect(axes,color='red'):
    axes.add_patch(Rectangle((-62.4,-9.56),50,8,linewidth=2,edgecolor=color,facecolor='none'))
    axes.add_patch(Rectangle((37.6,-8.5),50,8,linewidth=2,edgecolor=color,facecolor='none'))

def draw_shade(axes, x0=0, x=430, color='red'):
    axes.add_patch(Rectangle((x0,-9.8),x,10.8,linewidth=2,edgecolor='none',facecolor='lightcoral'))

def ranges():
    y_range = np.s_[0:114]
    x_range = np.s_[40:108]
    
    piv_range = np.index_exp[y_range,x_range]
    
    return piv_range

def hentdata(flow_case):
    
    filnamn =  "D:/Q{}.mat".format(flow_case)
    
    fil = h5py.File(filnamn, 'r') # https://docs.h5py.org/en/stable/quick.html#quick
     # list(f.keys())
     # ['#refs#', 'LEUC', 'LSUC', 'UEUC', 'USUC', 'Umx', 'Vmx', 'x', 'y']
     # x.shape
     
    x = fil['x'][0]
    y = fil['y'][0]
    
    Umx = np.array(fil['Umx'])*1000
    Vmx = np.array(fil['Vmx'])*1000
    V_mag = np.sqrt(Umx * Umx + Vmx * Vmx)
    
    u_bar = np.nanmean(Umx,0)
    v_bar = np.nanmean(Vmx,0)
    
    up = Umx - u_bar
    vp = Vmx - v_bar
    
    up_sq = (up*up)
    vp_sq = (vp*vp)
    
    up_sq_bar = np.nanmean(up_sq,0)
    vp_sq_bar = np.nanmean(vp_sq,0)
    
    Re_stressp = -1*up*vp*1e-3
    Re_stressm = np.nanmean(Re_stressp ,0)
    
    I = 126  # horisontal lengd
    J = 127  # vertikal lengd
    # m = 3 #define number of columns to be cut at the lefthand side of the window
    # n = 2 #define number of columns to be cut at the righthand side of the window
    # b = 3 #define number of columns to be cut at the bottom of the window
    # t = 3 #define number of columns to be cut at the top of the window
    
    x_reshape1 = x.reshape((J,I))      # x_reshape=(x_reshape1(t+1:J-b,m+1:I-n))
    y_reshape1 = y.reshape((J,I))      # y_reshape=(y_reshape1(t+1:J-b,m+1:I-n));
    u_reshape1 = u_bar.reshape((J,I))  # u_reshape=(u_reshape1(t+1:J-b,m+1:I-n));
    v_reshape1 = v_bar.reshape((J,I))  # v_reshape=(v_reshape1(t+1:J-b,m+1:I-n));
    v_bar_mag = np.sqrt(u_reshape1 * u_reshape1 + v_reshape1 * v_reshape1)
    
    Umx_reshape = Umx.reshape((len(Umx),J,I))
    Vmx_reshape = Vmx.reshape((len(Vmx),J,I))
    V_mag_reshape = V_mag.reshape((len(V_mag),J,I))
    t_3d,y_3d,x_3d = np.meshgrid(np.arange(3600.0),y_reshape1[:,0],x_reshape1[0,:],indexing='ij')
    
    Re_str_reshape1 = Re_stressm.reshape((J,I))   #   Re_str_reshape=(Re_str_reshape1(t+1:J-b,m+1:I-n));
    up_sq_bar_reshape1 = up_sq_bar.reshape((J,I))  #   up_sq_bar_reshape=(up_sq_bar_reshape1(t+1:J-b,m+1:I-n));
    vp_sq_bar_reshape1 = vp_sq_bar.reshape((J,I)) #   vp_sq_bar_reshape=(vp_sq_bar_reshape1(t+1:J-b,m+1:I-n));

    u_profile = np.nanmean(u_reshape1,1)


    vort = np.zeros((3600,J,I))

    d_l = 186/I

    for t in np.arange(3600):
        if t % 100 == 0: 
            print(t, end = '')
            print(' ', end = '')
        for j in np.arange(1,J-1):
            for i in np.arange(1,I-1):
                vort[t,j,i] = (Umx_reshape[t,j-1,i]-Umx_reshape[t,j+1,i]) / 2 + (Vmx_reshape[t,j,i+1]-Vmx_reshape[t,j,i-1]) / 2
                
    
    vort = vort/d_l
    
    vort_bar = np.nanmean(vort,0)
    
    
    # https://stackoverflow.com/questions/59071446/why-does-scipy-griddata-return-nans-with-cubic-interpolation-if-input-values

    #Her tek me vekk alle nan frå x, y og uv.
    nonanindex=np.invert(np.isnan(x)) * np.invert(np.isnan(y)) * np.invert(np.isnan(u_bar)) * np.invert(np.isnan(v_bar))
    nonancoords= np.transpose(np.vstack((x[nonanindex], y[nonanindex])))
    nonanu = u_bar[nonanindex]
    nonanv = v_bar[nonanindex]
    
    nonanxindex = np.invert(np.isnan(Umx))
    nonanyindex = np.invert(np.isnan(Vmx))
    
    

    loc = locals()
    return dict([(i,loc[i]) for i in loc])

# def calc_Re_stress(case):
#     up = case['up']
#     vp = case['vp']
    
#     Re_stressp=-1*np.array(up)*np.array(vp)*1e-3
    
#     Re_stressm = np.nanmean(Re_stressp,0)
    
#     Re_str_reshape1 = Re_stressm.reshape((127,126))
    
#     data = case['Re_str_reshape1']       # load the data
#     data[...] = Re_str_reshape1          # assign new values to data
#     rep = case['Re_stressp']
#     rep[...] = Re_stressp
#     rem = case['Re_stressm']
#     rem[...] = Re_stressm
#     fil.flush()
    
    
def calc_u_profile(case):

    
    piv_range = ranges()
    y_range = piv_range[0]
    
    # x_reshape1 = np.array(case['x_reshape1'])
    # x=x_reshape1[piv_range]
    y_reshape1 = np.array(case['y_reshape1'])
    y=y_reshape1[piv_range]


    u_reshape1 = np.array(case['u_reshape1'][piv_range])
    # v_reshape1 = np.array(case['v_reshape1'][piv_range])
    
    u_profile = np.nanmean(u_reshape1,1)
    
    gml_u_profile = np.array(case['u_profile'][y_range])
    
    myDPI = 300
    fig, axes = plt.subplots(figsize=(900/myDPI,900/myDPI),dpi=myDPI)
    
    axes.plot(u_profile, y[:,0], linewidth=.8, label="ny")
    axes.plot(gml_u_profile, y[:,0], linewidth=.8, label="gml")
    
    draw_shade(axes)
    # axes.set_title("Reynolds' turbulent shear stress")
    axes.set_xlabel(r'$u$ [mm/s]')
    axes.set_ylabel(r'$y$ [mm]')
    axes.legend()
    # axes[1].set_xlim(0,500)
    plt.tight_layout()
    
    filnamn = "u_nyoggamal.png"
    
    fig.savefig(filnamn)
    plt.close()
    
    
def fyllopp(discharges):
    cases = {}
    for q in discharges:
        print("byrja på Q", q)
        cases[q]=hentdata(q)
    return cases

    # plt.show()

# def f(t,yn, method='nearest'): # yn er array-like, altså np.array(xn,yn)
#     return np.hstack([interpolate.griddata((x,y), u_bar, yn, method=method), interpolate.griddata((x,y), v_bar, yn, method=method)]) 

def lag_nonan(case):
    print(case.name)
   
        
    # case.create_dataset('nonanx', data=nonanx, compression="gzip", compression_opts=9)
    # case.create_dataset('nonany', data=nonany, compression="gzip", compression_opts=9)
    # case.create_dataset('nonanu', data=nonanu, compression="gzip", compression_opts=9)
    # case.create_dataset('nonanv', data=nonanv, compression="gzip", compression_opts=9)
    # case.create_dataset('trix', data=trix, compression="gzip", compression_opts=9)
    # case.create_dataset('triy', data=triy, compression="gzip", compression_opts=9)
    
       
import scipy.spatial.qhull as qhull

# https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids

def interp_weights(tri, uv):
   
    simplex = tri.find_simplex(uv)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uv - temp[:, 2]
    bary = np.einsum('njk,nk->nj', temp[:2, :], delta)
    wts = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
    ret = np.einsum('nj,nj->n', np.take(uv, vertices), wts)

def interpol(coords, values, yn):
    
    #ret[np.any(wts < 0, axis=1)] = fill_value
    return ret



def interp_lin_near(coords,values, yn):
    new = interpolate.griddata(coords, values, yn, method='linear')
    if np.isnan(new):
        return interpolate.griddata(coords, values, yn, method='nearest')
    else:
        return new

def lag_ft(case, t_start, t_end, fps=20):
    ''' Funksjon for å laga eit kontinuerleg vektorfelt '''
    
    nonanxindex = np.array(case['nonanxindex'])
    nonanyindex = np.array(case['nonanyindex'])
    Umx = np.array(case['Umx'])
    Vmx = np.array(case['Vmx'])
    x = np.array(case['x'])
    y = np.array(case['y'])
    
    nonanx={}
    nonany={}
    nonanu={}
    nonanv={}
    trix={}
    triy={}
    
    for t in np.arange(t_start*fps, t_end*fps):
        
        nonanx[t]=np.vstack((x[nonanxindex[t]],y[nonanxindex[t]])).T
        nonany[t]=np.vstack((x[nonanyindex[t]],y[nonanyindex[t]])).T
        nonanu[t]=Umx[t,nonanxindex[t]]
        nonanv[t]=Vmx[t,nonanyindex[t]]
        trix[t] = qhull.Delaunay(nonanx[t])
        triy[t] = qhull.Delaunay(nonany[t])
        print(t, end = '')
        print(' ', end = '')
        
    def f_t(t, yn):
        
        if yn[0] > 100:
            return np.hstack([0,0])
        
        t_0 = floor(t)
        t_1 = ceil(t)
        
        if t_0 == t_1:
            u_0 = interp_lin_near((nonanx[t_0], nonany[t_0]), nonanu[t_0], yn, tri) #interpolate.griddata((x[nonanxindex[t_0]], y[nonanxindex[t_0]]), Umx[t_0,nonanxindex[t_0,:]], yn)
            v_0 = interp_lin_near((x[nonanyindex[t_0]], y[nonanyindex[t_0]]), Vmx[t_0,nonanyindex[t_0,:]], yn)
            
            return np.hstack([u_0,v_0])
        
        u_0 = interp_lin_near((x[nonanxindex[t_0]], y[nonanxindex[t_0]]), Umx[t_0,nonanxindex[t_0,:]], yn)
        v_0 = interp_lin_near((x[nonanyindex[t_0]], y[nonanyindex[t_0]]), Vmx[t_0,nonanyindex[t_0,:]], yn)
        
        u_1 = interp_lin_near((x[nonanxindex[t_1]], y[nonanxindex[t_1]]), Umx[t_1,nonanxindex[t_1,:]], yn)
        v_1 = interp_lin_near((x[nonanyindex[t_1]], y[nonanyindex[t_1]]), Vmx[t_1,nonanyindex[t_1,:]], yn)
        
        u_x = u_0 + (t- t_0) * (u_1 - u_0) / (t_1 - t_0) 
        v_y = v_0 + (t- t_0) * (v_1 - v_0) / (t_1 - t_0) 
        
        print("ferdig med interpolering")
        print(t,yn,np.hstack([u_x,v_y]))
        return np.hstack([u_x,v_y])
    return f_t
    
def rk(t0, y0, L, f, h=0.02):
    ''' Heimelaga Runge-Kutta-metode '''
    N=int(L/h)

    t=[0]*N # initialize lists
    y=[0]*N # initialize lists
    
    t[0] = t0
    y[0] = y0
    
    for n in range(0, N-1):
        #print(n,t[n], y[n], f(t[n],y[n]))
        k1 = h*f(t[n], y[n])
        k2 = h*f(t[n] + 0.5 * h, y[n] + 0.5 * k1)
        k3 = h*f(t[n] + 0.5 * h, y[n] + 0.5 * k2)
        k4 = h*f(t[n] + h, y[n] + k3)
        
        if (np.isnan(k4+k3+k2+k1).any()):
            #print(k1,k2,k3,k4)
            return t,y
        
        t[n+1] = t[n] + h
        y[n+1] = y[n] + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
        
    return t,y


def lag_sti(case, t_start,t_end,fps=20):
    

    
    f_t = lag_ft(case, t_start,t_end,fps=20)
    
    p_x,p_y = np.meshgrid([-90,-200],[85,75,65,55,45,35,25,15,5,0,-20,-30,-40,-50,-60])
    
    p_x = p_x.T.reshape(-1)
    p_y= p_y.T.reshape(-1)
    
    sti = []
    
 
    for par in np.column_stack((p_x,p_y)):
        sti.append(solve_ivp(f_t, [t_start,t_end*fps], par, t_eval=np.arange(t_start, t_end*fps, 1)))
        
    sti_ny=[]
    
    for el in sti:
        sti_ny.append(el.y.T)
    
    return np.array(sti_ny)
    

def sti_animasjon(case):
    
    x_reshape1= np.array(case['x_reshape1'])
    y_reshape1 = np.array(case['y_reshape1'])
    V_mag_reshape = np.array(case['V_mag_reshape'])
    sti = np.array(case['sti'])
    
    fig, ax = plt.subplots()
    
    field = ax.imshow(V_mag_reshape[0,:,:], extent=[x_reshape1[0,0],x_reshape1[0,-1], y_reshape1[-1,0], y_reshape1[0,0]])
    particle, =ax.plot(sti[:,0,0], sti[:,0,1], 'ro')
    ax.set_xlim([x_reshape1[0,0],x_reshape1[0,-1]])
    draw_rect(ax)
    
    def nypkt(i):
        field.set_data(V_mag_reshape[i,:,:])
        particle.set_data(sti[:,i,0], sti[:,i,1])
        return field,particle
    
    print("Skal byrja på filmen")
    #ax.axis('equal')
    ani = animation.FuncAnimation(fig, nypkt, frames=np.arange(1,600),interval=50)
    plt.show()
    print("ferdig med animasjon, skal lagra")
    
    filnamn = "stiQ{}.mp4".format(re.split(r'/',case.name)[-1])
    ani.save(filnamn)
    plt.close()


def lagra(dataset):
    f = h5py.File('alle2.hdf5','w')
    
    vassf = f.create_group("vassføringar")
    
    for q in dataset:
        gr = vassf.create_group(str(q))
        for k in dataset[q]:
            gr.create_dataset(k, data=dataset[q][k], compression="gzip", compression_opts=9)
    f.close()
  
def runsTest(l, l_median): 
  
    runs, n1, n2 = 0, 0, 0
      
    # Checking for start of new run 
    for i in range(len(l)): 
          
        # no. of runs 
        if (l[i] >= l_median and l[i-1] < l_median) or (l[i] < l_median and l[i-1] >= l_median): 
            runs += 1  
          
        # no. of positive values 
        if(l[i]) >= l_median: 
            n1 += 1   
          
        # no. of negative values 
        else: 
            n2 += 1   
  
    runs_exp = ((2*n1*n2)/(n1+n2))+1
    stan_dev = sqrt((2*n1*n2*(2*n1*n2-n1-n2))/(((n1+n2)**2)*(n1+n2-1))) 
  
    z = (runs-runs_exp)/stan_dev 
  
    return z 



 
