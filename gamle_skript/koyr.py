# -*- coding: utf-8 -*-
"""
Created on Tue May  4 11:53:21 2021

@author: havre
"""


#%% Førebu

Umx_lang, Vmx_lang = get_velocity_data(20)
tri = hent_tre()

#%% Utfør

stein = Particle([-88.5,87],1)
f_retur = stein.f(0,[[-88.5,87],[0,0]], tri, Umx_lang, Vmx_lang)

solve_ivp(stein.f,(0,2), np.array([-88.5,87,0,0]),t_eval=np.arange(0,2,0.1), args=(tri, Umx_lang, Vmx_lang))
