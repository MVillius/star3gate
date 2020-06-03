#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:21:53 2020

@author: Vil
"""

import numpy as np
# import qutip as qt
import matplotlib.pyplot as plt
# import os
# import matplotlib as mpl
# from matplotlib import cm

from sg1 import *

compute = True
fitting = False
saving  = False


''' Checking Hilbert space dimension using qubit absorption spectra '''
# Set parameters
Ltype = 'Hybrid'
dims  = [5,10,15,20]
w_cav = 6e3
d_cp  = 1e3
gamma = 1e-3*d_cp
g     = 1*gamma
S     = 12

if compute:
    # Do not touch
    span  = 50
    w_drive_list = []
    sreg_list, sweird_list = [], []
    for ii, dim in enumerate(dims):
        hilbert(dim)
        sys   = system(w_cav=w_cav, g=g, gamma=gamma, kappa=gamma)
        print('------------------------')
        e_pump, w_pump = feeder(S, d_cp, w_cav)
        sys.twopi_update({'e_pump':e_pump, 'w_pump':w_pump})
        sys.args.update(d_qp=sys.Omega())

        w_drive = -sys.Omega() + sys.args['gamma']*np.linspace(-span,span,1001)
    #    w_drive = 2*np.pi*np.linspace(-15e3,15e3,50001)
        w_drive_list.append(w_drive)

        if Ltype is 'Hybrid':
            sys.Dynamics(Htype='LRsqueez', Dtype='regular', isDrive=False)
        elif Ltype is 'Squeeze':
            sys.Dynamics(Htype='LRsqueez', Dtype='weird', isDrive=False)
        elif Ltype is 'Pump':
            sys.Dynamics(Htype='LRrot', Dtype='regular', isDrive=False)
        sweird_list.append(qubit_spectrum(sys, w_drive, solver='es'))

    print('done')



''' Plotting '''
plt.close('all')
fig1, ax1 = plt.subplots(1, figsize=(10,6))
fig2, ax2 = plt.subplots(1, figsize=(10,6))

dist    = np.zeros(len(dims))
dist2   = np.zeros(len(dims))
for ii, dim in enumerate(dims):
    e_pump, w_pump = feeder(S, d_cp, w_cav)
    sys.twopi_update({'e_pump':e_pump, 'w_pump':w_pump})

    wdriv = w_drive_list[ii]
    xaxis = (wdriv+sys.Omega())/sys.args['gamma']
#    xaxis = wdriv/2/np.pi
    s_abs = sweird_list[ii]/np.max(sweird_list[ii])
    ax1.plot(xaxis, s_abs, '*', label='dim=%.f'%(2*dim), color='C'+str(ii))

    if fitting:
        p0 = [-sys.Omega(),
              sys.args['gamma'],
              1,
              sys.args['g']*np.exp(sys.r_param())]

        popt, pcov = sc.curve_fit(func_double_lor, wdriv, s_abs, p0=p0)
        ax1.plot(xaxis, func_double_lor(wdriv,*popt), '--', color='C'+str(ii))
        print(np.array(p0)/2/np.pi)
        print(np.array(popt)/2/np.pi)

    if ii-len(dims)+1:
        s_next = sweird_list[ii+1]/np.max(sweird_list[ii+1])
    else:
        s_next = s_abs
    dist[ii] = sum(s_abs/s_next)/len(s_abs)
    dist2[ii] = np.sqrt(sum(np.abs(s_abs**2-s_next**2)))
    # ax2.plot(2*dims[ii], dist[ii], 'o', markersize=8)
    ax2.plot(2*dims[ii], dist2[ii], 'x', markersize=8)

# ax1.set_xlim([-20,20])

ax1.set_ylabel(r'Absorption Spectrum')
ax1.grid(linestyle='--')
ax1.set_xlabel(r'$(\omega-\Omega_c[r])/\gamma$')
ax1.legend()
ax1.set_title(Ltype+r': $\gamma=$%.f, $ g=$%.1f, $ \delta_c$=%.f, $ S=$%.fdB'%(gamma,g,d_cp,S))

ax2.set_ylabel(r'$d_N$')
ax2.grid(linestyle='--')
ax2.set_xlabel(r'N=dim($H$)')
# ax2.set_ylim([0.45,1.05])
ax2.set_title(Ltype+r': $\gamma=$%.f, $ g=$%.1f, $ \delta_c$=%.f, $ S=$%.fdB'%(gamma,g,d_cp,S))

plt.show()
if saving:
    fig1.tight_layout()
    fig1.savefig('truncation_'+Dtype+'_g%.1f_S%.fdB.png'%(g,S))
    fig2.tight_layout()
    fig2.savefig('distance_'+Dtype+'_g%.1f_S%.fdB.png'%(g,S))