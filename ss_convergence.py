#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:30:39 2020

@author: Vil
"""
import numpy as np
# import qutip as qt
import matplotlib.pyplot as plt
# import os
# import matplotlib as mpl
# from matplotlib import cm

from sg1 import *

compute = False

''' All 3 dynamic steadystates & comparison '''
# Set parameters
dlist = list(range(10,30))
dlist = np.array(dlist[0::10])
w_cav = 6e3
d_cp  = 1e3
gamma = 1e-3*d_cp
g     = 0.5*gamma
Slist = np.logspace(-2,1,21)
nth   = 0

if compute:
    # Do not touch
    shybrid, spump, ssqueez = [], [], []
    sUhybrid, sUsqueez = [], []
    for jj, Na in enumerate(dlist):
        print('------------------------')
        hilbert(Na)
        sys   = system(w_cav=w_cav, g=g, gamma=gamma, kappa=gamma)
        sH, sP, sS = [], [], []
        sUh, sUs = [], []
        for ii, S in enumerate(Slist):
            e_pump, w_pump = feeder(S, d_cp, w_cav, printing=False)
            sys.twopi_update({'e_pump':e_pump, 'w_pump':w_pump-1000})
            sys.args.update(d_qp=sys.Omega())

            sys.Dynamics(Htype='LRsqueez', Dtype='regular', isDrive=False, nth=nth)
            sH.append(steadystate(sys)) # Hybrid
            sys.Dynamics(Htype='LRrot', Dtype='regular', isDrive=False, nth=nth)
            sP.append(steadystate(sys)) # Pump
            sys.Dynamics(Htype='LRsqueez', Dtype='weird', isDrive=False, nth=nth)
            sS.append(steadystate(sys)) # Squeeze

            sys.Dynamics(Htype='LRUsqueez', Dtype='regular', isDrive=False, nth=nth)
            sUh.append(steadystate(sys)) # Unitary Hybrid
            sys.Dynamics(Htype='LRUsqueez', Dtype='Uweird', isDrive=False, nth=nth)
            sUs.append(steadystate(sys)) # Unitary Squeeze

        shybrid.append(sH)
        spump.append(sP)
        ssqueez.append(sS)
        sUhybrid.append(sUh)
        sUsqueez.append(sUs)
    print('done')



''' Plotting '''
plt.close('all')
fig1, ax1 = plt.subplots(1, figsize=(12,6))
fig2, ax2 = plt.subplots(1, figsize=(12,6))
#fig3, ax3 = plt.subplots(1, figsize=(12,6))
#fig4, ax4 = plt.subplots(1, figsize=(18,6))
saving  = False

Nd = len(dlist)
NS = len(Slist)

pump2squeez = np.zeros((Nd, NS))
squeez2Usqueez = np.zeros((Nd, NS))


for jj, Na in enumerate(dlist):
    for ii, S in enumerate(Slist):
        e_pump, w_pump = feeder(S, d_cp, w_cav, printing=False)
        sys.twopi_update({'e_pump':e_pump, 'w_pump':w_pump})
        U = sys.Usqueeze(Na=Na)
        sp = spump[jj][ii]
        ss = U.dag()*ssqueez[jj][ii]*U
        pump2squeez[jj][ii] = qt.fidelity(sp,ss)

        sus = sUsqueez[jj][ii]
        ss  = ssqueez[jj][ii]
        squeez2Usqueez[jj][ii] = qt.fidelity(sus,ss)

    ax1.semilogx(Slist, pump2squeez[jj], '*', color='C'+str(jj%10), label=r'dimH=%.f'%(2*dlist[jj]))

    # ax2.semilogx(Slist, hyb2Uhyb[jj], '*', color='C'+str(jj%10), label=r'dimH=%.f'%(2*dlist[jj]))

#[ax1.semilogx(Slist, pump2squeez[jj], '*', label=r'dimH=%.f'%(2*dlist[jj])) for jj in range(Nd)]

ax1.set_ylabel(r'$\mathcal{F}$ $[\rho_p, U^\dagger\rho_sU]$')
ax1.grid(linestyle='--')
ax1.set_xlabel(r'$S$ (dB)')
ax1.legend()
ax1.set_title(r'Pump dynamics $\partial_t\rho_p=\mathcal{L}_p\rho_p$ vs Squeeze dynamics $\partial_t\rho_s = \mathcal{L}_s\rho_s$')

ax2.set_ylabel(r'$\mathcal{F}$ $[\rho_s, \tilde{\rho}_s]$')
ax2.grid(linestyle='--')
ax2.set_xlabel(r'$S$ (dB)')
ax2.legend()
ax2.set_title(r'Analytical Squeeze dynamics vs Numerical Squeeze Dynamics')



plt.show()
if saving:
    fig1.tight_layout()
    fig1.savefig('Comparision_Hybrid')
