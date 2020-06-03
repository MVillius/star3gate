#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 17:04:53 2020

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

''' All 3 dynamic qubit spectra & comparison '''
# Set parameters
Slist = [0,3,6]
param = {'Hybrid':[10,10,10],
        'Pump':[10,40,60],
        'Squeeze':[20,20,20]}
w_cav = 6e3
d_cp  = 1e3
gamma = 1e-3*d_cp
g     = 1*gamma
span  = 50

spectrum = {key:[] for key in param.keys()}
w_drives = {key:[] for key in param.keys()}
if compute:
    for ii, S in enumerate(Slist):
        for style, dims in param.items():
            hilbert(dims[ii])
            sys = system(w_cav=w_cav, g=g, gamma=gamma, kappa=gamma)
            print('------------------------')
            e_pump, w_pump = feeder(S, d_cp, w_cav)
            sys.twopi_update({'e_pump':e_pump, 'w_pump':w_pump})
            sys.args.update(d_qp=sys.Omega())

            w_drive = -sys.Omega() + sys.args['gamma']*np.linspace(-span,span,1001)
            # w_drive = 2*np.pi*np.linspace(-1.1e3,0.1e3,50001)
            w_drives[style].append(w_drive)

            if style == 'Hybrid':
                sys.Dynamics(Htype='LRsqueez', Dtype='regular', isDrive=False, nth=0)
            elif style == 'Pump':
                sys.Dynamics(Htype='LRrot', Dtype='regular', isDrive=False, nth=0)
            elif style == 'Squeeze':
                sys.Dynamics(Htype='LRsqueez', Dtype='weird', isDrive=False, nth=0)
            spectrum[style].append(qubit_spectrum(sys, w_drive))

    print('done')




