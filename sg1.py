#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:29:43 2020

@author: Vil
"""

import numpy as np
import numpy.fft as nf
import qutip as qt
import matplotlib.pyplot as plt
import scipy.optimize as sc
import time as time
import os
import matplotlib as mpl
from matplotlib import cm

options = qt.Options()
options.nsetps = 10000

plt.style.use('ggplot')

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)   # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def hilbert(Na=15):
    ''' '''
    global I2, Ia, sm, sx, sz, a, II, vac, p_g, p_e

    I2 = qt.qeye(2)
    Ia = qt.qeye(Na)

    sm = qt.tensor(Ia, qt.sigmam())
    sx = qt.tensor(Ia, qt.sigmax())
    sz = qt.tensor(Ia, qt.sigmaz())
    a  = qt.tensor(qt.destroy(Na), I2)
    II = qt.tensor(Ia, I2)

    g_q = qt.basis(2,1)
    e_q = qt.basis(2,0)
    vac = qt.tensor(qt.basis(Na,0), g_q)
    p_g = qt.tensor(Ia, g_q*g_q.dag())
    p_e = qt.tensor(Ia, e_q*e_q.dag())

    dim = 2*Na
    print('dim(H) = %.f'%dim)

def state(n, gnd_or_exc):
    ''' '''
    if gnd_or_exc is 'exc':
        psi = sm.dag()*vac
    elif gnd_or_exc is 'gnd':
        psi = vac
    else:
        raise ValueError('Poor state definition')
    for ii in range(n):
        psi = a.dag()*psi
    return psi/psi.norm()


class system(object):

    def __init__(self, w_qubit=6e3, g=1e-1, gamma=5e0, kappa=5e0, e_drive=2e0, **kwargs):

        self.args = {'w_qubit': w_qubit,
                     'w_cav'  : w_qubit,
                     'g'      : g,
                     'gamma'  : gamma,
                     'kappa'  : kappa,
                     'w_pump' : 2*w_qubit+0.00001,
                     'e_pump' : 0,
                     'w_drive': w_qubit,
                     'e_drive': e_drive,
                     **kwargs}
        self.driv = {}         # filled when calling time_gen
        self.operators = {}    # filled when calling Dynamics

        for keys in self.args.keys():
            self.args[keys] = 2*np.pi*self.args[keys]

        self._compute()


    def _compute(self):
        ''' By default, all these parameters must be computed after the fact '''
        ''' They shall not be input parameters '''
        self.args['d_cp'] = self.args['w_cav'] - self.args['w_pump']/2
        self.args['d_qp'] = self.args['w_qubit'] - self.args['w_pump']/2
        self.args['r']    = self.r_param()
        self.args['SdB']  = self.Squeez()
        self.args['O']    = self.Omega()
        self.operators['U'] = self.Usqueeze()

    def twopi_update(self, dic):
        ''' Takes care of the two pis '''
        for keys in dic.keys():
            dic[keys] *= 2*np.pi
        self.args.update(dic)
        self._compute()

    def freqs(self, key):
        ''' Returns a frequency '''
        return self.args[key]/2/np.pi

    def print_args(self, where):
        ''' Self-explanatory '''
        skip  = ['e_drive', 'w_drive', 'gamma', 'kappa']
        title = ''
        if where is 'inline':
            largs = {'w_qubit':'wq', 'w_cav':'wc', 'g':'g',
                     'gamma':'gamma', 'kappa':'kappa',
                     'w_pump':'wp', 'e_pump':'ep',
                     'w_drive':'wd', 'e_drive':'ed',
                     'd_cp':'dcp', 'd_qp':'dqp',
                     'r':'r', 'O':'Omega', 'SdB':'SdB'}
        elif where is 'plot':
            largs = {'w_qubit':r'$\omega_q$', 'w_cav':r'$\omega_c$', 'g':r'$g$',
                     'gamma':r'$\gamma$', 'kappa':r'$\kappa$',
                     'w_pump':r'$\omega_p$', 'e_pump':r'$\varepsilon_p$',
                     'w_drive':r'$\omega_d$', 'e_drive':r'$\varepsilon_d$',
                     'd_cp':r'$\delta_{c-p}$', 'd_qp':r'$\delta_{q-p}$',
                     'r':r'$r$', 'O':r'$\Omega$', 'SdB':r'$S_{dB}$'}
        for keys in self.args.keys():
            val = self.args[keys]/2/np.pi
            if keys is 'r':
                val *= 2*np.pi
            if (keys in skip):
                continue
            elif keys is 'e_pump':
                title += largs[keys]+'=%.4f,  '%val
            else:
                title += largs[keys]+'=%.2f,  '%val
        return title[:-3]

    def w_larmor(self):
        ''' Larmor frequency '''
        ''' Must be called after time_gen '''
        ''' w_drive is a np.array() '''
        ed = self.driv['e_drive']
        wq = self.args['w_qubit']
        wd = self.driv['w_drive']
        return np.sqrt(ed**2+(wq-wd)**2)

    def p_rabi(self):
        ''' Maximum excited probability for dissipation-less system '''
        ''' Must be called after time_gen '''
        ''' w_drive is a np.array() '''
        ed = self.driv['e_drive']
        wq = self.args['w_qubit']
        wd = self.driv['w_drive']
        return (ed/w_larmor(args))**2

    def Purcell(self):
        ''' Purcell broadening '''
        k  = self.args['kappa']
        g  = self.args['g']
        wq = self.args['w_qubit']
        wc = self.args['w_cav']
        return k*g**2/(k**2/4+(wq-wc)**2)

    def r_param(self):
        ''' r param in Leroux '''
        ep = self.args['e_pump']
        dc = self.args['d_cp']
        return (1/2)*np.arctanh(ep/dc)

    def Omega(self):
        ''' Omega in Leroux '''
        dc = self.args['d_cp']
        r  = self.r_param()
        return dc/np.cosh(2*r)

    def Squeez(self):
        ''' S = e^(2r) expressed in dB'''
        r  = self.r_param()
        return 10*np.log10(np.exp(2*r))

    def Lambda(self):
        ''' Lambda = e_pump '''
        ''' Careful, e_pump must be defined first '''
        dc = self.args['d_cp']
        return d_c*np.tanh(np.log(10**(Squeez(self)/10)))

    def Usqueeze(self, Na=None):
        ''' U = exp((r/2)*(a**2-a.dag()**2) '''
        if not Na:
            Na = a.dims[0][0]
        r  = self.args['r']
        return qt.tensor(qt.squeeze(Na, r), I2)



    def Dynamics(self, Htype, Dtype, isDrive=False, nth=0):

        def Hd_coeff(t, args=self.args):
            ''' Cosine drive '''
            ''' (1/2) mimicks an exp(1j*w*t) drive in the spin locked frame '''
            return (1/2)*args['e_drive']*np.cos(args['w_drive']*t)

        def Hp_coeff(t, args=self.args):
            ''' Cosine drive '''
            ''' (-1) mimicks an (-1/2)*exp(1j*w*t) drive in the spin locked frame '''
            return (-1)*args['e_pump']*np.cos(args['w_pump']*t)

        def D0_coeff(t, args=self.args):
            ''' Qubit relaxation '''
            return np.sqrt(args['gamma'])

        def Da_coeff(t, args=self.args):
            ''' Cav relaxation '''
            return np.sqrt(args['kappa'])

#        print(self.r_param())

        if Htype is 'TLS':
            ''' '''
            H0 = self.args['w_qubit']*sz/2
            Hd = sx
            Hr = self.args['e_drive']/2*sx
            H  = [H0]
            if isDrive: H.append([Hd,Hd_coeff])

        elif Htype is 'Rabi':
            ''' '''
            H0 = self.args['w_qubit']*sz/2
            Ha = self.args['w_cav']*a.dag()*a
            Hc = self.args['g']*(a+a.dag())*sx
            Hd = a+a.dag()
            H  = [H0+Ha+Hc]
            if isDrive: H.append([Hd,Hd_coeff])

        elif Htype is 'JC':
            ''' Well-known Jesus-Christ Hamiltonian '''
            H0 = 0*II #self.args['w_qubit']*sz/2
            Ha = (self.args['w_cav']-self.args['w_qubit'])*a.dag()*a
            Hc = self.args['g']*(a*sm.dag()+a.dag()*sm)
            Hd = a+a.dag()
            H  = [H0+Ha+Hc]
            if isDrive: H.append([Hd,Hd_coeff])

        elif Htype is 'LRlab':
            ''' eqn (S8) '''
            H0 = self.args['w_qubit']*sz/2
            Ha = self.args['w_cav']*a.dag()*a
            Hc = self.args['g']*(a*sm.dag()+a.dag()*sm)
            Hd = a+a.dag()
            Hp = a**2+a.dag()**2
            H  = [H0+Ha+Hc,[Hp,Hp_coeff]]
            if isDrive: H.append([Hd,Hd_coeff])

        elif Htype is 'LRrot': #pump
            ''' eqn (1) '''
            H0 = (self.args['d_qp'])*sz/2
            Ha = (self.args['d_cp'])*a.dag()*a
            Hc = self.args['g']*(a*sm.dag()+a.dag()*sm)
            Hp = (-1/2)*self.args['e_pump']*(a**2+a.dag()**2)
            Hd = a+a.dag()
            H  = [H0+Ha+Hc+Hp]
            if isDrive: H.append([Hd,Hd_coeff])

        elif Htype is 'LRsqueez':
            ''' eqn (3) '''
            H0 = (self.args['d_qp'])*sz/2
            Ha = (self.Omega())*a.dag()*a
            Hc = (self.args['g']/2)*np.exp(self.r_param())*(a+a.dag())*sx
            HE = (-self.args['g']/2)*np.exp(-self.r_param())*(a.dag()-a)*(sm.dag()-sm)
            HD = 0*II
            Hd = a+a.dag()
            H  = [H0+Ha+Hc+HE+HD]
            if isDrive: H.append([Hd,Hd_coeff])

        elif Htype is 'LRUsqueez':
            ''' U * eqn (1) * U.dag() '''
            H0 = (self.args['d_qp'])*sz/2
            Ha = (self.args['d_cp'])*a.dag()*a
            Hc = self.args['g']*(a*sm.dag()+a.dag()*sm)
            Hp = (-1/2)*self.args['e_pump']*(a**2+a.dag()**2)
            Hd = a+a.dag()
            U  = self.operators['U']
            H  = [U*(H0+Ha+Hc+Hp)*U.dag()]
            if isDrive: H.append([Hd,Hd_coeff])

        self.operators.update(H = H)

        ### NO PHASE DAMPING np.sqrt(gamma_phi/2)*sz
        D = []
        if Dtype is 'regular':
            D.append(np.sqrt(self.args['gamma']*(1+nth))*sm)
            D.append(np.sqrt(self.args['gamma']*nth)*sm.dag())
            D.append(np.sqrt(self.args['kappa']*(1+nth))*a)
            D.append(np.sqrt(self.args['kappa']*nth)*a.dag())

        elif Dtype is 'weird':
            r = self.r_param()
            D.append(np.sqrt(self.args['gamma']*(1+nth))*sm)
            D.append(np.sqrt(self.args['gamma']*nth)*sm.dag())
            D.append(np.sqrt(self.args['kappa'])*a)
            kp = self.args['kappa']*(nth+np.sinh(r))*np.exp(+r)/2
            km = self.args['kappa']*(nth-np.sinh(r))*np.exp(-r)/2
            D.append(np.sqrt(kp)*(a+a.dag()))
            D.append(np.sign(km)*np.sqrt(np.abs(km))*(a-a.dag()))

        elif Dtype is 'Uweird':
            d = np.sqrt(self.args['gamma']*(1+nth))*sm
            d+= np.sqrt(self.args['gamma']*nth)*sm.dag()
            d+= np.sqrt(self.args['kappa']*(1+nth))*a
            d+= np.sqrt(self.args['kappa']*nth)*a.dag()
            U = self.operators['U']
            D.append(U*d*U.dag())

        self.operators.update(D = D)

#########################################
# Analysis
#########################################

def func_cos(x, w, phi, A, C):
    return A*np.cos(w*x+phi)+C

def fit_cos(x, y, printing=False):
    ''' Use Fourier Transform to feed initial values to cosine fit '''
    y_mean = np.mean(y)
    y_fft  = nf.rfft(y-y_mean, norm='ortho')
    freqs  = nf.rfftfreq(len(x), x[1]-x[0])
    i_gues = np.argmax(np.abs(y_fft))
    w_gues = 2*np.pi*freqs[i_gues]
    p_gues = np.pi*(1-np.sign(np.real(y_fft[i_gues])))/2
    A_gues = (np.max(y) - np.min(y))/2
    p0     = [w_gues, p_gues, A_gues, y_mean]
    popt, pcov = sc.curve_fit(func_cos, x, y, p0=p0)
    if printing:
        print('guess: ', np.round(p0,3))
        print('fit:   ', np.round(popt,3))
        print('------------------------')
    return popt, pcov

def func_lor(x, x0, gamma, A):
    return A/(1+((x-x0)*2/gamma)**2)

def func_double_lor(x, x0, gamma, A, g):
    return A/2*( 1/(1+((x-x0-g)*2/gamma)**2) + 1/(1+((x-x0+g)*2/gamma)**2) )

def func_double_decay(x, x0, gamma1, gamma2, A1, A2):
    return A1/(1+((x-x0)*2/gamma1)**2) + A2/(1+((x-x0)*2/gamma2)**2)


#########################################
# Absorption Spectrum
#########################################

def time_gen(sys, sp=3, N=11, nmax=5000):
    ''' Construction of the drive and time lists: '''
    ''' Crucially one must resolve the drive frequency according to Shannon criteria, '''
    ''' which sets 'tstep'. Ideally the slow dynamics should also be captured, which '''
    ''' sets 'tmax', but the number of points is limited to 'nmax' '''

    wq = sys.args['w_qubit']
    ga = sys.args['gamma']
    ed = sys.args['e_drive']
    g  = sys.args['g']
    gp = sys.Purcell()
#    ge = g*np.exp(sys.r_param())

    if ga < ed:
        print('Time list generator: expect Power Broadening')

    wspan = max(ga, ed, g)*sp
    wslow = ga

    wdriv = np.linspace(wq-wspan, wq+wspan, N)
    tstep = 2*np.pi/(2*wdriv[-1])
    tmax  = 2*np.pi/wslow
    nopt  = 2*int(tmax/tstep)
    n     = min(nopt, nmax)
    times = np.linspace(0, n*tstep, n+1)

    sys.driv = {}
    sys.driv['w_drive'] = wdriv
    sys.driv['tstep']   = tstep
    sys.driv['tmax']    = tmax
    sys.driv['nopt']    = nopt
    sys.driv['n']       = n
    sys.driv['times']   = times

    return times

def absorption_spectrum(sys, N=11, sp=3, nmax=5000, printing=False):
    ''' Generate the absorption spectrum of the Qubit for N drives '''
    ''' centered around w_qubit with span +/- sp*gamma '''

    H    = sys.operators['H']
    psi0 = vac
    D    = sys.operators['D']
    obs  = [sm*sm.dag()]

    times = time_gen(sys, N=N, sp=sp, nmax=nmax)
    freqs = nf.rfftfreq(len(times), times[1]-times[0])

    alert = 0
    if sys.driv['n'] is nmax:
        alert = 1
    print(sys.driv['n'], 'bins:', end='')
    start = time.time()

    s_abs = np.zeros(N)
    for ii, wd in enumerate(sys.driv['w_drive']):
        startloop = time.time()
        sys.args.update({'w_drive':wd})
        result    = qt.mesolve(H, psi0, times, D, obs, args=sys.args, options=options)
        smsp_fft  = nf.rfft(result.expect[0], norm='ortho')
        ii_wd     = np.argmin(np.abs(freqs-wd/2/np.pi))
        s_abs[ii] = np.abs(smsp_fft[ii_wd])

        if printing and ii in [int(N/4), int(N/2)]:
            fig, ax = plt.subplots(1,2, figsize=(12,4))
            ax[0].plot(times, result.expect[0])
            ax[0].set_ylabel(r'$<\sigma_-\sigma_+>$')
            ax[0].grid(linestyle='--')
            ax[1].plot(freqs[:], np.abs(smsp_fft[:]))
            ax[1].plot(f=abel(r'|TF $[<\sigma_-\sigma_+>]$|'))
            ax[1].grid(linestyle='--')
            ax[1].set_ylim([np.abs(smsp_fft[ii_wd])*(-0.1),np.abs(smsp_fft[ii_wd])*20])
#            title = r'$\omega_d=$%.2f, $\varepsilon_d=$%.2f'%(wd/2/np.pi, edrive)
            fig.suptitle(sys.print_args('plot'))

    if alert:
        print(' ', np.round(time.time()-start,1),'s - nmax reached ', str(sys.driv['nopt']))
    else:
        print(' ', np.round(time.time()-start,1),'s')

    return sys.driv['w_drive'], s_abs

def qubit_spectrum(sys, wlist, solver='es'):

    H   = sys.operators['H'][0]
    D   = sys.operators['D']
    A   = sm
    B   = sm.dag()

    return qt.spectrum(H, wlist, D, A, B, solver=solver)

def steadystate(sys, method='direct', solver=None):

    H   = sys.operators['H'][0]
    D   = sys.operators['D']

    return qt.steadystate(H, D, method='direct', solver=None)


#########################################
# Eigenspectrum
#########################################

def eigenspectrum(sys, Htype, sweep_key, sweep_key_list):

    nrj_list, state_list = [], []
    for ii, kk in enumerate(sweep_key_list):
        args = sys.twopi_update({sweep_key:kk})
#        print(sys.print_args('inline'))
        sys.Dynamics(Htype, isD=False)
        Ham = sys.operators['H'][0]
        nrj, state = Ham.eigenstates()
        nrj_list.append(nrj)
        state_list.append(state)
    nrj_list   = np.array(nrj_list).T/2/np.pi
    state_list = np.array(state_list).T

    return nrj_list, state_list


def feeder(S, d_cp, w_cav, printing=True):

    r_para = (1/2)*np.log(10**(S/10))
    Omega   = d_cp/np.cosh(2*r_para)
    if printing:
        print('feeder: r=%.2f, Omega=%.2f'%(r_para,Omega))
    e_pump  = d_cp*np.tanh(2*r_para)
    w_pump  = (w_cav-d_cp)*2

    return e_pump, w_pump


