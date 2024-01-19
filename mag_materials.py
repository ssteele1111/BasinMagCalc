'''
Sarah Steele
2023

Remagnetization equations from Lillis et al. (2013)
'''


import numpy as np
from scipy.special import gammainc
from scipy.special import gamma
from scipy.constants.codata import mu0
import matplotlib.pyplot as plt

## magnetic materials functions
# remagnetization 
def M_pyrr(T):    # from Lillis et al. (2013)
    TC = T-273
    M = 1- gammainc(0.355,(325 - TC)/152)/gammainc(0.355,(325)/152)
    M[np.isnan(M)] = 1   
    
    # susceptibility
    M_susc = 1

    return M, M_susc

def M_mag(T):    # from Lillis et al. (2013)
    TC = T-273
    M = 1- gammainc(0.211,(580 - TC)/7000)/gammainc(0.211,580/7000)
    M[M<0] = 0
    M[np.isnan(M)] = 1
    
    # susceptibility
    M_susc = 1

    return M, M_susc

def M_hem(T):    # from Lillis et al. (2013)
    TC = T-273
    M = 1- gammainc(1.06,(700 - TC)/20.6)/gammainc(1.06,700/20.6)
    M[np.isnan(M)] = 1
    
    # susceptibility
    M_susc = 1
    
    return M, M_susc

def M_composite(T,fP=1./3.,fM=1./3.,fH=1./3.,mag_dens=0.0005):
    
    MP = M_pyrr(T)
    MM = M_mag(T)
    MH = M_hem(T)
    
    remag_eqn = fP*MP[0] + fM*MM[0] + fH*MH[0] 
    
    remag_eqn[remag_eqn>1]=1
    susc = mag_dens*(fP*MP[1] + fM*MM[1] + fH*MH[1])
    
    return remag_eqn, susc

def M_Tissint(T):    # from fig 7, TRM of Gattacceca+ 2013
    
    TC = T-273
    
    poly7 = np.poly1d([ 1.88087428e-18, -3.81780020e-15,  2.79413427e-12, -8.78185007e-10,
        1.02997869e-07, -2.47798745e-06, -6.04214437e-04,  1.00511613e+00])
    
    M = (1-poly7(TC))
    M[TC<0] = 0.
    M[TC>600] = 1
    
    # susceptibility
    M_susc = 1
    
    return M, M_susc

