'''
Late impact remagnetization calculations

Sarah Steele
2023
'''
import numpy as np
import matplotlib.pyplot as plt
import time as tm
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import plotly.graph_objects as go
from scipy import ndimage
import numba
import asyncio
from joblib import Parallel, delayed
# from numba import jitclass   

## load important constants and parameters
# constants|
g = 3.721;               # gravitational acceleration at surface (m/s)

# material properties (assuming a basaltic lithology e.g. Abramov et al. (2016))
mats = {"Cp": 800.,      # heat capacity (J k^-1 C^-1) (Abramov et al. (2016))
        "K0": 19.3*1e9,  # adiabatic bulk modulus at zero pressure (GPa) (Abramov et al. (2016))
        "n": 5.5,        # derivative of the bulk modulus (Abramov et al. (2016))
        "rho": 3000.,    # uncompressed density (kg/m^3) (Abramov et al. (2016))
        "Dq":10000.,        # simple-to-complex crater transition diameter
        "Tliq":1250.,    # liquidus temperature (C) (Abramov et al. (2016))
        "Tsol":1100.,    # solidus temperature (C) (Abramov et al. (2016))
        "Tconv":1175,    # rheological transition temperature (Kolzenburg et al. (2018), Fig. 2)
        "gradT":13.,     # thermal gradient (C km^-1) (Babeyko & Zharkov (2000))
        "flux":35.       # (Plesa et al. (2015))
        }

# impact defaults
imdefs = {"vi"  :   10,     # impact velocity (km/s)
          "theta":  np.pi/4 # impact angle (radians)
        }


## impactor flux data files
data_fp = os.getcwd()+'\Marchi21_data'

## load cumulative crater density vs. time 
MBA_early = np.genfromtxt(data_fp+'/ajabe417f2/mars_N1_MBA_early.dat',
                     skip_header=2,
                     dtype=None,
                     delimiter='      ')

MBA_late = np.genfromtxt(data_fp+'/ajabe417f2/mars_N1_MBA_late.dat',
                     skip_header=2,
                     dtype=None,
                     delimiter='      ')

NEO_early = np.genfromtxt(data_fp+'/ajabe417f2/mars_N1_NEO_early.dat',
                     skip_header=2,
                     dtype=None,
                     delimiter='      ')

NEO_late= np.genfromtxt(data_fp+'/ajabe417f2/mars_N1_NEO_late.dat',
                     skip_header=2,
                     dtype=None,
                     delimiter='      ')

# load MPF
MPF_MBA = np.genfromtxt(data_fp+'/ajabe417f5/mars_MPF_MBA.dat',
                     skip_header=6,
                     dtype=None,
                     encoding=None)

MPF_NEO = np.genfromtxt(data_fp+'/ajabe417f5/mars_MPF_NEO.dat',
                     skip_header=6,
                     dtype=None,
                     encoding=None)



# make probability density distributions from MPFs
def MPF_prob(MPF_in):
    # interpolate MPF, restricting to craters > 1km diameter
    MPF_interp = interp1d(MPF_in[:,0],MPF_in[:,1])
    MPF = MPF_interp(np.logspace(0,np.log10(np.max(MPF_in[:,0]-1)),100))
    
    MPF_norm = MPF/np.sum(MPF)
    
    MPF_cum = interp1d(np.cumsum(MPF_norm),np.logspace(0,np.log10(np.max(MPF_in[:,0]-1)),100),fill_value="extrapolate")
    
    return MPF_cum

def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped


# @background
def time_evolve(outmat, xgrid,ygrid,zgrid,time, N, MPF, sizelims,scaling,fac):
    t0 = tm.time()
    for i,t in enumerate(time):
        
        if N[i] > 0:
            # draw basin sizes
            sizes = MPF(np.random.rand(N[i]))
                  
            sizes = sizes[np.logical_and(sizes>=sizelims[0],sizes<=sizelims[1])]
            Ntmp = len(sizes)

            outmat = add_impacts(sizes,xgrid,ygrid,zgrid,outmat,scaling=scaling,fac=fac)
    
    
    try:
        print(N.shape, tm.time()-t0)
    except:
        print(tm.time()-t0)
    return outmat


@numba.njit(fastmath=True)        
def add_impacts(sizes,xgrid,ygrid,zgrid,mag,scaling='excavation',fac=1):
    
    mag2 = mag.copy()
    width = np.abs(np.max(xgrid)-np.min(xgrid))
    
    for n in range(len(sizes)):
        x,y = [np.random.rand(1)*width+np.min(xgrid),np.random.rand(1)*width+np.min(ygrid)]
        # calculate transient crater diameter
        Dtc=((1/0.91)*10000.**(0.09)*sizes[n]*1000)**(1/1.125)/1.2   # Abramov & Kring (2005)
        
        gridshape = xgrid.shape
        for k in range(gridshape[2]):
            if np.sqrt(zgrid[0,0,k]**2)>(fac*Dtc/2000):
                break
            
            for i in range(gridshape[0]):
                for j in range(gridshape[1]):
                
                    if np.sqrt((xgrid[i,j,k] - x)**2 + (ygrid[i,j,k] - y)**2 + zgrid[i,j,k]**2)<(fac*Dtc/2000):
                        mag2[i,j,k] = 1
    return mag2

class ImDomain:
    def __init__(self, xgrid, ygrid, zgrid):
        self.xgrid, self.ygrid, self.zgrid = xgrid/1000, ygrid/1000, zgrid/1000
        
        self.width = np.abs(np.max(self.xgrid)-np.min(self.xgrid))
        self.mag = np.zeros(xgrid.shape)

        return
    
    def time_setup(self, age, model, MPF, sizelims):
        time = MBA_early[:,0][MBA_early[:,0] <= age]
        Ndens = model[:,1][model[:,0] <= age]

        N = np.round(self.width**2 * Ndens)
        Ndt = np.insert(N[1:]-N[:-1],0,N[0])
        
        MPF_cum = MPF_prob(MPF)
        
        return time, Ndt.astype('int'), MPF_cum
    

    def run_all(self, age, model, MPF, scaling='excavation',sizelims=[2,1000],fac=1):
        # do setup
        time, N, MPF = self.time_setup(age, model, MPF, sizelims)
        
        print('impacts time evolution: ', np.sum(N))
        t0_tot = tm.time()
        
        outmat1 = np.zeros(self.xgrid.shape)
        outmat2 = np.zeros(self.xgrid.shape)
        outmat3 = np.zeros(self.xgrid.shape)
        outmat4 = np.zeros(self.xgrid.shape)
        outmat5 = np.zeros(self.xgrid.shape)
        outmat6 = np.zeros(self.xgrid.shape)
        outmat7 = np.zeros(self.xgrid.shape)
        outmat8 = np.zeros(self.xgrid.shape)
        
        results = Parallel(n_jobs=8,prefer="threads")([delayed(time_evolve)(outmat1,self.xgrid, self.ygrid, self.zgrid, time[0:66], N[0:66], MPF, sizelims, scaling, fac),
                       delayed(time_evolve)(outmat2,self.xgrid, self.ygrid, self.zgrid, time[66:76], N[66:76], MPF, sizelims, scaling, fac),
                       delayed(time_evolve)(outmat3,self.xgrid, self.ygrid, self.zgrid, time[76:80], N[76:80], MPF, sizelims, scaling, fac),
                       delayed(time_evolve)(outmat4,self.xgrid, self.ygrid, self.zgrid, time[80:81], N[80:81], MPF, sizelims, scaling, fac),
                       delayed(time_evolve)(outmat5,self.xgrid, self.ygrid, self.zgrid, time[81:82], N[81:82], MPF, sizelims, scaling, fac),
                       delayed(time_evolve)(outmat6,self.xgrid, self.ygrid, self.zgrid, time[82:83], N[82:83], MPF, sizelims, scaling, fac),
                       delayed(time_evolve)(outmat7,self.xgrid, self.ygrid, self.zgrid, time[83:84], [int((N[83]-N[83]%3)/2)], MPF, sizelims, scaling, fac),
                       delayed(time_evolve)(outmat8,self.xgrid, self.ygrid, self.zgrid, time[83:84], [int(N[83]-2*(N[83]-N[83]%3)/2)], MPF, sizelims, scaling, fac)])
    
        
        res = results[0] + results[1] + results[2] + results[3] + results[4] + results[5] + results[6] + results[7]
        
        print('total runtime: ', tm.time()-t0_tot)
        
        self.mag = res
        
        return res
    
    def run_all_1500(self, age, model, MPF, scaling='excavation',sizelims=[2,1000],fac=1):
            # do setup
            time, N, MPF = self.time_setup(age, model, MPF, sizelims)
            
            # load one of 5 random small basin maps
            np.load(rf'C:\Users\SteeleSarah\Researches\ImpactCooling\Scripts\ImpactMaps\1500km\smalls_{np.random.randint(5):d}')
            
            print('impacts time evolution: ', np.sum(N))
            t0_tot = tm.time()
            
            outmat1 = np.zeros(self.xgrid.shape)
            outmat2 = np.zeros(self.xgrid.shape)
            outmat3 = np.zeros(self.xgrid.shape)
            outmat4 = np.zeros(self.xgrid.shape)
            outmat5 = np.zeros(self.xgrid.shape)
            outmat6 = np.zeros(self.xgrid.shape)
            outmat7 = np.zeros(self.xgrid.shape)
            outmat8 = np.zeros(self.xgrid.shape)
            
            results = Parallel(n_jobs=8,prefer="threads")([delayed(time_evolve)(outmat1,self.xgrid, self.ygrid, self.zgrid, time[0:66], N[0:66], MPF, sizelims, scaling, fac),
                        delayed(time_evolve)(outmat2,self.xgrid, self.ygrid, self.zgrid, time[66:76], N[66:76], MPF, sizelims, scaling, fac),
                        delayed(time_evolve)(outmat3,self.xgrid, self.ygrid, self.zgrid, time[76:80], N[76:80], MPF, sizelims, scaling, fac),
                        delayed(time_evolve)(outmat4,self.xgrid, self.ygrid, self.zgrid, time[80:81], N[80:81], MPF, sizelims, scaling, fac),
                        delayed(time_evolve)(outmat5,self.xgrid, self.ygrid, self.zgrid, time[81:82], N[81:82], MPF, sizelims, scaling, fac),
                        delayed(time_evolve)(outmat6,self.xgrid, self.ygrid, self.zgrid, time[82:83], N[82:83], MPF, sizelims, scaling, fac),
                        delayed(time_evolve)(outmat7,self.xgrid, self.ygrid, self.zgrid, time[83:84], [int((N[83]-N[83]%3)/2)], MPF, sizelims, scaling, fac),
                        delayed(time_evolve)(outmat8,self.xgrid, self.ygrid, self.zgrid, time[83:84], [int(N[83]-2*(N[83]-N[83]%3)/2)], MPF, sizelims, scaling, fac)])
        
            
            res = results[0] + results[1] + results[2] + results[3] + results[4] + results[5] + results[6] + results[7]
            
            print('total runtime: ', tm.time()-t0_tot)
            
            self.mag = res
            
            return res
        