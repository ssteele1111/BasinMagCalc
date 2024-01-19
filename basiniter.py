'''
Sarah Steele
2023
'''

# imports
import matplotlib.pyplot as plt
import time
import os
from numpy.random import randint

from basinremag import *
from mag_materials import *
from reversal_hists import Bprofile


# preset image grid
imgrid2=np.meshgrid(np.arange(-40e3,41e3,10e3),np.arange(-750e3,751e3,20e3))
imgrid3=np.meshgrid(np.arange(-40e3,41e3,10e3),np.arange(-1250e3,1251e3,20e3))
imgrid5=np.meshgrid(np.arange(-750e3,751e3,25e3),np.arange(-750e3,751e3,25e3))
imgrid6=np.meshgrid(np.arange(-500e3,501e3,15e3),np.arange(-500e3,501e3,15e3))
imgrid9=np.meshgrid(np.arange(-1250e3,1251e3,50e3),np.arange(-1250e3,1251e3,50e3))

# set input/output files
fpout = os.getcwd()+'/600km/mag_output/'
fpin= os.getcwd()+'/600km/'

# make output file if it doesn't exist
if not os.path.isdir(fpout):
        os.mkdir(fpout)
    
def do_basiniter_dual(basinsizes,nRevs,mu,thresh,magdir,magdirstr,liftoff,imgrid,fileapp='',late_remag='none',bg_mag=False,imfile=False):
    '''
    Iterates over specified basin sizes and saves magnetic field maps, susceptibility map slices, and reversal histories for
    a given number of randomly generated reversal histories shared by all basin sizes.
    
    basinsizes: list of basin sizes, must match file names
    nRevs: number of reversal histories to generate
    mu, thresh: parameters for relevant binomial distribution
    magdir: magnetization direction in normalized [x,y,z] format
    magdirstr: save file string
    liftoff: height above surface to perform mapping
    imgrid: grid of points over which to calculate magnetic fields
    '''
    # make base dir if necessary
    if not os.path.isdir(fpout + 'BMaps/'):
        os.mkdir(fpout + 'BMaps/')
        os.mkdir(fpout + 'BMaps_nolr/')
        os.mkdir(fpout + 'RevRates/')
        os.mkdir(fpout + 'Revs/')
        os.mkdir(fpout + 'SuscMaps/')
            
    # make reversal folder path if necessary
    if not os.path.isdir(fpout + 'Revs/' + magdirstr+'/'):
        os.mkdir(fpout + 'Revs/'+magdirstr+'/')
        
    # make reversal folder path if necessary
    if not os.path.isdir(fpout + 'Revs/' + magdirstr + '/Full/'):
        os.mkdir(fpout + 'Revs/' + magdirstr + '/Full/')
               
    # make common reversal histories
    RevsC = []
    
    for j in range(nRevs):
        RevsC.append(B_revs(mu, thresh, Nt=20000))
        
            
        
    for i in basinsizes:
        t0 = time.time()
        
        # make basin
        bmt1 = BasinMag(fpin,late_remag=late_remag)
        
        print('Starting ' + str(i) + 'km')
        RevsB = []
        
        # make label
        outtag = str(i)+'km_' + str(mu) +'_'+str(thresh) + '_' + str(nRevs) 
        output_fp = rf'\{i:d}km\output'
        bmt1.do_setup(M_Tissint,output_fp, tcoarsen=1,xcoarsen=2,curietrim=True,load_heat=False)
        
        # get reversals on basin timescale
        nmax = bmt1.ttot/1e8*10000 - 1 # find highest location in full reversal history array that overlaps cooling history
        nadj = 1e4/(bmt1.interpt*1)        # scale reversal history to map to time steps used in cooling model + adjust for coarsening
        
        # initialize reversal rates array
        revRates = np.zeros(len(RevsC))
        
        for k,revk in enumerate(RevsC):
            maxk = RevsC[k]
            scaledk = maxk[maxk<nmax]*nadj
            
            revsk=np.unique(scaledk.astype(int))
            
            RevsB.append(revsk)
            print(len(revsk)/bmt1.ttot)
            
            # calculate reversal rate
            revRates[k] = len(revsk)/bmt1.ttot
            
        
        # make dir if necessary
        if not os.path.isdir(fpout + 'BMaps/'+magdirstr+'/'):
            os.mkdir(fpout + 'BMaps/'+magdirstr+'/')
        if not os.path.isdir(fpout + 'BMaps_nolr/'+magdirstr+'/'):
            os.mkdir(fpout + 'BMaps_nolr/'+magdirstr+'/')
        if not os.path.isdir(fpout + 'RevRates/'+magdirstr+'/'):
            os.mkdir(fpout + 'RevRates/'+magdirstr+'/')
        if not os.path.isdir(fpout + 'SuscMaps/'+magdirstr+'/'):    
            os.mkdir(fpout + 'SuscMaps/'+magdirstr+'/')
        
        # save full reversal histories
        fp_R0 = fpout + 'Revs/'+magdirstr+'/Full/Revs_' + outtag + '.txt'
        np.save(fp_R0, RevsB)
        
        if isinstance(liftoff, list):
            
            fpRR = fpout + 'RevRates/'+magdirstr+'/RR_' + outtag + '.txt'
            
            for i in range(len(liftoff)):
                
                # make filepaths
                fpB_0 = fpout + 'BMaps_nolr/'+magdirstr+'/BMap_' + outtag + '_' + str(liftoff[i]) + '.txt'
                fpBz_0 = fpout + 'BMaps_nolr/'+magdirstr+'/BzMap_' + outtag + '_' + str(liftoff[i]) + '.txt'
                fpB = fpout + 'BMaps/'+magdirstr+'/BMap_' + outtag + '_' + str(liftoff[i])+ '.txt'
                fpBz = fpout + 'BMaps/'+magdirstr+'/BzMap_' + outtag + '_' + str(liftoff[i]) + '.txt'
                fpS = fpout + 'SuscMaps/'+magdirstr+'/SuscMap_'  + outtag + '_' + str(liftoff[i])  + '.txt'
                 
                # run stuff
                revs1,B1_0,Bz1_0,B1,Bz1,s1=bmt1.do_mult_revs_dual(RevsB,magdir,0,liftoff[i],imgrid)
                
                # save stuff
                np.savetxt(fpB_0,B1_0.reshape(B1_0.shape[0],-1)) 
                np.savetxt(fpBz_0,Bz1_0.reshape(Bz1_0.shape[0],-1)) 
                np.savetxt(fpB,B1.reshape(B1.shape[0],-1)) 
                np.savetxt(fpBz,Bz1.reshape(Bz1.shape[0],-1)) 
                np.savetxt(fpS,s1[:,::2,::2,::2].reshape(s1.shape[0],-1)) 
                
                
        else:
            revs1,B1_0,Bz1_0,B1,Bz1,s1=bmt1.do_mult_revs_dual(RevsB,magdir,0,liftoff,imgrid)
            
            # make filepaths
            fpB_0 = fpout + 'BMaps_nolr/'+magdirstr+'/BMap_' + outtag + '.txt'
            fpBz_0 = fpout + 'BMaps_nolr/'+magdirstr+'/BzMap_' + outtag + '.txt'
            fpB = fpout + 'BMaps/'+magdirstr+'/BMap_' + outtag + '.txt'
            fpBz = fpout + 'BMaps/'+magdirstr+'/BzMap_' + outtag + '.txt'
            fpRR = fpout + 'RevRates/'+magdirstr+'/RR_' + outtag + '.txt'
            fpS = fpout + 'SuscMaps/'+magdirstr+'/SuscMap_' + outtag + '.txt'
            
            # save stuff
            np.savetxt(fpB_0,B1_0.reshape(B1_0.shape[0],-1)) 
            np.savetxt(fpBz_0,Bz1_0.reshape(Bz1_0.shape[0],-1)) 
            np.savetxt(fpB,B1.reshape(B1.shape[0],-1)) 
            np.savetxt(fpBz,Bz1.reshape(Bz1.shape[0],-1)) 
            np.savetxt(fpRR,revRates) 
            np.savetxt(fpS,s1[:,::2,::2,::2].reshape(s1.shape[0],-1)) 
       
        print('Done!')
        print(str(i) + 'km time: ' + str(time.time()-t0))
        
        
    return 


do_basiniter_dual([600],5,10,15,[0,1,0],'010_200km_ig6',200,imgrid3,fileapp='Cold/',late_remag='excavation')
