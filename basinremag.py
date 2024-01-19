'''
Basin magnetization scripts

Sarah Steele
2023
'''

## imports
# external packages
import numpy as np
from numpy.lib import stride_tricks
import pandas as pd
import scipy
from scipy.interpolate import interp1d, interp2d
from scipy.interpolate import RectBivariateSpline
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import BoundaryNorm
import meshio
import time
import os

# other .py files
from mag_materials import *
from reversal_hists import *
from late_remag_wnumba import *
from dipole_sources import calc_all


## BasinMag class
class BasinMag:
    def __init__(self,basefp,late_remag='none', tadj=1.05,interpt=1e4,**kwargs):
        self.basefp = basefp
        self.makefps(**kwargs)
        self.tadj = tadj
        self.interpt = interpt
        self.lr_scaling = late_remag
        
        # get lines from config file
        cfg = open(self.basefp+'config.cfg', "r")
        list_of_lines = cfg.readlines()
        
        # set initial time step (dt0) and total run time (ttot)
        try:
            # set initial time step (dt0) and total run time (ttot)
            self.dt0 = float(list_of_lines[98][19:-2])
            self.ttot = float(list_of_lines[99][19:-2])
            
            # set # of prerefine steps
            self.preref = int(list_of_lines[74][27:-18])
        except:
            self.dt0 = float(list_of_lines[100][19:-2])
            self.ttot = float(list_of_lines[101][19:-2])
            
            # set # of prerefine steps
            self.preref = int(list_of_lines[75][27:-18])
        
        # # load in base mesh, collect mesh points into a np array
        # mesh_df = pd.read_csv(basefp + 'mesh.inp', sep=" ",names=[1,2,3,4,5,6,7,8,9,10,11], skiprows=1, engine='python');
        
        # mesh_points = mesh_df[mesh_df[3]!='quad'].astype('float')
        # self.mesh_points = mesh_points.loc[:,[3,4]].to_numpy()
        
        return
    
    def makefps(self,meshparamsfile = 'mesh_params.txt', heatsolfile = 'output/heat_solutions.txt'):
        ''' make important file paths '''
        self.meshparamsfp = self.basefp + meshparamsfile
        self.heatsolfp = self.basefp + heatsolfile
        
        return
    
    def get_dt(self):
        ''' set time step and total model run time using fractional adjsutment at each step and # of solution files '''
        
        self.dtlist = np.insert(self.tadj**np.arange(0,self.nt-1)*self.dt0,0,0)
        self.tlist = np.cumsum(self.dtlist)
        
        return
    
    def get_dt_const(self):
        ''' set time step and total model run time using config file, fractional adjsutment at each step, and # of solution files '''
        # get lines from config file
        cfg = open(self.basefp+'config.cfg', "r")
        list_of_lines = cfg.readlines()
        
        # set time step (dt) and total run time (ttot)
        self.dt = float(list_of_lines[98][19:-2])
        self.ttot = float(list_of_lines[99][19:-2])
        
        return
    
    def interp_t(self):
        # smooth out oscillation artifacts from modeling
        heat_sols_temp = savgol_filter(self.heat_sols,2,1, axis=0)
        
        # fix undershoot
        heat_sols_temp[heat_sols_temp<210] = 210
        
        # interpolate onto regular time grid
        tint = interp1d(self.tlist,heat_sols_temp,axis=0)
        self.tintlist = np.arange(0,self.ttot,self.interpt)
        self.heat_sols = tint(np.arange(0,self.ttot,self.interpt))
        
        return 
    
    def read_sol(self,output_fp,dx=2000,save_heat=False,load_heat=False,curietrim=True,tcoarsen=1,xcoarsen=1):
        # store tcoarsen choice
        self.tcoarsen = tcoarsen
        
        # load initial mesh to extract dimensions, etc.
        meshi = meshio.read(output_fp+'\solution-000.vtk')
            
        # get x, y, T data
        xi=meshi.points[:,0]
        yi=meshi.points[:,1]
        Ti=meshi.point_data['solution']
        # self.xi = xi
        # self.yi = yi
        # make base T array
        new_x = np.arange(np.min(xi),np.max(xi),dx)
        new_y = np.arange(np.max(yi),np.min(yi),-dx)
        
        if load_heat:
            n_steps = len(os.listdir(output_fp))-1
            heat_sols = np.load(output_fp+'/T_reg.npy')
            heat_sols = heat_sols.reshape(heat_sols.shape[0],-1,n_steps)
            
                # print("Heat file not found. Generating anew")
                # self.read_sol(output_fp,dx=dx,save_heat=True,load_heat=False,curietrim=curietrim,tcoarsen=tcoarsen,xcoarsen=xcoarsen)
                # return self.heat_sols
            
        else:
            n_steps = len(os.listdir(output_fp))
            
            heat_sols = np.zeros((len(new_y),len(new_x),n_steps))
            
            new_x_mesh,new_y_mesh = np.meshgrid(new_x,new_y)
            
            xy_out = np.stack([new_x_mesh.flatten(),new_y_mesh.flatten()]).T
            
            i =  0
            while os.path.isfile(output_fp + f'\solution-{i:03d}.vtk'):
                # load mesh
                meshi = meshio.read(output_fp + f'\solution-{i:03d}.vtk')
                if i%5 == 0:
                    print(i)
                # get x, y, T data
                xi=meshi.points[:,0]
                yi=meshi.points[:,1]
                xy_in = np.stack([xi,yi]).T
                Ti=meshi.point_data['solution']
                
                # interpolate onto set mesh
                T_out = scipy.interpolate.griddata(xy_in, Ti, xy_out)
                
                # add to bigger array
                heat_sols[:,:,i] = T_out.reshape((len(new_y),len(new_x)))
                
                i += 1
            
        if save_heat:
            np.save(output_fp+'/T_reg.npy',heat_sols.reshape(heat_sols.shape[0], -1))
        
        # reorient heat_sols for rest of script
        heat_sols = np.moveaxis(heat_sols,2,0)
        
        
        # trim and subsample array as necessary
        if curietrim and np.any(heat_sols[0,:,-1]>1200):
            cutind = np.where(heat_sols[0,:,-1]>1200)[0][0]
            self.xvals = new_x[::xcoarsen]
            self.zvals = new_y[:cutind:xcoarsen]
            self.cellsize = self.xvals[1]-self.xvals[0]#xv_un[1]-xv_un[0]
            self.xgrid,self.zgrid = np.meshgrid(self.xvals,self.zvals)
            self.nt = n_steps
            self.heat_sols = heat_sols[::tcoarsen,:cutind:xcoarsen,::xcoarsen]
            
        else:   
            self.xvals = new_x[::xcoarsen]
            self.zvals = new_y[::xcoarsen]
            self.cellsize = self.xvals[1]-self.xvals[0]#xv_un[1]-xv_un[0]
            self.xgrid,self.zgrid = np.meshgrid(self.xvals,self.zvals)
            self.nt = n_steps
            self.heat_sols = heat_sols[::tcoarsen,::xcoarsen,::xcoarsen]
        
                           
        return self.heat_sols

    
    def read_sol_txt(self,rdepth=1,cdepth=50,ldepth=100,save_heat=False,load_heat=False, curietrim=True,tcoarsen=1,xcoarsen=1):
        '''
        read solution files and make heat solutions array.
        
        curietrim: cut off section of matrix that stays above ~hematite Curie point at all time steps
        tcoarsen: only keep every n time stdeps
        xcoarsen: only keep every n grid points
        '''
        # store tcoarsen choice
        self.tcoarsen = tcoarsen
        
        # load heat solutions file
        heat_sols_raw = np.loadtxt(self.heatsolfp)
        
        # drop prerefine rows
        heat_sols_raw = np.concatenate([heat_sols_raw[0].reshape(1,-1),heat_sols_raw[self.preref*2+1:]])
        
        print('Done 1')
        
        # load mesh parameters
        mps = np.loadtxt(self.meshparamsfp)
        
        print('Done 2')
        xvals = mps[:,0]
        zvals = mps[:,1]
        xv_un = np.unique(xvals)
        zv_un = np.unique(zvals)
        
        # reshape heat solutions
        dimx = xv_un.shape[0]
        dimz = zv_un.shape[0]
        heat_sols_rs = np.reshape(heat_sols_raw,(-1,dimz,dimx))

        # initialize final heat solutions array
        heat_sols = np.zeros(heat_sols_rs.shape)
        
        # deal with weird solutions things resulting from different material properties
        if np.max(np.abs(zv_un)) <= rdepth*1000:
            ind_dr = np.max(np.where(zv_un > -rdepth*1000)[0]) - np.min(np.where(zv_un > -rdepth*1000)[0]) 
            ind_r = np.min(np.where(zv_un > -rdepth*1000)[0])
            
            for i in range(heat_sols_rs.shape[0]):
                temp = np.concatenate((heat_sols_rs[i,ind_dr+2:],np.roll(heat_sols_rs[i],-ind_dr-2,axis=0)[ind_r-1:ind_r+1].flatten().reshape(2,-1,order='F'),heat_sols_rs[i,2:ind_dr+2]))
                
                temp = temp[::-1,:]
                
                heat_sols[i,:,:] = temp
                
        elif np.max(np.abs(zv_un)) <= cdepth*1000:
            ind_dc = np.max(np.where(zv_un > -cdepth*1000)[0]) - np.min(np.where(zv_un > -cdepth*1000)[0]) 
            ind_r = np.min(np.where(zv_un > -rdepth*1000)[0])
            ind_c = np.min(np.where(zv_un > -cdepth*1000)[0]);
            
            for i in range(heat_sols_rs.shape[0]):
                temp = np.concatenate((heat_sols_rs[i,ind_dc+2:],np.roll(heat_sols_rs[i],-ind_dc-2,axis=0)[ind_c-1:ind_c+1].flatten().reshape(2,-1,order='F'),heat_sols_rs[i,2:ind_dc+2]))
                
                
                temp = temp[::-1,:]
                
                heat_sols[i,:,:] = temp
                
            
                
        elif np.max(np.abs(zv_un)) <= ldepth*1000:
            ind_dc = np.max(np.where(zv_un > -cdepth*1000)[0]) - np.min(np.where(zv_un > -cdepth*1000)[0]) 
            ind_r = np.min(np.where(zv_un > -rdepth*1000)[0])
            ind_c = np.min(np.where(zv_un > -cdepth*1000)[0]);
            
            for i in range(heat_sols_rs.shape[0]):
                hs_temp = heat_sols_rs[i]
                
                hs_temp[:2,:]=hs_temp[:2,:].flatten().reshape(2,-1,order='F')
                hs_temp[2:4,:]=hs_temp[2:4,:].flatten().reshape(2,-1,order='F')
                hs_temp[ind_r-ind_c+2:ind_r-ind_c+4,:]=hs_temp[ind_r-ind_c+2:ind_r-ind_c+4,:].flatten().reshape(2,-1,order='F')
                
                hs_temp = hs_temp[::-1,:]
                
                hs_temp = np.concatenate((hs_temp[-2:,:],
                               hs_temp[ind_c-1:-2,:],
                               hs_temp[:ind_c-1,:]))
                
                hs_temp[0,0],hs_temp[1,1] = hs_temp[1,1],hs_temp[0,0]
                
                heat_sols[i,:,:] = hs_temp
                
        else:
            ind_dl = np.max(np.where(zv_un > -ldepth*1000)[0]) - np.min(np.where(zv_un > -ldepth*1000)[0]) 
            ind_r = np.min(np.where(zv_un > -rdepth*1000)[0])
            ind_c = np.min(np.where(zv_un > -cdepth*1000)[0])
            ind_l = np.min(np.where(zv_un > -ldepth*1000)[0])
            
            for i in range(heat_sols_rs.shape[0]):
                # temp = np.concatenate((heat_sols_rs[i,ind_dl+2:],np.roll(heat_sols_rs[i],-ind_dl-2,axis=0)[ind_l-1:ind_l+1].flatten().reshape(2,-1,order='F'),heat_sols_rs[i,2:ind_dl+2]))
                
                hs_temp = heat_sols_rs[i]
                
                hs_temp[:2,:]=hs_temp[:2,:].flatten().reshape(2,-1,order='F')
                hs_temp[2:4,:]=hs_temp[2:4,:].flatten().reshape(2,-1,order='F')
                hs_temp[ind_r-ind_c+2:ind_r-ind_c+4,:]=hs_temp[ind_r-ind_c+2:ind_r-ind_c+4,:].flatten().reshape(2,-1,order='F')
                hs_temp[ind_r-ind_l+2:ind_r-ind_l+4,:]=hs_temp[ind_r-ind_l+2:ind_r-ind_l+4,:].flatten().reshape(2,-1,order='F')
                
                hs_temp = hs_temp[::-1,:]
                
                hs_temp = np.concatenate((hs_temp[-2:,:],
                               hs_temp[ind_c-1:-2,:],
                               hs_temp[ind_l-1:ind_c-1,:],
                               hs_temp[:ind_l-1,:]))
                
                hs_temp[0,0],hs_temp[1,1] = hs_temp[1,1],hs_temp[0,0]
                #temp = np.concatenate((heat_sols_rs[i,ind_dl+2:],np.roll(heat_sols_rs[i],0,axis=0)[ind_l-1:ind_l+1].flatten().reshape(2,-1,order='F'),heat_sols_rs[i,2:ind_dl+2]))
                
                heat_sols[i,:,:] = hs_temp
          
        
        

        # get properly ordered z values for final arrays
        zuinds = np.unique(zvals, return_index=True)[1]
        zv_un = np.array([zvals[ind] for ind in sorted(zuinds)])
        
        # trim and subsample array as necessary
        if curietrim and np.any(heat_sols[0,:,-1]>1200):
            cutind = np.where(heat_sols[0,:,-1]>1200)[0][0]
            self.xvals = xv_un[::xcoarsen]
            self.zvals = zv_un[:cutind:xcoarsen]
            self.cellsize = self.xvals[1]-self.xvals[0]#xv_un[1]-xv_un[0]
            self.xgrid,self.zgrid = np.meshgrid(self.xvals,self.zvals)
            self.nt = heat_sols.shape[0]
            self.heat_sols = heat_sols[::tcoarsen,:cutind:xcoarsen,::xcoarsen]
            
        
        else:   
            self.xvals = xv_un[::xcoarsen]
            self.zvals = zv_un[::-xcoarsen]
            self.cellsize = self.xvals[1]-self.xvals[0]
            self.xgrid,self.zgrid = np.meshgrid(self.xvals,self.zvals)
            self.nt = heat_sols.shape[0]
            print('!!!!')
            self.heat_sols = heat_sols[::tcoarsen,::xcoarsen,::xcoarsen]
        
        
        return self.heat_sols
    
    def make_map_arrays(self):
        ''' makes arrays for basin in full 3D '''
        # make x, y, r arrays
        #self.xmap = np.concatenate((self.xgrid[:,:0:-1],self.xgrid),axis=1)
        #self.zmap = np.concatenate((self.zgrid,self.zgrid[:,1:]),axis=1)
        #xy = np.meshgrid(self.xmap[0,:],self.xmap[0,:])
        xy = np.meshgrid(self.xvals,self.xvals)
        rmap = np.sqrt(xy[0]**2+xy[1]**2)
        #rmat = np.sqrt(self.xvals)
        
        # map elements of r array to closest value in x array
        rmap_inds = np.argmin(np.abs(rmap[:,:,None]-self.xvals[None,None,:]),axis=2);
        
        self.xy = xy
        self.rmap = rmap
        self.rmap_inds = rmap_inds
        return rmap   

    def make_dM(self, Mfun, heat_in=False, tmin=0, tmax=1e10):
        ''' returns remagnetized fraction at each time step and set important magnetic properties'''
        # tmin = 1e2
        # tmax = 1e7 + 2e6
        # tmin = 0
        # tmax = 0+1e5
        
        if isinstance(heat_in,bool):    
            heat_in = self.heat_sols
            
        mag_props = Mfun(heat_in)
        Mseries = mag_props[0]
        # Mseries[:np.searchsorted(self.tintlist, tmin),:,:] = 0
        # Mseries[np.searchsorted(self.tintlist, tmax):,:,:] = 0
        
        # set object magnetic properties
        self.Mfun = Mfun
        self.dM_mat = Mseries               # fractional remagnetization 
        self.susc = mag_props[1]            # susceptibility 
        self.maxdM = np.max(Mseries,axis=0) # maximum fractional remagnetization (for implementing background fields)
        self.ieq = -1#np.searchsorted(self.tintlist, tmax)-1
        
        return Mseries
    
    def Mnet(self, revR):
        ''' calculate magnetization fraction '''
        if len(revR) == 0:
            Bstack = np.max(self.dM_mat,axis=0)-self.dM_mat[self.ieq,:,:]
        else:
            Bprof = Bprofile(revR,self.dM_mat)[0:self.ieq,:,:]
            endArr = self.dM_mat[self.ieq,:,:]
            np.append(Bprof,np.reshape(endArr,(1,endArr.shape[0],endArr.shape[1])),axis=0)
    
            # loop to drop out chrons that are later overprinted
            keepChrons = np.zeros(Bprof.shape)
            isone = np.ones((Bprof.shape[1],Bprof.shape[2]))
            
            for i in reversed(range(Bprof.shape[0])):
                keepChrons[i,:,:] = np.max(Bprof[i:,:,:],axis=0)#Bprof[i,:,:]
            '''
            for i,r in enumerate(keepChrons[:-1,:,:]):
                stays2 = (r > np.max(keepChrons[i+1:,:,:],axis=0))
                
                keepChrons[i,:,:] = keepChrons[i,:,:]*stays2
                
                #isonei = Bprof[i,:,:] != 1
                #isone = isone*isonei
            '''    
            keepChrons[keepChrons>1] = 1
            
            # sum positive and negative contributions
            pos = np.sum(keepChrons[0::2,:,:], axis = 0) - np.sum(keepChrons[1::2,:,:], axis = 0)
            neg = np.sum(keepChrons[1::2,:,:], axis = 0) - np.sum(keepChrons[2::2,:,:], axis = 0)
            '''
            pos[pos < 0] = pos[pos < 0] + np.max(keepChrons[1::2,:,:][pos < 0],axis=0)
            elif neg < 0:
                neg = neg + np.max(keepChrons[0::2,:,:],axis=0)
            
            #pos[pos>1] = 1
            #neg[neg>1] = 1
            '''
            # get net magnetizations
            if len(revR)%2==0:
                Bstack = pos - neg + self.dM_mat[self.ieq,:,:]
                
                
            else:
                Bstack = pos - neg - self.dM_mat[self.ieq,:,:]
                
                
            #if pos.shape[0] == neg.shape[0]:
               
            #Bstack = pos - neg#- self.dM_mat[ieq,:,:]
            # subtract off contribution from things above Curie temp
            
        return Bstack
    '''
    def Mnet(self, revR, ieq=-1):
        
        if len(revR) == 0:
            Bstack = np.max(self.dM_mat,axis=0)*(1-self.dM_mat[ieq,:,:])
        else:
            print(np.shape(self.dM_mat))
            Bprof = Bprofile(revR,self.dM_mat)[0:ieq,:,:]
    
            # loop to drop out chrons that are later overprinted
            keepChrons = np.zeros(Bprof.shape)
            isone = np.ones((Bprof.shape[1],Bprof.shape[2]))
            
            for i in reversed(range(Bprof.shape[0])):
                stays = Bprof[i,:,:] == np.max(Bprof[i:,:,:],axis=0)
                keepChrons[i,:,:] = Bprof[i,:,:]*stays*isone
                
                isonei = Bprof[i,:,:] != 1
                isone = isone*isonei
                
            keepChrons[keepChrons>1] = 1
            
            # sum positive and negative contributions
            pos = np.sum(keepChrons[0::2,:,:], axis = 0)
            neg = np.sum(keepChrons[1::2,:,:], axis = 0)
            
            #pos[pos>1] = 1
            #neg[neg>1] = 1
            
            # get net magnetizations
            Bstack = pos - neg
            
            # subtract off contribution from things above Curie temp
            Bstack = Bstack*(1-self.dM_mat[ieq,:,:])# - Mfrac[:,-1]*((Bprof.shape[1]%2)*2-1)
        
        return Bstack
    '''
        
    def make_old_mag(self):
        # get useful lengths
        lx = len(self.xvals)
        lz = len(self.zvals)
        
        # flatten r map array
        rinds = self.rmap.flatten()
        
        # create interpolations
        itrpMmax = interp2d(self.xvals,self.zvals,self.maxdM)

        # sort array, make recovery index
        sorted_r = np.sort(rinds)
        recovinds_r = np.argsort(np.argsort(rinds))
        
        # evaluate interpolation and put back in usual coordinates
        quadr_mag_frac = itrpMmax(sorted_r,self.zvals)[:,recovinds_r].reshape(lz,lx,lx)
        
        self.olddM_quadr = 1-quadr_mag_frac
        
        mag_moms = quadr_mag_frac[::-1,:,:]
        mag_moms = np.concatenate((mag_moms[:,:0:-1,:],mag_moms),axis=1)
        mag_moms = np.concatenate((mag_moms[:,:,:0:-1],mag_moms),axis=2)
        return mag_moms
    
    
    def make_M_map(self,dM_arr):
        '''
        Makes map of net magnetization magnitude in one quadrant
        
        dM_arr: output of self.Mnet(revR)
        '''
        # get useful lengths
        lx = len(self.xvals)
        lz = len(self.zvals)
        
        # flatten r map array
        rinds = self.rmap.flatten()
        
        # create interpolations
        itrpM = interp2d(self.xvals,self.zvals,dM_arr)

        # sort array, make recovery index
        sorted_r = np.sort(rinds)
        recovinds_r = np.argsort(np.argsort(rinds))
        
        # evaluate interpolation and put back in usual coordinates
        quadr_mag_frac = itrpM(sorted_r,self.zvals)[:,recovinds_r].reshape(lz,lx,lx)
        
        return quadr_mag_frac
    
    def baseline_B(self,mag_old,lift_off,imgrid):
        bMag_bl = self.Mnet([])
        rMag_bl = self.make_M_map(bMag_bl)
        
        B_map_bl, Bz_map_bl, s_map_bl = self.Bnet(rMag_bl,self.mag_vec,mag_old,lift_off,imgrid=imgrid)
        print(np.max(B_map_bl),np.min(B_map_bl))
        return B_map_bl, Bz_map_bl, s_map_bl
    
    def Bcalc(self, mag_moms, xm_list, ym_list, zm_list, imgrid, lift_off):
        
        # np.save(basefp_1+'mid.npy',mag_moms[:,399,:])
        # np.save(basefp_1+'top.npy',mag_moms[0,:,:])

        limx = imgrid[0].shape[0]
        limy = imgrid[0].shape[1]
        
        Bmapx = np.zeros((limx,limy))
        Bmapy = np.zeros((limx,limy))
        Bmapz = np.zeros((limx,limy))
        
        # loop over depths and calculate magnetic field
        for zi in range(len(self.zvals)):
            
            if zi%10==0:
                print(zi)
                
            
            
            # make magnetic moment array
            mag_new = mag_moms[zi,:,:].reshape((-1,1))
            #mag_old = (mag_moms[zi,:,:].reshape((-1,1))-1)*np.array(og_vec).reshape((1,-1))*self.cellsize**3*self.susc
            mag_xyz =  mag_new #+ mag_old
            
            zero_inds = np.sum(mag_xyz**2,axis=1)**2>0.0001**2
            
            mag_xyz = mag_xyz[zero_inds,:]*np.array(self.mag_vec).reshape((1,-1))*self.cellsize**3*self.susc
            
            xl_z = xm_list[zi,:,:].reshape((-1,1))[zero_inds,:]
            yl_z = ym_list[zi,:,:].reshape((-1,1))[zero_inds,:]
            zl_z = np.abs(zm_list[zi,:,:].reshape((-1,1)))[zero_inds,:]
            
        
            loc_xyz = np.concatenate((xl_z,yl_z,zl_z),axis=1)
            
            t0 = time.time()
           
            # calculate magnetic fields 
            Bmapi = calc_all(imgrid[0],imgrid[1],loc_xyz[:,:],mag_xyz[:,:],lift_off*1000)
            #print('Mag calculation time: %1.3f'%(time.time() - t0))
            #calc_point_source_field.parallel_diagnostics(level=4)
            Bmapx += Bmapi[0]
            Bmapy += Bmapi[1]
            Bmapz += Bmapi[2]
             
            
            
        # put into nT
        Bmapx = Bmapx*1e9
        Bmapy = Bmapy*1e9
        Bmapz = Bmapz*1e9
        
        print([np.max(np.abs(Bmapx)),np.max(np.abs(Bmapy)),np.max(np.abs(Bmapz))])
        
        # calculate field magnitude
        Bmap = np.sqrt(Bmapx**2+Bmapy**2+Bmapz**2)
        
        print(np.max(np.abs(Bmap)))
        
        return  Bmap, Bmapz
    
    def Bnet_dual(self,quadr_mag_frac,mag_old,lift_off,imgrid=np.meshgrid(np.arange(-100e3,100e3,10e3),np.arange(-100e3,100e3,10e3)),manualcalc=True):
        '''
        Makes map of magnetic field over specific magnetization ensemble at a given measurement height
        
        quadr_mag_frac: magnetization intensity for one quadrant
        mag_vec: magnetization direction vector
        lift_off: measurement height (km)
        '''
        
        # get useful lengths
        lx = len(self.xvals)
        lz = len(self.zvals)
        
        
        # make magnetic moments matrix
        mag_moms = quadr_mag_frac[::-1,:,:]
        mag_moms = np.concatenate((mag_moms[:,:0:-1,:],mag_moms),axis=1)
        mag_moms = np.concatenate((mag_moms[:,:,:0:-1],mag_moms),axis=2)
        
        # make position arrays for dipole_sources function
        xv_whole = np.concatenate((-1*self.xvals[:0:-1],self.xvals))
        xyz_grid = np.meshgrid(xv_whole,self.zvals,xv_whole)
        
        zm_list = xyz_grid[1]
        xm_list = xyz_grid[0]
        ym_list = xyz_grid[2]
        
        
        # add background magnetization
        print('Adding random background magnetization')
        
        # add impact layer
        print('Adding late impacts')
        
        mag_moms[self.lr_out>0] = 0
        
        print('top 2km: ', self.lr_out[0,:,:].sum()/(801*801)*100)
        print('top 10km: ', self.lr_out[0:5,:,:].sum()/(801*801*5)*100)
        
        # Bmap_0, Bmapz_0 = self.Bcalc(mag_moms, xm_list,  ym_list, zm_list, imgrid, 0.01)
        
        
        # calculate field including late impacts
        Bmap, Bmapz = self.Bcalc(mag_moms, xm_list, ym_list, zm_list, imgrid, lift_off)
        
        Bmap_0 = Bmap*0
        Bmapz_0 = Bmapz*0
        
        return  Bmap_0, Bmapz_0, Bmap, Bmapz, mag_moms 
    
    def Bnet(self,quadr_mag_frac,mag_old,lift_off,imgrid=np.meshgrid(np.arange(-100e3,100e3,10e3),np.arange(-100e3,100e3,10e3)),manualcalc=True):
        '''
        Makes map of magnetic field over specific magnetization ensemble at a given measurement height
        
        quadr_mag_frac: magnetization intensity for one quadrant
        mag_vec: magnetization direction vector
        lift_off: measurement height (km)
        '''
        
        # get useful lengths
        lx = len(self.xvals)
        lz = len(self.zvals)
        limx = imgrid[0].shape[0]
        limy = imgrid[0].shape[1]
        
        # make magnetic moments matrix
        mag_moms = quadr_mag_frac[::-1,:,:]
        mag_moms = np.concatenate((mag_moms[:,:0:-1,:],mag_moms),axis=1)
        mag_moms = np.concatenate((mag_moms[:,:,:0:-1],mag_moms),axis=2)
        
        # make position arrays for dipole_sources function
        xv_whole = np.concatenate((-1*self.xvals[:0:-1],self.xvals))
        xyz_grid = np.meshgrid(xv_whole,self.zvals,xv_whole)
        
        zm_list = xyz_grid[1]
        xm_list = xyz_grid[0]
        ym_list = xyz_grid[2]
        
        Bmapx = np.zeros((limx,limy))
        Bmapy = np.zeros((limx,limy))
        Bmapz = np.zeros((limx,limy))
        
        # add background magnetization
        print('Adding random background magnetization')
        
        # add impact layer
        print('Adding late impacts')
        
        
        mag_moms[self.lr_out>0] = 0
        
        # calculate field including late impacts
        Bmap, Bmapz = self.Bcalc(mag_moms, xm_list, ym_list, zm_list, imgrid, lift_off)
        
        
        return  Bmap, Bmapz, mag_moms 
    
    def late_remag(self):
        if self.lr_scaling == 'none':
            self.lr_out = np.zeros((len(self.zvals),len(self.xvals)*2-1,len(self.xvals)*2-1))
            
            return 
        
        elif self.lr_scaling == 'excavation':  
            print('Adding late impacts...')
            xv_whole = np.concatenate((-1*self.xvals[:0:-1],self.xvals))
            xyz = np.meshgrid(xv_whole,xv_whole,self.zvals)
            
            lr_D = ImDomain(xyz[0],xyz[1],xyz[2])
            lr_D.run_all(4.1,MBA_late,MPF_MBA,scaling=self.lr_scaling)

            self.lr_out = lr_D.mag.transpose(2, 0, 1)
            self.lr_out = self.lr_out[:,::-1,::-1]
            
            return
    
    def do_setup(self,Mfunc,output_fp,**kwargs):
        ''' do essential setup for basin '''
        # read heat solutions output file
        try:
            self.read_sol_txt(**kwargs)
        except:
            self.read_sol(output_fp,**kwargs)
        
        # make time list
        self.get_dt()
        
        # interpolate to new time grid
        self.interp_t()
        
        print(self.heat_sols.shape)
        
        # make arrays for magnetic mapping
        self.make_map_arrays()
        
        # calculate remagnetized fraction for each pixel at each time step
        self.make_dM(Mfunc)
        
        # calculate fraction of magnetization that is old
        self.make_old_mag()
        
        # write params file
        self.write_params()
        
        return
    
    def write_params(self):
        ''' write mapping parameters file '''
        # set filepath
        params_fp = self.basefp + 'params.txt'
        
        # write file
        with open(params_fp, 'w') as f:
            f.write('dim_x: %d \n' % len(self.xvals))
            f.write('dim_z: %d \n' % len(self.zvals))
            f.write('dt0: %1.6e \n' % self.dt0) 
            f.write('t_tot: %1.6e \n' % self.ttot) 
            
        return
    
    def do_mult_revs_def(self,Revs,mag_vec,mag_old,lift_off,imgrid,**kwargs):
        '''
        calculate basin magnetic field maps and susceptibility profiles for a given set 
        of reversal histories.
        
        Revs: number of reversal histories
        mag_vec: magnetization direction vector
        liftoff: height above surface to perform mapping
        imgrid: grid of points over which to calculate magnetic fields
        '''
        # calculate useful dimensions for source and mapping arrays
        lx_im = imgrid[0].shape[0]
        ly_im = imgrid[0].shape[1]
        lx = len(self.xvals)
        lz = len(self.zvals)
        
        #  initialize output arrays
        revs = [];
        B_maps = np.zeros((len(Revs),lx_im,ly_im))
        Bz_maps = np.zeros((len(Revs),lx_im,ly_im))
        s_maps = np.zeros((len(Revs),lz,2*lx-1,2*lx-1))
        
        self.mag_vec = mag_vec
        
        print('nrevs: ', len(Revs))
        for i,revsi in enumerate(Revs):
            # compute late impact remag
            self.late_remag()
            
            print('Reversal history %d' %i)
            print(revsi)
            
            # make map 
            bMagi = self.Mnet(revsi)
            rMagi = self.make_M_map(bMagi)
            
            B_mapi, Bz_mapi, s_mapi = self.Bnet(rMagi,mag_old,lift_off,imgrid=imgrid)
            
            # save stuff
            revs.append(revsi)
            B_maps[i,:,:] = B_mapi
            Bz_maps[i,:,:] = Bz_mapi
            s_maps[i,:,:,:] = s_mapi
            
        return revs, B_maps, Bz_maps, s_maps
    

    def do_mult_revs_dual(self,Revs,mag_vec,mag_old,lift_off,imgrid,**kwargs):
        '''
        calculate basin magnetic field maps and susceptibility profiles for a given set 
        of reversal histories. calculates field maps with AND without late impacts
        
        Revs: number of reversal histories
        mag_vec: magnetization direction vector
        liftoff: height above surface to perform mapping
        imgrid: grid of points over which to calculate magnetic fields
        '''
        # calculate useful dimensions for source and mapping arrays
        lx_im = imgrid[0].shape[0]
        ly_im = imgrid[0].shape[1]
        lx = len(self.xvals)
        lz = len(self.zvals)
        
        #  initialize output arrays
        revs = [];
        B_maps_0 = np.zeros((len(Revs),lx_im,ly_im))
        Bz_maps_0 = np.zeros((len(Revs),lx_im,ly_im))
        B_maps = np.zeros((len(Revs),lx_im,ly_im))
        Bz_maps = np.zeros((len(Revs),lx_im,ly_im))
        s_maps = np.zeros((len(Revs),lz,2*lx-1,2*lx-1))
        
        self.mag_vec = mag_vec
        
        print('nrevs: ', len(Revs))
        for i,revsi in enumerate(Revs):
            # compute late impact remag
            self.late_remag()
        
            if len(revsi)==0:
                # make baseline field files
                print('Reversal history %d' %i)
                print(revsi)
                
                # make map 
                bMagi = self.Mnet(revsi)
                rMagi = self.make_M_map(bMagi)
                
                mag_moms = rMagi[::-1,:,:]
                
                self.mag_moms = mag_moms
                B_mapi_0, Bz_mapi_0, B_mapi, Bz_mapi, s_mapi = self.Bnet_dual(rMagi,mag_old,lift_off,imgrid=imgrid)
                
                # save stuff
                revs.append(revsi)
                B_maps_0[i,:,:] = B_mapi_0
                Bz_maps_0[i,:,:] = Bz_mapi_0
                B_maps[i,:,:] = B_mapi
                Bz_maps[i,:,:] = Bz_mapi
                s_maps[i,:,:,:] = s_mapi
            else:
                print('Reversal history %d' %i)
                print(revsi)
                
                # make map 
                bMagi = self.Mnet(revsi)
                rMagi = self.make_M_map(bMagi)
                
                mag_moms = rMagi[::-1,:,:]
                
                self.mag_moms = mag_moms
                B_mapi_0, Bz_mapi_0, B_mapi, Bz_mapi, s_mapi = self.Bnet_dual(rMagi,mag_old,lift_off,imgrid=imgrid)
                
                # save stuff
                revs.append(revsi)
                B_maps_0[i,:,:] = B_mapi_0
                Bz_maps_0[i,:,:] = Bz_mapi_0
                B_maps[i,:,:] = B_mapi
                Bz_maps[i,:,:] = Bz_mapi
                s_maps[i,:,:,:] = s_mapi
            
        return revs, B_maps_0, Bz_maps_0, B_maps, Bz_maps, s_maps

