'''
Sarah Steele
2023
'''
import numpy as np
np.random.seed(seed=100)

# make reversal histories
def B_revs(mu,thresh,Nt=20000):
    # initialize polarity array
    pol = np.ones(Nt)
    
    # randomly draw from a Poisson distribution
    steps = np.random.poisson(mu, Nt)
    
    # find where field reverses
    reversals = np.where(steps > thresh)[0]
    
    for i,r in enumerate(reversals[0:-1:2]):
        pol[r:reversals[2*i+1]] = -1
      
    if len(reversals)<0:
        return B_revs(mu,thresh,Nt)
    else:
        return reversals+1
    
    # make reversal histories
def B_revs_pol(mu,thresh,Nt=500):
    # initialize polarity array
    pol = np.ones(Nt)
    
    # randomly draw from a Poisson distribution
    steps = np.random.poisson(mu, Nt)
    
    # find where field reverses
    reversals = np.where(steps > thresh)[0]
    
    for i,r in enumerate(reversals[0:-1:2]):
        pol[r:reversals[2*i+1]] = -1
        
    if len(reversals)==0:
        return B_revs(mu,thresh,Nt)
    else:
        return reversals+1, pol


# calculate susceptible fraction at each time step
def Bprofile(revR,Mfrac):
    # find indices where polarity changes if they're given as 1s and 0s
    # v[:-1] != v[1:] 
    if len(revR) == 0:
        remag = np.zeros((1,Mfrac.shape[1],Mfrac.shape[2]))
        remag[0,:,:] = np.max(Mfrac,axis=0)
    else:
        if revR[0]!=0:
            revR = np.insert(revR,0,0)
        if revR[-1]!=Mfrac.shape[0]-1:
            revR = np.append(revR, Mfrac.shape[0]-1)
        
        # initialize remag array
        remag = np.zeros((len(revR)-1,Mfrac.shape[1],Mfrac.shape[2]))
        
        # find max percent reset in each region
        for i in range(len(revR)-1):
            remag[i,:,:] = np.max(Mfrac[revR[i]:revR[i+1],:,:],axis=0) # make sure this max is along time axis
    
    return remag


# calculate susceptible fraction at each time step (1D)
def Bprofile_1D(revR,Mfrac):
    # find indices where polarity changes if they're given as 1s and 0s
    # v[:-1] != v[1:] 
    if len(revR) == 0:
        remag = np.zeros((Mfrac.shape[0],1))
        remag[:,0] = np.max(Mfrac[:,:],axis=1)
    else:
        revR = np.insert(revR,0,0)
        revR = np.append(revR, Mfrac.shape[1]-1)
        
        # initialize remag array
        remag = np.zeros((Mfrac.shape[0],len(revR)-1))
        
        # find max percent reset in each region
        for i in range(len(revR)-1):
            remag[:,i] = np.max(Mfrac[:,revR[i]:revR[i+1]],axis=1) # make sure this max is along time axis
    
    return remag

# calculate magnetization fraction and net magnetization at altitude
def Bnet_1D(revR, Mfrac, dx=1, alt=200,ieq=-1):
    Bprof = Bprofile(revR,Mfrac)[:,0:ieq]
    
    # loop to drop out chrons that are later overprinted
    keepChrons = np.zeros(Bprof.shape)
    isone = np.ones(Bprof.shape[0])
    
    for i in reversed(range(Bprof.shape[1])):
        stays = Bprof[:,i] == np.max(Bprof[:,i:],axis=1)
        keepChrons[:,i] = Bprof[:,i]*stays*isone
    
        isonei = Bprof[:,i] != 1
        isone = isone*isonei

    # sum positive and negative contributions
    pos = np.sum(keepChrons[:,0::2], axis = 1)
    neg = np.sum(keepChrons[:,1::2], axis = 1)
    
    pos[pos>1] = 1
    neg[neg>1] = 1
    
    # get net magnetizations
    Bstack = pos - neg
    
    # subtract off contribution from things above Curie temp
    Bstack = Bstack*(1-Mfrac[:,ieq-1])# - Mfrac[:,-1]*((Bprof.shape[1]%2)*2-1)
    
    Bnet = np.sum(np.flip(Bstack)/(alt+np.arange(0,Bprof.shape[0],dx)/dx)**2)
    return np.abs(Bnet),Bstack

# get adjusted reversal indices
def rTimeline(revs,Basin):
    dt = Basin.evolvedt[1]-Basin.evolvedt[0]
    revsi = (revs - Basin.evolvedt[0])/dt
    
    revsi = np.round(np.array(revsi[revs < Basin.evolvedt[-1]])).astype(int)
    revsi = revsi[revsi > 0]
    
    return revsi[revsi < len(Basin.evolvedt)]


# put all craters on one timeline
def Btimeline(basins):
    # get basins ages
    
    # calculate offsets
    for basin in basins:
        dt = basin.evolvedt[1]-basin.evolvedt[0]
        
        rem = basin.remag
        dtind = int(np.round(basin.age/dt))
        basin.remagadj = np.concatenate((np.zeros((rem.shape[0],dtind)),rem[:,0:-dtind]),axis=1)
    
    return 

# generate a bunch of different reversals, get net magnetic field over each basin
def simRevs(basins,Nr=100):
    Nt0 = basins[0].remagadj.shape[1]
    Nb = len(basins)
    
    # make arrays
    Bnets = np.zeros((Nb,Nr))
    params = np.zeros((Nb,Nr))
    
    revRecs = np.zeros((40,Nr))
    nRevs = np.zeros(Nr)
    
    for j in range(Nr):
        # make reversal record
        p = np.random.randint(17,24)
        revs = B_revs(10,p,Nt=Nt0-2)[0]+1
        nRevs[j] = len(revs)
        
        revRecs[0:len(revs),j] = revs
        
        for i,basin in enumerate(basins):
            # get time-adjusted reversal record
            Brem = basin.remagadj
            
            Bnets[i,j] = Bnet_1D(revs,Brem)[0]
            params[i,j] = p
    
    
    return Bnets, nRevs, revRecs#, params
  
  