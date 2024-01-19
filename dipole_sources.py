import numpy as np
import numba
from math import sqrt as msqrt
from joblib import Parallel, delayed

def calc_all(
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        location: np.ndarray,
        source_vector: np.ndarray,
        lift_off: float = 0.0) -> np.ndarray:

    n3 = int(len(location)/10)
    #locs = [location[:n2,:],location[n2:,:]]
    #scs = [source_vector[:n2,:],source_vector[n2:,:]]
    results = Parallel(n_jobs=10)([delayed(calc_point_source_field)(x_grid,y_grid,location[:n3,:],source_vector[:n3,:],lift_off),
                                   delayed(calc_point_source_field)(x_grid,y_grid,location[n3:2*n3,:],source_vector[n3:2*n3,:],lift_off),
                                   delayed(calc_point_source_field)(x_grid,y_grid,location[2*n3:3*n3,:],source_vector[2*n3:3*n3,:],lift_off),
                                   delayed(calc_point_source_field)(x_grid,y_grid,location[3*n3:4*n3,:],source_vector[3*n3:4*n3,:],lift_off),
                                   delayed(calc_point_source_field)(x_grid,y_grid,location[4*n3:5*n3,:],source_vector[4*n3:5*n3,:],lift_off),
                                   delayed(calc_point_source_field)(x_grid,y_grid,location[5*n3:6*n3,:],source_vector[5*n3:6*n3,:],lift_off),
                                   delayed(calc_point_source_field)(x_grid,y_grid,location[6*n3:7*n3,:],source_vector[6*n3:7*n3,:],lift_off),
                                   delayed(calc_point_source_field)(x_grid,y_grid,location[7*n3:8*n3,:],source_vector[7*n3:8*n3,:],lift_off),
                                   delayed(calc_point_source_field)(x_grid,y_grid,location[8*n3:9*n3,:],source_vector[8*n3:9*n3,:],lift_off),
                                   delayed(calc_point_source_field)(x_grid,y_grid,location[9*n3:,:],source_vector[9*n3:,:],lift_off)])

    bx_tot = results[0][0] + results[1][0] + results[2][0] + results[3][0] + results[4][0] + results[5][0] + results[6][0] + results[7][0] + results[8][0] + results[9][0]
    by_tot = results[0][1] + results[1][1] + results[2][1] + results[3][1] + results[4][1] + results[5][1] + results[6][1] + results[7][1] + results[8][1] + results[9][1]
    bz_tot = results[0][2] + results[1][2] + results[2][2] + results[3][2] + results[4][2] + results[5][2] + results[6][2] + results[7][2] + results[8][2] + results[9][2]
    
    return bx_tot,by_tot,bz_tot

@numba.njit(fastmath=True)
def calc_point_source_field(
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        location: np.ndarray,
        source_vector: np.ndarray,
        lift_off: float = 0.0) -> np.ndarray:
    """
    Compute the field of a magnetic dipole point source

    Parameters
    ----------
    x_grid:  ndarray(pixel, pixel)
        grid to calculate the fields for
    y_grid: ndarray(pixel, pixel)
        grid to calculate the fields on
    location: ndarray (n_sources, 3)
        x,y,z-location of source
        z distance, not including the sensor height
    source_vector: ndarray(n_sources,3)
        xyz-components of vector not n sources
    lift_off: float
        distance between sensor and sample

    Note: rewritten in tensorflow, nut is not faster, 
    vectorizing it completeley, means you run out of RAM

    Examples
    --------
    >>> x, y = maps.calc_observation_grid(pixel_size=1e-6, pixel=50)
    >>> loc, vec, total = maps.get_point_sources(5000, 100)
    >>> i = randint(loc.shape[0])
    >>> calc_point_source_field(x, y, loc[i], vec[i], 5e-6)
    """
    pixel = 1#x_grid.shape[0]
    n_sources = location.shape[0]
    
    source_vector = source_vector.copy().reshape(n_sources, 1, 1, 3)
    location = location.copy().reshape(n_sources, 1, 1, 3)
    
    mx = source_vector[:, :, :, 0]
    my = source_vector[:, :, :, 1]
    mz = source_vector[:, :, :, 2]
    lx = location[:, :, :, 0]
    ly = location[:, :, :, 1]
    lz = location[:, :, :, 2] + lift_off
    
    x_grid = x_grid.reshape((1, x_grid.shape[0], -1))
    y_grid = y_grid.reshape((1, y_grid.shape[0], -1))
    
    dgridx = np.subtract(x_grid, lx)
    dgridy = np.subtract(y_grid, ly)
    
    lx = None
    ly = None
    
    squared_distance = dgridx*dgridx + dgridy*dgridy + lz*lz
    
    gridsum = mx * dgridx + my * dgridy + mz * lz 
    
    #aux = calc_loop(squared_distance,gridsum,dgridx,dgridy,lz,mx,my,mz)
    #sqrt_dist = np.sqrt(squared_distance*squared_distance*squared_distance*squared_distance*squared_distance)
    aux = np.empty(squared_distance.shape)
    
    for i in range(squared_distance.shape[0]):
        for j in range(squared_distance.shape[1]):
            for k in range(squared_distance.shape[2]):
                aux[i,j,k] = gridsum[i,j,k]/msqrt(squared_distance[i,j,k]**5)
    
    
    tmp = 1/ np.sqrt(squared_distance*squared_distance*squared_distance)
    
    squared_distance = None
    
    bx_dip = 3.0 * aux * dgridx - mx * tmp
    by_dip = 3.0 * aux * dgridy - my * tmp
    bz_dip = 3.0 * aux * lz - mz * tmp
    

    bx_tot = np.sum(bx_dip,axis=0)*9.9472e-8
    by_tot = np.sum(by_dip,axis=0)*9.9472e-8
    bz_tot = np.sum(bz_dip,axis=0)*9.9472e-8
    
    return bx_tot,by_tot,bz_tot


@numba.njit(fastmath=True,parallel=True)
def calc_loop(
        squared_distance: np.ndarray,
        gridsum: np.ndarray,
        dgridx: np.ndarray,
        dgridy: np.ndarray,
        lz: np.ndarray,
        mx: np.ndarray,
        my: np.ndarray,
        mz: np.ndarray) -> np.ndarray:
    
    aux = np.empty(squared_distance.shape)
    
    for i in numba.prange(squared_distance.shape[0]):
        for j in range(squared_distance.shape[1]):
            for k in range(squared_distance.shape[2]):
                aux[i,j,k] = gridsum[i,j,k]/msqrt(squared_distance[i,j,k]**5)
            
    #aux = gridsum / sqrt_dist
    
    return aux