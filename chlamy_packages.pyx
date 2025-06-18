import cython
import numpy as np
cimport numpy as np
import scipy.signal as sp
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
from tqdm import tqdm
ctypedef np.float64_t dtype_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef full_arrays(dtype_t[:] phi, dtype_t[:] x, dtype_t[:] y, 
                      dtype_t[:] W_phi, dtype_t[:] W_x, dtype_t[:] W_y,
                      dtype_t tau,
                      dtype_t R, dtype_t H, dtype_t eta_0, dtype_t kbT, 
                      dtype_t D_r, dtype_t v, int N,
                      dtype_t T, dtype_t Delta_P, dtype_t L, dtype_t W, bint oscillations):
    

    Rh = (1 - 0.63 * H / W) ** (-1) * (12 * eta_0 * L) / (H**3 * W)
    
    # cdef dtype_t time
    cdef dtype_t time = 0
    cdef int i
    for i in range(1, N):     
        phi[i], x[i], y[i] = update_positions(
            x[i - 1], y[i - 1], phi[i - 1], 
            W_x[i - 1], W_y[i - 1], W_phi[i - 1],
            R, H, eta_0, kbT, D_r, v, tau,
            T, Delta_P, L, W, Rh, time, oscillations
            )
        time += tau 

    return np.asarray(phi), np.asarray(x), np.asarray(y)

"""
______ H

--- z+R
particle top
particle bottom
--- z
______ 0

"""




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef (dtype_t,dtype_t,dtype_t) update_positions(
    dtype_t x_current, dtype_t y_current, dtype_t phi_current, dtype_t W_x_current, dtype_t W_y_current, dtype_t W_phi_current,
    dtype_t R, dtype_t H, dtype_t eta_0, dtype_t kbT, dtype_t D_r, dtype_t v, dtype_t tau, 
    dtype_t T, dtype_t Delta_P, dtype_t L, dtype_t W, dtype_t Rh, dtype_t time, bint oscillations):
    D_x = (kbT)/(6 * np.pi * eta_0 * R)
    D_y = (kbT)/(6 * np.pi * eta_0 * R)

    cdef dtype_t phi_next


    
    if oscillations == True :
        v_stokes = 1 / (2 * eta_0 ) * Delta_P / L * (y_current) * (W - y_current) * np.sin(2 * np.pi * time / T)
        phi_next = phi_current + np.sqrt(2 * D_r) * W_phi_current - (0.25 * Delta_P * np.sin(2 * np.pi * time / T) / (eta_0 * L) * (W - 2 * y_current)) * tau

    else :
        v_stokes =  1 / (2 * eta_0 ) * Delta_P / L * (y_current) * (W - y_current)
        phi_next = phi_current + np.sqrt(2 * D_r) * W_phi_current - (0.25 * Delta_P / (eta_0 * L) * (W - 2 * y_current)) * tau


    x_next = x_current + v * np.cos(phi_current) * tau + np.sqrt(2 * D_x) * W_x_current + v_stokes * tau
    y_next = y_current + v * np.sin(phi_current) * tau + np.sqrt(2 * D_y) * W_y_current

    if y_next > (W - R) :
        y_next = W - R
    elif y_next < R :
        y_next = R 
    
    return phi_next, x_next, y_next


def compute_full_trajectory(R, H, eta_0, kbT, 
                D_r, v, N,
                T, tau, Delta_P, L, W, oscillations) :

    cdef dtype_t[:] phi = np.zeros(N)
    phi[0] = np.random.random_sample() * 2 * np.pi
    cdef dtype_t[:] x = np.zeros(N)
    cdef dtype_t[:] y = np.zeros(N)
    y[0] = np.random.random_sample()* 0.9999 * R

    cdef dtype_t[:] W_phi = np.random.normal(0, np.sqrt(tau), N)

    cdef dtype_t[:] W_x = np.random.normal(0, np.sqrt(tau), N)

    cdef dtype_t[:] W_y = np.random.normal(0, np.sqrt(tau), N)

    full_arrays(phi, x, y, 
                W_phi, W_x, W_y,
                tau,
                R, H, eta_0, kbT, 
                D_r, v, N,
                T, Delta_P, L, W, oscillations)
    
    return np.asarray(x), np.asarray(y), np.asarray(phi), N, tau, R, H, eta_0, kbT, D_r, v, T, Delta_P, L, W, oscillations
