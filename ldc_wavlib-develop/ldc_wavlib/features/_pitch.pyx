import cython
import numpy as np
cimport numpy as np


cdef extern from "math.h":
    float sqrt(float x)




from scipy.linalg.blas import fblas
from cpython cimport PyCObject_AsVoidPtr

ctypedef float (*sdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
cdef sdot_ptr sdot=<sdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # float = dot(x, y) 

REAL = np.float32
ctypedef np.float32_t REAL_t
cdef int ONE = 1


@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False) # turn off negative-indices for entire function 
cdef float mean(float [:] x):
    cdef:
        unsigned int ii
        float mu = 0.0
    for ii in xrange(x.size):
        mu += x[ii];
    mu  /= x.size;

    return mu;

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False) # turn off negative-indices for entire function  
cdef float dot(float [:] x, float [:] y, int x_bi, int y_bi, int n) nogil:
    cdef:
        unsigned int ii
        float dp = 0.0
    for ii in xrange(n):
        dp += x[x_bi++ii]*y[y_bi+ii];
    return dp;






# LPC
@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False) # turn off negative-indices for entire function
cpdef levinson(float [:, :] r, n=None):
    """Find coefficients of autoregressive linear process by
    Levinson-Durbin recursion.

    Inputs:
        r: Autocorrelation sequence of process. If 2-D, each
           row is treated as being from a separate process.

        n: Order of denominator polynomial. If None, order is set
           to r.shape[0]-1
           (Default: None)

    Outputs:
        acoeff: coefficients of autoregressive model

        error: prediction error
    """
    cdef:
        unsigned int order
        unsigned int nr, nc, ii, jj, kk
        float kcoeff # reflection coefficient

    # determine order
    nr = r.shape[0];
    nc = r.shape[1];
    if n is None:
        order = nc-1;
    else:
        order = n;

    # allocate memory and set memory views
    acoeff = np.empty([nr, order+1], dtype=np.float32); # LP coefficients
    tmp_acoeff = np.empty(order+1, dtype=np.float32);
    error = np.zeros(nr, dtype=np.float32);
    cdef float[:, :] acoeff_mv = acoeff
    cdef float[:] error_mv = error
    cdef float[:] tmp_acoeff_mv = tmp_acoeff

    # compute LP coefficients
    for ii in xrange(nr):
        error_mv[ii] = r[ii, 0];
        acoeff_mv[ii, 0] = 1.0;   # order 0
        for jj in xrange(1, order+1):
            # next reflection coefficient
            kcoeff = r[ii, jj];
            for kk in xrange(1, jj):
                kcoeff += acoeff_mv[ii, kk]*r[ii, jj-kk];
            kcoeff = -kcoeff / error_mv[ii];
            acoeff_mv[ii, jj] = kcoeff;

            # new LP coefficients
            for kk in xrange(order):
                tmp_acoeff_mv[kk] = acoeff_mv[ii, kk];
            for kk in xrange(1, jj):
                acoeff_mv[ii, kk] += kcoeff*tmp_acoeff_mv[jj-kk];

            # new error
            error_mv[ii] *= 1 - kcoeff*kcoeff;

    return acoeff, error;



@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False) # turn off negative-indices for entire function
cpdef acorr(float [:, :] x):
    """Compute autocorrelation sequence for rows of a 2-D array.

    For each row of x, compute the cross-correlation of the row with
    itself at lags from 0 to n_cols - 1;

    Input:
        x:    2-D numpy array

    Output:
        corr: 2-D numpy array, the i-th row of which is the
              autocorrelation sequence of the i-th row of x.
    """
    cdef:
        unsigned int nr, nc, ii, jj, lag
        float accum

    # allocate array
    nr = x.shape[0];
    nc = x.shape[1];
    corr = np.empty([nr, nc], dtype='float32');
    cdef float[:, :] corr_mv = corr

    # compute
    for ii in xrange(nr):
        for lag in xrange(nc):
            accum = 0.0;
            for jj in xrange(nc - lag):
                accum += x[ii, jj]*x[ii, jj+lag];
            corr_mv[ii, lag] = accum;

    return corr;


def lpc(x, order=8):
    """
    """
    # calculate autocorrelation

    # find coefficients by Levinson-Durbin recursion

    # normalize
    return 0;



# correlations
def get_correlogram(x, indices, wl,
                     min_lag, max_lag,
                     afact=10000.):
    """                                     
    """
    cdef:
        unsigned int npl, npr, n_frames, n_lags

    # create zero-padded version of x for computing
    # cross-correlations
    npl = wl + np.abs(min_lag);
    npr = wl + max_lag;
    xzp = np.r_[np.zeros(npl, 'float32'), x, np.zeros(npr, 'float32')];
    ind_zp = indices + npl;

    # Cxy[i, j] = normalized cross-correlation of
    # i-th frame with itself at lag lags[j]
    n_frames = ind_zp.size;
    n_lags = max_lag - min_lag + 1;
    Cxy = np.empty([n_frames, n_lags], dtype='float32');
    #print Cxy.shape;

    if wl > 50:
        get_correlogram2(xzp, ind_zp, wl, min_lag, max_lag, afact, Cxy);
    else:
        get_correlogram1(xzp, ind_zp, wl, min_lag, max_lag, afact, Cxy);

    return Cxy;


@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False) # turn off negative-indices for entire function
cdef get_correlogram1(float [:] xzp, int [:] ind_zp, int wl,
                     int min_lag, int max_lag,
                     float afact, float [:, :] Cxy):
    """
    """
    cdef:
        unsigned int n_frames, n_lags, wl2
        unsigned int ii, jj, ref_bi, ref_ei, lagged_bi, lagged_ei
        int lag
        float corr, e1
        double mu, e2

    # compute
    wl2 = wl // 2;
    n_frames = Cxy.shape[0];
    n_lags = Cxy.shape[1];
    for ii in xrange(n_frames):
        # get reference window bounds
        ref_bi = ind_zp[ii] - wl2;
        ref_ei = ref_bi + wl;

        # calc reference window mean
        mu = 0.0;
        for jj in xrange(ref_bi, ref_ei):
            mu += xzp[jj];
        mu /= wl; # cast wl first?

        # calc reference window energy
        e1 = 0.0;
        for jj in xrange(ref_bi, ref_ei):
            e1 += (xzp[jj] - mu)**2;

        # compute energy at first lag
        lagged_bi = ref_bi + min_lag;
        lagged_ei = ref_ei + min_lag;
        e2 = 0.0;
        for jj in xrange(lagged_bi, lagged_ei):
            e2 += (xzp[jj] - mu)**2;

        # calculate nxc at each lag
        for lag in xrange(min_lag, max_lag + 1):
            # get lagged window bounds
            lagged_bi = ref_bi + lag;
            lagged_ei = ref_ei + lag;

            # calc corr
            corr = 0.0;
            for jj in xrange(wl):
                corr += (xzp[ref_bi + jj] - mu)*(xzp[lagged_bi + jj] - mu);

            # nxc
            Cxy[ii, lag - min_lag] = corr / sqrt(e1*e2 + afact);

            # adjust window energy for next lag
            e2 -= (xzp[lagged_bi] - mu)**2;
            e2 += (xzp[lagged_ei] - mu)**2;
#            if e2 < 0.0:
#                e2 = 0.0; # in case of roundoff error




####
# With an accelerated BLAS, this is faster than the above method when wl is large.
# In practice, this benefit is not seen until wl is > 40 or so. 
@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False) # turn off negative-indices for entire function
cdef get_correlogram2(float [:] xzp, int [:] ind_zp, int wl,
                     int min_lag, int max_lag,
                     float afact, float [:, :] Cxy):
    """
    """
    cdef:
        unsigned int wl2, n_frames, n_lags
        unsigned int local_win_bi, local_win_ei # indices of local window for each NCCF computation
        unsigned int local_win_len 
        unsigned int ref_bi, ref_ei # indices of reference window relative to local window
        unsigned int lagged_bi, lagged_ei, min_lagged_bi # indices of lagged window relative to local window
        unsigned int ii, jj
        int lag
        float corr, e1, mu
        double e2

    # local_win will hold just the samples relevant to the
    # cross-correlation computation for a single frame
    cdef np.ndarray[np.float32_t, ndim=1] local_win
    if min_lag < 0:
        local_win_len = max_lag - min_lag + wl + 1;  
        ref_bi = -min_lag;
        min_lagged_bi = 0;
    else:
        local_win_len = max_lag + wl + 1;
        ref_bi = 0;
        min_lagged_bi = min_lag;
    ref_ei = ref_bi + wl;
    local_win = np.empty(local_win_len, dtype='float32');
    cdef float [:] local_win_mv = local_win

    # compute
    wl2 = wl // 2;
    n_frames = Cxy.shape[0];
    n_lags = Cxy.shape[1];
    for ii in xrange(n_frames):
        # get local window
        local_win_bi = ind_zp[ii] - wl2 - ref_bi;
        local_win_ei = local_win_bi + local_win_len
        local_win[:] = xzp[local_win_bi:local_win_ei]

        # mean center
        mu = mean(local_win[ref_bi:ref_ei]);
        for jj in xrange(local_win_len):
            local_win_mv[jj] -= mu;

        # calc reference window energy
        e1 = sdot(&wl, &local_win_mv[ref_bi], &ONE, &local_win_mv[ref_bi], &ONE)

        # compute energy at first lag
        e2 = sdot(&wl, &local_win_mv[min_lagged_bi], &ONE, &local_win_mv[min_lagged_bi], &ONE)
        
        # calculate nxc at each lag
        for lag in xrange(min_lag, max_lag + 1):
            lagged_bi = min_lagged_bi + lag - min_lag;
            lagged_ei = lagged_bi + wl;

            # calc corr
            corr = sdot(&wl, &local_win_mv[ref_bi], &ONE, &local_win_mv[lagged_bi], &ONE)

            # nxc
            Cxy[ii, lag - min_lag] = corr / sqrt(e1*e2 + afact);

            # adjust window energy for next lag
            e2 -= local_win_mv[lagged_bi]**2;
            e2 += local_win_mv[lagged_ei]**2;
#            if e2 < 0.0:
#                e2 = 0.0;

