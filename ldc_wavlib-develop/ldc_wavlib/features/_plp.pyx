import cython
import numpy as np
cimport numpy as np


cdef extern from "math.h":
    float log(float x)


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




@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False) # turn off negative-indices for entire function
@cython.cdivision(True)
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
cpdef lpc_2_cepstrum(float [:, :] acoeff, float [:] e, unsigned int n_ceps):
    """
    """
    # see http://labrosa.ee.columbia.edu/matlab/rastamat/lpc2cep.m
    # but remember that they put frames in columns
    cdef:
        unsigned int n_frames, ii, jj, kk
        float accum

    # allocate memory
    n_frames = acoeff.shape[0];
    Cxy = np.empty([n_frames, n_ceps], dtype='float32');
    cdef float [:, :] Cxy_mv = Cxy

    # convert
    for ii in xrange(n_frames):
        Cxy_mv[ii, 0] = log(e[ii]);
        for jj in xrange(1, n_ceps):
            accum = 0.0;
            for kk in xrange(1, jj+1):
                accum += (jj - kk)*acoeff[ii, kk]*Cxy_mv[ii, jj - kk];
            Cxy_mv[ii, jj] = -(acoeff[ii, kk] + accum / jj);

    return Cxy
