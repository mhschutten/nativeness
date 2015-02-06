import cython
import numpy as np
cimport numpy as np


@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False) # turn off negative-indices for entire function
cpdef enframe(float[:] x, int [:] indices, float[:] window):
    """Efficiently extract and window frames from signal.

    Inputs:
        x:        numpy array of samples

        indices:  numpy array of indices of window centers

        window:   numpy array giving window function
    """
    cdef:
        unsigned int ii, jj, nf, wl, wl2, n_samp
        int bi, kk

    n_samp = x.size;
    nf = indices.size;
    wl = window.size;
    wl2 = wl // 2;

    # window
    F = np.empty([nf, wl], dtype='float32');
    cdef float [:, :] F_mv = F
    for ii in xrange(nf):
        bi = indices[ii] - wl2;
        for jj in xrange(wl):
            kk = bi + jj;
            if 0 <= kk < n_samp:
                F_mv[ii, jj] = x[kk]*window[jj]
            else:
                F_mv[ii, jj] = 0.0;

    return F;
