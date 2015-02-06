B#!/usr/bin/env python

from math import ceil, log;

import numpy as np;
from numpy import convolve, correlate, dot, ones, sqrt;




#################
# Derivs
#################
def get_deriv(x, step, n=0):
    """Compute derivative of timeseries.

    Inputs:
        x:              Nx1 numpy array of sample points

        step:           time in sec between successive samples

        n:              a nonnegative integer giving the order of
                        the derivative to be computed
                        (default: 0)
    """
    if n == 0:
        deriv = x;
    elif n == 1:
        # calc deriv using center differences in interior
        # and first differences at boundaries
        deriv = np.gradient(x, step);
    elif n > 1:
        # calc deriv using iterated forward differences
        deriv = np.zeros(x.size);
        deriv[:-n] = np.diff(x, n) / (step**n);
    return deriv;


#################
# Smoothing
#################
def smoothe_ts(x, window='hanning', windowLength=5):
    """Return smoothed version of timeseries.

    Inputs:
        x:            Nx1 numpy array to be smoothed

        window:       Type of window to be used. Available windows
                      are listed in documentation of scipy.signal.get_window.
                      If the window does not require parameters, WINDOW
                      can be a string. Otherwise, must be a tuple whose 
                      first arg is the string name of the window with
                      following arguments as paramters.
                      (default: hanning)

        windowLength: number of samples in the window
                      (default: 5)
    """
    w = sig.get_window(window, windowLength);
    return sig.convolve(x, w / w.sum(), mode='same');


def median_smoothe_ts(x, windowLength=3):
    """
    """
    return sig.medfilter(x, windowLength);


def tukey_smoothe_ts(x, windowLengthMed=5, windowLengthLin=3):
    """Return smoothed version of x using combination of median and
    linear filters.

    Details in Tukey (1974) and  Rabiner et al. (1975).

    Tukey, J.W. (1974). "Nonlinear (nonsuperposable) methods for smoothing
        data." in Cong. Rec., 1974 EASCON. p. 673

    Rabiner, L.R., Sambur, M.R. and Schmidt, C.S. (1975). "Applications of a
        nonlinear smoothing algorithm to speech processing." IEEE Transactions
        on Acoustics, Speech, and Signal Processing. 6: 552-557

    Inputs:
        x:               Nx1 numpy array to be smoothed

        windowLengthMed: number of samples in median filter window
                         (default: 5)

        windowLengthLin: number of samples in linear filter window
                         (default: 3)
    """
    pass;








def find_peaks(x, useQuadInterp=True, ws=20):
    """Find peaks.

    Peaks are defined as locations where signal changes sign,
    midpoints of plateaus, and shoulders.

    Outputs:
        locs:   Nx1 numpy array whose i-th entry is the index
                of the i-th peak in *x*

        vals:   Nx1 array whose i-th entry is the amplitude of
                the i-th peak in *x*
    """
    # WARNING. The machinations needed to vectorize this code
    # means the logic is not totally transparent, but it is
    # reasonably fast. Probably worthwhile to rewrite in pure
    # C.
    nx = x.size;

    # first difference of x, locations of rises, and
    # locations of falls
    dx = x[1:] - x[:-1];
    r = np.nonzero(dx > 0)[0] + 1;
    f = np.nonzero(dx <0)[0] + 1;

    # if at least one rise and one fall present,
    # proceed with computation
    locs = np.array([]);
    vals = np.array([]);
    if (r.size > 0) and (f.size > 0):
        # rises
        dr = r[1:] - r[:-1];

        tmp = np.ones(nx);
        tmp[r[1:]] = 1 - dr;
        tmp[r[0]] = -r[0];
        time_from_last_rise = np.cumsum(tmp);

        tmp = np.zeros(nx) - 1;
        tmp[r[:-1]] = dr - 1;
        tmp[0] = r[0] - 1; tmp[r[-1]] = nx - r[-1] - 1;
        time_til_next_rise = np.cumsum(tmp);

        # falls
        df = f[1:] - f[:-1];

        tmp = np.ones(nx);
        tmp[f[1:]] = 1 - df;
        tmp[f[0]] = -f[0];
        time_from_last_fall = np.cumsum(tmp);

        tmp = np.zeros(nx) - 1;
        tmp[f[:-1]] = df - 1;
        tmp[0] = f[0] - 1; tmp[f[-1]] = nx - f[-1] - 1;
        time_til_next_fall = np.cumsum(tmp);

        # locate peaks
        c1 = np.logical_and(time_from_last_rise < time_from_last_fall,
                            time_til_next_fall < time_til_next_rise)
        c2 = np.logical_and(c1, np.floor((time_til_next_fall - time_from_last_rise) / 2.0) == 0);
        locs = np.nonzero(c2)[0];
        vals = x[locs];

        # perform quadratic interpolation to refine amps
        if useQuadInterp:
            b = 0.5*(x[locs+1] - x[locs-1]);
            a = x[locs] - b - x[locs-1];
            jj = a > 0;
            vals[jj] = x[locs[jj]] + 0.25*b[jj]**2 / a[jj];

        # purge nearby peaks
        if ws > 0:
            cands = np.nonzero(locs[1:] - locs[0:-1] <= ws)[0];
            while cands.any():
                elim = cands + (vals[cands] >= vals[cands+1]);
                locs = np.delete(locs, elim);
                vals = np.delete(vals, elim);
                cands = np.nonzero(locs[1:] - locs[0:-1] <= ws)[0];

        return locs, vals;




