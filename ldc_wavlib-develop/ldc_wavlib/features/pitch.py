__all__ = ['get_norm_xcorr_at_lag', 'get_correlogram',
           'apply_ltm_correction'];

import numpy as np;

from .util import _get_timing_params;
import _pitch;

def get_norm_xcorr_at_lag(x, sr, wl=0.005, step=0.001,
                          start=0.0, end=None,
                          lag=1):
    """Compute normalized cross-correlation of signal with itself at specified
    lag in each window.
    """
    Cxy, lags, times = get_correlogram(x, sr, wl, step, start, end, minLag=lag, maxLag=lag);
    return Cxy[:, ], times;


def get_correlogram(x, sr, wl=0.005, step=0.001,
                     start=0.0, end=None,
                     min_lag=1, max_lag=None, afact=10000.):
    """Compute correllogram for signal.

    Inputs:
        x:           1-D numpy array of samples.

        sr:          Sample rate (Hz).

        wl:          Duration of analysis window (sec).
                     (Default: 0.025)

        step:        Time between onset of successive analysis
                     windows (sec)
                     (Default: 0.01)

        start:       Time (sec) of center of first frame.
                     (Default: 0)

        end:         Time (sec) of center of last frame. If None,
                     set to duration of recording.
                     (Default: None)

        min_lag:     Minimum lag for cross-correlation computation.
                     (Default: 1)

        max_lag:     Maximum lag for cross-correlation computation.
                     If None, then set to length of analysis window
                     (in samples).
                     (Default: None)

        afact:       Normalization constant.
                     (Default: 10000.)

    Outputs:
        Cxy:            nframes x nlags numpy array where Cxy[i,j] is the
                        normalized cross-correlation of the signal in the i-th
                        analysis window at lag lags[j]

        lags:           nlags x 1 numpy array of lags

        times:          nframes x 1 numpy array of times
    """
    wl, times, indices = _get_timing_params(x, sr, wl, step, start, end);

    if max_lag is None:
        max_lag = wl;
    lags = np.arange(min_lag, max_lag+1);
    Cxy = _pitch.get_correlogram(x, indices, wl, min_lag, max_lag, afact);

    return Cxy, lags, times;



def apply_ltm_correction(f0, n_iter=10):
    """
    Implements the Lognormal Tied Mixture-model approach to resolve octave ambiguity
    from the SRI work: M.K. Sonmez et al., "A lognormal model of pitch for prosody-based
    speaker recognition," in Proc. Eur. Conf. Speech Communication Technology, 1997,
    vol. 3, pp. 1391-1394.

    Based on Malcolm Slaney's LTMPitch.m.
    """
    log_sqrt_2pi = np.log(np.sqrt(2*np.pi));

    # transform pitch to octave scale
    is_voiced = f0 > 0;
    p = np.log2(f0[is_voiced]);
    mu = p.mean();
    sigma = p.std();

    # init weights
    # priors = [p(halve), p(corr), p(double)]
    weights = np.array([0.001, 1, 0.001]);
    weights /= weights.sum();

    # estimate mixture weights, mu, and sigma
    # via EM
    print('='*30);
    for ii in xrange(n_iter):
        # E-step
        y = np.maximum(-1, np.minimum(1, np.round(p-mu)));
        y = y.astype('int32');
        p_norm = p - y; # renormalize pitch
        w = weights[y + 1];
        ll = np.log(w) - np.log(sigma) - log_sqrt_2pi \
             - 0.5*((p_norm - mu)/sigma)**2;
        ll = ll.sum();
        print ii, ll;

        # M-step
        mu = np.mean(p_norm);
        sigma = np.std(p_norm);
        weights = np.mean([y==-1, y==0, y==1], axis=0);

    # renormalize
    f0_new = np.empty(f0.shape, dtype=f0.dtype);
    f0_new[~is_voiced] = 0;
    f0_new[is_voiced] = 2**(p_norm);

    return f0_new;


