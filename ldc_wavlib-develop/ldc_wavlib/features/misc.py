__all__ = ['get_energy', 'get_teager_energy',
           'get_dft_energy', 'get_zero_crossing_rate',
           'get_high_zero_crossing_rate_ratio',
          ];

import numpy as np;
from scipy.signal import hamming;

from .spectral import _spectral_slices_gen, _get_band_indices;
from .util import _get_timing_params;

np.pi = 3.141592653589793;

###########################
# Energy measures
###########################
def get_energy(x, sr, wl=0.05, step=0.01,
               start=0, end=None):
    """Return energy in signal.

    The energy of the signal in a window is defined as the average of the
    sum of squares of the signal in that window.
    """
    wl, times, indices = _get_timing_params(x, sr, wl, step, start, end);

    # calc energies
    window = hamming(wl);
    es = np.convolve(x**2, window**2, mode='same');

    # sample at desired freq and return
    es = es[indices, ];
    return es, times;


def get_teager_energy(x, sr, wl=0.05, step=0.01,
                      start=0, end=None):
    """Return teager energy of signal.

    Teager energy defined as in:
        Kaiser, J.F. (1990). "On a simple algorithm to calculate the 'energy' of a signal."
            In proceedings of ICASSP-90.
    """
    wl, times, indices = _get_timing_params(x, sr, wl, step, start, end);

    # calc energies
    shift_right = np.hstack([[0], x[:-1]]);
    shift_left = np.hstack([x[1:], [0]]);
    window = hamming(wl);
    es = np.convolve(x**2 - shift_left*shift_right, window**2, mode='same');
    es[es<0] = 0;

    # sample at desired freq and return
    es = es[indices, ];
    return es, times;


def get_dft_energy(x, sr, wl=0.05, step=0.01,
                   start=0, end=None,
                   nfft=512, win='hamming',
                   use_power=True,
                   bands=None, bsz=None,
                   ):
    """Return energy in signal (calculated using DFT).
    """
    wl, times, indices = _get_timing_params(x, sr, wl, step, start, end);

    # allocate memory
    band_indices = _get_band_indices(sr, nfft, bands);
    n_bands = len(band_indices);
    es = np.empty([indices.size, n_bands], dtype='float32');

    # compute energy lazily
    spec_gen = _spectral_slices_gen(x, sr, indices, wl, nfft, win, use_power, bsz);
    for Pxy, freqs, bi, ei in spec_gen:
        for ii, (band_bi, band_ei) in enumerate(band_indices):
                es[:, ii] = Pxy[:, band_bi:band_ei+1].sum(axis=1);

    return es, times;



###########################
# Miscellaneous
###########################
def get_zero_crossing_rate(x, sr, wl=0.05, step=0.01,
                           start=0.0, end=None):
    """Determine zero-crossing rates for x.

    A zero-crossing is defined to occur at sample x[n] if x[n]*x[n-1] < 0.
    Rates are calculated based on centered analysis windows (zero-padded at
    edges of signal).

    Inputs:
        x:              Nx1 numpy array of samples

        sr:             sampling frequency (Hz)

        wl:             duration of analysis window in sec
                        (default: 0.05)

        step:           time between onset of successive analysis
                        windows in sec
                        (default: 0.01)

        start:          onset in sec of region to be parameterized
                        (default: 0.0)

        end:            offset in sec of region to be parameterized
                        (default: duration of recording)
    """
    wl, times, indices = _get_timing_params(x, sr, wl, step, start, end);

    # excise interval [start, end]
    # (NOTE: as done this will cause some very
    #  minor artifacts at edges at larger window sizes.
    #  will fix in future edits)
    xt = x[indices[0]:indices[-1] + 1];
    indices -= indices[0];

    # calc zero crossings
    signs = np.sign(xt);
    diffs = signs[1:] - signs[:-1]; del signs;
    crossings = np.hstack([False, np.abs(diffs)==2]);

    # calc rates
    window = hamming(wl); window /= window.sum();
    rates = np.convolve(crossings, window, mode='same').astype('float32');

    return rates[indices], times;


def get_high_zero_crossing_rate_ratio(x, sr, wl=0.05, step=0.01,
                                      start=0.0, end=None):
    """Determine high zero-crossing rate ratios for x.

    High zero crossing rate ratio is defined in Lu and Zhang (2002) "Content analysis for audio classification and segmentation" as the proportion of frames in a window for which the zcr exceeds the 1.5*global mean.

    Inputs:
        x:              Nx1 numpy array of samples

        sr:             sampling frequency (Hz)

        wl:             duration of analysis window in sec
                        (default: 0.05)

        step:           time between onset of successive analysis
                        windows in sec
                        (default: 0.01)

        start:          onset in sec of region to be parameterized
                        (default: 0.0)

        end:            offset in sec of region to be parameterized
                        (default: duration of recording)
    """
    wl, times, indices = _get_timing_params(x, sr, wl, step, start, end);

    # excise interval [start, end]
    # (NOTE: as done this will cause some very
    #  minor artifacts at edges at larger window sizes.
    #  will fix in future edits)
    xt = x[indices[0]:indices[-1] + 1];
    indices -= indices[0];

    # calc zero crossings
    signs = np.sign(xt);
    diffs = signs[1:] - signs[:-1]; del signs;
    crossings = np.hstack([False, np.abs(diffs)==2]);

    # calc rates
    window = hamming(wl); window /= window.sum();
    rates = np.convolve(crossings, window, mode='same').astype('float32');

    # calc zcrr
    mu = rates.mean();
    zcrr = np.convolve(rates>1.5*mu, window, mode='same').astype('float32')

    return zcrr[indices], times;
