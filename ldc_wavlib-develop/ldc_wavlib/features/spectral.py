__all__ = ['get_spectral_slices', 'get_spectral_entropy',
           'get_spectral_renyi_entropy', 'get_wiener_entropy',
           'get_spectral_centroid', 'get_spectral_slope',
          ];

import numpy as np;
from numpy import log2, log10, mean, sum;
from scipy.fftpack import fft;
from scipy import signal as sig;
from scipy.stats import gmean;

from ._spectral import enframe;
from .util import _get_timing_params;


########################
# Spectral estimation
########################
def get_spectral_slices(x, sr, wl=0.025, step=0.01,
                        start=0.0, end=None,
                        nfft=512, win='hamming',
                        use_power=True, bsz=None,
                        ):
    """Compute spectral slices for real signal.

    Spectral slices are computed every *step* seconds based on centered
    analysis windows of length *wl* seconds.

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

        nfft:        Number of points to use in FFT. If None, uses window length
                     (in samples).
                     (Default: 512)

        use_power:   If True, return power spectrum. Otherwise, return magnitude
                     spectrum.
                     (Default: False)

        bsz:         If integer, filterbank outputs are computed in batches
                     of bsz frames at a time to reduce memory overhead. If
                     None, all coefficients are computed at once.
                     (Default: None)

    Outputs:
        Pxy:            2-D numpy array where Pxy[i,j] is the power at
                        freqs[j] at times[i].

        freqs:          1-D numpy array of frequencies.

        times:          1-D numpy array of times.
    """
    wl, times, indices = _get_timing_params(x, sr, wl, step, start, end);

    # compute slices
    n_freqs = nfft//2 + 1;
    Pxy = np.empty([indices.size, n_freqs], dtype='float32');
    spec_gen = _spectral_slices_gen(x, sr, indices, wl, nfft, win,
                                   use_power, bsz);
    for Pxy_, freqs, bi, ei in spec_gen:
        Pxy[bi:ei, ] = Pxy_

    return Pxy, freqs, times;


def _spectral_slices_gen(x, sr, indices, wl, nfft, win,
                        use_power, bsz):
    """Generate spectral slices in batches of length bsz.
    """
    x = x.astype('float32', copy=False);

    # generate appropriate window
    if win == 'gaussian':
        wl*=2;
        window = sig.gaussian(wl, wl / 10.); # praat style gaussian window
    elif win == 'chebwin':
        window = sig.chebwin(wl, 80);
    else:
        window = sig.get_window(win, wl);
    window /= np.linalg.norm(window); # scale by window norm to
                                      # to account for windowing loss
                                      # (Bendat and Piersol)
    window = window.astype('float32');

    # calc frequencies
    if nfft is None:
        nfft = wl;
    n_freqs = nfft//2 + 1;
    freqs = float(sr) / nfft * np.arange(n_freqs);

    # generate slices lazily
    if bsz is None:
        bsz = indices.size;
    bi = 0;
    while bi < indices.size:
        ei = bi + bsz;

        # compute mag spectrum
        xy = enframe(x, indices[bi:ei], window);
        Pxy = fft(xy, n=nfft); del xy;
        Pxy = Pxy[:, :freqs.size]; # keep positive freqs only
        Pxy = np.abs(Pxy);

        # convert to power spectrum if specified
        if use_power:
            Pxy = Pxy**2;

        # correct for one-sided density
        Pxy *= 2.;

        yield Pxy, freqs, bi, ei;
        bi = ei;



##########################
# Spectral shape measures
##########################
def get_spectral_entropy(x, sr, wl=0.025, step=0.01,
                         start=0.0, end=None,
                         nfft=512, win='hamming',
                         use_power=False,
                         bands=None, bsz=None):
    """Return shannon entropy of signal in window.
    """
    wl, times, indices = _get_timing_params(x, sr, wl, step, start, end);

    # allocate memory
    band_indices = _get_band_indices(sr, nfft, bands);
    n_bands = len(band_indices);
    se = np.empty([indices.size, n_bands], dtype='float32');

    # compute spectral entropy lazily
    spec_gen = _spectral_slices_gen(x, sr, indices, wl, nfft, win, use_power, bsz);
    for Pxy, freqs, bi, ei in spec_gen:
        Pxy += 1e-6;
        for ii, (band_bi, band_ei) in enumerate(band_indices):
            p = _spectrum_2_density(Pxy[:, band_bi:band_ei+1]);
            se[bi:ei, ii] = -1 * sum(p * log2(p), 1);

    return se, times;


def get_spectral_renyi_entropy(x, sr, wl=0.025, step=0.01,
                               start=0.0, end=None,
                               nfft=512, win='hamming',
                               use_power=False,
                               alpha=.5, bands=None, bsz=None):
    """Return renyi entropy of signal in window.
    """
    wl, times, indices = _get_timing_params(x, sr, wl, step, start, end);

    # allocate memory
    band_indices = _get_band_indices(sr, nfft, bands);
    n_bands = len(band_indices);
    sre = np.empty([indices.size, n_bands], dtype='float32');

    # compute spectral entropy lazily
    spec_gen = _spectral_slices_gen(x, sr, indices, wl, nfft, win, use_power, bsz);
    for Pxy, freqs, bi, ei in spec_gen:
        Pxy += 1e-6;
        for ii, (band_bi, band_ei) in enumerate(band_indices):
            p = _spectrum_2_density(Pxy[:, band_bi:band_ei+1]);
            sre[bi:ei, ii] = 1 / (1 - alpha) * log2(sum(p**alpha, axis=1));

    return sre, times;


def get_wiener_entropy(x, sr, wl=0.025, step=0.01,
                       start=0.0, end=None,
                       nfft=512, win='hamming',
                       use_power=False,
                       bands=None, bsz=None):
    """Return wiener entropy of signal in window.

    Wiener entropy (spectral flatness) is the ratio  of the geometric
    and arithmetic means of the spectrum.
    """
    wl, times, indices = _get_timing_params(x, sr, wl, step, start, end);

    # allocate memory
    band_indices = _get_band_indices(sr, nfft, bands);
    n_bands = len(band_indices);
    we = np.empty([indices.size, n_bands], dtype='float32');

    # compute wiener entropy lazily
    spec_gen = _spectral_slices_gen(x, sr, indices, wl, nfft, win, use_power, bsz);
    for Pxy, freqs, bi, ei in spec_gen:
        Pxy += 1e-6;
        for ii, (band_bi, band_ei) in enumerate(band_indices):
            p = _spectrum_2_density(Pxy[:, band_bi:band_ei+1]);
            bPxy = Pxy[:, band_bi:band_ei+1];
            we[bi:ei, ii] = 10*log10(gmean(bPxy, axis=1) / mean(bPxy, axis=1));

    return we, times;


def get_spectral_centroid(x, sr, wl=0.025, step=0.01,
                          start=0.0, end=None,
                          nfft=512, win='hamming',
                          use_power=False, smoothing=0.0,
                          bands=None, bsz=None):
    """
    """
    wl, times, indices = _get_timing_params(x, sr, wl, step, start, end);

    # allocate memory
    band_indices = _get_band_indices(sr, nfft, bands);
    n_bands = len(band_indices);
    sc = np.empty([indices.size, n_bands], dtype='float32');

    # compute wiener entropy lazily
    spec_gen = _spectral_slices_gen(x, sr, indices, wl, nfft, win, use_power, bsz);
    for Pxy, freqs, bi, ei in spec_gen:
        for ii, (band_bi, band_ei) in enumerate(band_indices):
            p = _spectrum_2_density(Pxy[:, band_bi:band_ei+1]);
            sc[bi:ei, ii] = np.dot(p, freqs[band_bi:band_ei+1]);

    return sc, times;


def get_spectral_slope(x, sr, wl=0.025, step=0.01,
                       start=0.0, end=None,
                       nfft=512, win='hamming',
                       use_power=False,
                       bands=None, bsz=None):
    """
    """
    wl, times, indices = _get_timing_params(x, sr, wl, step, start, end);

    # allocate memory
    band_indices = _get_band_indices(sr, nfft, bands);
    n_bands = len(band_indices);
    slopes = np.empty([indices.size, n_bands], dtype='float32');

    # compute wiener entropy lazily
    spec_gen = _spectral_slices_gen(x, sr, indices, wl, nfft, win, use_power, bsz);
    for Pxy, freqs, bi, ei in spec_gen:
        for ii, (band_bi, band_ei) in enumerate(band_indices):
            A = np.column_stack([np.ones(band_ei-band_bi+1), freqs[band_bi:band_ei+1]]);
            A = A.astype('float32');
            B = Pxy[:, band_bi:band_ei+1].T;
            X, res, rnk, s = np.linalg.lstsq(A, B);
            slopes[bi:ei, ii] = X[1, :];

    return slopes, times;



###########################
# Utility
###########################
def _spectrum_2_density(Pxy):
    """
    """
    n_frames, n_freqs = Pxy.shape;
    p = Pxy / np.tile(Pxy.sum(1), [n_freqs, 1]).T;
    return p;


def _get_band_indices(sr, nfft, bands):
    """
    """
    n_freqs = nfft//2 + 1;
    freqs = sr / float(nfft) * np.arange(n_freqs);
    if bands is None:
        bands = [[0, freqs[-1]]];
    pairs = [];
    for freq_min, freq_max in bands:
        bw = freqs[1] - freqs[0];
        bi = np.where((freqs - freq_min) > -bw/2.)[0][0];
        ei = np.where((freq_max - freqs) > -bw/2.)[0][-1];
        pairs.append([bi, ei]);
    return pairs;
