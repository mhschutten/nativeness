__all__ = ['get_plp'];

import numpy as np;
from scipy.fftpack import fft, ifft;
from scipy.signal import lfilter;

from .fbank import _bark_fbank_mat, _mel_fbank_mat;
from .mfcc import lifter;
from .spectral import _spectral_slices_gen;
from .util import _get_timing_params;
import _plp;


###########################
# PLP
###########################
def get_plp(x, sr, wl=0.025, step=0.01,
            start=0, end=None,
            nfft=512, win='hamming',
            freq_min=0, freq_max=8000,
            preemphasis=0, n_filts=21, n_ceps=13,
            order=12, htk_style=False,
            bsz=None):
    """Compute perceptual linear prediction (PLP) coefficient features.

    Both feacalc-style and HTK-style PLP features are supported via the
    argument htk_style. For details on the underlying algorithm consult
    Hermanksy (1989).

    References
    ----------
    Hermansky, H. (1989). "Perceptual linear predictive (PLP) analysis
    of speech."

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

        nfft:        Number of points to use in FFT when comupting
                     spectra. If None, uses window length (in samples).
                     (Default: 512)

        win:         Window to use in framing signal.
                     Must be one of 'blackman,' 'blackmanharris,'
                     'chebwin,' 'gaussian,' or 'hamming.'
                     (Default: hamming)

        freq_min:    Lowest band edge in filterbank.
                     (Default: 0)

        freq_max:    Highest band edge in filterbank.
                     (Default: 8000) 
                    
        preemphasis: Coefficient for preemphasis filter. If 0,
                     no preemphasis filtering is performed.
                     (Default: 0)

        n_filts:     Number of bands in filterbank.
                     (Default: 21)

        n_ceps:      Number of coefficients to keep.
                     (Default: 13)

        order:       PLP model order.
                     (Default: 12)

        htk_style:   If True, compute HTK-style PLP features.
                     Else, computes feacalc-style.
                     (Default: False)

        bsz:         If integer, PLP coefficients are computed
                     in batches of bsz frames at a time to
                     reduce memory overhead. If None, all 
                     coefficients are computed at once.
                     (Default: None)

    Outputs:
        Cxy:         2-D numpy array with each row a vector of
                     PLP coefficients.

        times:       1-D numpy array containing the times (sec)
                     of the frame centers.
    """
    wl, times, indices = _get_timing_params(x, sr, wl, step, start, end);

    # allocate memory
    n_frames = indices.size;
    Cxy = np.empty([n_frames, n_ceps], dtype='float32');

    # apply preemphasis
    if preemphasis:
        x_filt = lfilter([1., -preemphasis], 1, x).astype('float32');
    else:
        x_filt = x;

    # precompute filterbank matrix
    n_freqs = nfft//2 + 1;
    freqs = float(sr) / nfft * np.arange(n_freqs);
    if htk_style:
        wts, cfreqs, widths  = _mel_fbank_mat(freqs, n_filts, freq_min, freq_max, False);
    else:
        wts, cfreqs, widths  = _bark_fbank_mat(freqs, n_filts, freq_min, freq_max);

    # precompute weights for equal loudness preemphasis
    cfreqs_sq = cfreqs**2;
    eql = (cfreqs_sq/(cfreqs_sq + 1.6e5))**2 * (cfreqs_sq + 1.44e6)/(cfreqs_sq + 9.61e6);

    # determine lifter_exp:
    if htk_style:
        lifter_exp = -22;
    else:
        lifter_exp = 0.6;

    # compute PLPs lazily
    spec_gen = _spectral_slices_gen(x_filt, sr, indices, wl, nfft, win, True, bsz);
    for Pxy, freqs, bi, ei in spec_gen:
        # filter through triangular filterbank
        Mxy = np.dot(Pxy, wts);

        # equal loudness preemphasis
        Mxy *= eql;

        # cubic root amplitude compression
        Mxy = Mxy**0.33;

        # replace first and last bands with adjacent bands (they
        # are unreliable as calculated)
        Mxy[:, 0] = Mxy[:, 1];
        Mxy[:, -1] = Mxy[:, -2];

        # lpc analysis
        acoeff, e = lpc(Mxy, order, False);

        # convert lpc to cepstra
        Cxy_ = lpc_2_cepstrum(acoeff, e, n_ceps);

        # lifter
        lifter(Cxy_, lifter_exp, in_place=True);

        Cxy[bi:ei, ] = Cxy_;

    return Cxy, times;


def lpc_2_cepstrum(acoeff, e, n_ceps=None):
    """Calculate real cepstrum.

    Calculate cepstra from spectral samples.

    Inputs:
        Pxy:     2-D numpy array whose rows are LPC 'a' coefficients.

        e:

        n_ceps:  Number cepstral coefficient to keep.
                 If None, then full cepstrum is returned.
                 (Default: None)
    """
    if n_ceps is None:
        n_ceps = acoeff.shape[1];
    Cxy = _plp.lpc_2_cepstrum(acoeff, e, n_ceps)

    return Cxy.real;


def lpc(x, order=8, norm=False):
    """Perform LPC analysis.
    """
    nr, nc = x.shape;

    # calculate autocorrelation
    xt = np.column_stack([x, x[:, -2:0:-1]]);
    Cxy = ifft(xt, axis=1).real
    Cxy = Cxy[:, :nc];

    # LPC coeffs via Levinson-Durbin recursion
    acoeff, e = _plp.levinson(Cxy, order);

    # normalize polynomials by gain
    if norm:
        acoeff = (acoeff.T / acoeff.T).T;

    return acoeff, e;
