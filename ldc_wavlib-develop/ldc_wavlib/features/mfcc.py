__all__ = ['get_mfcc', 'mfcc_2_spectrum'];

import numpy as np;
from scipy.fftpack import fft, ifft;
from scipy.signal import lfilter;

from .fbank import _mel_fbank_mat;
from .spectral import _spectral_slices_gen;
from .util import _get_timing_params;


###########################
# MFCC
###########################
def get_mfcc(x, sr, wl=0.025, step=0.01,
             start=0, end=None,
             nfft=512, win='hamming',
             freq_min=0, freq_max=8000,
             preemphasis=0.97, n_filts=20, n_ceps=13,
             lifter_exp=-22, dct_type=3, slaney=False,
             bsz=None):
    """Compute mel frequency cepstral coefficient (MFCC) features.

    This is based on Dan Ellis' rastamat function melfcc and can be
    used to compute HTK or Auditory Toolbox style MFCC features. To
    compute HTK-style, set:
        lifter_exp = -22
        dct_type = 3
        slaney = False
   For Auditory Toolbox style:
        lifter_exp = 0
        dct_type = 2
        slaney = True

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

        nfft:        Number of points to use in FFT when computing
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

        lifter_exp:  Exponent for liftering. If 0, no liftering
                     is performed to the cepstral coefficients.
                     If <0, HTK-style sin liftering is performed.
                     (Default: -22)

        dct_type:    Type of DCT to use. For HTK, use 3.
                     For Auditory Toolbox use 2.
                     (Default: 3)

        slaney:      If True, Slaney's mel curve is used. Otherwise,
                     use O'Shaugnessy's as in HTK.
                     (Default: False)

        bsz:         If integer, MFCCs are computed in batches of bsz
                     frames at a time to reduce memory overhead. If None,
                     all coefficients are computed at once.
                     (Default: None)

    Outputs:
        Cxy:         2-D numpy array with each row a vector of MFCCs.

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

    # precompute triangular filterbank matrix
    n_freqs = nfft//2 + 1;
    freqs = float(sr) / nfft * np.arange(n_freqs);
    wts, cfreqs, widths  = _mel_fbank_mat(freqs, n_filts,
                                          freq_min, freq_max, slaney);

    # compute MFCCs lazily
    spec_gen = _spectral_slices_gen(x_filt, sr, indices, wl, nfft, win,
                                    False, bsz);
    for Pxy, freqs, bi, ei in spec_gen:
        # filter through triangular filterbank
        Mxy = np.dot(Pxy, wts);
        Mxy = Mxy**2;

        # convert to cepstrum via dct
        Cxy_ = spectrum_2_cepstrum(Mxy, n_ceps, dct_type); del Mxy;

        # lifter
        lifter(Cxy_, lifter_exp, in_place=True);

        Cxy[bi:ei, ] = Cxy_;

    return Cxy, times;


def mfcc_2_spectrum(Cxy, freqs, freq_min=0, freq_max=8000,
                    n_filts=20, lifter_exp=-22, dct_type=3,
                    slaney=False, use_power=True):
    """Convert from MFCCs back to spectra.
    """
    # undo liftering
    Cxy_lift = lifter(Cxy, lifter_exp, invert=True);

    # map back from ceptrsum to log mag spectrum
    Mxy = cepstrum_2_spectrum(Cxy_lift, n_filts, dct_type);
    del Cxy_lift;

    # undo mel-scaling
    wts, cfreqs, widths = _mel_fbank_mat(freqs, n_filts, freq_min, freq_max, slaney);
    ww = np.dot(wts, wts.T);
    ww_diag_mean_v = np.mean(diag(ww)) * np.ones(freqs.size);
    ww_sum = np.sum(ww, 0);
    v = np.max(np.r_[ww_diag_mean_v/100., ww_sum]);
    iwts = wts.T / np.tile(v, [n_filts, 1]);
    Pxy = np.dot(sqrt(Mxy), iwts);
    del Mxy;

    if use_power:
        Pxy = Pxy**2;

    return Pxy;


def spectrum_2_cepstrum(Pxy, n_ceps=None, dct_type=0):
    """Calculate real cepstrum

    Calculate cepstra from spectral samples.

    Inputs:
        Pxy:     nframes x nfreqs numpy array of
                 spectral slices

        n_ceps:   number cepstral coefficient to keep.
                 If None, then full cepstrum is returned.
                 (default: None)

        dct_type: if 0, cepstrum is computed using ifft.
                 if 2 or 3, dct of corresponding type is used
    """
    n_frames, n_freqs = Pxy.shape;
    if n_ceps is None:
        n_ceps = n_freqs;

    if dct_type == 0:
        Cxy = ifft(log(0.00001 + Pxy))[:, :n_ceps];
    elif dct_type in [2, 3]:
        dctm = np.zeros([n_freqs, n_ceps], dtype='float32');
        for ii in xrange(n_ceps):
            thetas = ii*np.pi*np.arange(1, 2*n_freqs, 2)/(2.*n_freqs);
            dctm[:, ii] = np.cos(thetas) * np.sqrt(2./n_freqs);
        if dct_type == 2:
            # make unitary
            dctm[:, 0] /= np.sqrt(2);
        Cxy = np.dot(np.log(0.00001 + Pxy), dctm);

    return Cxy.real;


def cepstrum_2_spectrum(Cxy, n_freqs=None, dct_type=0):
    """Recover spectrum.

    Recovers spectrum from real cepstrum.

    Inputs:
        Pxy:     nframes x nceps numpy array of
                 cepstral slices

        n_freqs:  num frequency components to reconstruct
                  in spectrum. If None, set to nceps.
                  NOTE: this option does not work when
                  dct_type set to 0
                  (default: None)

        dct_type: if 0, spectrum computed using fft.
                  if 2 or 3, dct of corresponding type is used
                  (default: 0)
    """
    n_frames, n_ceps = Cxy.shape;
    if n_freqs is None or dct_type == 0:
        n_freqs = n_ceps;

    if dct_type == 0:
        Pxy = fft(Cxy).real;
        Pxy = np.exp(Pxy);
    elif dct_type in [2, 3]:
        idctm = np.zeros([n_ceps, n_freqs]);
        for ii in xrange(n_ceps):
            thetas = ii*np.pi*np.arange(1, 2*n_freqs, 2)/(2.*n_freqs);
            idctm[ii, :] = np.cos(thetas) * np.sqrt(2./n_freqs);
        if dct_type == 2:
            # make unitary
            idctm[0, :] /= sqrt(2);
        else:
            idctm[0, :] /= 2;
        Pxy = exp(np.dot(Cxy, idctm));

    return Pxy;


def lifter(Cxy, lift, invert=False, in_place=False):
    """Apply lifter to matrix of cepstra.

    Inputs:
        Cxy:      n_frames x n_ceps numpy matrix of cepstra

        lift:     If positive, exponent of liftering.
                  If negative integer, the length of htk-style
                  sin-curve liftering. Alternately, may be an
                  nceps x 1 numpy array specifying the lifter
                  window

        in_place:
    """
    n_frames, n_ceps = Cxy.shape;
    if np.isscalar(lift):
        if lift == 0:
            return Cxy;
        elif lift > 0:
            liftwts = np.r_[1, np.arange(1, n_ceps)**lift];
        else:
            # HACK (htk-style liftering)
            L = -lift;
            liftwts = np.r_[1, (1 + L/2. * np.sin(np.arange(1, n_ceps)*np.pi/L))];
    else:
        liftwts = lift;

    if invert:
        liftwts = 1./liftwts;

    if in_place:
        Cxy[:, ] *= liftwts;
        return Cxy;

    return Cxy * liftwts;

