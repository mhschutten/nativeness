__all__ = ['get_fbank',
           'hz_2_mel', 'mel_2_hz',
           'hz_2_bark', 'bark_2_hz',
          ];

import numpy as np;
from numpy import cos, sin, exp, sqrt;
from scipy.signal import lfilter;

from .spectral import _spectral_slices_gen;
from .util import _get_timing_params;


def get_fbank(x, sr, wl=0.025, step=0.01,
              start=0, end=None,
              nfft=512, win='hamming',
              freq_min=0, freq_max=8000,
              preemphasis=0.97, n_filts=20,
              use_power=False, fb_type='mel',
              bsz=None):
    """Compute filterbank.

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

        use_power:   If True, compute power spectrum prior to integrating
                     bins. Otherwise, use magintude spectrum.
                     (Default: False)

        fb_type:     Type of filterbank to use. Must be one of
                     {mel, slaney_mel, bark}.
                     (Default: mel)

        bsz:         If integer, filterbank outputs are computed in batches
                     of bsz frames at a time to reduce memory overhead. If
                     None, all coefficients are computed at once.
                     (Default: None)
    """
    wl, times, indices = _get_timing_params(x, sr, wl, step, start, end);

    # allocate memory
    n_frames = indices.size;
    Mxy = np.empty([n_frames, n_filts], dtype='float32')

    # apply preemphasis
    if preemphasis:
        x_filt = lfilter([1., -preemphasis], 1, x).astype('float32');
    else:
        x_filt = x;

    # precompute filterbank matrix
    n_freqs = nfft//2 + 1;
    freqs = float(sr) / nfft * np.arange(n_freqs);
    if fb_type == 'mel':
        wts, cfreqs, widths  = _mel_fbank_mat(freqs, n_filts, freq_min, freq_max, False);
    elif fb_type == 'slaney_mel':
        wts, cfreqs, widths  = _mel_fbank_mat(freqs, n_filts, freq_min, freq_max, True);
    elif fb_type == 'bark':
        wts, cfreqs, widths  = _bark_fbank_mat(freqs, n_filts, freq_min, freq_max);
    elif fb_type == 'gammatone':
        wts, cfreqs, widths = _gammatone_fbank_mat(sr, freqs, n_filts, freq_min, freq_max);

    # compute MFCCs lazily
    spec_gen = _spectral_slices_gen(x_filt, sr, indices, wl, nfft, win, use_power, bsz);
    for Pxy, freqs, bi, ei in spec_gen:
        Mxy[bi:ei, ] = np.dot(Pxy, wts);

    return Mxy, cfreqs, times;


def _mel_fbank_mat(freqs, n_filts, freq_min, freq_max, slaney=False):
    """Generate matrix of weights to combine FFT bins into Mel bins.

    Inputs:
        freqs:       1-D numpy array of frequencies.

        n_filts:     Number of filters in the filterbank.

        freq_min:    Frequency of lowest band edge.

        freq_max:    Frequency of highest band edge.

        slaney:     If True, use the Mel curve implemented in Malcolm Slaney's
                    Auditory Toolboz. Else, use O'Shaughnessy's Mel curve (the
                    HTK default).
                    (Default: False)
    """
    # determine center frequencies and widths (in Hz) of each band
    mel_min, mel_max = hz_2_mel(np.array([freq_min, freq_max]), slaney);
    bin_freqs = mel_2_hz(np.linspace(mel_min, mel_max, n_filts+2), slaney);
    c_freqs = bin_freqs[1:-1];
    widths = bin_freqs[2:] - bin_freqs[:-2];

    # gen filterbank weights matrix
    wts = np.zeros([freqs.size, n_filts], dtype='float32');
    for ii in xrange(n_filts):
        fs = bin_freqs[np.arange(ii, ii+3)];

        # lower and upper slopes for all bins
        lo_slope = (freqs - fs[0]) / (fs[1] - fs[0]);
        hi_slope = (fs[2] - freqs) / (fs[2] - fs[1]);

        # intersect and zero
        mn = np.min(np.c_[lo_slope, hi_slope], 1);
        zeros = np.zeros(mn.size);
        wts[:, ii] = np.max(np.c_[zeros, mn], 1);

    if slaney:
        # scale so integration windows peak sum to 1
        # CHECK ME
        D = np.diag(2. / (bin_freqs[2:n_filts+2] - bin_freqs[:n_filts]));
        wts = np.dot(wts, D);

    return wts, c_freqs, widths;


def _bark_fbank_mat(freqs, n_filts, freq_min, freq_max):
    """Generate matrix of weights to combine FFT bins into Bark bins.

    Implements a trapezoidal filterbanks where the centers of the filters
    are equally spaced on the Bark scale.

    Based on feacalc.

    Inputs:
        freqs:       1-D numpy array of frequencies.

        n_filts:     Number of filters in the filterbank.

        freq_min:    Frequency of lowest band edge.

        freq_max:    Frequency of highest band edge.
    """
    # determine bin and center freqs in barks using Hermansky's Bark scale
    bin_freqs_bark = hz_2_bark(freqs);
    bark_min, bark_max = hz_2_bark(np.array([freq_min, freq_max]));
    c_freqs_bark = np.linspace(bark_min, bark_max, n_filts);

    # determine center freqs and wdiths in Hz
    c_freqs = bark_2_hz(c_freqs_bark);
    lo_freqs = bark_2_hz(c_freqs_bark - 0.5);
    hi_freqs = bark_2_hz(c_freqs_bark + 0.5);
    widths = hi_freqs - lo_freqs;

    # gen trapezoidal filterbank weights matrix
    wts = np.zeros([freqs.size, n_filts], dtype='float32');
    for ii in xrange(n_filts):
        lo_slope = bin_freqs_bark - c_freqs_bark[ii] - 0.5;
        hi_slope = bin_freqs_bark - c_freqs_bark[ii] + 0.5;
        log_wts = np.minimum(0., np.minimum(hi_slope, -2.5*lo_slope));
        wts[:, ii] = 10**log_wts;

    return wts, c_freqs, widths;


def _gammatone_fbank_mat(sr, freqs, n_filts, freq_min=200, freq_max=8000):
    """Generate matrix of weights to combine FFT bins into Gammatone
    filter bank outputs.

    Based on Malcom Slaney's MakeERBFilters and Dan Ellis' fft2gammatonemx.m.

    References
    ----------
    Ellis, D. (2009). "Gammatone-like spectrograms."
    http://www.ee.columbia.edu/ln/rosa/matlab/gammatonegram/

    Slaney, M. Auditory Toolbox Version 2.
    https://engineering.purdue.edu/~malcolm/interval/1998-010

    Inputs:
        sr:        Sample rate (Hz).

        freqs:     1-D numpy array of frequencies.

        n_filts:   Number of filters in the filterbank.

        freq_min:  Frequency of lowest band edge.

        freq_max:  Frequency of highest band edge.
    """
    # calculate gammatone filer parameters following MakeERBFilter.m
    fcoefs, c_freqs, bin_widths = _get_erb_filter_coefs(sr, n_filts, freq_min, freq_max);
    A11 = fcoefs[:, 1];
    A12 = fcoefs[:, 2];
    A13 = fcoefs[:, 3];
    A14 = fcoefs[:, 4];
    gain = fcoefs[:, 9];

    # translate to weights on fft bins
    n_freqs = freqs.size;
    ucirc = np.exp(1j*2*np.pi*np.linspace(0, 0.5, n_freqs));
    GTord = 4.;
    T = 1 / float(sr);
    wts = np.empty([n_freqs, n_filts], dtype='float32');
    for ii in xrange(n_filts):
        cf = c_freqs[ii];
        bw = bin_widths[ii];
        r = np.exp(-bw/float(sr));
        theta = 2*np.pi*cf/float(sr);
        pole = r*np.exp(1j*theta);
        wts[:, ii] = ((T**4)/gain[ii]) * \
                   np.abs(ucirc + A11[ii]/T) * \
                   np.abs(ucirc + A12[ii]/T) * \
                   np.abs(ucirc + A13[ii]/T) * \
                   np.abs(ucirc + A14[ii]/T) * \
                   (np.abs((pole - ucirc)*(np.conj(pole) - ucirc))**-GTord);
    return wts, c_freqs, bin_widths;


def _get_erb_filter_coefs(sr, n_filts, freq_min, freq_max):
    """Compute filter coefficients for a bank of Gammatone filters as
    defined by Patterson and Holdsworth for simulating the cochlea.

    Adapted from Malcom Slaney's MakeERBFilters.m from Auditory Toolbox.

    References
    ----------
    Slaney, M. Auditory Toolbox Version 2.
    https://engineering.purdue.edu/~malcolm/interval/1998-010

    Slaney, M. (1993). "An efficient implementation of the Patterson-Holdsworth
    auditory filter bank." Apple Technical Report #35

    Glassberg, B.R. and Moore, B.C.J. (1990). "Derivation of auditory filter shapes
    from notched-noise data."

    Patterson, R.D., Robinson, K., Holdsworth, J., McKeown, D., Zhang, C.,
    and Allerhand, M.H. (1992). "Complex sounds and auditory images."


    Inputs:
        sr:        Sample rate (Hz).

        n_filts:   Number of filters in the filterbank.

        freq_min:  Frequency of lowest band edge.

        freq_max:  Frequency of highest band edge.

    Outputs:
        fcoefs:    2-D numpy array, each row of which contains the
                   coefficients for four second order filters.
                   filters

        c_freqs:

        widths:
    """
    # Glassberg and Moore defaults
    EarQ = 9.26449;  # asymptotic filter quality at large frequencies
    minBW = 24.7;    # minimum bandwidth for low frequency channels
    order = 1;       # filter order
    T = 1/float(sr);

    # Determine center freqs (Hz) for filters (TR #35: 6)
    C = EarQ*minBW;
    inds = np.arange(1, n_filts+1);
    c_freqs = np.exp(inds*(-np.log(freq_max + C) + np.log(freq_min + C))/float(n_filts))*(freq_max + C) - C;
    c_freqs = c_freqs[::-1];

    # Determine filter widths (Hz) (TR #35: p. 2)
    widths = ((c_freqs/EarQ)**order + minBW**order);
    widths = widths**(1/float(order));
    widths *=1.019; # Patterson (1992) suggests using 1.019*ERB for bandwidth
    B = 2*np.pi*widths;  # bandwidth in radians

    # Determine filter coefficients
    # (this is pretty much a direct translation
    # of Slaney's MakeERBFilters.m)
    A0 = T;
    A2 = 0;
    B0 = 1;
    B1 = -2*cos(2*c_freqs*np.pi*T)/np.exp(B*T);
    B2 = np.exp(-2*B*T);

    A11 = -(2*T*cos(2*c_freqs*np.pi*T)/exp(B*T) + \
            2*sqrt(3+2**1.5)*T*sin(2*c_freqs*np.pi*T)/exp(B*T)) / 2.;
    A12 = -(2*T*cos(2*c_freqs*np.pi*T)/exp(B*T) - \
            2*sqrt(3+2**1.5)*T*sin(2*c_freqs*np.pi*T)/exp(B*T)) / 2.;
    A13 = -(2*T*cos(2*c_freqs*np.pi*T)/exp(B*T) + \
            2*sqrt(3-2**1.5)*T*sin(2*c_freqs*np.pi*T)/exp(B*T)) / 2.;
    A14 = -(2*T*cos(2*c_freqs*np.pi*T)/exp(B*T) - \
            2*sqrt(3-2**1.5)*T*sin(2*c_freqs*np.pi*T)/exp(B*T)) / 2.;

    gain = abs((-2*exp(4*1j*c_freqs*np.pi*T)*T + \
                 2*exp(-(B*T) + 2*1j*c_freqs*np.pi*T)*T* \
                                (cos(2*c_freqs*np.pi*T) - \
                                sqrt(3 - 2**1.5)*sin(2*c_freqs*np.pi*T))) * \
               (-2*exp(4*1j*c_freqs*np.pi*T)*T + \
                 2*exp(-(B*T) + 2*1j*c_freqs*np.pi*T)*T* \
                   (cos(2*c_freqs*np.pi*T) + \
                    sqrt(3 - 2**1.5)*sin(2*c_freqs*np.pi*T))) * \
               (-2*exp(4*1j*c_freqs*np.pi*T)*T + \
                 2*exp(-(B*T) + 2*1j*c_freqs*np.pi*T)*T* \
                 (cos(2*c_freqs*np.pi*T) - sqrt(3 + 2**1.5)* \
                                             sin(2*c_freqs*np.pi*T)))  * \
               (-2*exp(4*1j*c_freqs*np.pi*T)*T + \
                 2*exp(-(B*T) + 2*1j*c_freqs*np.pi*T)*T* \
                 (cos(2*c_freqs*np.pi*T) + \
                  sqrt(3 + 2**1.5)*sin(2*c_freqs*np.pi*T)))  / \
               (-2/exp(2.*B*T) - 2*exp(4*1j*c_freqs*np.pi*T) + \
                2*(1 + exp(4*1j*c_freqs*np.pi*T)) / exp(B*T))**4.);

    #
    fcoefs = np.column_stack([A0*np.ones(n_filts),
                              A11,
                              A12,
                              A13,
                              A14,
                              A2*np.ones(n_filts),
                              B0*np.ones(n_filts),
                              B1,
                              B2,
                              gain]);

    return fcoefs, c_freqs, widths;


############
# Scales
############
def hz_2_mel(freqs, slaney=False):
    """Convert frequencies (in Hz) to mel scale.

    Inputs:
        freqs:  1-D numpy array of freqs (Hz)

        slaney:     If True, use the Mel curve implemented in Malcolm Slaney's
                    Auditory Toolboz. Else, use O'Shaughnessy's Mel curve (the
                    HTK default).
                    (Default: False)
    """
    if slaney:
        # conversion params
        low_freq = 0;      # 133.33333 often used to skip LF region
        lin_step = 200/3.; # 66.666667
        log_step= np.exp(np.log(6.4) / 27);
        brk_frq = 1000.;                          # starting freq for log region
        brk_pt = (brk_frq - low_freq) / lin_step; # starting mel value for log
                                                  # region

        # now fill in the linear and log part seperately
        z = np.zeros(freqs.size);
        is_linear = freqs < brk_frq;
        z[is_linear] = (freqs[is_linear] - low_freq) / lin_step;
        z[~is_linear] = (brk_pt + (np.log(freqs[~is_linear] / brk_frq)) / np.log(log_step));
    else:
        z = 2595. * np.log10(1 + freqs/700.);
    return z;


def mel_2_hz(z, slaney=False):
    """Convert mel scale frequencies to Hz.

    Inputs:
        freqs:  1-D numpy array of freqs (Hz)

        slaney:     If True, use the Mel curve implemented in Malcolm Slaney's
                    Auditory Toolboz. Else, use O'Shaughnessy's Mel curve (the
                    HTK default).
                    (Default: False)
    """
    if slaney:
        # conversion params
        low_freq = 0;      # 133.33333 often used to skip LF region
        lin_step = 200/3.; # 66.666667
        log_step= np.exp(np.log(6.4) / 27);
        brk_frq = 1000.;                           # starting freq for log region
        brk_pt = (brk_frq - low_freq) / lin_step;  # starting mel value for log
                                                   # region

        # now fill in the linear and log part seperately
        freqs = np.zeros(z.size);
        is_linear = z < brk_pt;
        freqs[is_linear] = lin_step*z[is_linear] + low_freq;
        freqs[~is_linear] = brk_frq*np.exp(np.log(log_step)*(z[~is_linear] - brk_pt));
    else:
        freqs = 700 * (10**(z/2595.) - 1)
    return freqs;


def hz_2_bark(freqs):
    """Convert frequencies (in Hz) to bark.

    Equivalent to formula from Hermansky (1989).

    References
    ----------
    Hermansky, H. (1989). "Perceptual linear predictive (PLP) analysis
    of speech."

    Inputs:
        freqs:  1-D numpy array of freqs (Hz)
    """
    barks = 6*np.arcsinh(freqs/600.);
    return barks;


def bark_2_hz(barks):
    """Convert frequencies (in bark) to Hz.

    See also hz_2_bark.

    Inputs:
        barks:  1-D numpy array of freqs (barks)
    """
    freqs = 600*np.sinh(barks/6.);
    return freqs;
