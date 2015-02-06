import numpy as np;


# see sklearn.util.validation for array conversion utilities

def _get_timing_params(x, sr, wl, step, start=0.0, end=None):
    """Determine analysis window duration and centers (in indices).

    Inputs:
        x:              Nx1 numpy array of samples

        sr:             sampling frequency (Hz)

        wl:             duration of analysis window in sec

        step:           time between onset of successive analysis
                        windows in sec

        start:          onset in sec of region to be parameterized
                        (default: 0.0)

        end:            offset in sec of region to be parameterized.
                        If None, offset set to be duration of recording
                        (default: None)
    """
    # get window length and step size in indices
    if end is None:
        duration = float(x.size) / sr;
        end = duration;
    wl, stp = np.round(sr * np.array([wl, step])).astype('int');
    wl = int(wl);

    # determine indices for centers
    bi, ei = times_2_indices([start, end], x, sr)
    indices = np.arange(bi, ei + 1, stp, 'int32');

    # in seconds
    times = indices / float(sr);
    return wl, times, indices;


def times_2_indices(times, sample, sampleRate, startTime=0.0):
    """Convert array of timepoints to array of indices of sample intervals.

    Times outside the interval covered by signal are automatically mapped to
    nearest endpoint.

    Inputs:
        times:       N x 1 numpy array of sample times (in seconds)
        sample:      numSamples x 1 numpy array containing the sample intervals
        sampleRate:  sampling freq (in Hz)
        startTime:   time (in seconds) corresponding to first sample point of
                     samples
    """
    # map timepoint t to inf({n | s(n) <= t })
    # correct indices out of array bounds
    indices = np.round((np.array(times) - startTime) * sampleRate).astype('int32');
    indices[indices < 0 ] = 0;
    indices[indices > (sample.size - 1)] = sample.size - 1;
    return indices;

def rasta_filter(Pxy, step=0.001, lofreq=0.9, hifreq=12.8):
    """Perform RASTA filtering of spectrogram.

    Based on Malcom Slaney's implementation in auditory toolbox,
    which approximates the filters of Hermansky and Morgan (1994)
    with butterworth filters.

    Inputs:
        Pxy:     nframes x nfreqs numpy array where
                 Pxy[i,j] is the power at freqs[j]
                 at times[i]

        step:    duration in seconds between successive slices.
                 (default: 0.001)

        lofreq:  beginning, in Hz, of the passband

        hifreq:  end, in Hz, of the passband
    """
    # compress via log transform
    rPxy = np.log(Pxy+1);

    # filter
    theta = 1;
    w1 = lofreq*2*pi*step;
    w2 = hifreq*2*pi*step;
    a = np.cos((w1+w2)/2.) / np.cos((w2-w1)/2.);
    k = np.tan(theta/2.) / np.tan((w2-w1)/2);

    b = np.array([1, 0, -2, 0, 1]);
    a = np.array([1. + 2.*2.**0.5*k + 4.*k**2.,
                  -4.*2.**0.5*a*k - 16.*a*k**2.,
                  -2. + 8.*k**2. + 16.*a**2.*k**2.,
                  4.*2.**0.5*a*k - 16.*a*k**2.,
                  1. - 2.*2.**0.5*k + 4.*k**2.]);
    scale = b[0];
    b = b/scale;
    a = a/scale;

    rPxy = lfilter(b, a, rPxy, axis=0);

    # decompress
    rPxy = np.exp(rPxy);

    return rPxy;
