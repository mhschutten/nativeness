__all__ = ['plot_timeseries', 'plot_waveform', 'plot_spectrogram',
           'plot_correlogram', 'plot_segs',];

from matplotlib import pyplot as pyp;
import numpy as np;
from numpy import logical_and as log_and;

from .features import get_spectral_slices, get_correlogram;


from matplotlib.backends import pylab_setup;
#new_figure_manager, draw_if_interactive, _show = pylab_setup()[-3:];
_backend_mod, new_figure_manager, draw_if_interactive, _show = pylab_setup();


# generic
def plot_timeseries(times, x, labels=None, x_min=0.0, x_max=None,
                    y_min=None, y_max=None, y_label='', ax = None):
    """Plot timeseries data.

    Inputs:
        times:   1-D numpy array of times (sec).

        x:       1-D or 2-D numpy array. If 2-D, each column
                 contains a separate timeseries.

        labels:  List of labels for the timeseries to plot in
                 legend. If None, no legend is displayed.
                 (Default: None)

        x_min:   int
                 (Default: 0)

        x_max:   int. If None, set to times[-1].
                 (Default: None)

        y_min:   int. If None, set to x.min().
                 (Default: None)

        y_max:   int. If None, set to x.max().
                 (Default: None)

        y_label: Y-axis label.
                 (Default: '')

        ax:      Matplotlib Axes instance. If None, defaults to the
                 current axis.
                 (Default: None)
    """
    # get axis
    if ax is None:
        ax = pyp.gca();

    # plot
    ax.plot(times, x);
    if labels:
        ax.legend(labels, loc='upper right');

    # get x-axis/y-axis bounds
    if x_max is None:
        x_max = times[-1];
    if y_min is None:
        y_min = x.min();
    if y_max is None:
        y_max = x.max();

    # format
    _format_axes(ax, '',
                 x_min, x_max, 'Time (sec)',
                 y_min, y_max, y_label);

    draw_if_interactive();


# Waveforms
def plot_waveform(x, sr, start=0.0, end=None, y_min=None, y_max=None,
                  ax=None):
    """Plot waveform normalized to lie on interval [-1, 1].

    Inputs:
        x:       1-D numpy array of samples.

        sr:      Sample rate (Hz).

        x_min:   Time (sec) of beginning of segment to plot.
                 (Default: 0)

        x_max:   Time (sec) of end of segment to plot. If None,
                 set to duration of recording.
                 (Default: None)

        y_min:   int. If None, defaults to minimum of normalized x.
                 (Default: None)

        y_max:   int. If None, defaults to maximum of normalized x.
                 (Default: None)

        ax:      Matplotlib Axes instance. If None, defaults to the
                 current axis.
                 (Default: None)
    """
    # get axis
    if ax is None:
        ax = pyp.gca();

    times = np.arange(x.size) / float(sr);
    amps = x / 32768.;  # assume LPCM encoding
    if y_min is None:
        y_min = amps.min();
    if y_max is None:
        y_max = amps.max();
    plot_timeseries(times, amps, x_min=start, x_max=end,
                    y_min=y_min, y_max=y_max, ax=ax);


# Spectrogram plotting
def plot_spectrogram(x, sr, wl=0.005, step=0.001,
                     start=0.0, end=None,
                     nfft=512, win='hamming',
                     use_power=True, cmap='gray',
                     preemphasis = 6.0, dynamic_compression = 0,
                     top_clip = -10, dynamic_range = 50.0,
                     ax=None):
    """
    Inputs:
        x:           1-D numpy array of samples.

        sr:          Sample rate (Hz).

        wl:          Duration of analysis window (sec).
                     (Default: 0.005)

        step:        Time between onset of successive analysis
                     windows (sec)
                     (Default: 0.001)

        start:       Time (sec) of center of first frame.
                     (Default: 0)

        end:         Time (sec) of center of last frame. If None,
                     set to duration of recording.
                     (Default: None)

        nfft:        Number of points to use in FFT. If None, uses window length
                     (in samples).
                     (Default: 512)

        use_power:   If True, compute power spectrum. Otherwise, compute
                     magnitude spectrum.
                     (Default: True)

        preemphasis: dB/octave for high-pass filtering
                     (Default: 6.0)

        cmap:        Matplotlib Colormap instance or string naming a
                     Colormap instance defined in matplotlib.cm.
                     Useful values: 'gray', 'blues', 'jet', 'spectral',
                     'seismic', 'gnuplot', and 'heatmap'.
                     (Default: 'gray')

        dynamic_compression: Controls amount by which weaker frames are enhanced
                             in the direction of the strongest frame. Must be on
                             interval [0, 1] with 0 = no compression and
                             1 = complete compression.
                             (Default = 0.0)

        top_clip:    Maximum intensity (dB) displayed. If >0, maximum intensity
                     set to top_clip. If <=0, maximum intensity set to
                     maximum observed value in image - abs(top_clip).
                     (Default: -10)

        dynamic_range:  Difference (in dB) between maximum intensity and
                        lowest visible intensity
                        (Default: 50)

        ax:          matplotlib Axes instance
    """
    # get power spectrum
    Pxy, freqs, times = get_spectral_slices(x, sr, wl, step, start, end,
                                            nfft, win, use_power);
    _plot_spectrogram(Pxy, freqs, times, cmap, preemphasis, dynamic_compression,
                      top_clip, dynamic_range, ax);


def _plot_spectrogram(Pxy, freqs, times, cmap='gray',
                      preemphasis = 6.0, dynamic_compression = 0,
                      top_clip = -10, dynamic_range = 50.0,
                      ax=None):
    """Helper function for plotting spectrograms and spetrogram like
    outputs.

    Inputs:
        Pxy: 2-D numpy array containing power spectrum in rows.
    """
    # get axis
    if ax is None:
        ax = pyp.gca();

    # convert to dB
    Pxy += 1; # ensure min(Pxy) == 0
    Pxy = 10*np.log10(Pxy);

    # apply pre-emphasis
    if preemphasis:
        preemphasis_factors = preemphasis * np.log2(freqs / 1000. + 1);
        Pxy[:, ] += preemphasis_factors;
        Pxy += preemphasis_factors;

    # dynamic compression
    if dynamic_compression:
        dynamic_factors = dynamic_compression * (Pxy.max(axis=1) - Pxy.max(axis=1));
        Pxy = (Pxy.T + dynamic_factors).T;

    # set topclip and restrict dynamic range
    if top_clip <= 0:
        top_clip = np.max(Pxy) + top_clip; # interpret relative to image max
    bottom_clip = top_clip - dynamic_range;

    # plot the image with specified limits on freqs, and times
    if cmap == 'gray':
        cmap = 'gray_r';
    imgplot = ax.imshow(Pxy.T, cmap, aspect='auto', origin='lower',
                        extent=[times[0], times[-1], freqs[0], freqs[-1]],
                        interpolation='bicubic');
    imgplot.set_clim(bottom_clip, top_clip);

    # format axes
    _format_axes(ax, '',
                 times[0], times[-1], 'Time (sec)',
                 freqs[0], freqs[-1], 'Frequency (Hz)');

    draw_if_interactive();


def plot_aud_spectrogram(x, sr, wl=0.008, step=0.001,
                         tc=0.008, fac=0.1,
                         start=0.0, end=None,
                         cmap='gray',
                         ax=None):
    """
    """
    # get axis
    if ax is None:
        ax = pyp.gca();

    Pxy, freqs, times = wav_2_aud(x, sr, wl, step, start=start, end=end);

    # convert to dB
    Pxy = ne.evaluate('10 * log10(Pxy + 1)');

    # set appropriate color map
    # plot the image with specified limits on freqs, and times
    if cmap == 'gray':
        cmap = cm.gray_r;
    elif cmap == 'blues':
        cmap = cm.Blues;
    elif cmap == 'jet':
        cmap = cm.jet;
    elif cmap == 'spectral':
        cmap = cm.spectral;
    elif cmap == 'seismic':
        cmap = cm.seismic;
    elif cmap == 'gnuplot':
        cmap = cm.gnuplot;
    elif cmap == 'heatmap':
        cmap = cm.hot;

    # plot the image with specified limits on freqs, and times
    imgplot = ax.imshow(Pxy.T, cmap, aspect='auto', origin='lower',
                        extent=[times[0], times[-1], freqs[0], freqs[-1]],
                        interpolation='bicubic');

    # format axes
    _format_axes(ax, '',
                 times[0], times[-1], 'Time (sec)',
                 freqs[0], freqs[-1], 'Frequency (Hz)');

    draw_if_interactive();


# cross-correlogram
def plot_correlogram(x, sr, wl=0.005, step=0.001,
                      start=0.0, end=None,
                      min_f0=50, max_f0=500,
                      cmap='jet',
                      ax=None):
    """Plot correlogram.

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

        min_f0:      Minimum f0 value for cross-correlation computation.
                     (Default: 50)

        max_f0:      Maximum f0 value for cross-correlation computation.
                     (Default: 500)
    """
    # get axis
    if ax is None:
        ax = pyp.gca();

    # determine max lag based on minF0
    min_lag = int(sr / float(max_f0));
    max_lag = int(sr / float(min_f0));

    # compute nxcorrs
    Cxy, lags, times = get_correlogram(x, sr, wl, step, start, end,
                                        min_lag=min_lag, max_lag=max_lag);

    # plot
    imgplot = ax.imshow(Cxy.T, aspect='auto', origin='lower',
                        extent=[times[0], times[-1], lags[0], lags[-1]],
                        );

    # format axes
    _format_axes(ax, '',
                 times[0], times[-1], 'Time (sec)',
                 lags[0], lags[-1], 'Lag');

    draw_if_interactive();


# htk label files
def plot_segs(segs, start=0.0, end=None, seg_fontsize=12,
              ax=None):
    """
    """
    # get axis
    if ax is None:
        ax = pyp.gca();

    # load lab file
    onsets, offsets, labs = zip(*segs);
    onsets = np.array(onsets);
    offsets = np.array(offsets);
    durs = offsets - onsets;

    # determine bounds
    duration = offsets[-1];
    if end is None:
        end = duration;

    # plot
    ax.bar(onsets, np.ones(onsets.size), durs, color='white');

    # determine positions for centers of segment labels
    labLocs = onsets + durs/2;
    isLeftEdge = log_and(log_and(onsets < start, offsets > start), offsets <= end);
    labLocs[isLeftEdge] = (start + offsets[isLeftEdge])/2;
    isRightEdge = log_and(log_and(offsets > end, onsets >= start), onsets < end);
    labLocs[isRightEdge] = (onsets[isRightEdge] + end)/2;

    # add in segment labels
    for ii in xrange(onsets.size):
        if offsets[ii] >= start and onsets[ii] <= end:
            ax.text(labLocs[ii], 0.5, labs[ii], fontsize=seg_fontsize,
                    horizontalalignment='center', verticalalignment='center');

    # format axes
    _format_axes(ax, '',
                 start, end, 'Time (sec)',
                 0, 1, '');
    ax.yaxis.set_visible(False);

    draw_if_interactive();


# utility
def _format_axes(ax, title,
                 x_min, x_max, xlabel,
                 y_min, y_max, y_label):
    """
    """
    # set title
    ax.set_title(title);

    # format x-axis
    ax.set_xlim(x_min, x_max);
    ax.set_xlabel(xlabel);
    ax.xaxis.set_tick_params(which='both',
                             direction='out',
                             length=5, width=1,
                             bottom='on', top='off',
                             labelsize=10,
                             );

    # format y-axis
    ax.set_ylim(y_min, y_max);
    ax.set_ylabel(y_label);
    ax.yaxis.set_tick_params(which='both',
                             direction='out',
                             length=5, width=1,
                             left='on', right='off',
                             labelsize=10,
                             );
