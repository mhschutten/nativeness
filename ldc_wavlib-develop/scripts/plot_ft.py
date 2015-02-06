#!/usr/bin/env python

import argparse;
import os;
import sys;

import numpy as np;
from matplotlib import pyplot as pyp;
from matplotlib.backends.backend_pdf import PdfPages;

from ldc_wavlib.features import *;
from ldc_wavlib.io import read_htk_label_file, wav_2_array;
from ldc_wavlib.plots import *;
from ldc_wavlib.plots import _plot_spectrogram;


####################
# Ye olde' main
####################
if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser(description='''Display spectrogram, features,
                                                    and segments for .wav files.''',
                                     add_help=False,
                                     usage='%(prog)s [options] wfs');

    # analysis args
    group1 = parser.add_argument_group('analysis params');
    group1.add_argument('--ftype', nargs='?', default='zcr',
                        choices=['wf',
                                 'corr',
                                 'mfcc',
                                 'gammatone',
                                 'plp',
                                 'energy',
                                 'f0',
                                 'nxc',
                                 'zcr',
                                 'se', 'we', 'sc', 'ss',
                                 'test',
                                 'zcr'
                                 ],
                        metavar='ftype', dest='ftype',
                        help="Set feature to be displayed (default: %(default)s)");
    group1.add_argument('--wl', nargs='?', default=0.005,
                        type=float, metavar='x', dest='wl',
                        help='Set wl for feature sampling (default: %(default)s s)');
    group1.add_argument('--step', nargs='?', default=0.001,
                        type=float, metavar='x', dest='step',
                        help='Set step for feature sampling (default: %(default)s s)');
    group1.add_argument('-s', '--start', nargs='?', default=0.0,
                        type=float, metavar='t', dest='start',
                        help='start at time t seconds (default: %(default)s)');
    group1.add_argument('-e', '--end', nargs='?', default=None,
                        type=float, metavar='t', dest='end',
                        help='end at time t seconds (default: duration of wav)');

    # spectrogram args
    group2 = parser.add_argument_group('spectrogram params');
    group2.add_argument('--nfft', nargs='?', default=1024,
                        type=float, metavar='x', dest='nfft',
                        help='Set NFFT for spectrogram DFT (default: %(default)s)');
    group2.add_argument('--win', nargs='?', default='gaussian',
                        choices = ['gaussian',
                                   'hamming',
                                   'blackman',
                                   'blackmanharris',
                                   'chebwin',
                                  ],
                        metavar='x', dest='win',
                        help='Set window type (default: %(default)s)');
    group2.add_argument('--topclip', nargs='?', default=-10,
                        type=float, metavar='x', dest='top_clip',
                        help='Set topclip for spectrogram in dB. (default: max dB in image)');
    group2.add_argument('--dr', nargs='?', default=50.0,
                        type=float, metavar='x', dest='dynamic_range',
                        help='Set dynamic range for spectrogram in dB (default: %(default)s)');
    group2.add_argument('--cmap', nargs='?', default='gray',
                        metavar='s', dest='cmap',
                        help='Set spectrogram colormap (default: grayscale)');

    # io args
    group3 = parser.add_argument_group('io');
    group3.add_argument('wfs', nargs='*',
                        help='wav files to be processed');
    group3.add_argument('-S', nargs='?', default=None,
                        metavar='f', dest='scpf',
                        help='Set script file (default: %(default)s)');
    group3.add_argument('-L', nargs='?', default='./',
                        metavar='dir', dest='lab_dir',
                        help="Set input label dir (default: %(default)s)");
    group3.add_argument('-X', nargs='?', default='lab',
                        metavar='ext', dest='lab_ext',
                        help="Set input label file ext (default: %(default)s)");
    group3.add_argument('--in_htk_units',
                        default=True, action='store_false',
                        dest='in_sec',
                        help='dfdf');
    group3.add_argument('--pdf', nargs='?', default=None,
                        metavar='f', dest='pdff',
                        help="""Set pdf output file for plots
                                (default: %(default)s)""");
    args = parser.parse_args();

    if len(sys.argv) == 1:
        parser.print_help();


    # load wfs from scpf
    if not args.scpf is None:
        with open(args.scpf, 'r') as f:
            args.wfs = [l.strip() for l in f];

    # open pdf file for output if specified
    if not args.pdff is None:
        pp = PdfPages(args.pdff);

    # iterate through files, displaying or saving to pdf
    for wf in args.wfs:
        sr, x = wav_2_array(wf);

        # extract features
        step, wl = args.step, args.wl;
        start, end = args.start, args.end;
        if args.ftype == 'zcr':
            zcr, times = get_zero_crossing_rate(x, sr, wl, step, start, end);
            zcrr, times = get_high_zero_crossing_rate_ratio(x, sr, wl, step, start, end);
            y = np.column_stack([zcr, zcrr]);
            y_min=0;
            y_max=1.01;
            y_label = 'zcr';
            labels = ['zcr', 'zcrr'];
        elif args.ftype == 'nxc':
            min_freq = 50.;
            max_freq = 500.;
            min_lag = int(sr/max_freq);
            max_lag = int(sr/min_freq);
            Cxy, lags, times = get_correlogram(x, sr, wl, step,
                                                start=args.start, end=args.end,
                                                min_lag=min_lag, max_lag=max_lag,
                                                );
            y = np.max(Cxy, axis=1);
            y_min = y.min(); y_max = y.max();
            y_label = 'nxcorr';
            labels = ['max'];
        elif args.ftype == 'energy':
            bands = [(0, 8000),
                     (0, 4000),
                     (4000, 8000)];
            es, times = get_dft_energy(x, sr, wl, step,
                                       start=args.start, end=args.end,
                                       nfft=args.nfft, win=args.win,
                                       bands=bands);
            les = 10*np.log10(es + 1);
            y = les;
            y_min, y_max = 10, 120;
            y_label = 'dB';
            labels = ['<8k', '<4k', '>4k']
        elif args.ftype == 'f0':
            f0, times = get_f0_from_file(wf, wl, step,
                                        start=args.start, end=args.end);
            times[f0==0] = np.nan;
            f0[f0==0] = np.nan;
            y = f0;
            y_min, y_max =  50, 350;
            y_label = 'Hz';
            labels = [];
        elif args.ftype == 'se':
            se, times = get_spectral_entropy(x, sr, wl, step,
                                             start=args.start, end=args.end,
                                             nfft=args.nfft, win=args.win);
            y = se;
            y_min, y_max = 4, 8;
            y_label = 'se';
            labels = [];
        elif args.ftype == 'sc':
            sc, times = get_spectral_centroid(x, sr, wl, step,
                                              start=args.start, end=args.end,
                                              nfft=args.nfft, win=args.win);

            y = sc;
            y_min, y_max = 0, 8000.;
            y_label = 'Frequence (Hz)';
            labels = ['sc'];
        elif args.ftype == 'ss':
            ss, times = get_spectral_slope(x, sr, wl, step,
                                           start=args.start, end=args.end,
                                           nfft=args.nfft, win=args.win);
            y = ss;
            y_min, y_max = -1, 1;
            y_label = '';
            labels = ['ss'];
        elif args.ftype == 'we':
            we, times = get_wiener_entropy(x, sr, wl, step,
                                           start=args.start, end=args.end,
                                           nfft=args.nfft, win=args.win);
            y = we;
            y_min, y_max =  -15, 0;
            y_label = 'we';
            labels = ['we'];

        # create figure and subplots
        fig = pyp.figure(figsize=(24,8)); #figsize=(24,12)
        axs = [pyp.subplot2grid((5,1), (0,0), rowspan=2)];
        axs.append(pyp.subplot2grid((5,1), (2,0), rowspan=2, sharex=axs[-1]));
        axs.append(pyp.subplot2grid((5,1), (4,0), sharex=axs[-1]));
        fig.subplots_adjust(hspace=0.1);
        fig.suptitle(wf);

        # row 1: plot spectrogram
        pyp.sca(axs[0]);
        plot_spectrogram(x, sr, 0.005, 0.001,
                         start=args.start, end=args.end,
                         nfft=1024, win=args.win,
                         cmap=args.cmap, top_clip=args.top_clip,
                         dynamic_range=args.dynamic_range);
        # row 2: plot timeseries
        pyp.sca(axs[1]);
        if args.ftype == 'wf':
            plot_waveform(x, sr, args.start, args.end);
        elif args.ftype == 'corr':
            plot_correlogram(x, sr, wl, step,
                              start=args.start, end=args.end,
                              min_f0=50, max_f0=500);
        elif args.ftype == 'mfcc':
            Cxy, times = get_mfcc(x, sr, wl, step,
                                  start=args.start, end=args.end,
                                  nfft=args.nfft);
            Cxy, times = Cxy.astype('float32'), times.astype('float32');
            Cxy = Cxy[:, 1:];
            axs[1].matshow(Cxy.T, aspect='auto', origin='lower',
                           extent=[times[0], times[-1], 1, Cxy.shape[1]],
                           );
        elif args.ftype == 'plp':
            Cxy, times = get_plp(x, sr, wl, step,
                                 start=args.start, end=args.end,
                                 nfft=args.nfft);
            Cxy, times = Cxy.astype('float32'), times.astype('float32');
            Cxy = Cxy[:, 1:];
            axs[1].matshow(Cxy.T, aspect='auto', origin='lower',
                           extent=[times[0], times[-1], 1, Cxy.shape[1]],
                           );
        elif args.ftype == 'gammatone':
            Mxy, c_freqs, times = get_fbank(x, sr, wl, step,
                                            start=args.start, end=args.end,
                                            nfft=args.nfft, preemphasis=0.0,
                                            n_filts=64, use_power=True);
            _plot_spectrogram(Mxy, c_freqs, times,
                              cmap=args.cmap, top_clip=args.top_clip,
                              dynamic_range=args.dynamic_range);
        else:
            plot_timeseries(times, y, labels=labels,
                            x_min=args.start, x_max=args.end,
                            y_min=y_min, y_max=y_max, y_label=y_label);
            axs[1].axhline(0, times[0], times[-1], c='r');
        # row 3: plot labels
        try:
            pyp.sca(axs[2]);

            # read segs
            bn = os.path.basename(wf);
            lf = os.path.join(args.lab_dir, bn.replace('.wav', '.%s' % args.lab_ext));
            segs = read_htk_label_file(lf, in_sec=args.in_sec);

            # plot
            plot_segs(segs, start=args.start, end=args.end);
        except IOError:
            pass;

        # elim xlabels from first two panels
        for ax in axs[:2]:
            ax.set_xlabel('');
            ax.xaxis.set_visible(False);

        # display or save
        if args.pdff:
            pp.savefig();
        else:
            pyp.show();

        # clear memory
        pyp.close();

    # close file object of output specified
    if args.pdff:
        pp.close();
