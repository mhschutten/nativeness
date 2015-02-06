__all__ = ['get_f0_from_file'];

import os;
import shutil;
import subprocess;
import tempfile;

import numpy as np;

from ..io import wav_2_array;
from .util import _get_timing_params;


#########################################
# Exceptions
#########################################
class GetF0Error(Exception):
    pass;

#################
# get_f0
#################
def get_f0_from_file(wf, wl=0.005, step=0.01,
                     start=0.0, end=None,
                     min_f0=60.0, max_f0=650.0):
    """Extract f0 values from wav file using RAPT.

    This wraps the ESPS command line tool get_f0, which
    MUST be on your path.

    References
    ----------
    Talkin, D. (1995). "A Robust Algorithm for Pitch Tracking (RAPT)."


    Inputs:
        wf:     Path to audio file in PCM wav format.

        wl:     Duration of analysis window (sec).
                (Default: 0.005)

        step:   Time between onset of successive analysis
                windows (sec).
                (Default: 0.01)

        start:  Time (sec) of center of first frame.
                (Default: 0)

        end:    Time (sec) of center of last frame. If None,
                set to duration of recording.
                (Default: None)

        min_f0: Lower bound of F0 search range (Hz).
                (Default: 60.0)

        max_f0: Upper bound of F0 search range (Hz).
                (Default: 650.0)
    """
    # check that params in range expected by get_f0
    sr, x = wav_2_array(wf);
    duration = x.size / float(sr);
    if end is None:
        end = duration;
    _check_f0_params(sr, wl, step, min_f0, max_f0,
                     duration, start, end);

    # create tmp dir
    tmpdir = tempfile.mkdtemp();

    # gen param files for get_f0 and fea_print
    paramf = os.path.join(tmpdir, 'params');
    _gen_param(paramf, min_f0, max_f0,
               wl, step);
    layoutf = os.path.join(tmpdir, 'layout');
    _gen_layout(layoutf);

    # calc f0 (get_f0)
    f0_binf = os.path.join(tmpdir, 'tmp.f0');
    with open(os.devnull, 'w') as f:
        cmd = ['get_f0', '-P', paramf,
               '-s', '%f:%f' % (start, end),
               wf, f0_binf];
        subprocess.call(cmd, stdout=f, stderr=f);

    # convert binary to text (fea_print)
    f0_txtf = os.path.join(tmpdir, 'tmp.af0');
    with open(f0_txtf, 'w') as f:
        subprocess.call(['fea_print', layoutf, f0_binf], stdout=f);

    # read f0 vals
    X = np.loadtxt(f0_txtf, dtype='float32').reshape((-1, 4));
    f0 = X[:, 0];

    # calculate times and pad
    wl, times, indices = _get_timing_params(x, sr, wl, step, start, end);
    n_pad = indices.size - f0.size;
    f0 = np.pad(f0, [0, n_pad], mode='edge');

    # clean up
    shutil.rmtree(tmpdir);

    return f0, times;


def _check_f0_params(sr, wl, step, min_f0, max_f0,
                     duration, start, end):
    """
    """
    if start < 0 or start > duration:
        raise GetF0Error("ERROR: start must be in [0, duration]");
    if end < 0 or end > duration:
        raise GetF0Error("ERROR: end must be in [0, duration]");
    if end <= start:
        raise GetF0Error("ERROR: require start < end");
    if wl < 0.0001 or wl > 0.01:
        raise GetF0Error("ERROR: wl must be in [0.0001, 0.01].");
    if step > 0.1 or step < 1/sr:
        raise GetF0Error('ERROR: frame step parameters must be in [1/sample rate, 0.1] ');
    if max_f0 < min_f0 or max_f0 >= sr/2.0 or min_f0 < sr/10000.:
        raise GetF0Error('ERROR: min(max) f0 params inconsistent with sample rate.');


def _gen_param(fn, min_f0, max_f0, wl, step):
    """Write esps param file for get_f0.

    Inputs:
        fn:            path to param file

        minF0:         min f0 value to consider for
                       candidates (Hz)

        maxF0:         max f0 value to consider for
                       candidates (Hz)

       wl:             length of analysis window (in seconds)

        step:          duration between onsets of successive windows
                       (in seconds)
    """
    f = open(fn, 'w');
    f.write('float min_f0 = %f;\n' % min_f0);
    f.write('float max_f0 = %f;\n' % max_f0);
    f.write('float wind_dur = %f;\n' % wl);
    f.write('float frame_step = %f;\n' % step);
    f.close();

def _gen_layout(fn):
    """Write esps layout file for fea_print.

    Inputs:
        fn:    path to layout file
    """
    f = open(fn, 'w');
    f.write('layout=f0 \n');
    f.write('F0[0] %.2f \n');
    f.write('prob_voice[0] %.1f \n');
    f.write('rms[0] %.2f \n');
    f.write('ac_peak[0] %.3f \n');
    f.close();
