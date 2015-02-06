__all__ = ['wav_2_array', 
           'read_htk_label_file', 'write_htk_label_file'];

from struct import pack, unpack;
import sys;
import wave;

import numpy as np;
from numpy import tile;
from scipy.io import wavfile;
import scipy.stats as stat;


# wav io
def wav_2_array(wf, scale=False):
    """
    Inputs:
        wf:    path to single-channel wav file 

        scale: If True, recales signal to interval [-1, 1].
               (default: False)

    Outputs:
        sr:    sampling frequency (Hz)

        x:     Nx1 numpy array of samples
    """
    try:
        (sr, x) = wavfile.read(wf);
        x = x.astype('float32');
    except IOError:
        print('Error opening file: %s.' % wf);
    if scale:
        alpha = scale / stat.scoreatpercentile(np.abs(x), 99.);
        x *= alpha;

    return (sr, x);


def get_wav_dur(wf):
    """
    Inputs:
        wf:    path to single-channel wav file
    """
    f =  wave.open(wf, 'r');
    dur = f.getnframes() / float(f.getframerate());
    f.close();
    return dur;



# HTK label file io
def read_htk_label_file(lf, in_sec=True, label_sep=None):
    """
    Inputs:
        lf:        path to htk label file

        in_sec:    boolean indicating whether times in label
                   file are in seconds or htk 100 ns units

        label_sep: ...

    Outputs:
        segs:      list of tuples of the form (onset, offset, label),
                   where onset and offset are in seconds
    """
    segs = [];
    with open(lf, 'r') as f:
        for line in f:
            onset, offset, label = line.strip().split()[:3];
        
            # convert from htk units if needed
            if in_sec:
                onset = float(onset);
                offset = float(offset);
            else:
                onset = htk_units_2_seconds(int(onset));
                offset = htk_units_2_seconds(int(offset));

            # replace label_sep with whitespace
            if label_sep:
                label = label.replace(label_sep, ' ');

            segs.append([onset, offset, label]);

    return segs;


def write_htk_label_file(lf, segs, in_sec=True):
    """
    Inputs:
        lf:        path to htk label file

        segs:      list of tuples of the form (onset, offset, label),
                   where onset and offset are in seconds

        in_sec:    boolean indicating whether times in label
                   file are in seconds or htk 100 ns units
    """
    with open(lf, 'w') as f:
        for onset, offset, label in segs:
            if in_sec:
                f.write('%f %f %s\n' % (onset, offset, label));
            else:
                onset = seconds_2_htk_units(onset);
                offset = seconds_2_htk_units(offset);
                f.write('%d %d %s\n' % (onset, offset, label));


def htk_units_2_seconds(t):
    """Convert from 100ns units to seconds.
    """
    return t*10.**-7;


def seconds_2_htk_units(t):
    """Convert from seconds to 100ns units, rounded down to nearest integer.
    """
    return int(t*10**7);


# features
def _fromfile(fid, dtype='float', count=-1):
    """
    """
    X = np.fromfile(fid, dtype, count);
    if sys.byteorder == 'little':
        X.byteswap(True);
    return X;


_INT = set(['int8', 'int16', 'int32', 'int64', 'int128',
            'uint8', 'uint16', 'uint32', 'uint64', 'uint128']);

_FLOAT = set(['float16', 'float32', 'float64', 'float96', 'float128'
              'float256']);

def _tofile(X, fid, dtype='float'):
    """
    """
    # determine appropriate format string
    if dtype in _INT:
        fmt = '%d';
    elif dtype in _FLOAT:
        fmt = '%f';
    else:
        print('Error: unsupported dtype: %d' % dtype);
        sys.exit(1);
    Xtmp = X.astype(dtype);
    if sys.byteorder == 'little':
        Xtmp.byteswap(True);
    Xtmp.tofile(fid, format=fmt);


def read_htk(fp):
    """
    """
    f = open(fp, 'r');

    # read header
    (numSamples,
     samplePeriod,  # = step
     sampleSize,
     paramKind
    ) = unpack('>iihh', f.read(12));
    paramKindStr = _param_kind_2_str(paramKind);
    hasCheckSum = '_K' in paramKindStr;
    isCompressed = '_C' in paramKindStr;

    # read samples
    if isCompressed:
        # For details on htk compression methods, consult
        # chap 5 of the HTK manual
        numSamples -= 4;
        numFeatures = sampleSize // 2;
        A = tile(_fromfile(f, 'float32', numFeatures), (numSamples, 1));
        B = tile(_fromfile(f, 'float32', numFeatures), (numSamples, 1));
        S = _fromfile(f, 'uint16', numFeatures*numSamples);
        S = S.reshape([numSamples, numFeatures]);
        F = (S + B) / A;
    else:
        numFeatures = sampleSize // 4;
        F = _fromfile(f, 'float32').reshape([numSamples, numFeatures]);
    f.close();

    return F;


def save_htk(X, fp, step):
    """
    """
    # header
    numSamples, numFeatures = X.shape;
    samplePeriod = seconds_2_htk_units(step);
    sampleSize = 4*numFeatures;  # 4-byte floats
    paramKind = 9;               # code for other

    # write to file
    f = open(fp, 'wb');
    header = pack('>iihh', 
                  numSamples,
                  samplePeriod,
                  sampleSize,
                  paramKind
                 );
    f.write(header);
    _tofile(X, f, 'float32');
    f.close();

_BASEMASK = 077;
_BK2TXT = {  0 : 'WAVEFORM',
             1 : 'LPC',
             2 : 'LPREFC',
             3 : 'LPCEPSTRA',
             4 : 'LPDELCEP',
             5 : 'IREFC',
             6 : 'MFCC',
             7 : 'FBANK',
             8 : 'MELSPEC',
             9 : 'USER',
            10 : 'DISCRETE',
            11 : 'PLP',
          };
_AK2TXT = {  000100 : '_E',
             000200 : '_N',
             000400 : '_D',
             001000 : '_A',
             002000 : '_C',
             004000 : '_Z',
             010000 : '_K',
             020000 : '_0',
             040000 : '_V',
             0100000 : '_T',
          };
_AKS = [000100, 000400, 000200, 001000, 0100000, 002000, 010000, 004000, 020000, 040000];

def _param_kind_2_str(paramKind):
    baseKind = _BK2TXT[paramKind & _BASEMASK];
    auxKinds = ''.join([_AK2TXT[MASK] for MASK in _AKS if (paramKind & MASK)]);

    return baseKind + auxKinds;
