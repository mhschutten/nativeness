from ldc_wavlib.features import mfcc
from scipy.io import wavfile
from sys import argv
import numpy as np
import struct 

class MFCCExtractor:
    
    #read the specified wav file
    def readWav(self, location):
        [self.sr, self.wavdata] = wavfile.read(location)
        
    #getMFCCs of current wav file
    def getMFCCs(self):
        mfccs = mfcc.get_mfcc(self.wavdata, self.sr)

        return mfccs 
        
    #initialization
    def __init__(self):
        self.curWav = 0

if __name__ == "__main__":
    extractor = MFCCExtractor()
    extractor.readWav(argv[1]) 
    print extractor.getMFCCs()
