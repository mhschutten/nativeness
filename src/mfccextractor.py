from ldc_wavlib.features import mfcc
from scipy.io import wavfile
from sys import argv
import numpy as np
import struct 
import praatTextGrid
import math

class MFCCExtractor:
    
    #read the specified wav file
    def readWav(self, location):
        self.sr, self.wavData = wavfile.read(location)

        print self.wavData
        wavfile.write('tmp/original2.wav',self.sr,self.wavData)

        
    #getMFCCs of current wav file
    def getMFCCs(self):
        mfccs = mfcc.get_mfcc(self.wavData, self.sr)

        return mfccs 
        
    def getWordAlignment(self, location):
        textGrid = praatTextGrid.PraatTextGrid(0, 0)
        arrTiers = textGrid.readFromFile(location)
        
        self.alignments = arrTiers[1]
        return arrTiers[1]
        


    def extractWords(self):
        self.words = []

        for i in range (self.alignments.getSize()):
            word = self.alignments.get(i)

            if self.__isNonWord(word[2]):
                continue

            startIdx = int(word[0] * self.sr)
            endIdx = int(word[1] * self.sr)
    
            self.words.append(self.wavData[startIdx : endIdx])
                    
    #check whether something is an actual word
    def __isNonWord(self, word):
        if word == 'sp' or word == 'sil':
            return True
        
        return False

    #initialization
    def __init__(self):
        self.sr = 0 #sampling rate
        self.wavData = [] #current wav file
        self.alignments = [] #alignments per word for the current wav file
        self.words = [] #list of sound signals for all words

        #NOT IN USE YET
        self.location = "" #location of datafolder

if __name__ == "__main__":
    extractor = MFCCExtractor()
    tier = extractor.getWordAlignment(argv[1] + ".TextGrid")
    extractor.readWav(argv[1] + ".wav") 

    extractor.extractWords()

    filename = "tmp/original1.wav"
    wavfile.write(filename, extractor.sr, extractor.wavData)

    for i in range(len(extractor.words)):
        filename = "tmp/speaker1word" + str(i)
        print extractor.words[i]
        print extractor.sr, len(extractor.words[i])
        
        wavfile.write(filename, extractor.sr, np.array(extractor.words[i], dtype = float))

    """
    print tier  
    for i in range(tier.getSize()):
#        if tier.getLabel(i) == 'sounding':
        interval = tier.get(i)
        print "\t", interval

    print extractor.getMFCCs()
    """    
    
    tier = extractor.getWordAlignment(argv[2] + ".TextGrid")
    extractor.readWav(argv[2] + ".wav") 

    extractor.extractWords()

    filename = "tmp/original2.wav"
    wavfile.write(filename, extractor.sr, extractor.wavData)

    for i in range(len(extractor.words)):
        filename = "tmp/speaker2word" + str(i)
        wavfile.write(filename, extractor.sr, extractor.words[i])

    """
    print tier  
    for i in range(tier.getSize()):
#        if tier.getLabel(i) == 'sounding':
        interval = tier.get(i)
        print "\t", interval

    print extractor.getMFCCs()
    """
