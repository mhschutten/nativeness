from os import listdir
from os.path import isfile, join
from sys import argv

from scipy.io import wavfile
import numpy as np
import math
import struct 

import praatTextGrid
from sklearn import mixture
from ldc_wavlib.features import mfcc
from sklearn import svm

class MFCCExtractor:

    def run(self):
        allInputs = []
        for speaker in self.__allSpeakers:
            gmm = self.computeSpeakerProperties(self.__dataLocation + speaker)
            speakerInput = np.concatenate((gmm.means_, gmm.covars_))
            allInputs.append(speakerInput)

        datasetSize = len(self.__nativelikenessScores)
        boundary = int(0.9 * datasetSize)
        
        regressionSolver = svm.SVR()
        regressionSolver.fit(allInputs[0:boundary], self.__nativelikenessScores[0:boundary])
        
        result = 0.0
        for idx in range(boundary, datasetSize):
            output = regressionSolver.predict(allInputs[idx])
            
            result += (output - self.__nativelikenessScores[idx]) * (output - self.__nativelikenessScores[idx])

        result /= (datasetSize - boundary)

        print result



    #Extract the gmm info from the wav file for a given speaker
    def computeSpeakerProperties(self, location):
        #currently just for a single file at location
        sr, wavData = self.readWav(location + ".wav") 

        words = self.extractWords(wavData, location + ".TextGrid", sr)
        mfccs = self.getMFCCs(words, sr)
        gmm = self.getMixtureModel(mfccs)

        return gmm
        
    #extract words from a given wav file w.r.t. provided TextGrid file
    def extractWords(self, wavData, textGridFile, sr):

        #read the TextGrid file
        textGrid = praatTextGrid.PraatTextGrid(0, 0)
        arrTiers = textGrid.readFromFile(textGridFile)
        
        #Extract the alignments
        alignments = arrTiers[1]

        words = []
        for i in range (alignments.getSize()):
            word = alignments.get(i)

            #exclude non-words
            if self.__isNonWord(word[2]):
                continue
            
            #get the boundaries of the word
            startIdx = int(word[0] * sr)
            endIdx = int(word[1] * sr)

            #add the word to thelist of words
            words.append(wavData[startIdx : endIdx])
    
        return words
    

    #read the specified wav file
    def readWav(self, location):
        return wavfile.read(location)

        
    #getMFCCs of the current set of words with a given sampling rate
    def getMFCCs(self, words, sr):
        mfccs = []
        for word in words:
            mfccs.append(mfcc.get_mfcc(word, sr))
    
        return mfccs
        
        
    #create the GMM for a given set of MFCCs    
    def getMixtureModel(self, mfccs):
        gmm = mixture.GMM(n_components = 13)
        
        for mfcc in mfccs:
            tmpmfcc = np.reshape(mfcc[0], len(mfcc[0]) * 13)
            gmm.fit(tmpmfcc)
            
            gmm.means_ = np.reshape(gmm.means_, len(gmm.means_))
            gmm.covars_ = np.reshape(gmm.covars_, len(gmm.covars_))
        
        return gmm


    #check whether something is an actual word
    def __isNonWord(self, word):
        if word == 'sp' or word == 'sil':
            return True
        
        return False

    def __parseDatafile(self, datafile):
        f = open(datafile, 'r')
        
        #skip the first line
        f.readline() 

        speakers = []
        nativelikenessScores = []
        for line in f:
            columns = line.split('\t')
            if (columns[1] != "NA"):
                speakers.append(columns[0])
                nativelikenessScores.append(float(columns[1]))

        print "Number of speakers: ", len(speakers)

        return speakers, nativelikenessScores

        
    #initialization
    def __init__(self, datafile, dataLocation):
        self.__allSpeakers, self.__nativelikenessScores = self.__parseDatafile(datafile)
        if (dataLocation[-1] != '/'):
            dataLocation += '/'
        self.__dataLocation = dataLocation
        

        
if __name__ == "__main__":
    dataFileLocation = "../speakers.txt"
    wavDataFolder = "../aligned_data/"

    extractor = MFCCExtractor(dataFileLocation, wavDataFolder)
    extractor.run()

