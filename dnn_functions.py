import numpy as np
# Needed libraries
import matplotlib.pyplot as plt
import ROOT
import root_numpy as rnp
import pandas as pd
import tables
from samplesAOD2018 import *
from ROOT import gROOT, TFile, TTree, TObject, TH1, TH1F, AddressOf, TLorentzVector

###
## Thanks to Dirk
###

def plotLearningCurves(*histObjs):
    """This function processes all histories given in the tuple.
    Left losses, right accuracies
    """
    # too many plots
    if len(histObjs)>10: 
        print('Too many objects!')
        return
    # missing names
    for histObj in histObjs:
        if not hasattr(histObj, 'name'): histObj.name='?'
    names=[]
    # loss plot
    plt.figure(figsize=(12,6))
    plt.rcParams.update({'font.size': 15}) #Larger font size
    plt.subplot(1,2,1)
    # loop through arguments
    for histObj in histObjs:
        plt.plot(histObj.history['loss'])
        names.append('train '+histObj.name)
        plt.plot(histObj.history['val_loss'])
        names.append('validation '+histObj.name)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(names, loc='upper right')
    

    #accuracy plot
    plt.subplot(1,2,2)
    for histObj in histObjs:
        plt.plot(histObj.history['accuracy'])
        plt.plot(histObj.history['val_accuracy'])
    plt.title('model accuracy')
    #plt.ylim(0.5,1)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(names, loc='upper left')
    
    #plt.show()

    # min, max for loss and acc
    for histObj in histObjs:
        h=histObj.history
        maxIdxTrain = np.argmax(h['accuracy'])
        maxIdxTest  = np.argmax(h['val_accuracy'])
        minIdxTrain = np.argmin(h['loss'])
        minIdxTest  = np.argmin(h['val_loss'])
        
        strg='\tTrain: Min loss {:6.10f} at {:3d} --- Max accuracy {:6.10f} at {:3d} | '+histObj.name
        print(strg.format(h['loss'][minIdxTrain],minIdxTrain,h['accuracy'][maxIdxTrain],maxIdxTrain))
        strg='\tValidation : Min loss {:6.10f} at {:3d} --- Max accuracy {:6.10f} at {:3d} | '+histObj.name
        print(strg.format(h['val_loss'][minIdxTest],minIdxTest,h['val_accuracy'][maxIdxTest],maxIdxTest))
        print(len(strg)*'-')

    return plt
    
def plotLearningCurvesBDT(histObj, name):
    """This function processes all histories given in the tuple.
    Left losses, right accuracies
    """
    # too many plots
    if len(histObj.keys())>10: 
        print('Too many objects!')
        return
    # missing names
    if name == "": name='?'
    names=[]
    # loss plot
    plt.figure(figsize=(12,6))
    plt.rcParams.update({'font.size': 15}) #Larger font size
    plt.subplot(1,2,1)
    print(histObj.keys())
    plt.plot(histObj['validation_0']['logloss'])
    names.append('train '+name)
    plt.plot(histObj['validation_1']['logloss'])
    names.append('validation '+name)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(names, loc='upper right')

    #accuracy plot
    plt.subplot(1,2,2)
    plt.plot(np.ones(len(histObj['validation_0']['error'])) - np.array(histObj['validation_0']['error']))
    plt.plot(np.ones(len(histObj['validation_1']['error'])) - np.array(histObj['validation_1']['error']))
    plt.title('model accuracy')
    #plt.ylim(0.5,1)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(names, loc='upper left')
        
    # loop through arguments
    '''
    #plt.show()

    # min, max for loss and acc
    for histObj in histObjs:
        h=histObj.history
        maxIdxTrain = np.argmax(h['accuracy'])
        maxIdxTest  = np.argmax(h['val_accuracy'])
        minIdxTrain = np.argmin(h['loss'])
        minIdxTest  = np.argmin(h['val_loss'])
        
        strg='\tTrain: Min loss {:6.10f} at {:3d} --- Max accuracy {:6.10f} at {:3d} | '+histObj.name
        print(strg.format(h['loss'][minIdxTrain],minIdxTrain,h['accuracy'][maxIdxTrain],maxIdxTrain))
        strg='\tValidation : Min loss {:6.10f} at {:3d} --- Max accuracy {:6.10f} at {:3d} | '+histObj.name
        print(strg.format(h['val_loss'][minIdxTest],minIdxTest,h['val_accuracy'][maxIdxTest],maxIdxTest))
        print(len(strg)*'-')
    '''
    return plt
