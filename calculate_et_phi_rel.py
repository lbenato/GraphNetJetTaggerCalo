import pandas as pd
import numpy as np
import math
import awkward
import pickle
import uproot_methods
import os.path
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from root_numpy import array2tree, array2root
from dnn_functions import *
from samplesAOD2018 import *
from tf_keras_model import *

#My test folder. Fix with yours and create a new one with the new datasets
IN  = '/nfs/dust/cms/group/cms-llp/dataframes_lisa/phi_rel_test/'
OUT = '/nfs/dust/cms/group/cms-llp/dataframes_lisa/phi_rel_test/'
#Max n. of pf candidates
npf = 30

if not(os.path.exists(OUT)):
    os.mkdir(OUT)

#This is done only for validation; must be done also for training/validation
store = pd.HDFStore(IN+"val.h5")
#Note: stop=100 for testing purposes. Must be set to -1 (all events)
df = store.select("df",start=0,stop=100)

## Implementation of delta_phi and delta_eta functions from:
## https://github.com/scikit-hep/uproot-methods/blob/master/uproot_methods/classes/TLorentzVector.py#L82-L83

def delta_phi(phi1, phi2):
    return (phi1 - phi2 + math.pi) % (2*math.pi) - math.pi

def delta_eta(eta1, eta2):
    delta = eta1 - eta2
    jet_sign  = np.sign(eta2) 
    jet_sign[jet_sign==0] = 1
    return delta * jet_sign

for i in range(npf):
    #mask: keep memory of PF candidates that are empty
    mask = df['pt_'+str(i)].values<=-1.
    df['etarel_'+str(i)] = delta_eta(  df['Jet_eta'].values, df['eta_'+str(i)].values  )
    df['phirel_'+str(i)] = delta_phi(  df['Jet_phi'].values, df['phi_'+str(i)].values  )
    df['ptrel_'+str(i)]  = df['Jet_pt'].values/df['pt_'+str(i)].values
    df['logptrel_'+str(i)]  = np.log(df['ptrel_'+str(i)].values)

    #Assign zero-padded values to invalid (empty) PF candidates
    #Note: zero-padding taken as per original variables (eta, phi, pt)
    #for log pt: set to -999.
    df['etarel_'+str(i)]    = df['etarel_'+str(i)].mask(mask, -9.)
    df['phirel_'+str(i)]    = df['phirel_'+str(i)].mask(mask, -9.)
    df['ptrel_'+str(i)]     = df['ptrel_'+str(i)].mask(mask, -1.)
    df['logptrel_'+str(i)]  = df['logptrel_'+str(i)].mask(mask, -999.)


#only for testing purposes: pf candidate n.19
print(df['pt_'+str(19)])
print(df['ptrel_'+str(19)])
print(df['logptrel_'+str(19)])
print(df)

# !!
#here missing: save this new df into a new file in the OUT folder
