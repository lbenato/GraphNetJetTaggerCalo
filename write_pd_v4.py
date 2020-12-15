import ROOT
import os
import root_numpy as rnp
import numpy as np
import pandas as pd
import tables
import uproot
import time
import multiprocessing
from samplesAOD2017 import *
from math import ceil
from ROOT import gROOT, TFile, TTree, TObject, TH1, TH1F, AddressOf, TLorentzVector
from dnn_functions import *
gROOT.ProcessLine('.L Objects.h' )
from ROOT import JetType, CaloJetType, MEtType, CandidateType, DT4DSegmentType, CSCSegmentType, PFCandidateType#, TrackType
from collections import defaultdict

from samplesAOD2018 import *


variables = []
MEt = MEtType()
#CHSJets = JetType()


### NOTE:
### Includes important bug fix for c_nEvents!

def write_h5_v4(folder,output_folder,file_name,xs,LUMI,counter,cols,tree_name="",counter_hist="",sel_cut="",obj_sel_cut="",verbose=True):
    print("    Opening ", folder)
    print("\n")
    if verbose:
        print("\n")
        #print("   Initialized df for sample: ", file_name)
        print("   Initialized df for sample: ", file_name)
    #print(cols)
    
    
    # loop over files, called file_name
    oldFile = TFile(folder+file_name, "READ")
    if(oldFile.GetListOfKeys().Contains(counter_hist) == False):
        return
    #counter = oldFile.Get(counter_hist)#).GetBinContent(1)
    #nevents_gen = counter.GetBinContent(1)
    nevents_gen = counter
    print("  n events gen.: ", nevents_gen)
    if(nevents_gen==0):
        return
        print("   empty root file! ")
    oldTree = oldFile.Get(tree_name)
    nevents_tot = oldTree.GetEntries()#?#-1
    #tree_weight = oldTree.GetWeight()
    tree_weight = LUMI * xs / nevents_gen
    print("   Tree weight:   ",tree_weight)

    if verbose:
        print("   Reading n. events in tree: ", nevents_tot)
        #print("\n")

    if nevents_tot<=0:
        print("   Empty tree!!! ")
        return

    # First loop: check how many events are passing selections
    count = rnp.root2array(folder+file_name, selection = sel_cut, object_selection = obj_sel_cut, treename=tree_name, branches=["EventNumber"], start=0, stop=nevents_tot)
    nevents=count.shape[0]
    if verbose:
        print("   Cut applied: ", sel_cut)
        print("   Events passing cuts: ", nevents)
        print("\n")            

    #avoid loop over variables, read all together
    #we have already zero padded
    startTime = time.time()
    b = rnp.root2array(folder+file_name, selection = sel_cut, object_selection = obj_sel_cut, treename=tree_name, branches=cols, start=0, stop=nevents_tot)
    df = pd.DataFrame(b)#,columns=cols)

    #Remove dots from column names
    column_names = []
    for a in cols:
        if isinstance(a, tuple):
            column_names.append(a[0].replace('.','_').replace('s[','_').replace(']',''))
        else: column_names.append(a.replace('.','_'))
    df.columns = column_names
    print(df)

    #add is_signal flag
    df["is_signal"] = np.ones(nevents) if (("n3n2" in folder) or ("H2ToSSTobbbb" in folder) or ("TChiHH" in folder)) else np.zeros(nevents)
    df["c_nEvents"] = np.ones(nevents) * nevents_gen
    df["EventWeight"] = df["EventWeight"]*tree_weight
    df["SampleWeight"] = np.ones(nevents) * tree_weight
    #print(df)
    print("\n")
    print("  * * * * * * * * * * * * * * * * * * * * * * *")
    print("  Time needed root2array: %.2f seconds" % (time.time() - startTime))
    print("  * * * * * * * * * * * * * * * * * * * * * * *")
    print("\n")

    #df.rename(columns={"nJets" : "nCHSJets"},inplace=True)
    if verbose:
        print(df)

    #shuffle
    df.sample(frac=1).reset_index(drop=True)

    print("  -------------------   ")
    print("  Events : ", df.shape[0])
    
    # Write h5
    if ".root" in file_name:
        file_name = file_name[:-5]
    df.to_hdf(output_folder+'/'+file_name+'.h5', 'df', format='table' if (len(cols)<=2000) else 'fixed')
    print("  "+output_folder+"/"+file_name+".h5 stored")
    print("  -------------------   ")