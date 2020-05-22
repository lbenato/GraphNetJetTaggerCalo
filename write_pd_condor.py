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

from samplesAOD2017 import *

# define your variables here
'''
var_list = ['EventNumber',
            'RunNumber','LumiNumber','EventWeight','isMC',#not to be trained on!
            'isVBF','HT','MEt.pt','MEt.phi','MEt.sign','MinJetMetDPhi',
            'nCHSJets',
            'nElectrons','nMuons','nPhotons','nTaus','nPFCandidates','nPFCandidatesTrack'
            ]
#jets variables
nj = 10#10
jtype = ['Jet']
jvar = ['pt','eta','phi','mass','nConstituents','nTrackConstituents','nSelectedTracks','nHadEFrac', 'cHadEFrac','ecalE','hcalE',
        'muEFrac','eleEFrac','photonEFrac', 'eleMulti','muMulti','photonMulti','cHadMulti','nHadMulti',
        'nHitsMedian','nPixelHitsMedian', 'dRSVJet', 'nVertexTracks', 'CSV', 'SV_mass',
        #new
        'nRecHits', 'timeRecHits', 'timeRMSRecHits', 'energyRecHits', 'energyErrorRecHits',
        'ptAllTracks', 'ptAllPVTracks', 'ptPVTracksMax', 'nTracksAll', 'nTracksPVMax', 'medianIP2D', 
        #'medianTheta2D',#currently empty
        'alphaMax', 'betaMax', 'gammaMax', 'gammaMaxEM', 'gammaMaxHadronic', 'gammaMaxET', 'minDeltaRAllTracks', 'minDeltaRPVTracks',
        'dzMedian', 'dxyMedian',
]
jvar+=['isGenMatched']
jet_list = []

#pf candidates variables
npf = 100#100##2#1#00
pftype = ['PFCandidate']
pfvar = ['pt','eta','phi','mass','energy','px','py','pz','jetIndex',
         'pdgId','isTrack','hasTrackDetails', 'dxy', 'dz', 'POCA_x', 'POCA_y', 'POCA_z', 'POCA_phi', 
         'ptError', 'etaError', 'phiError', 'dxyError', 'dzError', 'theta', 'thetaError','chi2', 'ndof', 'normalizedChi2', 
         'nHits', 'nPixelHits', 'lostInnerHits'
]

pf_list = []
for n in range(nj):
    for t in jtype:
        for v in jvar:
            jet_list.append(str(t)+"_"+str(n)+"."+v)
        for p in range(npf):
            for tp in pftype:
                for pv in pfvar:
                    pf_list.append(str(t)+"_"+str(n)+"_"+str(tp)+"_"+str(p)+"."+str(pv))


var_list += jet_list
var_list += pf_list
'''

variables = []
MEt = MEtType()
#CHSJets = JetType()

def write_condor_h5(folder,output_folder,file_name,xs,LUMI,cols,test_split,val_split,tree_name="",counter_hist="",sel_cut="",obj_sel_cut="",verbose=True):
    print("    Opening ", folder)
    print("\n")
    # loop over files, called file_name
    oldFile = TFile(folder+file_name, "READ")
    counter = oldFile.Get(counter_hist)#).GetBinContent(1)
    nevents_gen = counter.GetBinContent(1)
    print("  n events gen.: ", nevents_gen)
    oldTree = oldFile.Get(tree_name)
    nevents_tot = oldTree.GetEntries()#?#-1
    #tree_weight = oldTree.GetWeight()
    tree_weight = LUMI * xs / nevents_gen
    print("   Tree weight:   ",tree_weight)

    if verbose:
        print("\n")
        #print("   Initialized df for sample: ", file_name)
        print("   Initialized df for sample: ", file_name)
        print("   Reading n. events in tree: ", nevents_tot)
        #print("\n")

    if nevents_tot<0:
        print("   Empty tree!!! ")
        return

    # First loop: check how many events are passing selections
    count = rnp.root2array(folder+file_name, selection = sel_cut, object_selection = obj_sel_cut, treename=tree_name, branches=cols[0], start=0, stop=nevents_tot)
    nevents=count.shape[0]
    if verbose:
        print("   Cut applied: ", sel_cut)
        print("   Events passing cuts: ", nevents)
        print("\n")            

    #Divide conversion in chuncks
    #chunksize = 100
    #n_chunks = int(ceil(float(nevents_tot) / chunksize))
    #print(n_chunks)
    ## loop over variables

    #avoid loop over variables, read all together
    #we have already zero padded
    startTime = time.time()
    b = rnp.root2array(folder+file_name, selection = sel_cut, object_selection = obj_sel_cut, treename=tree_name, branches=cols, start=0, stop=nevents_tot)
    df = pd.DataFrame(b,columns=cols)

    #Remove dots from column names
    column_names = []
    for a in cols:
        column_names.append(a.replace('.','_'))
    df.columns = column_names

    #add is_signal flag
    df["is_signal"] = np.ones(nevents) if (("n3n2" in folder) or ("H2ToSSTobbbb" in folder)) else np.zeros(nevents)
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

    #split training, test and validation samples
    #first shuffle
    df.sample(frac=1).reset_index(drop=True)

    #define train, test and validation samples
    n_events_train = int(df.shape[0] * (1-test_split-val_split) )#0
    n_events_test = int(df.shape[0] * (test_split))#1
    n_events_val = int(df.shape[0] - n_events_train - n_events_test)#2
    print("Train: ", n_events_train)
    print("Test: ", n_events_test)
    print("Val: ", n_events_val)
    print("Tot: ", n_events_train+n_events_test+n_events_val)
    df_train = df.head(n_events_train)
    df_left  = df.tail(df.shape[0] - n_events_train)
    df_test = df_left.head(n_events_test)
    df_val  = df_left.tail(df_left.shape[0] - n_events_test)

    # Add ttv column
    df_train.loc[:,"ttv"] = np.zeros(df_train.shape[0])
    df_test.loc[:,"ttv"] = np.ones(df_test.shape[0])
    df_val.loc[:,"ttv"] = np.ones(df_val.shape[0]) * 2
    print("  -------------------   ")
    print("  Events for training: ", df_train.shape[0])
    print("  Events for testing: ", df_test.shape[0])
    print("  Events for validation: ", df_val.shape[0])
    
    # Write h5
    if ".root" in file_name:
        file_name = file_name[:-5]
    df_train.to_hdf(output_folder+'/'+file_name+'_train.h5', 'df', format='table' if (len(cols)<=2000) else 'fixed')
    print("  "+output_folder+"/"+file_name+"_train.h5 stored")
    df_test.to_hdf(output_folder+'/'+file_name+'_test.h5', 'df', format='table' if (len(cols)<=2000) else 'fixed')
    print("  "+output_folder+"/"+file_name+"_test.h5 stored")
    df_val.to_hdf(output_folder+'/'+file_name+'_val.h5', 'df', format='table' if (len(cols)<=2000) else 'fixed')
    print("  "+output_folder+"/"+file_name+"_val.h5 stored")
    print("  -------------------   ")



'''
event_list = [
            'EventNumber',
            'RunNumber','LumiNumber','EventWeight','SampleWeight','isMC',
            #'isVBF','HT','MEt_pt','MEt_phi','MEt_sign','MinJetMetDPhi',
            'nCHSJets',
            #'nElectrons','nMuons','nPhotons','nTaus','nPFCandidates','nPFCandidatesTrack',
            'ttv','is_signal',
            ]


##This must be fixed, still depends on variables defined here

def convert_dataset_condor(folder,graphnet_folder,file_name,cols):
    print("  Transform per-event into per-jet dataframes...")
    print("\n")


    ##folder+file_name WILL be folder+s and no loop ! TODO!

    ##Prepare train/test/val sample
    df_pre_train = defaultdict()
    df_pre_test = defaultdict()
    df_pre_val = defaultdict()
    
    
    store_pre_train = pd.HDFStore(folder+file_name+"_train.h5")
    df_pre_train    = store_pre_train.select("df",start=0,stop=-1)#
    store_pre_test  = pd.HDFStore(folder+file_name+"_test.h5")
    df_pre_test     = store_pre_test.select("df",start=0,stop=-1)#
    store_pre_val   = pd.HDFStore(folder+file_name+"_val.h5")
    df_pre_val      = store_pre_val.select("df",start=0,stop=-1)#
        
        
    df_temp_train = defaultdict()
    df_temp_test = defaultdict()
    df_temp_val = defaultdict()
    df_conc_train = defaultdict()
    df_conc_test = defaultdict()
    df_conc_val = defaultdict()
    df_train = defaultdict()
    df_test = defaultdict()
    df_val = defaultdict()
    
    #Transform per-event into per-jet
    for j in range(nj):
        temp_list = []
        for l in cols:#all variables
            #for l in cols:#only the one we want to train?
            if ("Jet_"+str(j)) in l:
                #print(l)
                temp_list.append(l.replace('.','_'))
                 
            #print("Here doing per jet")
            #print(temp_list)
            
            ##Temp train
        df_temp_train = df_pre_train[temp_list+event_list]
        df_temp_train["Jet_index"] = np.ones(df_temp_train.shape[0])*j
        ##Temp test
        df_temp_test = df_pre_test[temp_list+event_list]
        df_temp_test["Jet_index"] = np.ones(df_temp_test.shape[0])*j
        ##Temp val
        df_temp_val = df_pre_val[temp_list+event_list]
        df_temp_val["Jet_index"] = np.ones(df_temp_val.shape[0])*j
        #print("\n")
        #print("Before renaming")
        #print(df_temp_val_s)
            

        #Rename columns
        for i, v in enumerate(train_features):
            for p in range(npf):
                df_temp_train.rename(columns={"Jet_"+str(j)+"_PFCandidate_"+str(p)+"_"+train_features[i]: train_features[i]+"_"+str(p)},inplace=True)
                df_temp_test.rename(columns={"Jet_"+str(j)+"_PFCandidate_"+str(p)+"_"+train_features[i]: train_features[i]+"_"+str(p)},inplace=True)
                df_temp_val.rename(columns={"Jet_"+str(j)+"_PFCandidate_"+str(p)+"_"+train_features[i]: train_features[i]+"_"+str(p)},inplace=True)
                    
        for v in jvar: 
            #print("Jet_"+str(j)+"_"+str(v))      
            df_temp_train.rename(columns={"Jet_"+str(j)+"_"+str(v): "Jet_"+str(v)},inplace=True)
            df_temp_test.rename( columns={"Jet_"+str(j)+"_"+str(v): "Jet_"+str(v)},inplace=True)
            df_temp_val.rename(  columns={"Jet_"+str(j)+"_"+str(v): "Jet_"+str(v)},inplace=True)
            
        #print(df_temp_val_s[s])
        #exit()

        #Concatenate jets
        if j==0:
            df_conc_train = df_temp_train
            df_conc_test = df_temp_test
            df_conc_val = df_temp_val
        else:
            df_conc_train = pd.concat([df_conc_train,df_temp_train])
            df_conc_test = pd.concat([df_conc_test,df_temp_test])
            df_conc_val = pd.concat([df_conc_val,df_temp_val])

    ##df_train_s[s] = df_conc_train_s[s][ df_conc_train_s[s]["Jet_isGenMatched"]==1 ]
    ##df_test_s[s] = df_conc_test_s[s][ df_conc_test_s[s]["Jet_isGenMatched"]==1 ]
    ##no selections at the moment
    df_train = df_conc_train
    df_test = df_conc_test
    df_val = df_conc_val
    print(s, df_train)
    ##write h5
    df_train.to_hdf(graphnet_folder+'/'+file_name+'_train.h5', 'df', format='table' if (len(cols)<=2000) else 'fixed')
    print("  "+graphnet_folder+"/"+file_name+"_train.h5 stored")
    df_test.to_hdf(graphnet_folder+'/'+file_name+'_test.h5', 'df', format='table' if (len(cols)<=2000) else 'fixed')
    print("  "+graphnet_folder+"/"+file_name+"_test.h5 stored")
    df_val.to_hdf(graphnet_folder+'/'+file_name+'_val.h5', 'df', format='table' if (len(cols)<=2000) else 'fixed')
    print("  "+graphnet_folder+"/"+file_name+"_val.h5 stored")
    print("  -------------------   ")
'''
