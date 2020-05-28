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

def convert_dataset_condor(folder,graphnet_folder,file_name,nj,npf,event_list,cols):
    print("  Transform per-event into per-jet dataframes...")
    print("\n")

    startTime = time.time()
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

    #print(cols)
    for j in range(nj):
        temp_list = []
        #print("Jet n. ",j)
        for l in cols:
            if ("Jet_"+str(j)) in l:
                #print(l)
                temp_list.append(l.replace('.','_'))
        if "Jet_"+str(j)+"_isGenMatched" not in temp_list:
            temp_list.append("Jet_"+str(j)+"_isGenMatched")
        if "Jet_"+str(j)+"_pt" not in temp_list:
            temp_list.append("Jet_"+str(j)+"_pt")
        if "Jet_"+str(j)+"_eta" not in temp_list:
            temp_list.append("Jet_"+str(j)+"_eta")
        if "Jet_"+str(j)+"_timeRecHits" not in temp_list:
            temp_list.append("Jet_"+str(j)+"_timeRecHits")

        if "Jet_"+str(j)+"_isGenMatchedCaloCorrLLPAccept" not in temp_list:
            temp_list.append("Jet_"+str(j)+"_isGenMatchedCaloCorrLLPAccept")
        #print(temp_list)



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
        #print(df_temp_val)

        #Rename columns
        for i, v in enumerate(temp_list):
            if("PFCandidate" in v):
                for p in range(npf):
                    feat = v.replace("Jet_"+str(j)+"_PFCandidate_"+str(p)+"_","")
                    df_temp_train.rename(columns={"Jet_"+str(j)+"_PFCandidate_"+str(p)+"_"+feat: feat+"_"+str(p)},inplace=True)
                    df_temp_test.rename(columns={"Jet_"+str(j)+"_PFCandidate_"+str(p)+"_"+feat: feat+"_"+str(p)},inplace=True)
                    df_temp_val.rename(columns={"Jet_"+str(j)+"_PFCandidate_"+str(p)+"_"+feat: feat+"_"+str(p)},inplace=True)
            else:
                feat = v.replace("Jet_"+str(j)+"_","")
                df_temp_train.rename(columns={str(v): "Jet_"+feat},inplace=True)
                df_temp_test.rename( columns={str(v): "Jet_"+feat},inplace=True)
                df_temp_val.rename(  columns={str(v): "Jet_"+feat},inplace=True)
                    
        #print("\n")
        #print("After renaming")            
        #print(df_temp_val["Jet_isGenMatched"])

        #Concatenate jets
        if j==0:
            df_conc_train = df_temp_train
            df_conc_test = df_temp_test
            df_conc_val = df_temp_val
        else:
            df_conc_train = pd.concat([df_conc_train,df_temp_train])
            df_conc_test = pd.concat([df_conc_test,df_temp_test])
            df_conc_val = pd.concat([df_conc_val,df_temp_val])

    #Remove empty jets from train and val
    df_train = df_conc_train[ df_conc_train["Jet_pt"]>0 ]
    df_test = df_conc_test
    df_val = df_conc_val[ df_conc_val["Jet_pt"]>0 ]
    #print(df_train[["Jet_isGenMatched","Jet_pt"]])
    
    print("\n")
    print("  * * * * * * * * * * * * * * * * * * * * * * *")
    print("  Time needed to convert: %.2f seconds" % (time.time() - startTime))
    print("  * * * * * * * * * * * * * * * * * * * * * * *")
    print("\n")
    ##write h5
    #print(graphnet_folder+'/'+file_name)
    
    df_train.to_hdf(graphnet_folder+'/'+file_name+'_train.h5', 'df', format='fixed')
    print("  "+graphnet_folder+"/"+file_name+"_train.h5 stored")
    df_test.to_hdf(graphnet_folder+'/'+file_name+'_test.h5', 'df', format='fixed')
    print("  "+graphnet_folder+"/"+file_name+"_test.h5 stored")
    df_val.to_hdf(graphnet_folder+'/'+file_name+'_val.h5', 'df', format='fixed')
    print("  "+graphnet_folder+"/"+file_name+"_val.h5 stored")
    
    #print(df_train)
    #print("  DONEEEEE")
    print("  -------------------   ")










'''
def convert_dataset_condor_per_jet(folder,graphnet_folder,file_name):
    print("  Transform per-event into per-jet dataframes...")
    print("\n")

    startTime = time.time()
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
        for l in var_list:#all variables
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
    print(df_train)
    
    print("\n")
    print("  * * * * * * * * * * * * * * * * * * * * * * *")
    print("  Time needed to convert: %.2f seconds" % (time.time() - startTime))
    print("  * * * * * * * * * * * * * * * * * * * * * * *")
    print("\n")
    ##write h5
    df_train.to_hdf(graphnet_folder+'/'+file_name+'_train.h5', 'df', format='table' if (len(var_list)<=2000) else 'fixed')
    print("  "+graphnet_folder+"/"+file_name+"_train.h5 stored")
    df_test.to_hdf(graphnet_folder+'/'+file_name+'_test.h5', 'df', format='table' if (len(var_list)<=2000) else 'fixed')
    print("  "+graphnet_folder+"/"+file_name+"_test.h5 stored")
    df_val.to_hdf(graphnet_folder+'/'+file_name+'_val.h5', 'df', format='table' if (len(var_list)<=2000) else 'fixed')
    print("  "+graphnet_folder+"/"+file_name+"_val.h5 stored")
    print("  -------------------   ")
'''
