import ROOT
import os
import root_numpy as rnp
import numpy as np
import pandas as pd
import tables
import uproot
import time
import multiprocessing
from samplesAOD2018 import *
from math import ceil
from ROOT import gROOT, TFile, TTree, TObject, TH1, TH1F, AddressOf, TLorentzVector
from dnn_functions import *
gROOT.ProcessLine('.L Objects.h' )
from ROOT import JetType, CaloJetType, MEtType, CandidateType, DT4DSegmentType, CSCSegmentType, PFCandidateType#, TrackType
from collections import defaultdict


from samplesAOD2018 import *

def convert_dataset_v4(folder,graphnet_folder,file_name,jet_type,nj,nfj,npf,cols,selections=''):
    print("  Transform per-event into per-jet dataframes...")
    print("\n")

    startTime = time.time()
    ##folder+file_name WILL be folder+s and no loop ! TODO!

    ##Prepare columns
    event_list = []
    pf_list = []
    jet_list = []
    fat_jet_list = []


    for i,c in enumerate(cols):
        if isinstance(c, tuple):
            if "_PFCandidate_" in c[0]:
                pf_list.append(c[0].replace('.','_').replace('s[','_').replace(']',''))
            elif "FatJet" in c[0]:
                fat_jet_list.append(c[0].replace('.','_').replace('s[','_').replace(']',''))
            else:
                jet_list.append(c[0].replace('.','_').replace('s[','_').replace(']',''))

        else:
            event_list.append(c.replace('.','_'))

    ##Prepare sample
    df_pre = defaultdict()
    print("going to open: ", folder+file_name+".h5")
    store_pre = pd.HDFStore(folder+file_name+".h5")
    if store_pre.keys()==[]:
        return
    df_pre    = store_pre.select("df",start=0,stop=-1)#
        
    df_temp = defaultdict()
    df_conc = defaultdict()
    df = defaultdict()

    nloop = 0

    if jet_type=="AK4jets": nloop=nj
    elif jet_type=="AK8jets": nloop=nfj  

    for j in range(nloop):
        temp_list = []
        loop_list = []
        jet_string = ""
        if jet_type=="AK4jets":
            loop_list = jet_list+pf_list
            jet_string = "Jet_"
        elif jet_type=="AK8jets":
            loop_list = fat_jet_list
            jet_string = "FatJet_"
            
        #Here loop, working for both
        for l in (loop_list):
            if (jet_string+str(j)) in l:
                #print(l)
                temp_list.append(l.replace('.','_'))
        if jet_string+str(j)+"_isGenMatched" not in temp_list:
            temp_list.append(jet_string+str(j)+"_isGenMatched")
        if jet_string+str(j)+"_pt" not in temp_list:
            temp_list.append(jet_string+str(j)+"_pt")
        if jet_string+str(j)+"_eta" not in temp_list:
            temp_list.append(jet_string+str(j)+"_eta")
        if jet_string+str(j)+"_phi" not in temp_list:
            temp_list.append(jet_string+str(j)+"_phi")
        if jet_string+str(j)+"_timeRecHitsEB" not in temp_list:
            temp_list.append(jet_string+str(j)+"_timeRecHitsEB")
        if jet_string+str(j)+"_isGenMatchedCaloCorr" not in temp_list:
            temp_list.append(jet_string+str(j)+"_isGenMatchedCaloCorr")
        if jet_string+str(j)+"_zLLP" not in temp_list:
            temp_list.append(jet_string+str(j)+"_zLLP")
        if jet_string+str(j)+"_radiusLLP" not in temp_list:
            temp_list.append(jet_string+str(j)+"_radiusLLP")
            
        #Variable currently existing only for AK4
        if jet_type=="AK4jets":     
            if jet_string+str(j)+"_isGenMatchedCaloCorrLLPAccept" not in temp_list:
                temp_list.append(jet_string+str(j)+"_isGenMatchedCaloCorrLLPAccept")
        #print("here temp list ", temp_list)

        df_temp = df_pre[temp_list+event_list]
        df_temp[jet_string+"index"] = np.ones(df_temp.shape[0])*j

        #print("\n")
        #print("Before renaming")
        #print(df_temp)

        #Rename columns
        for i, v in enumerate(temp_list):
            if("_PFCandidate_" in v):
                pre_feat = v.replace(jet_string+str(j)+"_PFCandidate_","")
                #fast way to rename, without looping through all the pf candidates
                sep = '_'
                num = pre_feat.split(sep, 1)[0]
                feat = pre_feat.split(sep, 1)[1]
                df_temp.rename(columns={jet_string+str(j)+"_PFCandidate_"+str(num)+"_"+feat: feat+"_"+str(num)},inplace=True)
            elif(jet_string in v and "_PFCandidate_" not in v):
                feat = v.replace(jet_string+str(j)+"_","")
                df_temp.rename(columns={str(v): jet_string+feat},inplace=True)
                #print(jet_string+feat)
          
        #print("After renaming")
        #print(df_temp[ ["Jet_pt","Jet_index","is_signal"] ])

        #Concatenate jets
        if j==0:
            df_conc = df_temp
        else:
            df_conc = pd.concat([df_conc, df_temp ])

    #Remove empty jets; apply kinematical cuts
    df = df_conc[ (df_conc[jet_string+"pt"]>0) & (df_conc[jet_string+"eta"]<1.48) & (df_conc[jet_string+"eta"]>-1.48) & (df_conc[jet_string+"timeRecHitsEB"]>-100.) & (df_conc[jet_string+"timeRecHitsHB"]>-100.)]

    #Remove empty jets from train and val; apply kinematical cuts
    #if("val" in file_name):
    #    df = df_conc[ (df_conc["Jet_pt"]>0) & (df_conc["Jet_eta"]<1.48) & (df_conc["Jet_eta"]>-1.48) & (df_conc["Jet_timeRecHitsEB"]>-100.) & (df_conc["Jet_timeRecHitsHB"]>-100.)]
    #elif("train" in file_name):
    #    df = df_conc[ (df_conc["Jet_pt"]>0) & (df_conc["Jet_eta"]<1.48) & (df_conc["Jet_eta"]>-1.48) & (df_conc["Jet_timeRecHitsEB"]>-100.) & (df_conc["Jet_timeRecHitsHB"]>-100.) ]
    #else:
    #    df = df_conc
    
    print("\n")
    print("  * * * * * * * * * * * * * * * * * * * * * * *")
    print("  Time needed to convert: %.2f seconds" % (time.time() - startTime))
    print("  * * * * * * * * * * * * * * * * * * * * * * *")
    print("\n")

    #Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    del df_conc  
    df_train = df[ (df["EventNumber"]%2 == 0) ]
    df_test  = df[ (df["EventNumber"]%2 != 0) ]
    del df
    
    print("   -> Train shape: ", df_train.shape)
    print("   -> Test shape: ", df_test.shape)
    ##write h5
    #print(graphnet_folder+'/'+file_name)
    new_name = file_name.replace("_unconverted_","_")
    df_train.to_hdf(graphnet_folder+'/'+new_name+'_'+jet_type+'_train.h5', 'df', format='fixed')
    df_test.to_hdf(graphnet_folder+'/'+new_name+'_'+jet_type+'_test.h5', 'df', format='fixed')
    print("  "+graphnet_folder+"/"+new_name+"_"+jet_type+"_train.h5 & test.h5 stored")
    


    #print(df)
    #print("  DONEEEEE")
    print("  -------------------   ")



def convert_dataset_AK8_v4(folder,graphnet_folder,file_name,nj,nfj,npf,cols,selections=''):
    print("  Transform per-event into per-jet dataframes...")
    print("\n")

    startTime = time.time()
    ##folder+file_name WILL be folder+s and no loop ! TODO!

    ##Prepare columns
    event_list = []
    pf_list = []
    jet_list = []
    fat_jet_list = []


    for i,c in enumerate(cols):
        if isinstance(c, tuple):
            if "FatJet" in c[0]:
                fat_jet_list.append(c[0].replace('.','_').replace('s[','_').replace(']',''))
            else:
                jet_list.append(c[0].replace('.','_').replace('s[','_').replace(']',''))

        else:
            event_list.append(c.replace('.','_'))

    ##Prepare sample
    df_pre = defaultdict()
    print("going to open: ", folder+file_name+".h5")
    store_pre = pd.HDFStore(folder+file_name+".h5")
    df_pre    = store_pre.select("df",start=0,stop=-1)#
        
    df_temp = defaultdict()
    df_conc = defaultdict()
    df = defaultdict()


    for j in range(nfj):
        temp_list = []
        for l in (fat_jet_list+pf_list):
            if ("FatJet_"+str(j)) in l:
                #print(l)
                temp_list.append(l.replace('.','_'))
        if "FatJet_"+str(j)+"_isGenMatched" not in temp_list:
            temp_list.append("FatJet_"+str(j)+"_isGenMatched")
        if "FatJet_"+str(j)+"_pt" not in temp_list:
            temp_list.append("FatJet_"+str(j)+"_pt")
        if "FatJet_"+str(j)+"_eta" not in temp_list:
            temp_list.append("FatJet_"+str(j)+"_eta")
        if "FatJet_"+str(j)+"_phi" not in temp_list:
            temp_list.append("FatJet_"+str(j)+"_phi")
        if "FatJet_"+str(j)+"_timeRecHitsEB" not in temp_list:
            temp_list.append("FatJet_"+str(j)+"_timeRecHitsEB")

        #if "FatJet_"+str(j)+"_isGenMatchedCaloCorrLLPAccept" not in temp_list:
        #    temp_list.append("FatJet_"+str(j)+"_isGenMatchedCaloCorrLLPAccept")
        if "FatJet_"+str(j)+"_isGenMatchedCaloCorr" not in temp_list:
            temp_list.append("FatJet_"+str(j)+"_isGenMatchedCaloCorr")
        if "FatJet_"+str(j)+"_zLLP" not in temp_list:
            temp_list.append("FatJet_"+str(j)+"_zLLP")
        if "FatJet_"+str(j)+"_radiusLLP" not in temp_list:
            temp_list.append("FatJet_"+str(j)+"_radiusLLP")
        #print("here temp list ", temp_list)

        df_temp = df_pre[temp_list+event_list]
        df_temp["FatJet_index"] = np.ones(df_temp.shape[0])*j

        #print("\n")
        #print("Before renaming")
        #print(df_temp)

        #Rename columns
        for i, v in enumerate(temp_list):
            if("FatJet_" in v):
                feat = v.replace("FatJet_"+str(j)+"_","")
                df_temp.rename(columns={str(v): "FatJet_"+feat},inplace=True)
                #print("FatJet_"+feat)
          
        #print("After renaming")
        #print(df_temp[ ["FatJet_pt","FatJet_index","is_signal"] ])

        #Concatenate jets
        if j==0:
            df_conc = df_temp
        else:
            df_conc = pd.concat([df_conc, df_temp ])

    #Remove empty jets; apply kinematical cuts
    df = df_conc[ (df_conc["FatJet_pt"]>0) & (df_conc["FatJet_eta"]<1.48) & (df_conc["FatJet_eta"]>-1.48) & (df_conc["FatJet_timeRecHitsEB"]>-100.) & (df_conc["FatJet_timeRecHitsHB"]>-100.)]

    #Remove empty jets from train and val; apply kinematical cuts
    #if("val" in file_name):
    #    df = df_conc[ (df_conc["FatJet_pt"]>0) & (df_conc["FatJet_eta"]<1.48) & (df_conc["FatJet_eta"]>-1.48) & (df_conc["FatJet_timeRecHitsEB"]>-100.) & (df_conc["FatJet_timeRecHitsHB"]>-100.)]
    #elif("train" in file_name):
    #    df = df_conc[ (df_conc["FatJet_pt"]>0) & (df_conc["FatJet_eta"]<1.48) & (df_conc["FatJet_eta"]>-1.48) & (df_conc["FatJet_timeRecHitsEB"]>-100.) & (df_conc["FatJet_timeRecHitsHB"]>-100.) ]
    #else:
    #    df = df_conc
    
    print("\n")
    print("  * * * * * * * * * * * * * * * * * * * * * * *")
    print("  Time needed to convert: %.2f seconds" % (time.time() - startTime))
    print("  * * * * * * * * * * * * * * * * * * * * * * *")
    print("\n")
    print(df.shape)

    #Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    ##write h5
    #print(graphnet_folder+'/'+file_name)
    new_name = file_name.replace("_unconverted_","_")
    df.to_hdf(graphnet_folder+'/'+new_name+'_AK8jets.h5', 'df', format='fixed')
    print("  "+graphnet_folder+"/"+new_name+"_AK8jets.h5 stored")
    print("  -------------------   ")