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

def convert_dataset_v2(folder,graphnet_folder,file_name,nj,npf,cols):
    print("  Transform per-event into per-jet dataframes...")
    print("\n")

    startTime = time.time()
    ##folder+file_name WILL be folder+s and no loop ! TODO!

    ##Prepare sample
    df_pre = defaultdict()
    print("going to open: ", folder+file_name+".h5")
    store_pre = pd.HDFStore(folder+file_name+".h5")
    df_pre    = store_pre.select("df",start=0,stop=-1)#
        
    df_temp = defaultdict()
    df_conc = defaultdict()
    df = defaultdict()

    #print(cols)
    #print(df_pre)
    #exit()

    event_list = []
    pf_list = []
    jet_list = []


    #Associate PFcandidates to jets
    #print(df_pre["nCHSJets"])
    #for j in range(nj):
    #    jet_num = np.ones(df_pre.shape[0]) * j
    #    count_pfcand = np.zeros(df_pre.shape[0])
    #    for p in range(npf):
    #        #print(df_pre["PFCandidate_"+str(p)+"_jetIndex"])
    #        #print("pf cand assoc: ", df_pre["PFCandidate_"+str(p)+"_jetIndex"].values)
    #        #print(jet_num)
    #        mask = (df_pre["PFCandidate_"+str(p)+"_jetIndex"].values == jet_num).astype(int)

    #        #list of zero and ones, we want to know the index of the pf candidate

    #        #print("mask ", mask)
    #        #mask_after = (df_pre["PFCandidate_"+str(p)+"_jetIndex"].values == jet_num).astype(int) + 1
    #        #print("mask +1 ", mask_after)
    #        count_pfcand += mask
    #        #print(count_pfcand)
    #        #if df_pre["PFCandidate_"+str(p)+"_jetIndex"]==j:
    #        #    count_pfcand +=1
    #    print(" ~~~ Jet n. ", j, " has count_pfcand ", count_pfcand)
    #    df_pre["Jet_"+str(j)+"_nPFCand"] = count_pfcand
    #    jet_list.append("Jet_"+str(j)+"_nPFCand")

    #print(df_pre)
    #print(cols)


    for i,c in enumerate(cols):
        if isinstance(c, tuple):
           if "_PFCandidate_" in c[0]:
               pf_list.append(c[0].replace('.','_').replace('s[','_').replace(']',''))
           else:
               jet_list.append(c[0].replace('.','_').replace('s[','_').replace(']',''))

        else:
           event_list.append(c.replace('.','_'))

    #print(jet_list)
    #print(pf_list)
    #print(event_list)
    #print(df_pre[jet_list])
    #exit()
    for j in range(nj):
        temp_list = []
        #print("Jet n. ",j)
        for l in (jet_list+pf_list):
            if ("Jet_"+str(j)) in l:
                #print(l)
                temp_list.append(l.replace('.','_'))
        if "Jet_"+str(j)+"_isGenMatched" not in temp_list:
            temp_list.append("Jet_"+str(j)+"_isGenMatched")
        if "Jet_"+str(j)+"_pt" not in temp_list:
            temp_list.append("Jet_"+str(j)+"_pt")
        if "Jet_"+str(j)+"_eta" not in temp_list:
            temp_list.append("Jet_"+str(j)+"_eta")
        if "Jet_"+str(j)+"_phi" not in temp_list:
            temp_list.append("Jet_"+str(j)+"_phi")
        if "Jet_"+str(j)+"_timeRecHitsEB" not in temp_list:
            temp_list.append("Jet_"+str(j)+"_timeRecHitsEB")

        if "Jet_"+str(j)+"_isGenMatchedCaloCorrLLPAccept" not in temp_list:
            temp_list.append("Jet_"+str(j)+"_isGenMatchedCaloCorrLLPAccept")
        #print("here temp list ", temp_list)

        df_temp = df_pre[temp_list+event_list]
        df_temp["Jet_index"] = np.ones(df_temp.shape[0])*j

        #print("\n")
        #print("Before renaming")
        #print(df_temp)

        #Rename columns
        for i, v in enumerate(temp_list):
            if("_PFCandidate_" in v):
                pre_feat = v.replace("Jet_"+str(j)+"_PFCandidate_","")
                #fast way to rename, without looping through all the pf candidates
                sep = '_'
                num = pre_feat.split(sep, 1)[0]
                feat = pre_feat.split(sep, 1)[1]
                df_temp.rename(columns={"Jet_"+str(j)+"_PFCandidate_"+str(num)+"_"+feat: feat+"_"+str(num)},inplace=True)
            elif("Jet_" in v and "_PFCandidate_" not in v):
                feat = v.replace("Jet_"+str(j)+"_","")
                df_temp.rename(columns={str(v): "Jet_"+feat},inplace=True)
                #print("Jet_"+feat)
          
        #print("After renaming")
        #print(df_temp[ ["Jet_pt","Jet_index","is_signal"] ])

        #Concatenate jets
        if j==0:
            df_conc = df_temp
        else:
            df_conc = pd.concat([df_conc,df_temp])

    #Remove empty jets from train and val
    #df = df_conc[ df_conc["Jet_pt"]>0 ]#think if it makes sense for test... maybe not
    df = df_conc
    
    print("\n")
    print("  * * * * * * * * * * * * * * * * * * * * * * *")
    print("  Time needed to convert: %.2f seconds" % (time.time() - startTime))
    print("  * * * * * * * * * * * * * * * * * * * * * * *")
    print("\n")
    ##write h5
    #print(graphnet_folder+'/'+file_name)
    
    df.to_hdf(graphnet_folder+'/'+file_name+'.h5', 'df', format='fixed')
    print("  "+graphnet_folder+"/"+file_name+".h5 stored")
    
    #print(df)
    #print("  DONEEEEE")
    print("  -------------------   ")