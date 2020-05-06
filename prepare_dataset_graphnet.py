import pandas as pd
from tensorflow import keras
import tensorflow as tf
import numpy as np
import os.path
from datetime import datetime
from collections import defaultdict
# Needed libraries
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from root_numpy import array2tree, array2root
from dnn_functions import *
from samplesAOD2017 import *

#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True
#tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

# Configure parameters
pd_folder = 'dataframes/v2_calo_AOD_2017/'
graphnet_pd_folder = 'dataframes_graphnet/v2_calo_AOD_2017_test/'
result_folder = 'model_weights/v2_calo_AOD_2017/'
graphnet_result_folder = 'model_weights_graphnet/v2_calo_AOD_2017/'


print("\n")
print("\n")

#sgn = ['ggH_MH1000_MS150_ctau1000']#,'ggH_MH1000_MS400_ctau1000']
sgn = ['SUSY_mh400_pl1000']
bkg = ['VV']
#bkg = ['ZJetsToNuNuRed']
#bkg = []


# Define your variables here
var_list = []

event_list = [
            'EventNumber',
            'RunNumber','LumiNumber','EventWeight','isMC',
            #'isVBF','HT','MEt_pt','MEt_phi','MEt_sign','MinJetMetDPhi',
            'nCHSJets',
            #'nElectrons','nMuons','nPhotons','nTaus','nPFCandidates','nPFCandidatesTrack',
            'ttv','is_signal',
            ]

#jets variables
nj = 10
jtype = ['Jet']
jvar = [
       'pt','eta','phi','mass','nConstituents',
       'nTrackConstituents','nSelectedTracks','nHadEFrac', 'cHadEFrac','ecalE','hcalE', 'muEFrac','eleEFrac','photonEFrac', 'eleMulti','muMulti','photonMulti','cHadMulti','nHadMulti', 'nHitsMedian','nPixelHitsMedian', 'dRSVJet', 'nVertexTracks', 'CSV', 'SV_mass',#FCN tagger
       ]
jvar+=['isGenMatched']
jet_list = []

#pf candidates variables
npf = 100
pftype = ['PFCandidate']
pfvar = [
        'energy', 'px','py','pz','pt',
        'eta','phi','mass',
        'pdgId','isTrack','hasTrackDetails', 'dxy', 'dz', 'POCA_x', 'POCA_y', 'POCA_z', 'POCA_phi',
        'ptError', 'etaError', 'phiError', 'dxyError', 'dzError', 'theta', 'thetaError','chi2', 'ndof', 'normalizedChi2',
        'nHits', 'nPixelHits', 'lostInnerHits',
        'jetIndex'
        ]
pf_list = []
for n in range(nj):
    for t in jtype:
        for v in jvar:
            jet_list.append(str(t)+"_"+str(n)+"_"+v)
        for p in range(npf):
            for tp in pftype:
                for pv in pfvar:
                    pf_list.append(str(t)+"_"+str(n)+"_"+str(tp)+"_"+str(p)+"_"+str(pv))


var_list += jet_list
var_list += pf_list
print(var_list)

if(len(var_list)>=2000):
    print(len(var_list))
    print("\n")
    print("\n")
    print(" Warning! Too many columns! Can't be handled by pandas tables! Will used fixed format!")
    print("\n")
    print("\n")
    

train_features = ['energy','px','py','pz','pt',]
top_features   = ['energy','px','py','pz','pt',]#['E', 'PX', 'PY', 'PZ','PT',]

train_features += [
    'eta','phi','mass',
    'pdgId','isTrack','hasTrackDetails', 'dxy', 'dz', 'POCA_x', 'POCA_y', 'POCA_z', 'POCA_phi',
    'ptError', 'etaError', 'phiError', 'dxyError', 'dzError', 'theta', 'thetaError','chi2', 'ndof', 'normalizedChi2',
    'nHits', 'nPixelHits', 'lostInnerHits', 'jetIndex',
]
top_features += [
    'eta','phi','mass',
    'pdgId','isTrack','hasTrackDetails', 'dxy', 'dz', 'POCA_x', 'POCA_y', 'POCA_z', 'POCA_phi',
    'ptError', 'etaError', 'phiError', 'dxyError', 'dzError', 'theta', 'thetaError','chi2', 'ndof', 'normalizedChi2',
    'nHits', 'nPixelHits', 'lostInnerHits', 'jetIndex',
]

cols = []

for n in range(nj):
    for t in jtype:
        for v in jvar:
            jet_list.append(str(t)+"_"+str(n)+"."+v)
        for p in range(npf):
            for tp in pftype:
                for pv in train_features:
                    cols.append(str(t)+"_"+str(n)+"_"+str(tp)+"_"+str(p)+"_"+str(pv))

print("\n")
print(cols)
print(len(cols)," training features!")
print("\n")

#exit()

##Time stamp for saving model
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%d%b%Y_%H_%M_%S")
print("Time:", timestampStr)
print("\n")


def convert_dataset(folder,graphnet_folder,sgn,bkg):
    print("  Transform per-event into per-jet dataframes...")
    print("\n")

    signal_list = []
    background_list = []
    for a in sgn:
        signal_list += samples[a]['files']

    for b in bkg:
        background_list += samples[b]['files']

    print(signal_list)
    print(background_list)


    ##Prepare train/test/val sample for signal
    df_pre_train_s = defaultdict()
    df_pre_test_s = defaultdict()
    df_pre_val_s = defaultdict()
    for n, s in enumerate(signal_list):
        print("   ",n, s)
        ##Load dataframes
        store_pre_train_s = pd.HDFStore(folder+s+"_train.h5")
        df_pre_train_s[s] = store_pre_train_s.select("df",start=0,stop=-1)#
        store_pre_test_s  = pd.HDFStore(folder+s+"_test.h5")
        df_pre_test_s[s]  = store_pre_test_s.select("df",start=0,stop=-1)#
        store_pre_val_s   = pd.HDFStore(folder+s+"_val.h5")
        df_pre_val_s[s]   = store_pre_val_s.select("df",start=0,stop=-1)#
        
        #print(df_pre_test_s[s])
        
    df_temp_train_s = defaultdict()
    df_temp_test_s = defaultdict()
    df_temp_val_s = defaultdict()
    df_conc_train_s = defaultdict()
    df_conc_test_s = defaultdict()
    df_conc_val_s = defaultdict()
    df_train_s = defaultdict()
    df_test_s = defaultdict()
    df_val_s = defaultdict()
    #exit()
    #Loop on signals
    for n, s in enumerate(signal_list):
        #Transform per-event into per-jet
        for j in range(nj):
            temp_list = []
            for l in var_list:#all variables
            #for l in cols:#only the one we want to train?
                if ("Jet_"+str(j)) in l:
                    #print(l)
                    temp_list.append(l)
                 
            #print("Here doing per jet")
            #print(temp_list)
            
            ##Temp train
            df_temp_train_s[s] = df_pre_train_s[s][temp_list+event_list]
            df_temp_train_s[s]["Jet_index"] = np.ones(df_temp_train_s[s].shape[0])*j
            ##Temp test
            df_temp_test_s[s] = df_pre_test_s[s][temp_list+event_list]
            df_temp_test_s[s]["Jet_index"] = np.ones(df_temp_test_s[s].shape[0])*j
            ##Temp val
            df_temp_val_s[s] = df_pre_val_s[s][temp_list+event_list]
            df_temp_val_s[s]["Jet_index"] = np.ones(df_temp_val_s[s].shape[0])*j
            #print("\n")
            #print("Before renaming")
            #print(df_temp_val_s)
            

            #Rename columns
            for i, v in enumerate(train_features):
                for p in range(npf):
                    #print(i,train_features[i],top_features[i])
                    #print("Jet_"+str(j)+"_PFCandidate_"+str(p)+"_"+train_features[i])
                    #print(top_features[i]+str(p))
                    #print(df_temp_val_s[s]["Jet_"+str(j)+"_PFCandidate_"+str(p)+"_"+train_features[i]])
                    df_temp_train_s[s].rename(columns={"Jet_"+str(j)+"_PFCandidate_"+str(p)+"_"+train_features[i]: train_features[i]+"_"+str(p)},inplace=True)
                    df_temp_test_s[s].rename(columns={"Jet_"+str(j)+"_PFCandidate_"+str(p)+"_"+train_features[i]: train_features[i]+"_"+str(p)},inplace=True)
                    df_temp_val_s[s].rename(columns={"Jet_"+str(j)+"_PFCandidate_"+str(p)+"_"+train_features[i]: train_features[i]+"_"+str(p)},inplace=True)
                    
            for v in jvar: 
                #print("Jet_"+str(j)+"_"+str(v))      
                df_temp_train_s[s].rename(columns={"Jet_"+str(j)+"_"+str(v): "Jet_"+str(v)},inplace=True)
                df_temp_test_s[s].rename( columns={"Jet_"+str(j)+"_"+str(v): "Jet_"+str(v)},inplace=True)
                df_temp_val_s[s].rename(  columns={"Jet_"+str(j)+"_"+str(v): "Jet_"+str(v)},inplace=True)
            
            #print(df_temp_val_s[s])
            #exit()

            #Concatenate jets
            if j==0:
                df_conc_train_s[s] = df_temp_train_s[s]
                df_conc_test_s[s] = df_temp_test_s[s]
                df_conc_val_s[s] = df_temp_val_s[s]
            else:
                df_conc_train_s[s] = pd.concat([df_conc_train_s[s],df_temp_train_s[s]])
                df_conc_test_s[s] = pd.concat([df_conc_test_s[s],df_temp_test_s[s]])
                df_conc_val_s[s] = pd.concat([df_conc_val_s[s],df_temp_val_s[s]])

        ##df_train_s[s] = df_conc_train_s[s][ df_conc_train_s[s]["Jet_isGenMatched"]==1 ]
        ##df_test_s[s] = df_conc_test_s[s][ df_conc_test_s[s]["Jet_isGenMatched"]==1 ]
        ##no selections at the moment
        df_train_s[s] = df_conc_train_s[s]
        df_test_s[s] = df_conc_test_s[s]
        df_val_s[s] = df_conc_val_s[s]
        print(s, df_train_s[s])
        ##write h5
        df_train_s[s].to_hdf(graphnet_folder+'/'+s+'_train.h5', 'df', format='table' if (len(var_list)<=2000) else 'fixed')
        print("  "+graphnet_folder+"/"+s+"_train.h5 stored")
        df_test_s[s].to_hdf(graphnet_folder+'/'+s+'_test.h5', 'df', format='table' if (len(var_list)<=2000) else 'fixed')
        print("  "+graphnet_folder+"/"+s+"_test.h5 stored")
        df_val_s[s].to_hdf(graphnet_folder+'/'+s+'_val.h5', 'df', format='table' if (len(var_list)<=2000) else 'fixed')
        print("  "+graphnet_folder+"/"+s+"_val.h5 stored")
        print("  -------------------   ")

    #exit()


    ##Prepare train/test/val sample for background
    df_pre_train_b = defaultdict()
    df_pre_test_b = defaultdict()
    df_pre_val_b = defaultdict()
    for n, b in enumerate(background_list):
        print("   ",n, s)
        ##Load dataframes
        store_pre_train_b = pd.HDFStore(folder+b+"_train.h5")
        df_pre_train_b[b] = store_pre_train_b.select("df",start=0,stop=-1)#
        store_pre_test_b  = pd.HDFStore(folder+b+"_test.h5")
        df_pre_test_b[b]  = store_pre_test_b.select("df",start=0,stop=-1)#
        store_pre_val_b   = pd.HDFStore(folder+b+"_val.h5")
        df_pre_val_b[b]   = store_pre_val_b.select("df",start=0,stop=-1)#
        
        
    df_temp_train_b = defaultdict()
    df_temp_test_b = defaultdict()
    df_temp_val_b = defaultdict()
    df_conc_train_b = defaultdict()
    df_conc_test_b = defaultdict()
    df_conc_val_b = defaultdict()
    df_train_b = defaultdict()
    df_test_b = defaultdict()
    df_val_b = defaultdict()
    
    #Loop on background
    for n, b in enumerate(background_list):
        #Transform per-event into per-jet
        for j in range(nj):
            temp_list = []
            for l in var_list:#all variables
            #for l in cols:#only the one we want to train?
                if ("Jet_"+str(j)) in l:
                    #print(l)
                    temp_list.append(l)
                 
            #print("Here doing per jet")
            #print(temp_list)
            
            ##Temp train
            df_temp_train_b[b] = df_pre_train_b[b][temp_list+event_list]
            df_temp_train_b[b]["Jet_index"] = np.ones(df_temp_train_b[b].shape[0])*j
            ##Temp test
            df_temp_test_b[b] = df_pre_test_b[b][temp_list+event_list]
            df_temp_test_b[b]["Jet_index"] = np.ones(df_temp_test_b[b].shape[0])*j
            ##Temp val
            df_temp_val_b[b] = df_pre_val_b[b][temp_list+event_list]
            df_temp_val_b[b]["Jet_index"] = np.ones(df_temp_val_b[b].shape[0])*j
            

            #Rename columns
            for i, v in enumerate(train_features):
                for p in range(npf):
                    df_temp_train_b[b].rename(columns={"Jet_"+str(j)+"_PFCandidate_"+str(p)+"_"+train_features[i]: train_features[i]+"_"+str(p)},inplace=True)
                    df_temp_test_b[b].rename(columns={"Jet_"+str(j)+"_PFCandidate_"+str(p)+"_"+train_features[i]: train_features[i]+"_"+str(p)},inplace=True)
                    df_temp_val_b[b].rename(columns={"Jet_"+str(j)+"_PFCandidate_"+str(p)+"_"+train_features[i]: train_features[i]+"_"+str(p)},inplace=True)
                    
            for v in jvar:     
                df_temp_train_b[b].rename(columns={"Jet_"+str(j)+"_"+str(v): "Jet_"+str(v)},inplace=True)
                df_temp_test_b[b].rename( columns={"Jet_"+str(j)+"_"+str(v): "Jet_"+str(v)},inplace=True)
                df_temp_val_b[b].rename(  columns={"Jet_"+str(j)+"_"+str(v): "Jet_"+str(v)},inplace=True)
            

            #Concatenate jets
            if j==0:
                df_conc_train_b[b] = df_temp_train_b[b]
                df_conc_test_b[b] = df_temp_test_b[b]
                df_conc_val_b[b] = df_temp_val_b[b]
            else:
                df_conc_train_b[b] = pd.concat([df_conc_train_b[b],df_temp_train_b[b]])
                df_conc_test_b[b] = pd.concat([df_conc_test_b[b],df_temp_test_b[b]])
                df_conc_val_b[b] = pd.concat([df_conc_val_b[b],df_temp_val_b[b]])

        df_train_b[b] = df_conc_train_b[b]
        df_test_b[b] = df_conc_test_b[b]
        df_val_b[b] = df_conc_val_b[b]
        print(b, df_train_b[b])
        ##write h5
        df_train_b[b].to_hdf(graphnet_folder+'/'+b+'_train.h5', 'df', format='table' if (len(var_list)<=2000) else 'fixed')
        print("  "+graphnet_folder+"/"+b+"_train.h5 stored")
        df_test_b[b].to_hdf(graphnet_folder+'/'+b+'_test.h5', 'df', format='table' if (len(var_list)<=2000) else 'fixed')
        print("  "+graphnet_folder+"/"+b+"_test.h5 stored")
        df_val_b[b].to_hdf(graphnet_folder+'/'+b+'_val.h5', 'df', format='table' if (len(var_list)<=2000) else 'fixed')
        print("  "+graphnet_folder+"/"+b+"_val.h5 stored")
        print("  -------------------   ")
        


def prepare_dataset(folder,sgn,bkg,upsample_signal_factor=0,signal_match_train=True):
    print("   Preparing input dataset.....   ")
    print("\n")
    #if model_label=="":
    #    model_label=timestampStr

    signal_list = []
    background_list = []
    for a in sgn:
        signal_list += samples[a]['files']

    for b in bkg:
        background_list += samples[b]['files']

    print(signal_list)
    print(background_list)

    ##Prepare train/test sample for signal
    for n, s in enumerate(signal_list):
        print("   ",n, s)
        #load train tables
        store_temp_train_s = pd.HDFStore(folder+s+"_train.h5")
        df_temp_train_s = store_temp_train_s.select("df")
        #load test tables
        store_temp_test_s = pd.HDFStore(folder+s+"_test.h5")
        df_temp_test_s = store_temp_test_s.select("df")
        #load validation tables
        store_temp_val_s = pd.HDFStore(folder+s+"_val.h5")
        df_temp_val_s = store_temp_val_s.select("df")
        if n==0:
            df_train_s = df_temp_train_s
            df_test_s = df_temp_test_s
            df_val_s = df_temp_val_s
        else:
            df_train_s = pd.concat([df_train_s,df_temp_train_s])
            df_test_s = pd.concat([df_test_s,df_temp_test_s])
            df_val_s = pd.concat([df_val_s,df_temp_val_s])

    if signal_match_train:
        print("  -------------------   ")
        print("    Training/validation signal only on gen matched jets!!!")
        print(" Size before: ", df_train_s.shape[0])
        df_train_s = df_train_s[ df_train_s["Jet_isGenMatched"] ==1 ]
        print(" Size before: ", df_train_s.shape[0])
        print("  -------------------   ")
        df_val_s = df_val_s[ df_val_s["Jet_isGenMatched"] ==1 ]
        print("\n")
        print(" ~~~~ For top tagging exercise recasted on LL: removing empty jets and events with negative weights also from test data!!! ~~~")
        print("\n")
        df_test_s = df_test_s[ df_test_s["Jet_isGenMatched"] ==1 ]
    
    if upsample_signal_factor>0:
        print("   df_train_s.shape[0] before upsampling", df_train_s.shape[0])
        print("   Will enlarge training/validation sample by factor ", upsample_signal_factor)
        df_train_s = pd.concat([df_train_s]*upsample_signal_factor)
        print("   df_train_s.shape[0] AFTER upsampling", df_train_s.shape[0])
        df_val_s = pd.concat([df_val_s]*upsample_signal_factor)

    ##Remove negative weights for training!
    ### !!! !! Removing negative weights also for test sample!!
    print("\n")
    print(" ~~~~ For top tagging exercise recasted on LL: removing empty jets and events with negative weights also from test data!!! ~~~")
    print("\n")
    print("----Signal training shape before removing negative weights: ")
    print("   df_train_s.shape[0]", df_train_s.shape[0])
    df_train_s = df_train_s[df_train_s['EventWeight']>=0]
    df_val_s = df_val_s[df_val_s['EventWeight']>=0]
    df_test_s  = df_test_s[df_test_s['EventWeight']>=0]
    print("----Signal training shape after removing negative weights: ")
    print("   df_train_s.shape[0]", df_train_s.shape[0])
    print(df_train_s)
    
    ##Normalize train/validation weights
    print("   df_train_s.shape[0]", df_train_s.shape[0])
    norm_train_s = df_train_s['EventWeight'].sum(axis=0)
    print("   renorm signal train: ", norm_train_s)
    df_train_s['EventWeightNormalized'] = df_train_s['EventWeight'].div(norm_train_s)
    df_train_s = df_train_s.sample(frac=1).reset_index(drop=True)#shuffle signals
    
    norm_val_s = df_val_s['EventWeight'].sum(axis=0)
    df_val_s['EventWeightNormalized'] = df_val_s['EventWeight'].div(norm_val_s)
    df_val_s = df_val_s.sample(frac=1).reset_index(drop=True)#shuffle signals

    ##Normalize test weights
    print("   df_test_s.shape[0]", df_test_s.shape[0])
    norm_test_s = df_test_s['EventWeight'].sum(axis=0)
    print("   renorm signal test: ", norm_test_s)
    df_test_s['EventWeightNormalized'] = df_test_s['EventWeight'].div(norm_test_s)
    df_test_s = df_test_s.sample(frac=1).reset_index(drop=True)#shuffle signals
    print("  -------------------   ")
    print("\n")
    ###n_events_s = int(all_sign.shape[0] * train_percentage)
    ###df_train_s = all_sign.head(n_events_s)
    ###df_test_s = all_sign.tail(all_sign.shape[0] - n_events_s)

    ##Prepare train sample for background
    for n, b in enumerate(background_list):
        print("   ",n, b)
        if not os.path.isfile(folder+b+"_train.h5"):
            print("!!!File ", folder+b+"_train.h5", " does not exist! Continuing")
            continue
        #load train tables
        store_temp_train_b = pd.HDFStore(folder+b+"_train.h5")
        df_temp_train_b = store_temp_train_b.select("df")
        #load test tables
        store_temp_test_b = pd.HDFStore(folder+b+"_test.h5")
        df_temp_test_b = store_temp_test_b.select("df")
        #load val tables
        store_temp_val_b = pd.HDFStore(folder+b+"_val.h5")
        df_temp_val_b = store_temp_val_b.select("df")
        if n==0:
            df_train_b = df_temp_train_b
            df_test_b = df_temp_test_b
            df_val_b = df_temp_val_b
        else:
            df_train_b = pd.concat([df_train_b,df_temp_train_b])
            df_test_b = pd.concat([df_test_b,df_temp_test_b])
            df_val_b = pd.concat([df_val_b,df_temp_val_b])


    print("Remove background empty jets from training/validation!")
    print("\n")
    print(" ~~~~ For top tagging exercise recasted on LL: removing empty jets and events with negative weights also from test data!!! ~~~")
    print("\n")
    df_train_b = df_train_b[ df_train_b["Jet_pt"] >-1 ]
    df_val_b = df_val_b[ df_val_b["Jet_pt"] >-1 ]
    df_test_b = df_test_b[ df_test_b["Jet_pt"] >-1 ]

    ##Remove negative weights for training/validation!
    print("----Background training shape before removing negative weights: ")
    print("   df_train_b.shape[0]", df_train_b.shape[0])
    print("\n")
    print(" ~~~~ For top tagging exercise recasted on LL: removing empty jets and events with negative weights also from test data!!! ~~~")
    print("\n")
    df_train_b = df_train_b[df_train_b['EventWeight']>=0]
    df_val_b  = df_val_b[df_val_b['EventWeight']>=0]
    df_test_b  = df_test_b[df_test_b['EventWeight']>=0]
    print("----Background training shape after removing negative weights: ")
    print("   df_train_b.shape[0]", df_train_b.shape[0])
    
    ##Normalize train weights
    print("   df_train_b.shape[0]", df_train_b.shape[0])
    norm_train_b = df_train_b['EventWeight'].sum(axis=0)
    print("   renorm background train: ", norm_train_b)
    df_train_b['EventWeightNormalized'] = df_train_b['EventWeight'].div(norm_train_b)
    df_train_b = df_train_b.sample(frac=1).reset_index(drop=True)#shuffle signals

    ##Normalize val weights
    print("   df_val_b.shape[0]", df_val_b.shape[0])
    norm_val_b = df_val_b['EventWeight'].sum(axis=0)
    print("   renorm background val: ", norm_val_b)
    df_val_b['EventWeightNormalized'] = df_val_b['EventWeight'].div(norm_val_b)
    df_val_b = df_val_b.sample(frac=1).reset_index(drop=True)#shuffle signals
    
    ##Normalize test weights
    print("   df_test_b.shape[0]", df_test_b.shape[0])
    norm_test_b = df_test_b['EventWeight'].sum(axis=0)
    print("   renorm background test: ", norm_test_b)
    df_test_b['EventWeightNormalized'] = df_test_b['EventWeight'].div(norm_test_b)
    df_test_b = df_test_b.sample(frac=1).reset_index(drop=True)#shuffle signals

    print("  -------------------   ")
    
    ###n_events_b = int(all_back.shape[0] * train_percentage)
    ###df_train_b = all_back.head(n_events_b)
    ###df_test_b = all_back.tail(all_back.shape[0] - n_events_b)

    print("\n")
    print("   Ratio nB/nS: ", df_train_b.shape[0]/df_train_s.shape[0])
    
    ##Prepare global train and test samples
    df_train = pd.concat([df_train_s,df_train_b])
    df_test = pd.concat([df_test_s,df_test_b])
    df_val = pd.concat([df_val_s,df_val_b])

    ##Reshuffle
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_test   = df_test.sample(frac=1).reset_index(drop=True)
    df_val    = df_val.sample(frac=1).reset_index(drop=True)
    

    print("train: ", df_train['Jet_pt'])
    print("test: ", df_test['is_signal'])
    print("val: ", df_val['energy_0'])
    
    df_train.to_hdf(folder+'train.h5', 'df', format='table' if (len(var_list)<=2000) else 'fixed')
    df_test.to_hdf(folder+'test.h5', 'df', format='table' if (len(var_list)<=2000) else 'fixed')
    df_val.to_hdf(folder+'val.h5', 'df', format='table' if (len(var_list)<=2000) else 'fixed')
    print("\n")
    print("   "+folder+"train.h5 stored")
    print("   "+folder+"test.h5 stored")
    print("   "+folder+"val.h5 stored")  
    print("\n")
    


##model 0, first attempt:

print("\n")
print(" ~~~~ For top tagging exercise recasted on LL: removing empty jets and events with negative weights also from test data!!! ~~~")
print("\n")

convert_dataset(pd_folder,graphnet_pd_folder,sgn,bkg)
prepare_dataset(graphnet_pd_folder,sgn,bkg,upsample_signal_factor=0,signal_match_train=True)#
