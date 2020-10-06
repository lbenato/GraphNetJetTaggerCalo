import pandas as pd
import numpy as np
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

DIR = "/nfs/dust/cms/group/cms-llp/dataframes_graphnet/v2_calo_AOD_2017_condor_JJ/"
OUT = "/nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/plot/"

DIR = "/nfs/dust/cms/group/cms-llp/dataframes_lisa/v3_calo_AOD_2018_dnn/"
OUT = "/nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/plot/v3_calo_AOD_2018_dnn/"

DIR = "/nfs/dust/cms/group/cms-llp/dataframes_lisa/v3_calo_AOD_2018_BDT_new_presel/"
OUT = "/nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/plot/v3_calo_AOD_2018_dnn_new_presel/"

DIR = "/nfs/dust/cms/group/cms-llp/dataframes_jh/v3_calo_AOD_2018_jh_partnet/"
OUT = "/nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/plot/v3_calo_AOD_2018_dnn_new_presel/"

def plot(filename,var,weight,minval,maxval,nbins):
    
    store = pd.HDFStore(DIR+filename+".h5")
    df_test = store.select("df")

    print("    Remove negative weights at testing!!!!!!")
    df_test = df_test.loc[df_test['EventWeight']>=0]
    
    back = np.array(df_test[var].loc[df_test["is_signal"]==0].values)
    sign = np.array(df_test[var].loc[df_test["is_signal"]==1].values)
    if weight!="":
        back_w = np.array(df_test[weight].loc[df_test["is_signal"]==0].values)
        sign_w = np.array(df_test[weight].loc[df_test["is_signal"]==1].values)
    else:
        back_w = np.array(np.ones(back.shape[0]))
        sign_w = np.array(np.ones(sign.shape[0]))   
    
    plt.figure(figsize=(8,7))
    plt.rcParams.update({'font.size': 15}) #Larger font size

    
    plt.hist(back, np.linspace(minval,maxval,nbins), weights=back_w, color='blue', edgecolor='blue', lw=2, label='background', alpha=0.3)#, density=True)
    plt.hist(sign, np.linspace(minval,maxval,nbins), weights=sign_w, color='red', edgecolor='red', lw=2, label='signal', alpha=0.3)#, density=True)

    plt.xlim([minval, maxval])
    plt.xlabel(var)
    plt.legend(loc="upper right", title='')
    plt.yscale('log')
    plt.grid(True)
    #plt.show()
    plt.savefig(OUT+filename+"_"+var+weight+'.png')
    plt.savefig(OUT+filename+"_"+var+weight+'.pdf')
    plt.close()
    
    
#for w in ["","SampleWeight","EventWeight","EventWeightNormalized"]:
for w in ["EventWeightNormalized"]:
    #plot("val","Jet_pt",w,0,500,50)
    #plot("val","Jet_eta",w,-2,2,50)
    plot("val","HT",w,0,500,50)
    plot("val","MEt_pt",w,0,200,50)#100, 3100
    #plot("val","Jet_eleEFrac",w,0,1,50)
    #plot("val","Jet_cHadEFrac",w,0,1,50)
    #plot("val","Jet_nHadEFrac",w,0,1,50)
    #plot("val","Jet_photonEFrac",w,0,1,50)
    #plot("val","Jet_timeRecHits",w,-15,15,50)
    #plot("val","Jet_gammaMaxET",w,0,5,50)
    #plot("val","Jet_betaMax",w,0,5,50)
    #plot("val","Jet_gammaMax",w,0,5,50)
    #plot("val","Jet_gammaMaxEM",w,0,5,50)
    #plot("val","Jet_gammaMaxHadronic",w,0,5,50)
    #plot("val","Jet_alphaMax",w,0,1,50)
    #plot("val","Jet_minDeltaRPVTracks",w,0,2,50)
    #plot("val","Jet_minDeltaRAllTracks",w,0,2,50)
    #plot("val","Jet_medianIP2D",w,-10000,10000,50)
    #plot("val","Jet_timeRecHitsEB",w,-25,25,50)
    #plot("val","Jet_timeRecHitsHB",w,-25,25,50)
    #plot("val","Jet_energyRecHitsEB",w,0,100,50)
    #plot("val","Jet_energyRecHitsHB",w,0,100,50)
    #plot("val","Jet_nRecHitsEB",w,0,100,50)
    #plot("val","Jet_nSelectedTracks",w,0,100,50)
    #plot("val","Jet_nTrackConstituents",w,0,100,50)
    #plot("val","Jet_nPixelHitsMedian",w,0,10,11)
    #plot("val","Jet_nHitsMedian",w,0,50,51)
    #plot("val","Jet_nTracksAll",w,0,100,101)
    #plot("val","Jet_nTracksPVMax",w,0,100,101)
    #plot("val","Jet_ptAllTracks",w,0,500,50)
    #plot("val","Jet_ptAllPVTracks",w,0,500,50)
    #plot("val","Jet_ptPVTracksMax",w,0,500,50)
    #plot("val","dxy_0",w,-50,50,50)
    #plot("val","dxy_29",w,-50,50,50)
    #plot("val","dxy_29",w,-20000,20000,50)

'ptAllTracks', 'ptAllPVTracks', 'ptPVTracksMax', 'nTracksAll', 'nTracksPVMax',


'timeRecHits', 
'gammaMaxET','minDeltaRPVTracks',
