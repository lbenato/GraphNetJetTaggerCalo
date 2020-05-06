import ROOT
import root_numpy as rnp
import numpy as np
import pandas as pd
import tables
import uproot
import time
from math import ceil
from ROOT import gROOT, TFile, TTree, TObject, TH1, TH1F, AddressOf, TLorentzVector
from dnn_functions import *
gROOT.ProcessLine('.L Objects.h' )
from ROOT import JetType, CaloJetType, MEtType, CandidateType, DT4DSegmentType, CSCSegmentType, PFCandidateType#, TrackType

# storage folder of the original root files
folder = '/nfs/dust/cms/group/cms-llp/v2_calo_AOD_2017/Skim/'
#sgn = ['ggH_MH1000_MS150_ctau1000']#,'ggH_MH1000_MS400_ctau1000'']
sgn = ['SUSY_mh400_pl1000','SUSY_mh300_pl1000','SUSY_mh250_pl1000','SUSY_mh200_pl1000','SUSY_mh175_pl1000','SUSY_mh150_pl1000','SUSY_mh127_pl1000']
bkg = ['VV']
#bkg = ['ZJetsToNuNuRed']
#bkg = []
sgn = []

#sgn = ['ggH_MH1000_MS400_ctau500','ggH_MH1000_MS400_ctau1000','ggH_MH1000_MS400_ctau2000','ggH_MH1000_MS400_ctau5000','ggH_MH1000_MS400_ctau10000',
#'ggH_MH1000_MS150_ctau500','ggH_MH1000_MS150_ctau1000','ggH_MH1000_MS150_ctau2000','ggH_MH1000_MS150_ctau5000','ggH_MH1000_MS150_ctau10000']
from samplesAOD2017 import *

# define your variables here
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
        'nHitsMedian','nPixelHitsMedian', 'dRSVJet', 'nVertexTracks', 'CSV', 'SV_mass']
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
#print(var_list)
#,'j0_pt','j1_pt','j0_nTrackConstituents','j1_nTrackConstituents','j0_nConstituents','j1_nConstituents','j0_nSelectedTracks','j1_nSelectedTracks','j0_nTracks3PixelHits','j1_nTracks3PixelHits','j0_nHadEFrac','j1_nHadEFrac','j0_cHadEFrac','j1_cHadEFrac']#,'c_nEvents']#,'is_signal']
if(len(var_list)>=2000):
    print(len(var_list))
    print("\n")
    print("\n")
    print(" Warning! Too many columns! Can't be handled by pandas tables! Will used fixed format!")
    print("\n")
    print("\n")

variables = []
MEt = MEtType()
#CHSJets = JetType()

def write_h5(folder,output_folder,file_list,test_split,val_split,tree_name="",counter_hist="",sel_cut="",obj_sel_cut="",verbose=True):
    print("    Opening ", folder)
    print("\n")
    # loop over files
    for a in file_list:
        print(a)
        for i, ss in enumerate(samples[a]['files']):
            #read number of entries
            oldFile = TFile(folder+ss+'.root', "READ")
            counter = oldFile.Get(counter_hist)#).GetBinContent(1)
            nevents_gen = counter.GetBinContent(1)
            print("  n events gen.: ", nevents_gen)
            oldTree = oldFile.Get(tree_name)
            nevents_tot = oldTree.GetEntries()#?#-1
            tree_weight = oldTree.GetWeight()
            print("   Tree weight:   ",tree_weight)

            if verbose:
                print("\n")
                #print("   Initialized df for sample: ", file_name)
                print("   Initialized df for sample: ", ss)
                print("   Reading n. events in tree: ", nevents_tot)
                #print("\n")

            if nevents_tot<0:
                print("   Empty tree!!! ")
                continue

            # First loop: check how many events are passing selections
            count = rnp.root2array(folder+ss+'.root', selection = sel_cut, object_selection = obj_sel_cut, treename=tree_name, branches=var_list[0], start=0, stop=nevents_tot)
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
            b = rnp.root2array(folder+ss+'.root', selection = sel_cut, object_selection = obj_sel_cut, treename=tree_name, branches=var_list, start=0, stop=nevents_tot)
            df = pd.DataFrame(b,columns=var_list)
            #Remove dots from column names
            column_names = []
            for a in var_list:
                column_names.append(a.replace('.','_'))
            #print(column_names)
            df.columns = column_names
            #if needed, rename columns: nJets --> nCHSJets
            #df.rename(columns={"nJets": "nCHSJets"})      
            #add is_signal flag
            df["is_signal"] = np.ones(nevents) if (("n3n2" in ss) or ("H2ToSSTobbbb" in ss)) else np.zeros(nevents)
            df["c_nEvents"] = np.ones(nevents) * nevents_gen
            df["EventWeight"] = df["EventWeight"]*tree_weight
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
            df_train.to_hdf(output_folder+'/'+ss+'_train.h5', 'df', format='table' if (len(var_list)<=2000) else 'fixed')
            print("  "+output_folder+"/"+ss+"_train.h5 stored")
            df_test.to_hdf(output_folder+'/'+ss+'_test.h5', 'df', format='table' if (len(var_list)<=2000) else 'fixed')
            print("  "+output_folder+"/"+ss+"_test.h5 stored")
            df_val.to_hdf(output_folder+'/'+ss+'_val.h5', 'df', format='table' if (len(var_list)<=2000) else 'fixed')
            print("  "+output_folder+"/"+ss+"_val.h5 stored")
            print("  -------------------   ")


'''
def read_h5(folder,file_names):
    for fi, file_name in enumerate(file_names):
        #read hd5
        store = pd.HDFStore(folder+'numpy/'+fileNas[fi]+'.h5')
        df = store.select("df")
        print(df)
'''

#print("write")
print("VERY IMPORTANT! selection in rnp.root2array performs an OR of the different conditions!!!!")
write_h5(folder,"dataframes/v2_calo_AOD_2017/",sgn+bkg,test_split=0.2,val_split=0.2,tree_name="skim",counter_hist="c_nEvents",sel_cut ="")

###write_h5(folder,"dataframes/v0_calo_AOD/",sgn+bkg,test_split=0.2,tree_name="ntuple/tree",counter_hist="counter/c_nEvents",sel_cut ="HT>200",obj_sel_cut="")

#print "read"
#read_h5(output_folder,"")
