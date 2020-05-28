import ROOT
import os
import root_numpy as rnp
import numpy as np
import pandas as pd
import tables
import uproot
import time
import multiprocessing
from math import ceil
from ROOT import gROOT, TFile, TTree, TObject, TH1, TH1F, AddressOf, TLorentzVector
from dnn_functions import *
gROOT.ProcessLine('.L Objects.h' )
from ROOT import JetType, CaloJetType, MEtType, CandidateType, DT4DSegmentType, CSCSegmentType, PFCandidateType#, TrackType

# storage folder of the original root files
in_folder  = '/nfs/dust/cms/group/cms-llp/v2_calo_AOD_2017_unmerged/'
#out_folder = '/nfs/dust/cms/group/cms-llp/dataframes/v2_calo_AOD_2017_condor/'#first test
out_folder = '/nfs/dust/cms/group/cms-llp/dataframes/v2_calo_AOD_2017_condor_graphnet/'#out dir for write_
#graphnet_out_folder = '/nfs/dust/cms/group/cms-llp/dataframes_graphnet/v2_calo_AOD_2017_condor/'
graphnet_out_folder = '/nfs/dust/cms/group/cms-llp/dataframes_graphnet/v2_calo_AOD_2017_condor_graphnet/'#out dir for convert_
graphnet_out_folder_SMALL = '/nfs/dust/cms/group/cms-llp/dataframes_graphnet/v2_calo_AOD_2017_condor_SMALL/'
#sgn = ['ggH_MH1000_MS150_ctau1000']#,'ggH_MH1000_MS400_ctau1000'']
sgn = ['SUSY_mh400_pl1000','SUSY_mh300_pl1000','SUSY_mh250_pl1000','SUSY_mh200_pl1000','SUSY_mh175_pl1000','SUSY_mh150_pl1000','SUSY_mh127_pl1000']
#sgn = ['SUSY_mh250_pl1000','SUSY_mh200_pl1000',]
#sgn = ['SUSY_mh250_pl1000','SUSY_mh200_pl1000','SUSY_mh175_pl1000','SUSY_mh150_pl1000','SUSY_mh127_pl1000']
#sgn = ['SUSY_mh400_pl1000']
#bkg = ['VV']
bkg = ['ZJetsToNuNu']
#bkg = ['WJetsToLNu']
bkg = ['ZJetsToNuNu','WJetsToLNu','VV']
#bkg = ['QCD']
#bkg = ['ZJetsToNuNu','WJetsToLNu','VV','QCD']
bkg = []
#bkg = ['ZJetsToNuNu','QCD','VV']
#sgn = []

from samplesAOD2017 import *
from prepare_condor import *

# define your variables here

##
## INPUTS for write_
##
## - - - - - - - - - - - - - - -
## Event variables exsisting as tree branches in root files
## - - - - - - - - - - - - - - -
var_list_tree = []

event_var_in_tree = ['EventNumber',
                     'RunNumber','LumiNumber','EventWeight','isMC',
                     'isVBF','HT','MEt.pt','MEt.phi','MEt.sign','MinJetMetDPhi',
                     'nCHSJets',
                     'nElectrons','nMuons','nPhotons','nTaus','nPFCandidates','nPFCandidatesTrack',
                     #new for signal
                     #'nLLPInCalo',
                 ]

## Per-jet variables
nj = 10
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
        'isGenMatched',
        #new for signal
        'isGenMatchedCaloCorr', 'isGenMatchedLLPAccept',
        'isGenMatchedCaloCorrLLPAccept'
]

## Per-jet PF candidates variables
npf = 50#100#50#100#20
pftype = ['PFCandidate']
pfvar = ['pt','eta','phi',
         #'mass',
         'energy',
         #'px','py','pz',
         'jetIndex',
         'pdgId','isTrack',
         #'hasTrackDetails',
         'dxy', 'dz', 'POCA_x', 'POCA_y', 'POCA_z', 'POCA_phi', 
         #'ptError', 'etaError', 'phiError', 'dxyError', 'dzError', 
         'theta', 
         #'thetaError',
         'chi2', 'ndof', 'normalizedChi2', 
         'nHits', 'nPixelHits', 'lostInnerHits'
]

jet_list_tree = []
pf_list_tree = []
for n in range(nj):
    for t in jtype:
        for v in jvar:
            jet_list_tree.append(str(t)+"_"+str(n)+"."+v)
        for p in range(npf):
            for tp in pftype:
                for pv in pfvar:
                    pf_list_tree.append(str(t)+"_"+str(n)+"_"+str(tp)+"_"+str(p)+"."+str(pv))

var_list_tree = event_var_in_tree + jet_list_tree
var_list_tree += pf_list_tree

##
## INPUTS for convert_
##
## - - - - - - - - - - - - - - -
## Event variables exsisting after writing the h5
## - - - - - - - - - - - - - - -
var_list = []
event_list = []
#Rename "." --> "_"
for v in event_var_in_tree: event_list.append(v.replace('.','_'))
if "is_signal" not in event_list:
    event_list.append("is_signal")
if "c_nEvents" not in event_list:
    event_list.append("c_nEvents")
event_list += ['SampleWeight']
#SampleWeight is created in write function. Not existing in original tree (and not existing at the moment, to be added)
jet_list_h5 = []
pf_list_h5 = []
#jvar+= ["index"]
#Not available in original tree
for n in range(nj):
    for t in jtype:
        for v in jvar:
            jet_list_h5.append(str(t)+"_"+str(n)+"_"+v)
        for p in range(npf):
            for tp in pftype:
                for pv in pfvar:
                    pf_list_h5.append(str(t)+"_"+str(n)+"_"+str(tp)+"_"+str(p)+"_"+str(pv))

var_list+= event_list

#Here: specify if you want LEADER features or GraphNet features
var_list+= jet_list_h5
var_list+= pf_list_h5


##
## INPUTS for merge_
##
## - - - - - - - - - - - - - - -
## Per-jet variables
## - - - - - - - - - - - - - - -
var_list_per_jet = []
event_list = []
#Rename "." --> "_"
for v in event_var_in_tree: event_list.append(v.replace('.','_'))
if "is_signal" not in event_list:
    event_list.append("is_signal")
if "c_nEvents" not in event_list:
    event_list.append("c_nEvents")
if "SampleWeight" not in event_list:
    event_list.append("SampleWeight")
per_jet_list_h5 = []
per_pf_list_h5 = []
for t in jtype:
    for v in jvar:
        per_jet_list_h5.append(str(t)+"_"+v)

for p in range(npf):
    for pv in pfvar:
        per_pf_list_h5.append(str(pv)+"_"+str(p))

var_list_per_jet+= event_list

if "Jet_pt" not in var_list_per_jet:
    var_list_per_jet.append("Jet_pt")
if "Jet_index" not in var_list_per_jet:
    var_list_per_jet.append("Jet_index")
if "Jet_isGenMatched" not in var_list_per_jet:
    var_list_per_jet.append("Jet_isGenMatched")
if "Jet_eta" not in var_list_per_jet:
    var_list_per_jet.append("Jet_eta")


# # # # # # # # # # # # # 

variables = []
MEt = MEtType()
#CHSJets = JetType()


from write_pd_condor import *
NCPUS   = 1
MEMORY  = 8000#1500 orig#2000#10000#tried 10 GB for a job killed by condor automatically
RUNTIME = 3600*24#4 #4 hours

def do_write(in_folder, out_folder, cols=var_list_tree):
    for a in sgn+bkg:
        for i, s in enumerate(samples[a]['files']):
            #calculate CMS weight
            xs = sample[s]['xsec']
            LUMI = 1#41557#2017 lumi with normtag, from pdvm2017 twiki
            print(s, "xsec: ",xs)
            print("Being read in root file: ")
            #print(cols)
            #to be used in write_condor_h5

            IN = in_folder + s + '/'
            OUT = out_folder+s+'/'
            if not(os.path.exists(OUT)):
                os.mkdir(OUT)
            root_files = [x for x in os.listdir(IN) if os.path.isfile(os.path.join(IN, x))]
        
            print("Prepare condor submission scripts")
            if not(os.path.exists('condor/'+s)):
                os.mkdir('condor/'+s)
            #copy write_pd_condor to be sure it exists
            from shutil import copyfile
            #copyfile('/nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/write_pd_condor.py', 'condor/'+s+'/write_pd_condor.py')

            ##print(root_files)
            max_n = 20000
            print("Max number of root files considered: ", max_n)
            #for n,f in enumerate([root_files[0]]):
            #for n, f in enumerate(root_files):
            max_loop = min(max_n,len(root_files))
            for n in range(max_loop):#enumerate([root_files[0]]):
                os.chdir('condor/'+s+'/')
                print("Loop n. ", n)
                print(root_files[n])  
                #write_condor_h5(IN,OUT,f,xs,LUMI,test_split=0.2,val_split=0.2,tree_name="skim",counter_hist="c_nEvents",sel_cut ="")
            
                #write python macro
                with open('write_macro_'+str(n)+'.py', 'w') as fout:
                    fout.write('#!/usr/bin/env python \n')
                    fout.write('import os \n')
                    fout.write('import ROOT as ROOT \n')
                    fout.write('import sys \n')
                    fout.write('sys.path.insert(0, "/nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/") \n')
                    fout.write('from write_pd_condor import * \n')
                    fout.write('IN  = "'+IN+'" \n')
                    fout.write('OUT = "'+OUT+'" \n')
                    fout.write('xs = '+str(xs)+' \n')
                    fout.write('LUMI = '+str(LUMI)+' \n')
                    fout.write('cols = '+str(cols)+' \n')
                    fout.write(' \n')
                    fout.write('write_condor_h5(IN,OUT,"'+root_files[n]+'",xs,LUMI,cols,test_split=0.2,val_split=0.2,tree_name="skim",counter_hist="c_nEvents",sel_cut ="") \n')

                #From here now, to be fixed
                with open('job_write_'+str(n)+'.sh', 'w') as fout:
                    fout.write('#!/bin/sh \n')
                    fout.write('source /etc/profile.d/modules.sh \n')
                    fout.write('export PATH=/nfs/dust/cms/user/lbenato/anaconda2/bin:$PATH \n')
                    fout.write('cd /nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/ \n')
                    fout.write('source activate /nfs/dust/cms/user/lbenato/anaconda2/envs/particlenet \n')
                    fout.write('python /nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor/'+s+'/write_macro_'+str(n)+'.py'  +' \n')
                os.system('chmod 755 job_write_'+str(n)+'.sh')
                ###os.system('sh job_skim_'+str(n)+'.sh')
    
                #write submit config
                with open('submit_write_'+str(n)+'.submit', 'w') as fout:
                    fout.write('executable   = /nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor/'+s+'/job_write_'+ str(n) + '.sh \n')
                    fout.write('output       = /nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor/'+s+'/out_write_'+ str(n) + '.txt \n')
                    fout.write('error        = /nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor/'+s+'/error_write_'+ str(n) + '.txt \n')
                    fout.write('log          = /nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor/'+s+'/log_write_'+ str(n) + '.txt \n')
                    fout.write(' \n')
                    fout.write('#Requirements = OpSysAndVer == "CentOS7" \n')
                    fout.write('##Requirements = OpSysAndVer == "CentOS7" && CUDADeviceName == "GeForce GTX 1080 Ti" \n')
                    fout.write('#Request_GPUs = 1 \n')
                    fout.write(' \n')
                    fout.write('## uncomment this if you want to use the job specific variables $CLUSTER and $PROCESS inside your batchjob \n')
                    fout.write('##environment = "CLUSTER=$(Cluster) PROCESS=$(Process)" \n')
                    fout.write(' \n')
                    fout.write('## uncomment this to specify a runtime longer than 3 hours (time in seconds) \n')
                    fout.write('Request_Cpus = ' + str(NCPUS) + ' \n')
                    fout.write('Request_Memory = ' + str(MEMORY) + ' \n')
                    fout.write('+RequestRuntime = ' + str(RUNTIME) + ' \n')
                    fout.write('batch_name = w_'+s[:2]+str(n)+' \n')
                    fout.write('queue 1 \n')
            
                ##submit condor
                os.chdir('../../.')
                os.system('condor_submit condor/'+s+'/submit_write_'+str(n)+'.submit' + ' \n')
            ###os.chdir('../../.')



def do_convert(inp,out,nj,npf,event_list,cols):
    for a in sgn+bkg:
        for i, s in enumerate(samples[a]['files']):
            print(s)

            IN = inp + s + '/'
            OUT = out + s +'/'
            if not(os.path.exists(OUT)):
                os.mkdir(OUT)
            all_files = [x for x in os.listdir(IN) if os.path.isfile(os.path.join(IN, x))]
        
            print("Prepare condor submission scripts")
            if not(os.path.exists('condor_conv_partnet/'+s)):
                os.mkdir('condor_conv_partnet/'+s)
            
            ##print("Pre root files: ", all_files)
            skim_list = []
            for f in all_files:
                if ("train" in f):
                    skim_list.append(f[:-9])
            ####root_files.remove(f)
            
            ##print("Post root files: ", skim_list)
            
            for n, f in enumerate(skim_list):
            #for n, f in enumerate([skim_list[0]]):
                ###convert_dataset_condor(IN,OUT,f)
                os.chdir('condor_conv_partnet/'+s+'/')
                print("Loop n. ", n)
                print(f)  

                #convert_dataset_condor(IN,OUT,f,nj,npf,event_list,cols)
                #exit()
                #write python macro
                with open('convert_macro_'+str(n)+'.py', 'w') as fout:
                    fout.write('#!/usr/bin/env python \n')
                    fout.write('import os \n')
                    #fout.write('import ROOT as ROOT \n')
                    fout.write('import sys \n')
                    fout.write('sys.path.insert(0, "/nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/") \n')
                    fout.write('from prepare_condor import * \n')
                    fout.write('IN  = "'+IN+'" \n')
                    fout.write('OUT = "'+OUT+'" \n')
                    fout.write('nj = '+str(nj)+' \n')
                    fout.write('npf = '+str(npf)+' \n')
                    fout.write('event_list = '+str(event_list)+' \n')
                    fout.write('cols = '+str(cols)+' \n')
                    fout.write(' \n')
                    fout.write('convert_dataset_condor(IN,OUT,"'+f+'",nj,npf,event_list,cols) \n')

                #From here now, to be fixed
                with open('job_convert_'+str(n)+'.sh', 'w') as fout:
                    fout.write('#!/bin/sh \n')
                    fout.write('source /etc/profile.d/modules.sh \n')
                    fout.write('export PATH=/nfs/dust/cms/user/lbenato/anaconda2/bin:$PATH \n')
                    fout.write('cd /nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/ \n')
                    fout.write('source activate /nfs/dust/cms/user/lbenato/anaconda2/envs/particlenet \n')
                    fout.write('python /nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor_conv_partnet/'+s+'/convert_macro_'+str(n)+'.py'  +' \n')
                os.system('chmod 755 job_convert_'+str(n)+'.sh')
                ###os.system('sh job_convert_'+str(n)+'.sh')
    
                #write submit config
                with open('submit_convert_'+str(n)+'.submit', 'w') as fout:
                    fout.write('executable   = /nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor_conv_partnet/'+s+'/job_convert_'+ str(n) + '.sh \n')
                    fout.write('output       = /nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor_conv_partnet/'+s+'/out_convert_'+ str(n) + '.txt \n')
                    fout.write('error        = /nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor_conv_partnet/'+s+'/error_convert_'+ str(n) + '.txt \n')
                    fout.write('log          = /nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor_conv_partnet/'+s+'/log_convert_'+ str(n) + '.txt \n')
                    fout.write(' \n')
                    fout.write('#Requirements = OpSysAndVer == "CentOS7" \n')
                    fout.write('##Requirements = OpSysAndVer == "CentOS7" && CUDADeviceName == "GeForce GTX 1080 Ti" \n')
                    fout.write('#Request_GPUs = 1 \n')
                    fout.write(' \n')
                    fout.write('## uncomment this if you want to use the job specific variables $CLUSTER and $PROCESS inside your batchjob \n')
                    fout.write('##environment = "CLUSTER=$(Cluster) PROCESS=$(Process)" \n')
                    fout.write(' \n')
                    fout.write('## uncomment this to specify a runtime longer than 3 hours (time in seconds) \n')
                    fout.write('Request_Cpus = ' + str(NCPUS) + ' \n')
                    fout.write('Request_Memory = ' + str(MEMORY) + ' \n')
                    fout.write('+RequestRuntime = ' + str(RUNTIME) + ' \n')
                    fout.write('batch_name = '+s[:15]+str(n)+' \n')
                    fout.write('queue 1 \n')
            
                ##submit condor
                os.chdir('../../.')
                os.system('condor_submit condor_conv_partnet/'+s+'/submit_convert_'+str(n)+'.submit' + ' \n')

'''
jet_cols = []
for t in jtype:
    for v in jvar:
        jet_cols.append(str(t)+"_"+v)
'''

def do_merge(type_dataset,inp,out,cols,max_n_jets=10):    
    #for a in bkg+sgn:
    for a in sgn+bkg:
        if a in sgn:
            max_jetindex = max_n_jets
        else:
            max_jetindex = 1

        for i, s in enumerate(samples[a]['files']):
            print("\n")
            print(s)

            IN  = inp + s + '/'
            OUT = out +'/'
            if not(os.path.exists(OUT)):
                os.mkdir(OUT)
            all_files = [x for x in os.listdir(IN) if os.path.isfile(os.path.join(IN, x))]
        

            files_list = []
            for f in all_files:
                if (type_dataset in f):
                    files_list.append(f)

            if( len(files_list)==0 ):
                print("Type of dataset (train, test, val) not recognized, abort . . .")
                exit()
            #print(files_list)
            
            startTime = time.time()
            df_list = []

            max_n = len(files_list)
            #if (s=="ZJetsToNuNu_HT-200To400_13TeV-madgraph-v1" or s=="ZJetsToNuNu_HT-400To600_13TeV-madgraph-v1"):
            #    max_n = 300
            #    print("Problematic sample, too large!!!")
            #    print("Max number of root files considered: ", max_n)
            #max_n = 10
            max_loop = min(max_n,len(files_list))
            
            #for n, f in enumerate(files_list):
            #for n, f in enumerate([files_list[0]]):
            for n in range(max_loop):
                #store = pd.HDFStore(IN+f)
                store = pd.HDFStore(IN+files_list[n])
                df = store.select("df",start=0,stop=-1)
                df_list.append(df[cols][(df["Jet_pt"]>0) & (df["Jet_index"]<max_jetindex)])
                if(n % 100 == 0):
                    print("  * * * * * * * * * * * * * * * * * * * * * * *")
                    print("  Time needed to open file n. %s: %.2f seconds" % (str(n), time.time() - startTime))
                    print("  * * * * * * * * * * * * * * * * * * * * * * *")
                    print("\n")
                store.close()
                del df
                del store
                
                #print("\n")

            time_open = time.time()
            print("  * * * * * * * * * * * * * * * * * * * * * * *")
            print("  Time needed to open datasets: %.2f seconds" % (time.time() - startTime))
            print("  * * * * * * * * * * * * * * * * * * * * * * *")
            print("\n")
            final_df = pd.concat(df_list,ignore_index=True)
            final_df = final_df.loc[:,~final_df.columns.duplicated()]#remove duplicates
            print("  * * * * * * * * * * * * * * * * * * * * * * *")
            print("  Time needed to concatenate: %.2f seconds" % (time.time() - time_open))
            print("  * * * * * * * * * * * * * * * * * * * * * * *")
            print("\n")
            print(final_df)
            
            final_df.to_hdf(OUT+s+'_'+type_dataset+'.h5', 'df', format='table' if (len(cols)<=2000) else 'fixed')
            print("  "+OUT+s+"_"+type_dataset+".h5 stored")
            del final_df
            del df_list

def do_merge_partnet(type_dataset,inp,out,cols,max_n_jets=10):    
    #for a in bkg+sgn:
    for a in sgn+bkg:
        if a in sgn:
            max_jetindex = max_n_jets
        else:
            max_jetindex = 1

        for i, s in enumerate(samples[a]['files']):
            print("\n")
            print(s)

            IN  = inp + s + '/'
            OUT = out +'/'
            if not(os.path.exists(OUT)):
                os.mkdir(OUT)
            all_files = [x for x in os.listdir(IN) if os.path.isfile(os.path.join(IN, x))]
        

            files_list = []
            for f in all_files:
                if (type_dataset in f):
                    files_list.append(f)

            if( len(files_list)==0 ):
                print("Type of dataset (train, test, val) not recognized, abort . . .")
                exit()
            #print(files_list)
            
            startTime = time.time()
            df_list = []

            max_n = len(files_list)
            #if (s=="ZJetsToNuNu_HT-200To400_13TeV-madgraph-v1" or s=="ZJetsToNuNu_HT-400To600_13TeV-madgraph-v1"):
            #    max_n = 300
            #    print("Problematic sample, too large!!!")
            #    print("Max number of root files considered: ", max_n)
            max_n = 50#20
            max_loop = min(max_n,len(files_list))
            
            #for n, f in enumerate(files_list):
            #for n, f in enumerate([files_list[0]]):
            for n in range(max_loop):
                #store = pd.HDFStore(IN+f)
                store = pd.HDFStore(IN+files_list[n])
                df = store.select("df",start=0,stop=-1)
                #Already JJ preselections
                #cols+=["Jet_isGenMatchedCaloCorrLLPAccept"]
                df_list.append(df[cols][(df["Jet_pt"]>0) & (df["Jet_index"]<max_jetindex) & (df["EventWeight"]>0) & (df["Jet_eta"]<1.48) & (df["Jet_eta"]>-1.48) & (df["Jet_timeRecHits"]>-99.) & (df["MEt_pt"]>200) ])
                if(n % 100 == 0):
                    print("  * * * * * * * * * * * * * * * * * * * * * * *")
                    print("  Time needed to open file n. %s: %.2f seconds" % (str(n), time.time() - startTime))
                    print("  * * * * * * * * * * * * * * * * * * * * * * *")
                    print("\n")
                store.close()
                del df
                del store
                
                #print("\n")

            time_open = time.time()
            print("  * * * * * * * * * * * * * * * * * * * * * * *")
            print("  Time needed to open datasets: %.2f seconds" % (time.time() - startTime))
            print("  * * * * * * * * * * * * * * * * * * * * * * *")
            print("\n")
            final_df = pd.concat(df_list,ignore_index=True)
            print("  * * * * * * * * * * * * * * * * * * * * * * *")
            print("  Time needed to concatenate: %.2f seconds" % (time.time() - time_open))
            print("  * * * * * * * * * * * * * * * * * * * * * * *")
            print("\n")
            print(final_df)
            
            final_df.to_hdf(OUT+s+'_'+type_dataset+'.h5', 'df', format='table' if (len(cols)<=2000) else 'fixed')
            print("  "+OUT+s+"_"+type_dataset+".h5 stored")
            del final_df
            del df_list


def do_mix_background(type_dataset,folder,cols):
    #for a in bkg+sgn:
    IN  = folder+ '/'
    OUT = folder+'/'
    if not(os.path.exists(OUT)):
        os.mkdir(OUT)

    files_to_mix = []
    for b in bkg:
        for i, s in enumerate(samples[b]['files']):
            files_to_mix.append(s+"_"+type_dataset+".h5")

    startTime = time.time()
    df_list = []
    for n, f in enumerate(files_to_mix):
        print("Adding... ", f)
        store = pd.HDFStore(IN+f)
        df = store.select("df",start=0,stop=-1)
        df_list.append(df[cols])
        store.close()
        del df
        del store
                
    time_open = time.time()
    print("  * * * * * * * * * * * * * * * * * * * * * * *")
    print("  Time needed to open datasets: %.2f seconds" % (time.time() - startTime))
    print("  * * * * * * * * * * * * * * * * * * * * * * *")
    print("\n")

    df_b = pd.concat(df_list,ignore_index=True)
    print("  * * * * * * * * * * * * * * * * * * * * * * *")
    print("  Time needed to concatenate: %.2f seconds" % (time.time() - time_open))
    print("  * * * * * * * * * * * * * * * * * * * * * * *")
    print("\n")

    #Retain only valid jets and non negative weights
    df_b = df_b.loc[:,~df_b.columns.duplicated()]#remove duplicates
    ##df_b = df_b[  (df_b["EventWeight"]>0) ]
    ##df_b = df_b[ (df_b["Jet_pt"] >-1) & (df_b["EventWeight"]>0) & (df_b["Jet_index"]<1) ]
    ##JJ
    #already done in partnet
    #df_b = df_b[ (df_b["Jet_pt"] >-1) & (df_b["EventWeight"]>0) & (df_b["Jet_index"]<1) & (df_b["Jet_eta"]<1.48) & (df_b["Jet_eta"]>-1.48) & (df_b["Jet_timeRecHits"]>-99.) & (df_b["MEt_pt"]>200)]

    #Normalize later!
    
    #Shuffle later
    #df_b = df_b.sample(frac=1).reset_index(drop=True)
    print(df_b)
            
    df_b.to_hdf(OUT+'back_'+type_dataset+'.h5', 'df', format='fixed')
    print("  Saving full back dataset: "+OUT+"back_"+type_dataset+".h5 stored")
    del df_b
    del df_list

def do_mix_signal(type_dataset,folder,cols,upsample_factor=0):
    #for a in bkg+sgn:
    IN  = folder+ '/'
    OUT = folder+'/'
    if not(os.path.exists(OUT)):
        os.mkdir(OUT)

    files_to_mix = []
    for b in sgn:
        for i, s in enumerate(samples[b]['files']):
            files_to_mix.append(s+"_"+type_dataset+".h5")

    startTime = time.time()
    df_list = []
    for n, f in enumerate(files_to_mix):
        store = pd.HDFStore(IN+f)
        df = store.select("df",start=0,stop=-1)
        df_list.append(df[cols])
        store.close()
        del df
        del store
                
    time_open = time.time()
    print("  * * * * * * * * * * * * * * * * * * * * * * *")
    print("  Time needed to open datasets: %.2f seconds" % (time.time() - startTime))
    print("  * * * * * * * * * * * * * * * * * * * * * * *")
    print("\n")

    if upsample_factor>0:
        print("\n")
        print("   Upsampling signal by a factor: ", upsample_factor)
        print("\n")
        df_s = pd.concat(df_list * upsample_factor,ignore_index=True)
    else:
        df_s = pd.concat(df_list,ignore_index=True)
        
    print("  * * * * * * * * * * * * * * * * * * * * * * *")
    print("  Time needed to concatenate: %.2f seconds" % (time.time() - time_open))
    print("  * * * * * * * * * * * * * * * * * * * * * * *")
    print("\n")

    #Retain only valid jets and non negative weights, plus optional cuts if needed
    df_s = df_s.loc[:,~df_s.columns.duplicated()]#remove duplicates
    ##df_s = df_s[ (df_s["Jet_pt"] >-1) & (df_s["EventWeight"]>0) ]
    ##JJ
    df_s = df_s[ (df_s["Jet_pt"] >-1) & (df_s["EventWeight"]>0) & (df_s["Jet_eta"]<1.48) & (df_s["Jet_eta"]>-1.48) & (df_s["Jet_timeRecHits"]>-99.) ]

    ##Normalize weights
    #norm_s = df_s['EventWeight'].sum(axis=0)
    #df_s['EventWeightNormalized'] = df_s['EventWeight'].div(norm_s)

    #Shuffle later
    #df_s = df_s.sample(frac=1).reset_index(drop=True)
    print(df_s)
            
    df_s.to_hdf(OUT+'sign_'+type_dataset+'.h5', 'df', format='fixed')
    print("  Saving full sign dataset: "+OUT+"sign_"+type_dataset+".h5 stored")
    del df_s
    del df_list

def do_mix_signal_new(type_dataset,folder,cols,gen_matched,upsample_factor=0):
    if "Jet_isGenMatched" in cols:
        cols.remove("Jet_isGenMatched")
    if "Jet_isGenMatchedCaloCorrLLPAccept" not in cols:
        cols.append("Jet_isGenMatchedCaloCorrLLPAccept")
    print(cols)
    print(len(cols))
    #for a in bkg+sgn:
    IN  = folder+ '/'
    OUT = folder+'/'
    if not(os.path.exists(OUT)):
        os.mkdir(OUT)

    files_to_mix = []
    for b in sgn:
        for i, s in enumerate(samples[b]['files']):
            files_to_mix.append(s+"_"+type_dataset+".h5")

    startTime = time.time()
    df_list = []
    for n, f in enumerate(files_to_mix):
        store = pd.HDFStore(IN+f)
        df = store.select("df",start=0,stop=-1)
        df_list.append(df[cols])
        store.close()
        del df
        del store
                
    time_open = time.time()
    print("  * * * * * * * * * * * * * * * * * * * * * * *")
    print("  Time needed to open datasets: %.2f seconds" % (time.time() - startTime))
    print("  * * * * * * * * * * * * * * * * * * * * * * *")
    print("\n")

    if upsample_factor>0:
        print("\n")
        print("   Upsampling signal by a factor: ", upsample_factor)
        print("\n")
        df_s = pd.concat(df_list * upsample_factor,ignore_index=True)
    else:
        df_s = pd.concat(df_list,ignore_index=True)
        
    print("  * * * * * * * * * * * * * * * * * * * * * * *")
    print("  Time needed to concatenate: %.2f seconds" % (time.time() - time_open))
    print("  * * * * * * * * * * * * * * * * * * * * * * *")
    print("\n")

    #Retain only valid jets and non negative weights, plus optional cuts if needed
    ##df_s = df_s[ (df_s["Jet_pt"] >-1) & (df_s["EventWeight"]>0) ]
    df_s = df_s.loc[:,~df_s.columns.duplicated()]#remove duplicates
    ##JJ
    #df_s = df_s[ (df_s["Jet_pt"] >-1) & (df_s["EventWeight"]>0) & (df_s["Jet_eta"]<1.48) & (df_s["Jet_eta"]>-1.48) & (df_s["Jet_timeRecHits"]>-99.) & (df_s["MEt_pt"]>200)]
    df_s.rename(columns={"Jet_isGenMatchedCaloCorrLLPAccept": "Jet_isGenMatched"},inplace=True)
    df_s = df_s.loc[:,~df_s.columns.duplicated()]#remove duplicates
    if gen_matched:
        print("\n")
        print("  accept only gen matched jets!!! ")
        print("\n")
        df_s = df_s[ df_s["Jet_isGenMatched"]==1 ]

    ##Normalize weights
    #norm_s = df_s['EventWeight'].sum(axis=0)
    #df_s['EventWeightNormalized'] = df_s['EventWeight'].div(norm_s)

    #Shuffle later
    #df_s = df_s.sample(frac=1).reset_index(drop=True)
    print(df_s)

    df_s.to_hdf(OUT+'sign_'+type_dataset+'.h5', 'df', format='fixed')
    print("  Saving full sign dataset: "+OUT+"sign_"+type_dataset+".h5 stored")
    del df_s
    del df_list

def do_mix_s_b(type_dataset,folder,cols,upsample_signal_factor=0,fraction_of_background=1):
    #for a in bkg+sgn:
    IN  = folder+ '/'
    OUT = folder+'/'
    if not(os.path.exists(OUT)):
        os.mkdir(OUT)

    startTime = time.time()
    df_list = []

    store_s = pd.HDFStore(IN+"sign_"+type_dataset+".h5")
    df_pre = store_s.select("df",start=0,stop=-1)

    if upsample_signal_factor>1:
        print("Upsample signal by a factor: ", upsample_signal_factor)
        df_s = pd.concat([df_pre[cols]] * upsample_signal_factor,ignore_index=True)
    else:
        df_s = df_pre[cols]

    store_s.close()
    del store_s
    del df_pre

    #Normalize sgn weights after upsampling
    norm_s = df_s['EventWeight'].sum(axis=0)
    df_s['EventWeightNormalized'] = df_s['EventWeight'].div(norm_s)


    store_b = pd.HDFStore(IN+"back_"+type_dataset+".h5")
    stop = -1
    if fraction_of_background<1:
        print("   Using only this amount of bkg: ", fraction_of_background*100,"%")
        size = store_b.get_storer('df').shape[0]
        stop = int(size*fraction_of_background)

    df_b = store_b.select("df",start=0,stop=stop)
    df_b = df_b[cols]
    store_b.close()
    del store_b

    #Normalize bkg weights
    norm_b = df_b['EventWeight'].sum(axis=0)
    df_b['EventWeightNormalized'] = df_b['EventWeight'].div(norm_b)
                
    print("Ratio nB/nS: ", df_b.shape[0]/df_s.shape[0])

    df = pd.concat([df_s,df_b],ignore_index=True)
    df = df.loc[:,~df.columns.duplicated()]#remove duplicates
    del df_s, df_b
        
    print("  * * * * * * * * * * * * * * * * * * * * * * *")
    print("  Time needed to concatenate: %.2f seconds" % (time.time() - startTime))
    print("  * * * * * * * * * * * * * * * * * * * * * * *")
    print("\n")


    #Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    print(df[ ["Jet_isGenMatched","is_signal"] ])
            
    df.to_hdf(OUT+type_dataset+'.h5', 'df', format='fixed')
    print("  Saving full sign dataset: "+OUT+type_dataset+".h5 stored")
    del df
    

#do_write(in_folder, out_folder, cols=var_list_tree)
#exit()
write_folder = '/nfs/dust/cms/group/cms-llp/dataframes/v2_calo_AOD_2017_condor_graphnet/'#first test
out_leader = "/nfs/dust/cms/group/cms-llp/dataframes_graphnet/v2_calo_AOD_2017_condor_LEADER/"
out_JJ = "/nfs/dust/cms/group/cms-llp/dataframes_graphnet/v2_calo_AOD_2017_condor_JJ/"
out_partnet = "/nfs/dust/cms/group/cms-llp/dataframes_graphnet/v2_calo_AOD_2017_condor_partnet/"
out_partnet_JJ = "/nfs/dust/cms/group/cms-llp/dataframes_graphnet/v2_calo_AOD_2017_condor_partnet_JJ_presel/"
#do_convert(write_folder,out_leader,nj,npf,event_list,jet_list_h5)
# REMEBER TO MODIFY!!!
##NJ  = 1
##NPF = 50
#do_convert(write_folder,out_partnet_JJ,nj,npf,event_list,pf_list_h5)
#exit()

###out = "/nfs/dust/cms/group/cms-llp/dataframes_graphnet/v2_calo_AOD_2017_condor_graphnet/"
###do_convert(out_folder,out,nj,npf,event_list,pf_list_h5)
###exit()

#Here: specify if you want LEADER features or GraphNet features
#var_list_per_jet+= per_jet_list_h5
var_list_per_jet+= per_pf_list_h5

#print(var_list_per_jet)
#exit()

### These are faster operations, do not require condor:
'''
for a in ["train","test","val"]:      
#for a in ["test","val"]:
    print("Preparing: ", a)
    #do_merge(a,out_leader,out_leader,var_list_per_jet)
    #do_mix_background(a,out_leader,var_list_per_jet)
    #do_mix_signal_new(a,out_leader,var_list_per_jet,False if a=="test" else True, 0)
    do_mix_s_b(a,out_leader,var_list_per_jet,50 if a=="val" else 10,1.0)
    #do_mix_s_b(a,out_leader,var_list_per_jet,10,1.0)
'''


for a in ["train","test","val"]:      
#for a in ["val","test"]:
    print("Preparing: ", a)
    #do_merge_partnet(a,out_partnet_JJ,out_partnet_JJ,var_list_per_jet+["Jet_isGenMatchedCaloCorrLLPAccept"],5)#5 max jets this time
    #do_mix_background(a,out_partnet_JJ,var_list_per_jet)
    #do_mix_signal_new(a,out_partnet_JJ,var_list_per_jet,False if a=="test" else True, 0)
    ##do_mix_signal(a,out_partnet_JJ,var_list_per_jet,50)
    do_mix_s_b(a,out_partnet_JJ,var_list_per_jet,10,1)

