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
#from dnn_functions import *
gROOT.ProcessLine('.L Objects.h' )
from ROOT import JetType, CaloJetType, MEtType, CandidateType, DT4DSegmentType, CSCSegmentType, PFCandidateType#, TrackType

# storage folder of the original root files
in_folder  = '/nfs/dust/cms/group/cms-llp/v3_calo_AOD_2018_skimAccept_unmerged/'
out_folder = '/nfs/dust/cms/group/cms-llp/dataframes_jh/v3_calo_AOD_2018/'#out dir for write_

in_convert = '/nfs/dust/cms/group/cms-llp/dataframes_jh/v3_calo_AOD_2018/'
out_convert = '/nfs/dust/cms/group/cms-llp/dataframes_jh/v3_calo_AOD_2018_jh_dnn_partnet/'

sgn = ['SUSY_mh400_pl1000_XL']
#sgn = ['SUSY_mh400_pl1000','SUSY_mh300_pl1000','SUSY_mh250_pl1000','SUSY_mh200_pl1000','SUSY_mh175_pl1000','SUSY_mh150_pl1000','SUSY_mh127_pl1000']
#sgn = []
#bkg = ['ZJetsToNuNu']
#bkg = ['ZJetsToNuNu','WJetsToLNu','VV','QCD','TTbar']
#bkg = ['ZJetsToNuNu']
#bkg = ['QCD']
#bkg = ['TTbar']
#bkg = ['WJetsToLNu']
#bkg = ['VV']
bkg = []

from samplesAOD2018 import *
from prepare_v2 import *

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
event_dict = {
'EventNumber' : -1,
'RunNumber' : -1,
'LumiNumber' : -1,
'EventWeight' : -9999.,
'isMC' : 0,
'isVBF' : 0,
'HT' : -1.,
'MEt.pt' : -1.,
'MEt.phi' : -9.,
'MEt.sign' : -9.,
'MinJetMetDPhi' : 999.,
'nCHSJets' : 0,
'nElectrons' : 0,
'nMuons' : 0,
'nPhotons' : 0,
'nTaus' : 0,
'nPFCandidates' : -1,
'nPFCandidatesTrack' : -1,
}

## Per-jet variables
nj = 10
jtype = ['Jets']

jdict = {
'pt' : -1.,
'eta' : -9.,
'phi' : -9.,
'mass' : -1.,
'nConstituents' : -1,
'nTrackConstituents' : -1,
'nSelectedTracks' : -1,
'nHadEFrac' : -1.,
'cHadEFrac' : -1.,
'ecalE' : -100.,
'hcalE' : -100.,
'muEFrac' : -1.,
'eleEFrac' : -1.,
'photonEFrac' : -1.,
'eleMulti' : -1,
'muMulti' : -1,
'photonMulti' : -1,
'cHadMulti' : -1,
'nHadMulti' : -1,
'nHitsMedian' : -1.,
'nPixelHitsMedian' : -1.,
'dRSVJet' : -100.,
'nVertexTracks' : -1,
'CSV' : -99.,
'SV_mass' : -100.,
#new
'nRecHitsEB' : -1,
'timeRecHitsEB' : -100.,
'timeRMSRecHitsEB' : -1.,
'energyRecHitsEB' : -1.,
'energyErrorRecHitsEB' : -1.,
'nRecHitsHB' : -1,
'timeRecHitsHB' : -100.,
'timeRMSRecHitsHB' : -1.,
'energyRecHitsHB' : -1.,
'energyErrorRecHitsHB' : -1.,    
'ptAllTracks' : -1.,
'ptAllPVTracks' : -1.,
'ptPVTracksMax' : -1.,
'nTracksAll' : -1,
'nTracksPVMax' : -1,
'medianIP2D' : -10000.,
#'medianTheta2D',#currently empty
'alphaMax' : -100.,
'betaMax' : -100.,
'gammaMax' : -100.,
'gammaMaxEM' : -100.,
'gammaMaxHadronic' : -100.,
'gammaMaxET' : -100.,
'minDeltaRAllTracks' : 999.,
'minDeltaRPVTracks' : 999.,
'dzMedian' : -9999.,
'dxyMedian' : -9999.,
'isGenMatched' : 0,
#new for signal
'isGenMatchedCaloCorr' : 0,
'isGenMatchedLLPAccept' : 0,
'isGenMatchedCaloCorrLLPAccept' : 0,
#for gen matching
'radiusLLP' : -1000.,
'zLLP' : -1000.,    
}

## Per-jet PF candidates variables
npf = 30
pftype = ['PFCandidates']
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
pfdict = {
'pt' : -1.,
'eta' : -9.,
'phi' : -9.,
#'mass' : -1.,
'energy' : -1.,
#'px' : -9999.,
#'py' : -9999.,
#'pz' : -9999.,
'jetIndex' : -1,
'pdgId' : 0,
'charge' : -99,
'isTrack' : 0,
#'hasTrackDetails',
'dxy' : -9999.,
'dz' : -9999.,
'POCA_x' : -9999.,
'POCA_y' : -9999.,
'POCA_z' : -9999.,
'POCA_phi' : -9., 
#'ptError' : -1.,
#'etaError' : -1.,
#'phiError' : -1.,
#'dxyError' : -99.,
#'dzError' : -99., 
'theta' : -9., 
#'thetaError' : -1.,
'chi2' : -1.,
'ndof' : -1,
'normalizedChi2' : -1., 
'nHits' : -1,
'nPixelHits' : -1,
'lostInnerHits' : -1,
}

jet_list_tree = []
pf_list_tree = []
for n in range(nj):
    for t in jtype:
        for v in jdict.keys():
            #print(v, jdict[v])
            jet_list_tree.append( (str(t)+"["+str(n)+"]."+v , jdict[v], 1) )

for n in range(nj):
    for t in jtype:
        for p in range(npf):
            for tp in pftype:
                for pv in pfdict.keys():
                    pf_list_tree.append( ( "Jet_"+str(n)+"_"+ str(tp) + "["+str(p)+"]."+pv , pfdict[pv], 1) )


event_list_tree = []
for e in event_dict.keys():
    event_list_tree.append( (e, event_dict[e], 1) )
var_list_tree = event_var_in_tree + jet_list_tree
#var_list_tree = jet_list_tree
var_list_tree += pf_list_tree
#print(var_list_tree)

convert_var_list_tree = []
for a in var_list_tree:
    if isinstance(a, tuple):
        b = (a[0].replace("s[","_").replace("].","_"), a[1], a[2])
        convert_var_list_tree.append(b)
    else:
        convert_var_list_tree.append(a)
if "is_signal" not in convert_var_list_tree:
    convert_var_list_tree.append("is_signal")
if "c_nEvents" not in convert_var_list_tree:
    convert_var_list_tree.append("c_nEvents")
if "SampleWeight" not in convert_var_list_tree:
    convert_var_list_tree.append("SampleWeight")

#exit()
#exit()
#print(var_list_tree[0])

'''
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
if "Jet_phi" not in var_list_per_jet:
    var_list_per_jet.append("Jet_phi")
'''
# # # # # # # # # # # # # 

variables = []
MEt = MEtType()
#CHSJets = JetType()


from write_pd_v2 import *
NCPUS   = 1
MEMORY  = 1500#1500 orig#2000#10000#tried 10 GB for a job killed by condor automatically
RUNTIME = 3600*4#4 #4 hours
def do_write(in_folder, out_folder, cols=var_list_tree):
    for a in sgn+bkg:
        for i, s in enumerate(samples[a]['files']):
            #calculate CMS weight
            if 'GluGluH2_H2ToSSTobbbb' in s:
                xs = 1
            else:
                xs = sample[s]['xsec']
            LUMI = 59690#2018 lumi with normtag, from pdvm2018 twiki
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
            if not(os.path.exists('/nfs/dust/cms/user/heikenju/ML_LLP/GraphNetJetTaggerCalo/condor_write_v2/'+s)):
                os.mkdir('/nfs/dust/cms/user/heikenju/ML_LLP/GraphNetJetTaggerCalo/condor_write_v2/'+s)

            max_n = 20000
            print("Max number of root files considered: ", max_n)
            #for n,f in enumerate([root_files[0]]):
            #for n, f in enumerate(root_files):
            max_loop = min(max_n,len(root_files))
            for n in range(max_loop):
                os.chdir('/nfs/dust/cms/user/heikenju/ML_LLP/GraphNetJetTaggerCalo/condor_write_v2/'+s+'/')
                print("Loop n. ", n)
                print(root_files[n])
                #for local test:
                #write_h5_v2(IN,OUT,root_files[n],xs,LUMI,cols,tree_name="skim",counter_hist="c_nEvents",sel_cut="")
                #exit()
                #write python macro
                with open('write_macro_'+str(n)+'.py', 'w') as fout:
                    fout.write('#!/usr/bin/env python \n')
                    fout.write('import os \n')
                    fout.write('import ROOT as ROOT \n')
                    fout.write('import sys \n')
                    fout.write('sys.path.insert(0, "/nfs/dust/cms/user/heikenju/ML_LLP/GraphNetJetTaggerCalo/") \n')
                    fout.write('from write_pd_v2 import * \n')
                    fout.write('IN  = "'+IN+'" \n')
                    fout.write('OUT = "'+OUT+'" \n')
                    fout.write('xs = '+str(xs)+' \n')
                    fout.write('LUMI = '+str(LUMI)+' \n')
                    fout.write('cols = '+str(cols)+' \n')
                    fout.write(' \n')
                    fout.write('write_h5_v2(IN,OUT,"'+root_files[n]+'",xs,LUMI,cols,tree_name="skim",counter_hist="c_nEvents",sel_cut ="") \n')

                #From here now, to be fixed
                with open('job_write_'+str(n)+'.sh', 'w') as fout:
                    fout.write('#!/bin/sh \n')
                    fout.write('source /etc/profile.d/modules.sh \n')
                    fout.write('export PATH=/nfs/dust/cms/user/heikenju/anaconda2/bin:$PATH \n')
                    fout.write('cd /nfs/dust/cms/user/heikenju/ML_LLP/GraphNetJetTaggerCalo/ \n')
                    fout.write('source activate /nfs/dust/cms/user/heikenju/anaconda2/envs/particlenet \n')
                    fout.write('python /nfs/dust/cms/user/heikenju/ML_LLP/GraphNetJetTaggerCalo/condor_write_v2/'+s+'/write_macro_'+str(n)+'.py'  +' \n')
                os.system('chmod 755 job_write_'+str(n)+'.sh')
                ###os.system('sh job_skim_'+str(n)+'.sh')
        
                #write submit config
                with open('submit_write_'+str(n)+'.submit', 'w') as fout:
                    fout.write('executable   = /nfs/dust/cms/user/heikenju/ML_LLP/GraphNetJetTaggerCalo/condor_write_v2/'+s+'/job_write_'+ str(n) + '.sh \n')
                    fout.write('output       = /nfs/dust/cms/user/heikenju/ML_LLP/GraphNetJetTaggerCalo/condor_write_v2/'+s+'/out_write_'+ str(n) + '.txt \n')
                    fout.write('error        = /nfs/dust/cms/user/heikenju/ML_LLP/GraphNetJetTaggerCalo/condor_write_v2/'+s+'/error_write_'+ str(n) + '.txt \n')
                    fout.write('log          = /nfs/dust/cms/user/heikenju/ML_LLP/GraphNetJetTaggerCalo/condor_write_v2/'+s+'/log_write_'+ str(n) + '.txt \n')
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
                os.system('condor_submit /nfs/dust/cms/user/heikenju/ML_LLP/GraphNetJetTaggerCalo/condor_write_v2/'+s+'/submit_write_'+str(n)+'.submit' + ' \n')


def do_convert(inp,out,nj,npf,cols):
    for a in sgn+bkg:
        for i, s in enumerate(samples[a]['files']):
            print(s)

            IN = inp + s + '/'
            OUT = out + s +'/'
            if not(os.path.exists(OUT)):
                os.mkdir(OUT)
            all_files = [x for x in os.listdir(IN) if os.path.isfile(os.path.join(IN, x))]
        
            print("Prepare condor submission scripts")
            if not(os.path.exists('/nfs/dust/cms/user/heikenju/ML_LLP/GraphNetJetTaggerCalo/condor_conv_v2/'+s)):
                os.mkdir('/nfs/dust/cms/user/heikenju/ML_LLP/GraphNetJetTaggerCalo/condor_conv_v2/'+s)
            
            ##print("Pre root files: ", all_files)
            skim_list = []
            for f in all_files:
                skim_list.append(f[:-3])
            print("files? ", skim_list)
            ####root_files.remove(f)
            
            ##print("Post root files: ", skim_list)
            
            for n, f in enumerate(skim_list):
            #for n, f in enumerate([skim_list[0]]):
                ###convert_dataset_condor(IN,OUT,f)
                os.chdir('/nfs/dust/cms/user/heikenju/ML_LLP/GraphNetJetTaggerCalo/condor_conv_v2/'+s+'/')
                print("Loop n. ", n)
                print(f)
                
                #convert_dataset_v2(IN,OUT,f,nj,npf,cols)
                #exit()
                
                #write python macro
                with open('convert_macro_'+str(n)+'.py', 'w') as fout:
                    fout.write('#!/usr/bin/env python \n')
                    fout.write('import os \n')
                    #fout.write('import ROOT as ROOT \n')
                    fout.write('import sys \n')
                    fout.write('sys.path.insert(0, "/nfs/dust/cms/user/heikenju/ML_LLP/GraphNetJetTaggerCalo/") \n')
                    fout.write('from prepare_v2 import * \n')
                    fout.write('IN  = "'+IN+'" \n')
                    fout.write('OUT = "'+OUT+'" \n')
                    fout.write('nj = '+str(nj)+' \n')
                    fout.write('npf = '+str(npf)+' \n')
                    ###out.write('pf_dict = '+str(pf_dict)+' \n')
                    fout.write('cols = '+str(cols)+' \n')
                    fout.write(' \n')
                    fout.write('convert_dataset_v2(IN,OUT,"'+f+'",nj,npf,cols) \n')

                #From here now, to be fixed
                with open('job_convert_'+str(n)+'.sh', 'w') as fout:
                    fout.write('#!/bin/sh \n')
                    fout.write('source /etc/profile.d/modules.sh \n')
                    fout.write('export PATH=/nfs/dust/cms/user/heikenju/anaconda2/bin:$PATH \n')
                    fout.write('cd /nfs/dust/cms/user/heikenju/ML_LLP/GraphNetJetTaggerCalo/ \n')
                    fout.write('source activate /nfs/dust/cms/user/heikenju/anaconda2/envs/particlenet \n')
                    fout.write('python /nfs/dust/cms/user/heikenju/ML_LLP/GraphNetJetTaggerCalo/condor_conv_v2/'+s+'/convert_macro_'+str(n)+'.py'  +' \n')
                os.system('chmod 755 job_convert_'+str(n)+'.sh')
                ###os.system('sh job_convert_'+str(n)+'.sh')
    
                #write submit config
                with open('submit_convert_'+str(n)+'.submit', 'w') as fout:
                    fout.write('executable   = /nfs/dust/cms/user/heikenju/ML_LLP/GraphNetJetTaggerCalo/condor_conv_v2/'+s+'/job_convert_'+ str(n) + '.sh \n')
                    fout.write('output       = /nfs/dust/cms/user/heikenju/ML_LLP/GraphNetJetTaggerCalo/condor_conv_v2/'+s+'/out_convert_'+ str(n) + '.txt \n')
                    fout.write('error        = /nfs/dust/cms/user/heikenju/ML_LLP/GraphNetJetTaggerCalo/condor_conv_v2/'+s+'/error_convert_'+ str(n) + '.txt \n')
                    fout.write('log          = /nfs/dust/cms/user/heikenju/ML_LLP/GraphNetJetTaggerCalo/condor_conv_v2/'+s+'/log_convert_'+ str(n) + '.txt \n')
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
                os.system('condor_submit condor_conv_v2/'+s+'/submit_convert_'+str(n)+'.submit' + ' \n')
                


def do_merge(inp,out,cols,max_n_jets=10,train_split=0.8,test_split=0.2,val_split=0.2):
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
                files_list.append(f)

            
            startTime = time.time()
            df_list = []

            #max_n = len(files_list)
            max_n = 70
            max_loop = min(max_n,len(files_list))
            
            for n in range(max_loop):
                #store = pd.HDFStore(IN+f)
                print("Opening... ", IN+files_list[n])
                store = pd.HDFStore(IN+files_list[n])
                df = store.select("df",start=0,stop=-1)
                df_list.append(df[(df["Jet_pt"]>0) & (df["Jet_index"]<max_jetindex) & (df["EventWeight"]>0)  & (df["Jet_eta"]<1.48) & (df["Jet_eta"]>-1.48)])
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

            #here split train, test, val
            n_events_train = int(final_df.shape[0] * (1-test_split-val_split) )#0
            n_events_test = int(final_df.shape[0] * (test_split))#1
            n_events_val = int(final_df.shape[0] - n_events_train - n_events_test)#2
            print("Train: ", n_events_train)
            print("Test: ", n_events_test)
            print("Val: ", n_events_val)
            print("Tot: ", n_events_train+n_events_test+n_events_val)
            df_train = final_df.head(n_events_train)
            df_left  = final_df.tail(final_df.shape[0] - n_events_train)
            del final_df
            df_test = df_left.head(n_events_test)
            df_val  = df_left.tail(df_left.shape[0] - n_events_test)
            del df_left

            df_train.to_hdf(OUT+s+'_train.h5', 'df', format='table' if (len(cols)<=2000) else 'fixed')
            df_val.to_hdf(OUT+s+'_val.h5', 'df', format='table' if (len(cols)<=2000) else 'fixed')
            df_test.to_hdf(OUT+s+'_test.h5', 'df', format='table' if (len(cols)<=2000) else 'fixed')
            print("  "+OUT+s+"_train/val/test.h5 stored")

            del df_list
            del df_train
            del df_val
            del df_test


def do_mix_signal(type_dataset,folder,features,upsample_factor):
    #for a in bkg+sgn:
    IN  = folder+ '/'
    OUT = folder+'/'
    if not(os.path.exists(OUT)):
        os.mkdir(OUT)

    files_to_mix = []
    for b in sgn:
        for i, s in enumerate(samples[b]['files']):
            files_to_mix.append(s+"_"+type_dataset+".h5")
            print(s)

    startTime = time.time()
    df_list = []
    for n, f in enumerate(files_to_mix):
        store = pd.HDFStore(IN+f)
        df = store.select("df",start=0,stop=-1)
        df_list.append(df[features])
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
    if type_dataset=="test":
        df_s = df_s[ (df_s["Jet_pt"] >-1) & (df_s["EventWeight"]>0) ]
    else:
        print("Applying jet matching to train and validation... (Jet_isGenMatchedCaloCorrLLPAccept)")
        df_s = df_s[ (df_s["Jet_pt"] >-1) & (df_s["EventWeight"]>0) & (df_s["Jet_isGenMatchedCaloCorrLLPAccept"]==1) ]
    ##JJ
    #df_s = df_s[ (df_s["Jet_pt"] >-1) & (df_s["EventWeight"]>0) & (df_s["Jet_eta"]<1.48) & (df_s["Jet_eta"]>-1.48) & (df_s["Jet_timeRecHits"]>-99.) ]

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

def do_mix_background(type_dataset,folder,features):
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
        df_list.append(df[features])
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
    df_b = df_b[ (df_b["Jet_pt"] >-1) & (df_b["EventWeight"]>0) ]
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

def do_mix_s_b(type_dataset,folder,features,upsample_signal_factor=5,fraction_of_background=0.25):
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
        df_s = pd.concat([df_pre[features]] * upsample_signal_factor,ignore_index=True)
    else:
        df_s = df_pre[features]

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
    df_b = df_b[features]
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
    #print(df[ ["Jet_isGenMatched","is_signal"] ])
            
    df.to_hdf(OUT+type_dataset+'.h5', 'df', format='fixed')
    print("  Saving complete dataset: "+OUT+type_dataset+".h5 stored")
    del df

#do_write(in_folder, out_folder, cols=var_list_tree)
#do_convert(in_convert,out_convert,nj,npf,cols=convert_var_list_tree)
#do_merge(out_convert,out_convert,cols=convert_var_list_tree)
#exit()
pf_list = []
jet_list = []
event_list = []
for i,c in enumerate(convert_var_list_tree):
    if not isinstance(c, tuple):
        event_list.append(c.replace('.','_'))
for a in jdict.keys():
    jet_list.append("Jet_"+a)
if "Jet_index" not in jet_list:
    jet_list.append("Jet_index")
for n in range(npf):
    for a in pfdict.keys():
        pf_list.append(a+"_"+str(n))
#print(event_list)
#print(jet_list)
#print(pf_list)
#for a in ["train","test","val"]:
    #do_mix_background(a,out_convert,event_list+pf_list+["Jet_pt","Jet_eta","Jet_phi","Jet_index","Jet_isGenMatchedCaloCorrLLPAccept"])
#for a in ["test","val","train"]:
 #   do_mix_signal(a,out_convert,event_list+jet_list+pf_list+["Jet_pt","Jet_eta","Jet_phi","Jet_index","Jet_isGenMatchedCaloCorrLLPAccept"],upsample_factor=0)
for a in ["test","val","train"]:
    do_mix_s_b(a,out_convert,event_list+pf_list+jet_list+["Jet_pt","Jet_eta","Jet_phi","Jet_index","Jet_isGenMatchedCaloCorrLLPAccept"])
    
## mix considers feature names as a list!



