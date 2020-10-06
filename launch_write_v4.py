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
out_folder = '/nfs/dust/cms/group/cms-llp/dataframes_lisa/v3_calo_AOD_2018__v4/'#out dir for write_

in_convert = '/nfs/dust/cms/group/cms-llp/dataframes_lisa/v3_calo_AOD_2018__v4/'
out_convert = '/nfs/dust/cms/group/cms-llp/dataframes_lisa/v3_calo_AOD_2018_dnn__v4/'
OUT_XL = '/nfs/dust/cms/group/cms-llp/dataframes_lisa/v3_calo_AOD_2018_dnn__v4_XL/'
OUT_no = '/nfs/dust/cms/group/cms-llp/dataframes_lisa/v3_calo_AOD_2018_dnn__v4_no_upsampling/'

OUT  = '/nfs/dust/cms/group/cms-llp/dataframes_lisa/v3_calo_AOD_2018_dnn__v4_2018/'

sgn = [
    #'SUSY_mh400_pl1000','SUSY_mh300_pl1000','SUSY_mh250_pl1000','SUSY_mh200_pl1000','SUSY_mh175_pl1000','SUSY_mh150_pl1000','SUSY_mh127_pl1000',
    #'ggH_MH125_MS25_ctau500',  'ggH_MH125_MS25_ctau1000',  'ggH_MH125_MS25_ctau2000',  'ggH_MH125_MS25_ctau5000',  'ggH_MH125_MS25_ctau10000', 
    #'ggH_MH125_MS55_ctau500',  'ggH_MH125_MS55_ctau1000',  'ggH_MH125_MS55_ctau2000',  'ggH_MH125_MS55_ctau5000',  'ggH_MH125_MS55_ctau10000', 
    #'ggH_MH200_MS50_ctau500',  'ggH_MH200_MS50_ctau1000',  'ggH_MH200_MS50_ctau2000',  'ggH_MH200_MS50_ctau5000',  'ggH_MH200_MS50_ctau10000', 
    #'ggH_MH200_MS25_ctau500',  'ggH_MH200_MS25_ctau1000',  'ggH_MH200_MS25_ctau2000',  'ggH_MH200_MS25_ctau5000',  'ggH_MH200_MS25_ctau10000', 
    #'ggH_MH400_MS100_ctau500', 'ggH_MH400_MS100_ctau1000', 'ggH_MH400_MS100_ctau2000', 'ggH_MH400_MS100_ctau5000', 'ggH_MH400_MS100_ctau10000',
    #'ggH_MH400_MS50_ctau500',  'ggH_MH400_MS50_ctau1000',  'ggH_MH400_MS50_ctau2000',  'ggH_MH400_MS50_ctau5000',  'ggH_MH400_MS50_ctau10000',
    #'ggH_MH600_MS150_ctau500', 'ggH_MH600_MS150_ctau1000', 'ggH_MH600_MS150_ctau2000', 'ggH_MH600_MS150_ctau5000', 'ggH_MH600_MS150_ctau10000',
    #'ggH_MH600_MS50_ctau500',  'ggH_MH600_MS50_ctau1000',  'ggH_MH600_MS50_ctau2000',  'ggH_MH600_MS50_ctau5000',  'ggH_MH600_MS50_ctau10000',
    #'ggH_MH1000_MS150_ctau500','ggH_MH1000_MS150_ctau1000','ggH_MH1000_MS150_ctau2000','ggH_MH1000_MS150_ctau5000','ggH_MH1000_MS150_ctau10000',
    #'ggH_MH1000_MS400_ctau500','ggH_MH1000_MS400_ctau1000','ggH_MH1000_MS400_ctau2000','ggH_MH1000_MS400_ctau5000','ggH_MH1000_MS400_ctau10000',
    #'ggH_MH1500_MS200_ctau500','ggH_MH1500_MS200_ctau1000','ggH_MH1500_MS200_ctau2000','ggH_MH1500_MS200_ctau5000','ggH_MH1500_MS200_ctau10000',
    #'ggH_MH1500_MS500_ctau500','ggH_MH1500_MS500_ctau1000','ggH_MH1500_MS500_ctau2000','ggH_MH1500_MS500_ctau5000','ggH_MH1500_MS500_ctau10000',
    #'ggH_MH2000_MS250_ctau500','ggH_MH2000_MS250_ctau1000','ggH_MH2000_MS250_ctau2000','ggH_MH2000_MS250_ctau5000','ggH_MH2000_MS250_ctau10000',
    #'ggH_MH2000_MS600_ctau500','ggH_MH2000_MS600_ctau1000','ggH_MH2000_MS600_ctau2000','ggH_MH2000_MS600_ctau5000','ggH_MH2000_MS600_ctau10000',
]

sgn_2017 = [
    'SUSY_mh400_pl1000','SUSY_mh300_pl1000','SUSY_mh250_pl1000','SUSY_mh200_pl1000','SUSY_mh175_pl1000','SUSY_mh150_pl1000','SUSY_mh127_pl1000',
]

sgn = ['SUSY_mh400_pl1000_XL']


sgn_resolved = [
    'ggH_MH200_MS50_ctau500',  'ggH_MH200_MS50_ctau1000',  'ggH_MH200_MS50_ctau2000',  'ggH_MH200_MS50_ctau5000',  'ggH_MH200_MS50_ctau10000', 
    'ggH_MH400_MS100_ctau500', 'ggH_MH400_MS100_ctau1000', 'ggH_MH400_MS100_ctau2000', 'ggH_MH400_MS100_ctau5000', 'ggH_MH400_MS100_ctau10000',
    'ggH_MH600_MS150_ctau500', 'ggH_MH600_MS150_ctau1000', 'ggH_MH600_MS150_ctau2000', 'ggH_MH600_MS150_ctau5000', 'ggH_MH600_MS150_ctau10000',
    'ggH_MH1000_MS400_ctau500','ggH_MH1000_MS400_ctau1000','ggH_MH1000_MS400_ctau2000','ggH_MH1000_MS400_ctau5000','ggH_MH1000_MS400_ctau10000',
    'ggH_MH1500_MS500_ctau500','ggH_MH1500_MS500_ctau1000','ggH_MH1500_MS500_ctau2000','ggH_MH1500_MS500_ctau5000','ggH_MH1500_MS500_ctau10000',
    'ggH_MH2000_MS600_ctau500','ggH_MH2000_MS600_ctau1000','ggH_MH2000_MS600_ctau2000','ggH_MH2000_MS600_ctau5000','ggH_MH2000_MS600_ctau10000',
]

sgn_boosted = [
    ##200 not very boosted
    ##'ggH_MH200_MS25_ctau500',  'ggH_MH200_MS25_ctau1000',  'ggH_MH200_MS25_ctau2000',  'ggH_MH200_MS25_ctau5000',  'ggH_MH200_MS25_ctau10000', 
    #'ggH_MH400_MS50_ctau500',  'ggH_MH400_MS50_ctau1000',  'ggH_MH400_MS50_ctau2000',  'ggH_MH400_MS50_ctau5000',  'ggH_MH400_MS50_ctau10000',
    #'ggH_MH600_MS50_ctau500',  'ggH_MH600_MS50_ctau1000',  'ggH_MH600_MS50_ctau2000',  'ggH_MH600_MS50_ctau5000',  'ggH_MH600_MS50_ctau10000',
    'ggH_MH1000_MS150_ctau500','ggH_MH1000_MS150_ctau1000','ggH_MH1000_MS150_ctau2000','ggH_MH1000_MS150_ctau5000','ggH_MH1000_MS150_ctau10000',
    'ggH_MH1500_MS200_ctau500','ggH_MH1500_MS200_ctau1000','ggH_MH1500_MS200_ctau2000','ggH_MH1500_MS200_ctau5000','ggH_MH1500_MS200_ctau10000',
    'ggH_MH2000_MS250_ctau500','ggH_MH2000_MS250_ctau1000','ggH_MH2000_MS250_ctau2000','ggH_MH2000_MS250_ctau5000','ggH_MH2000_MS250_ctau10000',
]

#sgn = sgn_resolved
#print("Using resolved signal!!!")
sgn = sgn_boosted
print("Using boosted signal!!!")

#print(" Using SUSY + resolved Heavy Higgs for training!!! ")
#sgn += sgn_resolved
#print(sgn)

#sgn = []
bkg = ['ZJetsToNuNu','WJetsToLNu','VV','QCD','TTbar']
#bkg = ['ZJetsToNuNu']
#bkg = ['WJetsToLNu']
#bkg = ['QCD']
#bkg = ['TTbar']
#bkg = ['VV','TTbar']
#bkg = ['ZJetsToNuNu','WJetsToLNu','QCD']
#bkg = []

from samplesAOD2018 import *
from prepare_v4 import *

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
                     'nCHSJets','nCHSFatJets',
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
'nCHSFatJets' : 0,
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
'energy' : -1.,
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
#new
'nRecHitsEE' : -1,
'timeRecHitsEE' : -100.,
'timeRMSRecHitsEE' : -1.,
'energyRecHitsEE' : -1.,
'energyErrorRecHitsEE' : -1.,
#new
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


## Per-fat jet variables
nfj = 6
fjtype = ['FatJets']

fjdict = {
'pt' : -1.,
'energy' : -1.,
'eta' : -9.,
'phi' : -9.,
'mass' : -1.,
#AK8
'puppiTau21' : -1.,
'softdropPuppiMass' : -1.,
'nConstituents' : -1,
'nTrackConstituents' : -1,
#'nSelectedTracks' : -1,
'nHadEFrac' : -1.,
'cHadEFrac' : -1.,
#'ecalE' : -100.,
#'hcalE' : -100.,
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
#'dRSVJet' : -100.,
#'nVertexTracks' : -1,
#'CSV' : -99.,
#'SV_mass' : -100.,
#new
'nRecHitsEB' : -1,
'timeRecHitsEB' : -100.,
'timeRMSRecHitsEB' : -1.,
'energyRecHitsEB' : -1.,
'energyErrorRecHitsEB' : -1.,
#new
'nRecHitsEE' : -1,
'timeRecHitsEE' : -100.,
'timeRMSRecHitsEE' : -1.,
'energyRecHitsEE' : -1.,
'energyErrorRecHitsEE' : -1.,
#new
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
#'isGenMatchedLLPAccept' : 0,
#'isGenMatchedCaloCorrLLPAccept' : 0,
#for gen matching
'radiusLLP' : -1000.,
'zLLP' : -1000.,
}


## Per-jet PF candidates variables
npf = 50
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
fat_jet_list_tree = []
pf_list_tree = []
for n in range(nj):
    for t in jtype:
        for v in jdict.keys():
            #print(v, jdict[v])
            jet_list_tree.append( (str(t)+"["+str(n)+"]."+v , jdict[v], 1) )

for n in range(nfj):
    for t in fjtype:
        for v in fjdict.keys():
            #print(v, jdict[v])
            fat_jet_list_tree.append( (str(t)+"["+str(n)+"]."+v , fjdict[v], 1) )


for n in range(nj):
    for t in jtype:
        for p in range(npf):
            for tp in pftype:
                for pv in pfdict.keys():
                    pf_list_tree.append( ( "Jet_"+str(n)+"_"+ str(tp) + "["+str(p)+"]."+pv , pfdict[pv], 1) )


event_list_tree = []
for e in event_dict.keys():
    event_list_tree.append( (e, event_dict[e], 1) )
var_list_tree = event_var_in_tree + jet_list_tree + fat_jet_list_tree
#var_list_tree = jet_list_tree
#var_list_tree += pf_list_tree
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

# # # # # # # # # # # # # 

variables = []
MEt = MEtType()
#CHSJets = JetType()


from write_pd_v4 import *
NCPUS   = 1
MEMORY  = 1500#1500 orig#2000#10000#tried 10 GB for a job killed by condor automatically
RUNTIME = 3600*12#4 #4 hours
root_files_per_job = 50#50

def do_write(in_folder, out_folder, cols=var_list_tree):
    for a in sgn+bkg:
        for i, s in enumerate(samples[a]['files']):

            #print("Being read in root file: ")
            #print(cols)
            #to be used in write_condor_h5

            IN = in_folder + s + '/'
            OUT = out_folder+s+'/'
            if not(os.path.exists(OUT)):
                os.mkdir(OUT)
            root_files = [x for x in os.listdir(IN) if os.path.isfile(os.path.join(IN, x))]
       


            #Determine counter number
            oldFile = TFile(in_folder+s+"_counter.root", "READ")
            counter = oldFile.Get("c_nEvents").GetBinContent(1)
            oldFile.Close()

            #calculate CMS weight
            if 'GluGluH2_H2ToSSTobbbb' in s:
                xs = 1
            else:
                xs = sample[s]['xsec']
            LUMI = 59690#2018 lumi with normtag, from pdvm2018 twiki
            print('\n')
            print(s, "xsec: ",xs)
            print("counter: ", counter)
            
            print("Prepare condor submission scripts . . .")
            if not(os.path.exists('/nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor_write_v4/'+s)):
                os.mkdir('/nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor_write_v4/'+s)
                
                        
            ###Loop on root files
            max_n = 20000
            #print("Max number of root files considered: ", max_n)
            max_loop = min(max_n,len(root_files))
            print("Max number of root files: ", max_loop)
            print("Max root files per condor job: ", root_files_per_job)
            j_num = 0
            for b in range(0,max_loop,root_files_per_job):
                start = b
                stop = min(b+root_files_per_job-1,max_loop-1)
                #print("Start & stop: ", start, stop)
                print("Submitting job n. : ", j_num)
                os.chdir('/nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor_write_v4/'+s+'/')

                #PYTHON 
                with open('write_macro_'+str(j_num)+'.py', 'w') as fout:
                    fout.write('#!/usr/bin/env python \n')
                    fout.write('import os \n')
                    fout.write('import ROOT as ROOT \n')
                    fout.write('import sys \n')
                    fout.write('sys.path.insert(0, "/nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/") \n')
                    fout.write('from write_pd_v4 import * \n')
                    fout.write('IN  = "'+IN+'" \n')
                    fout.write('OUT = "'+OUT+'" \n')
                    fout.write('xs = '+str(xs)+' \n')
                    fout.write('LUMI = '+str(LUMI)+' \n')
                    fout.write('COUNT  = '+str(counter)+' \n')
                    fout.write('cols = '+str(cols)+' \n')
                    fout.write(' \n')
                    for c in np.arange(start,stop+1):
                        fout.write('write_h5_v4(IN,OUT,"'+root_files[c]+'",xs,LUMI,COUNT,cols,tree_name="skim",counter_hist="c_nEvents",sel_cut ="") \n')

                #BASH
                with open('job_write_'+str(j_num)+'.sh', 'w') as fout:
                    fout.write('#!/bin/sh \n')
                    fout.write('source /etc/profile.d/modules.sh \n')
                    fout.write('export PATH=/nfs/dust/cms/user/lbenato/anaconda2/bin:$PATH \n')
                    fout.write('cd /nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/ \n')
                    fout.write('source activate /nfs/dust/cms/user/lbenato/anaconda2/envs/particlenet \n')
                    fout.write('python /nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor_write_v4/'+s+'/write_macro_'+str(j_num)+'.py'  +' \n')
                os.system('chmod 755 job_write_'+str(j_num)+'.sh')

                #CONDOR
                with open('submit_write_'+str(j_num)+'.submit', 'w') as fout:
                    fout.write('executable   = /nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor_write_v4/'+s+'/job_write_'+ str(j_num) + '.sh \n')
                    fout.write('output       = /nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor_write_v4/'+s+'/out_write_'+ str(j_num) + '.txt \n')
                    fout.write('error        = /nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor_write_v4/'+s+'/error_write_'+ str(j_num) + '.txt \n')
                    fout.write('log          = /nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor_write_v4/'+s+'/log_write_'+ str(j_num) + '.txt \n')
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
                    fout.write('batch_name = w_'+s[:2]+str(j_num)+' \n')
                    fout.write('queue 1 \n')
               
                ##submit condor
                os.chdir('../../.')
                #os.system('condor_submit /nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor_write_v4/'+s+'/submit_write_'+str(j_num)+'.submit' + ' \n')
                os.system('python /nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor_write_v4/'+s+'/write_macro_'+str(j_num)+'.py' + ' \n')

                j_num +=1


def do_merge_and_split(inp,out,cols,max_n_jets=10,train_split=0.5,test_split=0.5):
    for a in sgn+bkg:
        for i, s in enumerate(samples[a]['files']):
            print("\n")
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

            max_n = len(files_list)
            #max_n = 300
            max_loop = min(max_n,len(files_list))
            
            for n in range(max_loop):
                #store = pd.HDFStore(IN+f)
                #print("Opening... ", IN+files_list[n])
                store = pd.HDFStore(IN+files_list[n])
                df = store.select("df",start=0,stop=-1)
                if(n % 500 == 0):
                    print("  * * * * * * * * * * * * * * * * * * * * * * *")
                    print("  Time needed to open file n. %s: %.2f seconds" % (str(n), time.time() - startTime))
                    print("  * * * * * * * * * * * * * * * * * * * * * * *")
                    print("\n")
                df_list.append(df)
                store.close()
                del df
                del store
                
                #print("\n")

            time_open = time.time()
            #print("  * * * * * * * * * * * * * * * * * * * * * * *")
            #print("  Time needed to open datasets: %.2f seconds" % (time.time() - startTime))
            #print("  * * * * * * * * * * * * * * * * * * * * * * *")
            #print("\n")
            final_df = pd.concat(df_list,ignore_index=True)
            final_df = final_df.loc[:,~final_df.columns.duplicated()]#remove duplicates
            print("  * * * * * * * * * * * * * * * * * * * * * * *")
            print("  Time needed to concatenate: %.2f seconds" % (time.time() - time_open))
            print("  * * * * * * * * * * * * * * * * * * * * * * *")
            print("\n")
            print(final_df)

            #New splitting, easier for other analysis teams: even and odd event number
            df_train = final_df[ (final_df["EventNumber"]%2 == 0) ]
            df_test = final_df[ (final_df["EventNumber"]%2 != 0) ]
            df_train["EventSplitRatio"] = np.ones(df_train.shape[0])*(df_train.shape[0]/final_df.shape[0])
            df_test["EventSplitRatio"] = np.ones(df_test.shape[0])*(df_test.shape[0]/final_df.shape[0])


            print("\n")
            print("% events for training: ", df_train.shape[0]/final_df.shape[0]*100)
            print("% events for testing: ", df_test.shape[0]/final_df.shape[0]*100)
            del final_df
            df_train.to_hdf(OUT+s+'_unconverted_train.h5', 'df', format='table' if (len(cols)<=2000) else 'fixed')
            df_test.to_hdf(OUT+s+'_unconverted_test.h5', 'df', format='table' if (len(cols)<=2000) else 'fixed')
            print("  "+OUT+s+"_unconverted_*.h5 stored")

            del df_list
            del df_train
            del df_test
            
            #here split train, test, val
            '''
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

            #Here: select only events with weight>0 and max jet index for train and validation
            df_train = df_train[ (df_train["EventWeight"]>0) ]
            df_val = df_val[ (df_val["EventWeight"]>0) ]
            print("Train with positive event weight: ", df_train.shape[0])
            print("Val with positive event weight: ", df_val.shape[0])

            df_train.to_hdf(OUT+s+'_unconverted_train.h5', 'df', format='table' if (len(cols)<=2000) else 'fixed')
            df_val.to_hdf(OUT+s+'_unconverted_val.h5', 'df', format='table' if (len(cols)<=2000) else 'fixed')
            df_test.to_hdf(OUT+s+'_unconverted_test.h5', 'df', format='table' if (len(cols)<=2000) else 'fixed')
            print("  "+OUT+s+"_unconverted_*.h5 stored")

            del df_list
            del df_train
            del df_val
            del df_test
            '''

def do_convert_from_split(inp,out,nj,nfj,npf,cols):
    for a in sgn+bkg:
        for i, s in enumerate(samples[a]['files']):
            #print(s)
            IN = inp
            OUT = out
            #print(IN)
            #print(OUT)
            #for a in ['train','val','test']:
            #    print(IN+s+'_unconverted_'+a+'.h5')
        
            print("Prepare condor submission scripts")
            if not(os.path.exists('/nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor_conv_v4/'+s)):
                os.mkdir('/nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor_conv_v4/'+s)
            
            #for n,a in enumerate(['train','val','test']):
            for n,a in enumerate(['train','test']):
                print("Convert ", a)
                f = s+'_unconverted_'+a
                os.chdir('/nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor_conv_v4/'+s+'/')

                #write python macro
                with open('convert_macro_'+str(n)+'.py', 'w') as fout:
                    fout.write('#!/usr/bin/env python \n')
                    fout.write('import os \n')
                    #fout.write('import ROOT as ROOT \n')
                    fout.write('import sys \n')
                    fout.write('sys.path.insert(0, "/nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/") \n')
                    fout.write('from prepare_v4 import * \n')
                    fout.write('IN  = "'+IN+'" \n')
                    fout.write('OUT = "'+OUT+'" \n')
                    fout.write('nj = '+str(nj)+' \n')
                    fout.write('nfj = '+str(nfj)+' \n')
                    fout.write('npf = '+str(npf)+' \n')
                    ###out.write('pf_dict = '+str(pf_dict)+' \n')
                    fout.write('cols = '+str(cols)+' \n')
                    fout.write(' \n')
                    fout.write('convert_dataset_v4(IN,OUT,"'+f+'",nj,nfj,npf,cols,"") \n')
                    fout.write('convert_dataset_AK8_v4(IN,OUT,"'+f+'",nj,nfj,npf,cols,"") \n')

                #Run without condor:
                #convert_dataset_v4(IN,OUT,f,nj,nfj,npf,cols,"")
                #convert_dataset_AK8_v4(IN,OUT,f,nj,nfj,npf,cols,"")
                #os.system('python /nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor_conv_v4/'+s+'/convert_macro_'+str(n)+'.py')

                #From here now, to be fixed
                with open('job_convert_'+str(n)+'.sh', 'w') as fout:
                    fout.write('#!/bin/sh \n')
                    fout.write('source /etc/profile.d/modules.sh \n')
                    fout.write('export PATH=/nfs/dust/cms/user/lbenato/anaconda2/bin:$PATH \n')
                    fout.write('cd /nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/ \n')
                    fout.write('source activate /nfs/dust/cms/user/lbenato/anaconda2/envs/particlenet \n')
                    fout.write('python /nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor_conv_v4/'+s+'/convert_macro_'+str(n)+'.py'  +' \n')
                os.system('chmod 755 job_convert_'+str(n)+'.sh')
                #os.system('sh job_convert_'+str(n)+'.sh')#!
    
                #write submit config
                with open('submit_convert_'+str(n)+'.submit', 'w') as fout:
                    fout.write('executable   = /nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor_conv_v4/'+s+'/job_convert_'+ str(n) + '.sh \n')
                    fout.write('output       = /nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor_conv_v4/'+s+'/out_convert_'+ str(n) + '.txt \n')
                    fout.write('error        = /nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor_conv_v4/'+s+'/error_convert_'+ str(n) + '.txt \n')
                    fout.write('log          = /nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor_conv_v4/'+s+'/log_convert_'+ str(n) + '.txt \n')
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
                os.system('condor_submit condor_conv_v4/'+s+'/submit_convert_'+str(n)+'.submit' + ' \n')



def do_convert(inp,out,nj,nfj,npf,cols):
    for a in sgn+bkg:
        for i, s in enumerate(samples[a]['files']):
            print(s)

            IN = inp + s + '/'
            OUT = out + s +'/'
            if not(os.path.exists(OUT)):
                os.mkdir(OUT)
            all_files = [x for x in os.listdir(IN) if os.path.isfile(os.path.join(IN, x))]
            print(IN)
            print(all_files)    
        
        
            print("Prepare condor submission scripts")
            if not(os.path.exists('/nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor_conv_v4/'+s)):
                os.mkdir('/nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor_conv_v4/'+s)
            
            ##print("Pre root files: ", all_files)
            skim_list = []
            for f in all_files:
                skim_list.append(f[:-3])
            print("files? ", skim_list)
            ####root_files.remove(f)
            
            ##print("Post root files: ", skim_list)


            ###Loop on root files
            max_n = 200000
            #print("Max number of root files considered: ", max_n)
            max_loop = min(max_n,len(skim_list))
            print("Max number of root files: ", max_loop)
            print("Max root files per condor job: ", root_files_per_job)
            j_num = 0
            for b in range(0,max_loop,root_files_per_job):
                start = b
                stop = min(b+root_files_per_job-1,max_loop-1)
                #print("Start & stop: ", start, stop)
                print("Submitting job n. : ", j_num)






            
            #test#for n, f in enumerate(skim_list):
            #for n, f in enumerate([skim_list[0]]):
                ###convert_dataset_condor(IN,OUT,f)
                os.chdir('/nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor_conv_v4/'+s+'/')
                print("Loop n. ", j_num)
                #test#print(f)
                
                #convert_dataset_v4(IN,OUT,f,"AK4jets",nj,nfj,npf,cols)
                #convert_dataset_v4(IN,OUT,f,"AK8jets",nj,nfj,npf,cols)
                #exit()
                
                #write python macro
                with open('convert_macro_'+str(j_num)+'.py', 'w') as fout:
                    fout.write('#!/usr/bin/env python \n')
                    fout.write('import os \n')
                    #fout.write('import ROOT as ROOT \n')
                    fout.write('import sys \n')
                    fout.write('sys.path.insert(0, "/nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/") \n')
                    fout.write('from prepare_v4 import * \n')
                    fout.write('IN  = "'+IN+'" \n')
                    fout.write('OUT = "'+OUT+'" \n')
                    fout.write('nj = '+str(nj)+' \n')
                    fout.write('nfj = '+str(nfj)+' \n')
                    fout.write('npf = '+str(npf)+' \n')
                    ###out.write('pf_dict = '+str(pf_dict)+' \n')
                    fout.write('cols = '+str(cols)+' \n')
                    fout.write(' \n')
                    for c in np.arange(start,stop+1):
                        fout.write('convert_dataset_v4(IN,OUT,"'+skim_list[c]+'","AK4jets",nj,nfj,npf,cols) \n')
                        fout.write('convert_dataset_v4(IN,OUT,"'+skim_list[c]+'","AK8jets",nj,nfj,npf,cols) \n')

                for c in np.arange(start,stop+1):
                    convert_dataset_v4(IN,OUT,skim_list[c],"AK4jets",nj,nfj,npf,cols)
                    convert_dataset_v4(IN,OUT,skim_list[c],"AK8jets",nj,nfj,npf,cols)
                #From here now, to be fixed
                
                with open('job_convert_'+str(j_num)+'.sh', 'w') as fout:
                    fout.write('#!/bin/sh \n')
                    fout.write('source /etc/profile.d/modules.sh \n')
                    fout.write('export PATH=/nfs/dust/cms/user/lbenato/anaconda2/bin:$PATH \n')
                    fout.write('cd /nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/ \n')
                    fout.write('source activate /nfs/dust/cms/user/lbenato/anaconda2/envs/particlenet \n')
                    fout.write('python /nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor_conv_v4/'+s+'/convert_macro_'+str(j_num)+'.py'  +' \n')
                os.system('chmod 755 job_convert_'+str(j_num)+'.sh')
                #os.system('sh job_convert_'+str(n)+'.sh')
    
                #write submit config
                with open('submit_convert_'+str(j_num)+'.submit', 'w') as fout:
                    fout.write('executable   = /nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor_conv_v4/'+s+'/job_convert_'+ str(j_num) + '.sh \n')
                    fout.write('output       = /nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor_conv_v4/'+s+'/out_convert_'+ str(j_num) + '.txt \n')
                    fout.write('error        = /nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor_conv_v4/'+s+'/error_convert_'+ str(j_num) + '.txt \n')
                    fout.write('log          = /nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/condor_conv_v4/'+s+'/log_convert_'+ str(j_num) + '.txt \n')
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
                    fout.write('batch_name = '+s[:15]+str(j_num)+' \n')
                    fout.write('queue 1 \n')
            
                
                ##submit condor
                os.chdir('../../.')
                #os.system('condor_submit condor_conv_v4/'+s+'/submit_convert_'+str(j_num)+'.submit' + ' \n')
                j_num +=1


def do_merge(inp,out,jet_type,cols,max_n_jets=10):
    for a in sgn+bkg:
        if a in sgn:
            max_jetindex = max_n_jets
        else:
            max_jetindex = 10

        for i, s in enumerate(samples[a]['files']):
            print("\n")
            print(s)

            IN  = inp + s + '/'
            OUT = out +'/'
            if not(os.path.exists(OUT)):
                os.mkdir(OUT)
            all_files = [x for x in os.listdir(IN) if os.path.isfile(os.path.join(IN, x))]
        

            files_list = []
            train_list = []
            test_list = []
            for f in all_files:
                files_list.append(f)
                if jet_type+"_train" in f:
                    train_list.append(f)
                elif jet_type+"_test" in f:
                    test_list.append(f)
            
            startTime = time.time()
            df_list_train = []
            df_list_test = []

            # TRAIN
            max_n = len(train_list)
            #max_n = 300
            max_loop = min(max_n,len(train_list))
            for n in range(max_loop):
                #store = pd.HDFStore(IN+f)
                #print("Opening... ", IN)
                store_train = pd.HDFStore(IN+train_list[n])
                df = store_train.select("df",start=0,stop=-1)
                #event weight
                df_list_train.append( df[(df["EventWeight"]>0)] )
                store_train.close()
                del df
                del store_train

            df_train = pd.concat(df_list_train,ignore_index=True)
            del df_list_train
            df_train = df_train.loc[:,~df_train.columns.duplicated()]#remove duplicates
            #Do a shuffling, otherwise jets are poorly sorted in pt
            df_train = df_train.sample(frac=1).reset_index(drop=True)
            df_train.to_hdf(OUT+s+'_'+jet_type+'_train.h5', 'df', format='table' if (len(cols)<=2000) else 'fixed')
            print("  * * * * * * * * * * * * * * * * * * * * * * *")
            print("  Time needed to prepare train: %.2f seconds" % (time.time() - startTime))
            print("  * * * * * * * * * * * * * * * * * * * * * * *")
            #print("\n")
            print(df_train.shape)
            print("  "+OUT+s+"_"+jet_type+"_train.h5 stored")
            del df_train


            # TEST
            max_n = len(test_list)
            #max_n = 300
            max_loop = min(max_n,len(test_list))
            timeTest = time.time()
            for n in range(max_loop):
                #store = pd.HDFStore(IN+f)
                #print("Opening... ", IN)
                store_test = pd.HDFStore(IN+test_list[n])
                df = store_test.select("df",start=0,stop=-1)
                #event weight
                df_list_test.append( df[(df["EventWeight"]>0)] )
                store_test.close()
                del df
                del store_test

            df_test = pd.concat(df_list_test,ignore_index=True)
            del df_list_test
            df_test = df_test.loc[:,~df_test.columns.duplicated()]#remove duplicates
            df_test = df_test.sample(frac=1).reset_index(drop=True)
            df_test.to_hdf(OUT+s+'_'+jet_type+'_test.h5', 'df', format='table' if (len(cols)<=2000) else 'fixed')
            print("  * * * * * * * * * * * * * * * * * * * * * * *")
            print("  Time needed to prepare test: %.2f seconds" % (time.time() - timeTest))
            print("  * * * * * * * * * * * * * * * * * * * * * * *")
            #print("\n")
            print(df_test.shape)
            print("  "+OUT+s+"_"+jet_type+"_test.h5 stored")
            del df_test

def do_mix_signal(type_dataset,type_jets,folder,out_folder,upsample_factor=0):
    #for a in bkg+sgn:
    IN  = folder+ '/'
    OUT = out_folder+'/'
    if not(os.path.exists(OUT)):
        os.mkdir(OUT)
    
    print("\n")
    print(" - - - - - - - - - - - - - - - - - - - - - - - - - ")
    print("Mixing dataset and jets: ", type_dataset,type_jets)

    files_to_mix = []
    for b in sgn:
        for i, s in enumerate(samples[b]['files']):
            #print(s+"_"+type_dataset+"_"+type_jets+".h5")
            files_to_mix.append(s+"_"+type_jets+"_"+type_dataset+".h5")
            #print(s)

    startTime = time.time()
    df_list = []
    for n, f in enumerate(files_to_mix):
        store = pd.HDFStore(IN+f)
        df = store.select("df",start=0,stop=-1)
        df_list.append(df)#[features])
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
    print("Only positive Event Weight!!!")
    df_s = df_s[ (df_s["EventWeight"]>0) ]

    #Already done
    #if type_dataset=="test":
    #    df_s = df_s[ (df_s["Jet_pt"] >-1) & (df_s["EventWeight"]>0) ]
    #else:
    #    print("Applying jet matching to train and validation... (Jet_isGenMatchedCaloCorrLLPAccept)")
    #    df_s = df_s[ (df_s["Jet_pt"] >-1) & (df_s["EventWeight"]>0) & (df_s["Jet_isGenMatchedCaloCorrLLPAccept"]==1) ]
    ##JJ
    #df_s = df_s[ (df_s["Jet_pt"] >-1) & (df_s["EventWeight"]>0) & (df_s["Jet_eta"]<1.48) & (df_s["Jet_eta"]>-1.48) & (df_s["Jet_timeRecHits"]>-99.) ]

    #Create column with AK8 proper gen matching
    if "AK8" in type_jets:
        #Select column "FatJet_isGenMatchedCaloCorr" and mask when LLP not in acceptance
        df_s["FatJet_isGenMatchedCaloCorrLLPAccept"] = df_s["FatJet_isGenMatchedCaloCorr"].mask( (df_s["FatJet_zLLP"]>=376.) | (df_s["FatJet_zLLP"]<=-376.) | (df_s["FatJet_radiusLLP"]>=184.) | (df_s["FatJet_radiusLLP"]<=30.) , 0 )

    #Gen matching for everything!
    #if type_dataset!="test":
    if "AK4" in type_jets:
        print("Applying jet matching to train and validation... (Jet_isGenMatchedCaloCorrLLPAccept)")
        df_s = df_s[ (df_s["Jet_pt"] >-1) & (df_s["EventWeight"]>0) & (df_s["Jet_isGenMatchedCaloCorrLLPAccept"]==1) ]
    elif "AK8" in type_jets:
        print("Applying jet matching to train and validation... (FatJet_isGenMatchedCaloCorrLLPAccept)")
        df_s = df_s[ (df_s["FatJet_pt"] >-1) & (df_s["EventWeight"]>0) & (df_s["FatJet_isGenMatchedCaloCorrLLPAccept"]==1)]

    ##Normalize weights
    #norm_s = df_s['EventWeight'].sum(axis=0)
    #df_s['EventWeightNormalized'] = df_s['EventWeight'].div(norm_s)

    #Shuffling never hurts
    df_s = df_s.sample(frac=1).reset_index(drop=True)
    print(df_s.shape)
            
    df_s.to_hdf(OUT+'sign_'+type_jets+'_'+type_dataset+'.h5', 'df', format='fixed')
    print("  Saving full sign dataset: "+OUT+"sign_"+type_jets+"_"+type_dataset+".h5 stored")
    del df_s
    del df_list


def do_mix_background(type_dataset,type_jets,folder,out_folder):
    #for a in bkg+sgn:
    IN  = folder+ '/'
    OUT = out_folder+'/'
    if not(os.path.exists(OUT)):
        os.mkdir(OUT)
    
    print("\n")
    print(" - - - - - - - - - - - - - - - - - - - - - - - - - ")
    print("Mixing dataset and jets: ", type_dataset,type_jets)

    files_to_mix = []
    for b in bkg:
        for i, s in enumerate(samples[b]['files']):
            files_to_mix.append(s+"_"+type_jets+"_"+type_dataset+".h5")

    startTime = time.time()
    df_list = []
    for n, f in enumerate(files_to_mix):
        print("Adding... ", f)
        store = pd.HDFStore(IN+f)
        if store.keys()==[]:
            print("Empty dataset! Skipping . . .")
            continue
            #continue goes to the next element of the loop, if the dataframe is empty
        df = store.select("df",start=0,stop=-1)
        df_list.append(df)#[features])
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

    #Retain only valid jets and non negative weights, plus optional cuts if needed
    df_b = df_b.loc[:,~df_b.columns.duplicated()]#remove duplicates
    print("Only positive Event Weight!!!")
    df_b = df_b[ (df_b["EventWeight"]>0) ]

    #Create column with AK8 proper gen matching
    if "AK8" in type_jets:
        #Select column "FatJet_isGenMatchedCaloCorr" and mask when LLP not in acceptance
        df_b["FatJet_isGenMatchedCaloCorrLLPAccept"] = df_b["FatJet_isGenMatchedCaloCorr"].mask( (df_b["FatJet_zLLP"]>=376.) | (df_b["FatJet_zLLP"]<=-376.) | (df_b["FatJet_radiusLLP"]>=184.) | (df_b["FatJet_radiusLLP"]<=30.) , 0 )


    ##Normalize weights
    #norm_s = df_b['EventWeight'].sum(axis=0)
    #df_b['EventWeightNormalized'] = df_b['EventWeight'].div(norm_s)

    #Shuffle: we might need only a part of background!
    df_b = df_b.sample(frac=1).reset_index(drop=True)
    print(df_b.shape)
            
    df_b.to_hdf(OUT+'back_'+type_jets+'_'+type_dataset+'.h5', 'df', format='fixed')
    print("  Saving full back dataset: "+OUT+"back_"+type_jets+"_"+type_dataset+".h5 stored")
    del df_b
    del df_list




def do_mix_background_old(type_dataset,folder,features):
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

def do_mix_s_b(type_dataset,type_jets,folder,out_folder,upsample_signal_factor=0,fraction_of_background=1):
    #for a in bkg+sgn:
    IN  = folder+ '/'
    OUT = out_folder+'/'
    if not(os.path.exists(OUT)):
        os.mkdir(OUT)

    startTime = time.time()
    df_list = []

    store_s = pd.HDFStore(IN+"sign_"+type_jets+"_"+type_dataset+".h5")
    df_s = store_s.select("df",start=0,stop=-1)

    #Shuffle
    df_s = df_s.sample(frac=1).reset_index(drop=True)
    store_s.close()
    del store_s
    #//#del df_pre

    if type_dataset == "train":
        val_split = 0.2
        print("   Preparing signal validation; 20% of training sample")
        n_s_train = int(df_s.shape[0] * (1-val_split) )#0
        n_s_val = int(df_s.shape[0] - n_s_train)#2
        print("   -> Train: ", n_s_train)
        print("   -> Val: ", n_s_val)
        df_s_val_pre   = df_s.tail(df_s.shape[0] - n_s_train)
        df_s_train_pre = df_s.head(n_s_train) 

        print('\n')
        if upsample_signal_factor>1:
            print("   Upsample signal by a factor: ", upsample_signal_factor)
            print("   --> before, train: ", df_s_train_pre.shape[0])
            df_s_train = pd.concat([df_s_train_pre] * upsample_signal_factor,ignore_index=True)
            print("   --> after, train: ", df_s_train.shape[0])
            print("   --> before, val: ", df_s_val_pre.shape[0])
            df_s_val = pd.concat([df_s_val_pre] * upsample_signal_factor,ignore_index=True)
            print("   --> after, val: ", df_s_val.shape[0])
        else:
            df_s_train = df_s_train_pre
            df_s_val = df_s_val_pre

        del df_s_train_pre
        del df_s_val_pre
        del df_s

        #Normalize sgn weights after upsampling:
        #TRAIN
        norm_s_train = df_s_train['EventWeight'].sum(axis=0)
        df_s_train.loc[:,'EventWeightNormalized'] = df_s_train['EventWeight'].div(norm_s_train)
        df_s_train.loc[:,'EventWeightNormFactor'] = np.ones(df_s_train.shape[0])*norm_s_train

        norm_sampl_s_train = df_s_train['SampleWeight'].sum(axis=0)
        df_s_train.loc[:,'SampleWeightNormalized'] = df_s_train['SampleWeight'].div(norm_sampl_s_train)
        df_s_train.loc[:,'SampleWeightNormFactor'] = np.ones(df_s_train.shape[0])*norm_sampl_s_train

        #Normalize sgn weights after upsampling:
        #VAL
        norm_s_val = df_s_val['EventWeight'].sum(axis=0)
        df_s_val.loc[:,'EventWeightNormalized'] = df_s_val['EventWeight'].div(norm_s_val)
        df_s_val.loc[:,'EventWeightNormFactor'] = np.ones(df_s_val.shape[0])*norm_s_val

        norm_sampl_s_val = df_s_val['SampleWeight'].sum(axis=0)
        df_s_val.loc[:,'SampleWeightNormalized'] = df_s_val['SampleWeight'].div(norm_sampl_s_val)
        df_s_val.loc[:,'SampleWeightNormFactor'] = np.ones(df_s_val.shape[0])*norm_sampl_s_val


    else:

        print('\n')
        if upsample_signal_factor>1:
            print("   Upsample signal by a factor: ", upsample_signal_factor)
            print("   --> before, train: ", df_s.shape[0])
            df_s_test = pd.concat([df_s] * upsample_signal_factor,ignore_index=True)
            print("   --> after, train: ", df_s_test.shape[0])
        else:
            df_s_test = df_s

        del df_s

        #Normalize sgn weights after upsampling
        norm_s = df_s_test['EventWeight'].sum(axis=0)
        df_s_test.loc[:,'EventWeightNormalized'] = df_s_test['EventWeight'].div(norm_s)
        df_s_test.loc[:,'EventWeightNormFactor'] = np.ones(df_s_test.shape[0])*norm_s

        norm_sampl_s = df_s_test['SampleWeight'].sum(axis=0)
        df_s_test.loc[:,'SampleWeightNormalized'] = df_s_test['SampleWeight'].div(norm_sampl_s)
        df_s_test.loc[:,'SampleWeightNormFactor'] = np.ones(df_s_test.shape[0])*norm_sampl_s

    store_b = pd.HDFStore(IN+"back_"+type_jets+"_"+type_dataset+".h5")
    stop = -1
    if fraction_of_background<1:
        print("   Using only this amount of bkg: ", fraction_of_background*100,"%")
        size = store_b.get_storer('df').shape[0]
        stop = int(size*fraction_of_background)

    df_b = store_b.select("df",start=0,stop=stop)
    df_b = df_b
    store_b.close()
    del store_b

    print('\n')
    if type_dataset == "train":
        val_split = 0.2
        print("   Preparing background validation; 20% of training sample")
        n_b_train = int(df_b.shape[0] * (1-val_split) )#0
        n_b_val = int(df_b.shape[0] - n_b_train)#2
        print("   -> Train: ", n_b_train)
        print("   -> Val: ", n_b_val)
        df_b_val   = df_b.tail(df_b.shape[0] - n_b_train)
        df_b_train = df_b.head(n_b_train) 

        #Normalize sgn weights after upsampling:
        #TRAIN
        norm_b_train = df_b_train['EventWeight'].sum(axis=0)
        df_b_train.loc[:,'EventWeightNormalized'] = df_b_train['EventWeight'].div(norm_b_train)
        df_b_train.loc[:,'EventWeightNormFactor'] = np.ones(df_b_train.shape[0])*norm_b_train

        norm_sampl_b_train = df_b_train['SampleWeight'].sum(axis=0)
        df_b_train.loc[:,'SampleWeightNormalized'] = df_b_train['SampleWeight'].div(norm_sampl_b_train)
        df_b_train.loc[:,'SampleWeightNormFactor'] = np.ones(df_b_train.shape[0])*norm_sampl_b_train

        #Normalize sgn weights after upsampling:
        #VAL
        norm_b_val = df_b_val['EventWeight'].sum(axis=0)
        df_b_val.loc[:,'EventWeightNormalized'] = df_b_val['EventWeight'].div(norm_b_val)
        df_b_val.loc[:,'EventWeightNormFactor'] = np.ones(df_b_val.shape[0])*norm_b_val

        norm_sampl_b_val = df_b_val['SampleWeight'].sum(axis=0)
        df_b_val.loc[:,'SampleWeightNormalized'] = df_b_val['SampleWeight'].div(norm_sampl_b_val)
        df_b_val.loc[:,'SampleWeightNormFactor'] = np.ones(df_b_val.shape[0])*norm_sampl_b_val
        print("   ---- Ratio nB/nS train: ", df_b_train.shape[0]/df_s_train.shape[0])
        print("   ---- Ratio nB/nS val: ", df_b_val.shape[0]/df_s_val.shape[0])

        df_train = pd.concat([df_s_train,df_b_train],ignore_index=True)
        df_train = df_train.loc[:,~df_train.columns.duplicated()]#remove duplicates
        del df_s_train, df_b_train

        df_val = pd.concat([df_s_val,df_b_val],ignore_index=True)
        df_val = df_val.loc[:,~df_val.columns.duplicated()]#remove duplicates
        del df_s_val, df_b_val

        #Shuffle
        df_train = df_train.sample(frac=1).reset_index(drop=True)
        df_train.to_hdf(OUT+type_jets+'_train.h5', 'df', format='fixed')
        del df_train

        df_val = df_val.sample(frac=1).reset_index(drop=True)
        df_val.to_hdf(OUT+type_jets+'_val.h5', 'df', format='fixed')
        del df_val

    else:
        #Normalize sgn weights after upsampling
        norm_b = df_b['EventWeight'].sum(axis=0)
        df_b.loc[:,'EventWeightNormalized'] = df_b['EventWeight'].div(norm_b)
        df_b.loc[:,'EventWeightNormFactor'] = np.ones(df_b.shape[0])*norm_b

        norm_sampl_b = df_b['SampleWeight'].sum(axis=0)
        df_b.loc[:,'SampleWeightNormalized'] = df_b['SampleWeight'].div(norm_sampl_b)
        df_b.loc[:,'SampleWeightNormFactor'] = np.ones(df_b.shape[0])*norm_sampl_b
        print("   ---- Ratio nB/nS test: ", df_b.shape[0]/df_s_test.shape[0])


        df = pd.concat([df_s_test,df_b],ignore_index=True)
        df = df.loc[:,~df.columns.duplicated()]#remove duplicates
        del df_s_test, df_b
        #Shuffle
        df = df.sample(frac=1).reset_index(drop=True)
        df.to_hdf(OUT+type_jets+'_'+type_dataset+'.h5', 'df', format='fixed')
        del df
        
    print("  * * * * * * * * * * * * * * * * * * * * * * *")
    print("  Time needed to concatenate: %.2f seconds" % (time.time() - startTime))
    print("  * * * * * * * * * * * * * * * * * * * * * * *")
    #print("\n")
            
    print("  Saving mixed dataset: "+OUT+type_jets+'_'+type_dataset+".h5 stored")

def do_mix_s_b_old(type_dataset,folder,features,upsample_signal_factor=0,fraction_of_background=1):
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
    print(df[ ["Jet_isGenMatched","is_signal"] ])
            
    df.to_hdf(OUT+type_dataset+'.h5', 'df', format='fixed')
    print("  Saving mixed dataset: "+OUT+type_dataset+".h5 stored")
    del df

#do_write(in_folder, out_folder, cols=var_list_tree)
#exit()
##do_merge_and_split(in_convert,out_convert,cols=var_list_tree)
###do_convert_from_split(out_convert,out_convert,nj,nfj,npf,cols=convert_var_list_tree)
#do_convert(in_convert,out_convert,nj,nfj,npf,cols=convert_var_list_tree)
#do_merge(out_convert,out_convert,"AK4jets",cols=convert_var_list_tree,max_n_jets=10)
#do_merge(out_convert,out_convert,"AK8jets",cols=convert_var_list_tree,max_n_jets=10)
#exit()
##for i,a in enumerate(["train","test","val"]):

print(sgn)
#exit()

upsampl = [10,10,20]
for i,a in enumerate(["train","test"]):
    #do_mix_signal(a,"AK4jets",out_convert,OUT_no,upsample_factor=0)
    do_mix_signal(a,"AK8jets",out_convert,OUT,upsample_factor=0)
    #do_mix_background(a,"AK4jets",out_convert,OUT_no)
    #do_mix_background(a,"AK8jets",out_convert,OUT)


    #FIX: only 2018 signal! otherwise large differences in HCAL
    #For AK4, SUSY XL only
    #do_mix_s_b(a,"AK4jets",OUT,OUT,5,0.25)
    #For AK4, SUSY XL only, no upsampling
    #do_mix_s_b(a,"AK4jets",OUT_no,OUT_no,1,0.25)
    #For AK8, Heavy Higgs upsampled
    #do_mix_s_b(a,"AK8jets",OUT,OUT,5,0.25)#try to keep 5
    do_mix_s_b(a,"AK8jets",OUT,OUT,20,0.25)#go up and check if it works better
    #do_mix_s_b(a,"AK8jets",OUT,OUT,20,0.25)



    #Best current results: upsampling 20, back 0.25, only SUSY without XL
    #do_mix_s_b(a,"AK4jets",OUT,OUT,10,0.50)
    #do_mix_s_b(a,"AK4jets",OUT_XL,OUT_XL,3,0.25)#0.5)#overfitting!!!
    #do_mix_s_b(a,"AK8jets",out_convert,out_convert,20,0.25)#0.5)

    ##high stat:
    #do_mix_s_b(a,"AK4jets",OUT,OUT,20,0.25)#0.5)
    #do_mix_s_b(a,"AK8jets",OUT,OUT,20,0.25)#0.5)
exit()
#do_convert(in_convert,out_convert,nj,npf,cols=convert_var_list_tree)

#OUT_NEW_PRESEL = '/nfs/dust/cms/group/cms-llp/dataframes_lisa/v3_calo_AOD_2018_dnn__v4/'
#do_merge(out_convert,OUT_NEW_PRESEL,cols=convert_var_list_tree)
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
78.9,27.1,162.8
upsampl = [10,10,20]
for i,a in enumerate(["train","test","val"]):
    do_mix_signal(a,OUT_NEW_PRESEL,event_list+jet_list,upsample_factor=0)
    do_mix_background(a,OUT_NEW_PRESEL,event_list+jet_list)
    do_mix_s_b(a,OUT_NEW_PRESEL,event_list+jet_list,upsampl[i],1)
