#!/bin/sh 
source /etc/profile.d/modules.sh 
export PATH=/nfs/dust/cms/user/heikenju/anaconda2/bin:$PATH 
cd /nfs/dust/cms/user/heikenju/ML_LLP/GraphNetJetTaggerCalo/ 
source activate /nfs/dust/cms/user/heikenju/anaconda2/envs/particlenet 
python /nfs/dust/cms/user/heikenju/ML_LLP/GraphNetJetTaggerCalo/condor_write_v2/ZJetsToNuNu_HT-200To400_13TeV-madgraph-v1/write_macro_96.py 
