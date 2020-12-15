#!/bin/sh 
source /etc/profile.d/modules.sh 
export PATH=/nfs/dust/cms/user/heikenju/anaconda2/bin:$PATH 
cd /nfs/dust/cms/user/heikenju/ML_LLP/GraphNetJetTaggerCalo/ 
source activate /nfs/dust/cms/user/heikenju/anaconda2/envs/particlenet 
python /nfs/dust/cms/user/heikenju/ML_LLP/GraphNetJetTaggerCalo/condor_write_partnet/WZ_TuneCP5_13TeV-pythia8-v3/write_macro_1.py 
