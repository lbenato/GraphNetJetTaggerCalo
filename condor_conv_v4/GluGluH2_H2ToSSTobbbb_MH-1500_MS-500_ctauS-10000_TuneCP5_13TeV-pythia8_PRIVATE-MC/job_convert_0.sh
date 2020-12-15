#!/bin/sh 
source /etc/profile.d/modules.sh 
export PATH=/nfs/dust/cms/user/heikenju/anaconda2/bin:$PATH 
cd /nfs/dust/cms/user/heikenju/ML_LLP/GraphNetJetTaggerCalo/ 
source activate /nfs/dust/cms/user/heikenju/anaconda2/envs/particlenet 
python /nfs/dust/cms/user/heikenju/ML_LLP/GraphNetJetTaggerCalo/condor_conv_v4/GluGluH2_H2ToSSTobbbb_MH-1500_MS-500_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/convert_macro_0.py 
