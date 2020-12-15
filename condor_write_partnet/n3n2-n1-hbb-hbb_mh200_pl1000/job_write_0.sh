#!/bin/sh 
source /etc/profile.d/modules.sh 
export PATH=/nfs/dust/cms/user/heikenju/anaconda2/bin:$PATH 
cd /nfs/dust/cms/user/heikenju/ML_LLP/GraphNetJetTaggerCalo/ 
source activate /nfs/dust/cms/user/heikenju/anaconda2/envs/particlenet 
python /nfs/dust/cms/user/heikenju/ML_LLP/GraphNetJetTaggerCalo/condor_write_partnet/n3n2-n1-hbb-hbb_mh200_pl1000/write_macro_0.py 
