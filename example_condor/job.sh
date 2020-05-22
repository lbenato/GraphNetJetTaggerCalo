#!/bin/sh 
source /etc/profile.d/modules.sh 
export PATH=/nfs/dust/cms/user/<username>/anaconda2/bin:$PATH 
cd /nfs/dust/cms/user/<username>/ML_LLP/GraphNetJetTaggerCalo/ 
source activate /nfs/dust/cms/user/<username>/anaconda2/envs/particlenet 

python <path_here>/macro.py 
