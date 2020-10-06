from graphnet_tagger import *


## ## Configure parameters ## ##

#~~~~~~~~~~~~~~~~~~~~~~~~~
### Folders ###
#~~~~~~~~~~~~~~~~~~~~~~~~~
graphnet_pd_folder = '/nfs/dust/cms/group/cms-llp/dataframes_graphnet/v2_calo_AOD_2017_condor_SMALL/'#'dataframes_graphnet/v2_calo_AOD_2017_test/'
graphnet_result_folder = 'model_weights_graphnet/v2_calo_AOD_2017_condor_SMALL/'#'model_weights_graphnet/v2_calo_AOD_2017_test/'


#~~~~~~~~~~~~~~~~~~~~~~~~~
## Define variables we want to keep in the output root file
#~~~~~~~~~~~~~~~~~~~~~~~~~
var_list = []
event_list = [
            'EventNumber',
            'RunNumber','LumiNumber','EventWeight','isMC',
            #'isVBF','HT','MEt_pt','MEt_phi','MEt_sign','MinJetMetDPhi',
            'nCHSJets',
            #'nElectrons','nMuons','nPhotons','nTaus','nPFCandidates','nPFCandidatesTrack',
            'ttv','is_signal',
            ]

###########################
### Define jet features ###
###########################

nj=10

j_gen = ['isGenMatched']
j_nottrain = [
'pt','eta','phi','mass',
]

# These are the variables we will use for training
j_features = [
#Lisa, old:
#'nTrackConstituents','nSelectedTracks','nHadEFrac', 'cHadEFrac','ecalE','hcalE',
#'muEFrac','eleEFrac','photonEFrac',
#'eleMulti','muMulti','photonMulti','cHadMulti','nHadMulti',
#'nHitsMedian','nPixelHitsMedian',
#'dRSVJet', 'nVertexTracks',
##'CSV',
#'SV_mass',
###JiaJing uses only:
#'timeRecHitsEB', 
#'cHadEFrac', 'nHadEFrac', 'eleEFrac','photonEFrac',
#'gammaMaxET','minDeltaRPVTracks',

#new:
#'nRecHits', 'timeRecHits', #'timeRMSRecHits', 
#'energyRecHits', #'energyErrorRecHits',
#'ptAllTracks', 'ptAllPVTracks', 'ptPVTracksMax', 'nTracksAll', 'nTracksPVMax', 'medianIP2D',
#'alphaMax', 'betaMax', 'gammaMax', 'gammaMaxEM', 'gammaMaxHadronic', 'gammaMaxET', 'minDeltaRAllTracks', 'minDeltaRPVTracks',
#'dzMedian', 'dxyMedian',

#v3 variables include ECAL/HCAL recHits
# # # The best so far:
# # # # # # #
# # # # # # #
# # # # # # #
# # # # # # #
'nTrackConstituents','nSelectedTracks',

'timeRecHitsEB',#'timeRecHitsHB',
'energyRecHitsEB',#'energyRecHitsHB',
'nRecHitsEB',#'nRecHitsHB', 

'cHadEFrac', 'nHadEFrac', 'eleEFrac','photonEFrac',
'ptAllTracks', 'ptAllPVTracks',
#'ptPVTracksMax', 'nTracksAll', 'nTracksPVMax',#Si does not have these!

###'medianIP2D',#?
'alphaMax', 'betaMax', 'gammaMax', 'gammaMaxEM', 'gammaMaxHadronic', 'gammaMaxET',
'minDeltaRAllTracks','minDeltaRPVTracks',
###'dzMedian', 'dxyMedian', #wait for those
# # # # # # #
# # # # # # #
# # # # # # #
# # # # # # #

#variables for partnet + jet constituents
#'nTrackConstituents',#already embedded in pf constituents
#'cHadEFrac', 'nHadEFrac', 'eleEFrac','photonEFrac',#these are somehow already embedded in pf constituents
#'nSelectedTracks',
#'timeRecHitsEB','timeRecHitsHB','energyRecHitsEB','energyRecHitsHB','nRecHitsEB','nRecHitsHB', 
#'ptAllTracks', 'ptAllPVTracks', 'ptPVTracksMax', 'nTracksAll', 'nTracksPVMax',
#'alphaMax', 'betaMax', 'gammaMax', 'gammaMaxEM', 'gammaMaxHadronic', 'gammaMaxET', 'minDeltaRAllTracks','minDeltaRPVTracks',

#Here: test for v3__v4, find the most important variables
#'nTrackConstituents','nSelectedTracks',
#'timeRecHitsEB','timeRecHitsHB','energyRecHitsEB','energyRecHitsHB','nRecHitsEB','nRecHitsHB', 
#'cHadEFrac', 'nHadEFrac', 'eleEFrac','photonEFrac',
#'ptAllTracks', 'ptAllPVTracks', 'ptPVTracksMax', 'nTracksAll', 'nTracksPVMax',
###'medianIP2D',#?
#'alphaMax', 'betaMax', 'gammaMax', 'gammaMaxEM', 'gammaMaxHadronic', 'gammaMaxET',
#'minDeltaRAllTracks','minDeltaRPVTracks',
###'dzMedian', 'dxyMedian', #wait for those
]

#variables for tag NoMedian = all but medianIP2D,dzMedian,dxyMedian

'''
j_features = [
###JiaJing uses only:
'timeRecHitsEB', 
'cHadEFrac', 'nHadEFrac', 'eleEFrac','photonEFrac',
'gammaMaxET','minDeltaRPVTracks',
]
'''

jet_features_list = []
for f in j_features:
    jet_features_list.append("Jet_"+f)

# These are all the jet variables we want to save in the final output root file; needed to fully reconstruct the event
j_var = j_gen+j_features+j_nottrain
jet_list = []
for v in j_var:
    jet_list.append("Jet_"+v)

##############################
### Define FatJet features ###
##############################
fj_features = [
'nConstituents','nTrackConstituents',#'nSelectedTracks',
'timeRecHitsEB','timeRecHitsHB','energyRecHitsEB','energyRecHitsHB','nRecHitsEB','nRecHitsHB', 
'cHadEFrac', 'nHadEFrac', 'eleEFrac','photonEFrac',
'ptAllTracks', 'ptAllPVTracks', 'ptPVTracksMax', 'nTracksAll', 'nTracksPVMax',
'alphaMax', 'betaMax', 'gammaMax', 'gammaMaxEM', 'gammaMaxHadronic', 'gammaMaxET', 'minDeltaRAllTracks','minDeltaRPVTracks',
'puppiTau21',
#'softdropPuppiMass',
]
fat_jet_features_list = []
for f in fj_features:
    fat_jet_features_list.append("FatJet_"+f)


###################################
### Define PFCandidate features ###
###################################

# Number of pf candidates used in the model
npf=30 #npf are 100 at maximum

#PF features we have in the dataset but that we don't use for training
pf_nottrain = [
    #'energy',
    #'px','py','pz',
    #'pt',
    'pdgId',
    #'isTrack',
    'hasTrackDetails',
    #'dxy', 'dz',
    'POCA_x', 'POCA_y', 'POCA_z', 'POCA_phi',
    'ptError', 'etaError', 'phiError', 'dxyError', 'dzError', 'theta', 'thetaError','chi2', 'ndof', 'normalizedChi2',
    #'nHits', 'nPixelHits',
    'lostInnerHits',
]

#PF features used for training
pf_features = [
    'energy',
    #'px','py','pz',
    'pt',
    #'pdgId',
    'isTrack',
    #'hasTrackDetails',
    'dxy', 'dz',
    #'POCA_x', 'POCA_y', 'POCA_z', 'POCA_phi',
    #'ptError', 'etaError', 'phiError', 'dxyError', 'dzError', 'theta', 'thetaError','chi2', 'ndof', 'normalizedChi2',
    'nHits', 'nPixelHits',
    #'lostInnerHits',
]
pf_points = [
    'eta','phi',
]
pf_mask = [
    'pt',
]

pf_features_list = []
for n in range(npf):
    for f in pf_features:
        pf_features_list.append(f+str(n))

# These are all the pf variables we want to save in the final output root file; needed to fully reconstruct the event
pf_var = pf_features+pf_nottrain
pf_list = []
for n in range(npf):
    for v in pf_var:
        pf_list.append(v+str(n))
##print(pf_features_list)
##print(pf_list)


##Time stamp for saving model
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%d%b%Y_%H_%M_%S")
print("Time:", timestampStr)
print("\n")

## Number of output classes for classification
n_class=2



##compare_folder = 'model_weights_graphnet/compare_folder/'
##compare_models(["BDT","LEADER","particle_net_lite"],compare_folder,"is_signal",["SampleWeight","SampleWeight","SampleWeight"],use_weight=True,model_labels=["SampleWeight","1_SampleWeight","test"],signal_match_test=False,ignore_empty_jets_test=True)
#compare_folder = 'model_weights/v3_calo_AOD_2018_dnn_balance_val_train_new_presel/'
#compare_folder = 'compare_all/'
#compare_models(["BDT","FCN","particle_net","FCN_constituents","particle_net"],compare_folder,"is_signal",["SampleWeight","EventWeightNormalized","EventWeightNormalized","EventWeightNormalized","EventWeightNormalized"],use_weight=True,model_labels=["SampleWeight_NoMedian","2_EventWeightNormalized_NoMedian","08-22","15_EWN_rel","09-23"],plot_labels=["BDT","FCN","particle_net","FCN_constituents","particle_net + jets"],signal_match_test=True,ignore_empty_jets_test=True)
#compare_models(["BDT","FCN","FCN_constituents","particle_net"],compare_folder,"is_signal",["SampleWeight","EventWeightNormalized","EventWeightNormalized","EventWeightNormalized"],use_weight=True,model_labels=["SampleWeight_NoMedian_200epochs_patience50","2_EventWeightNormalized_NoMedian_Adam_ReLU_200epochs_patience200_batch_size_512_dropout_0p2","15_EWN_rel","09-23"],plot_labels=["BDT","FCN","FCN on PF","particle net"],signal_match_test=True,ignore_empty_jets_test=True)

#compare_folder = 'model_weights//v3_calo_AOD_2018_dnn__v4_20Upsampling_0p25Background_BUGFIX/SUSY/AK4jets/'
#compare_models(["FCN","FCN","FCN"],compare_folder,"is_signal",["EventWeightNormalized","EventWeightNormalized","EventWeightNormalized"],use_weight=True,model_labels=["2_EventWeightNormalized_NoMedian_Adam_ReLU_200epochs_patience200_batch_size_512_dropout_0p2","2_EventWeightNormalized_NoMedian_NoRecHitsEB_Adam_ReLU_200epochs_patience200_batch_size_512_dropout_0p2","2_EventWeightNormalized_NoMedian_NoRecHitsEBHB_Adam_ReLU_200epochs_patience200_batch_size_512_dropout_0p2"],plot_labels=["FCN","FCN no HCAL rec hits","FCN no rec hits"],signal_match_test=True,ignore_empty_jets_test=True)

compare_folder = 'model_weights/v3_calo_AOD_2018_dnn__v4_2018_5Upsampling_0p25Background/SUSY/AK4jets/'
#compare_models(["FCN","FCN","FCN","BDT","BDT","BDT"],compare_folder,"is_signal",["EventWeightNormalized","EventWeightNormalized","EventWeightNormalized","SampleWeight","SampleWeight","SampleWeight"],use_weight=True,model_labels=["2_EventWeightNormalized_NoMedian_Adam_ReLU_200epochs_patience200_batch_size_512_dropout_0p2","2_EventWeightNormalized_NoMedian_NoRecHitsHB_Adam_ReLU_200epochs_patience200_batch_size_512_dropout_0p2","2_EventWeightNormalized_NoMedian_NoRecHitsEBHB_Adam_ReLU_200epochs_patience200_batch_size_512_dropout_0p2","SampleWeight_NoMedian_200epochs_patience25","SampleWeight_NoMedian_NoRecHitsHB_200epochs_patience25","SampleWeight_NoMedian_NoRecHitsEBHB_200epochs_patience25"],plot_labels=["FCN","FCN no HCAL rec hits","FCN no rec hits","BDT","BDT no HCAL rec hits", "BDT no rec hits"],signal_match_test=True,ignore_empty_jets_test=True)

##FCN
compare_models(["FCN","FCN","FCN","FCN","FCN","FCN"],compare_folder,"is_signal",["EventWeightNormalized","EventWeightNormalized","EventWeightNormalized","EventWeightNormalized","EventWeightNormalized","EventWeightNormalized"],use_weight=True,model_labels=["2_EventWeightNormalized_NoMedian_Adam_ReLU_200epochs_patience200_batch_size_512_dropout_0p2","2_EventWeightNormalized_NoMedian_NoRecHitsHB_Adam_ReLU_200epochs_patience200_batch_size_512_dropout_0p2","2_EventWeightNormalized_NoMedian_NoRecHitsEBHB_Adam_ReLU_200epochs_patience200_batch_size_512_dropout_0p2","2_EventWeightNormalized_NoMedian_NoRecHitsHB_NoSi_Adam_ReLU_200epochs_patience200_batch_size_512_dropout_0p2","2_EventWeightNormalized_NoMedian_NoRecHitsHB_NoSi_OnlyECALtime_Adam_ReLU_200epochs_patience200_batch_size_512_dropout_0p2","1_EventWeightNormalized_NoMedian_NoRecHitsHB_NoSi_OnlyECALtime_Adam_ReLU_200epochs_patience200_batch_size_512_dropout_0p2"],plot_labels=["FCN","FCN no HCAL rec hits","FCN no rec hits","FCN no HCAL/Si","FCN EB time, no Si","FCN EB time, no Si, model 1"],signal_match_test=True,ignore_empty_jets_test=True)

#FCN & BDT
#compare_models(
#    ["FCN","FCN","FCN","FCN","FCN","BDT","BDT","BDT","BDT"],
#    compare_folder,
#    "is_signal",
#    ["EventWeightNormalized","EventWeightNormalized","EventWeightNormalized","EventWeightNormalized","EventWeightNormalized","SampleWeight","SampleWeight","SampleWeight","SampleWeight"],
#    use_weight=True,
#    model_labels=["2_EventWeightNormalized_NoMedian_Adam_ReLU_200epochs_patience200_batch_size_512_dropout_0p2","2_EventWeightNormalized_NoMedian_NoRecHitsHB_Adam_ReLU_200epochs_patience200_batch_size_512_dropout_0p2","2_EventWeightNormalized_NoMedian_NoRecHitsEBHB_Adam_ReLU_200epochs_patience200_batch_size_512_dropout_0p2","2_EventWeightNormalized_NoMedian_NoRecHitsHB_NoSi_Adam_ReLU_200epochs_patience200_batch_size_512_dropout_0p2","2_EventWeightNormalized_NoMedian_NoRecHitsHB_NoSi_OnlyECALtime_Adam_ReLU_200epochs_patience200_batch_size_512_dropout_0p2","SampleWeight_NoMedian_200epochs_patience25","SampleWeight_NoMedian_NoRecHitsHB_200epochs_patience25","SampleWeight_NoMedian_NoRecHitsEBHB_200epochs_patience25","SampleWeight_NoMedian_NoRecHitsHB_NoSi_OnlyECALtime200epochs_patience25"],
#    plot_labels=["FCN","FCN no HCAL rec hits","FCN no rec hits","FCN no HCAL/Si","FCN EB time, no Si","BDT","BDT no HCAL rec hits", "BDT no rec hits","BDT EB time, no Si"],
#    signal_match_test=True,
#    ignore_empty_jets_test=True
#)

exit()

graphnet_pd_folder = '/nfs/dust/cms/group/cms-llp/dataframes_graphnet/v2_calo_AOD_2017_condor_LEADER/'#'dataframes_graphnet/v2_calo_AOD_2017_test/'
graphnet_pd_BDT = '/nfs/dust/cms/group/cms-llp/dataframes_graphnet/v2_calo_AOD_2017_condor_BDT/'#'dataframes_graphnet/v2_calo_AOD_2017_t
graphnet_pd_JJ = '/nfs/dust/cms/group/cms-llp/dataframes_graphnet/v2_calo_AOD_2017_condor_JJ/'#'dataframes_graphnet/v2_calo_AOD_2017_t
graphnet_result_folder = 'model_weights_graphnet/v2_calo_AOD_2017_condor_LEADER/'#'model_weights_graphnet/v2_calo_AOD_2017_test/'
graphnet_result_folder = 'model_weights_graphnet/v2_calo_AOD_2017_condor_LEADER_JJ_preselections/'#'model_weights_graphnet/v2_calo_AOD_2017_test/'

graphnet_pd_partnet = '/nfs/dust/cms/group/cms-llp/dataframes_graphnet/v2_calo_AOD_2017_condor_partnet_JJ_presel/'#'dataframes_graphnet/v2_calo_AOD_2017_test/'
graphnet_result_partnet = 'model_weights_graphnet/v2_calo_AOD_2017_condor_partnet_JJ_presel/'#'model_weights_graphnet/v2_calo_AOD_2017_test/'

graphnet_pd_JJ_MET = '/nfs/dust/cms/group/cms-llp/dataframes_graphnet/v2_calo_AOD_2017_condor_JJ_MET/'
graphnet_result_folder = 'model_weights_graphnet/v2_calo_AOD_2017_condor_JJ_MET/'

folder_dnn_v3 = '/nfs/dust/cms/group/cms-llp/dataframes_lisa/v3_calo_AOD_2018_dnn_new_presel/'
folder_BDT_v3 = '/nfs/dust/cms/group/cms-llp/dataframes_lisa/v3_calo_AOD_2018_BDT_new_presel/'
result_v3 = 'model_weights/v3_calo_AOD_2018_dnn_balance_val_train_new_presel/'

folder_dnn_v3 = '/nfs/dust/cms/group/cms-llp/dataframes_lisa/v3_calo_AOD_2018_dnn_new_presel_50_50_also_neg_weights/'
result_v3 = 'model_weights/v3_calo_AOD_2018_dnn_balance_val_train_new_presel_50_50_also_neg_weights/'

#only positive weights, 50 training, 50 testing
folder_dnn_v3 = '/nfs/dust/cms/group/cms-llp/dataframes_lisa/v3_calo_AOD_2018_dnn_new_presel_50_50/'
result_v3 = 'model_weights/v3_calo_AOD_2018_dnn_balance_val_train_new_presel_50_50/'

##just for redoing the roc with correct gen matching and fair comparison
#folder_dnn_v3 = '/nfs/dust/cms/group/cms-llp/dataframes_lisa/v3_calo_AOD_2018_dnn_new_presel/'
#folder_BDT_v3 = '/nfs/dust/cms/group/cms-llp/dataframes_lisa/v3_calo_AOD_2018_BDT_new_presel/'
#result_v3 = 'model_weights/v3_calo_AOD_2018_dnn_balance_val_train_new_presel/'

### test for part net with jet features
#folder_test = '/nfs/dust/cms/group/cms-llp/dataframes_lisa/v3_calo_AOD_2018_dnn_partnet/'
#result_test = 'model_weights/test/'

#New dataset with AK4 and AK8
#jet_type = "AK8jets"
#jet_string = "FatJet_"
#jet_features_list = fat_jet_features_list
jet_type = "AK4jets"
jet_string = "Jet_"
#folder_dnn_v3 = '/nfs/dust/cms/group/cms-llp/dataframes_lisa/v3_calo_AOD_2018_dnn__v4/'+jet_type+'_'
#folder_BDT_v3 = '/nfs/dust/cms/group/cms-llp/dataframes_lisa/v3_calo_AOD_2018_BDT__v4/'+jet_type+'_'
#!#result_v3 = 'model_weights/v3_calo_AOD_2018_dnn__v4_20Upsampling_0p25Background/SUSY/'+jet_type+'/'
#result_v3 = 'model_weights/v3_calo_AOD_2018_dnn__v4_20Upsampling_0p25Background_BUGFIX/SUSY/'+jet_type+'/'

#more stat:
#folder_dnn_v3 = '/nfs/dust/cms/group/cms-llp/dataframes_lisa/v3_calo_AOD_2018_dnn__v4_XL/'+jet_type+'_'
#folder_BDT_v3 = '/nfs/dust/cms/group/cms-llp/dataframes_lisa/v3_calo_AOD_2018_BDT__v4_XL/'+jet_type+'_'
#result_v3 = 'model_weights/v3_calo_AOD_2018_dnn__v4_3Upsampling_0p25Background_XL/SUSY/'+jet_type+'/'
#more stat new attempt:
#folder_dnn_v3 = '/nfs/dust/cms/group/cms-llp/dataframes_lisa/v3_calo_AOD_2018_dnn__v4_XL_balance/'+jet_type+'_'
#folder_BDT_v3 = '/nfs/dust/cms/group/cms-llp/dataframes_lisa/v3_calo_AOD_2018_BDT__v4_XL_balance/'+jet_type+'_'
#result_v3 = 'model_weights/v3_calo_AOD_2018_dnn__v4_10Upsampling_0p25Background_XL_balance/SUSY/'+jet_type+'/'


#Fixed: only 2018 without mixing data eras
folder_dnn_v3 = '/nfs/dust/cms/group/cms-llp/dataframes_lisa/v3_calo_AOD_2018_dnn__v4_2018/'+jet_type+'_'
folder_BDT_v3 = '/nfs/dust/cms/group/cms-llp/dataframes_lisa/v3_calo_AOD_2018_BDT__v4_2018/'+jet_type+'_'
result_v3 = 'model_weights/v3_calo_AOD_2018_dnn__v4_2018_5Upsampling_0p25Background/SUSY/'+jet_type+'/'

#no upsampling, cross-check
#folder_dnn_v3 = '/nfs/dust/cms/group/cms-llp/dataframes_lisa/v3_calo_AOD_2018_dnn__v4_no_upsampling/'+jet_type+'_'
#folder_BDT_v3 = '/nfs/dust/cms/group/cms-llp/dataframes_lisa/v3_calo_AOD_2018_dnn__v4_no_upsampling/'+jet_type+'_'
#result_v3 = 'model_weights/v3_calo_AOD_2018_dnn__v4_no_upsampling_0p25Background/SUSY/'+jet_type+'/'


#~~~~~~~~~~~~~~~~~~~~~~~~~
## Signal and background samples, defined in samplesAOD201X.py
#~~~~~~~~~~~~~~~~~~~~~~~~~
#sgn = ['SUSY_mh400_pl1000','SUSY_mh300_pl1000','SUSY_mh250_pl1000','SUSY_mh200_pl1000','SUSY_mh175_pl1000','SUSY_mh150_pl1000','SUSY_mh127_pl1000']
#bkg = ['VV','WJetsToLNu','ZJetsToNuNu']
#bkg = ['WJetsToLNu']

##############################################################
### Here we need a switch between jet features and pf features
TRAIN_MODEL = "BDT"
TRAIN_MODEL = "FCN"
#TRAIN_MODEL = "FCN_constituents"
#TRAIN_MODEL = "particle_net_lite"
#TRAIN_MODEL = "particle_net"
#TRAIN_MODEL = "particle_net_jet"


if TRAIN_MODEL != "BDT":
    from tensorflow import keras
    import tensorflow as tf
    from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

    ## Create list of available GPUs:
    import GPUtil as gput

    gpus = gput.getGPUs()
    tf_list = tf.config.experimental.list_physical_devices('GPU')
    print("Number of available GPUs: {}".format(len(tf_list)))
    print(tf_list)
    print("\n")
    print(gpus[0].name)

    if len(tf_list) == 1:
        if gpus[0].name=='Tesla P100-PCIE-16GB':
            print("using single GPU setup on login node; use allow_growth")
            tf_gpus = tf.config.experimental.list_physical_devices('GPU')
            try:
                config = tf.compat.v1.ConfigProto()
                config.gpu_options.allow_growth = True
                tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config)) 

                #Toby's
                #tf.config.experimental.set_virtual_device_configuration(
                #    tf_gpus[0],
                #    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=args.memory)])
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)
        
        else:
            print("using single GPU setup")
    else:
        print("using multi-GPU setup")
    
        # search for GPUs with requested parameters:
        gpu_idx = -1
        max_mem = 0
    
        # take GPU with maximum available memory
        for gpu in gpus:
            if gpu.memoryFree >= args.memory and gpu.load <= args.maxload and max_mem < gpu.memoryFree:
                max_mem = gpu.memoryFree
                gpu_idx = gpu.id

        if gpu_idx == -1:
            sys.exit("Unable to allocate memory as required by user. Please rerun with new criteria!")
    
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(physical_devices[gpu_idx], 'GPU')

    '''
    ## Tensorflow allow_growth for jobs submitted in the login node
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config)) 
    '''


if TRAIN_MODEL == "FCN":
    print("\n")
    print("   Training FCN on jet features (FCN)    ")
    print("\n")
    print(jet_features_list)
    print(len(jet_features_list)," training features!")
    print("\n")
    
    #Test generator!
    #fit_generator("FCN", n_class, graphnet_pd_folder, graphnet_result_folder,0,[],jet_features_list,[],"Jet_isGenMatched","EventWeightNormalized",use_weight=True,n_epochs=50,batch_size=2000,patience_val=5,val_split=0.0,model_label="more_var_2_generator",ignore_empty_jets_train=True)

    #evaluate performances
    #evaluate_model("FCN", n_class, graphnet_pd_folder, graphnet_result_folder,0,[],jet_features_list,[],"Jet_isGenMatched","EventWeightNormalized",use_weight=True,n_batch_size=2000,model_label="more_var_2_generator",signal_match_test=True,ignore_empty_jets_test=True)


    #b_size=2000 #default
    b_size = 2048
    n_epochs = 200
    #b_size=4096

    #Improving architecture
    #Leaky Relu
    #name = "2_CASTFLOAT_EventWeightNormalized_NoMedian_Adam_LeakyReLU_"+str(n_epochs)+"epochs_patience200_batch_size_"+str(b_size)+"_dropout_0p2"

    #input normalization

    #less dropout
    #name = "2_CASTFLOAT_EventWeightNormalized_NoMedian_Adam_ReLU_"+str(n_epochs)+"epochs_patience200_batch_size_"+str(b_size)+"_dropout_0p1"

    #batch norm first layer
    #name = "2_CASTFLOAT_EventWeightNormalized_NoMedian_Adam_LayerNorm_LeakyReLU_"+str(n_epochs)+"epochs_patience200_batch_size_"+str(b_size)+"_dropout_0p1"
    b_size = 2048
    b_size=1024
    b_size = 512 #the best
    n_epochs = 200 #faster
    #name = "2_CASTFLOAT_EventWeightNormalized_NoMedian_Adam_EACHLAYER_LayerNorm_ReLU_"+str(n_epochs)+"epochs_patience200_batch_size_"+str(b_size)+"_NO_dropout"
    #name = "2_CASTFLOAT_EventWeightNormalized_NoMedian_Adam_EACHLAYER_BatchNorm_ReLU_"+str(n_epochs)+"epochs_patience200_batch_size_"+str(b_size)+"_dropout_0p2"
    #Try model 1!!
    #name = "1_CASTFLOAT_EventWeightNormalized_NoMedian_Adam_EACHLAYER_LayerNorm_ReLU_"+str(n_epochs)+"epochs_patience200_batch_size_"+str(b_size)+"_dropout_0p2"# --> best 1
    #name = "1_CASTFLOAT_EventWeightNormalized_NoMedian_Adam_EACHLAYER_LayerNorm_ReLU_"+str(n_epochs)+"epochs_patience200_monitor_val_acc_batch_size_"+str(b_size)+"_dropout_0p2"
    #name = "1_CASTFLOAT_EventWeightNormalized_NoMedian_Adam_EACHLAYER_LayerNorm_ReLU_"+str(n_epochs)+"epochs_patience200_batch_size_"+str(b_size)+"_dropout_0p2_StandardScaler"


    #Without layer norm
    name = "1_EventWeightNormalized_NoMedian_NoRecHitsHB_NoSi_OnlyECALtime_Adam_ReLU_"+str(n_epochs)+"epochs_patience200_batch_size_"+str(b_size)+"_dropout_0p2"
    #name = "2_EventWeightNormalized_NoMedian_Adam_ReLU_"+str(n_epochs)+"epochs_patience200_batch_size_"+str(b_size)+"_dropout_0p2"
    #name = "2_EventWeightNormalized_NoMedian_Adam_LeakyReLU_"+str(n_epochs)+"epochs_patience200_batch_size_"+str(b_size)+"_dropout_0p2"


    #name = "2_CASTFLOAT_EventWeightNormalized_NoMedian_Adam_EACHLAYER_LayerNorm_ReLU_"+str(n_epochs)+"epochs_patience200_batch_size_"+str(b_size)+"_dropout_0p2"
    #name = "2_CASTFLOAT_EventWeightNormalized_NoMedian_Adam_EACHLAYER_LayerNorm_ReLU_"+str(n_epochs)+"epochs_patience200_batch_size_"+str(b_size)+"_dropout_0p2_StadardScaler"


    #Train longer
    #name = "2_EventWeightNormalized_NoMedian_Adam_200epochs_patience200_batch_size_"+str(b_size)
    #name = "2_EventWeightNormalized_NoMedian_Adamax_200epochs_patience200_batch_size_"+str(b_size)
    #name = "1_CASTFLOAT_EventWeightNormalized_NoMedian_Adamax_200epochs_patience200_batch_size_"+str(b_size)
    #name = "thumb_CASTFLOAT_EventWeightNormalized_NoMedian_Adam_200epochs_patience200_batch_size_"+str(b_size)+"_dropout_0p3"
    #name = "2_CASTFLOAT_EventWeightNormalized_NoMedian_Adam_"+str(n_epochs)+"epochs_patience200_batch_size_"+str(b_size)+"_dropout_0p2"
    #name = "2_CASTFLOAT_EventWeightNormalized_CaloHitsAndEnergyFractionsAndXMax_Adam_"+str(n_epochs)+"epochs_patience200_batch_size_"+str(b_size)+"_dropout_0p2"

    #name = "2_SampleWeight_NoMedian_Adam_patience40_batch_size_"+str(b_size)
    #name = "2_EventWeightNormalized_NoMedian_Adam_patience40_batch_size_"+str(b_size)

    #name = "2_EventWeightNormalized_NoMedian_Adam_patience20_lr_0p0001_batch_size_"+str(b_size)
    #name = "2_EventWeightNormalized_NoMedian_Nadam_patience20_batch_size_"+str(b_size)
    #name = "2_EventWeightNormalized_NoMedian_Adadelta_patience20_lr_0p0001_batch_size_"+str(b_size)
    #name = "2_EventWeightNormalized_NoMedian_Adamax_patience20_batch_size_4096"
    #name = "2_EventWeightNormalized_NoMedian_Adamax_patience20_lr_0p0001_batch_size_4096"
    #name = "2_EventWeightNormalized_NoMedian"

    #fit function
    fit_model("FCN", n_class, folder_dnn_v3, result_v3,0,[],jet_features_list,[],[],"is_signal","EventWeightNormalized",use_weight=True,n_epochs=n_epochs,n_batch_size=b_size,patience_val=200,val_split=0.0,model_label=name,ignore_empty_jets_train=True)

    #evaluate performances
    ##evaluate_model("FCN", n_class, folder_dnn_v3, result_v3,0,[],jet_features_list,[],[],"is_signal","Jet_isGenMatchedCaloCorrLLPAccept","EventWeightNormalized",use_weight=True,n_batch_size=b_size,model_label=name,signal_match_test=False,ignore_empty_jets_test=True)
    evaluate_model("FCN", n_class, folder_dnn_v3, result_v3,0,[],jet_features_list,[],[],"is_signal",jet_string+"isGenMatchedCaloCorrLLPAccept","EventWeightNormalized",use_weight=True,n_batch_size=b_size,model_label=name,signal_match_test=True,ignore_empty_jets_test=True)


elif TRAIN_MODEL == "FCN_constituents":
    print("\n")
    print("   Training FCN_cons on constituent features (FCN)    ")
    print("\n")
    print(pf_features)
    print(len(pf_features)," training features!")
    print("\n")
    print(pf_points)
    print(len(pf_points)," points coordinates")
    print("\n")
    
    name = "2_SW"

    #fit function
    fit_model("FCN_constituents", n_class, graphnet_pd_partnet, fcn_result_folder, npf, pf_points, pf_features,[],[],"is_signal","EventWeightNormalized",use_weight=True,n_epochs=100,n_batch_size=1000,patience_val=100,val_split=0.0,model_label=name,ignore_empty_jets_train=True)

    #evaluate performances
    evaluate_model("FCN_constituents", n_class, graphnet_pd_partnet, fcn_result_folder, npf, pf_points, pf_features,[],[],"is_signal","Jet_isGenMatchedCaloCorrLLPAccept","EventWeightNormalized",use_weight=True,n_batch_size=1000,model_label=name,signal_match_test=True,ignore_empty_jets_test=True)

    #evaluate_model("FCN", n_class, graphnet_pd_JJ_MET, graphnet_result_folder,0,[],jet_features_list,[],[],"is_signal","Jet_isGenMatchedCaloCorrLLPAccept","EventWeightNormalized",use_weight=True,n_batch_size=2000,model_label=name,signal_match_test=False,ignore_empty_jets_test=True)     

elif TRAIN_MODEL == "BDT":
    print("\n")
    print("   Training BDT on jet features (same as FCN)    ")
    print("\n")
    print("   WARNING! Taking event weight NOT normalized!    ")
    print(jet_features_list)
    print(len(jet_features_list)," training features!")
    print("\n")

    n_epochs = 200
    patience = 25
    name = "SampleWeight_NoMedian_NoRecHitsHB_NoSi_OnlyECALtime"+str(n_epochs)+"epochs_patience"+str(patience)
    #name = "SampleWeight_NoMedian_"+str(n_epochs)+"epochs_patience"+str(patience)
    #name = "SampleWeight_CaloHitsAndEnergyFractionsAndXMax_"+str(n_epochs)+"epochs"

    fit_BDT("BDT", n_class, folder_BDT_v3, result_v3,0,[],jet_features_list,[],"is_signal","SampleWeight",use_weight=True,n_epochs=n_epochs,n_batch_size=2000,patience_val=patience,val_split=0.0,model_label=name,ignore_empty_jets_train=True)

    evaluate_BDT("BDT", n_class, folder_BDT_v3, result_v3,0,[],jet_features_list,[],"is_signal",jet_string+"isGenMatchedCaloCorrLLPAccept","SampleWeight",use_weight=True,n_batch_size=2000,model_label=name,signal_match_test=True,ignore_empty_jets_test=True)

    #evaluate_BDT("BDT", n_class, folder_BDT_v3, result_v3,0,[],jet_features_list,[],"is_signal","Jet_isGenMatchedCaloCorrLLPAccept","SampleWeight",use_weight=True,n_batch_size=2000,model_label=name,signal_match_test=False,ignore_empty_jets_test=True)


elif TRAIN_MODEL == "particle_net_lite":
    print("\n")
    print("   Training ParticleNet lite on ", npf, "jet constituents features (particle_net_lite)    ")
    print("\n")
    print(pf_features)
    print(len(pf_features)," training features")
    print("\n")
    print(pf_points)
    print(len(pf_points)," points coordinates")
    print("\n")
    print(pf_mask)
    print(len(pf_mask)," mask")
    print("\n")

    name = "EventWeightNormalized"

    fit_model("particle_net_lite", n_class, graphnet_pd_partnet, graphnet_result_partnet, npf, pf_points, pf_features, pf_mask, [], "is_signal","EventWeightNormalized",use_weight=True,n_epochs=50,n_batch_size=2000,patience_val=5,val_split=0.0,model_label=name,ignore_empty_jets_train=True)

    evaluate_model("particle_net_lite", n_class, graphnet_pd_partnet, graphnet_result_partnet, npf, pf_points, pf_features, pf_mask, [], "is_signal","Jet_isGenMatchedCaloCorrLLPAccept","EventWeightNormalized",use_weight=True,n_batch_size=2000,model_label=name,signal_match_test=True,ignore_empty_jets_test=True)

    evaluate_model("particle_net_lite", n_class, graphnet_pd_partnet, graphnet_result_partnet, npf, pf_points, pf_features, pf_mask, [], "is_signal","Jet_isGenMatchedCaloCorrLLPAccept","EventWeightNormalized",use_weight=True,n_batch_size=2000,model_label=name,signal_match_test=False,ignore_empty_jets_test=True)

elif TRAIN_MODEL == "particle_net":
    print("\n")
    print("   Training ParticleNet on ", npf, "jet constituents features (particle_net)    ")
    print("\n")
    print(pf_features)
    print(len(pf_features)," training features")
    print("\n")
    print(pf_points)
    print(len(pf_points)," points coordinates")
    print("\n")
    print(pf_mask)
    print(len(pf_mask)," mask")
    print("\n")

    name = "test"

    fit_model("particle_net", n_class, graphnet_pd_partnet, graphnet_result_partnet, npf, pf_points, pf_features, pf_mask, [], "is_signal","SampleWeight",use_weight=True,n_epochs=50,n_batch_size=500,patience_val=5,val_split=0.0,model_label=name,ignore_empty_jets_train=True)

    evaluate_model("particle_net", n_class, graphnet_pd_partnet, graphnet_result_partnet, npf, pf_points,pf_features, pf_mask, [], "is_signal","Jet_isGenMatchedCaloCorrLLPAccept","SampleWeight",use_weight=True,n_batch_size=500,model_label=name,signal_match_test=True,ignore_empty_jets_test=True)

    evaluate_model("particle_net", n_class, graphnet_pd_partnet, graphnet_result_partnet, npf, pf_points,pf_features, pf_mask, [], "is_signal","Jet_isGenMatchedCaloCorrLLPAccept","SampleWeight",use_weight=True,n_batch_size=500,model_label=name,signal_match_test=False,ignore_empty_jets_test=True)

elif TRAIN_MODEL == "particle_net_jet":
    print("\n")
    print("   Training ParticleNet on ", npf, "jet constituents features (particle_net)    ")
    print("   and on ", nj, "jet high level features    ")
    print("\n")
    print(pf_features)
    print(len(pf_features)," training features")
    print("\n")
    print(pf_points)
    print(len(pf_points)," points coordinates")
    print("\n")
    print(pf_mask)
    print(len(pf_mask)," mask")
    print("\n")
    print(jet_features_list)
    print(len(jet_features_list)," jet features")
    print("\n")

    name = "test_lisa"

    fit_model("particle_net_jet", n_class, folder_test, result_test, npf, pf_points, pf_features, pf_mask, jet_features_list, "is_signal","SampleWeight",use_weight=True,n_epochs=50,n_batch_size=500,patience_val=5,val_split=0.0,model_label=name,ignore_empty_jets_train=True)

    evaluate_model("particle_net_jet", n_class, folder_test, result_test, npf, pf_points,pf_features, pf_mask, jet_features_list, "is_signal","Jet_isGenMatchedCaloCorrLLPAccept","SampleWeight",use_weight=True,n_batch_size=500,model_label=name,signal_match_test=True,ignore_empty_jets_test=True)

    evaluate_model("particle_net_jet", n_class, folder_test, result_test, npf, pf_points,pf_features, pf_mask, jet_features_list, "is_signal","Jet_isGenMatchedCaloCorrLLPAccept","SampleWeight",use_weight=True,n_batch_size=500,model_label=name,signal_match_test=False,ignore_empty_jets_test=True)

####################
###To be tested:

##write_discriminator_output(graphnet_pd_folder,graphnet_result_folder,jet_features_list,"Jet_isGenMatched","EventWeightNormalized",n_batch_size=2000,model_label="graph_0",sample_list=sgn+bkg)
##var = jet_features_list + ["EventNumber","RunNumber","LumiNumber","EventWeight","isMC","Jet_isGenMatched","Jet_sigprob","Jet_index"]
##output_root_files = "root_files_tagger/v2_calo_AOD_2017/"
##var+= ["nDTSegments","nStandAloneMuons","nDisplacedStandAloneMuons"]
##test_to_root(graphnet_pd_folder,graphnet_result_folder,output_root_files,event_list+jvar,"is_signal",model_label="graph_0",sample_list=sgn+bkg)
