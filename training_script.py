from graphnet_tagger import *

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

## ## Configure parameters ## ##

#~~~~~~~~~~~~~~~~~~~~~~~~~
### Folders ###
#~~~~~~~~~~~~~~~~~~~~~~~~~
graphnet_pd_folder = 'dataframes_graphnet/v2_calo_AOD_2017_test/'
graphnet_result_folder = 'model_weights_graphnet/v2_calo_AOD_2017_test/'

#~~~~~~~~~~~~~~~~~~~~~~~~~
## Signal and background samples, defined in samplesAOD201X.py
#~~~~~~~~~~~~~~~~~~~~~~~~~
sgn = ['SUSY_mh400_pl1000']#,'SUSY_mh300_pl1000','SUSY_mh250_pl1000','SUSY_mh200_pl1000','SUSY_mh175_pl1000','SUSY_mh150_pl1000','SUSY_mh127_pl1000']
bkg = ['VV']

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
'nTrackConstituents','nSelectedTracks','nHadEFrac', 'cHadEFrac','ecalE','hcalE',
'muEFrac','eleEFrac','photonEFrac',
'eleMulti','muMulti','photonMulti','cHadMulti','nHadMulti',
'nHitsMedian','nPixelHitsMedian',
'dRSVJet', 'nVertexTracks', 'CSV', 'SV_mass',
]
jet_features_list = []
for f in j_features:
    jet_features_list.append("Jet_"+f)

# These are all the jet variables we want to save in the final output root file; needed to fully reconstruct the event
j_var = j_gen+j_features+j_nottrain
jet_list = []
for v in j_var:
    jet_list.append("Jet_"+v)

###################################
### Define PFCandidate features ###
###################################

# Number of pf candidates used in the model
npf=20 #npf are 100 at maximum

#PF features we have in the dataset but that we don't use for training
pf_nottrain = [
    #'energy','px','py','pz',
    'pt',
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
    'energy','px','py','pz',
    #'pt',
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

### Here we need a switch between jet features and pf features
TRAIN_MODEL = "LEADER"
#TRAIN_MODEL = "particle_net_lite"

if TRAIN_MODEL == "LEADER":
    print("\n")
    print("   Training FCN on jet features (LEADER)    ")
    print("\n")
    print(jet_features_list)
    print(len(jet_features_list)," training features!")
    print("\n")
    
    #fit function
    fit_model("LEADER", n_class, graphnet_pd_folder, graphnet_result_folder,0,[],jet_features_list,[],"Jet_isGenMatched","EventWeightNormalized",use_weight=True,n_epochs=50,n_batch_size=2000,patience_val=5,val_split=0.0,model_label="0",ignore_empty_jets_train=True)

    #evaluate performances
    evaluate_model("LEADER", n_class, graphnet_pd_folder, graphnet_result_folder,0,[],jet_features_list,[],"Jet_isGenMatched","EventWeightNormalized",use_weight=True,n_batch_size=2000,model_label="0",signal_match_test=True,ignore_empty_jets_test=True)

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

    fit_model("particle_net_lite", n_class, graphnet_pd_folder, graphnet_result_folder, npf, pf_points, pf_features, pf_mask,"Jet_isGenMatched","EventWeightNormalized",use_weight=True,n_epochs=50,n_batch_size=2000,patience_val=5,val_split=0.0,model_label="0",ignore_empty_jets_train=True)

    evaluate_model("particle_net_lite", n_class, graphnet_pd_folder, graphnet_result_folder, npf, pf_points,pf_features, pf_mask,"Jet_isGenMatched","EventWeightNormalized",use_weight=True,n_batch_size=2000,model_label="0",signal_match_test=True,ignore_empty_jets_test=True)



####################
###To be tested:

##write_discriminator_output(graphnet_pd_folder,graphnet_result_folder,jet_features_list,"Jet_isGenMatched","EventWeightNormalized",n_batch_size=2000,model_label="graph_0",sample_list=sgn+bkg)
##var = jet_features_list + ["EventNumber","RunNumber","LumiNumber","EventWeight","isMC","Jet_isGenMatched","Jet_sigprob","Jet_index"]
##output_root_files = "root_files_tagger/v2_calo_AOD_2017/"
##var+= ["nDTSegments","nStandAloneMuons","nDisplacedStandAloneMuons"]
##test_to_root(graphnet_pd_folder,graphnet_result_folder,output_root_files,event_list+jvar,"is_signal",model_label="graph_0",sample_list=sgn+bkg)
