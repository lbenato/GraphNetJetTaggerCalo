import pandas as pd
from tensorflow import keras
import tensorflow as tf
import numpy as np
import awkward
import uproot_methods
import os.path
from datetime import datetime
from collections import defaultdict
# Needed libraries
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from root_numpy import array2tree, array2root
from dnn_functions import *
from samplesAOD2017 import *
from tf_keras_model import *

# Configure parameters
##pd_folder = 'dataframes/v2_calo_AOD_2017_test/'
graphnet_pd_folder = 'dataframes_graphnet/v2_calo_AOD_2017_test/'
##result_folder = 'model_weights/v2_calo_AOD_2017_test/'
graphnet_result_folder = 'model_weights_graphnet/v2_calo_AOD_2017_test/'

sgn = ['SUSY_mh400_pl1000']#,'SUSY_mh300_pl1000','SUSY_mh250_pl1000','SUSY_mh200_pl1000','SUSY_mh175_pl1000','SUSY_mh150_pl1000','SUSY_mh127_pl1000']
bkg = ['VV']

# define your variables here
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
print(jet_list)


###################################
### Define PFCandidate features ###
###################################

npf=20#100
pf_nottrain = [
    #'energy','px','py','pz',
    'pt',
    'pdgId','isTrack','hasTrackDetails', 'dxy', 'dz', 'POCA_x', 'POCA_y', 'POCA_z', 'POCA_phi',
    'ptError', 'etaError', 'phiError', 'dxyError', 'dzError', 'theta', 'thetaError','chi2', 'ndof', 'normalizedChi2',
    'nHits', 'nPixelHits', 'lostInnerHits', 'jetIndex',
]
pf_features = [
    'energy','px','py','pz',
    #'pt',
    #'pdgId','isTrack','hasTrackDetails', 'dxy', 'dz', 'POCA_x', 'POCA_y', 'POCA_z', 'POCA_phi',
    #'ptError', 'etaError', 'phiError', 'dxyError', 'dzError', 'theta', 'thetaError','chi2', 'ndof', 'normalizedChi2',
    #'nHits', 'nPixelHits', 'lostInnerHits',
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
print(pf_features_list)
print(pf_list)


##Time stamp for saving model
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%d%b%Y_%H_%M_%S")
print("Time:", timestampStr)
print("\n")


def _col_list(prefix):
    return ['%s_%d'%(prefix,i) for i in range(npf)]


def fit_model(model_def,n_class,folder,result_folder,points,features,mask,is_signal,weight,n_epochs,n_batch_size,patience_val,val_split,model_label="",ignore_empty_jets_train=True):

    ##Read train/validation sample
    store_train = pd.HDFStore(folder+"train.h5")
    df_train = store_train.select("df")
    store_val = pd.HDFStore(folder+"val.h5")
    df_val = store_val.select("df")
    
    if(model_def=="LEADER"):
        X_train, y_train, w_train = get_FCN_jets_dataset(df_train,features,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
        X_val,   y_val,   w_val   = get_FCN_jets_dataset(df_val,features,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
        model = get_FCN_jets(num_classes=n_class, input_shapes=X_train.shape[1:])
    elif(model_def=="particle_net_lite"):
        X_train, y_train, w_train, input_shapes = get_particle_net_dataset(df_train,points,features,mask,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
        X_val,   y_val,   w_val, _   = get_particle_net_dataset(df_val,points,features,mask,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
        model = get_particle_net_lite(n_class, input_shapes) 
    else:
        print("  Model not recognized, abort . . .")
        exit()

    model.summary()

    if model_label=="":
        model_label=model_def+"_"+timestampStr
        
    if not os.path.isdir(result_folder+'/model_'+model_def+"_"+model_label):
        os.mkdir(result_folder+'/model_'+model_def+"_"+model_label)
        
    result_folder += 'model_'+model_def+"_"+model_label+"/"
    
    print("\n")
    print("   Fitting model.....   ")
    print("\n")
    
    ##Compile
    model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics = ["accuracy"])
    #custom_adam:
    #custom_adam = keras.optimizers.Adam(learning_rate=0.001/2., beta_1=0.9, beta_2=0.999, amsgrad=False)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=custom_adam, metrics = ["accuracy"])

    ##Callbacks
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience_val, verbose=0, mode='auto')
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=result_folder+'best_model_'+model_label+'.h5', monitor='val_loss', save_best_only=True)


    ##Fit model
    #train is 60%, test is 20%, val is 20%
    print(" WITHOUT WEIGHTS!! ")
    histObj = model.fit(X_train, y_train, epochs=n_epochs, batch_size=n_batch_size, validation_split=val_split, validation_data=None if val_split>0 else (X_val, y_val), callbacks=[early_stop, checkpoint])

    #print(" WITH WEIGHTS!! ")
    #histObj = model.fit(df_train[features].values, df_train[is_signal].values, epochs=n_epochs, batch_size=n_batch_size, sample_weight=df_train[weight].values, validation_split=val_split, validation_data=None if val_split>0 else (df_val[features].values, df_val[is_signal].values, df_val[weight].values), callbacks=[early_stop, checkpoint])
    #validation_data=(df_val[features].values, df_val["is_signal"].values, df_val["EventWeight"].values))#, batch_size=128) 
    histObj.name='model_'+model_def+model_label # name added to legend
    plot = plotLearningCurves(histObj)# the above defined function to plot learning curves
    plot.savefig(result_folder+'loss_accuracy_'+model_label+'.png')
    plot.savefig(result_folder+'loss_accuracy_'+model_label+'.pdf')
    print("Plot saved in: ", result_folder+'loss_accuracy_'+model_label+'.png')
    output_file = 'model_'+model_label
    model.save(result_folder+output_file+'.h5')
    del model
    print("Model saved in ", result_folder+output_file+'.h5')
    #plot.show()

def evaluate_model(model_def,n_class,folder,result_folder,points,features,mask,is_signal,weight,n_batch_size,model_label,signal_match_test,ignore_empty_jets_test):

    print("\n")
    print("   Evaluating performances of the model.....   ")
    print("\n")

    ##Read test sample
    store = pd.HDFStore(folder+"test.h5")
    df_test = store.select("df")

    print("    Remove negative weights at testing!!!!!!")
    df_test = df_test.loc[df_test['EventWeight']>=0]

    add_string = ""
    if ignore_empty_jets_test:
        print("\n")
        print("    Ignore empty jets at testing!!!!!!")
        print("\n")
        df_test = df_test.loc[df_test["Jet_isGenMatched"]!=-1]
        add_string+="_ignore_empty_jets"

    if signal_match_test:
        print("\n")
        print("    Ignore not matched jets in signal at testing!!!!!!")
        print("\n")
        df_s = df_test.loc[df_test[is_signal]==1]
        df_b = df_test.loc[df_test[is_signal]==0]
        df_s = df_s.loc[df_s["Jet_isGenMatched"]==1]
        df_test = pd.concat([df_b,df_s])
        print(df_test.shape[0],df_s.shape[0],df_b.shape[0])
        add_string+="_signal_matched"

    
    if(model_def=="LEADER"):
        X_test, y_test, w_test = get_FCN_jets_dataset(df_test,features,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
    elif(model_def=="particle_net_lite"):
        X_test, y_test, w_test, input_shapes = get_particle_net_dataset(df_test,points,features,mask,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
    else:
        print("  Model not recognized, abort . . .")
        exit()


    if model_label=="":
        model_label=model_def+"_"+timestampStr

    result_folder += 'model_'+model_def+"_"+model_label+"/"
    output_file = 'model_'+model_label

    #if not os.path.isdir(result_folder+'/model_'+model_def+"_"+model_label):
    #    print("Result folder ",result_folder, " does not exist! Have you trained the model? Aborting . . .")
    #    exit()

    print("Loading model... ", result_folder+output_file+'.h5')
    model = keras.models.load_model(result_folder+output_file+'.h5')
    model.summary()
    print("Running on test sample. This may take a moment. . .")


    probs = model.predict(X_test)#predict probability over test sample
    #print("Negative weights?")
    #print(df_test[df_test[weight]<0])
    #df_test = df_test[df_test[weight]>=0]
    #print(df_test)

    print(" WITHOUT WEIGHTS!! ")
    AUC = roc_auc_score(y_test, probs[:,1])

    #print(" WITH WEIGHTS!! ")
    #AUC = roc_auc_score(df_test[is_signal], probs[:,1],sample_weight=df_test[weight])
    print("Test Area under Curve = {0}".format(AUC))
    #exit()
    df_test["sigprob"] = probs[:,1]

    df_test.to_hdf(result_folder+'test_score_'+model_label+add_string+'.h5', 'df', format='fixed')
    print("   "+result_folder+"test_score_"+model_label+add_string+".h5 stored")

    back = np.array(df_test["sigprob"].loc[df_test[is_signal]==0].values)
    sign = np.array(df_test["sigprob"].loc[df_test[is_signal]==1].values)
    back_w = np.array(df_test["EventWeightNormalized"].loc[df_test[is_signal]==0].values)
    sign_w = np.array(df_test["EventWeightNormalized"].loc[df_test[is_signal]==1].values)
    #saves the df_test["sigprob"] column when the event is signal or background
    plt.figure(figsize=(8,7))
    plt.rcParams.update({'font.size': 15}) #Larger font size
    #Let's plot an histogram:
    # * y-values: back/sign probabilities
    # * 50 bins
    # * alpha: filling color transparency
    # * density: it should normalize the histograms to unity
    print(" WITHOUT WEIGHTS!! ")
    plt.hist(back, 50, color='blue', edgecolor='blue', lw=2, label='background', alpha=0.3)#, density=True)
    plt.hist(sign, 50, color='red', edgecolor='red', lw=2, label='signal', alpha=0.3)#, density=True)

    #print(" WITH WEIGHTS!! ")
    #plt.hist(back, 50, weights=back_w, color='blue', edgecolor='blue', lw=2, label='background', alpha=0.3)#, density=True)
    #plt.hist(sign, 50, weights=sign_w, color='red', edgecolor='red', lw=2, label='signal', alpha=0.3)#, density=True)

    plt.xlim([0.0, 1.05])
    plt.xlabel('Event probability of being classified as signal')
    plt.legend(loc="upper right")
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(result_folder+'probability_'+output_file+add_string+'.png')
    plt.savefig(result_folder+'probability_'+output_file+add_string+'.pdf')
    #plt.show()

    print(" WITHOUT WEIGHTS!! ")
    fpr, tpr, _ = roc_curve(df_test[is_signal], df_test["sigprob"])

    #print(" WITH WEIGHTS!! ")
    #fpr, tpr, _ = roc_curve(df_test[is_signal], df_test["sigprob"], sample_weight=df_test[weight]) #extract true positive rate and false positive rate

    plt.figure(figsize=(8,7))
    plt.rcParams.update({'font.size': 15}) #Larger font size
    plt.plot(fpr, tpr, color='crimson', lw=2, label='ROC curve (area = {0:.4f})'.format(AUC))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(result_folder+'ROC_'+output_file+add_string+'.pdf')
    plt.savefig(result_folder+'ROC_'+output_file+add_string+'.png')
    #plt.show()
    print("   Plots printed in "+result_folder)

    plt.figure(figsize=(8,7))
    plt.rcParams.update({'font.size': 15}) #Larger font size
    plt.plot(fpr, tpr, color='crimson', lw=2, label='ROC curve (area = {0:.4f})'.format(AUC))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.xlim([0.0001, 1.05])
    plt.xscale('log')
    plt.savefig(result_folder+'ROC_'+output_file+add_string+'_logx.pdf')
    plt.savefig(result_folder+'ROC_'+output_file+add_string+'_logx.png')
    #plt.show()

def write_discriminator_output(folder,result_folder,features,is_signal,weight,n_batch_size,model_label,sample_list=[]):
    if model_label=="":
        model_label=timestampStr
    output_file = 'model_'+model_label
    print("Loading model... ", result_folder+output_file+'.h5')
    model = keras.models.load_model(result_folder+output_file+'.h5')
    model.summary()
    print("Running on test sample. This may take a moment. . .")
    
    if sample_list==[]:
        ##Read test sample
        store = pd.HDFStore(folder+"test_"+model_label+".h5")
        df = store.select("df")
        df_valid = df.loc[df["Jet_isGenMatched"]!=-1]
        df_invalid = df.loc[df["Jet_isGenMatched"]==-1]

        probs = model.predict(df_valid[features].values)#predict probability over test sample
        df_valid["Jet_sigprob"] = probs[:,1]
        df_invalid["Jet_sigprob"] = np.ones(df_invalid.shape[0])*(-1)
        df_test = pd.concat([df_valid,df_invalid])
        df_test.to_hdf(result_folder+'test_score_'+model_label+'.h5', 'df', format='table')
        print("   "+result_folder+"test_score_"+model_label+".h5 stored")


    else:

        full_list = []
        for sl in sample_list:
            full_list += samples[sl]['files']

        for sample in full_list:
            print(" ********************* ")
            print(folder+sample+"_test.h5")
            ##Read test sample
            if not os.path.isfile(folder+sample+"_test.h5"):
                print("!!!File ", folder+sample+"_test.h5", " does not exist! Continuing")
                continue
            store = pd.HDFStore(folder+sample+"_test.h5")
            df = store.select("df")
            df_valid = df.loc[df["Jet_isGenMatched"]!=-1]
            df_invalid = df.loc[df["Jet_isGenMatched"]==-1]

            probs = model.predict(df_valid[features].values)#predict probability over test sample
            df_valid["Jet_sigprob"] = probs[:,1]
            df_invalid["Jet_sigprob"] = np.ones(df_invalid.shape[0])*(-1)
            df_test = pd.concat([df_valid,df_invalid])
            df_test.to_hdf(result_folder+sample+'_score_'+model_label+'.h5', 'df', format='table')
            print("   "+result_folder+sample+"_score_"+model_label+".h5 stored")


def test_to_root(folder,result_folder,output_root_folder,variables,is_signal,model_label,sample_list=[]):

    if not os.path.isdir(output_root_folder+'/model_'+model_label): os.mkdir(output_root_folder+'/model_'+model_label)

    if sample_list==[]:
        print("   Empty sample list, will use full sample . . .")
        ##Read test sample
        store = pd.HDFStore(result_folder+'test_score_'+model_label+'.h5')
        df_test = store.select("df")

        for n, a in enumerate(variables):
            back = np.array(df_test[a].loc[df_test[is_signal]==0].values, dtype=[(a, np.float64)])
            sign = np.array(df_test[a].loc[df_test[is_signal]==1].values, dtype=[(a, np.float64)])
            print(a," back: ", back)
            print(a," sign: ", sign)
            array2root(back, output_root_folder+'/model_'+model_label+'/test_bkg.root', mode='recreate' if n==0 else 'update')
            array2root(sign, output_root_folder+'/model_'+model_label+'/test_sgn.root', mode='recreate' if n==0 else 'update')
        print("  Signal and background root files written : ", output_root_folder+'/'+model_label+'/test_*.root')

    else:
        full_list = []
        for sl in sample_list:
            full_list += samples[sl]['files']

        for sample in full_list:
            print("   Reading sample: ", sample)
            ##Read test sample
            if not os.path.isfile(folder+sample+"_test.h5"):
                print("!!!File ", folder+sample+"_test.h5", " does not exist! Continuing")
                continue

            store = pd.HDFStore(result_folder+sample+"_score_"+model_label+".h5")
            df_test = store.select("df")

            #smaller for testing
            #df_test = df_test.sample(frac=1).reset_index(drop=True)#shuffle
            #df_test = df_test[0:10]

            #print(df_test)
            df_j = defaultdict()
            #Transform per-event into per-jet
            for j in range(nj):
                #print(j)
                df_j[j] = df_test.loc[ df_test["Jet_index"]==float(j) ]
                #print(df_j[j]["Jet_index"])
                #if df_j[j].shape[0]>0: print(df_j[j])
                #temp_list = []
                for f in jvar:
                    #print("Jet_"+f)
                    #print("Jets"+str(j)+"_"+f)
                    df_j[j].rename(columns={"Jet_"+f: "Jets"+str(j)+"_"+f},inplace=True)
                    #if str(j) in l:
                    #    print("\n")
                    #    #temp_list.append(l)
                df_j[j].rename(columns={"Jet_isGenMatched": "Jets"+str(j)+"_isGenMatched"},inplace=True)
                df_j[j].rename(columns={"Jet_index": "Jets"+str(j)+"_index"},inplace=True)
                df_j[j].rename(columns={"Jet_sigprob": "Jets"+str(j)+"_sigprob"},inplace=True)
                #if df_j[j].shape[0]>0: print(df_j[j])

                if j==0:
                    df = df_j[j]
                else:
                    df_temp = pd.merge(df, df_j[j], on=event_list, how='inner')
                    df = df_temp

            #Here, count how many jets are tagged!
            #Reject events with zero jets
            #print(df)
            df = df[ df['nCHSJets']>0]
            #Define variables to counts nTags
            var_tag_sigprob = []
            var_tag_cHadEFrac = []
            for j in range(nj):
                var_tag_sigprob.append("Jets"+str(j)+"_sigprob")
                var_tag_cHadEFrac.append("Jets"+str(j)+"_cHadEFrac")
            #print(var_tag_sigprob)
            wp_sigprob = [0.5,0.6,0.7,0.8,0.9,0.95]
            wp_cHadEFrac = [0.2,0.1,0.05,0.02]
            for wp in wp_sigprob:
                name = str(wp).replace(".","p")
                df['nTags_sigprob_wp'+name] = df[ df[var_tag_sigprob] > wp ].count(axis=1)
            for wp in wp_cHadEFrac:
                name = str(wp).replace(".","p")
                df['nTags_cHadEFrac_wp'+name] = df[ (df[var_tag_cHadEFrac] < wp) & (df[var_tag_cHadEFrac]>-1) ].count(axis=1)

            #!!!#
            #print("\n")
            #print(df)
            #print(nj)
            #print(len(jfeatures)+3)
            #print(len(event_list))
            #print(list(df.columns))
            #df_0 = df_j[0]
            #df_3 = df_j[3]     
            #mergedStuff = pd.merge(df_0, df_3, on=event_list, how='inner')

            #Here I must compare df_j with the same event number and merge it

            newFile = TFile(output_root_folder+'/model_'+model_label+'/'+sample+'.root', 'recreate')
            newFile.cd()
            for n, a in enumerate(list(df.columns)):
                arr = np.array(df[a].values, dtype=[(a, np.float64)])
                ###print(a, " values: ", arr)
                ###array2root(arr, output_root_folder+'/model_'+model_label+'/'+sample+'.root', mode='update')#mode='recreate' if n==0 else 'update')
                if n==0: skim = array2tree(arr)
                else: array2tree(arr, tree=skim)#mode='recreate' if n==0 else 'update')

            skim.Write()
            ##Recreate c_nEvents histogram
            counter = TH1F("c_nEvents", "Event Counter", 1, 0., 1.)
            counter.Sumw2()
            ##Fill counter histogram with the first entry of c_nEvents
            counter.Fill(0., df["c_nEvents"].values[0])
            ##print("counter bin content: ", counter.GetBinContent(1))
            counter.Write()
            newFile.Close()
            ##counter.Delete()

            
            print("  Root file written : ", output_root_folder+'/model_'+model_label+'/'+sample+'.root')


def get_FCN_jets_dataset(dataframe,features,weight,is_signal="is_signal",ignore_empty_jets=True):

    if ignore_empty_jets:
        print("\n")
        print("    Ignore empty jets!!!!!!")
        print("\n")
        dataframe = dataframe[ dataframe["Jet_isGenMatched"]!=-1 ]

    X = dataframe[features].values
    y = dataframe[is_signal].values
    w = dataframe[weight].values
        
    return X, y, w

def get_particle_net_dataset(dataframe,points_var,features_var,mask_var,weight,is_signal="is_signal",ignore_empty_jets=True):

    if ignore_empty_jets:
        print("\n")
        print("    Ignore empty jets!!!!!!")
        print("\n")
        dataframe = dataframe[ dataframe["Jet_isGenMatched"]!=-1 ]
        
    points_arr = []
    features_arr = []
    mask_arr = []
    
    for p_var in points_var:
        points_arr.append(dataframe[_col_list(p_var)].values)
    points = np.stack(points_arr,axis=-1)

    for f_var in features_var:
        features_arr.append(dataframe[_col_list(f_var)].values)
    features = np.stack(features_arr,axis=-1)
    
    for m_var in mask_var:
        mask_arr.append(dataframe[_col_list(m_var)].values)
    mask = np.stack(mask_arr,axis=-1)

    input_shapes = defaultdict()
    input_shapes['points'] = points.shape[1:]
    input_shapes['features'] = features.shape[1:]
    input_shapes['mask'] = mask.shape[1:]


    X = [points,features,mask]
    y = dataframe[is_signal].values
    w = dataframe[weight].values
        
    return X, y, w, input_shapes
    
# # # # # #


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config)) 

### Here we need a switch between jet features and pf features
cols = jet_features_list
print("\n")
print(cols)
print(len(cols)," training features!")
print("\n")

###Here we must get the different models and the different training features
n_class=2

#####################

### FCN works fine!!!
#fit_model("LEADER", n_class, graphnet_pd_folder, graphnet_result_folder,[],cols,[],"Jet_isGenMatched","EventWeightNormalized",n_epochs=50,n_batch_size=2000,patience_val=5,val_split=0.0,model_label="0",ignore_empty_jets_train=True)

#evaluate_model("LEADER", n_class, graphnet_pd_folder, graphnet_result_folder,[],cols,[],"Jet_isGenMatched","EventWeightNormalized",n_batch_size=2000,model_label="0",signal_match_test=True,ignore_empty_jets_test=True)

###

####################
### now graph net
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

fit_model("particle_net_lite", n_class, graphnet_pd_folder, graphnet_result_folder,['eta', 'phi'],pf_features,['pt'],"Jet_isGenMatched","EventWeightNormalized",n_epochs=50,n_batch_size=2000,patience_val=5,val_split=0.0,model_label="0",ignore_empty_jets_train=True)

evaluate_model("particle_net_lite", n_class, graphnet_pd_folder, graphnet_result_folder,['eta', 'phi'],pf_features,['pt'],"Jet_isGenMatched","EventWeightNormalized",n_batch_size=2000,model_label="0",signal_match_test=True,ignore_empty_jets_test=True)
###

####################

###
exit()

##write_discriminator_output(graphnet_pd_folder,graphnet_result_folder,cols,"Jet_isGenMatched","EventWeightNormalized",n_batch_size=2000,model_label="graph_0",sample_list=sgn+bkg)
##var = cols + ["EventNumber","RunNumber","LumiNumber","EventWeight","isMC","Jet_isGenMatched","Jet_sigprob","Jet_index"]
##output_root_files = "root_files_tagger/v2_calo_AOD_2017/"
##var+= ["nDTSegments","nStandAloneMuons","nDisplacedStandAloneMuons"]
##test_to_root(graphnet_pd_folder,graphnet_result_folder,output_root_files,event_list+jvar,"is_signal",model_label="graph_0",sample_list=sgn+bkg)



'''
At the moment, this function is not needed. We might restore it later for relative eta/phi

def transform(dataframe, max_particles=10, start=0, stop=-1):
    from collections import OrderedDict
    v = OrderedDict()

    df = dataframe.iloc[start:stop]
    def _col_list(prefix):
        return ['%s_%d'%(prefix,i) for i in range(max_particles)]
    
    _px = df[_col_list('px')].values
    _py = df[_col_list('py')].values
    _pz = df[_col_list('pz')].values
    _pt = df[_col_list('pt')].values
    _e = df[_col_list('energy')].values
    _eta = df[_col_list('eta')].values
    _phi = df[_col_list('phi')].values
    
    mask = _pt>-1.
    n_particles = np.sum(mask, axis=1)
    print("n_particles post pt>-1. cut: ", n_particles)

    px = awkward.JaggedArray.fromcounts(n_particles, _px[mask])
    py = awkward.JaggedArray.fromcounts(n_particles, _py[mask])
    pz = awkward.JaggedArray.fromcounts(n_particles, _pz[mask])
    energy = awkward.JaggedArray.fromcounts(n_particles, _e[mask])

    p4 = uproot_methods.TLorentzVectorArray.from_cartesian(px, py, pz, energy)
    pt = p4.pt

    jet_p4 = p4.sum()
    
    print("Compare new/original eta:")
    print(p4.eta)
    print(_eta)
    print("Compare new/original pt:")
    print(p4.pt)
    print(_pt)
    
    # outputs
    _label = df['is_signal'].values
    v['label'] = np.stack((_label, 1-_label), axis=-1)
    v['train_val_test'] = df['ttv'].values
    
    v['jet_pt'] = jet_p4.pt
    
    print("Jet pt old/new")
    print(df[('Jet_pt')].values)
    print(jet_p4.pt)
    
    v['jet_eta'] = jet_p4.eta
    
    print("Jet eta old/new")
    print(df[('Jet_eta')].values)
    print(jet_p4.eta)
    
    v['jet_phi'] = jet_p4.phi
    v['jet_mass'] = jet_p4.mass
    v['n_parts'] = n_particles

    v['part_px'] = px
    v['part_py'] = py
    v['part_pz'] = pz
    v['part_energy'] = energy

    v['part_pt_log'] = np.log(pt)
    v['part_ptrel'] = pt/v['jet_pt']
    v['part_logptrel'] = np.log(v['part_ptrel'])

    v['part_e_log'] = np.log(energy)
    v['part_erel'] = energy/jet_p4.energy
    v['part_logerel'] = np.log(v['part_erel'])

    v['part_raw_etarel'] = (p4.eta - v['jet_eta'])
    _jet_etasign = np.sign(v['jet_eta'])
    _jet_etasign[_jet_etasign==0] = 1
    v['part_etarel'] = v['part_raw_etarel'] * _jet_etasign

    v['part_phirel'] = p4.delta_phi(jet_p4)
    v['part_deltaR'] = np.hypot(v['part_etarel'], v['part_phirel'])

    return v
'''
