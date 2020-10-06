import pandas as pd
#from tensorflow import keras
#import tensorflow as tf
import numpy as np
import awkward
import pickle
import uproot_methods
import os.path
from datetime import datetime
import time
from collections import defaultdict
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
#from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from root_numpy import array2tree, array2root
from dnn_functions import *
from samplesAOD2018 import *
from tf_keras_model import *

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def _col_list(prefix,npf):
    return ['%s_%d'%(prefix,i) for i in range(npf)]


def get_FCN_jets_dataset(dataframe,features,weight,is_signal="is_signal",ignore_empty_jets=True):

    #if ignore_empty_jets:
    #    #print("\n")
    #    #print("    Ignore empty jets!!!!!!")
    #    #print("\n")
    #    dataframe = dataframe[ dataframe["Jet_pt"]>-1 ]

    X = dataframe[features].values.astype(dtype = 'float32')
    y = dataframe[is_signal].values.astype(dtype = 'float32')
    w = dataframe[weight].values.astype(dtype = 'float32')
        
    return X, y, w

def get_FCN_constituents_dataset(dataframe,n_points,features_var,weight,is_signal="is_signal",ignore_empty_jets=True):

    #if ignore_empty_jets:
    #    #print("\n")
    #    #print("    Ignore empty jets!!!!!!")
    #    #print("\n")
    #    dataframe = dataframe[ dataframe["Jet_pt"]>-1 ]
    
    #points_arr = []
    #features_arr = []

    features_list = []
    for n in range(n_points):
        for f_var in features_var:
            features_list.append(f_var+"_"+str(n))
    #L: do not stack!#features = np.stack(features_arr,axis=-1)
    #L: the _col_list was not doing what I thought. Beter a simpler loop
    #for p_var in points_var:
     #   points_arr.append(dataframe[_col_list(p_var,n_points)].values)
    #points = np.stack(points_arr,axis=-1)

    print("\n")
    print("Here features_list", features_list)
    print("Here the dataset")
    print(dataframe[features_list])
    
    X = dataframe[features_list].values
    y = dataframe[is_signal].values
    w = dataframe[weight].values
        
    return X, y, w

def get_FCN_jets_dataset_generator(dataframe,features,weight,is_signal="is_signal",ignore_empty_jets=True):

    while True:
        df = next(dataframe)
        #if ignore_empty_jets:
        #    #print("\n")
        #    #print("    Ignore empty jets!!!!!!")
        #    #print("\n")
        #    df = df[ df["Jet_pt"]>-1 ]

        X = df[features].values
        y = df[is_signal].values
        w = df[weight].values
        
        yield X, y, w

def get_BDT_dataset(dataframe,features,weight,is_signal="is_signal",ignore_empty_jets=True):

    #if ignore_empty_jets:
    #    #print("\n")
    #    #print("    Ignore empty jets!!!!!!")
    #    #print("\n")
    #    dataframe = dataframe[ dataframe["Jet_pt"]>-1 ]

    X = dataframe[features]
    y = dataframe[is_signal]
    w = dataframe[weight]
        
    return X, y, w

def get_BDT_DMatrix_dataset(dataframe,features,weight,is_signal="is_signal",ignore_empty_jets=True):

    #if ignore_empty_jets:
    #    #print("\n")
    #    #print("    Ignore empty jets!!!!!!")
    #    #print("\n")
    #    dataframe = dataframe[ dataframe["Jet_pt"]>-1 ]

    X = dataframe[features].values
    y = dataframe[is_signal].values
    w = dataframe[weight].values
    D = xgb.DMatrix(X, label=y, weight=w)
        
    return X, y, w, D

def get_particle_net_dataset(dataframe,n_points,points_var,features_var,mask_var,weight,is_signal="is_signal",ignore_empty_jets=True):

    #if ignore_empty_jets:
    #    #print("\n")
    #    #print("    Ignore empty jets!!!!!!")
    #    #print("\n")
    #    dataframe = dataframe[ dataframe["Jet_pt"]>-1 ]
        
    points_arr = []
    features_arr = []
    mask_arr = []
    
    for p_var in points_var:
        points_arr.append(dataframe[_col_list(p_var,n_points)].values)
    points = np.stack(points_arr,axis=-1)

    for f_var in features_var:
        features_arr.append(dataframe[_col_list(f_var,n_points)].values)
    features = np.stack(features_arr,axis=-1)
    
    for m_var in mask_var:
        mask_arr.append(dataframe[_col_list(m_var,n_points)].values)
    mask = np.stack(mask_arr,axis=-1)

    input_shapes = defaultdict()
    input_shapes['points'] = points.shape[1:]
    input_shapes['features'] = features.shape[1:]
    input_shapes['mask'] = mask.shape[1:]


    X = [points,features,mask]
    y = dataframe[is_signal].values
    w = dataframe[weight].values
        
    return X, y, w, input_shapes

def get_particle_net_jet_dataset(dataframe,n_points,points_var,features_var,mask_var,jet_var,weight,is_signal="is_signal",ignore_empty_jets=True):

    #if ignore_empty_jets:
    #    #print("\n")
    #    #print("    Ignore empty jets!!!!!!")
    #    #print("\n")
    #    dataframe = dataframe[ dataframe["Jet_pt"]>-1 ]
        
    points_arr = []
    features_arr = []
    mask_arr = []
    jet_arr = []
    
    for p_var in points_var:
        points_arr.append(dataframe[_col_list(p_var,n_points)].values)
    points = np.stack(points_arr,axis=-1)

    for f_var in features_var:
        features_arr.append(dataframe[_col_list(f_var,n_points)].values)
    features = np.stack(features_arr,axis=-1)
    
    for m_var in mask_var:
        mask_arr.append(dataframe[_col_list(m_var,n_points)].values)
    mask = np.stack(mask_arr,axis=-1)

    for j_var in jet_var:
        jet_arr.append(dataframe[j_var].values)
    jetvars = np.stack(jet_arr,axis=-1)#TBC

    input_shapes = defaultdict()
    input_shapes['points'] = points.shape[1:]
    input_shapes['features'] = features.shape[1:]
    input_shapes['mask'] = mask.shape[1:]
    input_shapes['jetvars'] = jetvars.shape[1:]


    X = [points,features,mask,jetvars]#TBC
    y = dataframe[is_signal].values
    w = dataframe[weight].values
        
    return X, y, w, input_shapes


def generator_batch(file_name,b_size):
    #print(file_name)
    batch_size = b_size
    store = pd.HDFStore(file_name)
    size = store.get_storer('df').shape
    print("Size: ",size)
    #print("\n")
    i_start = 0
    step = 0

    while True:
        #print("while true, size-start-step: ", size, i_start, step)
        if size >= i_start+batch_size:
            #print("before yield, size-start-step: ", size, i_start, step)
            foo = store.select('df',
                               start = i_start,
                               stop  = i_start + batch_size)
                        
            yield foo
            i_start += batch_size
            step += 1
            #print("after yield, size-start-step: ", size, i_start, step)

            if size < i_start+batch_size:
                #print("after yield, second if, i_start has incremented, size-start-step: ", size, i_start, step)
                #print("Closing " + file_name)
                print("\n")
                print("EOF, closing! ", file_name)
                store.close()
                #print("Closed " + file_name)

        else:
            #print("yield failed, size-start-step: ", size, i_start, step)
            #print("Opening " + file_name)
            print("\n")
            print("opening from scratch! ", file_name)
            store = pd.HDFStore(file_name)
            size = store.get_storer('df').shape
            #print("Opened " + file_name)

            i_start = 0
            #print("i_start back to zero, size-start-step: ", size, i_start, step)

def check(gen_s):
    s = next(gen_s)
    yield s

def concat_generators(gen_s, gen_b):
    s = next(gen_s)
    b = next(gen_b)
    df = pd.concat([s,b])
    yield df.sample(frac=1).reset_index(drop=True)

def concat_generators_list(gen_list):
    while True:
        next_list = []
        for gen in gen_list:
            next_list.append(next(gen))
        df = pd.concat(next_list)
        #normalize separately signal and background
        df['EventWeightNormalized'] = df['EventWeight']
        norm_s = df['EventWeight'][df['is_signal']==1].sum(axis=0)
        df.loc[df['is_signal']==1,'EventWeightNormalized'] = (df['EventWeight'][df['is_signal']==1]).div(norm_s)
        norm_b = df['EventWeight'][df['is_signal']==0].sum(axis=0)
        df.loc[df['is_signal']==0,'EventWeightNormalized'] = (df['EventWeight'][df['is_signal']==0]).div(norm_b)
        df['EventWeight2'] = df['EventWeight']
        yield df.sample(frac=1).reset_index(drop=True)

    '''
    Usage of generator:

    batch_size = 3
    df_chunck = gen('output.h5',batch_size)
    store = pd.HDFStore('output.h5')
    size = store.get_storer('df').shape[0]
    print("size: ", size)
    n_batches = int(size/batch_size)
    if(size%batch_size>0): n_batches+=1
    print("n. batches", n_batches)
    for n in range(n_batches):
    print("\n")
    print("Batch n. ", n)
    print(next(df_chunck))
    '''

##Please note!! This is work in progress!! Do NOT use yet!
def fit_generator(model_def,n_class,folder,result_folder,n_points,points,features,mask,is_signal,weight,use_weight,n_epochs,batch_size,patience_val,val_split,model_label="",ignore_empty_jets_train=True):

    file_train = folder+"train.h5"
    file_val = folder+"val.h5"

    store_train = pd.HDFStore(file_train)
    size_train = store_train.get_storer('df').shape
    store_train.close()

    store_val = pd.HDFStore(file_val)
    size_val = store_val.get_storer('df').shape
    store_val.close()

    print("Size train: ", size_train)
    print("Size val: ", size_val)

    n_steps_train = 1000#int(size_train/batch_size)
    n_steps_val = 1000#int(size_val/batch_size)
    
    print("steps train: ", n_steps_train)
    print("steps val: ", n_steps_val)

    #to create a generator:
    #generator_batch(s,n_batch_sampling)

    #Mixed background and signal, and shuffled
    df_gen_train = generator_batch(file_train,batch_size)
    df_gen_val   = generator_batch(file_val,batch_size)

    if model_label=="":
        model_label=model_def+"_"+timestampStr
        
    if not use_weight: model_label+="_no_weights"

    if not os.path.isdir(result_folder+'/model_'+model_def+"_"+model_label):
        os.mkdir(result_folder+'/model_'+model_def+"_"+model_label)

    result_folder += 'model_'+model_def+"_"+model_label+"/"
    
    print("\n")
    print("    Fitting model.....   ")
    print("\n")

    if(model_def=="FCN"):
        train_gen = get_FCN_jets_dataset_generator(df_gen_train,features,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
        train_val = get_FCN_jets_dataset_generator(df_gen_val,features,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
        model = get_FCN_jets(num_classes=n_class, input_shapes=(len(features),))
    else:
        print("    Model not recognized, abort . . .")
        exit()

    model.summary()

    ##Compile
    model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics = ["accuracy"])
    ##custom_opt:
    #custom_opt = keras.optimizers.Adam(learning_rate=0.001/10., beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name="custom_opt")
    ##Nadam
    #custom_opt = keras.optimizers.NAdam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="custom_opt")
    ##Adadelta
    #custom_opt = keras.optimizers.Adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-07, name="custom_opt")    
    #custom_opt = keras.optimizers.Adadelta(learning_rate=0.001/10., rho=0.95, epsilon=1e-07, name="custom_opt")    
    ##Adamax
    #custom_opt = keras.optimizers.Adamax(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="custom_opt")
    #custom_opt = keras.optimizers.Adamax(learning_rate=0.001/10., beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="custom_opt")
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=custom_opt, metrics = ["accuracy"])

    ##Callbacks
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience_val, verbose=0, mode='auto')
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=result_folder+'best_model_'+model_label+'.h5', monitor='val_loss', save_best_only=True)

    histObj = model.fit(train_gen, epochs=50, steps_per_epoch = n_steps_train, validation_data = train_val, validation_steps = n_steps_val, callbacks=[early_stop, checkpoint])##, use_multiprocessing=True, workers=6 )

    histObj.name='model_'+model_def+model_label # name added to legend
    plot = plotLearningCurves(histObj)# the above defined function to plot learning curves
    plot.savefig(result_folder+'loss_accuracy_'+model_label+'.png')
    plot.savefig(result_folder+'loss_accuracy_'+model_label+'.pdf')
    print("\n")
    print("    Plot saved in: ", result_folder+'loss_accuracy_'+model_label+'.png')
    output_file = 'model_'+model_label
    model.save(result_folder+output_file+'.h5')
    del model
    print("    Model saved in ", result_folder+output_file+'.h5')
    print("\n")
    #plot.show()


def fit_test(model_def,n_class,sign,back,folder,result_folder,n_points,points,features,mask,is_signal,weight,use_weight,n_epochs,n_batch_size,patience_val,val_split,model_label="",ignore_empty_jets_train=True):

    '''
    filename = "provetta.h5"
    batch_size = 5
    n_events = 20
    n_steps = int(n_events/batch_size)
    gen = generator_batch(filename,batch_size)
    #print(next(gen))
    for n in range(n_steps):
        print("Step n. ", n)
        print(next(gen))

    exit()
    '''
    #####

    print("GENERATOR???")
    print("\n")
    #features=["Jet_pt"]
    signal = []
    background = []
    signal_train = []
    background_train = []
    signal_val = []
    background_val = []
    for a in sign:
        signal += samples[a]['files']

    for b in back:
        background += samples[b]['files']

    for s in signal:
        signal_train.append(folder+s+"_train.h5")
        signal_val.append(folder+s+"_val.h5")

    for b in background:
        background_train.append(folder+b+"_train.h5")
        background_val.append(folder+b+"_val.h5")
    print(signal_train)
    print(background_train)

    n_batch_size = 500
    n_batch_sampling = int(n_batch_size/2)
    print("batch size: ", n_batch_size)
    print("batch sampling: ", n_batch_sampling)
    print("num of bkg: ", len(background_train))
    generator_list_train = []
    generator_list_val = []
    for s in signal_train:
        #df_gen_s = generator_batch(s,n_batch_size)
        generator_list_train.append(generator_batch(s,n_batch_sampling))
    for n,b in enumerate(background_train):
        #If more than one background, sample through backgrounds the same number of events
        print("\n")
        print("background: ", b)
        n_back = int(n_batch_sampling/len(background_train))
        if n==len(background_train)-1 and (n_batch_sampling%len(background_train)!=0):#add lefotver events
            n_back = int(n_batch_sampling/len(background_train)) + n_batch_sampling%len(background_train)
        print("sampled events per bkg: ", n_back)
        print("\n")
        #df_gen_b = generator_batch(b,n_back)
        generator_list_train.append(generator_batch(b,n_back))

    for s in signal_val:
        generator_list_val.append(generator_batch(s,n_batch_sampling))
    for n,b in enumerate(background_val):
        #If more than one background, sample through backgrounds the same number of events
        n_back = int(n_batch_sampling/len(background_val))
        if n==len(background_val)-1 and (n_batch_sampling%len(background_val)!=0):#add lefotver events
            n_back = int(n_batch_sampling/len(background_val)) + n_batch_sampling%len(background_val)
        generator_list_val.append(generator_batch(b,n_back))

    #Mixed background and signal, and shuffled
    df_gen_train = concat_generators_list(generator_list_train)
    df_gen_val   = concat_generators_list(generator_list_val)
    print("check gen train shape: ", next(df_gen_train).shape)

    #df_gen_s = generator_batch(signal_train[0],n_batch_size)
    #test = check(df_gen_s)
    #wait for that
    #X_train_gen, y_train_gen, w_train_gen =

    train_gen = get_FCN_jets_dataset_generator(concat_generators_list(generator_list_train),features,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
    train_val = get_FCN_jets_dataset_generator(concat_generators_list(generator_list_val),features,weight=weight,is_signal="is_signal",ignore_empty_jets=True)

    model = get_FCN_jets(num_classes=n_class, input_shapes=(len(features),))        
    model.summary()
    ##Compile
    model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics = ["accuracy"])
    #custom_opt:
    #custom_opt = keras.optimizers.Adam(learning_rate=0.001/2., beta_1=0.9, beta_2=0.999, amsgrad=False)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=custom_opt, metrics = ["accuracy"])

    ##Callbacks
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience_val, verbose=0, mode='auto')
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=result_folder+'best_model_'+model_label+'.h5', monitor='val_loss', save_best_only=True)

    train_steps = 100000/n_batch_size
    histObj = model.fit(train_gen, epochs=50, steps_per_epoch = train_steps, validation_data = train_val, validation_steps = int(train_steps*0.2), callbacks=[early_stop, checkpoint])##, use_multiprocessing=True, workers=6 )
    plot = plotLearningCurves(histObj)# the above defined function to plot learning curves
    plot.show()
    for a in range(1):
        print("This is the loop number ",a, "!")
        #print( next( concat_generators_list(generator_list_train) ) )
        ###print(next(df_gen_s))
        ###print(next(df_gen_b))
        #a = next(df_gen_train)
        #print(a)

        #wait for that
        #X_train, y_train, w_train = get_FCN_jets_dataset( a ,features,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
        X_train, y_train, w_train = next(train_gen)
        print("X_train: ", X_train)
        print("y_train: ", y_train)
        print("w_train: ", w_train)
        print(X_train.shape[1:])
        print(np.array(len(features)))
        #X_train_gen, y_train_gen, w_train_gen = get_FCN_jets_dataset_generator( concat_generators_list(generator_list_train) ,features,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
    exit()

    if(model_def=="FCN"):
        X_train, y_train, w_train = get_FCN_jets_dataset(df_train,features,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
        X_val,   y_val,   w_val   = get_FCN_jets_dataset(df_val,features,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
        model = get_FCN_jets(num_classes=n_class, input_shapes=X_train.shape[1:])        
    else:
        print("    Model not recognized, abort . . .")
        exit()

    model.summary()

    if model_label=="":
        model_label=model_def+"_"+timestampStr
        
    if not use_weight: model_label+="_no_weights"

    if not os.path.isdir(result_folder+'/model_'+model_def+"_"+model_label):
        os.mkdir(result_folder+'/model_'+model_def+"_"+model_label)

    result_folder += 'model_'+model_def+"_"+model_label+"/"
    
    print("\n")
    print("    Fitting model.....   ")
    print("\n")
    
    ##Compile
    model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics = ["accuracy"])
    #custom_opt:
    #custom_opt = keras.optimizers.Adam(learning_rate=0.001/2., beta_1=0.9, beta_2=0.999, amsgrad=False)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=custom_opt, metrics = ["accuracy"])

    ##Callbacks
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience_val, verbose=0, mode='auto')
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=result_folder+'best_model_'+model_label+'.h5', monitor='val_loss', save_best_only=True)


    ##Fit model
    #train is 60%, test is 20%, val is 20%
    if use_weight:
        histObj = model.fit(X_train, y_train, epochs=n_epochs, batch_size=n_batch_size, sample_weight=w_train, validation_split=val_split, validation_data=None if val_split>0 else (X_val, y_val, w_val), callbacks=[early_stop, checkpoint])
    else:
        histObj = model.fit(X_train, y_train, epochs=n_epochs, batch_size=n_batch_size, validation_split=val_split, validation_data=None if val_split>0 else (X_val, y_val), callbacks=[early_stop, checkpoint])

    histObj.name='model_'+model_def+model_label # name added to legend
    plot = plotLearningCurves(histObj)# the above defined function to plot learning curves
    plot.savefig(result_folder+'loss_accuracy_'+model_label+'.png')
    plot.savefig(result_folder+'loss_accuracy_'+model_label+'.pdf')
    print("\n")
    print("    Plot saved in: ", result_folder+'loss_accuracy_'+model_label+'.png')
    output_file = 'model_'+model_label
    model.save(result_folder+output_file+'.h5')
    del model
    print("    Model saved in ", result_folder+output_file+'.h5')
    print("\n")
    #plot.show()
    


def fit_model(model_def,n_class,folder,result_folder,n_points,points,features,mask,jvars,is_signal,weight,use_weight,n_epochs,n_batch_size,patience_val,val_split,model_label="",ignore_empty_jets_train=True):

    ##Read train/validation sample
    store_train = pd.HDFStore(folder+"train.h5")
    df_train = store_train.select("df")
    store_val = pd.HDFStore(folder+"val.h5")
    df_val = store_val.select("df")
    
    if(model_def=="FCN"):
        X_train, y_train, w_train = get_FCN_jets_dataset(df_train,features,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
        X_val,   y_val,   w_val   = get_FCN_jets_dataset(df_val,features,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
        model = get_FCN_jets(num_classes=n_class, input_shapes=X_train.shape[1:])
    elif(model_def=="FCN_constituents"):
        X_train, y_train, w_train = get_FCN_constituents_dataset(df_train,n_points,features+points,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
        X_val,   y_val,   w_val   = get_FCN_constituents_dataset(df_val,n_points,features+points,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
        model = get_FCN_constituents(num_classes=n_class, input_shapes=X_train.shape[1:])    
    elif(model_def=="particle_net_lite"):
        X_train, y_train, w_train, input_shapes = get_particle_net_dataset(df_train,n_points,points,features,mask,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
        X_val,   y_val,   w_val, _   = get_particle_net_dataset(df_val,n_points,points,features,mask,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
        model = get_particle_net_lite(n_class, input_shapes, contains_angle = True if 'phi' in points else False)
# Julia: add particle_net        
    elif(model_def=="particle_net"):
        X_train, y_train, w_train, input_shapes = get_particle_net_dataset(df_train,n_points,points,features,mask,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
        X_val,   y_val,   w_val, _   = get_particle_net_dataset(df_val,n_points,points,features,mask,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
        model = get_particle_net(n_class, input_shapes, contains_angle = True if 'phi' in points else False) 
    elif(model_def=="particle_net_jet"):#TBC
        print("L: here go to particle net jet")
        X_train, y_train, w_train, input_shapes = get_particle_net_jet_dataset(df_train,n_points,points,features,mask,jvars,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
        X_val,   y_val,   w_val, _   = get_particle_net_jet_dataset(df_val,n_points,points,features,mask,jvars,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
        model = get_particle_net_jet(n_class, input_shapes, contains_angle = True if 'phi' in points else False) 
        
    else:
        print("    Model not recognized, abort . . .")
        exit()

    model.summary()

    if model_label=="":
        model_label=model_def+"_"+timestampStr
        
    if not use_weight: model_label+="_no_weights"

    if not os.path.isdir(result_folder+'/model_'+model_def+"_"+model_label):
        os.mkdir(result_folder+'/model_'+model_def+"_"+model_label)

    result_folder += 'model_'+model_def+"_"+model_label+"/"
    
    print("\n")
    print("    Fitting model.....   ")
    print("\n")
    
    ##Compile
    model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics = ["accuracy"])
    #custom_opt:
    #custom_opt = keras.optimizers.Adam(learning_rate=0.001/2., beta_1=0.9, beta_2=0.999, amsgrad=False)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=custom_opt, metrics = ["accuracy"])

    ##Callbacks
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience_val, verbose=0, mode='auto')
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=result_folder+'best_model_'+model_label+'.h5', monitor='val_loss', save_best_only=True)

    #Input normalization!
    #print('n')
    #print(' INPUT NORM! ')
    #scaler_t = StandardScaler()
    #scaled_X_train = scaler_t.fit_transform(X_train)
    #scaler_v = StandardScaler()
    #scaled_X_val = scaler_v.fit_transform(X_val)

    #X_train = scaled_X_train
    #X_val = scaled_X_val

    ##Fit model
    #train is 60%, test is 20%, val is 20%
    if use_weight:
        histObj = model.fit(X_train, y_train, epochs=n_epochs, batch_size=n_batch_size, sample_weight=w_train, validation_split=val_split, validation_data=None if val_split>0 else (X_val, y_val, w_val), callbacks=[early_stop, checkpoint])
    else:
        histObj = model.fit(X_train, y_train, epochs=n_epochs, batch_size=n_batch_size, validation_split=val_split, validation_data=None if val_split>0 else (X_val, y_val), callbacks=[early_stop, checkpoint])

    histObj.name='model_'+model_def+model_label # name added to legend
    plot = plotLearningCurves(histObj)# the above defined function to plot learning curves
    plot.savefig(result_folder+'loss_accuracy_'+model_label+'.png')
    plot.savefig(result_folder+'loss_accuracy_'+model_label+'.pdf')
    print("\n")
    print("    Plot saved in: ", result_folder+'loss_accuracy_'+model_label+'.png')
    output_file = 'model_'+model_label
    model.save(result_folder+output_file+'.h5')
    del model
    print("    Model saved in ", result_folder+output_file+'.h5')
    print("\n")
    #plot.show()

def fit_BDT(model_def,n_class,folder,result_folder,n_points,points,features,mask,is_signal,weight,use_weight,n_epochs,n_batch_size,patience_val,val_split,model_label="",ignore_empty_jets_train=True):

    ##Read train/validation sample
    store_train = pd.HDFStore(folder+"train.h5")
    df_train = store_train.select("df")
    store_val = pd.HDFStore(folder+"val.h5")
    df_val = store_val.select("df")
    
    #X_train, y_train, w_train, D_train = get_BDT_DMatrix_dataset(df_train,features,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
    #D_test = get_BDT_DMatrix_dataset(df_test,features,weight=weight,is_signal="is_signal",ignore_empty_jets=True)

    if(model_def=="BDT"):
        X_train, y_train, w_train = get_BDT_dataset(df_train,features,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
        X_val,   y_val,   w_val   = get_BDT_dataset(df_val,features,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
        model = get_BDT(n_epochs,features)##num_classes=n_class, input_shapes=X_train.shape[1:])
    else:
        print("    Model not recognized, abort . . .")
        exit()

    print(model)
    if model_label=="":
        model_label=model_def+"_"+timestampStr
        
    if not use_weight: model_label+="_no_weights"

    if not os.path.isdir(result_folder+'/model_'+model_def+"_"+model_label):
        os.mkdir(result_folder+'/model_'+model_def+"_"+model_label)

    result_folder += 'model_'+model_def+"_"+model_label+"/"

    print("\n")
    print("    Fitting model.....   ")
    print("\n")
    
    
    ##Compile
    ##model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics = ["accuracy"])
    ##custom_opt:
    ##custom_opt = keras.optimizers.Adam(learning_rate=0.001/2., beta_1=0.9, beta_2=0.999, amsgrad=False)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=custom_opt, metrics = ["accuracy"])

    ###Callbacks
    #early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience_val, verbose=0, mode='auto')
    #checkpoint = keras.callbacks.ModelCheckpoint(filepath=result_folder+'best_model_'+model_label+'.h5', monitor='val_loss', save_best_only=True)


    ##Fit model
    #train is 60%, test is 20%, val is 20%
    
    ########  ### L
    ##print("Attempt at training DMatrix . . .")
    #pickle.load(open("model_weights/v3_calo_AOD_2018_dnn_balance_val_train_new_presel/model_BDT_SampleWeight_NoMedian/model_SampleWeight_NoMedian.dat","rb"))
    #print(model.get_booster().get_dump(dump_format="json"))
    #exit()
    # 
    #X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
    #D_train = xgb.DMatrix(X_train.values,label = y_train.values,weight=w_train.values)
    #D_val =   xgb.DMatrix(X_val.values,label = y_val.values,weight=w_val.values)
    
    #param = {
    #'eta': 0.3, 
    #'max_depth': 4,  
    #'objective': 'multi:softmax',
    #'eval_metric' : ["error","logloss"],
    #'num_class': 2} 

    #steps = 50  # The number of training iterations
    ##model = xgb.train(param, D_train, steps)
    ##print("Save model...")
    ##model.get_dump("/nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/BDT_dump.txt")#crash
    
    #model.fit(D_train, eval_metric=["error","logloss"], early_stopping_rounds = patience_val)
    ##from sklearn.metrics import precision_score, recall_score, accuracy_score
    ##preds = model.predict(D_train)
    ##best_preds = np.asarray([np.argmax(line) for line in preds])
    #print("Precision = {}".format(precision_score(X_train, best_preds, average='macro')))
    ##print("Recall = {}".format(recall_score(Y_test, best_preds, average='macro')))
    #print("Accuracy = {}".format(accuracy_score(y_train, best_preds)))
    ##histObj = model.evals_result()
    #exit()
    ####### ### L
    
    if use_weight:
        print("yes weight")
        eval_set = [(X_train, y_train), (X_val, y_val)]
        sample_w = [w_train,w_val]
        model.fit(X_train, y_train, sample_weight=w_train, eval_set = eval_set, sample_weight_eval_set=sample_w, eval_metric=["error","logloss"], early_stopping_rounds = patience_val)
        histObj = model.evals_result()

    else:
        print("no weight")
        eval_set = [(X_train, y_train), (X_val, y_val)]
        model.fit(X_train, y_train, eval_set = eval_set, eval_metric=["error","logloss"], early_stopping_rounds = patience_val)
        histObj = model.evals_result()

    histObj_name='model_'+model_def+model_label # name added to legend
    plot = plotLearningCurvesBDT(histObj,histObj_name)# the above defined function to plot learning curves
    #plot.show()
    plot.savefig(result_folder+'loss_accuracy_'+model_label+'.png')
    plot.savefig(result_folder+'loss_accuracy_'+model_label+'.pdf')
    print("\n")
    print("    Plot saved in: ", result_folder+'loss_accuracy_'+model_label+'.png')
    output_file = 'model_'+model_label
    pickle.dump(model, open(result_folder+output_file+".dat", "wb"))
    ####model.save_model(result_folder+output_file)
    del model
    print("    Model saved in ", result_folder+output_file+".dat")
    print("\n")
    #plot.show()


def evaluate_model(model_def,n_class,folder,result_folder,n_points,points,features,mask,jvars,is_signal,jet_matching,weight,use_weight,n_batch_size,model_label,signal_match_test,ignore_empty_jets_test):

    print("\n")
    print("    Evaluating performances of the model.....   ")

    cut_fpr = 0.0006325845

    ##Read test sample
    store = pd.HDFStore(folder+"test.h5")
    df_test = store.select("df")

    print("    Remove negative weights at testing!!!!!!")
    df_test = df_test.loc[df_test['EventWeight']>=0]

    add_string = ""
    #if ignore_empty_jets_test:
    #    #print("\n")
    #    #print("    Ignore empty jets at testing!!!!!!")
    #    #print("\n")
    #    df_test = df_test.loc[df_test["Jet_pt"]>-1]
    #    #add_string+="_ignore_empty_jets"

    if signal_match_test:
        #print("\n")
        #print("    Ignore not matched jets in signal at testing!!!!!!")
        #print("\n")
        df_s = df_test.loc[df_test[is_signal]==1]
        df_b = df_test.loc[df_test[is_signal]==0]
        df_s = df_s.loc[df_s[jet_matching]==1]
        df_test = pd.concat([df_b,df_s])
        #print(df_test.shape[0],df_s.shape[0],df_b.shape[0])
        add_string+="_signal_matched_"+jet_matching

    
    if(model_def=="FCN"):
        X_test, y_test, w_test = get_FCN_jets_dataset(df_test,features,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
    elif(model_def=="FCN_constituents"):
        #Lisa: here you also need features+points
        X_test, y_test, w_test = get_FCN_constituents_dataset(df_test,n_points,features+points,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
    elif(model_def=="particle_net_lite" or model_def=="particle_net"):
        X_test, y_test, w_test, input_shapes = get_particle_net_dataset(df_test,n_points,points,features,mask,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
    elif(model_def=="particle_net_jet"):
        X_test, y_test, w_test, input_shapes = get_particle_net_jet_dataset(df_test,n_points,points,features,mask,jvars,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
    else:
        print("    Model not recognized, abort . . .")
        exit()

    #Input normalization!
    #print('n')
    #print(' INPUT NORM! ')
    #scaler = StandardScaler()
    #scaled_X_test = scaler.fit_transform(X_test)

    #X_test = scaled_X_test


    if model_label=="":
        model_label=model_def+"_"+timestampStr

    if not use_weight: model_label+="_no_weights"

    result_folder += 'model_'+model_def+"_"+model_label+"/"
    output_file = 'best_model_'+model_label

    #if not os.path.isdir(result_folder+'/model_'+model_def+"_"+model_label):
    #    print("Result folder ",result_folder, " does not exist! Have you trained the model? Aborting . . .")
    #    exit()

    print("    Loading model... ", result_folder+output_file+'.h5')
    print("\n")
    model = keras.models.load_model(result_folder+output_file+'.h5')
    model.summary()
    print("\n")
    print("    Running on test sample. This may take a moment. . .")


    probs = model.predict(X_test)#predict probability over test sample
    #print("Negative weights?")
    #print(df_test[df_test[weight]<0])
    #df_test = df_test[df_test[weight]>=0]
    #print(df_test)

    if use_weight:
        AUC = roc_auc_score(y_test, probs[:,1], sample_weight=w_test)        
    else:
        AUC = roc_auc_score(y_test, probs[:,1])

    print("\n")
    print("    Test Area under Curve = {0}".format(AUC))
    print("\n")
    df_test["sigprob"] = probs[:,1]
    print(" ... x-check, probs size: \n", df_test["sigprob"])

    df_test.to_hdf(result_folder+'test_score_'+model_label+add_string+'.h5', 'df', format='fixed')
    print("    "+result_folder+"test_score_"+model_label+add_string+".h5 stored")

    back = np.array(df_test["sigprob"].loc[df_test[is_signal]==0].values)
    sign = np.array(df_test["sigprob"].loc[df_test[is_signal]==1].values)
    back_w = np.array(df_test[weight].loc[df_test[is_signal]==0].values)
    sign_w = np.array(df_test[weight].loc[df_test[is_signal]==1].values)
    #saves the df_test["sigprob"] column when the event is signal or background
    print(" ... x-check, back size: ", len(back))
    print(" ... x-check, sign size: ", len(sign))
    print(" ... x-check, back naive integral: ", sum(back))
    print(" ... x-check, sign naive integral: ", sum(sign))
    print(" ... x-check, back weight integral: ", sum(back_w))
    print(" ... x-check, sign weight integral: ", sum(sign_w))
    print(" ... x-check, back comb integral: ", sum(back_w*back))
    print(" ... x-check, sign comb integral: ", sum(sign_w*sign))
    plt.figure(figsize=(8,7))
    plt.rcParams.update({'font.size': 15}) #Larger font size
    #Let's plot an histogram:
    # * y-values: back/sign probabilities
    # * 50 bins
    # * alpha: filling color transparency
    # * density: it should normalize the histograms to unity

    if use_weight:
        nb, binsb, _ = plt.hist(back, np.linspace(0,1,50), weights=back_w, color='blue', edgecolor='blue', lw=2, label='background', alpha=0.3)#, density=True)
        ns, binss, _ = plt.hist(sign, np.linspace(0,1,50), weights=sign_w, color='red', edgecolor='red', lw=2, label='signal', alpha=0.3)#, density=True)
    else:
        nb, binsb, _ = plt.hist(back, np.linspace(0,1,50), color='blue', edgecolor='blue', lw=2, label='background', alpha=0.3)#, density=True)
        ns, binss, _ = plt.hist(sign, np.linspace(0,1,50), color='red', edgecolor='red', lw=2, label='signal', alpha=0.3)#, density=True)

    bin_widthb = binsb[1] - binsb[0]
    bin_widths = binss[1] - binss[0]
    integrals = bin_widths * sum(ns)
    integralb = bin_widthb * sum(nb)
    print("Integral S: ",integrals)
    print("Integral B: ",integralb)

    plt.xlim([0.0, 1.05])
    plt.xlabel('Event probability of being classified as signal')
    plt.legend(loc="upper right", title=model_label)
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(result_folder+'probability_'+output_file+add_string+'.png')
    plt.savefig(result_folder+'probability_'+output_file+add_string+'.pdf')

    if use_weight:
        fpr, tpr, thresholds = roc_curve(df_test[is_signal], df_test["sigprob"])
        idx, _ = find_nearest(fpr,cut_fpr)
    else:
        fpr, tpr, thresholds = roc_curve(df_test[is_signal], df_test["sigprob"], sample_weight=w_test)
        idx, _ = find_nearest(fpr,cut_fpr)

    plt.figure(figsize=(8,7))
    plt.rcParams.update({'font.size': 15}) #Larger font size
    plt.plot(fpr, tpr, color='crimson', lw=2, label='AUC = {0:.4f}'.format(AUC))
    plt.plot(fpr[idx], tpr[idx],'ro',color='crimson',label="w.p. {0:.4f}".format(thresholds[idx]))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot(0.0006325845,0.22,'ro',color='blue',label="cut based")
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right", title=model_label)
    plt.grid(True)
    plt.savefig(result_folder+'ROC_'+output_file+add_string+'.pdf')
    plt.savefig(result_folder+'ROC_'+output_file+add_string+'.png')
    #plt.show()
    print("    Plots printed in "+result_folder)
    print("\n")

    plt.figure(figsize=(8,7))
    plt.rcParams.update({'font.size': 15}) #Larger font size
    plt.plot(fpr, tpr, color='crimson', lw=2, label='AUC = {0:.4f}'.format(AUC))
    plt.plot(fpr[idx], tpr[idx],'ro',color='crimson',label="w.p. {0:.4f}".format(thresholds[idx]))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot(0.0006325845,0.22,'ro',color='blue',label="cut based")
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right", title=model_label)
    plt.grid(True)
    plt.xlim([0.0001, 1.05])
    plt.xscale('log')
    plt.savefig(result_folder+'ROC_'+output_file+add_string+'_logx.pdf')
    plt.savefig(result_folder+'ROC_'+output_file+add_string+'_logx.png')
    #plt.show()


def evaluate_BDT(model_def,n_class,folder,result_folder,n_points,points,features,mask,is_signal,jet_matching,weight,use_weight,n_batch_size,model_label,signal_match_test,ignore_empty_jets_test):

    print("\n")
    print("    Evaluating performances of the model.....   ")

    ##Read test sample
    store = pd.HDFStore(folder+"test.h5")
    df_test_pre = store.select("df")

    print("    Remove negative weights at testing!!!!!!")
    #df_test_pre = df_test_pre.loc[(df_test_pre['EventWeight']>=0) & (df_test_pre['Jet_pt']>-1)]
    df_test_pre = df_test_pre.loc[df_test_pre['EventWeight']>=0]

    add_string = ""

    if signal_match_test:
        print("\n")
        print("    Ignore not matched jets in signal at testing!!!!!!")
        print("\n")
        df_s = df_test_pre.loc[(df_test_pre[is_signal]==1) & (df_test_pre[jet_matching]==1)]
        df_b = df_test_pre.loc[df_test_pre[is_signal]==0]
        df_test = pd.concat([df_b,df_s])
        add_string+="_signal_matched_"+jet_matching
    else:
        df_test = df_test_pre

    if(model_def=="BDT"):
        X_test, y_test, w_test = get_BDT_dataset(df_test,features,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
    else:
        print("    Model not recognized, abort . . .")
        exit()

    if model_label=="":
        model_label=model_def+"_"+timestampStr

    if not use_weight: model_label+="_no_weights"

    result_folder += 'model_'+model_def+"_"+model_label+"/"
    output_file = 'model_'+model_label

    #if not os.path.isdir(result_folder+'/model_'+model_def+"_"+model_label):
    #    print("Result folder ",result_folder, " does not exist! Have you trained the model? Aborting . . .")
    #    exit()

    print("    Loading model... ", result_folder+output_file)
    print("\n")
    # load model from file
    model = pickle.load(open(result_folder+output_file+".dat", "rb"))
    print(model)
    
    print("\n")
    print("    Running on test sample. This may take a moment. . .")

    #model.get_booster().feature_names = features
    #print(model.get_booster().feature_names)

    fig, ax = plt.subplots(figsize=(20, 10))#12,8
    xgb.plot_importance(model, max_num_features=len(features), xlabel="F score (weight)",ax=ax)
    plt.savefig(result_folder+'feature_importance_'+output_file+add_string+'.pdf')
    plt.savefig(result_folder+'feature_importance_'+output_file+add_string+'.png')

    probs = model.predict_proba(X_test)#predict probability over test sample
    #print("BDT probs: ", probs[:,1])
    #print("Negative weights?")
    #print(df_test[df_test[weight]<0])
    #df_test = df_test[df_test[weight]>=0]
    #print(df_test)

    if use_weight:
        AUC = roc_auc_score(y_test, probs[:,1], sample_weight=w_test)        
    else:
        AUC = roc_auc_score(y_test, probs[:,1])

    print("\n")
    print("    Test Area under Curve = {0}".format(AUC))
    print("\n")
    df_test["sigprob"] = probs[:,1]
    print(" ... x-check, probs size: \n", df_test["sigprob"])

    df_test.to_hdf(result_folder+'test_score_'+model_label+add_string+'.h5', 'df', format='fixed')
    print("    "+result_folder+"test_score_"+model_label+add_string+".h5 stored")

    back = np.array(df_test["sigprob"].loc[df_test[is_signal]==0].values)
    sign = np.array(df_test["sigprob"].loc[df_test[is_signal]==1].values)
    back_w = np.array(df_test[weight].loc[df_test[is_signal]==0].values)
    sign_w = np.array(df_test[weight].loc[df_test[is_signal]==1].values)
    print(" ... x-check, back size: ", len(back))
    print(" ... x-check, sign size: ", len(sign))
    print(" ... x-check, back naive integral: ", sum(back))
    print(" ... x-check, sign naive integral: ", sum(sign))
    print(" ... x-check, back weight integral: ", sum(back_w))
    print(" ... x-check, sign weight integral: ", sum(sign_w))
    print(" ... x-check, back comb integral: ", sum(back_w*back))
    print(" ... x-check, sign comb integral: ", sum(sign_w*sign))
    #saves the df_test["sigprob"] column when the event is signal or background
    plt.figure(figsize=(8,7))
    plt.rcParams.update({'font.size': 15}) #Larger font size
    #Let's plot an histogram:
    # * y-values: back/sign probabilities
    # * 50 bins
    # * alpha: filling color transparency
    # * density: it should normalize the histograms to unity

    if use_weight:
        nb, binsb, _ = plt.hist(back, np.linspace(0,1,50), weights=back_w, color='blue', edgecolor='blue', lw=2, label='background', alpha=0.3)#, density=True)
        ns, binss, _ = plt.hist(sign, np.linspace(0,1,50), weights=sign_w, color='red', edgecolor='red', lw=2, label='signal', alpha=0.3)#, density=True)
    else:
        nb, binsb, _ = plt.hist(back, np.linspace(0,1,50), color='blue', edgecolor='blue', lw=2, label='background', alpha=0.3)#, density=True)
        ns, binss, _ = plt.hist(sign, np.linspace(0,1,50), color='red', edgecolor='red', lw=2, label='signal', alpha=0.3)#, density=True)

    bin_widthb = binsb[1] - binsb[0]
    bin_widths = binss[1] - binss[0]
    integrals = bin_widths * sum(ns)
    integralb = bin_widthb * sum(nb)
    print("Integral S: ",integrals)
    print("Integral B: ",integralb)
    plt.xlim([-0.05, 1.05])
    plt.xlabel('Event probability of being classified as signal')
    plt.legend(loc="upper right")
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(result_folder+'probability_'+output_file+add_string+'.png')
    plt.savefig(result_folder+'probability_'+output_file+add_string+'.pdf')

    if use_weight:
        fpr, tpr, _ = roc_curve(df_test[is_signal], df_test["sigprob"])
    else:
        fpr, tpr, _ = roc_curve(df_test[is_signal], df_test["sigprob"], sample_weight=w_test)

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
    print("    Plots printed in "+result_folder)
    print("\n")

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



def compare_models(model_list,result_folder,is_signal,weight_list,use_weight,model_labels,plot_labels,signal_match_test,ignore_empty_jets_test,jets="AK4"):

    sigprob = {}
    target = {}
    weight = {}
    AUC = {}
    fpr = {}
    tpr = {}
    thresholds = {}
    idx = {}
    add_string = ""

    cut_fpr = 0.0006325845


    #if ignore_empty_jets_test:
    #    add_string+="_ignore_empty_jets"

    if signal_match_test:
        if jets=="AK4":
            add_string+="_signal_matched_Jet_isGenMatchedCaloCorrLLPAccept"
        else:
            add_string+="_signal_matched_FatJet_isGenMatchedCaloCorrLLPAccept"

    orig_result = result_folder
    for i,m in enumerate(model_list):
        if not use_weight: model_labels[i]+="_no_weights"
        result_folder = orig_result + 'model_'+m+"_"+model_labels[i]+"/"
        print("Opening: ", result_folder+'test_score_'+model_labels[i]+add_string+'.h5')
        store = pd.HDFStore(result_folder+'test_score_'+model_labels[i]+add_string+'.h5')
        df = store.select("df")
        print(df)
        print(df[is_signal])
        print(df["sigprob"])
        print(df[weight_list[i]].values)
        print(df[weight_list[i]].values.shape)
        #print( "AUC: ",roc_auc_score(df[is_signal], df["sigprob"],sample_weight = df[weight]) )
        sigprob[i] = df["sigprob"].values
        target[i] = df[is_signal].values
        weight[i] = df[weight_list[i]].values
        if use_weight:
            AUC[i] = roc_auc_score(target[i], sigprob[i], sample_weight=weight[i])        
            fpr[i], tpr[i], thresholds[i] = roc_curve(target[i], sigprob[i], sample_weight=weight[i])
            idx[i], _ = find_nearest(fpr[i],cut_fpr)
        else:
            AUC[i] = roc_auc_score(target[i], sigprob[i])
            fpr[i], tpr[i], thresholds[i] = roc_curve(target[i], sigprob[i])
            idx[i], _ = find_nearest(fpr[i],cut_fpr)

        del df
        store.close()
        del store

    colors = ['crimson','green','skyblue','orange','gray','magenta','chocolate','yellow','black','olive']
    linestyles = ['-', '--', '-.', ':','-','--','-.',':']
    linestyles = ['-', '--', '-.', '-','--','-.',':','-', '--', '-.',]

    plt.figure(figsize=(8,7))
    plt.rcParams.update({'font.size': 15}) #Larger font size
    for i,m in enumerate(model_list):
        #lab = m if not m=="LEADER" else "FCN"
        #plt.plot(fpr[i], tpr[i], color=colors[i], linestyle=linestyles[i], lw=2, label=lab+' (AUC = {0:.4f})'.format(AUC[i]))
        plt.plot(fpr[i], tpr[i], color=colors[i], linestyle=linestyles[i], lw=2, label=plot_labels[i]+' (AUC = {0:.4f})'.format(AUC[i]))
        plt.plot(fpr[i][idx[i]], tpr[i][idx[i]],'ro',color=colors[i],label="w.p. {0:.4f}".format(thresholds[i][idx[i]]))
        #plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label=m+model_labels[i]+' (AUC = {0:.4f})'.format(AUC[i]))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot(0.0006325845,0.22,'ro',color='blue',label="cut based")

    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(orig_result+'Compare_ROC_'+add_string+'.pdf')
    plt.savefig(orig_result+'Compare_ROC_'+add_string+'.png')
    #plt.show()
    print("    Plots printed in "+orig_result)
    print("\n")

    plt.figure(figsize=(8,7))
    plt.rcParams.update({'font.size': 15}) #Larger font size
    for i,m in enumerate(model_list):
        lab = m if not m=="LEADER" else "FCN"
        #plt.plot(fpr[i], tpr[i], color=colors[i], linestyle=linestyles[i], lw=2, label=lab+' (AUC = {0:.4f})'.format(AUC[i]))
        plt.plot(fpr[i], tpr[i], color=colors[i], linestyle=linestyles[i], lw=2, label=plot_labels[i]+' (AUC = {0:.4f})'.format(AUC[i]))
        plt.plot(fpr[i][idx[i]], tpr[i][idx[i]],'ro',color=colors[i],label="w.p. {0:.4f}".format(thresholds[i][idx[i]]))
        #plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label=m+model_labels[i]+' (AUC = {0:.4f})'.format(AUC[i]))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot(0.0006325845,0.22,'ro',color='blue',label="cut based")
    plt.ylim([0.0, 1.05])

    print('\n')
    print('Example: take the first entry')
    print(tpr.keys())
    print(tpr[0])
    print(fpr[0])
    print(thresholds[0])
    idx, val = find_nearest(fpr[0],cut_fpr)
    print('closest?')
    print(idx, val, fpr[0][idx], tpr[0][idx], thresholds[0][idx], cut_fpr, 0.22)
    
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="center right")
    plt.grid(True)
    plt.xlim([0.0001, 1.05])
    plt.xscale('log')
    plt.savefig(orig_result+'Compare_ROC_'+add_string+'_logx.pdf')
    plt.savefig(orig_result+'Compare_ROC_'+add_string+'_logx.png')
    #plt.show()


    '''
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

    if use_weight:
        plt.hist(back, 50, color='blue', edgecolor='blue', lw=2, label='background', alpha=0.3)#, density=True)
        plt.hist(sign, 50, color='red', edgecolor='red', lw=2, label='signal', alpha=0.3)#, density=True)
    else:
        plt.hist(back, 50, weights=back_w, color='blue', edgecolor='blue', lw=2, label='background', alpha=0.3)#, density=True)
        plt.hist(sign, 50, weights=sign_w, color='red', edgecolor='red', lw=2, label='signal', alpha=0.3)#, density=True)

    plt.xlim([0.0, 1.05])
    plt.xlabel('Event probability of being classified as signal')
    plt.legend(loc="upper right")
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(result_folder+'probability_'+output_file+add_string+'.png')
    plt.savefig(result_folder+'probability_'+output_file+add_string+'.pdf')
    '''



# # # # # # # # # #
# These functions must be tested for graphnet:
'''
def write_discriminator_output(model_def,folder,model_folder,result_folder,event_dict, jet_dict,nj,features,jvar,event_var,is_signal,jet_matching,weight,n_batch_size,model_label,sample_list=[]):
    if model_label=="":
        model_label=timestampStr

    model_folder += 'model_'+model_def+"_"+model_label+"/"
    output_file = 'best_model_'+model_label

    print("Loading model... ", model_folder+output_file+'.h5')
    model = keras.models.load_model(model_folder+output_file+'.h5')
    model.summary()
    print("Running on test sample. This may take a moment. . .")
    
    if sample_list==[]:
        ##Read test sample
        store = pd.HDFStore(folder+"test_"+model_label+".h5")
        df = store.select("df")
        df_valid = df.loc[df[jet_matching]!=-1]
        df_invalid = df.loc[df[jet_matching]==-1]

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
            df_valid = df.loc[df["Jet_pt"]>-1]
            df_invalid = df.loc[df["Jet_pt"]<=-1]
            print("L: valid ", df_valid)
            print("L: invalid ", df_invalid)

            probs = model.predict(df_valid[features].values)#predict probability over test sample
            df_valid["Jet_sigprob"] = probs[:,1]
            df_invalid["Jet_sigprob"] = np.ones(df_invalid.shape[0])*(-1)
            df_test = pd.concat([df_valid,df_invalid])
            ## Here writes the output if needed
            #df_test.to_hdf(result_folder+sample+'_score_'+model_label+'.h5', 'df', format='table')
            #print("   "+result_folder+sample+"_score_"+model_label+".h5 stored")

            #Here root conversion
            #print(df_test)
            df_j = defaultdict()


            #Must redo zero padding!
            #for j in range(nj):
            print(df_test["Jet_index"])

            #Transform per-event into per-jet; this requires proper zero padding (hence, also empty jets)
            for j in range(nj):
                print(j)
                df_j[j] = df_test.loc[ df_test["Jet_index"]==float(j) ]
                print(df_j[j]["Jet_index"])

                #if df_j[j].shape[0]>0: print(df_j[j])
                #temp_list = []
                for f in jvar:
                    #print("Jet_"+f)
                    #print("Jets"+str(j)+"_"+f)
                    df_j[j].rename(columns={"Jet_"+f: "Jets"+str(j)+"_"+f},inplace=True)
                    #if str(j) in l:
                    #    print("\n")
                    #    #temp_list.append(l)
                df_j[j].rename(columns={jet_matching: "Jets"+str(j)+"_isGenMatched"},inplace=True)
                df_j[j].rename(columns={"Jet_index": "Jets"+str(j)+"_index"},inplace=True)
                df_j[j].rename(columns={"Jet_sigprob": "Jets"+str(j)+"_sigprob"},inplace=True)
                #if df_j[j].shape[0]>0: print(df_j[j])

                if j==0:
                    df = df_j[j]
                else:
                    df_temp = pd.merge(df, df_j[j], on=event_var, how='inner')
                    df = df_temp

            #Here, count how many jets are tagged!
            #Reject events with zero jets
            #print(df)
            df = df[ df['nCHSJets']>0]
            print("\n")
            print("work up to here?")
            #Define variables to counts nTags
            #HERE IT COMPLAINS, TOBEFIXED!
            ######
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

            ######
            #!!!#
            #print("\n")
            #print(df)
            #print(nj)
            #print(len(jfeatures)+3)
            #print(len(event_var))
            #print(list(df.columns))
            #df_0 = df_j[0]
            #df_3 = df_j[3]     
            #mergedStuff = pd.merge(df_0, df_3, on=event_var, how='inner')

            #Here I must compare df_j with the same event number and merge it

            newFile = TFile(result_folder+'/model_'+model_label+'/'+sample+'.root', 'recreate')
            newFile.cd()
            #Here, since it does not work, use only df_test
            #TOBEFIXED
            #####
            for n, a in enumerate(list(df.columns)):
                arr = np.array(df[a].values, dtype=[(a, np.float64)])
                ###print(a, " values: ", arr)
                ###array2root(arr, output_root_folder+'/model_'+model_label+'/'+sample+'.root', mode='update')#mode='recreate' if n==0 else 'update')
                if n==0: skim = array2tree(arr)
                else: array2tree(arr, tree=skim)#mode='recreate' if n==0 else 'update')
            #####

            for n, a in enumerate(list(df_test.columns)):
                arr = np.array(df_test[a].values, dtype=[(a, np.float64)])
                ###print(a, " values: ", arr)
                ###array2root(arr, output_root_folder+'/model_'+model_label+'/'+sample+'.root', mode='update')#mode='recreate' if n==0 else 'update')
                if n==0: skim = array2tree(arr)
                else: array2tree(arr, tree=skim)#mode='recreate' if n==0 else 'update')

            skim.Write()
            ##Recreate c_nEvents histogram
            #Giving errors, skip, TOBEFIXED!
            #####
            counter = TH1F("c_nEvents", "Event Counter", 1, 0., 1.)
            counter.Sumw2()
            ##Fill counter histogram with the first entry of c_nEvents
            counter.Fill(0., df["c_nEvents"].values[0])
            ##print("counter bin content: ", counter.GetBinContent(1))
            counter.Write()
            #####
            newFile.Close()
            ##counter.Delete()
'''

def write_discriminator_output_simplified(model_def,folder,model_folder,result_folder,jet_type,signal_model,event_dict,jet_dict,fat_jet_dict,nj,nfj,j_features,fj_features,is_signal,jet_matching,weight,n_batch_size,model_label,sample_list=[]):

    if model_label=="":
        print("No keras/BDT model specified, aborting . . . ")
        exit()

    model_folder += signal_model + '/' + jet_type + '/'
    model_folder += 'model_'+model_def+'_'+model_label+'/'
    output_file = 'best_model_'+model_label

    print("Loading model... ", model_folder+output_file+'.h5')
    model = keras.models.load_model(model_folder+output_file+'.h5')
    model.summary()

    if jet_type == "AK4jets":
        jet_string = "Jet_"
        features = j_features
    elif jet_type == "AK8jets":
        jet_string = "FatJet_"
        features = fj_features
    else:
        print("Jet type not recognized, aborting . . . ")
        exit()

    if sample_list==[]:
        print(" Not fixed, exit")
        exit()
        ##Read test sample
        '''
        store = pd.HDFStore(folder+"test_"+model_label+".h5")
        df = store.select("df",start=0,stop=20)
        df_valid = df.loc[df[jet_matching]!=-1]
        df_invalid = df.loc[df[jet_matching]==-1]

        probs = model.predict(df_valid[features].values.astype(dtype = 'float32'))#predict probability over test sample
        df_valid[jet_string+"sigprob"] = probs[:,1].astype(dtype = 'float32')
        df_invalid[jet_string+"sigprob"] = np.ones(df_invalid.shape[0])*(-1)
        df_test = pd.concat([df_valid,df_invalid])
        df_test.to_hdf(result_folder+'test_score_'+model_label+'.h5', 'df', format='table')
        print("   "+result_folder+"test_score_"+model_label+".h5 stored")
        '''

    else:
        full_list = []
        for sl in sample_list:
            full_list += samples[sl]['files']

        for sample in full_list:
            print('\n')
            print(" ********************* ")
            IN = folder+sample+'/'#+"_"+jet_type+"_test.h5"
            print(IN)
            #Concatenate all the files in one dataset
            files_list = [x for x in os.listdir(IN) if os.path.isfile(os.path.join(IN, x))]
            #print(files_list)
            df_list = []
            count_events = 0
            for n in range(len(files_list)):
                ###print(IN+files_list[n])
                store = pd.HDFStore(IN+files_list[n])
                if store.keys() ==[]:
                    print("Warning, empty dataframe, skipping...")
                    continue
                if n%50==0:
                    print('Opening file n. ', n)
                df = store.select("df",start=0,stop=20)
                count_events += df.shape[0]#store.get_storer('df').shape
                df_list.append( df.loc[df["EventNumber"]%2!=0]  )
                del df
                store.close()

            # Calculate even/odd ratio and keep only odd
            # Result of get_storer and df.shape differ by 1...
            df = pd.concat(df_list,ignore_index=True)
            if df.shape[0]==0:
                print("   Empty dataset, go to next sample.......")
                continue
            print("   df shape: ", df.shape)
            print("   count storer: ", count_events)
            del df_list

            #df_train = df.loc[df["EventNumber"]%2==0]
            df_test = df.loc[df["EventNumber"]%2!=0]
            ratio_test = df_test.shape[0]/count_events
            ratio_train = (count_events - df_test.shape[0])/count_events
            print("   ratio test ", ratio_test)
            print("   ratio train ", ratio_train)
            print("   train shape: ", count_events - df_test.shape[0])
            print("   test shape: ", df_test.shape)
            df_test["TestEventRatio"] = np.ones(df_test.shape[0])*(df_test.shape[0]/count_events)
            del df
            #print(df_test)

            nloop = 0
            if jet_type=="AK4jets":
                nloop=nj
                
            elif jet_type=="AK8jets":
                nloop=nfj

            #Create a dictionary of features lists
            features_list = {}
            for j in range(nloop):
                feat = []
                for f in features:
                    feat.append(jet_string+str(j)+"_"+f)
                features_list[j] = feat

            #Now make prediction per each jet
            #Empty jets (i.e. with pt<0) assigned probability is -1

            print('\n')
            print("   Running on test sample. This may take a moment. . .")
            
            #print("ATTENTION!!!! ONLY 1 JET ON PURPOSEEEE")
            #nloop = 1

            for j in range(nloop):
                #print("Jet n. : ", j)
                #print(df_test[ features_list[j] ])
                mask = df_test[jet_string+str(j)+"_pt"].values<=0.
                probs = model.predict(df_test[features_list[j]].values.astype(dtype='float32'))
                #First assign predicted probabilities
                df_test[jet_string + str(j) + "_sigprob"] = probs[:,1]
                #Then remove probabilities of empty jets, setting them to -1.
                df_test[jet_string + str(j) + "_sigprob"] = df_test[jet_string + str(j) + "_sigprob"].mask(mask, -1.)
                #print(df_test[ [ jet_string + str(j) + "_pt" ,  jet_string + str(j) + "_sigprob"] ])


            #Now we want to count the number of tagged jets
            print("   Counting number of tagged jets, cut based approach. . .")

            #Define variables to counts nTags
            var_tag_sigprob = []
            if jet_type=="AK4jets":
            ######
                cut_based_vars = {
                    "pt" : {"min" : 30, "max" : 1.e10},
                    "timeRecHitsEB" : {"min" : 0.09, "max" : 999.e+10},
                    "gammaMaxET" : {"min" : -100.-10., "max" : 0.16},
                    "minDeltaRPVTracks" : {"min" : 0.06, "max" : 999.+10.},
                    "cHadEFrac" : {"min" : -1., "max" : 0.06},
                    "muEFrac" : {"min" : -1., "max" : 0.6},
                    "eleEFrac" : {"min" : -1., "max" : 0.6},
                    "photonEFrac" : {"min" : -1., "max" : 0.8},
                }
                var_tag_cutbased_JJ = {}
                for j in range(nloop):
                    var_tag_sigprob.append(jet_string+str(j)+"_sigprob")
                for c in cut_based_vars.keys():
                    temp_list = []
                    for j in range(nloop):
                        temp_list.append(jet_string+str(j)+"_"+c)
                        var_tag_cutbased_JJ[c] = temp_list

                #print(cut_based_vars)

                mask_list = []
                for c in cut_based_vars.keys():
                    mask_temp = (df_test[ var_tag_cutbased_JJ[c]  ].values<cut_based_vars[c]['max']) & (df_test[ var_tag_cutbased_JJ[c]  ].values>cut_based_vars[c]['min'])
                    #print("mask[c] ", mask[c])
                    mask_list.append(mask_temp)

                ms = mask_list[0]
                for  m in mask_list:
                    ms = (ms) & (m)


                #print("with loop:  ", ms)
                #print("how many survived?")
                surv = np.sum(ms == True, axis=1)
                print(surv)
                df_test['nTags_cutbased'] = surv


            elif jet_type=="AK8jets":
            ######
                cut_based_vars = {
                    "pt" : {"min" : 170, "max" : 1.e15},
                    #missing one cut: FatJets->at(j).energyRecHitsHB/FatJets->at(j).energy>0.03 and FatJets->at(j).energyRecHitsHB>-1.
                    #"timeRecHitsEB" : {"min" : -0.296, "max" : 0.296},
                    "gammaMaxET" : {"min" : -100.-10., "max" : 0.02},
                    "cHadEFrac" : {"min" : -1., "max" : 0.03},
                }
                var_tag_cutbased_JJ = {}
                for j in range(nloop):
                    var_tag_sigprob.append(jet_string+str(j)+"_sigprob")
                for c in cut_based_vars.keys():
                    temp_list = []
                    for j in range(nloop):
                        temp_list.append(jet_string+str(j)+"_"+c)
                        var_tag_cutbased_JJ[c] = temp_list

                #print(var_tag_cutbased_JJ)
                #print(cut_based_vars)

                mask_list = []
                for c in cut_based_vars.keys():
                    mask_temp = (df_test[ var_tag_cutbased_JJ[c]  ].values<cut_based_vars[c]['max']) & (df_test[ var_tag_cutbased_JJ[c]  ].values>cut_based_vars[c]['min'])
                    #print("mask[c] ", mask[c])
                    mask_list.append(mask_temp)

                ms = mask_list[0]
                for  m in mask_list:
                    ms = (ms) & (m)


                #print("with loop:  ", ms)
                #print("how many survived?")
                surv = np.sum(ms == True, axis=1)
                #print(surv)
                df_test['nTags_cutbased'] = surv




            print("   Calculating n. tagged jets with different DNN output scores...")
            #wp_sigprob = [0.5,0.6,0.7,0.8,0.9,0.95,0.975]
            wp_sigprob = [0.9,0.95,0.96,0.97,0.98,0.99,0.995,0.9975]
            #wp_cHadEFrac = [0.2,0.1,0.05,0.02]

            for wp in wp_sigprob:
                print("   working point: ", wp)
                startTime = time.time()
                name = str(wp).replace(".","p")
                #This loop works but it's slow
                #df_test['nTags_sigprob_wp'+name] = df_test[ df_test[var_tag_sigprob] > wp ].count(axis=1)
                #print(df_test['nTags_sigprob_wp'+name].values)
                oneTime = time.time()

                cut_vars = {
                    "pt" : {"min" : 30, "max" : 1.e10},
                    "sigprob" : {"min" : wp, "max" : 1.e10},
                    "muEFrac" : {"min" : -1., "max" : 0.6},
                    "eleEFrac" : {"min" : -1., "max" : 0.6},
                    "photonEFrac" : {"min" : -1., "max" : 0.8},
                }
                var_tag = {}
                for c in cut_vars.keys():
                    temp_list = []
                    for j in range(nloop):
                        temp_list.append(jet_string+str(j)+"_"+c)
                        var_tag[c] = temp_list

                mask_list = []
                for c in cut_vars.keys():
                    mask_temp = (df_test[ var_tag[c]  ].values<cut_vars[c]['max']) & (df_test[ var_tag[c]  ].values>cut_vars[c]['min'])
                    #print("mask[c] ", mask[c])
                    mask_list.append(mask_temp)

                ms = mask_list[0]
                for  m in mask_list:
                    ms = (ms) & (m)


                #print("with loop:  ", ms)
                #print("how many survived?")
                surv = np.sum(ms == True, axis=1)
                #print(surv)
                df_test['nTags_sigprob_wp'+name] = surv

                '''
                mask = df_test[ var_tag_sigprob ].values>wp
                df_test['nTags_sigprob_wp'+name] = np.sum(mask == True, axis=1)
                #print(df_test['nTags_sigprob_wp'+name].values)
                '''
                twoTime = time.time()
                
                print("  * * * * * * * * * * * * * * * * * * * * * * *")
                #print("  Time needed with old method: %.2f seconds" % (oneTime - startTime))
                print("  Time needed with new method: %.2f seconds" % (twoTime - oneTime))
                print("  * * * * * * * * * * * * * * * * * * * * * * *")
            #Now we can write the test output

            OUT = result_folder+signal_model+'/'
            if not os.path.isdir(OUT):
                os.mkdir(OUT)

            OUT += jet_type + '/'
            if not os.path.isdir(OUT):
                os.mkdir(OUT)

            OUT += 'model_'+model_def+'_'+model_label+'/'
            if not os.path.isdir(OUT):
                os.mkdir(OUT)

            #print(OUT)
            print("   Creating root file . . .")

            newFile = TFile(OUT+sample+'.root', 'recreate')
            newFile.cd()

            for n, a in enumerate(list(df_test.columns)):
                arr = np.array(df_test[a].values, dtype=[(a, np.float32)])#64?
                ###print(a, " values: ", arr)
                ###array2root(arr, output_root_folder+'/model_'+model_label+'/'+sample+'.root', mode='update')#mode='recreate' if n==0 else 'update')
                if n==0: skim = array2tree(arr)
                else: array2tree(arr, tree=skim)#mode='recreate' if n==0 else 'update')

            skim.Write()
            ##Recreate c_nEvents histogram
            #Giving errors, skip, TOBEFIXED!
            
            counter = TH1F("c_nEvents", "Event Counter", 1, 0., 1.)
            counter.Sumw2()
            ##Fill counter histogram with the first entry of c_nEvents
            counter.Fill(0., df_test["c_nEvents"].values[0])
            ##print("counter bin content: ", counter.GetBinContent(1))
            counter.Write()
            
            newFile.Close()
            ##print('\n')
            print("   Written: ", OUT+sample+'.root')
            ##counter.Delete()





'''
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
                df_j[j].rename(columns={jet_matching: "Jets"+str(j)+"_isGenMatched"},inplace=True)
                df_j[j].rename(columns={"Jet_index": "Jets"+str(j)+"_index"},inplace=True)
                df_j[j].rename(columns={"Jet_sigprob": "Jets"+str(j)+"_sigprob"},inplace=True)
                #if df_j[j].shape[0]>0: print(df_j[j])

                if j==0:
                    df = df_j[j]
                else:
                    df_temp = pd.merge(df, df_j[j], on=event_var, how='inner')
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
            #print(len(event_var))
            #print(list(df.columns))
            #df_0 = df_j[0]
            #df_3 = df_j[3]     
            #mergedStuff = pd.merge(df_0, df_3, on=event_var, how='inner')

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
'''

    
###
# At the moment, this function is not needed. We might restore it later for relative eta/phi.
# Taken form ParticleNet notebooks

'''

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

