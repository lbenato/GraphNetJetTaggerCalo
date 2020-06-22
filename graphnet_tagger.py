import pandas as pd
from tensorflow import keras
import tensorflow as tf
import numpy as np
import awkward
import pickle
import uproot_methods
import os.path
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from root_numpy import array2tree, array2root
from dnn_functions import *
from samplesAOD2017 import *
from tf_keras_model import *

def _col_list(prefix,npf):
    return ['%s_%d'%(prefix,i) for i in range(npf)]


def get_FCN_jets_dataset(dataframe,features,weight,is_signal="is_signal",ignore_empty_jets=True):

    if ignore_empty_jets:
        #print("\n")
        #print("    Ignore empty jets!!!!!!")
        #print("\n")
        dataframe = dataframe[ dataframe["Jet_pt"]>-1 ]

    X = dataframe[features].values
    y = dataframe[is_signal].values
    w = dataframe[weight].values
        
    return X, y, w

def get_FCN_jets_dataset_generator(dataframe,features,weight,is_signal="is_signal",ignore_empty_jets=True):

    while True:
        df = next(dataframe)
        if ignore_empty_jets:
            #print("\n")
            #print("    Ignore empty jets!!!!!!")
            #print("\n")
            df = df[ df["Jet_pt"]>-1 ]

        X = df[features].values
        y = df[is_signal].values
        w = df[weight].values
        
        yield X, y, w

def get_BDT_dataset(dataframe,features,weight,is_signal="is_signal",ignore_empty_jets=True):

    if ignore_empty_jets:
        #print("\n")
        #print("    Ignore empty jets!!!!!!")
        #print("\n")
        dataframe = dataframe[ dataframe["Jet_pt"]>-1 ]

    X = dataframe[features]
    y = dataframe[is_signal]
    w = dataframe[weight]
        
    return X, y, w


def get_particle_net_dataset(dataframe,n_points,points_var,features_var,mask_var,weight,is_signal="is_signal",ignore_empty_jets=True):

    if ignore_empty_jets:
        #print("\n")
        #print("    Ignore empty jets!!!!!!")
        #print("\n")
        dataframe = dataframe[ dataframe["Jet_pt"]>-1 ]
        
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
    #custom_adam:
    #custom_adam = keras.optimizers.Adam(learning_rate=0.001/2., beta_1=0.9, beta_2=0.999, amsgrad=False)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=custom_adam, metrics = ["accuracy"])

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
    #custom_adam:
    #custom_adam = keras.optimizers.Adam(learning_rate=0.001/2., beta_1=0.9, beta_2=0.999, amsgrad=False)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=custom_adam, metrics = ["accuracy"])

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
    #custom_adam:
    #custom_adam = keras.optimizers.Adam(learning_rate=0.001/2., beta_1=0.9, beta_2=0.999, amsgrad=False)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=custom_adam, metrics = ["accuracy"])

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
    


def fit_model(model_def,n_class,folder,result_folder,n_points,points,features,mask,is_signal,weight,use_weight,n_epochs,n_batch_size,patience_val,val_split,model_label="",ignore_empty_jets_train=True):

    ##Read train/validation sample
    store_train = pd.HDFStore(folder+"trainptnorm.h5")
    df_train = store_train.select("df")
    store_val = pd.HDFStore(folder+"valptnorm.h5")
    df_val = store_val.select("df")
    
    if(model_def=="FCN"):
        X_train, y_train, w_train = get_FCN_jets_dataset(df_train,features,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
        X_val,   y_val,   w_val   = get_FCN_jets_dataset(df_val,features,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
        model = get_FCN_jets(num_classes=n_class, input_shapes=X_train.shape[1:])
    elif(model_def=="particle_net_lite"):
        X_train, y_train, w_train, input_shapes = get_particle_net_dataset(df_train,n_points,points,features,mask,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
        X_val,   y_val,   w_val, _   = get_particle_net_dataset(df_val,n_points,points,features,mask,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
        model = get_particle_net_lite(n_class, input_shapes, contains_angle = True if 'phi' in points else False)
# Julia: add particle_net        
    elif(model_def=="particle_net"):
        X_train, y_train, w_train, input_shapes = get_particle_net_dataset(df_train,n_points,points,features,mask,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
        X_val,   y_val,   w_val, _   = get_particle_net_dataset(df_val,n_points,points,features,mask,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
        model = get_particle_net(n_class, input_shapes, contains_angle = True if 'phi' in points else False) 
        
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
    #custom_adam:
    #custom_adam = keras.optimizers.Adam(learning_rate=0.001/2., beta_1=0.9, beta_2=0.999, amsgrad=False)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=custom_adam, metrics = ["accuracy"])

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

def fit_BDT(model_def,n_class,folder,result_folder,n_points,points,features,mask,is_signal,weight,use_weight,n_epochs,n_batch_size,patience_val,val_split,model_label="",ignore_empty_jets_train=True):

    ##Read train/validation sample
    store_train = pd.HDFStore(folder+"train.h5")
    df_train = store_train.select("df")
    store_val = pd.HDFStore(folder+"val.h5")
    df_val = store_val.select("df")
    
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
    ##custom_adam:
    ##custom_adam = keras.optimizers.Adam(learning_rate=0.001/2., beta_1=0.9, beta_2=0.999, amsgrad=False)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=custom_adam, metrics = ["accuracy"])

    ###Callbacks
    #early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience_val, verbose=0, mode='auto')
    #checkpoint = keras.callbacks.ModelCheckpoint(filepath=result_folder+'best_model_'+model_label+'.h5', monitor='val_loss', save_best_only=True)


    ##Fit model
    #train is 60%, test is 20%, val is 20%
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


def evaluate_model(model_def,n_class,folder,result_folder,n_points,points,features,mask,is_signal,weight,use_weight,n_batch_size,model_label,signal_match_test,ignore_empty_jets_test):

    print("\n")
    print("    Evaluating performances of the model.....   ")

    ##Read test sample
    store = pd.HDFStore(folder+"testptnorm.h5")
    df_test = store.select("df")

    print("    Remove negative weights at testing!!!!!!")
    df_test = df_test.loc[df_test['EventWeight']>=0]

    add_string = ""
    if ignore_empty_jets_test:
        #print("\n")
        #print("    Ignore empty jets at testing!!!!!!")
        #print("\n")
        df_test = df_test.loc[df_test["Jet_pt"]>-1]
        #add_string+="_ignore_empty_jets"

    if signal_match_test:
        #print("\n")
        #print("    Ignore not matched jets in signal at testing!!!!!!")
        #print("\n")
        df_s = df_test.loc[df_test[is_signal]==1]
        df_b = df_test.loc[df_test[is_signal]==0]
        df_s = df_s.loc[df_s["Jet_isGenMatched"]==1]
        df_test = pd.concat([df_b,df_s])
        #print(df_test.shape[0],df_s.shape[0],df_b.shape[0])
        add_string+="_signal_matched"

    
    if(model_def=="FCN"):
        X_test, y_test, w_test = get_FCN_jets_dataset(df_test,features,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
    elif(model_def=="particle_net_lite" or model_def=="particle_net"):
        X_test, y_test, w_test, input_shapes = get_particle_net_dataset(df_test,n_points,points,features,mask,weight=weight,is_signal="is_signal",ignore_empty_jets=True)
    else:
        print("    Model not recognized, abort . . .")
        exit()


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
        fpr, tpr, _ = roc_curve(df_test[is_signal], df_test["sigprob"])
    else:
        fpr, tpr, _ = roc_curve(df_test[is_signal], df_test["sigprob"], sample_weight=w_test)

    plt.figure(figsize=(8,7))
    plt.rcParams.update({'font.size': 15}) #Larger font size
    plt.plot(fpr, tpr, color='crimson', lw=2, label='AUC = {0:.4f}'.format(AUC))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
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
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
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


def evaluate_BDT(model_def,n_class,folder,result_folder,n_points,points,features,mask,is_signal,weight,use_weight,n_batch_size,model_label,signal_match_test,ignore_empty_jets_test):

    print("\n")
    print("    Evaluating performances of the model.....   ")

    ##Read test sample
    store = pd.HDFStore(folder+"test.h5")
    df_test_pre = store.select("df")

    print("    Remove negative weights at testing!!!!!!")
    df_test_pre = df_test_pre.loc[(df_test_pre['EventWeight']>=0) & (df_test_pre['Jet_pt']>-1)]

    add_string = ""

    if signal_match_test:
        print("\n")
        print("    Ignore not matched jets in signal at testing!!!!!!")
        print("\n")
        df_s = df_test_pre.loc[(df_test_pre[is_signal]==1) & (df_test_pre["Jet_isGenMatched"]==1)]
        df_b = df_test_pre.loc[df_test_pre[is_signal]==0]
        df_test = pd.concat([df_b,df_s])
        add_string+="_signal_matched"
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

    fig, ax = plt.subplots(figsize=(12, 8))
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



def compare_models(model_list,result_folder,is_signal,weight_list,use_weight,model_labels,signal_match_test,ignore_empty_jets_test):

    sigprob = {}
    target = {}
    weight = {}
    AUC = {}
    fpr = {}
    tpr = {}
    add_string = ""

    #if ignore_empty_jets_test:
    #    add_string+="_ignore_empty_jets"

    if signal_match_test:
        add_string+="_signal_matched"

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
            fpr[i], tpr[i], _ = roc_curve(target[i], sigprob[i], sample_weight=weight[i])
        else:
            AUC[i] = roc_auc_score(target[i], sigprob[i])
            fpr[i], tpr[i], _ = roc_curve(target[i], sigprob[i])

        del df
        store.close()
        del store

    colors = ['crimson','green','skyblue','chocolate','orange']
    linestyles = ['-', '--', '-.', ':']

    plt.figure(figsize=(8,7))
    plt.rcParams.update({'font.size': 15}) #Larger font size
    for i,m in enumerate(model_list):
        lab = m if not m=="LEADER" else "FCN"
        plt.plot(fpr[i], tpr[i], color=colors[i], linestyle=linestyles[i], lw=2, label=lab+' (AUC = {0:.4f})'.format(AUC[i]))
        #plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label=m+model_labels[i]+' (AUC = {0:.4f})'.format(AUC[i]))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
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
        plt.plot(fpr[i], tpr[i], color=colors[i], linestyle=linestyles[i], lw=2, label=lab+' (AUC = {0:.4f})'.format(AUC[i]))
        #plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label=m+model_labels[i]+' (AUC = {0:.4f})'.format(AUC[i]))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
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

'''

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

