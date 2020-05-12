# GraphNetJetTaggerCalo

Graph Network aiming at jet tagging for binary classification (signal vs background) for searches for long lived particles with CMS detector.

Currently implemented in ```keras```.

Input and output root files are designed to be compatible with LLP repository.

## Preliminary naf gpu setup
See: https://github.com/lbenato/ParticleNet-LLP-fork

Once in your environment folder, install the additional packages:
```
pip install GPUtil
pip install xgboost
```

## Prepare folders
```
mkdir dataframes
mkdir dataframes_graphnet
mkdir model_weights
mkdir model_weights_graphnet
mkdir root_files
```

## write_pd_graphnet.py
It reads input root files and transforms them into h5 files.

## samples*.py
Same as LLP repo; used to load sample names.

## dnn_functions.py
Simple function to draw training and validation losses and accuracies.

## tf_keras_model.py
Original ParticleNet functions and architectures, added FCN jet tagger architecture (https://github.com/lbenato/LEADER). Added some modifications to take into account angular variables for matrix distance.

## prepare_dataset_graphnet.py
- Convert per-event into per-jet dataframes
- Split training/test samples for both signal and background
- Transform them into train, validation, test h5

## graphnet_tagger.py
- Load FCN or ParticleNet models
- Training function
- Calculate performances
- Write the output scores of test samples
- Convert h5 back to root files, compatible with any macro of LLP repository

## training_script.py
- Define parameters and submits routines defined in graphnet_tagger.py 