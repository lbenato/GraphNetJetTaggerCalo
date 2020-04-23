# GraphNetJetTaggerCalo

Graph Network aiming at jet tagging for binary classification (signal vs background) for searches for long lived particles with CMS detector.

Currently implemented in ```keras```.

Input and output root files are designed to be compatible with LLP repository.

## Preliminary naf gpu setup
See: https://confluence.desy.de/pages/viewpage.action?spaceKey=UHHML&title=Using+GPUs+in+naf

## Prepare folders
```
mkdir dataframes
mkdir model_weights
mkdir root_files
```

## write_pd_skim.py
It reads input root files and transforms them into h5 files.

## samples.py
Same as LLP repo; used to load sample names.

## dnn_functions.py
Simple function to draw training and validation losses and accuracies.

## dnn.py
- Split training/test samples for both signal and background
- Transform them into h5
- Training function
- Calculate performances
- Write the output scores of test samples
- Convert h5 back to root files, compatible with any macro of LLP repository
