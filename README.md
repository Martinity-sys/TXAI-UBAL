# TXAI-UBAL
Trustworthy and Explainable AI Project - Uncertainty Based Active Learning

For this project we aim to reproce <em>[The power of ensembles for active learning in image classification (Beluch et al.)](https://ieeexplore.ieee.org/document/8579074)</em>. In addition to trying to reproduce the results we add a dropconnect model, and we test the usage of variance in stead of variation ratio. We trained 7 different models for uncertainty based active learning. We train Monte Carlo Dropout with variation ratio and variance, Monte Carlo Dropconnect with variation ratio and variance, ensembles with variation ratio and variance, and a random acquisition model. 

---

This repository is structured as:

```
.
├── analysis
├── data
│   ├── FashionMNIST
│   │   └── raw
│   └── model_data
├── models
│   ├── varR
│   └── variance
└── src
```

---

- analysis: Files used for the analysis of our models
    - calc_accuracy.py
    - plots.ipynb

- data: Files for training data and model accuracy data
    - model_data: contains csv files with the run, training size, loss and accuracy of each model

- models: Final models for each training run for each model
- images: Images generated with the analysis


### RUN

Note that running the model with overwrite the data in the model_data folder.
The models can be ran as follows (MCDropout used as example):

```
usage: MCDropout.py [-h] [--runs RUNS] [--save SAVE] [--init INIT] [--acq ACQ] [--max MAX] [--epochs EPOCHS] [--t T]

Active Learning with Monte Carlo MCDropout

options:
  -h, --help       show this help message and exit
  --runs RUNS      number of runs
  --save SAVE      save model
  --init INIT      initial labeled set size
  --acq ACQ        acquisition size
  --max MAX        maximum labeled set size
  --epochs EPOCHS  number of epochs for training
  --t T            number of forward passes (or number of models in an ensemble) for uncertainty quantification
```