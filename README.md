# TXAI-UBAL
Trustworthy and Explainable AI Project - Uncertainty Based Active Learning

For this project we aim to reproce <em>[The power of ensembles for active learning in image classification (Beluch et al.)](https://ieeexplore.ieee.org/document/8579074)</em>. In addition to trying to reproduce the results we add a dropconnect model, and we test the usage of variance in stead of variation ratio. We trained 7 different models for uncertainty based active learning. We train Monte Carlo Dropout with variation ratio and variance, Monte Carlo Dropconnect with variation ratio and variance, ensembles with variation ratio and variance, and a random acquisition model. 

This repo is structured as:

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

- analysis: Files used for the analysis of our models
    - calc_accuracy.py
    - plots.ipynb

- data: Files for training data and model accuracy data
    - model_data: contains csv files with the run, training size, loss and accuracy of each model

- models: Final models for each training run for each model
- images: Images generated with the analysis