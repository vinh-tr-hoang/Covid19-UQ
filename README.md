# Covid19-UQ
Uncertainty quantification in modelling of Covid19 pandemic evolution

We identify the parameters of the SIR model using Bayesian approach.
Data from the COVID-19 Data Repository by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University are used for the parameter identification.

Some results are collected in documents folder.
## Code structure
- main.py:  main file
- AlgorithParam.py: setting parameters and likelihood functions
- SIRparamIden.py: classes for mcmc parameter identification
- models.py: implementation of SIR and SEIR models
- mh_mcmc.py: Metropolis-Hasting algorithm
- postProcessing.py: post processing (save, load results, and plot)
