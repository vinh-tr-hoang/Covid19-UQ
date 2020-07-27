# Covid19-UQ
Uncertainty quantification in modelling of Covid19 pandemic evolution

The SIR model's parameters are identified using Bayesian approach.
The inference is based on the COVID-19 Data provided by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University.

Some results are documented in documents folder.
## Code structure
- main.py:  main file
- AlgorithParam.py: setting parameters and likelihood functions
- SIRparamIden.py: classes for mcmc parameter identification
- models.py: implementation of SIR and SEIR models
- mh_mcmc.py: Metropolis-Hasting MCMC algorithm
- postProcessing.py: post processing (save, load results, and plot)
