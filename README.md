# Covid19-UQ
Uncertainty quantification in the modelling of Covid19 pandemic evolution

Based on the COVID-19 Data provided by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University, we identify the SIR model's parameters using the Bayesian approach.
 
We have documented some results in the documents folder. 
## Code structure
- main.py:  main file
- AlgorithParam.py: setting parameters and likelihood functions
- SIRparamIden.py: classes for Bayesian parameter identification
- models.py: implementation of SIR and SEIR models
- mh_mcmc.py: Metropolis-Hasting Markov Chain Monte Carlo algorithm
- postProcessing.py: post-processing (save and load results, and plot)
