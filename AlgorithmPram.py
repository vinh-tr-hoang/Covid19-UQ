'''
All parameter for the algorithms are described in class AlgorithmParam():
Likelihood functions are implemented in class LikelihoodFunctions (): 
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform

class AlgorithmParam(): # All the parameter use in the code
    def __init__(self, country_region):
        self.country_region = country_region
        # Country poulation
        self.N_country ={'Italy': 60.36e6, 'Germany': 83.02e6, 'Saudi Arabia': 33.7e6, 'Uruguay': 3.449e6}
        self.N = self.N_country[country_region]
        # Init time to perform identifcation
        self.initT_country = {'Italy': 40, 'Germany': 55,  'Saudi Arabia': 70, 'Uruguay': 74}
        self.initT = self.initT_country[country_region]
        # ratio of standard deviatio between kernel in mcmc and prior
        self.kernel_std_ratio_country =  {'Italy': 0.01, 'Germany': 0.01,  'Saudi Arabia': 0.01, 'Uruguay': 0.01}
        self.kernel_std_ratio = self.kernel_std_ratio_country[country_region]
        # directory where the computed data are stored
        self.result_dir = "results/" 
        # deltaT: interval of the the picewise linear functions used ti intepolate gamma and beta
        if country_region == 'Uruguay':
            self.deltaT = 10
        else: self.deltaT = 20
        # number of samples in the Markov Chain Monte Carlo
        self.n_MC = int(500e3)
        # number of samples are keep 
        self.n_MC_keep = int(100e3)
        # Time inteval for observation
        self.observation_deltaT = 14
        # targeted acceptance ration for mcmc       
        self.targeted_acceptance_ratio = 0.25
        # prior bound of theta, log(theta) follow uniform distributions bettween the bounds
        # where theta is vector of beta, gamma in SIR model       
        self.priorTheta = np.log(np.array([[1.e-3, 1.], [1.e-2, 1.]]))

# Class of likelihood functions
class LikelihoodFunctions ():
    def __init__ (self):
        self.cenBelief = np.array([0., 0.2])
        self.cenBeliefInfection = np.array([0., 0.2])
        self.cenBeliefRecover = np.array([0., 0.2])
        self.cenBeliefActive = np.array([-0., 0.2])
        self.RwingStd = 0.05
        self.LwingStd = 0.05
        self.stdActive = 0.3 
        self.errorInfectionsCases = {'CenBelief': uniform (loc = self.cenBelief[0], scale = self.cenBelief[1]), 
                          'Rwing': norm(loc= 0, scale = self.RwingStd),
                          'Lwing': norm (loc= 0., scale =self.LwingStd)}
        self.errorDeathRecoverCases = self.errorInfectionsCases
    def eval (self,x, xobserved):
        _DEpdf =1.
        errorDist = self.errorInfectionsCases
        xCenBeliefInfection = xobserved[0]* (1.+self.cenBeliefInfection ) 
        xCenBeliefRecover = xobserved[1]* (1.+self.cenBeliefRecover) 
        if x[0] < xCenBeliefInfection[0]: 
            _DEpdf *= errorDist['Lwing'].pdf((x[0]-xCenBeliefInfection[0])/xobserved[0])/errorDist['Lwing'].pdf(0)
        if x[1] < xCenBeliefRecover[0]: 
            _DEpdf *= errorDist['Lwing'].pdf((x[1]-xCenBeliefRecover[0])/xobserved[1])/errorDist['Lwing'].pdf(0)
        if x[0] > xCenBeliefInfection[1]:
            _DEpdf *= errorDist['Rwing'].pdf ((x[0]-xCenBeliefInfection[1])/xobserved[0])/errorDist['Rwing'].pdf(0)
        if x[1] > xCenBeliefRecover[1]:
            _DEpdf *= errorDist['Rwing'].pdf ((x[1]-xCenBeliefRecover[1])/xobserved[1])/errorDist['Rwing'].pdf(0)
        return _DEpdf    
    
    def _logdataErrorPdfNormalUniform (self,x, xobserved):
        _logDEpdf =0.
        xCenBeliefInfection = xobserved[0]* (1.+self.cenBeliefInfection ) 
        xCenBeliefRecover = xobserved[1]* (1.+self.cenBeliefRecover) 
        if x[0] < xCenBeliefInfection[0]: 
            _logDEpdf -= ((x[0]-xCenBeliefInfection[0])/xobserved[0]/self.LwingStd)**2
        if x[1] < xCenBeliefRecover[0]: 
            _logDEpdf -= ((x[1]-xCenBeliefRecover[0])/xobserved[1]/self.LwingStd)**2
        if x[0] > xCenBeliefInfection[1]:
            _logDEpdf -=((x[0]-xCenBeliefInfection[1])/xobserved[0]/self.RwingStd)**2
        if x[1] > xCenBeliefRecover[1]:
            _logDEpdf -= ((x[1]-xCenBeliefRecover[1])/xobserved[1]/self.RwingStd)**2
        return _logDEpdf
    
    def eval_log (self,x, xobserved):
        return self._logdataErrorPdfNormalUniform(x,xobserved) 
            
    def plotLikelihood(self):
        x= np.arange(self.cenBelief[0]-0.2, self.cenBelief[1] + 0.2, 0.02)
        x = np.array([x,x])
        print (x)
        _likelihood = np.array([self.eval(x[:,i]+1., np.array([1.,1.]))
                               for i in range (x.shape[1])])
        plt.plot(x[0,:], np.sqrt(_likelihood), linewidth= 2)
        plt.xlabel('normalized misfit', fontsize=16)
        plt.ylabel('likelihood function', fontsize=16)
