'''
This file contains two class
SirIden, and simSirIden 
They deals with identification of parameter for two models:
'''
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from ndmcmc_mh import ndmcmc_mh as mcmc
from getdata import GetData
from models import SIR
from AlgorithmPram import AlgorithmParam as AlgParam
from AlgorithmPram import LikelihoodFunctions as LikelihoodFunc

class SirIden (mcmc,AlgParam):
    def __init__ (self,country_region, **kwargs):
        AlgParam.__init__(self, country_region)
        self.LikelihoodFunc = LikelihoodFunc()
        self.data = GetData(countryRegion = country_region)#Geta the data set
        self.time =np.arange(0, self.data.confirmed.shape[0]- self.initT,1) 
     
        self.observations = self.data._timeInterval_data (self.initT, self.time.size)        
        self.init_condition = self.observations[:,0]  
        self.model = SIR (self.init_condition, 
                          theta = np.zeros (shape = (2, self.time.size)), 
                          time = self.time, N = self.N)
        self.observation_infected_removed = np.array ([self.observations[0,:], 
                                        self.observations[1,:] + self.observations[2,:]])# infection and recovered + deaths
        self.time_node = np.arange(0, self.data.confirmed.shape[0]-self.initT,self.deltaT)
        self.time_node[-1] = self.data.confirmed.shape[0]-self.initT
        self.theta_dim = self.model.paramDim*self.time_node.size
        self.theta_shape = (self.model.paramDim, self.time_node.size)
        ##self.observation_deltaT = observation_deltaT
        self.time_4_eval_marginal_likelihood = self.time[np.arange(0,self.time.size, self.observation_deltaT, dtype = int )]
        self.time_4_eval_marginal_likelihood[-1] = self.time[-1]
        
        self.observation_initT = self.observation_infected_removed[:,0]
        self.identified_param_dim = self.theta_dim + self.observation_initT.size

        mcmc.__init__(self,_func_prior_pdf = self.log_prior_pdf, 
                         _func_prior_gen = self.prior_pdf_rvs, 
                         _func_likelihood = self.log_likelihood,
                         _func_model_ = self.forward_model,
                         _func_kernel_gen = self.adaptive_kernel_rvs, 
                         _func_kernelRatio = self.logkernelRatio,
                         n_MC = self.n_MC, dim = self.identified_param_dim,
                         loglikelihood = True)
        #MCMC
        index = np.where(self.observation_initT ==0)
        self.observation_initT[index] = 10.
        
        self.identified_param_dim = self.theta_dim + self.observation_initT.size

    def run_MCMC (self,**kwargs):
        self.kernel_cov = self.getPriorCov()     
        return mcmc.run_MCMC (self,**kwargs)   
    
    def x2theta_x0(self,x):
        theta = x[0:self.theta_dim].reshape (self.theta_shape)
        x0 = x[self.theta_dim:]
        return theta,x0
    
    def theta_x0_2_theta (self,theta,x0, **kwargs):
        list = []
        for i in range (theta.shape[0]):
            list.append(theta[i,:])
        list.append(x0)
        for key in kwargs.keys():
            list.append(kwargs['key'])
        return np.concatenate(list)
    
    def prior_pdf (self, x):
        theta, x0 = self.x2theta_x0(x)
        return self.prior_theta_pdf(theta)*self.LikelihoodFunc.eval(x0, self.observation_initT)
    
    def log_prior_pdf (self, x):
        theta, x0 = self.x2theta_x0(x)
        return np.log(self.prior_theta_pdf(theta)) + np.log(self.LikelihoodFunc.eval(x0, self.observation_initT))                     
    
    def prior_theta_pdf (self, theta):
        #log(beta) and log(gamma) follow uniform distribution
        _prior = 1.
        for i in range (0,self.theta_shape[0]):
            _prior *= float(np.min (self.priorTheta[i,1]>= theta[i,:])*
                          np.min (self.priorTheta[i,0]<= theta[i,:]))
        return _prior 
    
    def prior_init_condition(self,x0):
        prior = 1.
        for i in range (0,x0.size):
            ratio = x0[i]/self.observation_initT[i] - 1.
            prior *= float((ratio>= self.LikelihoodFunc.cenBelief[0])*
                          (ratio <= self.LikelihoodFunc.cenBelief[1]))
        return prior
    
    def prior_pdf_rvs (self):
        x0 = self.prior_init_condition_rvs()
        theta = self.prior_theta_rvs()
        return  self.theta_x0_2_theta(theta, x0)
    
    def prior_init_condition_rvs (self):
        return np.random.uniform(low = self.observation_initT*(1.+self.LikelihoodFunc.cenBelief[0]), 
                           high = self.observation_initT*(1.+self.LikelihoodFunc.cenBelief[1]))
    
    def prior_theta_rvs(self):
        shape = (self.time_node.size,self.model.paramDim)
        return (np.random.uniform(low = self.priorTheta[:,0], 
                             high = self.priorTheta[:,1], 
                             size = shape).transpose() )
    
    def likelihood (self,y):        
        _likelihood  = 1.
        y = y.reshape((2,y.size//2))
        for ti in range (1,self.time.size-1):
            xobserved = self.observation_infected_removed[:,ti] - self.observation_infected_removed[:,ti-1]
            _likelihood *= self.LikelihoodFunc.eval(y[:,ti-1], xobserved)
        return _likelihood
    
    # Log-likelihood
    def log_likelihood (self,y, **kwargs):        
        if 'observationTime' in kwargs.keys():
            observationTime= kwargs['observationTime']
        else: 
            observationTime = self.time_4_eval_marginal_likelihood
        _loglikelihood  = 0.
        y = y.reshape((2,y.size//2))
        for ti in range(1,observationTime.size):
            t = observationTime[ti]
            t1 = observationTime[ti]
            t0 = observationTime[ti-1]
            xobserved = self.observation_infected_removed[:,t1] - self.observation_infected_removed[:,t0]
            #xobserved = self.observation_infected_removed[:,t]
            lg = (self.LikelihoodFunc.eval_log(y[:,ti-1], xobserved))
            _loglikelihood += lg
        return _loglikelihood
    
    # new cases during intervals of observationTime       
    def forward_model (self, x, **kwargs):
        if 'observationTime' in kwargs.keys():
            observationTime= kwargs['observationTime']
        else: 
            observationTime = self.time_4_eval_marginal_likelihood
        vtheta = x[0:self.theta_dim].reshape (self.theta_shape)
        self.model.theta = self.interpolate_theta(vtheta)
        self.model.init_condition = x [self.theta_dim:self.theta_dim+2]
        __modelPrediction = self.model.eval()
        t1 = observationTime[1:]
        t0 = observationTime [0:-1]
        # New infection cases,and new recovered + deaths, and 
        incrementModel = __modelPrediction[:,t1] - __modelPrediction[:, t0]
        incrementModel = np.concatenate((incrementModel[0,:],incrementModel[1,:]))
        return incrementModel
    
    # picewise linear approximation of beta and gamma
    def interpolate_theta (self, logvtheta):
        vtheta = np.exp (logvtheta)
        ftheta = np.array ([np.interp(self.time, self.time_node, vtheta[i,:])
                            for i in range (vtheta.shape[0])])
        return ftheta
    
    def getPriorCov(self):
        # Sampling the prior with N samples to compute the covariance
        N = 10000
        x = np.zeros(shape = (N, self.identified_param_dim))
        for i in range (N):
            x[i,:] = self.prior_pdf_rvs()
        return np.cov(x.transpose())
    
    def getcorrPriorCov (self):
        _cov = self.getPriorCov ()
        for i in range (0, self.time_node.size-1):
            _cov[i,i+1] = 0.5*np.sqrt(cov[i,i]*_cov[i+1,i+1])
            _cov[i+1,i] = _cov[i,i+1]
        for i in range (self.time_node.size, self.time_node.size*2-1):
            _cov[i,i+1] = 0.5*np.sqrt(cov[i,i]*_cov[i+1,i+1])
            _cov[i+1,i] = _cov[i,i+1]
        return _cov
    
    def adaptive_kernel_std_ratio (self):
        if self.acceptanceRatio > self.targeted_acceptance_ratio*1.15 and self.currNbRepeatedSample <= 100:
            self.kernel_std_ratio = min (self.kernel_std_ratio*1.15, 0.1)
        if self.acceptanceRatio < self.targeted_acceptance_ratio/1.15 or self.currNbRepeatedSample >= 100:
            self.kernel_std_ratio = self.kernel_std_ratio/1.15
        
    def adaptive_kernel_rvs (self, x):
        if ((self.mcmcStep > 2000 and self.mcmcStep%100 == 0) 
            or (self.currNbRepeatedSample > 100 and self.currNbRepeatedSample % 20 ==0)):
            self.adaptive_kernel_std_ratio()
        if self.mcmcStep%500 == 0:
            print ('kernel_std_ratio', self.kernel_std_ratio)
        self.kernel = multivariate_normal(mean = np.zeros_like(x), cov = self.kernel_cov)   
        _prior_pdf = 0.
        while _prior_pdf == 0.:
            if self.mcmcStep < 1:
                _xN = self.prior_pdf_rvs()
            else:
                _xN = np.sqrt(1-self.kernel_std_ratio**2)*x + self.kernel.rvs()*self.kernel_std_ratio
            _prior_pdf = self.prior_pdf(_xN)
        return _xN 
    def logkernelRatio (self,x_0,x_1): 
        if self.mcmcStep < 1:
            _a = 0.
        else:
            _a = (self.kernel.logpdf (x_1-np.sqrt(1-self.kernel_std_ratio**2)*x_0)
            -self.kernel.logpdf (x_0-np.sqrt(1-self.kernel_std_ratio**2)*x_1))
        return _a
            
        
    
    def plot (self, x):
        theta,x0 = self.x2theta_x0(x)
        self.model.theta = self.interpolate_theta(theta)
        self.model.init_condition= x0
        self.model.eval()
        plt.plot(self.time + self.initT, self.model.simData[0,:], '-^',label = 'confirmed cases')
        plt.plot (self.time+  self.initT,self.model.simData[1,:], '--*',label = 'recovered + deaths')
        plt.plot(self.time +  self.initT,self.model.simData[0,:] -  self.model.simData[1,:],'--s',
                 label = 'active cases')
        plt.plot(self.data.confirmed, '-', label = 'data confirmed')
        plt.plot(self.data.deaths+ self.data.recovered,  '*-', label = 'data recovered deaths')
        plt.plot(self.data.confirmed -self.data.deaths - self.data.recovered, '--', label = 'active cases' )

    def modelBayesCriterion (self, samples, **kwargs):
        # real-likelihood = scaledBC *np.exp(logscale)
        if 'observationTime' in kwargs.keys():
            observationTime= kwargs['observationTime']
        else: 
            observationTime = self.time_4_eval_marginal_likelihood
        _BC = np.zeros_like(samples[:,0])
        for i in range (samples.shape[0]):
            _y = self.forward_model(samples[i,:], observationTime = observationTime)
            _BC [i] = ( self.log_likelihood (_y,observationTime = observationTime))
        _logscaledConst = np.mean (_BC)
        _scaledBC = 0.
        for i in range (samples.shape[0]):
            _scaledBC += np.exp(_BC[i] - _logscaledConst)/samples.shape[0] 
        return  _scaledBC, _logscaledConst 

    def plotLikelihood(self):
        x= np.arange(self.LikelihoodFunc.cenBelief[1]-0.2, self.LikelihoodFunc.cenBelief[1] + 0.2, 0.02)
        plt.plot(x, self.LikelihoodFunc.eval(x+1, 1))


class simpleSirIden (SirIden, mcmc):
    def __init__(self,
                  country_region,**kwargs):
        SirIden.__init__(self, country_region)     
        self.theta_dim = self.time_node.size + 1
        self.theta_shape = (self.theta_dim,)
        self.identified_param_dim = self.theta_dim + self.observation_initT.size
        
        # mcmc.__init__(self,_func_prior_pdf = self._logpriorPdf, 
        #                  _func_prior_gen = self._priorPdf_rvs, 
        #                  _func_likelihood = self._logd2dlikelihood,
        #                  _func_model_ = self._d2dforwardModel,
        #                  _func_kernel_gen = self._func_kernel_gen_AdaptiveMethod, 
        #                  _func_kernelRatio = self.logkernelRatio,
        #                  n_MC = self.n_MC, dim = self.identifiedParamDim,
        #                  loglikelihood = True)
        mcmc.__init__(self,_func_prior_pdf = self.log_prior_pdf, 
                         _func_prior_gen = self.prior_pdf_rvs, 
                         _func_likelihood = self.log_likelihood,
                         _func_model_ = self.forward_model,
                         _func_kernel_gen = self.adaptive_kernel_rvs, 
                         _func_kernelRatio = self.logkernelRatio,
                         n_MC = self.n_MC, dim = self.identified_param_dim,
                         loglikelihood = True)
        
    def prior_theta_rvs(self):
        betashape = (self.time_node.size,)
        beta = (np.random.uniform(low = self.priorTheta[0,0], 
                             high = self.priorTheta[0,1], 
                             size = betashape))
        gamma = (np.random.uniform(low = self.priorTheta[1,0], 
                             high = self.priorTheta[1,1]))
        theta = np.concatenate((beta[:],np.array([gamma]))).reshape(self.theta_shape)
        return theta

        
    def prior_theta_pdf (self, theta):
        #Uniform distribution
        __prior = 1.
        beta = theta[0:-1]
        gamma = theta[-1]
        for i in range (0,beta.size):
            __prior *= float(np.min (self.priorTheta[0,1]>= beta[i])*
                          np.min (self.priorTheta[0,0]<= beta[i]))
        __prior *= float(np.min (self.priorTheta[1,1]>= gamma)*
                          np.min (self.priorTheta[1,0]<= gamma))
        return __prior 
    def interpolate_theta (self, logvtheta):# picewise linear approximation
        vtheta = np.exp (logvtheta)
        ftheta = np.zeros(shape = (self.model.paramDim, self.time.size))
        ftheta[0,:] = np.interp(self.time, self.time_node, vtheta[0:-1])
        ftheta[1,:] = vtheta[-1]
        return ftheta
    def theta_x0_2_theta (self,theta,x0):
        list = []
        list.append(theta)
        list.append(x0)
        return np.concatenate(list)



    
    
    