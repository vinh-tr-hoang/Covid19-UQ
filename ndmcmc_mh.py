"""
Metropolis Hasting MCMC algorithm
author Truong-Vinh Hoang
"""
from scipy.stats import uniform
import numpy as np
import timeit
class ndmcmc_mh:
    def __init__(self,_func_prior_pdf, _func_prior_gen, _func_likelihood,_func_model_,
                 _func_kernel_gen, n_MC, dim,**kwargs):
        self.n_MC = n_MC
        self.dim  = dim
        self.x_MCMC = np.zeros (shape = (n_MC, dim))       
        self._prior_pdf = _func_prior_pdf
        self._prior_gen = _func_prior_gen
        self._likelihood = _func_likelihood
        self._forward_model = _func_model_
        self._kernel_gen = _func_kernel_gen
        if 'loglikelihood' in kwargs:
            self.loglikelihoodFlag = kwargs['loglikelihood']
        else:
            self.loglikelihoodFlag = False
        if '_func_kernelRatio' in kwargs:
            self.kernelRatio = kwargs['_func_kernelRatio']
        else:
            self.kernelRatio = self.unitRatio
        
        self.mcmcStep = 0
    def unitRatio(self, **kwargs):
        return 1.
    def run_MCMC(self, verbose = True, **kwargs): 
        startT = timeit.timeit()
        # kwargs: keep_rejected_point, initial value       
        if 'x_init' in kwargs:
            self.x_MCMC[0,:] = kwargs['x_init']
            x = kwargs['x_init'] 
        else:
            x = self._prior_gen()
            self.x_MCMC[0,:] = x 
            print('initial value: ', self.x_MCMC[0,:])
        y = self._forward_model(x)
        self.y_MCMC = np.zeros (shape = (self.n_MC, y.size))
        self.y_MCMC[0,:] = y
        pdf_x_prev = self._prior_pdf (x)
        pdf_epsilon_prev =  self._likelihood (y)
        pdf_x_curr = 0.
        pdf_epsilon_curr = 0.
        self.currNbRepeatedSample = 0
        numberAcceptedSamples = 0.
        self.acceptanceRatio  = 0.
        if 'keep_rejected_point' in kwargs:
            self.x_computed = np.zeros (shape = (self.n_MC, self.dim))
            self.x_computed [0,:] = x
            self.y_computed = np.zeros (shape = (self.n_MC, y.size))
            self.y_computed [0,:] = y
        successful_update = False
        for i in range (1,self.n_MC): 
            self.mcmcStep = i                             
            if ((self.currNbRepeatedSample > 100 and self.currNbRepeatedSample%100== 0)): # start the chain again if no update for 100 steps                     
                print ("MCMC is repeated form step ", i-self.currNbRepeatedSample, ' to step ', i, 
                       )#" with value: ", self.x_MCMC[i-1,:])
            # Generate new samples 
            x = self._kernel_gen(self.x_MCMC[i-1,:])  
            y = self._forward_model(x)
            if 'keep_rejected_point' in kwargs:
                self.x_computed[i] = x 
                self.y_computed[i] = y 
            pdf_x_curr = self._prior_pdf (x)
            pdf_epsilon_curr = self._likelihood (y)
            
            # Acept or refuse new sample
            temp = uniform.rvs(size = 1)
            if self.loglikelihoodFlag:
                ratio = (pdf_x_curr + pdf_epsilon_curr) - (pdf_x_prev + pdf_epsilon_prev) + self.kernelRatio(self.x_MCMC[i-1,:],x)
                successful_update = (np.log(temp) <= ratio) 
            else:
                if (pdf_x_prev*pdf_epsilon_prev):
                    kernelRatio = self.kernelRatio(self.x_MCMC[i-1,:],x)
                    ratio = (pdf_x_curr*pdf_epsilon_curr)/(pdf_x_prev*pdf_epsilon_prev)
                    ratio = ratio*kernelRatio 
                else:
                    ratio =1.                                    
                successful_update = (temp <= ratio) 
            if (successful_update):
                self.x_MCMC[i,:] = x
                self.y_MCMC[i,:] = y
                self.currNbRepeatedSample = 0
                pdf_x_prev = pdf_x_curr
                pdf_epsilon_prev = pdf_epsilon_curr
                numberAcceptedSamples += 1
            else:
                self.x_MCMC [i,:] = self.x_MCMC[i-1,:]
                self.y_MCMC [i,:] = self.y_MCMC[i-1,:]
                self.currNbRepeatedSample = self.currNbRepeatedSample +1
            self.acceptanceRatio = numberAcceptedSamples/ float(i)

            if verbose and i%500 == 0:
                print ("MCMC current step:", i )     
                print ("mean value", [self.x_MCMC[0:i,j].mean() for j in range(0,self.dim)] )
                print ("acceptance ratio", self.acceptanceRatio)

        endT = timeit.timeit()
        self.executionTime = endT-startT
        print('mcmc execution time', endT - startT)
###############################################################################                
## Ouput 
###############################################################################
        if self.dim == 1:
            self.x_MCMC = self.x_MCMC.reshape((self.x_MCMC.size,))
            self.y_MCMC = self.y_MCMC.reshape((self.y_MCMC.size,))
        if 'keep_rejected_point' in kwargs:
            return self.x_MCMC, self.x_computed #, self.std_converge_flag  
        else: 
            return self.x_MCMC

     