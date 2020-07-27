'''
This class contains functions to
1) run_MCMC(): execute the MCMC algorithm for SIR models,and save results
2) load_data(): load nummerical results (the Markov Chain)
3) report(): compute the marginal likelihood
4) remove_burning (): remove first smaples of the chain
5) plot (): for visualize the obtain results
'''
import numpy as np
import matplotlib.pyplot as plt
from SIRparamIden import SirIden as SirIden
from SIRparamIden import simpleSirIden as simpleSirIden
from xSIRparamIden import hyperSirIden as hyperSirIden
from uqVisual import uqPlot
import genLib as genLib
from AlgorithmPram import AlgorithmParam as AlgParam


class postProcessing(AlgParam):
    def __init__(self,country_region,model, **kwargs):
        AlgParam.__init__(self, country_region)
        self.model = model
        if model == 'SIR':
            self.SIR = SirIden(country_region= self.country_region) 
        elif model =='simSIR':
            self.SIR = simpleSirIden(country_region= self.country_region) 
            #gammaSize = 1
        elif model == 'hyperSIR':
            self.SIR = hyperSirIden(initT, deltaT, N, n_MC = self.n_MC, kernel_std_ratio = self.kernel_std_ratio, country_region= self.country_region, observation_deltaT = 14) 
    
        if 'loadInit' in kwargs.keys():
            self.loadInit = kwargs['loadInit']
            self.ithrun = kwargs['ithrun']
            if self.loadInit:
                self.filename = model +country_region + str(self.ithrun)
                self.loadfile = model +country_region + str(self.ithrun-1)
                if self.ithrun == 1:
                    self.loadfile = model +country_region
            else: self.loadInit = 0
        if self.loadInit == 0:
            self.filename = model + country_region
        f = plt.figure()
        self.SIR.LikelihoodFunc.plotLikelihood()
        f.savefig(self.result_dir+'likelihoodfunction.pdf',bbox_inches='tight')
        self.observationTime4Val = self.SIR.time[np.arange(5,self.SIR.time.size, 20, dtype = int )]
        

    def run_mcmc (self,**kwargs):
        #### Perform MCMC
        if self.loadInit == 0:
            np.random.seed(100)
            x_init = self.SIR.prior_pdf_rvs()
            print (x_init)
            print(self.SIR.x2theta_x0(x_init))
            x_init[0:self.SIR.theta_dim//2] = np.log(0.2)
            x_init[self.SIR.theta_dim//2:self.SIR.theta_dim] = np.log(0.2)
        else:
            x_prev = self.load_data(self.result_dir+self.loadfile)
            x_init = np.mean(x_prev, axis = 0)
        x = self.SIR.run_MCMC(x_init = x_init)
        #### End MCMC process
        
        np.save(self.result_dir+self.filename+'.npy', x)
        xremovedBurning = self.remove_burning(x)
        self.report(xremovedBurning)
        self.plot(xremovedBurning)
        return xremovedBurning
    def report(self,xremovedBurning, **kwargs):
        scaledBC, logscaledConst = self.SIR.modelBayesCriterion(xremovedBurning, observationTime = self.observationTime4Val)
        print ('marginal Likelihood', np.exp(np.log(scaledBC) + logscaledConst)) 
        if self.loadInit:
            filename = self.model + self.country_region + str(self.ithrun) + 'Cont.csv'  
        else:
            filename = self.model + self.country_region +'.csv'  
        genLib.report(filename, model = self.model, country_region = self.country_region, initT = self.SIR.initT, N = self.SIR.N,
               n_MC = self.SIR.n_MC,
               deltaT = self.SIR.deltaT, observation_deltaT = self.SIR.observation_deltaT,
               scaledBC= scaledBC, logscaledConst= logscaledConst,
               #executionTime = self.SIR.executionTime,
               marginalLikelihood = np.exp(np.log(scaledBC) + logscaledConst),x_mean = xremovedBurning.mean(axis =0))
        ####
    def remove_burning(self,x):
        return x[-self.n_MC_keep:,:]
    def load_data(self,loadfile):
        x = np.load(loadfile+'.npy')
        xremovedBurning = self.remove_burning(x)
        return xremovedBurning
    def plot(self,x,qtPlot = 0.95): 
        x_mean = x.mean(axis = 0)
        if self.model != 'hyperSIR':
            logthetaM, x0M = self.SIR.x2theta_x0(x_mean)
        else:
            logthetaM, x0M, hyperParam = self.SIR.x2theta_x0(x_mean)
    
        print (x_mean)
        fxmean = self.SIR.interpolate_theta(logthetaM)
        plt.figure('beta')
        plt.plot(self.SIR.model.time,(fxmean[0,:]))
        plt.figure('gamma')
        plt.plot(self.SIR.model.time,(fxmean[1,:]))
        
        plt.figure('mean value')    
        self.SIR.plot(x_mean)
        plt.xlim ([0, self.SIR.time_4_eval_marginal_likelihood.max() + self.SIR.initT])
        plt.ylim([0,self.SIR.data.confirmed.max()])
        plt.legend()
        
        f= plt.figure('ACF Beta')
        steps = np.arange(0,1000,1,dtype = int)
        uqPlot.autocorrelation(x[:,0],steps)
        f.savefig(self.result_dir+self.model + self.country_region + "ACFBeta.pdf", bbox_inches='tight')
        plt.show()
        
        
        
        #Posprocessing data 
        active_cases_posterior_samples = np.zeros(shape = (x.shape[0], self.SIR.model.time.size ))# infection and recovered
        total_infections_posterior_samples = np.zeros(shape = (x.shape[0], self.SIR.model.time.size ))# infection and recovered
        beta = np.zeros(shape = (x.shape[0], self.SIR.time_node.size ))# infection and recovered
        gamma = np.zeros(shape = (x.shape[0],self.SIR.time_node.size ))# infection and recovered = np.zeros(shape = (x.shape[0],SIR.time_node.size ))# infection and recovered
        if self.model == 'hyperSIR':
            alpha = np.zeros(shape = (x.shape[0],self.SIR.priorHyperParamDim))# infection and recovered
        
        for i in range(x.shape[0]):
            if self.model != 'hyperSIR':
                theta, x0 = self.SIR.x2theta_x0(x[i,:])
            else:
                theta, x0, hyperParam = self.SIR.x2theta_x0(x[i,:])
            if self.model == 'SIR':
                beta[i,:] = np.exp(theta[0,:])
                gamma[i,:] = np.exp(theta[1,:])
            elif self.model =='simSIR':
                beta[i,:] = np.exp(theta[0:-1])
                gamma[i,:] = np.exp(theta[-1])*np.ones_like(gamma[i,:])      
            elif self.model == 'hyperSIR':
                beta[i,:] = np.exp(theta[0,:])
                gamma[i,:] = np.exp(theta[1,:])
                alpha[i,:] = hyperParam
                
            self.SIR.model.theta = self.SIR.interpolate_theta(theta)
            self.SIR.model.initCondition= x0
            self.SIR.model.eval()
            active_cases_posterior_samples[i,:] = self.SIR.model.state[1,:]
            total_infections_posterior_samples[i,:] = self.SIR.model.state[1,:] + self.SIR.model.state[2,:]
        f=plt.figure('Active cases')
        uqPlot.quantilePlot(active_cases_posterior_samples, self.SIR.model.time + self.SIR.initT, qtPlot)
        plt.plot(self.SIR.data.confirmed -self.SIR.data.deaths - self.SIR.data.recovered, '*-', label = 'data' )
        plt.legend(fontsize=15)
        plt.xlabel('day', fontsize= 18)
        plt.ylabel('active cases', fontsize= 18)
        f.savefig(self.result_dir+self.model + self.country_region + "ActiveCases.pdf", bbox_inches='tight')
        plt.show()
        
        f = plt.figure('Total infected cases')
        uqPlot.quantilePlot(total_infections_posterior_samples, self.SIR.model.time + self.SIR.initT, qtPlot)
        plt.plot(self.SIR.data.confirmed, '*-', label = 'data' )
        plt.legend(fontsize=15)
        plt.xlabel('day', fontsize= 18)
        plt.ylabel('Cumulative infections', fontsize= 18)
        f.savefig(self.result_dir+self.model + self.country_region +"totalCases.pdf", bbox_inches='tight')
        plt.show()
        
        f = plt.figure('beta')
        uqPlot.quantilePlot(beta, self.SIR.time_node + self.SIR.initT, qtPlot)
        plt.legend(fontsize=15)
        plt.xlabel('day', fontsize= 18)
        plt.ylabel(r'$\beta$', fontsize= 18)
        if self.country_region == 'Germany':
            plt.ylim([0.0, 0.8])
        if self.country_region == 'Uruguay':
            plt.ylim([0., 1.])
        if self.country_region == 'Saudi Arabia':
            plt.ylim([0.02, 0.18])
        if self.country_region == 'Italy':
            plt.ylim([0, 0.35])
        f.savefig(self.result_dir+self.model + self.country_region +"beta.pdf", bbox_inches='tight')
        plt.show()
        
        f = plt.figure('gamma')
        uqPlot.quantilePlot(gamma, self.SIR.time_node + self.SIR.initT, qtPlot)
        plt.legend(fontsize=15)
        plt.xlabel('day', fontsize= 18)
        plt.ylabel(r'$\gamma$', fontsize= 18)
        if self.country_region == 'Germany':
            plt.ylim([0.0, 0.8])
        if self.country_region == 'Uruguay':
            plt.ylim([0., 0.55])
        if self.country_region == 'Saudi Arabia':
            plt.ylim([0, 0.14])
        if self.country_region == 'Italy':
            plt.ylim([0, 0.1])
        f.savefig(self.result_dir+self.model + self.country_region + "gamma.pdf", bbox_inches='tight')
        
        
        if self.model == 'hyperSIR':
            f = plt.figure('alpha')
            uqPlot.kde(alpha[:,0])
            plt.legend()
            plt.xlabel(r'$\alpha$', fontsize= 18)
            plt.ylabel('pdf', fontsize= 18)
            f.savefig(self.result_dir+self.model + self.country_region + "alpha.pdf", bbox_inches='tight')
        plt.show()    
        print('end')