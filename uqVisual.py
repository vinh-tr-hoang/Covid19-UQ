'''
This file contains basic statistical methods for analysis data and plots 
'''
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
		
class dataAnalysis ():
    def autocorrelation (samples,x):
        ACF = np.zeros_like(x, dtype = float) 
        samples = samples - samples.mean()
        for i in range(x.size):
            ACF[i] = np.mean (samples[0:-1-x[i]]*samples[x[i]:-1])
        return ACF  
    def kdePdf(samples):
        pass
    def eCdf (samples):
        pass
        

class uqPlot(dataAnalysis):
    def quantilePlot(ysamples,x, quantile):
        mvalue = np.mean (ysamples, axis = 0)
        quantile95 = np.quantile(ysamples,1 -(1-quantile)/2.,axis =0)
        quantile5 = np.quantile(ysamples,(1. - quantile)/2., axis =0)
        plt.plot(x,mvalue,'-b', label = 'mean value', linewidth=2 )
        plt.plot(x, quantile95, '--b', label = str(quantile)+' quantile', linewidth=2)
        plt.plot(x, quantile5, '--b',linewidth=2)
    
    def autocorrelation(samples,x):
        ACF = dataAnalysis.autocorrelation(samples, x)
        ACF = ACF/ACF[0]
        plt.plot(ACF, '-b', linewidth = 2, label = 'normalized autocorellation ')
        plt.ylabel('Normalized ACF', fontsize= 14)
        plt.xlabel('lag', fontsize = 14)
    def kde(samples):
        kernel = stats.gaussian_kde(samples)
        smin = samples.min()
        smax = samples.max()
        x= np.arange(smin/1.1, smax*1.1, smax/100.-smin/100.)
        plt.plot(x,kernel(x),linewidth = 2, label = 'pdf')
    def cdf(samples):
        pass
    
    