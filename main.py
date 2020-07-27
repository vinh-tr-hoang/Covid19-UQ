'''
Here is the main file which comunicates with postProcessing class
'''
from postProcessing import postProcessing as pp
import matplotlib.pyplot as plt

# SIR is the model in which beta and gamma vary with time
# simSIR is the model in which only beta is time-dependent
model = 'SIR'
# True if we want to run another run take the last member of previous run as initial value
# False it it is the first run 
loadInit = False 
ithrun = 0 # 0 for the first run
#running MCMC
for country in ('Germany',): #,'Uruguay', 'Saudi Arabia', 'Italy'): #, ,
    print(country)    
    sirpp = pp(country, model,loadInit = loadInit, ithrun = ithrun+1) 
    x = sirpp.run_mcmc()  

# load data and plot 
for country in ('Germany',): #,'Uruguay', 'Saudi Arabia', 'Italy'): #
    sirpp = pp(country, model,loadInit = loadInit, ithrun = ithrun) 
    sirpp.loadfile = sirpp.result_dir + model + country + str(ithrun)
    x = sirpp.load_data(sirpp.loadfile)
    x = sirpp.remove_burning(x)
    sirpp.report(x)
    sirpp.plot(x)
    f = plt.figure ()
    sirpp.SIR.data.plotLikelihood()
    f.savefig('likelihoodfunction.pdf', bbox_inches='tight')

 

# model = 'simSIR'
# loadInit = False
# for country in ('Germany','Italy','Uruguay', 'Saudi Arabia'): #, 
#     print(country)    
#     SIR = pp._setting(country, model)
#     x = pp._run(country, model,SIR, loadInit = loadInit, ithrun = 2)  
    

# model = 'hyperSIR'
# loadInit = False
# for country in ('Germany','Italy','Uruguay', 'Saudi Arabia'): #, 
#     print(country)    
#     if country == 'xx':
#         SIR = pp._setting(country, model)
#         x = pp._run(country, model,SIR, loadInit = loadInit, ithrun = 3)   

