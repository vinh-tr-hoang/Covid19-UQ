'''
This class load data from directory:
/jhudata/COVID-19/csse_covid_19_data/csse_covid_19_time_series//
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
class GetData:
    def __init__(self, countryRegion):
        cwd = os.getcwd()
        datad = "/jhudata/COVID-19/csse_covid_19_data/csse_covid_19_time_series//"
        data_set_confirmed = pd.read_csv(cwd + datad+"time_series_covid19_confirmed_global.csv")
        data_set_recovered = pd.read_csv(cwd + datad+"time_series_covid19_recovered_global.csv")
        data_set_deaths = pd.read_csv(cwd + datad+"time_series_covid19_deaths_global.csv")

        headers = list(data_set_confirmed)
        iItalia = np.where(data_set_confirmed['Country/Region']== countryRegion)[0][0]

        self.confirmed = np.array ([data_set_confirmed[headers[i]][iItalia]
                           for i in range (4, len(headers))])
        
        iItalia = np.where(data_set_recovered['Country/Region']== countryRegion)[0][0]
        self.recovered = np.array ([data_set_recovered[headers[i]][iItalia]
                           for i in range (4, len(headers))])

        iItalia = np.where(data_set_deaths['Country/Region']== countryRegion)[0][0]
        self.deaths = np.array ([data_set_deaths[headers[i]][iItalia]
                           for i in range (4, len(headers))])
        self.dates = headers[4:]
        self._compact = np.array([self.confirmed, self.recovered,
                                  self.deaths])
        self._compactHeaders = ["confirmed", "recovered",  "deaths"]
                # Data errors
    def plot (self):
        plt.plot(self.confirmed, label = 'data confirmedCase')
        plt.plot(self.recovered, '--', label = 'data recoveredCase')
        plt.plot(self.deaths,  '*-', label = 'data deaths')
        plt.yscale('log')
        #plt.show()       
    def _timeInterval_data (self, initT, deltaT):
        data = np.array([self.confirmed[initT:initT+deltaT],
                         self.recovered[initT:initT+deltaT],
                         self.deaths[initT:initT+deltaT]])
        return data


