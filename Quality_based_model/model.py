# ©2020-2021 ETH Zurich, Pagan Nicolò; D-ITET; Automatic Control Lab
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm    
from scipy import interpolate, stats
import os
import tikzplotlib
import pandas as pd
import powerlaw

from functions import *
from simulation import Simulation, Simulation_parameters
        
class Model_parameters:
    def __init__(self, N, nsim, save_results, folder, verbose):
        self.N = N
        self.nsim = nsim
        self.save_results = save_results
        self.folder = folder
        self.verbose = verbose
        
# The class Model can contain several instances of simulations with the same network size N, as well as for several values of N.
# The number of simulations for each value of N is nsim. N is a list of values for the network size.
class Model:
    def __init__(self, model_parameters, simulation_parameters):
        self.simulation_parameters = simulation_parameters
        self.N                     = model_parameters.N
        self.nsim                  = model_parameters.nsim
        self.save_results          = model_parameters.save_results
        self.folder                = model_parameters.folder
        self.verbose               = model_parameters.verbose
        
        self.time_steps_average         = np.zeros(len(self.N))
        self.time_steps_1q              = np.zeros(len(self.N))
        self.time_steps_2q              = np.zeros(len(self.N))
        self.time_steps_3q              = np.zeros(len(self.N))
        self.time_steps_std             = np.zeros(len(self.N))
        self.diameter_average           = np.zeros(len(self.N))
        self.diameter_std               = np.zeros(len(self.N))
        self.distance_average           = np.zeros(len(self.N))
        self.distance_std               = np.zeros(len(self.N))
        self.clustering_average         = np.zeros(len(self.N))
        self.clustering_std             = np.zeros(len(self.N))
        
        self.theoretical_pdf = []
        self.theoretical_ccdf= []
        self.numerical_dist  = []
        self.zipf_dist       = []
        self.powerlawfit     = []
        self.lognfit         = []
        self.mu              = []
        self.sigma           = []
    
        
        if (self.save_results):
            if not os.path.exists(self.folder):
                os.makedirs(self.folder)
            self.save_parameters()
            
        self.simulations = []
        for n in self.N:
            self.simulations.append([Simulation(n, self.simulation_parameters) for sim in range(self.nsim)])
        
    
    def save_parameters(self):
        file = open(self.folder+'parameters.txt', 'w')
        file.write('N: '       + str(self.N)                              + '\n')
        file.write('nsim: '    + str(self.nsim)                           + '\n')
        file.write('pPA: '     + str(self.simulation_parameters.pPA)      + '\n')
        file.write('speed_up:' + str(self.simulation_parameters.speed_up) + '\n')
        file.close()
            
            
    # The run function loops over the list of N. For each values, it runs nsim simulations.
    def run(self):
        for n in range(len(self.N)):
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print('N:', self.N[n])
            for sim in range(0,self.nsim):
                if(self.verbose):
                    print('-----------------------------------------------------------------------------')
                    print('Simulation', sim+1)
                self.simulations[n][sim].run()
                if(self.simulations[n][sim].verbose):
                    self.simulations[n][sim].print_statistics()
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                  
 
    ###################################
    
    # For time-steps analysis
    
    # Performs the time-steps analysis   
    def time_steps_analysis(self):
        self.compute_time_steps_statistics()   
        print('Time steps averages', self.time_steps_average)
        self.plot_time_steps_statistics()
        if (self.save_results):
            self.save_time_steps_statistics()
      
    # For each value of N in the model, the function computes the statistics related to the time-steps.
    def compute_time_steps_statistics(self):
        self.timesteps = []
        for n in range(len(self.N)):
            row = []
            self.timesteps.append(row)
            for sim in range(self.nsim):
                self.timesteps[n].append(self.simulations[n][sim].timesteps)
            self.time_steps_average[n] = np.average(row)
            self.time_steps_1q[n]      = np.quantile(row, 0.25)
            self.time_steps_2q[n]      = np.quantile(row, 0.5)
            self.time_steps_3q[n]      = np.quantile(row, 0.75)
            self.time_steps_std[n]     = np.std(row)
            
    # Time-steps statistics are plotted.
    def plot_time_steps_statistics(self):
        fig = plt.figure(figsize =(10, 7))
        ax = fig.add_axes([0, 0, 1, 1])
        # Creating plot
        bp = ax.boxplot(self.timesteps)
        plt.xlabel('N')
        plt.ylabel('time-steps')
        plt.title('Numerical Analysis time-steps')
        if (self.save_results):
            tikzplotlib.save(self.folder+'time_steps.tikz',  table_row_sep='\\\\\n')
        plt.show()


#     For each value of N in the model, the function computes the statistics related to the time-steps.
#     def compute_time_steps_statistics(self):
#         for n in range(len(self.N)):
#             timesteps = []
#             for sim in range(self.nsim):
#                 timesteps.append(self.simulations[n][sim].timesteps)
#             self.time_steps_average[n] = np.average(timesteps)
#             self.time_steps_1q[n]      = np.quantile(timesteps, 0.25)
#             self.time_steps_2q[n]      = np.quantile(timesteps, 0.5)
#             self.time_steps_3q[n]      = np.quantile(timesteps, 0.75)
#             self.time_steps_std[n]     = np.std(timesteps)
    
#     # Time-steps statistics are plotted.
#     def plot_time_steps_statistics(self):
#         plt.errorbar(self.N, self.time_steps_average, self.time_steps_std, linestyle='None', marker='^')
#         plt.xlabel('N')
#         plt.ylabel('time-steps')
#         plt.title('Numerical Analysis time-steps')
#         if (self.save_results):
#             tikzplotlib.save(self.folder+'time_steps.tikz',  table_row_sep='\\\\\n')
#         plt.show()
   

    # Saves the results of the time-steps analysis
    def save_time_steps_statistics(self):
        dict = {'N': self.N, 'Average Time Steps': self.time_steps_average.tolist(), 'Std Time Steps': self.time_steps_std.tolist(), 'First Quantile': self.time_steps_1q.tolist(), 'Second Quantile': self.time_steps_2q.tolist(), 'Third Quantile': self.time_steps_3q.tolist()}
        df = pd.DataFrame(dict)
        df.to_csv(self.folder+'time_steps_statistics.csv', index=False, sep ='\t')
   

    ###################################
    
    # For diameter analysis
    
    # Performs the diameter analysis. For each value of N, and for each simulation, it collects the diameter and the average distance. Data are aggregate per each value of N.
    # Average values are computed and stored. Plotting function is then called
    def diameter_analysis(self):    
        print('Diameter Analysis...')
        for n in range(len(self.N)):
            print('N:', self.N[n])
            diameters = []
            distances = []
            for sim in range(self.nsim):
                self.simulations[n][sim].compute_diameter()
                diameters.append(self.simulations[n][sim].diameter)
                for ele in self.simulations[n][sim].distances:
                    distances.append(ele)                
            self.diameter_average[n]             = np.average(diameters)
            self.diameter_std[n]                 = np.std(diameters)
            self.distance_average[n]     = np.average(distances)
            self.distance_std[n]         = np.std(distances)
            
        self.plot_diameter_statistics()
        if (self.save_results):
            self.save_diameter_statistics()
    
    # The functions plots the average diameter (with std) and average distance (with std), for each value of N.
    def plot_diameter_statistics(self):
        print('Plotting diameter statistics...')
        plt.errorbar(self.N, self.diameter_average, self.diameter_std, linestyle='None', marker='^', label='Diameter')
        plt.errorbar(self.N, self.distance_average, self.distance_std, linestyle='None', marker='^', label='Distance')
        plt.xlabel('N')
        plt.title('Numerical Analysis diameter and distance')
        plt.legend(loc="upper right")
        if (self.save_results):
            tikzplotlib.save(self.folder+'diameters.tikz',  table_row_sep='\\\\\n')
        plt.show()
     
    # Diameter analysis is saved
    def save_diameter_statistics(self):
        dict = {'N': self.N, 'Average Diameter': self.diameter_average.tolist(), 'Std Diameter': self.diameter_std.tolist(), 
                'Average Distance': self.distance_average.tolist(), 'Std distance': self.distance_std.tolist()}
        df = pd.DataFrame(dict)
        df.to_csv(self.folder+'diameter_statistics.csv', index=False, sep ='\t')
    
    
    ###################################
    
    # For clustering analysis
      
    # The clustering analysis is performed. Data are aggregate per value of N, across all the simulations.
    # An histogram (for each value of N) of the clustering coefficient is provided
    def clustering_analysis(self):
        for n in range(len(self.N)):
            clustering = []
            for sim in range(self.nsim):
                self.simulations[n][sim].compute_clustering()
                for value in self.simulations[n][sim].clustering:
                    clustering.append(value)
            plt.hist(clustering, bins='auto',  density=False, alpha=0.7, label='N='+str(self.N[n]))
            self.clustering_average[n] = np.average(clustering)
            self.clustering_std[n]     = np.std(clustering)
        plt.xlabel('Clustering')
        plt.ylabel('Frequency')
        plt.title('Clustering Distribution')
        plt.legend(loc="upper right")
        plt.show()
        if (self.save_results):
            tikzplotlib.save(self.folder+'clustering_hist_N='+str(self.N[n])+'.tikz',  table_row_sep='\\\\\n')
        self.plot_clustering_statistics()
    
    # The function plots the clustering coefficient (with std) as a function of N.
    def plot_clustering_statistics(self):
        plt.errorbar(self.N, self.clustering_average, self.clustering_std, linestyle='None', marker='^', label='Clustering')
        plt.xlabel('N')
        plt.title('Numerical Analysis of clustering')
        plt.legend(loc="upper right")
        dict = {'N': self.N, 'Average clustering': self.clustering_average.tolist(), 'Std clustering': self.clustering_std.tolist()}          
        df = pd.DataFrame(dict)
        if (self.save_results):
            tikzplotlib.save(self.folder+'clustering.tikz', table_row_sep='\\\\\n')
            df.to_csv(self.folder+'clustering.csv', index=False, sep ='\t')
        plt.show()
    
    # The following function saves the results related to the clustering analysis
    def save_clustering_statistics(self):
        dict = {'N': self.N, 'Average clustering': self.clustering_average.tolist(), 'Std clustering': self.clustering_std.tolist()}
        df = pd.DataFrame(dict)
        df.to_csv(self.folder+'clustering_statistics.csv', index=False, sep ='\t')
    
    
      
    ###################################
    
    # For overlap analysis
    
    # The function computes the overlap matrix for each simulation, and each value of N. For each value of N, the average overlap matrix (across the simulation with the same N)
    # is computed and plotted.
    def overlap_analysis(self, size):
        for n in range(len(self.N)):
            overlap_matrix = np.zeros([size+1,size+1])
            for sim in range(self.nsim):
                self.simulations[n][sim].compute_overlap(size)
                overlap_matrix = (overlap_matrix*sim+ self.simulations[n][sim].overlap_matrix)/(sim+1)
            plot_overlap(size, overlap_matrix, self.folder, self.save_results)
           
    ###################################
    
    # For Out-degree analysis
    
    # In the numerical outdegree analysis, the outdegree pdf is computed and plotted (against the theoretical, at equilibrium). 
    # n_tol defines the approximation for the theoretical distribution.
    # The analysis is done separately for each value of N.
    def numerical_outdegree_analysis(self, n_tol):
        for n in range(len(self.N)):
            outdegrees   = []
            outdegrees_pdf = np.zeros(self.N[n])
            for sim in range(self.nsim):
                for element in self.simulations[n][sim].outdegree:
                    outdegrees.append(element)
            outdegrees_pdf = count_occurences(outdegrees, self.N[n])
            outdegrees_pdf = outdegrees_pdf/len(outdegrees)
            plot_outdegree_probability(outdegrees_pdf, self.N[n], str(self.simulation_parameters.T), n_tol, self.save_results, self.folder+'N='+str(self.N[n])+'_')
    
    # The following function computes the fitting of the numerical outdegree distribution with:
    # - a powerlaw distribution
    # - a lognormal distribution
    # - a gamma distribution
    # It also prints the stastistics related to the fits, and returns the fitting parameters.
    # For each value of N, a different fit is computed.
    def fitting_outdegree_analysis(self):
        if (self.verbose):
            print('Fitting outdegree...')
        outdegrees = []
        for n in range(len(self.N)):
            for sim in range(self.nsim):
                for element in self.simulations[n][sim].outdegree: 
                    outdegrees.append(element)
            # numerical distribution
            self.numerical_dist.append(compute_numerical_dist(outdegrees, self.N[n]))
            # powerlaw fit
            self.powerlawfit.append(powerlaw.Fit(outdegrees, discrete=True, estimate_discrete=True, fit_method='Likelihood'))
            print('powerlaw', stats.kstest(outdegrees, "powerlaw", args=(self.powerlawfit[n].alpha, self.powerlawfit[n].xmin), N=len(outdegrees)))
            # lognormal fit
            sigma, loc, scale = stats.lognorm.fit(outdegrees, floc=0)
            self.sigma.append(sigma)
            self.mu.append(np.log(scale))
            self.lognfit.append(stats.lognorm(sigma, loc, scale))
            print('lognorm', stats.kstest(outdegrees, "lognorm", args=(np.mean(outdegrees), np.std(outdegrees)), N=len(outdegrees)))
            # gamma distribution fit
            fit_alpha, fit_loc, fit_beta=stats.gamma.fit(outdegrees)
            print('gamma', stats.kstest(outdegrees, "gamma", args=(fit_alpha, fit_loc, fit_beta))) 
            return outdegrees, self.numerical_dist[n], fit_alpha, fit_loc, fit_beta, self.powerlawfit, self.lognfit, self.mu, self.sigma

  
   
      ###################################
    
    # For In-degree analysis
  
    # The function aggregates the in-degree of the nodes from each simulation, but for the same value of N. By doing so, it removes null elements 
    # (which cannot be fit with powerlaw or lognormal distributions)
    def compute_indegrees(self):
        if (self.verbose):
            print('compute indegrees...')
        self.indegrees = []
        for n in range(len(self.N)):
            self.indegrees.append([])
            indegrees_with_zero = []
            for sim in range(self.nsim):
                for element in self.simulations[n][sim].indegree: 
                    indegrees_with_zero.append(element)
            indegrees_with_zero = np.array(indegrees_with_zero)
            self.indegrees[n] = indegrees_with_zero[indegrees_with_zero != 0]
    
    # The function provides the numerical analysis of the in-degree distribution as a function of the nodes rank. The analysis is done separately for each value of N.
    def numerical_indegree_analysis(self):
        for n in range(len(self.N)):
            indegrees   = np.zeros([self.N[n], self.nsim])
            averages    = np.zeros(self.N[n])
            indegrees_pdf = np.zeros([self.N[n], self.N[n]])
            for sim in range(self.nsim):
                for i in range(self.N[n]):
                    indegrees[i,sim] = self.simulations[n][sim].indegree[i]
            for i in range(self.N[n]):
                averages[i] = np.mean(indegrees[i,:])
                indegrees_pdf[i] = count_occurences(indegrees[i], self.N[n])
            indegrees_pdf = indegrees_pdf/self.nsim
            plot_indegree_probability(indegrees_pdf, averages, self.N[n], 'infinity', self.save_results, self.folder)         
    
    # The following function provides different fits for the average in-degree probability function.
    def fitting_indegree_analysis(self):
        if (self.verbose):
            print('Fitting indegree...')
        for n in range(len(self.N)):
            indegrees_with_zero = []
            for sim in range(self.nsim):
                for element in self.simulations[n][sim].indegree: 
                    indegrees_with_zero.append(element)
            indegrees_with_zero = np.array(indegrees_with_zero)
            indegrees = indegrees_with_zero[indegrees_with_zero != 0]
            # theoretical distribution
            indegree_table_inf = indegree_prob_table_inf(self.N[n])
            self.theoretical_pdf.append(compute_indegree_pdf(indegree_table_inf, self.N[n]))
            self.theoretical_ccdf.append(compute_indegree_ccdf(indegree_table_inf, self.N[n]))
            # numerical distribution
            self.numerical_dist.append(compute_numerical_dist(indegrees_with_zero, self.N[n]))
            # Zipf distribution
            zipfs_law_x = range(1,self.N[n])
            zipfs_law_y = (self.N[n]*np.power(np.linspace(1,self.N[n],self.N[n]), -1))
            zipfs_law_y[0] = self.N[n]-1
            indegrees_zipf = np.tile(zipfs_law_y, self.nsim)
            self.zipf_dist.append(compute_numerical_dist(indegrees_zipf, self.N[n]))
            # powerlaw fit
            self.powerlawfit.append(powerlaw.Fit(indegrees, discrete=True, estimate_discrete=True, fit_method='Likelihood'))
            print ('power law fit data', str(self.powerlawfit[n].alpha), str(self.powerlawfit[n].xmin))
            print('powerlaw', stats.kstest(indegrees, "powerlaw", args=(self.powerlawfit[n].alpha, self.powerlawfit[n].xmin), N=len(indegrees)))
            # lognormal fit
            sigma, loc, scale = stats.lognorm.fit(indegrees, floc=0)
            self.sigma.append(sigma)
            self.mu.append(np.log(scale))
            self.lognfit.append(stats.lognorm(sigma, loc, scale)) 
            print('lognorm', stats.kstest(indegrees, "lognorm", args=(np.mean(indegrees), np.std(indegrees)), N=len(indegrees)))
    
    # The function provides the binned fit of the average in-degree distribution
    def plot_binned_hist_pdf(self, nbins):
        if (self.verbose):
            print('Plot binned histogram')
        for n in range(len(self.N)):                   
            # log-scaled bins
            bins = np.logspace(0, np.log10(self.N[n]), nbins)
            widths = (bins[1:] - bins[:-1])
            hist = np.histogram(self.indegrees[n], bins=bins)
            hist_norm = hist[0]/widths
            hist_norm = hist_norm/sum(hist_norm)
            plt.scatter(bins[:-1], hist_norm, alpha=0.5)
            plt.bar(bins[:-1], hist_norm, widths, alpha=0.5)
            plt.xscale('log')
            plt.yscale('log')
            if (self.save_results):
                tikzplotlib.save(self.folder+'indegree_binned_hist_'+str(self.N[n])+'.tikz',  table_row_sep='\\\\\n')
            plt.show()
    
    # For each value of N, the function plots both the average in-degree pdf and ccdf, with their fits.
    def indegree_analysis_plot(self, desired_xmin, ymin):
        if (self.verbose):
            print('Indegree analysis plot...')
        for n in range(len(self.N)):
            plot_fit_pdf(self.theoretical_pdf[n], self.numerical_dist[n], self.powerlawfit[n], desired_xmin[0], self.lognfit[n], self.mu[n], self.sigma[n], self.N[n], ymin[0], self.save_results, self.folder)
            plot_fit_ccdf(self.theoretical_ccdf[n], self.numerical_dist[n], self.powerlawfit[n], desired_xmin[1], self.lognfit[n], self.mu[n], self.sigma[n], self.N[n], ymin[1], self.zipf_dist[n], self.save_results, self.folder)
    
    # The following function compares the powerlaw fit with an alternative distribution, e.g., lognormal.
    def compare_fit_distribution(self, alternative_distribution):
        if (self.verbose):
            print('Comparing fit distributions...')
        for n in range(len(self.N)):
            print('N:', self.N[n])  
            R, p = self.powerlawfit[n].distribution_compare('power_law', alternative_distribution)
            print ('R: ', str(R), 'p:', str(p))
            print ('If R < 0, the alternative distribution (', alternative_distribution, ') is favoured with respect to the powerlaw.')
       
