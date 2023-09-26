# ©2020-2021 ETH Zurich, Pagan Nicolò; D-ITET; Automatic Control Lab
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"]=[30,15]
import pandas as pd
import networkx as nx
from pytictoc import TicToc
import scipy.sparse
from scipy.sparse import csr_matrix, lil_matrix

from functions import *



class Simulation_parameters:
    def __init__(self, T, pPA, speed_up, verbose):
        # maximum number of time-steps. If set to NaN, it will run until convergence
        self.T = T
        # the parameter defines whether the simulation will try to implement a speed-up. When active, the number of time-steps is significantly reduced, but the resulting network is unchanged.
        self.speed_up = speed_up
        # defines whether a Preferential attachment process is used or not. If pPA == 0, the selection is purely random, if pPA == 1, the selection is purely based on Pref. attachment.
        self.pPA = pPA
        if (self.pPA == -1):
            self.extremePA = True
        else:
            self.extremePA = False
        # for extra information on the simulation
        self.verbose = verbose
        
class Simulation:
    def __init__(self, N, simulation_parameters):
        # number of agents
        self.N        = N
        # same parameters as defined in Simulation_parameters
        self.T        = simulation_parameters.T
        self.speed_up = simulation_parameters.speed_up
        self.pPA      = simulation_parameters.pPA
        self.extremePA= simulation_parameters.extremePA
        self.verbose  = simulation_parameters.verbose
        
        # Creating Adjacency matrix of the graph
        self.A = lil_matrix((self.N, self.N))
        # Creating q and sorting in descending order
        self.q = np.random.rand(self.N)
        self.q = np.sort(self.q, axis=0)[::-1]
        
        # Initializing the arrays for the network analysis
        self.indegree         = np.zeros(self.N, dtype=np.int64)
        self.outdegree        = np.zeros(self.N, dtype=np.int64)
        self.triads           = np.zeros(self.N, dtype=np.int64)
        self.closed_triads    = np.zeros(self.N, dtype=np.int64)
        self.clustering       = np.zeros(self.N)
        self.diameter         = np.NaN
        self.average_distance = np.NaN
        self.timesteps        = np.NaN
        
        # The target of all the nodes (but node 1), is node 1 (which has rank 0 in the code). The target of node 1 (which has rank 0 in the code)
        # is node 2 (which has rank 1 in the code).
        self.optimal_utility    = np.zeros((self.N), dtype=np.int64)
        self.optimal_utility[0] = 1
        self.utility            = (self.N)*np.ones((self.N), dtype=np.int64)


    # The function checks convergence of the simulation. To do so, it checks whether each node has found its optimal target.
    def converged(self, t):
        if (np.all(self.optimal_utility == self.utility)):
            if (self.verbose):
                print('Simulation has converged in ', t, 'timesteps')
            return True
        else:
            return False
    
    # The simulation is run until convergence, or until t=T time-steps, if T is defined.
    def run(self):
        t = 0
        if (self.verbose): 
            tt=TicToc()
            tt.tic()
        while (t<self.T or np.isnan(self.T)):
            if (self.converged(t)==True):
                break
            else:
                t = t+1
                self.run_timestep(t)
        if (self.verbose): 
            tt.toc()
        self.timesteps = t

        
    # Time-step of the simulation. The meeting process is either from a uniform distribution (if self.pPA==0), or from a PA process (if self.pPA==1), or
    # in between.
    # Speed-up only works for non-PA. It makes meeting always being successful. It affects the time-step analysis, but not the other analysis, nor 
    # the converged state.
    def run_timestep(self,t):
        indegree = self.indegree
        if (self.extremePA):
            max_indegree = np.max(indegree)
            p_extremePA = np.double(indegree == max_indegree)
            p_extremePA = p_extremePA/sum(p_extremePA)
            n_max = np.count_nonzero(p_extremePA)
        for i in range(self.N):
            # if the agent i has not found the optimal target, it searches for a new connection
            if (self.extremePA):
                j=i
                if (n_max>1):
                    while(j==i):
                        j = np.random.choice(range(0, self.N), 1, p=p_extremePA)
                else:
                    j = np.random.choice(range(0, self.N), 1, p=p_extremePA)
                # Best response is performed
                if (j!=i):
                    self.BR(i, j)
            else:
                if (self.optimal_utility[i] != self.utility[i]):
                    j=i
                    # The new connection is drawn according to the meeting scenario
                    while (j==i):
                        if (self.speed_up==False):
                            # the following statement is almost always true if pPA is set to 0. In this case, the new connection is drawn uniformly at random
                            if (np.random.rand()>self.pPA):
                                j = np.random.randint(0, self.N)
                            # Otherwise, with probability pPA, the agent j is selected proportionally to their indegree, and with probability 1-pPA at random (previous case).
                            else:
                                j = np.random.choice(range(0, self.N), 1, p=(indegree+np.ones(self.N))/sum(indegree+np.ones(self.N)))
                        else:
                            # If the speed-up is activated (and the meeting distribution process is uniform), 
                            # the  agent j is drawn from the list of agents to which i would form a succesfull connection.
                            if (np.random.rand()>self.pPA):
                                j = np.random.randint(0, self.utility[i])
                            else:
                                print("Not implemented.")
                                break
                    # Best response is performed
                    self.BR(i, j)
                              
#    Core function of the dynamics:
#    if the new agent j has a quality ranking lower than the current utility of i, than i follows j.
    def BR(self, i,j):
        # if i does not follow anyone, its utility is initialized to N.
        if (self.utility[i]==self.N):
            # i follows j
            self.A[i,j]=1
            # the utility of i is updated with the rank j. This can be done assuming agents are ranked in decreasing quality order
            self.utility[i] = j
            # the in-degree and out-degree of the nodes are updated accordingly
            self.indegree[j] = self.indegree[j]+1
            self.outdegree[i] = self.outdegree[i]+1
        else:
            # assuming agent are ranked in decreasing quality order, the conditions checks whether the quality ranking of j is better than the utility of i, 
            # in which the ranking of the last connection is stored.
            if (j<self.utility[i]):
                # if i does not already follow j
                if (self.A[i,j]!=1):
                    # i starts following j
                    self.A[i,j]=1
                    # the current utility of i is updated to j (the rank j is used instead of the quality of j)
                    self.utility[i] = j
                    # the in-degree and out-degree of the nodes are updated accordingly
                    self.indegree[j] = self.indegree[j]+1
                    self.outdegree[i] = self.outdegree[i]+1

    # Tools functions for printing
    def print_statistics(self):
        self.print_adjacency()
        self.print_indegree()
        self.print_outdegree()
    
    def print_adjacency(self):
        print('Adjacency matrix:')
        print(self.A)
    
    def print_indegree(self):
        print('Indegree:')
        print(self.indegree)
            
    def print_outdegree(self):
        print('Outdegree:')
        print(self.outdegree)
        
    # Compute overlap matrix between different followers sets
    # size: is the number of rows/columns of the computed overlap matrix.
    def compute_overlap(self, size):
        self.overlap_matrix = np.zeros((size+1, size+1))
        for i in range(size+1):
            for j in range(size+1):
                common_followers = 0
                for k in range(self.N):
                    if (self.A[k,i]*self.A[k,j]==1):
                        common_followers = common_followers+1
                i_indegree = self.indegree[i]
                j_indegree = self.indegree[j]
                if (self.A[j,i] == 1):
                    i_indegree = i_indegree-1
                if (self.A[i,j] == 1):
                    j_indegree = j_indegree-1
                if (i_indegree > 0):
                    self.overlap_matrix[i,j] = common_followers/i_indegree
                if (j_indegree > 0):
                    self.overlap_matrix[j,i] = common_followers/j_indegree
    
    # Saves the overlap into a text file
    def save_overlap(self, folder):
        np.savetxt(folder+'overlap_matrix.txt', self.overlap_matrix)
    
    # Computes the clustering coefficient of each node and stores it in the structure named clustering. 
    # It is computed only for the nodes with outdegree strictly larger than 1.
    def compute_clustering(self):
        self.clustering     = []
        sparseA = csr_matrix(self.A)
        B = csr_matrix(sparseA.dot(sparseA))
        sparseAT = csr_matrix(sparseA.transpose())
        for i in range(self.N):
            if (self.outdegree[i]>1):
                paths = np.dot(B[i,:].todense(), sparseAT[:,i].todense())
                self.clustering.append(paths[0,0]/(self.outdegree[i]*(self.outdegree[i]-1)))
        
    # Plots the histogram of the clustering coefficient
    def plot_clustering(self):
        plt.hist(self.clustering, bins='auto',  density=False, color='#0504aa', alpha=0.7)
        plt.xlabel('Clustering')
        plt.ylabel('Frequency')
        plt.title('Clustering')
        plt.show()
    
    # Computes the diameter and the average distance in the network
    def compute_diameter(self):
        graph=nx.from_scipy_sparse_matrix(self.A, create_using=nx.DiGraph)
        spl = dict(nx.all_pairs_shortest_path_length(graph))
        self.distances = []
        for i in spl:
            for j in spl[i]:
                self.distances.append(spl[i][j])
        self.diameter = max(self.distances)
        self.average_distance = np.mean(self.distances)
    
    # Plots the network
    def plot_graph(self):
        graph=nx.from_scipy_sparse_matrix(self.A, create_using=nx.DiGraph)
        nx.draw_networkx(graph)
    
    # Saves the graph in a format that can be opened with, e.g., Gephi. In particular, the nodes list is saved, with the node ID and the quality,
    # and the edges list.
    def save_graph(self, folder):
        dict = {'Id': np.linspace(1,self.N,self.N, dtype=np.int64), 'Quality': self.q}
        nodes_df = pd.DataFrame(dict)
        nodes_df.to_csv(folder+'nodes_list.csv', index=False)
        G = nx.from_scipy_sparse_matrix(self.A, create_using=nx.DiGraph)
        edges = list(G.edges())
        source = []
        target = []
        for edge in edges:
            source.append(edge[0]+1)
            target.append(edge[1]+1)
        edges_dict = {'Source': source, 'Target': target}
        edges_df = pd.DataFrame(edges_dict)
        edges_df.to_csv(folder+'edges_list.csv', index=False)
        

 