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
from matplotlib.colors import LinearSegmentedColormap
from scipy import interpolate, stats
from scipy.stats import rv_histogram
import tikzplotlib
import powerlaw
import pandas as pd
from pytictoc import TicToc


# The following function computes the theoretical in-degree probability density function, and stores it in two tables (p and p_inf). 
# The first one refers to the non-equilibrium state, and the second one to the equilibrium state).
# The function uses some of the sub-functions below, to speed-up the process that builds the large matrices, e.g., the computation of the powers of p_i and 1-p_i, 
# as defined in Theorem 2 and Corollary 1. 
def compute_indegree_probability(n, d, i, nchoosek_table, p1table, p1table_inf, p2table, p2table_inf, oneminus_p1table, oneminus_p1table_inf, oneminus_p2table, oneminus_p2table_inf):
    p = 0
    p_inf = 0
    rank = i+1
    if (rank==1):
        b2 = nchoosek_table[n-rank,d]
        if (b2>0):
            p      = b2*p2table[i][d] *oneminus_p2table[i][n-rank-d]
            p_inf  = b2*p2table_inf[i][d] *oneminus_p2table_inf[i][n-rank-d]
    else:
        for k in range(d+1):
            if (k>i):
                b1=0
            else:
                b1 = nchoosek_table[rank-1,k]
            if (d>n-i+k):
                b2 = 0
            else:
                b2 = nchoosek_table[n-rank, d-k]
            if (b1>0 and b2>0):
                p      = p     +  b1*p1table[i][k] *oneminus_p1table[i][rank-1-k]*b2*p2table[i][d-k] *oneminus_p2table[i][(n-rank)-(d-k)]
                p_inf  = p_inf +  b1*p1table_inf[i][k] *oneminus_p1table_inf[i][rank-1-k]*b2*p2table_inf[i][d-k] *oneminus_p2table_inf[i][(n-rank)-(d-k)]     
    return p, p_inf

def indegree_prob_table_inf(n):
    [indegree_table, indegree_table_inf] = indegree_prob_table(n, 1)
    return indegree_table_inf

def indegree_prob_table(n, T):
    nchoosek_table = np.eye(n)
    nchoosek_table[:,0] = 1
    for i in range(1,n):
        for j in range(1,i):
            nchoosek_table[i,j] = nchoosek_table[i-1, j-1]+ nchoosek_table[i-1,j]
    p1table               = []
    p2table               = []
    p1table_inf           = []
    p2table_inf           = []
    oneminus_p1table      = []
    oneminus_p2table      = []
    oneminus_p1table_inf  = []
    oneminus_p2table_inf  = []
    
    for i in range(n):
        rank = i+1
        if (rank>1):
            p1                    = 1/(rank-1)*(1-((n-rank)/(n-1))**T)
            p1_inf                = 1/(rank-1)
        else:
            p1                    = np.NaN
            p1_inf                = np.NaN
            
        p2                        = 1/rank*(1-((n-rank-1)/(n-1))**T) 
        p2_inf                    = 1/rank
        p1table.append(fill_probability_table_row(p1, n))
        p2table.append(fill_probability_table_row(p2, n))
        p1table_inf.append(fill_probability_table_row(p1_inf, n))
        p2table_inf.append(fill_probability_table_row(p2_inf, n))
        oneminus_p1table.append(fill_probability_table_row(1-p1, n))
        oneminus_p2table.append(fill_probability_table_row(1-p2, n))
        oneminus_p1table_inf.append(fill_probability_table_row(1-p1_inf, n))
        oneminus_p2table_inf.append(fill_probability_table_row(1-p2_inf, n))  
    indegree_prob_table = np.zeros([n,n])
    indegree_prob_table_inf = np.zeros([n,n])
    for i in range(n):
        for d in range(n):
            indegree_prob_table[i, d], indegree_prob_table_inf[i,d] = compute_indegree_probability(n, d, i, nchoosek_table, p1table, p1table_inf, p2table, p2table_inf, 
                                                                                                     oneminus_p1table, oneminus_p1table_inf, oneminus_p2table, oneminus_p2table_inf)
    return indegree_prob_table, indegree_prob_table_inf

def fill_probability_table_row(p, n):
    row = []
    row.append(1)
    row.append(p)
    for i in range(2,n):
        row.append(row[i-1]*p)
    return row


############################################################################################

  
# The function computes the expected in-degree (as a function of the quality ranking), once receiving the in-degree probability density function.
def compute_expected_indegree(indegree_table):
    n = len(indegree_table)
    expected_indegree = np.zeros(n)
    for i in range(n):
        for j in range(n):
            expected_indegree[i] = expected_indegree[i]+j*indegree_table[i,j]
    return expected_indegree
  
# The function plots the colormap with the in-degree probability distribution as a function of the quality ranking.
# It also computes the linear fit of the average values by using the function below.
def plot_indegree_probability(indegree_table, average, n, t, save_results, folder):
#     n_max, linear_model = fit_log_log(average)
#     linear_model_fn=np.poly1d(linear_model)
#     fit_slope = linear_model[0]
    y, x = np.mgrid[slice(1, n, 1), slice(0.5, n+0.5, 1)]
#    plt.pcolormesh(x, y, indegree_table[:,1:-1].T, vmin=0, vmax=1, cmap=scaled_cmap((cm.gray_r), 5), label='') # gray_r for reverse map
    plt.pcolormesh(x, y, indegree_table[:,1:-1].T, vmin=0, vmax=1, cmap=scaled_cmap((cm.gray), 5), label='') 
#     zipfs_law_x = range(1,n+1)
#     zipfs_law_y = n* np.power(np.linspace(1,n,n), -1)
#     zipfs_law_y[0] = n-1
    plt.xlim(1,n)
    plt.ylim(1,n-1)
    plt.xscale('log')
    plt.yscale('log')
    plt.axis('off')
    plt.savefig(folder+'indegree_rank_t='+t+'.png', bbox_inches='tight', transparent=True, pad_inches=0)
    plt.colorbar()
    plt.axis('on')
    plt.plot(range(1,n+1), average, marker='*', label='Average value', alpha=0.7)
#     plt.plot(range(1,n_max), np.exp(linear_model_fn(np.log(range(1,n_max)))), label='Fit of average, n_max='+str(n_max)+', slope='+str(fit_slope), alpha = 0.7  )
#     plt.plot(zipfs_law_x, zipfs_law_y, marker='*', label='Zipf\'s law', alpha=0.7)
    plt.xlabel('rank')
    plt.ylabel('indegree')
    plt.title(['Theoretical indegree pdf, t=' + t])
    plt.legend(loc="upper right")
    if (save_results):
        tikzplotlib.save(folder+'indegree_rank_t='+t+'.tikz',  table_row_sep='\\\\\n')
        np.savetxt(folder+'indegree_rank_t='+t+'.txt', indegree_table)
    plt.show()
    
# The following function computes the linear fit (after transforming the data in log-log scale).    
def fit_log_log(values):
    n = len(values)
    n_min = 1
    n_max = n
    min_values = np.where(values<=1)[0]
    if (len(min_values)>0):
        n_max = min_values[0]
    linear_model=np.polyfit(np.log(range(n_min, n_max+1)),np.log(values[n_min-1:n_max]),1)
    return n_max, linear_model
    

# The following function computes the theoretical outdegree probability density function and stores it in a matrix.
# In each row i, the probability density function of a network with i agents is stored.
# To spee-up the process, the matrix is computed with n_tol columns, as the probability of large out-degree for the nodes is decreasing very fast.
def compute_outdegree_table(n, n_tol):
    outdegree_table = np.zeros([n, min(n,n_tol+1)])
    for row in range(n):
        outdegree_table[row,0] = 0
    for row in range(1,n):
        outdegree_table[row,1] = 1/row
    for row in range(2,n):
        for col in range(2,min(row,n_tol)+1):
            for l in range(row):
                outdegree_table[row,col] = outdegree_table[row,col] + outdegree_table[l, col-1]
            outdegree_table[row,col] = outdegree_table[row,col]/row
    return outdegree_table

# Similarly to the above function, the following function computes the theoretical outdegree probability density function and stores it in a matrix.
# The process is accelerated by storing some temporary elements
def compute_outdegree_table_fast(n, n_tol):
    outdegree_table = np.zeros([n, min(n,n_tol+1)])
    outdegree_table_cum = np.zeros([n, min(n,n_tol+1)])
    for row in range(n):
        outdegree_table[row,0] = 0
        outdegree_table_cum[row,0] = 0
    for row in range(1,n):
        outdegree_table[row,1] = 1/row
        outdegree_table_cum[row,1] = outdegree_table_cum[row-1,1]+outdegree_table[row,1]
    for col in range(2, min(n,n_tol)+1):
        for row in range(col, n):
            outdegree_table[row,col] = outdegree_table_cum[row-1, col-1]/row
            outdegree_table_cum[row,col] = outdegree_table_cum[row-1,col] + outdegree_table[row,col]
    return outdegree_table, outdegree_table_cum

# The function returns the expected out-degree, as in Theorem 3.
def compute_expected_outdegree(n):
    expected_outdegree = np.zeros(n+1)
    for i in range(1, n+1):
        expected_outdegree[i] = expected_outdegree[i-1]+1/i
    return expected_outdegree

# The function returns the std-dev of the out-degree by using the out-degree probability density function stored in the table.
def compute_stddev_outdegree(n, outdegree_table, expected_outdegree, n_tol):
    variance_outdegree = np.zeros(n)
    for i in range(n):
        for j in range(min(i, n_tol)):
            variance_outdegree[i] = variance_outdegree[i]+outdegree_table[i,j]*(j-expected_outdegree[i])**2
    return np.sqrt(variance_outdegree)

# The function plots the numerical out-degree probability density function and compares it with the theoretical one for the same value of N.
def plot_outdegree_probability(outdegree_pdf, n, T, n_tol, save_results, folder):
    plt.plot(range(n), outdegree_pdf, label='Numerical')
    theoretical_pdf = compute_outdegree_table(n, n_tol)
    plt.plot(range(0,min(n,n_tol)), theoretical_pdf[n-1, 0:min(n,n_tol)], label='Theoretical N='+str(n))
    plt.ylim([10**(-6),1])
    plt.xlim([1,30])
    plt.xlabel('Outdegree')
    plt.title('Outdegree Probability density function')
    plt.legend(loc="upper right")
    if (save_results):
        tikzplotlib.save(folder+'outdegree_pdf.tikz',  table_row_sep='\\\\\n')
        np.savetxt(folder+'outdegree_pdf.txt', outdegree_pdf)
    plt.show()


# The following function computes the numerical distribution of a set of data.
def compute_numerical_dist(data, n):
    r = rv_histogram(np.histogram(data, bins=range(n)))
    return r

# The following function computes the average in-degree probability density function, from the pdf of each agent.
def compute_indegree_pdf(indegree_table,n):
    indegree_pdf = indegree_table.sum(axis=0)
    indegree_pdf = indegree_pdf/n
    return indegree_pdf

# The following function returns the average complementary cumulative distribution function.
def compute_indegree_ccdf(indegree_table, n):
    return np.cumsum(compute_indegree_pdf(indegree_table, n)[::-1])[::-1]

# The next function computes the ccdf but excluding from the data those with in-degree < xmin, to ease the comparison with power-law fits.
def compute_indegree_ccdf_without_xmin(indegree_table, n, xmin):
    pdf = compute_indegree_pdf(indegree_table, n)
    pdf = pdf[xmin:]
    pdf_sum = sum(pdf)
    pdf = pdf/pdf_sum
    ccdf = np.cumsum(pdf[::-1])[::-1]
    return ccdf   


# The function plots the pdf of the average in-degree.
def plot_indegree_pdf(indegree_table, n, save_results, folder):
    indegree_pdf = compute_indegree_pdf(indegree_table, n)
    sum_indegree_pdf = sum(indegree_pdf)
    indegree_pdf = indegree_pdf/sum_indegree_pdf
    plt.scatter(range(n), indegree_pdf)
    plt.ylim([10**(-8),1])
    plt.xlim([1,n])
    plt.xlabel('Indegree')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Theoretical Indegree Probability density function')
    if (save_results):
        tikzplotlib.save(folder+'indegree_pdf.tikz',  table_row_sep='\\\\\n')
        np.savetxt(folder+'indegree_pdf.txt', indegree_pdf)
    plt.show()

# The function plots the ccdf of the average in-degree.
def plot_indegree_ccdf(indegree_table, n, save_results, folder):
    indegree_ccdf = compute_indegree_ccdf(indegree_table, n)
    plt.scatter(range(n), indegree_ccdf)
    plt.ylim([10**(-3),1])
    plt.xlim([1,n])
    plt.xlabel('Indegree')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Theoretical Indegree Complementary Cumulative (Survival) distribution function')
    if (save_results):
        tikzplotlib.save(folder+'indegree_ccdf.tikz',  table_row_sep='\\\\\n')
        np.savetxt(folder+'indegree_ccdf.txt', indegree_ccdf)
    plt.show()

# The function plots the ccdf of the numerical and theoretical distribution of the average in-degree with the different fits (powerlaw and lognormal)    
def plot_fit_ccdf(theoretical_ccdf, numerical_dist, powerlawfit, desired_xmin, lognfit, mu, sigma, n, ymin, zipf_dist, save_results, folder):
    xmin = int(powerlawfit.power_law.xmin)
    powerlaw_fit_distribution = powerlaw.Power_Law(xmin=desired_xmin, parameters=[powerlawfit.alpha], discrete=True) # xmin=2 might not be good for different values of N. This is a bit of a hack, because of the powerlaw interpolation package
    plt.loglog(range(xmin,n), powerlaw_fit_distribution.ccdf(range(xmin,n)), color='b', label='fit ccdf: alpha=' +str(powerlawfit.power_law.alpha)+' sigma='+str(powerlawfit.power_law.sigma)+' xmin='+str(int(powerlawfit.power_law.xmin)), alpha=0.5)
    plt.scatter(range(1,n+1), 1-numerical_dist.cdf(range(n)), s=25, color='r', marker='o', label='Numerical data', alpha=0.5)
    plt.scatter(range(n), 1-zipf_dist.cdf(range(n)), s=25, color='black', marker='o', label='Zipf', alpha=0.5)
    plt.scatter(range(n), theoretical_ccdf,  s=25, color='g', marker='o', label='theoretical data', alpha=0.5)
    plt.plot(range(n), 1-lognfit.cdf(range(n)), color='black', label='lognormal fit, mu='+str(mu)+' sigma='+str(sigma), alpha=0.5)
    plt.title('Indegree complementary Cumulative Distribution Function and power-law fit, normalized after xmin')
    plt.legend(loc="upper right")
    plt.xlim([1,n])
    plt.ylim([ymin,1])
    plt.xlabel('indegree')
    plt.ylabel('ccdf')
    if (save_results):
        tikzplotlib.save(folder+'indegree_fit_ccdf.tikz',  table_row_sep='\\\\\n')
        np.savetxt(folder+'indegree_fit_ccdf.txt', powerlawfit.power_law.ccdf())
    plt.show()

# The function plots the pdf of the numerical and theoretical distribution of the average in-degree with the different fits (powerlaw and lognormal)    
def plot_fit_pdf(theoretical_pdf, numerical_dist, powerlawfit, desired_xmin, lognfit, mu, sigma, n, ymin, save_results, folder):
    xmin = int(powerlawfit.power_law.xmin)
    powerlaw_fit_distribution = powerlaw.Power_Law(xmin=desired_xmin, parameters=[powerlawfit.alpha], discrete=True) # xmin=2 might not be good for different values of N. This is a bit of a hack, because of the powerlaw interpolation package
    plt.loglog(range(xmin,n), powerlaw_fit_distribution.pdf(range(xmin,n)), color='b', label='fit pdf: alpha=' +str(powerlawfit.power_law.alpha)+' sigma='+str(powerlawfit.power_law.sigma)+' xmin='+str(int(powerlawfit.power_law.xmin)), alpha=0.5)
    plt.scatter(range(n), numerical_dist.pdf(range(n)), s=25, color='r', marker='o', label='Numerical data', alpha=0.5)
        
    plt.scatter(range(n), theoretical_pdf,  s=25, color='g', marker='o', label='theoretical data', alpha=0.5)
    plt.plot(range(n), lognfit.pdf(range(n)), color='black', label='lognormal fit, mu='+str(mu)+' sigma='+str(sigma), alpha=0.5)
    plt.title('Indegree Probability Density Function and power-law fit, normalized after xmin')
    plt.legend(loc="upper right")
    plt.xlim([1,n])
    plt.ylim([ymin,1])
    plt.xlabel('indegree')
    plt.ylabel('pdf')
    if (save_results):
        dict = {'d': range(n), 'Numerical': numerical_dist.pdf(range(n)).tolist(), 'Theoretical': theoretical_pdf.tolist(), 'Lognormal': lognfit.pdf(range(n)).tolist(),'Powerlaw': np.concatenate((np.NaN*np.ones(xmin),powerlaw_fit_distribution.pdf(range(xmin,n))), axis=0)}          
        df = pd.DataFrame(dict)
        tikzplotlib.save(folder+'indegree_fit_pdf.tikz',  table_row_sep='\\\\\n')
        df.to_csv(folder+'indegree_fit_pdf.csv', index=False, sep ='\t')
    plt.show()
       
# The function plots the overlap matrix and stores it.        
def plot_overlap(size, matrix, folder, save_results):
    y, x = np.mgrid[slice(0, size+1, 1), slice(0, size+1, 1)]
    plt.pcolor(x, y, matrix[0:size+1, 0:size+1], cmap=scaled_cmap(cm.hot, 2))
    plt.xlim(0,size)
    plt.ylim(0,size)
    plt.gca().invert_yaxis()
    plt.colorbar()
    dict = {'x': (np.reshape(x+0.5, ((size+1)**2),1)), 'y': (np.reshape(y+0.5, ((size+1)**2),1)), 'c': (np.reshape(matrix, ((size+1)**2),1))}
    df = pd.DataFrame(dict)
    if (save_results):
        tikzplotlib.save(folder+'overlap.tikz',  table_row_sep='\\\\\n')
        df.to_csv(folder+'overlap_matrix.csv', index=False, sep ='\t')
    plt.show()


# The function is used to make the colormap scale non-linear 
def scaled_cmap(original_map, scalingIntensity):
    R = []
    G = []
    B = [] 
    for i in range(255):
        R.append(original_map(i)[0])
        G.append(original_map(i)[1])
        B.append(original_map(i)[2])
    dataMax = 1;
    dataMin = 0;
    centerPoint = 0;
    x = np.linspace(1,255, 255)
    x = x - (centerPoint-dataMin)*len(x)/(dataMax-dataMin)
    x = scalingIntensity * x/np.max(np.abs(x))
    x = np.sign(x)* np.exp(np.abs(x))
    x = x - min(x); 
    x = x*511/max(x)+1
    fR = interpolate.interp1d(x, R)
    fG = interpolate.interp1d(x, G)
    fB = interpolate.interp1d(x, B)
    colormapR = fR(np.linspace(1,512,512))
    colormapG = fG(np.linspace(1,512,512))
    colormapB = fB(np.linspace(1,512,512))
    colormap = np.zeros([512,3])
    for i in range(512):
        colormap[i] = [colormapR[i], colormapG[i], colormapB[i]]
    colormap[-1] = [1, 1, 1]
    return LinearSegmentedColormap.from_list('mycmap', colormap, N=512)

# the function returns the occurrences in a vector.
def count_occurences(vector, n):
    vector_pdf = np.zeros(n)
    for i in vector:
        vector_pdf[int(i)]=vector_pdf[int(i)]+1
    return vector_pdf
    
