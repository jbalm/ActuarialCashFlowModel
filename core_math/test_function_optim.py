# Python Programs
from . import functions_credit
from . import function_optim
## Python packages
import numpy as np
import unittest
import pickle 
from scipy.optimize import minimize
import numpy.linalg as la
import xlwings as xw
import matplotlib.pyplot as plt

        
        
def test_function_spread():
    mu=5
    alpha=0.1   
    sigma=0.75
    recovery_rate=0.35
    
    with open('data\pickle\historical_transition_matrix.pkl', 'rb') as input:
        historical_transition_matrix = pickle.load(input)
            
    historical_generator_matrix = functions_credit.generator_matrix(historical_transition_matrix)
        
    w, v = la.eig(historical_generator_matrix)
    eigenval_hist_gen = w.real
    eigenvect_hist_gen = (v.T).real
    for l in range(len(eigenvect_hist_gen)):
        eigenvect_hist_gen[l] = eigenvect_hist_gen[l]/la.norm(eigenvect_hist_gen[l])
    
    eigenvect_hist_gen = eigenvect_hist_gen.T
    
    with open('data\pickle\spread.pkl', 'rb') as input:
        spread_list = pickle.load(input)
        col_index = pickle.load(input)
        row_index = pickle.load(input)

    AAA_AA=True
    
    def f_penal(v):
        return function_optim.function_optim(v[0], v[1], v[2], v[3], recovery_rate, eigenvect_hist_gen, eigenval_hist_gen, row_index, col_index, spread_list, AAA_AA ) 
        + 1000 *(v[1] - 0.1)**2 + 1000 * (v[2] - 5 )**2 + 100* (v[3] -0.75)**2
                 
    bdss = [(0.001,None), (0.01, 5), (1,20), (0.01,1)]
    res_penal = minimize(f_penal, x0=[3, 0.1, 5, 0.75], bounds = bdss)
    
    pi_0 = res_penal.x[0]
    alpha =res_penal.x[1]
    mu = res_penal.x[2]
    sigma = res_penal.x[3]
    
    spread = function_optim.function_spread(pi_0, alpha,mu, sigma, recovery_rate, eigenvect_hist_gen, eigenval_hist_gen)    
    
    fig = plt.figure()
    fig.set_size_inches(6,5)
    plt.plot(range(20), np.asarray(spread)[:,0], label = 'spreads AAA')
    plt.plot(range(20), np.asarray(spread)[:,1], label = 'spreads AA')
    plt.plot(range(20), np.asarray(spread)[:,2], label = 'spreads A')
    plt.plot(range(20), np.asarray(spread)[:,3], label = 'spreads BBB')
    plt.plot(range(20), np.asarray(spread)[:,4], label = 'spreads BB')
    plt.plot(range(20), np.asarray(spread)[:,5], label = 'spreads B')
    plt.plot(range(20), np.asarray(spread)[:,6], label = 'spreads CCC')
    plt.xlabel('Time (in years)', fontsize=18)
    plt.ylabel('spreads', fontsize=16)
    plt.legend()
    xw.sheets['Testing'].range('E4').clear_contents()
    xw.sheets['Testing'].pictures.add(fig, name='Spread Calibré', 
                                         left = xw.sheets['Testing'].range('E4').left, top = xw.sheets['Testing'].range('E4').top, update=True)        
def Test_FunctionsOptim():
    test_function_spread()

if __name__ == '__main__':
    unittest.main()
