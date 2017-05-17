import numpy as np
from scipy.linalg import eigh, cholesky
from pylab import plot, show, axis, subplot, xlabel, ylabel, grid
from .generator_correlated_variables import generator_correlated_variables
import matplotlib.pyplot as plt
import xlwings as xw

def test_generator_correlated_variables():
    corr_matrix = np.array([
        [ 1., 0.2, 0.9],
        [ 0.2, 1., 0.5],
        [ 0.9, 0.5, 1.]
        ])
    time_horizon = 1000
    fixed_seed = 1000
    dw = generator_correlated_variables(corr_matrix = corr_matrix, time_horizon = time_horizon, fixed_seed = fixed_seed)

    fig = plt.figure()
    fig.set_size_inches(3,2.5)
    plt.plot(dw[0], dw[1], 'b.')
    plt.xlabel('dw[0]')
    plt.ylabel('dw[1]')
    plt.legend()
    xw.sheets['Testing'].range('G34').clear_contents()
    xw.sheets['Testing'].pictures.add(fig, name='graph221', 
                                         left = xw.sheets['Testing'].range('G34').left, top = xw.sheets['Testing'].range('G34').top, update=True) 
    
    fig = plt.figure()
    fig.set_size_inches(3,2.5)
    plt.plot(dw[0], dw[2], 'b.')
    plt.xlabel('dw[0]')
    plt.ylabel('dw[2]')
    plt.legend()
    xw.sheets['Testing'].range('K34').clear_contents()
    xw.sheets['Testing'].pictures.add(fig, name='graph223', 
                                         left = xw.sheets['Testing'].range('K34').left, top = xw.sheets['Testing'].range('K34').top, update=True) 
    
    fig = plt.figure()
    fig.set_size_inches(3,2.5)
    plt.plot(dw[1], dw[2], 'b.')
    plt.xlabel('dw[1]')
    plt.ylabel('dw[2]')
    plt.legend()
    xw.sheets['Testing'].range('L34').clear_contents()
    xw.sheets['Testing'].pictures.add(fig, name='graph224', 
                                         left = xw.sheets['Testing'].range('L34').left, top = xw.sheets['Testing'].range('L34').top, update=True) 