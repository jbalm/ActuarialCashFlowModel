# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 22:57:32 2016

@author: Quang Dien DUONG
"""

### Progam packages
from .PCA import pca_swap_rate
from ..core_math import excel_toolbox as et
### Python packages
import matplotlib.pyplot as plt
import pickle
import xlwings as xw



Working_URL = r'Feuille_de_calcul_ALM(Working).xlsm'

@xw.func
def PCA_swap():
    # ====================================================================================
    # VL1, VL2, VL3 correspond respectively to the first, second and third vector loadings
    # ====================================================================================
    eigenvalues, eigenvectors, VL1, VL2, VL3, alpha_min, alpha_max, beta_min, beta_max, gamma_min, gamma_max, maturities = pca_swap_rate(Working_URL)
    with open(r'data\pickle\pca_swap.pkl', 'wb') as output:
        pickle.dump(VL1, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(VL2, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(VL3, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(alpha_min, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(alpha_max, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(beta_min, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(beta_max, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(gamma_min, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(gamma_max, output, pickle.HIGHEST_PROTOCOL)
    et.dump_array(VL1, filename = r'data\pickle\First_vector_loading.csv')
    et.dump_array(VL2, filename = r'data\pickle\Second_vector_loading.csv')
    et.dump_array(VL3, filename = r'data\pickle\Third_vector_loading.csv')
    # ============================================================================================================
    # Affich alpha_max, alpha_min, beta_max, beta_min, gamma_max, gamma_min
    xw.sheets['Swap Rates'].range('M54').value = alpha_max
    xw.sheets['Swap Rates'].range('M56').value = alpha_min
    xw.sheets['Swap Rates'].range('P54').value = beta_max
    xw.sheets['Swap Rates'].range('P56').value = beta_min
    xw.sheets['Swap Rates'].range('S54').value = gamma_max
    xw.sheets['Swap Rates'].range('S56').value = gamma_min
    # ============================================================================================================
    # Plot VL1, VL2, VL3
    # ============================================================================================================
    fig = plt.figure()
    plt.plot(maturities, VL1, label = 'Level')
    plt.plot(maturities, VL2, label = 'Slope')
    plt.plot(maturities, VL3, label = 'Curvature')
    plt.legend()
    xw.sheets['Swap Rates'].range('K8').clear_contents()
    xw.sheets['Swap Rates'].pictures.add(fig, name='Principal Component Analysis of the Swap Curve', 
                                         left = xw.sheets['Swap Rates'].range('K8').left, top = xw.sheets['Swap Rates'].range('K8').top, update=True)