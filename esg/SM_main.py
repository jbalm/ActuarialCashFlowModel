# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 22:57:32 2016

@author: Quang Dien DUONG
"""

## Progam packages
from .Smith_Wilson import Smith_Wilson
## Python packages
import matplotlib.pyplot as plt
import pickle
import xlwings as xw

Working_URL = r'Feuille_de_calcul_ALM(Working).xlsm'

@xw.func
def SM_extrapolation(save_data = True):
    sm = Smith_Wilson(file_name = Working_URL)
    # =========================================================
    # get time horizon
    end_time = int(xw.sheets['Smith Wilson extrapolation'].range('C3').value)
    # =========================================================
    # get time step
    s = int(xw.sheets['Smith Wilson extrapolation'].range('C9').value)
    sm.get_time_step(s)
    # =========================================================
    # get UFR
    UFR = xw.sheets['Smith Wilson extrapolation'].range('C5').value/100
    sm.get_UFR(UFR)
    # =========================================================
    # get alpha
    alpha = xw.sheets['Smith Wilson extrapolation'].range('C7').value
    sm.get_alpha(alpha)
    # =========================================================
    sm.SM_extrapolation(end_time = end_time)
    # ============================================================================================================
    # Plot ZCB_Price
    # ============================================================================================================
    fig = plt.figure()
    plt.plot(range(1, len(sm.swap_rate_lists['Spot_Rates'])+1), sm.swap_rate_lists['Spot_Rates'], label = 'Extrapolated Zero Coupon Yield')
    plt.plot(range(1, len(sm.swap_rate_lists['Forward_Rates'])+1), sm.swap_rate_lists['Forward_Rates'], '--', label = 'Extrapolated Forward Rates')
    plt.legend()    
    xw.sheets['Smith Wilson extrapolation'].range('J5').clear_contents()
    xw.sheets['Smith Wilson extrapolation'].pictures.add(fig, name='Extrapolated Zero Coupon Yield',
                                         left=xw.sheets['Swap Rates'].range('J5').left, top=xw.sheets['Swap Rates'].range('J5').top,
                                         update=True)
    # ============================================================================================================
    # Affich PC1, PC2, PC3
    xw.sheets['Loss Function'].range('B5').value = sm.PC1
    xw.sheets['Loss Function'].range('D5').value = sm.PC2
    xw.sheets['Loss Function'].range('F5').value = sm.PC3
    if save_data:
        with open(r'data\pickle\SM_extra_data.pkl', 'wb') as output:
            pickle.dump(sm, output, pickle.HIGHEST_PROTOCOL)