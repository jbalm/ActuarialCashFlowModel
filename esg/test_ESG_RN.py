# Python Programs
from ..asset.Asset_data import Asset_data
from .model_equity.EQ_model_classes import EQ_model_base as EQ_model
from .interest_rate.IR_model_classes import IR_model as IR_model
from .credit_risk.credit_model_classes import credit_model_base as credit_model
from .ESG_RN import ESG_RN
from .ESG_main import *

## Python packages
#import unittest
import xlwings as xw
import matplotlib.pyplot as plt
from xlrd import open_workbook
import pandas as pd

def test_test_martingale_EQ():
    temp_mk_name = xw.sheets['ESG'].range('D5').value 
    temp_time_horizon = xw.sheets['ESG'].range('D3').value 
    temp_number_trajectories = xw.sheets['ESG'].range('D1').value 
    temp_volatilité_EQ = xw.sheets['Market_Environment'].range('C7').value 
    
    xw.sheets['ESG'].range('D5').value = "EUR"
    xw.sheets['ESG'].range('D3').value = 50
    xw.sheets['ESG'].range('D1').value = 1
    xw.sheets['Market_Environment'].range('C7').value = 0
    
    ESG_calibrate()
    ESG_generate_scenarios(modif = False)
    ESG_test_martingale_EQ()
    
    df = pd.read_excel('data\Test_martingale_EQ.xls')
    df = df.as_matrix()
    
    fig = plt.figure()
    fig.set_size_inches(6,5)
    plt.plot(range(1, 50), df[0,], label = "Borne supérieure de l'IC à 95%")
    plt.plot(range(1, 50), df[2,], label = "Borne inférieure de l'IC à 95%")
    plt.plot(range(1, 50), df[4,], label = "Moyenne de MC actualisée")
    plt.xlabel('maturity')
    plt.ylabel('test de martingalité', fontsize=16)
    plt.legend()
   
    # Restore previous values
    xw.sheets['ESG'].range('D5').value = temp_mk_name
    xw.sheets['ESG'].range('D3').value = temp_time_horizon  
    xw.sheets['ESG'].range('D1').value  = temp_number_trajectories
    xw.sheets['Market_Environment'].range('C7').value = temp_volatilité_EQ

    # Plot the graph
    xw.sheets['Testing'].range('E60').clear_contents()
    xw.sheets['Testing'].pictures.add(fig, name='test martingale EQ', 
                                     left = xw.sheets['Testing'].range('E60').left, top = xw.sheets['Testing'].range('E60').top, update=True)  


def test_test_martingale_IR():  
        temp_mk_name = xw.sheets['ESG'].range('D5').value 
        temp_time_horizon = xw.sheets['ESG'].range('D3').value 
        temp_number_trajectories = xw.sheets['ESG'].range('D1').value 
        temp_volatilité_IR = xw.sheets['Market_Environment'].range('C4').value 
        
        xw.sheets['ESG'].range('D5').value = "EUR"
        xw.sheets['ESG'].range('D3').value = 50
        xw.sheets['ESG'].range('D1').value = 1
        xw.sheets['Market_Environment'].range('C4').value = 0
        
        ESG_calibrate()
        ESG_generate_scenarios(modif = False)
        ESG_test_martingale_IR()
        
        df = pd.read_excel('data\Test_martingale_IR.xls')
        df = df.as_matrix()
        
        fig = plt.figure()
        fig.set_size_inches(6,5)
        plt.plot(range(1, 50), df[0,], label = "Borne supérieure de l'IC à 95%")
        plt.plot(range(1, 50), df[2,], label = "Borne inférieure de l'IC à 95%")
        plt.plot(range(1, 50), df[4,], label = "Prix de Monte Carlo")
        plt.xlabel('maturity')
        plt.ylabel('test de martingalité', fontsize=16)
        plt.legend()
   
        # Restore previous values
        xw.sheets['ESG'].range('D5').value = temp_mk_name
        xw.sheets['ESG'].range('D3').value = temp_time_horizon  
        xw.sheets['ESG'].range('D1').value  = temp_number_trajectories
        xw.sheets['Market_Environment'].range('C4').value = temp_volatilité_IR

        # Plot the graph
        xw.sheets['Testing'].range('E32').clear_contents()
        xw.sheets['Testing'].pictures.add(fig, name='test martingale IR', 
                                         left = xw.sheets['Testing'].range('E32').left, top = xw.sheets['Testing'].range('E32').top, update=True)    
    
    
