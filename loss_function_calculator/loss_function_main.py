# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 22:57:32 2016

@author: Quang Dien DUONG
"""
# Program Python
from ..esg.ESG_main import ESG_calibrate,ESG_generate_scenarios
from ..alm.alm_main import Compute_BEL
from ..core_math import excel_toolbox as et
# Packages Python
import matplotlib.pyplot as plt
import pickle
import numpy as np
import xlwings as xw


Working_URL = r'Feuille_de_calcul_ALM(Working).xlsm'
			
			
@xw.func
def Loss_Function_generate_stressed_scenarios():
    # ===============================================================
    # Get alpha_max/min; beta_max/min; gamma_max/min 
    alpha_max = xw.sheets['Loss Function'].range('B2').value
    alpha_min = xw.sheets['Loss Function'].range('B3').value
    nb_test_alpha = int(xw.sheets['Loss Function'].range('B7').value)
    beta_max = xw.sheets['Loss Function'].range('D2').value
    beta_min = xw.sheets['Loss Function'].range('D3').value
    nb_test_beta = xw.sheets['Loss Function'].range('D7').value
    gamma_max = xw.sheets['Loss Function'].range('F2').value
    gamma_min = xw.sheets['Loss Function'].range('F3').value
    nb_test_gamma = xw.sheets['Loss Function'].range('F7').value
    
    PC1 = xw.sheets['Loss Function'].range('B5').value 
    PC2 = xw.sheets['Loss Function'].range('D5').value
    PC3 = xw.sheets['Loss Function'].range('F5').value
    # ================================================================
    alpha_arr = np.linspace(alpha_min, alpha_max, num = nb_test_alpha)     
    beta_arr = np.linspace(beta_min, beta_max, num = nb_test_beta)
    gamma_arr = np.linspace(gamma_min, gamma_max, num = nb_test_gamma)
    
    alpha_arr = list(alpha_arr)
    beta_arr = list(beta_arr)
    gamma_arr = list(gamma_arr)
    
    alpha_arr.append(PC1)
    beta_arr.append(PC2)
    gamma_arr.append(PC3)
    
    alpha_arr.sort()
    beta_arr.sort()
    gamma_arr.sort()
    
    # =================================================================================================
    # Point out the positions of PC1, PC2, PC3 within respectively alpha_arr, beta_arr
    indexPC1 = np.where(np.asarray(alpha_arr) == PC1)[0][0]
    indexPC2 = np.where(np.asarray(beta_arr) == PC2)[0][0]
    indexPC3 = np.where(np.asarray(gamma_arr) == PC3)[0][0]
    xw.sheets['Loss Function'].range('B6').value = indexPC1
    xw.sheets['Loss Function'].range('D6').value = indexPC2
    xw.sheets['Loss Function'].range('F6').value = indexPC3
    # =================================================================================================
    
    # =================================================================================================
    # get time horizon
    end_time = int(xw.sheets['Smith Wilson extrapolation'].range('C3').value)
    # =================================================================================================
    #SM = None
    #with open(r'C:\Users\FR011526\Documents\ALM_QDD_Interface_Excel\SM_extra_data.pkl', 'rb') as input:
    #    SM = pickle.load(input)
    zcb_curves_alpha = []
    zcb_curves_beta = []
    zcb_curves_gamma = []
    for alpha in alpha_arr:
        with open(r'data\pickle\SM_extra_data.pkl', 'rb') as input:
            SM_alpha = pickle.load(input)
        SM_alpha.swap_rate_lists['Coupon_yields'] = SM_alpha.swap_rate_lists['Coupon_yields'] + (alpha - SM_alpha.PC1) * np.array(SM_alpha.a1)
        SM_alpha.SM_extrapolation(end_time = end_time, output_excel = False)
        zcb_curves_alpha.append(SM_alpha.swap_rate_lists['Spot_Rates'])
    for beta in beta_arr:
        with open(r'data\pickle\SM_extra_data.pkl', 'rb') as input:
            SM_beta = pickle.load(input)
        SM_beta.swap_rate_lists['Coupon_yields'] = SM_beta.swap_rate_lists['Coupon_yields'] + (beta - SM_beta.PC2) * np.array(SM_beta.a2)        
        SM_beta.SM_extrapolation(end_time = end_time, output_excel = False)
        zcb_curves_beta.append(SM_beta.swap_rate_lists['Spot_Rates'])
    for gamma in gamma_arr:
        with open(r'data\pickle\SM_extra_data.pkl', 'rb') as input:
            SM_gamma = pickle.load(input)
        SM_gamma.swap_rate_lists['Coupon_yields'] = SM_gamma.swap_rate_lists['Coupon_yields'] + (gamma - SM_gamma.PC3) * np.array(SM_gamma.a3)
        SM_gamma.SM_extrapolation(end_time = end_time, output_excel = False)
        zcb_curves_gamma.append(SM_gamma.swap_rate_lists['Spot_Rates'])
    with open(r'data\pickle\adjusted_alpha_zcb_curves.pkl', 'wb') as output:
         pickle.dump(zcb_curves_alpha, output, pickle.HIGHEST_PROTOCOL)
    with open(r'data\pickle\adjusted_beta_zcb_curves.pkl', 'wb') as output:
        pickle.dump(zcb_curves_beta, output, pickle.HIGHEST_PROTOCOL)
    with open(r'data\pickle\adjusted_gamma_zcb_curves.pkl', 'wb') as output:
        pickle.dump(zcb_curves_gamma, output, pickle.HIGHEST_PROTOCOL)
    
    
    # =========================================================================
    # Export zcb_curves_alpha.xlsx, zcb_curves_beta.xlsx, zcb_curves_gamma.xlsx
    # =========================================================================
    et.Write2DListtoExcel(file_name = r'data\pickle\zcb_curves_alpha.csv', obj = zcb_curves_alpha)
    et.Write2DListtoExcel(file_name = r'data\pickle\zcb_curves_beta.csv', obj = zcb_curves_beta)
    et.Write2DListtoExcel(file_name = r'data\pickle\zcb_curves_gamma.csv', obj = zcb_curves_gamma)
    # ===========================================================================================================================================
    # get_adjusted_zero_coupon_curves 
    # ===========================================================================================================================================
    fig = plt.figure(figsize = (10,8))
    adj = plt.subplots_adjust(hspace = 0.4, wspace = 0.4)
    
    sp = plt.subplot(2,2,1)
    for zcb_curve in zcb_curves_alpha:
        x = range(1, len(zcb_curve)+1)
        y = zcb_curve
        l1 = plt.plot(x,y, linewidth = 1)
    lx = plt.xlabel("Maturity")
    ly = plt.ylabel("Zero-coupon bond curve")
    tl = plt.title("Choc on the first principal component")
    
    sp = plt.subplot(2,2,2)
    for zcb_curve in zcb_curves_beta:
        x = range(1, len(zcb_curve)+1)
        y = zcb_curve
        l1 = plt.plot(x,y, linewidth = 1)
    lx = plt.xlabel("Maturity")
    ly = plt.ylabel("Zero-coupon bond curve")
    tl = plt.title("Choc on the second principal component")
    
    sp = plt.subplot(2,2,3)
    for zcb_curve in zcb_curves_gamma:
        x = range(1, len(zcb_curve)+1)
        y = zcb_curve
        l1 = plt.plot(x,y, linewidth = 1)
    lx = plt.xlabel("Maturity")
    ly = plt.ylabel("Zero-coupon bond curve")
    tl = plt.title("Choc on the third principal component")
    xw.sheets['Loss Function'].range('I2').clear_contents()
    xw.sheets['Loss Function'].pictures.add(fig,name='Adjusted Zero Coupon Bond Curves',
                                                left=xw.sheets['Loss Function'].range('I2').left, top=xw.sheets['Loss Function'].range('I2').top,
                                                update=True)

@xw.func
def Compute_BEL_sensitivities_alpha():
    nb_test_alpha = int(xw.sheets['Loss Function'].range('B7').value)
    BEL = []
    FondsPropres = []
    for alpha in range(nb_test_alpha+1):
        ESG_calibrate(alpha = alpha)
        ESG_generate_scenarios(modif = True)
        BE,FP = Compute_BEL(balance_sheet = False, modif = True) 
        BEL.append(BE)
        FondsPropres.append(FP)
    BEL0 = xw.sheets['ALM'].range('L2').value
    FP0 = xw.sheets['ALM'].range('T13').value
    BEL = np.asarray(BEL)
    deltaBEL = (BEL - BEL0)/BEL0
    FondsPropres = np.asarray(FondsPropres)
    deltaFP = (FondsPropres - FP0)/FP0
    with open(r'data\pickle\Compute_BEL_sensitivities_alpha.pkl', 'wb') as output:
        pickle.dump(deltaBEL, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(deltaFP, output, pickle.HIGHEST_PROTOCOL)
    return deltaBEL, deltaFP
          
@xw.func
def Display_BEL_sensitivities_alpha():
    deltaBEL = None
    with open(r'data\pickle\Compute_BEL_sensitivities_alpha.pkl', 'rb') as input:
        deltaBEL = pickle.load(input)
        ignored = pickle.load(input)
    # =============================================
    # Configure X axes
    indexPC1 = xw.sheets['Loss Function'].range('B6').value
    nb_test_alpha = int(xw.sheets['Loss Function'].range('B7').value)
    x_axis = np.arange(nb_test_alpha+1) - indexPC1
    
    xw.sheets['Loss Function'].range('C79').value = x_axis
    xw.sheets['Loss Function'].range('C80').value = deltaBEL
    
@xw.func
def Display_FP_sensitivities_alpha():
    deltaFP = None
    with open(r'data\pickle\Compute_BEL_sensitivities_alpha.pkl', 'rb') as input:
        ignored = pickle.load(input)
        deltaFP = pickle.load(input)
    
    xw.sheets['Loss Function'].range('C81').value = deltaFP
    

def Compute_BEL_sensitivities_equity_index():
    level_max = xw.sheets['Loss Function'].range('C15').value
    level_min = xw.sheets['Loss Function'].range('F15').value
    
    nb_tests_max = int(xw.sheets['Loss Function'].range('C17').value)
    nb_tests_min = int(xw.sheets['Loss Function'].range('F17').value)
    
    assert level_max >= 0, "Level max must be positive"
    assert level_min <= 0, "Level min must be negative"

    
    # Merge two arrays and remove duplicates
    l1 = np.linspace(0, level_max, nb_tests_max+1)
    l2 = np.linspace(level_min, 0, nb_tests_min+1)
    
    test_list = list(l2)
    test_list.extend(x for x in l1 if x not in test_list)
    test_list = np.asarray(test_list)
    test_list = np.around(test_list, decimals = 2)

    BEL = []
    FondsPropres = []
    for s in test_list:
        ESG_calibrate(market_equity_shock = s)
        ESG_generate_scenarios(modif = True)
        BE, FP = Compute_BEL(balance_sheet = False, modif = True)
        BEL.append(BE)
        FondsPropres.append(FP)
    BEL0 = xw.sheets['ALM'].range('L2').value
    FP0 = xw.sheets['ALM'].range('T13').value
    BEL = np.asarray(BEL)
    deltaBEL = (BEL - BEL0)/BEL0
    FondsPropres = np.asarray(FondsPropres)
    deltaFP = (FondsPropres - FP0)/FP0
    with open(r'data\pickle\Compute_BEL_sensitivities_equity_index.pkl', 'wb') as output:
        pickle.dump(deltaBEL, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(deltaFP, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_list, output, pickle.HIGHEST_PROTOCOL)
    return deltaBEL, deltaFP, test_list
    
def Display_BEL_sensitivities_equity_index():
    test_list = None
    deltaBEL = None
    with open(r'data\pickle\Compute_BEL_sensitivities_equity_index.pkl', 'rb') as input:
        deltaBEL = pickle.load(input)
        ignored = pickle.load(input)
        test_list = pickle.load(input)
    xw.sheets['Loss Function'].range('C119').value = test_list
    xw.sheets['Loss Function'].range('C120').value = deltaBEL
    
def Display_FP_sensitivities_equity_index():
    deltaFP = None
    with open(r'data\pickle\Compute_BEL_sensitivities_equity_index.pkl', 'rb') as input:
        ignored1 = pickle.load(input)
        deltaFP = pickle.load(input)
        ignored2 = pickle.load(input)
    xw.sheets['Loss Function'].range('C121').value = deltaFP   