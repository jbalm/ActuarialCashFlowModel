# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 22:57:32 2016

@author: Quang Dien DUONG
"""
from Asset_data0 import Asset_data0
from Liabilities_data0 import Liabilities_data_m, Liabilities_data0
from ALM_v1 import ALM
#from ALM_v2_2 import ALM
from ESG_RN import ESG_RN
from Technical_Provision import Technical_Provision
from Liabilities import Liabilities
from Assets_v1 import Assets
#from Bond import Bond
from Bonds_Portfolio_v1 import Bonds_Portfolio
from Balance_Sheets import Balance_Sheets
from PCA import pca_swap_rate
from Smith_Wilson import Smith_Wilson
import EQ_model_classes as EQ_model
import IR_model_classes as IR_model
import credit_model_classes as credit_model
import matplotlib.pyplot as plt
import pickle
import numpy as np
import excel_toolbox as et
import xlwings as xw
import xlwt
import collections


Working_URL = r'..\Feuille_de_calcul_ALM(Working).xlsm'

@xw.func
def PCA_swap():
    # ====================================================================================
    # VL1, VL2, VL3 correspond respectively to the first, second and third vector loadings
    # ====================================================================================
    eigenvalues, eigenvectors, VL1, VL2, VL3, alpha_min, alpha_max, beta_min, beta_max, gamma_min, gamma_max, maturities = pca_swap_rate(Working_URL)
    with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\pca_swap.pkl', 'wb') as output:
        pickle.dump(VL1, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(VL2, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(VL3, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(alpha_min, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(alpha_max, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(beta_min, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(beta_max, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(gamma_min, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(gamma_max, output, pickle.HIGHEST_PROTOCOL)
    et.dump_array(VL1, filename = r'C:\Users\FR011526\Documents\ALM_credit(working)\First_vector_loading.csv')
    et.dump_array(VL2, filename = r'C:\Users\FR011526\Documents\ALM_credit(working)\Second_vector_loading.csv')
    et.dump_array(VL3, filename = r'C:\Users\FR011526\Documents\ALM_credit(working)\Third_vector_loading.csv')
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
        with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\SM_extra_data.pkl', 'wb') as output:
            pickle.dump(sm, output, pickle.HIGHEST_PROTOCOL)

    

def ESG_calibrate(**kwords):
    data = Asset_data0()
    data.update(Working_URL)
    mk_name = str(xw.sheets['ESG'].range('D5').value)
    market = data.get_list_market(mk_name)
    # ==============================================
    # Define Equity model
    Equity = EQ_model.GBM_constant_volatility()
    # ==============================================
    # Define Interest Rate model
    Interest_rate = IR_model.Hull_White_one_factor()
    # ==============================================
    # credit_model
    # ==============================================
    Credit = credit_model.JLT()
    # Initialise ESG
    ESG = ESG_RN(data, Interest_rate, Equity, Credit)
    ESG.add_corr_matrix(corr_matrix = market.corr_matrix)
    ESG.add_market_name(market_name = 'EUR')
    if bool(kwords):
        for kw in kwords:
            setattr(ESG, kw, kwords[kw])
        ESG.calibrate_models()
        with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\modif_ESG.pkl', 'wb') as output:
            pickle.dump(ESG, output, pickle.HIGHEST_PROTOCOL)
    else:
        ESG.calibrate_models()
        with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\ESG.pkl', 'wb') as output:
            pickle.dump(ESG, output, pickle.HIGHEST_PROTOCOL)

def ESG_generate_scenarios(modif = False):
    ESG = None
    if modif:
        with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\modif_ESG.pkl', 'rb') as input:
            ESG = pickle.load(input)
    else:
        with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\ESG.pkl', 'rb') as input:
            ESG = pickle.load(input)
    # ==============================================
    # update time horizon
    time_horizon = xw.sheets['ESG'].range('D3').value
    time_horizon = int(time_horizon)
    # ==============================================
    # get number of scenarios
    ESG.number_trajectories = int(xw.sheets['ESG'].range('D1').value)
    # ==============================================
    ESG.get_seed()
    ESG.update_time_horizon(time_horizon)
    
    for traj in range(ESG.number_trajectories):
        ESG.get_scenario(traj_i = traj)
    # ==============================================
    # Save ESG
    # ==============================================
    if modif:
        with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\modif_ESG_updated.pkl', 'wb') as output:
            pickle.dump(ESG, output, pickle.HIGHEST_PROTOCOL)
    else:
        with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\ESG_updated.pkl', 'wb') as output:
            pickle.dump(ESG, output, pickle.HIGHEST_PROTOCOL)
 
@xw.func   
def ESG_generate_EQ_prices():
    ESG = None
    with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\ESG_updated.pkl', 'rb') as input:
        ESG = pickle.load(input)
    fig = plt.figure()
    resu_price_index = []
    resu_total_return = []
    for traj in range(ESG.number_trajectories):
        plt.plot(range(ESG.time_horizon), ESG.scenarios[traj]['EQ_prices'])
        resu_price_index.append(ESG.scenarios[traj]['EQ_prices'])
        resu_total_return.append(ESG.scenarios[traj]['EQ_total_returns'])
    plt.legend()
    
    xw.sheets['ESG'].range('F12').clear_contents()
    xw.sheets['ESG'].pictures.add(fig, name='Black Schole',
                                       left=xw.sheets['ESG'].range('F12').left, top=xw.sheets['ESG'].range('F12').top,
                                       width=450, height=325, update=True)
    et.Write2DListtoExcel(file_name = r'C:\Users\FR011526\Documents\ALM_credit(working)\EQ_Price_Index.xls', obj = resu_price_index)
    et.Write2DListtoExcel(file_name = r'C:\Users\FR011526\Documents\ALM_credit(working)\EQ_Total_Return.xls', obj = resu_total_return)

@xw.func   
def ESG_generate_short_rates():
    ESG = None
    with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\ESG_updated.pkl', 'rb') as input:
        ESG = pickle.load(input)
    fig = plt.figure()
    short_rates = []
    for traj in range(ESG.number_trajectories):
        plt.plot(range(ESG.time_horizon+1), ESG.scenarios[traj]['Short_rates'])
        short_rates.append(ESG.scenarios[traj]['Short_rates'])
    plt.plot()
    
    xw.sheets['ESG'].range('S12').clear_contents()
    xw.sheets['ESG'].pictures.add(fig, name='Short rates',
                                       left=xw.sheets['ESG'].range('S12').left, top=xw.sheets['ESG'].range('S12').top,
                                       width= 450, height=325,
                                       update=True)

    et.Write2DListtoExcel(file_name = r'C:\Users\FR011526\Documents\ALM_credit(working)\Short_rates.xls', obj = short_rates)


def ESG_generate_spread_curve():
    ESG = None
    with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\ESG_updated.pkl', 'rb') as input:
        ESG = pickle.load(input)
    fig = plt.figure()
    Tmax = int(xw.sheets['ESG'].range('AG38').value)
    spreads = np.asarray(ESG.scenarios[0]['spreads_spot'][0][:Tmax])
    plt.plot(range(Tmax),spreads.T[0], label='AAA')
    plt.plot(range(Tmax),spreads.T[1], label='AA')
    plt.plot(range(Tmax),spreads.T[2], label='A')
    plt.plot(range(Tmax),spreads.T[3], label='BBB')
    plt.plot(range(Tmax),spreads.T[4], label='BB')
    plt.plot(range(Tmax),spreads.T[5], label='B')
    plt.plot(range(Tmax),spreads.T[6], label='CCC')
    plt.legend(loc = 'best')
    plt.plot()
    
    xw.sheets['ESG'].range('AF12').clear_contents()
    xw.sheets['ESG'].pictures.add(fig, name='Spread Curves',
                                       left=xw.sheets['ESG'].range('AF12').left, top=xw.sheets['ESG'].range('AF12').top,
                                       width= 450, height=325,
                                       update=True)

@xw.func
def ESG_test_martingale_IR():
    ESG = None
    with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\ESG_updated.pkl', 'rb') as input:
        ESG = pickle.load(input)
    epsilon_IR, var, mean = ESG.test_martingale_IR()
    xw.sheets['ESG'].range('T47').clear_contents()
    xw.sheets['ESG'].range('T47').value = epsilon_IR
    mean = np.asarray(mean)
    var = np.asarray(var)
    N = ESG.number_trajectories
    sup = ESG.IR_model.zcb_price[0, :ESG.time_horizon-1] + 1.96 / np.sqrt(N) * var
    inf = ESG.IR_model.zcb_price[0, :ESG.time_horizon-1] - 1.96 / np.sqrt(N) * var
    fig = plt.figure()
    plt.plot(range(1, ESG.time_horizon), sup, label = "Borne supérieure de l'IC à 95%")
    plt.plot(range(1, ESG.time_horizon), inf, label = "Borne inférieure de l'IC à 95%")
    plt.plot(range(1, ESG.time_horizon), mean, label = "Prix Monte Carlo")
    plt.legend()
    xw.sheets['ESG'].range('F56').clear_contents()
    xw.sheets['ESG'].pictures.add(fig, name='Test martingale pour le modèle de taux',
                                       left = xw.sheets['ESG'].range('F56').left, top = xw.sheets['ESG'].range('F56').top,
                                        update=True)
    # ==============================================================    
    # Create and save Excel Workbook    
    # ==============================================================
    book = xlwt.Workbook()
    sheet = book.add_sheet("Main")
    style1 = xlwt.easyxf('font: bold 1, color red;')
    sheet.write(0,0, "BORNE_SUP", style1)
    sheet.write(2,0, "MEAN", style1)
    sheet.write(4,0, "BONRE_INF", style1)
    for col in range(len(mean)):
        sheet.write(1, col, sup[col])
        sheet.write(3, col, mean[col])
        sheet.write(5, col, inf[col])
    book.save(r'C:\Users\FR011526\Documents\ALM_credit(working)\Test_martingale_IR.xls')
    
def ESG_test_martingale_EQ():
    ESG = None
    with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\ESG_updated.pkl', 'rb') as input:
        ESG = pickle.load(input)
    epsilon_EQ, var, mean = ESG.test_martingale_EQ()
    xw.sheets['ESG'].range('T50').clear_contents()
    xw.sheets['ESG'].range('T50').value = epsilon_EQ
    mean = np.asarray(mean)
    var = np.asarray(var)
    N = ESG.number_trajectories
    sup = np.ones(ESG.time_horizon-1) + 1.96 / np.sqrt(N) * var
    inf = np.ones(ESG.time_horizon-1) - 1.96 / np.sqrt(N) * var
    fig = plt.figure()
    plt.plot(range(1, ESG.time_horizon), sup, label = "Borne supérieure de l'IC à 95%")
    plt.plot(range(1, ESG.time_horizon), inf, label = "Borne inférieure de l'IC à 95%")
    plt.plot(range(1, ESG.time_horizon), mean, label = "Moyenne de MC actualisée")
    plt.legend()
    xw.sheets['ESG'].range('W56').clear_contents()
    xw.sheets['ESG'].pictures.add(fig, name='Test martingale pour le modèle des actions',
                                       left = xw.sheets['ESG'].range('W56').left, top = xw.sheets['ESG'].range('W56').top,
                                       update=True)
    #xw.Plot(fig).show('Test martingale pour le modèle des actions', left = xw.sheets['ESG'].range('W56').left, top = xw.sheets['ESG'].range('W56').top)
    # ==============================================================    
    # Create and save Excel Workbook    
    # ==============================================================
    book = xlwt.Workbook()
    sheet = book.add_sheet("Main")
    style1 = xlwt.easyxf('font: bold 1, color red;')
    sheet.write(0,0, "BORNE_SUP", style1)
    sheet.write(2,0, "MEAN", style1)
    sheet.write(4,0, "BONRE_INF", style1)
    for col in range(len(mean)):
        sheet.write(1, col, sup[col])
        sheet.write(3, col, mean[col])
        sheet.write(5, col, inf[col])
    book.save(r'C:\Users\FR011526\Documents\ALM_credit(working)\Test_martingale_EQ.xls')    
    
def ESG_generate_deflators():
    ESG = None
    with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\ESG_updated.pkl', 'rb') as input:
        ESG = pickle.load(input)
    Tmax = int(xw.sheets['ESG'].range('Y38').value)

    # ======================================
    # Create Excel Workbook
    # ======================================    
    book = xlwt.Workbook()
    sheet = book.add_sheet("Main")
    begin = 0
    for traj in range(ESG.number_trajectories):
        for col in range(ESG.time_horizon+1):
            for row in range(Tmax):
                sheet.write(row + begin, col, ESG.scenarios[traj]['Deflators'][col, row])
        begin += Tmax + 1
    book.save(r'C:\Users\FR011526\Documents\ALM_credit(working)\Deflators_to_Tmax.xls')
    
  
def ESG_generate_1y_ZCY():
    ESG = None
    with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\ESG_updated.pkl', 'rb') as input:
        ESG = pickle.load(input)
    
    DFdict = []    
    for traj in range(ESG.number_trajectories):
        DFlist = [np.exp(-ESG.scenarios[traj]['Short_rates'][t]) for t in range(1, ESG.time_horizon + 1)]
        DFdict.append(DFlist)
    et.Write2DListtoExcel(file_name = r'C:\Users\FR011526\Documents\ALM_credit(working)\One_year_Zero_coupon_yields.xls', obj = DFdict)
       
def Liabilities_data_update():
    liabilities_data = Liabilities_data_m()
    liabilities_data.update(Working_URL)
    with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\Liabilities_data1.pkl', 'wb') as output:
        pickle.dump(liabilities_data, output, pickle.HIGHEST_PROTOCOL)
        
def Liabilities_data_test_update():
    liabilities_data = Liabilities_data0()
    liabilities_data.update(Working_URL)
    with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\Liabilities_data0.pkl', 'wb') as output:
        pickle.dump(liabilities_data, output, pickle.HIGHEST_PROTOCOL)

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
        with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\SM_extra_data.pkl', 'rb') as input:
            SM_alpha = pickle.load(input)
        SM_alpha.swap_rate_lists['Coupon_yields'] = SM_alpha.swap_rate_lists['Coupon_yields'] + (alpha - SM_alpha.PC1) * np.array(SM_alpha.a1)
        SM_alpha.SM_extrapolation(end_time = end_time, output_excel = False)
        zcb_curves_alpha.append(SM_alpha.swap_rate_lists['Spot_Rates'])
    for beta in beta_arr:
        with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\SM_extra_data.pkl', 'rb') as input:
            SM_beta = pickle.load(input)
        SM_beta.swap_rate_lists['Coupon_yields'] = SM_beta.swap_rate_lists['Coupon_yields'] + (beta - SM_beta.PC2) * np.array(SM_beta.a2)        
        SM_beta.SM_extrapolation(end_time = end_time, output_excel = False)
        zcb_curves_beta.append(SM_beta.swap_rate_lists['Spot_Rates'])
    for gamma in gamma_arr:
        with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\SM_extra_data.pkl', 'rb') as input:
            SM_gamma = pickle.load(input)
        SM_gamma.swap_rate_lists['Coupon_yields'] = SM_gamma.swap_rate_lists['Coupon_yields'] + (gamma - SM_gamma.PC3) * np.array(SM_gamma.a3)
        SM_gamma.SM_extrapolation(end_time = end_time, output_excel = False)
        zcb_curves_gamma.append(SM_gamma.swap_rate_lists['Spot_Rates'])
    with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\adjusted_alpha_zcb_curves.pkl', 'wb') as output:
         pickle.dump(zcb_curves_alpha, output, pickle.HIGHEST_PROTOCOL)
    with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\adjusted_beta_zcb_curves.pkl', 'wb') as output:
        pickle.dump(zcb_curves_beta, output, pickle.HIGHEST_PROTOCOL)
    with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\adjusted_gamma_zcb_curves.pkl', 'wb') as output:
        pickle.dump(zcb_curves_gamma, output, pickle.HIGHEST_PROTOCOL)
    
    
    # =========================================================================
    # Export zcb_curves_alpha.xlsx, zcb_curves_beta.xlsx, zcb_curves_gamma.xlsx
    # =========================================================================
    et.Write2DListtoExcel(file_name = r'C:\Users\FR011526\Documents\ALM_credit(working)\zcb_curves_alpha.csv', obj = zcb_curves_alpha)
    et.Write2DListtoExcel(file_name = r'C:\Users\FR011526\Documents\ALM_credit(working)\zcb_curves_beta.csv', obj = zcb_curves_beta)
    et.Write2DListtoExcel(file_name = r'C:\Users\FR011526\Documents\ALM_credit(working)\zcb_curves_gamma.csv', obj = zcb_curves_gamma)
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
    with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\Compute_BEL_sensitivities_alpha.pkl', 'wb') as output:
        pickle.dump(deltaBEL, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(deltaFP, output, pickle.HIGHEST_PROTOCOL)
    return deltaBEL, deltaFP
          
@xw.func
def Display_BEL_sensitivities_alpha():
    deltaBEL = None
    with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\Compute_BEL_sensitivities_alpha.pkl', 'rb') as input:
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
    with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\Compute_BEL_sensitivities_alpha.pkl', 'rb') as input:
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
    with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\Compute_BEL_sensitivities_equity_index.pkl', 'wb') as output:
        pickle.dump(deltaBEL, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(deltaFP, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_list, output, pickle.HIGHEST_PROTOCOL)
    return deltaBEL, deltaFP, test_list
    
def Display_BEL_sensitivities_equity_index():
    test_list = None
    deltaBEL = None
    with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\Compute_BEL_sensitivities_equity_index.pkl', 'rb') as input:
        deltaBEL = pickle.load(input)
        ignored = pickle.load(input)
        test_list = pickle.load(input)
    xw.sheets['Loss Function'].range('C119').value = test_list
    xw.sheets['Loss Function'].range('C120').value = deltaBEL
    
def Display_FP_sensitivities_equity_index():
    deltaFP = None
    with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\Compute_BEL_sensitivities_equity_index.pkl', 'rb') as input:
        ignored1 = pickle.load(input)
        deltaFP = pickle.load(input)
        ignored2 = pickle.load(input)
    xw.sheets['Loss Function'].range('C121').value = deltaFP   
    
def Compute_BEL(balance_sheet = True, modif = False):
    ESG = None
    if modif:
        with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\modif_ESG_updated.pkl', 'rb') as input:
            ESG = pickle.load(input)
    else: 
        with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\ESG_updated.pkl', 'rb') as input:
            ESG = pickle.load(input)    
        
    nb_traj = int(xw.sheets['ALM'].range('D1').value)
    time_horizon = int(xw.sheets['ALM'].range('D3').value)
    assert nb_traj <= ESG.number_trajectories, "More economic scenarios generated by ESG are required in order to compute Best-Estimate Liabilities"
    assert time_horizon <= ESG.time_horizon, "ESG projection horizon must be larger than ALM projection horizon"
    
    spot_rates = ESG.asset_data0.get_list_market(ESG.market_name).spot_rates
    #initial_deflator = np.power(1./np.add(1,spot_rates), np.arange(1, len(spot_rates)+1))
    
    own_fund = xw.sheets['ALM'].range('C12').value
    capitalization_reserve = xw.sheets['ALM'].range('C14').value
    PRE = xw.sheets['ALM'].range('C16').value
    profit_sharing_reserve = xw.sheets['ALM'].range('C18').value
    
    
    # =====================================================================================================
    # Test version of Liabilities_data
    with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\Liabilities_data0.pkl', 'rb') as input:
        data = pickle.load(input)
    # =====================================================================================================
    #with open(r'C:\Users\FR011526\Documents\ALM_QDD_Interface_Excel\Liabilities_data1.pkl', 'rb') as input:
    #    data = pickle.load(input)
        
    PM0 = 0
    for mdlp in data.model_points:
        PM0 += mdlp.actual_math_provision
    
    if not modif:
        # Display the 'bilan comptable'
        xw.sheets['ALM'].range('O12').value = own_fund
        xw.sheets['ALM'].range('O13').value = capitalization_reserve
        xw.sheets['ALM'].range('O14').value = PRE
        xw.sheets['ALM'].range('O15').value = profit_sharing_reserve
        xw.sheets['ALM'].range('O20').value = PM0
        xw.sheets['ALM'].range('N18').value = own_fund + capitalization_reserve + PRE + profit_sharing_reserve + PM0
    
    allocation_EQ = xw.sheets['ALM'].range('C30').value
    allocation_bond = xw.sheets['ALM'].range('C32').value
    
    # ===================================================
    # update rating target allocation
    # ===================================================
    target_allocation = np.asarray(xw.sheets['Rating Target Allocation'].range('C3:V9').value)   
    
    
    deterministic = xw.sheets['ALM'].range('D23').value
    assert type(deterministic) == bool, "deterministic must be a boolean TRUE or FALSE"
    
    BEL = []
    Asset0 = own_fund + capitalization_reserve + PRE + profit_sharing_reserve
    for traj in range(nb_traj):
        with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\Liabilities_data0.pkl', 'rb') as input:
            data = pickle.load(input)
        profit_sharing_reserve = profit_sharing_reserve
        technical_provision = Technical_Provision()
        technical_provision.update(profit_sharing_reserve = profit_sharing_reserve)
        # ==========================================
        # Initialize Liabilities
        # ==========================================
        liabilities = Liabilities()
        own_fund = own_fund
        PRE = PRE
        liabilities.update(capitalization_reserve = capitalization_reserve, technical_provision = technical_provision, own_fund = own_fund, PRE = PRE, liabilities_data0 = data)
        # ==========================================
        # Initialize Assets
        # ==========================================
        generator = ESG        
        # Initialize Bond Portfolios
        bonds_portfolio = Bonds_Portfolio(time_horizon=time_horizon, ESG_RN_scenarios_traj_i = ESG.scenarios[traj], target_allocation= target_allocation, init_IR_curve=spot_rates)
        bonds_portfolio.initialize_unit_bonds_book_value()
        bonds_portfolio.initialize_unit_bonds_market_value()
        
        assets = Assets(generator, bonds_portfolio)
        # ================================================================================================================================================
        # By default, the initial profit sharing rate is supposed to be TME_0, so there is no any dynamical surrenders during the first year of projection
        # If the initial profit sharing rate is known previously, then it can be modified within the following section
        # ================================================================================================================================================
        for mdlp in liabilities.liabilities_data0.model_points:
            mdlp.profit_sharing_rate[0] = mdlp.TMG[mdlp.seniority]
        # ==========================================
        # Initialize ALM
        # ==========================================
        alm = ALM(assets, liabilities, time_horizon)
        # ==========================================
        # Set up asset allocation
        # ==========================================
        alm.assets.add_allocation(allocation_EQ = allocation_EQ, allocation_bond = allocation_bond)
        value = alm.liabilities.technical_provision.mathematical_provision[0]
        alm.assets.initialize_EQ_value(value = assets.allocations['EQ']*value)
        alm.assets.bonds.initialize_allocation(amount=assets.allocations['bonds']*value)
        # ===============================
        # Set up deflators
        # ===============================
        alm.assets.deflators = alm.assets.ESG_RN.scenarios[traj]['Deflators']
        # ==============================
        #          Cash
        # ==============================
        alm.assets.add_cash(amount = alm.assets.allocations['cash'] * value)
        # ================================
        # Display 'Actifs en valeur de marché'
        if traj == 0:
            Asset0 += alm.assets.cash[0] + alm.assets.EQ_market_value[0] + alm.assets.get_bond_market_value(0)
            if not modif:
                xw.sheets['ALM'].range('S18').clear_contents()
                xw.sheets['ALM'].range('S18').value = Asset0
        # ================================
        # Initialize Balance Sheet
        # ================================
        BS = Balance_Sheets()
        BS.update(key_lv1 = 'assets', key_lv2 = 'cash', value = alm.assets.cash[0])
        BS.update(key_lv1 = 'assets', key_lv2 = 'equity', value = alm.assets.EQ_book_value[0])
        BS.update(key_lv1 = 'assets', key_lv2 = 'bond', value = alm.assets.get_bond_book_value(valuation_date = 0))
        BS.update(key_lv1 = 'liabilities', key_lv2 = 'own_fund', value = liabilities.own_fund[0])
        BS.update(key_lv1 = 'liabilities', key_lv2 = 'capitalization_reserve', value = liabilities.capitalization_reserve[0])
        BS.update(key_lv1 = 'liabilities', key_lv2 = 'profit_sharing_reserve', value = liabilities.technical_provision.profit_sharing_reserve[0])
        BS.update(key_lv1 = 'liabilities', key_lv2 = 'PRE', value = liabilities.PRE[0])
        BS.update(key_lv1 = 'liabilities', key_lv2 = 'mathematical_provision', value = liabilities.technical_provision.mathematical_provision[0])
        alm.balance_sheets.append(BS)
        # ==================================
        # Main Part
        # Compute BEL
        # ==================================
        resu, resu_own_fund, discount_factor, cash_flow_out, trajectory = alm.compute_BEL_cashflows_scenario(traj_i = traj, deterministic = deterministic)
        BEL.append(resu)
        
    if modif:
        output = np.mean(BEL)
    else:
        xw.sheets['ALM'].range('L2').clear_contents()
        output = np.mean(BEL)
        xw.sheets['ALM'].range('L2').value = output
        xw.sheets['ALM'].range('T20').value = output
        xw.sheets['ALM'].range('T13').value = Asset0 - output
    # ===============================
    # Save Balance Sheets to excel
    # ===============================
    # At the begining of period
    # ===============================
    if balance_sheet:
        cash_0_list = [alm.balance_sheets[time_step].get_value('assets','cash_0') for time_step in range (1, alm.time_horizon)]
        equity_market_0_list = [alm.balance_sheets[time_step].get_value('assets', 'equity_market_value_0') for time_step in range(1, alm.time_horizon)]
        equity_bond_0_list = [alm.balance_sheets[time_step].get_value('assets', 'equity_book_value_0') for time_step in range(1, alm.time_horizon)]
        bond_market_0_list = [alm.balance_sheets[time_step].get_value('assets', 'bond_market_value_0') for time_step in range(1, alm.time_horizon)]
        bond_book_0_list = [alm.balance_sheets[time_step].get_value('assets', 'bond_book_value_0') for time_step in range(1, alm.time_horizon)]
        PMVR_action_0_list = [alm.balance_sheets[time_step].get_value('assets', 'PMVR_action_0') for time_step in range(1, alm.time_horizon)]
        PMVR_obligation_0_list = [alm.balance_sheets[time_step].get_value('assets', 'PMVR_obligation_0') for time_step in range(1, alm.time_horizon)]
        available_wealth_0_list = [alm.balance_sheets[time_step].get_value('assets', 'available_wealth_0') for time_step in range(1, alm.time_horizon)]
        own_fund_0_list = [alm.balance_sheets[time_step].get_value('liabilities', 'own_fund_0') for time_step in range(1, alm.time_horizon)]
        capitalization_reserve_0_list = [alm.balance_sheets[time_step].get_value('liabilities', 'capitalization_reserve_0') for time_step in range(1, alm.time_horizon)]
        PRE_0_list = [alm.balance_sheets[time_step].get_value('liabilities','PRE_0') for time_step in range(1, alm.time_horizon)]
        PPE_0_list = [alm.balance_sheets[time_step].get_value('liabilities', 'profit_sharing_reserve_0') for time_step in range(1, alm.time_horizon)]
        PM_0_list = [alm.balance_sheets[time_step].get_value('liabilities', 'mathematical_provision_0') for time_step in range(1, alm.time_horizon)]
        # ============================================
        # After the bond_reinvestment
        # ============================================
        cash_1_list = [alm.balance_sheets[time_step].get_value('assets','cash_1') for time_step in range (1, alm.time_horizon)]
        equity_market_1_list = [alm.balance_sheets[time_step].get_value('assets', 'equity_market_value_1') for time_step in range(1, alm.time_horizon)]
        equity_bond_1_list = [alm.balance_sheets[time_step].get_value('assets', 'equity_book_value_1') for time_step in range(1, alm.time_horizon)]
        bond_market_1_list = [alm.balance_sheets[time_step].get_value('assets', 'bond_market_value_1') for time_step in range(1, alm.time_horizon)]
        bond_book_1_list = [alm.balance_sheets[time_step].get_value('assets', 'bond_book_value_1') for time_step in range(1, alm.time_horizon)]
        PMVR_action_1_list = [alm.balance_sheets[time_step].get_value('assets', 'PMVR_action_1') for time_step in range(1, alm.time_horizon)]
        PMVR_obligation_1_list = [alm.balance_sheets[time_step].get_value('assets', 'PMVR_obligation_1') for time_step in range(1, alm.time_horizon)]
        available_wealth_1_list = [alm.balance_sheets[time_step].get_value('assets', 'available_wealth_1') for time_step in range(1, alm.time_horizon)]
        own_fund_1_list = [alm.balance_sheets[time_step].get_value('liabilities', 'own_fund_1') for time_step in range(1, alm.time_horizon)]
        capitalization_reserve_1_list = [alm.balance_sheets[time_step].get_value('liabilities', 'capitalization_reserve_1') for time_step in range(1, alm.time_horizon)]
        PPE_1_list = [alm.balance_sheets[time_step].get_value('liabilities', 'profit_sharing_reserve_1') for time_step in range(1, alm.time_horizon)]
        PM_1_list = [alm.balance_sheets[time_step].get_value('liabilities', 'mathematical_provision_1') for time_step in range(1, alm.time_horizon)]
        # ============================================
        # After computing amortizing value
        # ============================================
        cash_2_list = [alm.balance_sheets[time_step].get_value('assets','cash_2') for time_step in range (1, alm.time_horizon)]
        equity_market_2_list = [alm.balance_sheets[time_step].get_value('assets', 'equity_market_value_2') for time_step in range(1, alm.time_horizon)]
        equity_bond_2_list = [alm.balance_sheets[time_step].get_value('assets', 'equity_book_value_2') for time_step in range(1, alm.time_horizon)]
        bond_market_2_list = [alm.balance_sheets[time_step].get_value('assets', 'bond_market_value_2') for time_step in range(1, alm.time_horizon)]
        bond_book_2_list = [alm.balance_sheets[time_step].get_value('assets', 'bond_book_value_2') for time_step in range(1, alm.time_horizon)]
        PMVR_action_2_list = [alm.balance_sheets[time_step].get_value('assets', 'PMVR_action_2') for time_step in range(1, alm.time_horizon)]
        PMVR_obligation_2_list = [alm.balance_sheets[time_step].get_value('assets', 'PMVR_obligation_2') for time_step in range(1, alm.time_horizon)]
        available_wealth_2_list = [alm.balance_sheets[time_step].get_value('assets', 'available_wealth_2') for time_step in range(1, alm.time_horizon)]
        own_fund_2_list = [alm.balance_sheets[time_step].get_value('liabilities', 'own_fund_2') for time_step in range(1, alm.time_horizon)]
        capitalization_reserve_2_list = [alm.balance_sheets[time_step].get_value('liabilities', 'capitalization_reserve_2') for time_step in range(1, alm.time_horizon)]
        PPE_2_list = [alm.balance_sheets[time_step].get_value('liabilities', 'profit_sharing_reserve_2') for time_step in range(1, alm.time_horizon)]
        PM_2_list = [alm.balance_sheets[time_step].get_value('liabilities', 'mathematical_provision_2') for time_step in range(1, alm.time_horizon)]
        # ============================================
        # After paying surrenders out
        # ============================================
        cash_3_list = [alm.balance_sheets[time_step].get_value('assets','cash_3') for time_step in range (1, alm.time_horizon)]
        equity_market_3_list = [alm.balance_sheets[time_step].get_value('assets', 'equity_market_value_3') for time_step in range(1, alm.time_horizon)]
        equity_bond_3_list = [alm.balance_sheets[time_step].get_value('assets', 'equity_book_value_3') for time_step in range(1, alm.time_horizon)]
        bond_market_3_list = [alm.balance_sheets[time_step].get_value('assets', 'bond_market_value_3') for time_step in range(1, alm.time_horizon)]
        bond_book_3_list = [alm.balance_sheets[time_step].get_value('assets', 'bond_book_value_3') for time_step in range(1, alm.time_horizon)]
        PMVR_action_3_list = [alm.balance_sheets[time_step].get_value('assets', 'PMVR_action_3') for time_step in range(1, alm.time_horizon)]
        PMVR_obligation_3_list = [alm.balance_sheets[time_step].get_value('assets', 'PMVR_obligation_3') for time_step in range(1, alm.time_horizon)]
        available_wealth_3_list = [alm.balance_sheets[time_step].get_value('assets', 'available_wealth_3') for time_step in range(1, alm.time_horizon)]
        own_fund_3_list = [alm.balance_sheets[time_step].get_value('liabilities', 'own_fund_3') for time_step in range(1, alm.time_horizon)]
        capitalization_reserve_3_list = [alm.balance_sheets[time_step].get_value('liabilities', 'capitalization_reserve_3') for time_step in range(1, alm.time_horizon)]
        PPE_3_list = [alm.balance_sheets[time_step].get_value('liabilities', 'profit_sharing_reserve_3') for time_step in range(1, alm.time_horizon)]
        PM_3_list = [alm.balance_sheets[time_step].get_value('liabilities', 'mathematical_provision_3') for time_step in range(1, alm.time_horizon)]
        # ============================================
        # After paying mortalities out
        # ============================================
        cash_4_list = [alm.balance_sheets[time_step].get_value('assets','cash_4') for time_step in range (1, alm.time_horizon)]
        equity_market_4_list = [alm.balance_sheets[time_step].get_value('assets', 'equity_market_value_4') for time_step in range(1, alm.time_horizon)]
        equity_bond_4_list = [alm.balance_sheets[time_step].get_value('assets', 'equity_book_value_4') for time_step in range(1, alm.time_horizon)]
        bond_market_4_list = [alm.balance_sheets[time_step].get_value('assets', 'bond_market_value_4') for time_step in range(1, alm.time_horizon)]
        bond_book_4_list = [alm.balance_sheets[time_step].get_value('assets', 'bond_book_value_4') for time_step in range(1, alm.time_horizon)]
        PMVR_action_4_list = [alm.balance_sheets[time_step].get_value('assets', 'PMVR_action_4') for time_step in range(1, alm.time_horizon)]
        PMVR_obligation_4_list = [alm.balance_sheets[time_step].get_value('assets', 'PMVR_obligation_4') for time_step in range(1, alm.time_horizon)]
        available_wealth_4_list = [alm.balance_sheets[time_step].get_value('assets', 'available_wealth_4') for time_step in range(1, alm.time_horizon)]
        own_fund_4_list = [alm.balance_sheets[time_step].get_value('liabilities', 'own_fund_4') for time_step in range(1, alm.time_horizon)]
        capitalization_reserve_4_list = [alm.balance_sheets[time_step].get_value('liabilities', 'capitalization_reserve_4') for time_step in range(1, alm.time_horizon)]
        PPE_4_list = [alm.balance_sheets[time_step].get_value('liabilities', 'profit_sharing_reserve_4') for time_step in range(1, alm.time_horizon)]
        PM_4_list = [alm.balance_sheets[time_step].get_value('liabilities', 'mathematical_provision_4') for time_step in range(1, alm.time_horizon)]
        # ============================================
        # After computing distributed wealth
        # ============================================
        richesse_max_list = [alm.balance_sheets[time_step].get_value('assets','richesse_max') for time_step in range (1, alm.time_horizon)]
        richesse_min_list = [alm.balance_sheets[time_step].get_value('assets','richesse_min') for time_step in range (1, alm.time_horizon)]
        richesse_TMG_list = [alm.balance_sheets[time_step].get_value('assets','richesse_TMG') for time_step in range (1, alm.time_horizon)]
        richesse_voulue_list = [alm.balance_sheets[time_step].get_value('assets','richesse_voulue') for time_step in range (1, alm.time_horizon)]
        abondement_list = [alm.balance_sheets[time_step].get_value('assets','abondement') for time_step in range (1, alm.time_horizon)]
        richesse_distribuee_list = [alm.balance_sheets[time_step].get_value('assets','richesse_distribuee') for time_step in range (1, alm.time_horizon)]
        equity_market_5_list = [alm.balance_sheets[time_step].get_value('assets', 'equity_market_value_5') for time_step in range(1, alm.time_horizon)]
        bond_market_5_list = [alm.balance_sheets[time_step].get_value('assets', 'bond_market_value_5') for time_step in range(1, alm.time_horizon)]
        PMVR_action_5_list = [alm.balance_sheets[time_step].get_value('assets', 'PMVR_action_5') for time_step in range(1, alm.time_horizon)]
        PMVR_obligation_5_list = [alm.balance_sheets[time_step].get_value('assets', 'PMVR_obligation_5') for time_step in range(1, alm.time_horizon)]
        available_wealth_5_list = [alm.balance_sheets[time_step].get_value('assets', 'available_wealth_5') for time_step in range(1, alm.time_horizon)]
        own_fund_5_list = [alm.balance_sheets[time_step].get_value('liabilities', 'own_fund_5') for time_step in range(1, alm.time_horizon)]
        capitalization_reserve_5_list = [alm.balance_sheets[time_step].get_value('liabilities', 'capitalization_reserve_5') for time_step in range(1, alm.time_horizon)]
        PPE_5_list = [alm.balance_sheets[time_step].get_value('liabilities', 'profit_sharing_reserve_5') for time_step in range(1, alm.time_horizon)]
        PM_5_list = [alm.balance_sheets[time_step].get_value('liabilities', 'mathematical_provision_5') for time_step in range(1, alm.time_horizon)]
        # ============================================
        # Traitement en fin de période
        # ============================================
        # Assets
        cash_list = [alm.balance_sheets[time_step].get_value('assets','cash') for time_step in range (1, alm.time_horizon)]
        equity_list = [alm.balance_sheets[time_step].get_value('assets', 'equity') for time_step in range(1, alm.time_horizon)]
        bond_list = [alm.balance_sheets[time_step].get_value('assets', 'bond') for time_step in range(1, alm.time_horizon)]
        # Liabilities
        own_fund_list = [alm.balance_sheets[time_step].get_value('liabilities', 'own_fund') for time_step in range(1, alm.time_horizon)]
        capitalization_reserve_list = [alm.balance_sheets[time_step].get_value('liabilities', 'capitalization_reserve') for time_step in range(1, alm.time_horizon)]
        PPE_list = [alm.balance_sheets[time_step].get_value('liabilities', 'profit_sharing_reserve') for time_step in range(1, alm.time_horizon)]
        PRE_list = [alm.balance_sheets[time_step].get_value('liabilities', 'PRE') for time_step in range(1, alm.time_horizon)]
        PM_list = [alm.balance_sheets[time_step].get_value('liabilities', 'mathematical_provision') for time_step in range(1, alm.time_horizon)]
        # Cash flows    
        prime_list = [alm.balance_sheets[time_step].get_value('cash_flows_in', 'premium') for time_step in range(1, alm.time_horizon)]
    
        surrender_list = [alm.balance_sheets[time_step].get_value('cash_flows_out', 'surrender_value') for time_step in range(1, alm.time_horizon)]
        mortality_list = [alm.balance_sheets[time_step].get_value('cash_flows_out', 'mortality_value') for time_step in range(1, alm.time_horizon)]
    
        revenu_list = [alm.balance_sheets[time_step].get_value('cash_flows_in', 'revenu') for time_step in range(1, alm.time_horizon)]
    
        amortizing_in_list = [alm.balance_sheets[time_step].get_value('cash_flows_in', 'amortizing_value') for time_step in range(1, alm.time_horizon)]
        amortizing_out_list = [alm.balance_sheets[time_step].get_value('cash_flows_out', 'amortizing_value') for time_step in range(1, alm.time_horizon)]
        
        PMVR_in_list = [alm.balance_sheets[time_step].get_value('cash_flows_in', 'PMVR') for time_step in range(1, alm.time_horizon)]
        PMVR_out_list = [alm.balance_sheets[time_step].get_value('cash_flows_out', 'PMVR') for time_step in range(1, alm.time_horizon)]
    
        PMVR_in_rachat_list = [alm.balance_sheets[time_step].get_value('cash_flows_in', 'PMVR_rachat') for time_step in range(1, alm.time_horizon)]
        PMVR_out_rachat_list = [alm.balance_sheets[time_step].get_value('cash_flows_out', 'PMVR_rachat') for time_step in range(1, alm.time_horizon)]
    
        delta_PRE_in_list = [alm.balance_sheets[time_step].get_value('cash_flows_in', 'delta_PRE') for time_step in range(1, alm.time_horizon)]
        delta_PRE_out_list = [alm.balance_sheets[time_step].get_value('cash_flows_out', 'delta_PRE') for time_step in range(1, alm.time_horizon)]
    
        PMVR_in_deces_list = [alm.balance_sheets[time_step].get_value('cash_flows_in', 'PMVR_deces') for time_step in range(1, alm.time_horizon)]
        PMVR_out_deces_list = [alm.balance_sheets[time_step].get_value('cash_flows_out', 'PMVR_deces') for time_step in range(1, alm.time_horizon)]

        Cash_Return_in = [alm.balance_sheets[time_step].get_value('cash_flows_in', 'Cash_Return') for time_step in range(1, alm.time_horizon)]
        Cash_Return_out = [alm.balance_sheets[time_step].get_value('cash_flows_out', 'Cash_Return') for time_step in range(1, alm.time_horizon)]        
        
        PMVR_aft_res_allo_in_list  = [alm.balance_sheets[time_step].get_value('cash_flows_in', 'PMVR_aft_res_allo') for time_step in range(1, alm.time_horizon)]
        PMVR_aft_res_allo_out_list = [alm.balance_sheets[time_step].get_value('cash_flows_out', 'PMVR_aft_res_allo') for time_step in range(1, alm.time_horizon)]
    
        MVR__TF_rachat_list = [alm.balance_sheets[time_step].get_value('cash_flows_out', 'MVR_TF_residuelle_rachat') for time_step in range(1, alm.time_horizon)]
    
        MVR__TF_deces_list = [alm.balance_sheets[time_step].get_value('cash_flows_out', 'MVR_TF_residuelle_deces') for time_step in range(1, alm.time_horizon)]
        
        sum_cash_list = [alm.balance_sheets[time_step].get_cash_flow() for time_step in range(1, alm.time_horizon)]
    
        delta_own_fund_list = [alm.balance_sheets[time_step].get_value('liabilities', 'own_fund')
                               - alm.balance_sheets[time_step].get_value('liabilities', 'own_fund_0') for time_step in range(1, time_horizon)]
        delta_PPE_list = [alm.balance_sheets[time_step].get_value('liabilities', 'profit_sharing_reserve')
                          - alm.balance_sheets[time_step].get_value('liabilities', 'profit_sharing_reserve_0') for time_step in range(1, alm.time_horizon)]
        delta_Kpi_list = [alm.balance_sheets[time_step].get_value('liabilities', 'capitalization_reserve')
                          - alm.balance_sheets[time_step].get_value('liabilities', 'capitalization_reserve_0') for time_step in range(1, alm.time_horizon)]
        delta_PM_list = [alm.balance_sheets[time_step].get_value('liabilities', 'mathematical_provision')
                         - alm.balance_sheets[time_step].get_value('liabilities', 'mathematical_provision_0') for time_step in range(1, alm.time_horizon)]
        delta_PRE_list = [alm.balance_sheets[time_step].get_value('liabilities', 'PRE')
                          - alm.balance_sheets[time_step].get_value('liabilities','PRE_0') for time_step in range(1, alm.time_horizon)]
        delta_liabilities_list = [x + y + w for x,y,w in zip(delta_own_fund_list, delta_PPE_list, delta_PM_list)]

        filename = r'C:\Users\FR011526\Documents\ALM_credit(working)\Balance_Sheet.xls'
        filename_short = r'C:\Users\FR011526\Documents\ALM_credit(working)\Balance_Sheet_short_version.xls'
        # =========================
        # At the begining of period
        # =========================
        assets_0_dict = collections.OrderedDict()
        assets_0_dict['cash_0'] = cash_0_list
        assets_0_dict['equity_market_0'] = equity_market_0_list
        assets_0_dict['equity_book_0'] = equity_bond_0_list
        assets_0_dict['bond_market_0'] = bond_market_0_list
        assets_0_dict['bond_book_0'] = bond_book_0_list
        assets_0_dict['PMVR_action_0'] = PMVR_action_0_list
        assets_0_dict['PMVR_obligation_0'] = PMVR_obligation_0_list
        assets_0_dict['available_wealth_0'] = available_wealth_0_list
        liabilities_0_dict = collections.OrderedDict()
        liabilities_0_dict['own_fund_0'] = own_fund_0_list
        liabilities_0_dict['capitalization_reserve_0'] = capitalization_reserve_0_list
        liabilities_0_dict['PRE_0'] = PRE_0_list
        liabilities_0_dict['profit_sharing_reserve_0'] = PPE_0_list
        liabilities_0_dict['mathematical_provision_0'] = PM_0_list
        # =========================
        # After bond reinvestment
        # =========================
        assets_1_dict = collections.OrderedDict()
        assets_1_dict['cash_1'] = cash_1_list
        assets_1_dict['equity_market_1'] = equity_market_1_list
        assets_1_dict['equity_book_1'] = equity_bond_1_list
        assets_1_dict['bond_market_1'] = bond_market_1_list
        assets_1_dict['bond_book_1'] = bond_book_1_list
        assets_1_dict['PMVR_action_1'] = PMVR_action_1_list
        assets_1_dict['PMVR_obligation_1'] = PMVR_obligation_1_list
        assets_1_dict['available_wealth_1'] = available_wealth_1_list
        liabilities_1_dict = collections.OrderedDict()
        liabilities_1_dict['own_fund_1'] = own_fund_1_list
        liabilities_1_dict['capitalization_reserve_1'] = capitalization_reserve_1_list
        liabilities_1_dict['profit_sharing_reserve_1'] = PPE_1_list
        liabilities_1_dict['mathematical_provision_1'] = PM_1_list
        # ================================
        # After computing amortizing value
        # ================================
        assets_2_dict = collections.OrderedDict()
        assets_2_dict['cash_2'] = cash_2_list
        assets_2_dict['equity_market_2'] = equity_market_2_list
        assets_2_dict['equity_book_2'] = equity_bond_2_list
        assets_2_dict['bond_market_2'] = bond_market_2_list
        assets_2_dict['bond_book_2'] = bond_book_2_list
        assets_2_dict['PMVR_action_2'] = PMVR_action_2_list
        assets_2_dict['PMVR_obligation_2'] = PMVR_obligation_2_list
        assets_2_dict['available_wealth_2'] = available_wealth_2_list
        liabilities_2_dict = collections.OrderedDict()
        liabilities_2_dict['own_fund_2'] = own_fund_2_list
        liabilities_2_dict['capitalization_reserve_2'] = capitalization_reserve_2_list
        liabilities_2_dict['profit_sharing_reserve_2'] = PPE_2_list
        liabilities_2_dict['mathematical_provision_2'] = PM_2_list
        # ================================
        # After paying surrenders out
        # ================================
        assets_3_dict = collections.OrderedDict()
        assets_3_dict['cash_3'] = cash_3_list
        assets_3_dict['equity_market_3'] = equity_market_3_list
        assets_3_dict['equity_book_3'] = equity_bond_3_list
        assets_3_dict['bond_market_3'] = bond_market_3_list
        assets_3_dict['bond_book_3'] = bond_book_3_list
        assets_3_dict['PMVR_action_3'] = PMVR_action_3_list
        assets_3_dict['PMVR_obligation_3'] = PMVR_obligation_3_list
        assets_3_dict['available_wealth_3'] = available_wealth_3_list
        liabilities_3_dict = collections.OrderedDict()
        liabilities_3_dict['own_fund_3'] = own_fund_3_list
        liabilities_3_dict['capitalization_reserve_3'] = capitalization_reserve_3_list
        liabilities_3_dict['profit_sharing_reserve_3'] = PPE_3_list
        liabilities_3_dict['mathematical_provision_3'] = PM_3_list
        # ================================
        # After paying mortalities out
        # ================================
        assets_4_dict = collections.OrderedDict()
        assets_4_dict['cash_4'] = cash_4_list
        assets_4_dict['equity_market_4'] = equity_market_4_list
        assets_4_dict['equity_book_4'] = equity_bond_4_list
        assets_4_dict['bond_market_4'] = bond_market_4_list
        assets_4_dict['bond_book_4'] = bond_book_4_list
        assets_4_dict['PMVR_action_4'] = PMVR_action_4_list
        assets_4_dict['PMVR_obligation_4'] = PMVR_obligation_4_list
        assets_4_dict['available_wealth_4'] = available_wealth_4_list
        liabilities_4_dict = collections.OrderedDict()
        liabilities_4_dict['own_fund_4'] = own_fund_4_list
        liabilities_4_dict['capitalization_reserve_4'] = capitalization_reserve_4_list
        liabilities_4_dict['profit_sharing_reserve_4'] = PPE_4_list
        liabilities_4_dict['mathematical_provision_4'] = PM_4_list
        # ==================================
        # After computing distributed wealth
        # ==================================
        assets_5_dict = collections.OrderedDict()
        assets_5_dict['richesse_min'] = richesse_min_list
        assets_5_dict['richesse_max'] = richesse_max_list
        assets_5_dict['richesse_TMG'] = richesse_TMG_list
        assets_5_dict['richesse_voulue'] = richesse_voulue_list
        assets_5_dict['richesse_distribuee'] = richesse_distribuee_list
        assets_5_dict['equity_market_5'] = equity_market_5_list
        assets_5_dict['bond_market_5'] = bond_market_5_list
        assets_5_dict['PMVR_action_5'] = PMVR_action_5_list
        assets_5_dict['PMVR_obligation_5'] = PMVR_obligation_5_list
        assets_5_dict['available_wealth_5'] = available_wealth_5_list
        assets_5_dict['abondement'] = abondement_list
        liabilities_5_dict = collections.OrderedDict()
        liabilities_5_dict['own_fund_4'] = own_fund_5_list
        liabilities_5_dict['capitalization_reserve_4'] = capitalization_reserve_5_list
        liabilities_5_dict['profit_sharing_reserve_4'] = PPE_5_list
        liabilities_5_dict['mathematical_provision_4'] = PM_5_list
        
        assets_dict = collections.OrderedDict()
        assets_dict['Cash'] = cash_list
        assets_dict['Equity'] = equity_list
        assets_dict['Bond'] = bond_list
        liabilities_dict = collections.OrderedDict()
        liabilities_dict['Own_Fund'] = own_fund_list
        liabilities_dict['Capitalization_Reserve'] = capitalization_reserve_list
        liabilities_dict['Profit_Sharing_Reserve'] = PPE_list
        liabilities_dict['PRE'] = PRE_list
        liabilities_dict['Mathematical_Provision'] = PM_list
        delta_liabilities_dict = collections.OrderedDict()
        delta_liabilities_dict['Delta_Liabilities'] = delta_liabilities_list
        
        cash_in_dict = collections.OrderedDict()
        cash_in_dict['Prime'] = prime_list
        cash_in_dict['Revenu'] = revenu_list
        cash_in_dict['Amortizing_Value'] = amortizing_in_list
        cash_in_dict['PMVR'] = PMVR_in_list
        cash_in_dict['PMVR_rachat'] = PMVR_in_rachat_list
        cash_in_dict['PMVR_deces'] = PMVR_in_deces_list
        cash_in_dict['Cash_Return'] = Cash_Return_in
        cash_in_dict['PMVR_aft_res_allo'] = PMVR_aft_res_allo_in_list
        cash_in_dict['delta_PRE'] = delta_PRE_in_list
        
        cash_out_dict = collections.OrderedDict()
        cash_out_dict['Surrender_Value'] = surrender_list
        cash_out_dict['Mortality_Value'] = mortality_list
        cash_out_dict['Amortizing_Value'] = amortizing_out_list
        cash_out_dict['PMVR'] = PMVR_out_list
        cash_out_dict['PMVR_rachat'] = PMVR_out_rachat_list
        cash_out_dict['PMVR_deces'] = PMVR_out_deces_list
        cash_out_dict['Cash_Return'] = Cash_Return_out
        cash_out_dict['PMVR_aft_res_allo'] = PMVR_aft_res_allo_out_list
        cash_out_dict['MVR_rachat'] = MVR__TF_rachat_list
        cash_out_dict['MVR_deces'] = MVR__TF_deces_list
        cash_out_dict['delta_PRE'] = delta_PRE_out_list
        sum_cash_dict = collections.OrderedDict()
        sum_cash_dict['Sum_of_cash_flows'] = sum_cash_list
        # ======================
        # Build Up Balance Sheet
        # ======================
        balance_sheet = collections.OrderedDict()
        balance_sheet['Assets_0'] = assets_0_dict
        balance_sheet['Liabilities_0'] = liabilities_0_dict
        balance_sheet['Assets_1'] = assets_1_dict
        balance_sheet['Liabilities_1'] = liabilities_1_dict
        balance_sheet['Assets_2'] = assets_2_dict
        balance_sheet['Liabilities_2'] = liabilities_2_dict
        balance_sheet['Assets_3'] = assets_3_dict
        balance_sheet['Liabilities_3'] = liabilities_3_dict
        balance_sheet['Assets_4'] = assets_4_dict
        balance_sheet['Liabilities_4'] = liabilities_4_dict
        balance_sheet['Assets_5'] = assets_5_dict
        balance_sheet['Liabilities_5'] = liabilities_5_dict
        balance_sheet['Assets'] = assets_dict
        balance_sheet['Liabilities'] = liabilities_dict
        balance_sheet['Delta_Liabilities'] = delta_liabilities_dict
        balance_sheet['Cash_Flows_In'] = cash_in_dict
        balance_sheet['Cash_Flows_Out'] = cash_out_dict
        balance_sheet['Sum_of_cash_flows'] = sum_cash_dict
        
        balance_sheet_short = collections.OrderedDict()
        balance_sheet_short['Assets'] = assets_dict
        balance_sheet_short['Liabilities'] = liabilities_dict
        balance_sheet_short['Cash Flows coming in'] = cash_in_dict
        balance_sheet_short['Cash Flows going out'] = cash_out_dict
        
        et.output_excel(file_name = filename, dictionary = balance_sheet)
        et.output_excel(file_name = filename_short, dictionary = balance_sheet_short) 
    
    return output, Asset0-output
        


if __name__ == "__main__":
    Liabilities_data_test_update()
    Liabilities_data = None
    with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\Liabilities_data0.pkl', 'rb') as input:
        Liabilities_data = pickle.load(input)
        
    ESG = None
    with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\ESG_updated.pkl', 'rb') as input:
        ESG = pickle.load(input)
    #deltaBEL, deltaFP = Compute_BEL_sensitivities_alpha()
    #deltaBEL, deltaFP = Compute_BEL_sensitivities_equity_index()
    BEL, FP = Compute_BEL()
    
