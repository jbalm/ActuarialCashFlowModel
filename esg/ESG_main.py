## Program packages
from ..asset.Asset_data import Asset_data
from .ESG_RN import ESG_RN
#from .interest_rate import IR_model_classes as IR_model
from .model_equity.GBM_constant_volatility import GBM_constant_volatility 
#from .credit_risk import credit_model_classes as credit_model
from .credit_risk.JLT import JLT
from .interest_rate.Hull_White_one_factor import Hull_White_one_factor
from .interest_rate.IR_model_util import IR_model_util
from ..core_math import excel_toolbox as et
## Python packages
import matplotlib.pyplot as plt
import pickle
import numpy as np
import xlwings as xw
import xlwt


Working_URL = r'Feuille_de_calcul_ALM(Working).xlsm'

def ESG_calibrate(**kwords):
    data = Asset_data()
    data.update(Working_URL)
    mk_name = str(xw.sheets['ESG'].range('D5').value)
    market = data.get_list_market(mk_name)
    # ==============================================
    # Define Equity model
    Equity = GBM_constant_volatility()
    # ==============================================
    # Define Interest Rate model
    Interest_rate = Hull_White_one_factor()
    # ==============================================
    # credit_model
    # ==============================================
    Credit = JLT()
    # Initialise ESG
    ESG = ESG_RN(data, Interest_rate, Equity, Credit)
    ESG.add_corr_matrix(corr_matrix = market.corr_matrix)
    ESG.add_market_name(market_name = 'EUR')
    if bool(kwords):
        for kw in kwords:
            setattr(ESG, kw, kwords[kw])
        ESG.calibrate_models()
        with open(r'data\pickle\modif_ESG.pkl', 'wb') as output:
            pickle.dump(ESG, output, pickle.HIGHEST_PROTOCOL)
    else:
        ESG.calibrate_models()
        with open(r'data\pickle\ESG.pkl', 'wb') as output:
            pickle.dump(ESG, output, pickle.HIGHEST_PROTOCOL)

def ESG_generate_scenarios(modif = False):
    ESG = None
    if modif:
        with open(r'data\pickle\modif_ESG.pkl', 'rb') as input:
            ESG = pickle.load(input)
    else:
        with open(r'data\pickle\ESG.pkl', 'rb') as input:
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
    # Update the variable time_horizon of the objects deriving from the following classes IR_model, EQ_model, credit_model
    ESG.update_time_horizon(time_horizon)
    
    for traj in range(ESG.number_trajectories):
        ESG.get_scenario(traj_i = traj)
    # ==============================================
    # Save ESG
    # ==============================================
    if modif:
        with open(r'data\pickle\modif_ESG_updated.pkl', 'wb') as output:
            pickle.dump(ESG, output, pickle.HIGHEST_PROTOCOL)
    else:
        with open(r'data\pickle\ESG_updated.pkl', 'wb') as output:
            pickle.dump(ESG, output, pickle.HIGHEST_PROTOCOL)
 
@xw.func   
def ESG_generate_EQ_prices():
    ESG = None
    with open(r'data\pickle\ESG_updated.pkl', 'rb') as input:
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
    et.Write2DListtoExcel(file_name = r'data\EQ_Price_Index.xls', obj = resu_price_index)
    et.Write2DListtoExcel(file_name = r'data\EQ_Total_Return.xls', obj = resu_total_return)

@xw.func   
def ESG_generate_short_rates():
    ESG = None
    with open(r'data\pickle\ESG_updated.pkl', 'rb') as input:
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

    et.Write2DListtoExcel(file_name = r'data\Short_rates.xls', obj = short_rates)


def ESG_generate_spread_curve():
    ESG = None
    with open(r'data\pickle\ESG_updated.pkl', 'rb') as input:
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
    with open(r'data\pickle\ESG_updated.pkl', 'rb') as input:
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
    book.save(r'data\Test_martingale_IR.xls')
    
def ESG_test_martingale_EQ():
    ESG = None
    with open(r'data\pickle\ESG_updated.pkl', 'rb') as input:
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
    book.save(r'data\Test_martingale_EQ.xls')    
    
def ESG_generate_deflators():
    ESG = None
    with open(r'data\pickle\ESG_updated.pkl', 'rb') as input:
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
    book.save(r'data\Deflators_to_Tmax.xls')
    
  
def ESG_generate_1y_ZCY():
    ESG = None
    with open(r'data\pickle\ESG_updated.pkl', 'rb') as input:
        ESG = pickle.load(input)
    
    DFdict = []    
    for traj in range(ESG.number_trajectories):
        DFlist = [np.exp(-ESG.scenarios[traj]['Short_rates'][t]) for t in range(1, ESG.time_horizon + 1)]
        DFdict.append(DFlist)
    et.Write2DListtoExcel(file_name = r'data\One_year_Zero_coupon_yields.xls', obj = DFdict)
       

#if __name__ == "__main__":
#    Liabilities_data_test_update()
#    Liabilities_data = None
#    with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\Liabilities_data0.pkl', 'rb') as input:
#        Liabilities_data = pickle.load(input)
#        
#    ESG = None
#    with open(r'C:\Users\FR011526\Documents\ALM_credit(working)\ESG_updated.pkl', 'rb') as input:
#        ESG = pickle.load(input)
#    #deltaBEL, deltaFP = Compute_BEL_sensitivities_alpha()
#    #deltaBEL, deltaFP = Compute_BEL_sensitivities_equity_index()
#    BEL, FP = Compute_BEL()
#    
