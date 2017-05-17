# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 15:39:51 2016

@author: Quang Dien DUONG-Jun Jing
"""
## Progam packages
#from ..esg.ESG_RN import ESG_RN
#from .Asset_data import Asset_data
#from .bond.Bonds_Portfolio import Bonds_Portfolio
#from ..esg.model_equity import EQ_model_classes as EQ_model
#from ..esg.interest_rate import IR_model_classes as IR_model
#from ..esg.credit_risk import credit_model_classes as credit_model
## Python packages
import numpy as np

class Assets(object):
    """
        This class is meant to:
        1. Compute asset values across time steps and scenarios,
        2. Record the asset allocation across time steps and scenarios (via the ALM.py class)

        Attributes:
        ===========
        Input:

        1. ESG_RN:
            Type: instance of ESG_RN class
            Function: Economic scenario generator.

        2. allocation:
            Type: list
            Function: asset allocation.

        3. number_EQ: (default parameter)
            Type: float (positive only)
            Function: number of equity shares.

        4. EQ_initial_price:
            Type: float (positive only)
            Function: initial equity purchase price.

        5. deflators:
            Type: 2 dimensional array
            Function: deflators(valuation_date, maturity)

        Output:

        1. target_rates:
            Type: array
            Function: corresponding list of asset target rate over time horizon.

        2. asset_income:
            Type: dictionary
            Function: corresponding list of all the possible asset incomes values over time horizon (see EQ_model_classes.py for more details).

        3. cash:
            Type: array
            Function: corresponding list of cash instrument values over time horizon

        Methods:
        ========

        1. initialize_EQ_value

        *2. add_bond

        3. get_target_bond_rate

        4. get_target_rate

        *5. add_asset_income

        6. get_asset_income_and_EQ_value

        *7. get_bond_book_value

        *8. get_bond_market_value

        9. reset_asset_income_and_EQ_value

        *10. add_allocation

        11. get_allocation

        12. add_EQ

        13. delete_EQ

        14. add_cash

        15. execute_unrealised_EQ_gain

        16. execute_unrealised_EQ_loss
    """

    def __init__(self, ESG_RN, Bonds_Portfolio):
        # N'oublie pas la dépendance entre les deux arguments dans la suite
        # à réfléchir sur cette dépendance
    
        self.ESG_RN = ESG_RN
        self.allocations = {}
        
        self.target_rates = [0]
        self.asset_income = [0]
        self.number_EQ = 1
        self.EQ_initial_price = None
        self.deflators = None
        self.cash = []
        
        self.bonds = Bonds_Portfolio
        # bonds_portfolio mis en argument est déjà initialisé
        
        
    def initialize_EQ_value(self, value):
        """
            Method: initialize_EQ_value

            Function: initialize the equity values (initial equity purchase price, equity's initial book value and equity's initial market value)

            Parameter:
                1. value
                    Type: float (positive only)
                    Function: initial equity purchase price.
        """
        self.EQ_initial_price = value
        self.EQ_book_value = [self.number_EQ * value]
        self.EQ_market_value = [self.number_EQ * (1 + self.ESG_RN.market_equity_shock) * value]
        self.EQ_total_returns = [self.number_EQ * (1 + self.ESG_RN.market_equity_shock) * value]
        PMVL0 = self.ESG_RN.market_equity_shock * value
        PVL0 = max(0, PMVL0)
        MVL0 = max(0, -PMVL0)
        PVL_hors_obligation_0 = PVL0
        self.asset_income[0] = {'PMVL': PMVL0, 'PVL': PVL0, 'MVL': MVL0, 'Revenu': 0, 'PVL_obligation_TF': 0, 'MVL_obligation_TF': 0, 'PMVR_hors_obligation': 0, 'PMVR_obligation_TF': 0, 'PVL_hors_obligation': PVL_hors_obligation_0}

    

        
        # Determiner le taux de rendement cible de la poche obligatoire du concurrent à chaque pas temps
    def get_target_bond_rate(self, traj_i, valuation_date):
        """
            Method: get_target_bond_rate

            Function: return the target return rate on bond evaluated at "valuation_date" which corresponds to the i-th economic scenario.

            Parameters:
                1. traj_i:
                    Type: int
                    Function: the i-th economic scenario
                2. valuation_date:
                    Type: int
                    Function: valuation date
        """
        self.target_bond_rate = max(self.ESG_RN.scenarios[traj_i]['IR_curves'][valuation_date,:10])
        return self.target_bond_rate

    # Determiner le taux cible à chaque pas temps
    def get_target_rate(self, traj_i, valuation_date, conc_margin = 0.):
        """
            Method: get_target_rate

            Function: return the asset portfolio target return rate evaluated at "valuation_date" which corresponds to the i-th economic scenario.

            Parameters:
                1. traj_i:
                    Type: int
                    Function: the i-th economic scenario
                2. valuation_date:
                    Type: int
                    Function: valuation date
                3. conc_margin:
                    Type: float (positive only)
                    Function: contractual margin.
        """
        # Compute target_bond_rate
        target_bond_rate = self.get_target_bond_rate(traj_i, valuation_date)
        # Compute mix_conc_rate
        EQ_return_rate = self.ESG_RN.scenarios[traj_i]['EQ_return_rates'][valuation_date]
        Cash_return_rate = self.ESG_RN.scenarios[traj_i]['IR_curves'][valuation_date,0] # Taux de rendement 1 ans 
        mix_conc_rate = 0.2 * EQ_return_rate + 0.7 *target_bond_rate + 0.1 * Cash_return_rate
        
        # Compute target_rate
        return max(mix_conc_rate, target_bond_rate) - conc_margin

    # Gérer le revenu du portefeuille à chaque pas temps
    def add_asset_income(self, list_income, valuation_date, key = None):
        """
            Method: add_asset_income

            Function: update the asset income list given a new instrument income list (e.g. bond income list, real estate income list, etc).

            Parameters:
                1. list_income:
                    Type: dictionary
                    Function: corresponding income list of a new instrument.
                2. valuation_date:
                    Type: int
                    Function: valuation date
                3. key: (this is a default parameter which is set to None)
                    Type: string
                    Function: a specific type of income which must be one of the nine following keys:
                        PVL    : (plus value latente) unrealised gains
                        MVL    : (moins value latente) unrealised losses
                        Revenu : (revenus dégagés par les actifs)
                        PVL_obligation_TF : unrealised gains of the fixed rate bonds
                        PVL_hors_obligation : unrealised losses out of the fixed rate bonds
                        MVL_obligation_TF : unrealised losses of the fixed rate bonds
                        PMVR_hors_obligation : realised gains and losses out of fixed rate bonds
                        PMVR_obligation_TF : realised gains and losses of the fixed rate bonds
        """
        if key is not None:
            self.asset_income[valuation_date][key] += list_income[key]
        else:
            self.asset_income[valuation_date]['Revenu'] += list_income['Revenu']
            self.asset_income[valuation_date]['PMVL'] += list_income['PMVL']
            self.asset_income[valuation_date]['PVL'] += list_income['PVL']
            self.asset_income[valuation_date]['MVL'] += list_income['MVL']
            self.asset_income[valuation_date]['PVL_obligation_TF'] += list_income['PVL_obligation_TF']
            self.asset_income[valuation_date]['MVL_obligation_TF'] += list_income['MVL_obligation_TF']
            self.asset_income[valuation_date]['PVL_hors_obligation'] += list_income['PVL_hors_obligation']
            self.asset_income[valuation_date]['PMVR_hors_obligation'] += list_income['PMVR_hors_obligation']
            self.asset_income[valuation_date]['PMVR_obligation_TF'] += list_income['PMVR_obligation_TF']

    def get_asset_income_and_EQ_value(self, traj_i, valuation_date):
        """
            Method: get_asset_income_and_EQ_value

            Function: get asset income list and equity market value at "valuation_date" which correspond to the i-th economic scenario.

            Parameters:
                1. traj_i:
                    Type: int
                    Function: the i-th economic scenario
                2. valuation_date:
                    Type: int
                    Function: valuation date
        """
        #=============================================
        # The key in asset_income list:
        #    1. Revenu
        #    2. PMVL
        #    3. PVL
        #    4. MVL
        #    5. PVL_hors_obligation
        #    6. PVL_obligation_TF
        #    7. MVL_obligation_TF
        #    8. PMVR_hors_obligation
        #    9. PMVR_obligation_TF
        #==============================================================================================
        self.EQ_total_returns.append(self.ESG_RN.scenarios[traj_i]['EQ_total_returns'][valuation_date])
        self.EQ_market_value.append(self.ESG_RN.scenarios[traj_i]['EQ_prices'][valuation_date])
        #==============================================================================================

    def get_bond_market_value(self, valuation_date):
        """
            Method: get_bond_market_value

            Function: return the total market value of the bond instruments evaluated at "valuation_date"

            Parameter:
                1. valuation_date
                    Type: int
                    Function: valuation_date
        """
        market_value_list = self.bonds.get_market_value_list(valuation_date)
        market_value = np.sum(np.asarray(market_value_list))
        
        return market_value
        
        
    def get_bond_book_value(self, valuation_date):
        """
            Method: get_bond_book_value

            Function: return the total book value of the bond instruments evaluated at "valuation_date"

            Parameter:
                1. valuation_date
                    Type: int
                    Function: valuation_date
        """
        book_value_list = self.bonds.get_book_value_list(valuation_date)
        book_value = np.sum(np.asarray(book_value_list))
        
        return book_value
        
        
    def reset_asset_income_and_EQ_value(self, valuation_date):
        """
            Method: reset_asset_income_and_EQ_value

            Function: update asset income list at the beginning of each time steps

            Parameter:
                1. valuation_date:
                    Type: int
                    Function: valuation_date
        """
        self.EQ_book_value.append(self.EQ_book_value[-1])
        self.EQ_market_value[valuation_date] = self.number_EQ * self.EQ_initial_price * self.EQ_market_value[valuation_date]
        self.EQ_total_returns[valuation_date] = self.number_EQ * self.EQ_initial_price * self.EQ_total_returns[valuation_date]

    def add_allocation(self, allocation_EQ, allocation_bond):
        """
            Method: add_allocation

            Function: initialise asset allocation

            Parameters:
                1. allocation_EQ:
                    Type: float (between 0 and 1)
                    Function: stock allocation
                2. allocation_bond:
                    Type: float (between 0 and 1)
                    Function: bond allocation
        """
        self.allocations['EQ'] = allocation_EQ
        self.allocations['bonds'] = allocation_bond 
        self.allocations['cash'] = 1 - allocation_EQ - allocation_bond
    
    

    def add_EQ(self, amount, traj_i, valuation_date):
        """
            Method: add_EQ

            Function: invest a quantity of money in the stock market and update its market value and book value as well as the current number of equity shares.

            Parameters:
                1. amount:
                    Type: float (positive only)
                    Function: the quantity of money invested in the stock market.
        """
        if self.number_EQ == 0 or self.EQ_market_value[valuation_date]== 0:
            self.number_EQ = 1
            self.EQ_book_value[valuation_date] = amount
            self.EQ_market_value[valuation_date] = amount
            self.EQ_initial_price = self.EQ_market_value[valuation_date]/(self.ESG_RN.scenarios[traj_i]['EQ_prices'][valuation_date])
        else:
            self.number_EQ = (1 + amount/self.EQ_market_value[valuation_date]) * self.number_EQ
            self.EQ_market_value[-1] += amount
            self.EQ_book_value[-1] += amount
        
        
        

    
    def delete_EQ(self, amount, valuation_date, book_value = False):
        """
            Method: delete_EQ

            Function: divest a quantity of money in the stock market and update its market value and book value as well as the current number of equity shares.

            Parameters:
                1. amount:
                    Type: float (positive only)
                    Function: the quantity of money divested in the stock market.
                2. valuation_date
                    Type: int
                    Function: valuation date
                3. book_value: (optional parameter)
                    Type: boolean
                    Function: True if we want to decrease the book value of equity by a certain amount.
        """
        
        if book_value:
            if self.EQ_book_value[valuation_date] == 0:
                if self.number_EQ !=0:
                    raise ValueError('number of equity non updated! ')
            else:
                EQ_PMVL = self.EQ_market_value[valuation_date] - self.EQ_book_value[valuation_date]
                self.number_EQ = (1.0 - amount/self.EQ_book_value[valuation_date]) * self.number_EQ
                self.asset_income[valuation_date]['PMVL'] -= (amount/self.EQ_book_value[valuation_date])*EQ_PMVL
                self.asset_income[valuation_date]['PMVR_hors_obligation'] += (amount/self.EQ_book_value[valuation_date])*EQ_PMVL
                if EQ_PMVL >= 0:
                    self.asset_income[valuation_date]['PVL'] -= (amount/self.EQ_book_value[valuation_date])*EQ_PMVL
                    self.asset_income[valuation_date]['PVL_hors_obligation'] -= (amount/self.EQ_book_value[valuation_date])*EQ_PMVL
                else:
                    self.asset_income[valuation_date]['MVL'] -= (amount/self.EQ_book_value[valuation_date])*np.absolute(EQ_PMVL)
                self.EQ_market_value[valuation_date] = self.EQ_market_value[valuation_date] * (1.0 - amount/self.EQ_book_value[valuation_date])
                self.EQ_book_value[valuation_date] -= amount
        
        else:
            if self.EQ_market_value[valuation_date] == 0:
                if self.number_EQ != 0:
                    raise ValueError('number of equity non updated!')
            else:
                EQ_PMVL = self.EQ_market_value[valuation_date] - self.EQ_book_value[valuation_date]
                self.number_EQ = (1.0 - amount/self.EQ_market_value[valuation_date]) * self.number_EQ
                self.asset_income[valuation_date]['PMVL'] -= (amount/self.EQ_market_value[valuation_date])*EQ_PMVL
                self.asset_income[valuation_date]['PMVR_hors_obligation'] += (amount/self.EQ_market_value[valuation_date])*EQ_PMVL
                if EQ_PMVL >= 0:
                    self.asset_income[valuation_date]['PVL'] -= (amount/self.EQ_market_value[valuation_date])*EQ_PMVL
                    self.asset_income[valuation_date]['PVL_hors_obligation'] -= (amount/self.EQ_market_value[valuation_date])*EQ_PMVL
                else:
                    self.asset_income[valuation_date]['MVL'] -= (amount/self.EQ_market_value[valuation_date])*np.absolute(EQ_PMVL)
                self.EQ_book_value[valuation_date] = self.EQ_book_value[valuation_date] * (1.0 - amount/(self.EQ_market_value[valuation_date]))
                self.EQ_market_value[valuation_date] -= amount


    def add_bonds(self, amount, valuation_date):
        """
            Method: add_bonds
            
            Function: to invest a quantity of money in bonds at valuation date
            
            Parameters:
                1. amount
                    Type: float
                    Function: the amount of money to invest in bonds
                    
                2. valuation_date
                    Type: int
                    Function: valuation_date
        """
        self.bonds.add_bonds(amount, valuation_date)
                       
        
        
    def delete_bonds(self, amount, valuation_date, book_value = True):

        """
            Method: delete_bonds
            
            Function: to disinvest a quantity of money in bonds at valuation date
            
            Parameters:
                1. amount
                    Type: float
                    Function: the amount of money to disinvest in bonds
                    
                2. valuation_date
                    Type: int
                    Function: valuation_date
                    
                3. book_value
                    Type: boolean
                    Function: if True, the amount corresponds to the book value that we want to disinvest; otherwise, it is in market value
                    
        """
        # Return PMVR_obligation_TF back to 0
        self.asset_income[valuation_date]['PMVR_obligation_TF'] = 0
                     
        
        bonds_book_value = self.get_bond_book_value(valuation_date)        
        bonds_market_value = self.get_bond_market_value(valuation_date)
        
              
        
        if book_value:
            if bonds_book_value == 0:
                pass
            
            else:
                PMVR = self.bonds.delete_bonds(amount,  valuation_date, book_value)
                self.asset_income[valuation_date]['PMVR_obligation_TF'] += PMVR
        else:
            if bonds_market_value == 0:
                pass
            
            else:
                PMVR = self.bonds.delete_bonds(amount, valuation_date, book_value)
                self.asset_income[valuation_date]['PMVR_obligation_TF'] += PMVR
                
        # on modifie que PMVR_obligation_TF
            
            
        
        
    def delete_asset(self, amount, valuation_date, book_value = True):
        """
            Method: delete_asset
            
            Function: to disinvest a quantity of money in assets at valuation date
            
            Parameters:
                1. amount
                    Type: float
                    Function: the amount of money to disinvest in assets
                    
                2. valuation_date
                    Type: int
                    Function: valuation_date
                    
                3. book_value
                    Type: boolean
                    Function: if True, the amount corresponds to the book value that we want to disinvest; otherwise, it is in market value
                    
        """
        
        if self.cash[valuation_date] >= amount:
            self.cash[valuation_date] -= amount
        else:
            amount2 = amount - self.cash[valuation_date]
            self.cash[valuation_date] = 0
            if book_value:
                if self.EQ_book_value[valuation_date] >= amount2:
                    self.delete_EQ(amount = amount2, valuation_date = valuation_date, book_value = book_value)
                else:
                    amount3 = amount2 - self.EQ_book_value[valuation_date]
                    self.delete_EQ(amount = self.EQ_book_value[valuation_date], valuation_date = valuation_date, book_value = book_value)
                    self.delete_bonds(amount = amount3, valuation_date = valuation_date, book_value = book_value)
            else:
                if self.EQ_market_value[valuation_date] >= amount2:
                    self.delete_EQ(amount = amount2, valuation_date = valuation_date, book_value = book_value)
                else:
                    amount3 = amount2 - self.EQ_market_value[valuation_date]
                    self.delete_EQ(amount = self.EQ_market_value[valuation_date], valuation_date = valuation_date, book_value = book_value)
                    self.delete_bonds(amount = amount3, valuation_date = valuation_date, book_value = book_value)
        
        
        if self.EQ_market_value[valuation_date] < 0:
            raise ValueError("Equity market price cannot be negative !")
        if self.get_bond_market_value(valuation_date) < 0:
            raise ValueError("Bonds market price cannot be negative !")


    def add_cash(self, amount):
        """
            Method: add_cash

            Function: update cash in the asset portfolio

            Parameter:
                1. amount:
                    Type: float (positive only)
                    Function: corresponding cash that we want to add or update.
        """
        self.cash.append(amount)

    



#if __name__ == '__main__':
#    # =============================
#    # Correlated Matrix
#    # =============================
#    corr_matrix = np.array([
#        [1., 0.22,0.1],
#        [0.22, 1., 0.05],
#        [0.1, 0.05,1.]
#        ])
#
#    # =============================
#    # Number of instruments
#    # =============================
#    num_instrument = 1
#    #===================
#    # Update asset_data
#    #===================
#    path = 'Market_Environment.xls'
#    data = Asset_data()
#    data.update(path)
#    #=========
#    # EQ_model
#    #=========
#    #Equity = EQ_model.EQ_model_const()
#    #Equity = EQ_model.BS_constant_volatility()
#    Equity = EQ_model.GBM_constant_volatility()
#    #=========
#    # IR_model
#    #=========
#    #Interest_rate = IR_model.IR_model_const()
#    Interest_rate = IR_model.Hull_White_one_factor()
#    # =============================================
#    # 
#    credit = credit_model.JLT()
#    #=============
#    # time_horizon
#    #=============
#    time_horizon = 40
#    #====================
#    # number_trajectories
#    #====================
#    number_trajectories = 1
#    
#    market_equity_shock = 0.
#    # ==========================
#    ESG = ESG_RN(data,time_horizon,number_trajectories,Interest_rate,Equity,credit)
#    ESG.add_corr_matrix(corr_matrix = corr_matrix)
#    ESG.add_num_instrument(num_instrument = num_instrument)
#    ESG.add_market_name(market_name = 'EUR')
#    ESG.calibrate_models()
#    #==================================================
#    # This step aims to generate the economic scenarios
#    #==================================================
#    
#    fixed_seed = [1664 + i for i in range(number_trajectories)]
#    
#    for traj in range(number_trajectories):
#        ESG.update_seed(fixed_seed = fixed_seed[traj])
#        ESG.get_scenario(traj_i = traj, market_equity_shock = market_equity_shock)
#    
#    bonds_target_allocation = np.asarray([
#        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,6,7,8,9,10],
#        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,7,8,9],
#        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,7,8],
#        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,7],
#        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]        
#        ])
#
#    bonds_portfolio = Bonds_Portfolio( time_horizon = time_horizon, ESG_RN_scenarios_traj_i = ESG.scenarios[0], target_allocation= bonds_target_allocation)
#    bonds_portfolio.initialize_allocation(amount=10000)
#    bonds_portfolio.initialize_unit_bonds_book_value()
#    bonds_portfolio.initialize_unit_bonds_market_value()
#    
#    Portfolio = Assets(ESG, bonds_portfolio)
#    Portfolio.initialize_EQ_value(value = 100)
#    #for time_step in range(1,time_horizon):
#        #Portfolio.get_asset_income_and_EQ_value(traj_i = 0, valuation_date = time_step)
#        #Portfolio.reset_asset_income_and_EQ_value(valuation_date = time_step)
#        #PMVL = Portfolio.EQ_market_value[time_step] - Portfolio.EQ_book_value[time_step]
#        #PVL = max(0, PMVL)
#        #MVL = max(0, -PMVL)
#        #PVL_hors_obligation = PVL
#        #Revenu = Portfolio.ESG_RN.EQ_model.dividend_rate * Portfolio.EQ_market_value[time_step - 1]
#        #income_dict = {'PMVL': PMVL, 'PVL': PVL, 'MVL': MVL, 'Revenu': Revenu, 'PVL_obligation_TF': 0, 'MVL_obligation_TF': 0, 'PMVR_hors_obligation': 0, 'PMVR_obligation_TF': 0, 'PVL_hors_obligation': PVL_hors_obligation}
#        #Portfolio.asset_income.append(income_dict)
#        # tester les méthodes codées:
#        #Portfolio.add_bonds(amount=100 ,valuation_date = time_step)
#        #Portfolio.delete_bonds(amount=50, valuation_date = time_step)
#        #Portfolio.add_cash(amount=100)
#        # test:
#        #delta_num_bond = Portfolio.bonds.get_num_of_bond(amount = 1, purchase_date = time_step, rating = 1, TtM = 10)
#        #print(delta_num_bond)   
#    
#    print('=====================================================================')
#    print('=================   EXEMPLE OF BONDS_PORTFOLIO   ====================')
#    print('=====================================================================')
#    print('Year | Num of bonds |  Book Vakue | Market Value |  Coupons  |  PMVL | PMVR |')
#    print('-----------------------------------------------------------------------------')
#    pmvr = 0
#    mv_0 = sum(sum(Portfolio.bonds.get_market_value_list(0)))
#    bv_0 = sum(sum(Portfolio.bonds.get_book_value_list(0)))
#    pmvl = mv_0 - bv_0
#    print('{0:.2f}    | {1:.2f}  |  {2:.2f}  |   {3:.2f}   | {4:.2f}  | {5:.2f}  | {6:.2f} |'.format(0, sum(sum(sum(np.asarray(Portfolio.bonds.num_of_bonds)))), sum(sum(Portfolio.bonds.get_book_value_list(0))),sum(sum(Portfolio.bonds.get_market_value_list(0))), 
#              sum(sum(np.asarray(Portfolio.bonds.get_coupon_list(0)))) , pmvl, pmvr))
#
#    for time_step in range(1,time_horizon):
#        Portfolio.get_asset_income_and_EQ_value(traj_i = 0, valuation_date = time_step)
#        Portfolio.reset_asset_income_and_EQ_value(valuation_date = time_step)
#        PMVL = Portfolio.EQ_market_value[time_step] - Portfolio.EQ_book_value[time_step]
#        PVL = max(0, PMVL)
#        MVL = max(0, -PMVL)
#        PVL_hors_obligation = PVL
#        Revenu = Portfolio.ESG_RN.EQ_model.dividend_rate * Portfolio.EQ_market_value[time_step - 1]
#        income_dict = {'PMVL': PMVL, 'PVL': PVL, 'MVL': MVL, 'Revenu': Revenu, 'PVL_obligation_TF': 0, 'MVL_obligation_TF': 0, 'PMVR_hors_obligation': 0, 'PMVR_obligation_TF': 0, 'PVL_hors_obligation': PVL_hors_obligation}
#        Portfolio.asset_income.append(income_dict)
#        # ===========================================================================
#        # Tester bond_portfolio
#        # ===========================================================================
#        pmvr =0
#        # ===============================================================================================================
#        # on ajoute une dimension au num_of_bonds et allocation_matrix si on appelle cette méthode en début de la période
#        # ===============================================================================================================
#        if len(Portfolio.bonds.allocation_matrix) == time_step:
#            Portfolio.bonds.allocation_matrix.append(Portfolio.bonds.allocation_matrix[-1])
#            new_num=[]
#            for k in range(7):
#                num_k = []
#                for t in range(time_step):
#                    num_k.append(Portfolio.bonds.num_of_bonds[k][t,:])
#                num_k.append(np.zeros(20))    
#                new_num.append(np.asarray(num_k))
#            Portfolio.bonds.num_of_bonds = new_num
#            
#        if len(Portfolio.bonds.allocation_matrix) != time_step+1 or len(Portfolio.bonds.num_of_bonds[0]) != time_step+1:
#            raise ValueError('Bonds_Portofolio object has a wrong dimension in number of bonds')
#                        
#       
#        # on modifie la TtM (Time to maturity) dans num_of_bonds et allocation_matrix
#        for k in range(7):
#            for TtM in range(2,21):
#                Portfolio.bonds.num_of_bonds[k][:,TtM-2] = Portfolio.bonds.num_of_bonds[k][:,TtM-1]
#                Portfolio.bonds.allocation_matrix[time_step][k, TtM-2] = Portfolio.bonds.allocation_matrix[time_step][k, TtM-1]
#            Portfolio.bonds.num_of_bonds[k][:,19] = np.zeros(len(Portfolio.bonds.num_of_bonds[k]))
#            Portfolio.bonds.allocation_matrix[time_step][k,19] = 0
#        
#        income_list =   Portfolio.bonds.valorize_bonds_income(valuation_date = time_step)
#        
#        coupon_list = Portfolio.bonds.get_coupon_list(valuation_date = time_step)
#        
#        # ===============================================================================
#        # on vérifie que le book value augmente et diminue du montant passant en argument
#        # ===============================================================================
#        if time_step <time_horizon:
#            print('{0:.2f}    | {1:.2f}  |  {2:.2f}  |   {3:.2f}   | {4:.2f}  | {5:.2f}  | {6:.2f} |'.format(
#                  time_step, 
#                  np.sum(np.asarray(Portfolio.bonds.num_of_bonds)), 
#                  np.sum(Portfolio.bonds.get_book_value_list(time_step)), 
#                  np.sum(Portfolio.bonds.get_market_value_list(time_step)), 
#                  np.sum(np.asarray(Portfolio.bonds.get_coupon_list(time_step))),
#                  income_list['PMVL'],
#                  pmvr))  
#              
#        print()
#        if time_step == 2:
#            Portfolio.delete_bonds(amount=500, valuation_date = time_step, book_value= True)
#            income_list =   Portfolio.bonds.valorize_bonds_income(valuation_date = time_step)
#            print('Sell of bonds for an amount of 500 (BV)')
#            print('----------------------------------------------')  
#            print('{0:.2f}    | {1:.2f}  |  {2:.2f}  |   {3:.2f}   | {4:.2f}  | {5:.2f}  | {6:.2f} |'.format(
#                    time_step,
#                    np.sum(np.asarray(Portfolio.bonds.num_of_bonds)),
#                    np.sum(Portfolio.bonds.get_book_value_list(time_step)),
#                    np.sum(Portfolio.bonds.get_market_value_list(time_step)), 
#                    np.sum(np.asarray(Portfolio.bonds.get_coupon_list(time_step))),
#                    income_list['PMVL'],
#                    Portfolio.asset_income[time_step]['PMVR_obligation_TF']))         