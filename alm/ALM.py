# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 15:44:33 2016

@author: Quang Dien DUONG-Jun Jing
"""

from ..liability.liability_data.Liabilities_data_m import Liabilities_data_m
from ..liability.liability_data.Liabilities_data import Liabilities_data
#from ..asset.Assets import Assets
from ..esg.ESG_RN import ESG_RN
#from ..esg.model_equity import EQ_model_classes as EQ_model
#from ..esg.interest_rate import IR_model_classes as IR_model
#from ..esg.credit_risk import credit_model_classes as credit_model
# This package aims to return the balance sheet at each time steps
from .Balance_Sheets import Balance_Sheets
## Python packages
import numpy as np

class ALM(object):
    """This is here that we will implement the asset liabilities interactions

        Attributes:
        ===========

        Input:
        _______
        1. assets:
            Type: instance of Assets class
            Function: see Assets class for more details

        2. liabilities:
            Type: instance of Liabilities class
            Function: see Liabilities class for more details

        3. time_horizon:
            Type: int
            Function: time horizon of the simulation for the overall system analysis (By default, this parameter is None)

        Output:
        _______
        1. BEL_cashflows:
            Type: array
            Function: corresponding list of BEL cashflows values over time horizon

        2. liabilities_book_value:
            Type: array
            Function: corresponding list of liabilities book values over time horizon

        3. assets_book_value:
            Type: array
            Function: corresponding list of assets book values over time horizon


        Methods:
        ========
        1. reset_allocation:

        2. reinvestment:

        3. treatment_begin_period:

        4. treatment_end_period:

        5. commpute_BEL_cashflows_scenario
    """

    def __init__(self, Assets, Liabilities, time_horizon):
        self.assets = Assets
        self.liabilities = Liabilities
        self.balance_sheets = []
        self.time_horizon = time_horizon
        self.assets_book_value = []
        self.assets_book_value.append(self.liabilities.technical_provision.mathematical_provision[0])
        
        self.assets_market_value = []
        self.assets_market_value.append(self.liabilities.technical_provision.mathematical_provision[0])
        # Temporary surrender paramaters
        self.RC_alpha = -0.05
        self.RC_beta = - 0.01
        self.RC_gamma = 0.01
        self.RC_delta = 0.03
        self.RC_min = -0.05
        self.RC_max = 0.3

        self.bf_migr = []   # Attribue utilisée pour tester le moteur
        self.aft_migr = []  # Atriibue utilisée pour tester le moteur
        self.aft_rell = [] # Attribue utilisée pour tester le moteur 

        self.coupon = []
        
    def execute_unrealised_bonds_loss(self, amount, valuation_date):
        """
            Method : execute_unrealised_bonds_loss
            
            Function : execute the unrealised bonds loss for an amount of "amount" at time ""valuation_date"
            
            Parameters : 
                1. amount:
                    Type: float (positive only)
                    Function: corresponding amount of bonds' unrealised loss that we want to realise.
                2. valuation_date
                    Type: int
                    Function: valuation date -- the moment where we realise the bonds' unrealised loss
        """
        reserve_capi =  self.liabilities.capitalization_reserve[valuation_date]
        
        
        
        if round(self.assets.asset_income[valuation_date]['MVL_obligation_TF'],3) >= round(reserve_capi,3):
        
            self.assets.bonds.execute_unrealised_bonds_loss(amount = reserve_capi, valuation_date = valuation_date)
            self.assets.asset_income[valuation_date]['PMVL'] += reserve_capi
            self.assets.asset_income[valuation_date]['MVL'] -= reserve_capi
            self.assets.asset_income[valuation_date]['MVL_obligation_TF'] -= reserve_capi
        # Liabilities
        self.liabilities.capitalization_reserve[valuation_date] = 0

        if round(self.assets.asset_income[valuation_date]['MVL_obligation_TF'],3) >= round(amount,3):
            self.assets.bonds.execute_unrealised_bonds_loss(amount = amount, valuation_date = valuation_date)
            self.assets.asset_income[valuation_date]['PMVL'] += amount
            self.assets.asset_income[valuation_date]['MVL'] -= amount
            self.assets.asset_income[valuation_date]['MVL_obligation_TF'] -= amount
            
        else:
            raise ValueError('Residual MVL is not sufficient for the amount demanded')
            
        
    
    # Pour les deux méthodes qui suivent: execute_unrealised_asset_gain et execute_unrealised_asset_loss, on modifie pas le compte PMVR_hors_obligation;
    # En effet, c'est un compte résiduel dans lequel on garde que les "changements"
    # Et ce qui était réalisé va directement impacter le compte available_wealth    
        
        
        
    def execute_unrealised_asset_gain(self, amount, valuation_date):
        """
            Method: execute_unrealised_asset_gain

            Function: return the modified market value of equity and its corresponding incomes after the realisation of capital gains

            Parameters:
                1. amount:
                    Type: float (positive only)
                    Function: corresponding amount of equity's unrealised gains that we want to realise.
                2. valuation_date
                    Type: int
                    Function: valuation date-the moment where we realise the equity's unrealised gains
        """

        assert amount >= 0, "We cannot realise PVL with a negative amount"
        PVL_action = max(0,self.assets.EQ_market_value[valuation_date] - self.assets.EQ_book_value[valuation_date])
        if PVL_action == 0:
            pass
        else:
            assert round(amount, 5) <= round(PVL_action,5), "Equity unrealised profits are not sufficent to reach the target amount"
            if self.assets.EQ_market_value[valuation_date] == 0:
                pass
            else:
                self.assets.number_EQ = (1 - amount/self.assets.EQ_market_value[valuation_date])*self.assets.number_EQ
                self.assets.asset_income[valuation_date]['PMVL'] -= amount
                self.assets.asset_income[valuation_date]['PVL'] -= amount
                self.assets.asset_income[valuation_date]['PVL_hors_obligation'] -= amount
                self.assets.EQ_market_value[valuation_date] -= amount

    def execute_unrealised_asset_loss(self, amount, valuation_date):
        """
            Method: execute_unrealised_asset_loss

            Function: return the modified market value of equity and its corresponding incomes after the realisation of capital losses.

            Parameters:
                1. amount:
                    Type: float (positive only)
                    Function: corresponding amount of equity's unrealised losses that we want to realise.
                2. valuation_date
                    Type: int
                    Function: valuation date-the moment where we realise the equity's unrealised losses.
        """
        # ===============================================
        # Modify asset_income due to the execution
        # ===============================================
        self.assets.asset_income[valuation_date]['PMVL'] += amount
        MVL_action = self.assets.asset_income[valuation_date]['MVL'] - self.assets.asset_income[valuation_date]['MVL_obligation_TF']
        # Consistency check
        delta = self.assets.EQ_market_value[valuation_date] - self.assets.EQ_book_value[valuation_date]
        if (delta < 0) and (np.absolute(round(MVL_action,0)) != round(np.absolute(delta),0)) :
            raise ValueError("MVLs are not matched !")
        assert round(amount,3) >= 0, "We cannot realise MVL with a negative amount"
            
        if amount <= MVL_action:
            if self.assets.EQ_market_value[valuation_date]== 0:
                pass
            else:
                self.assets.number_EQ = (1 + amount/self.assets.EQ_market_value[valuation_date]) * self.assets.number_EQ
                self.assets.EQ_market_value[valuation_date] += amount
                self.assets.asset_income[valuation_date]['MVL'] -= amount            
        else:
            if round(MVL_action,5) == 0:
                pass
            else:
                assert self.assets.EQ_market_value[valuation_date] > 0, " Numerical error in MVL"
                self.assets.number_EQ = self.assets.EQ_book_value[valuation_date]/self.assets.EQ_market_value[valuation_date] * self.assets.number_EQ
                self.assets.EQ_market_value[valuation_date] = self.assets.EQ_book_value[valuation_date]
                self.assets.asset_income[valuation_date]['MVL'] -= MVL_action
            
                amount -= MVL_action
                # Execute unrealised bonds loss
                self.execute_unrealised_bonds_loss(amount = amount, valuation_date = valuation_date)
            
                        
         

    def update_capitalization_reserve(self, valuation_date):
        """
            Method: update_capitalization_reserve

            Function: update the capitalization reserve value after revaluation.

            Parameters:
                1. valuation_date:
                    Type: int
                    Function: the moment of execution

        """
        
        rescapi = self.liabilities.capitalization_reserve[valuation_date]
        resu = max(0, rescapi + self.assets.asset_income[valuation_date]['PMVR_obligation_TF'])
        self.liabilities.capitalization_reserve[valuation_date] = resu
        if resu > 0 :
            self.assets.asset_income[valuation_date]['PMVR_obligation_TF'] = 0.
        else:
            self.assets.asset_income[valuation_date]['PMVR_obligation_TF'] += rescapi



    def reset_allocation(self, traj_i, valuation_date):
        """
            Method: reset_allocation

            Function: recovers asset's allocations at valuation date, not only the 1:2:7 ratio is verified among the asset classes 
                        but also the allocation matrix is verified inside the bond portfolio.

            Parameter:
                1. valuation_date:
                    Type: int
                    Function: valuation_date
        """
        
        # ========================
        # reset_allocation['cash']
        # ========================
        self.assets.cash[valuation_date] = (1 - self.assets.allocations['EQ'] - self.assets.allocations['bonds']) * self.assets_market_value[valuation_date]
        targer_EQ_market_value = self.assets.allocations['EQ'] * self.assets_market_value[valuation_date]
        targer_bond_market_value = self.assets.allocations['bonds'] * self.assets_market_value[valuation_date]
        # ======================
        # reset_allocation['EQ']
        # ======================
        if self.assets.EQ_market_value[valuation_date] >= targer_EQ_market_value:
            value = self.assets.EQ_market_value[valuation_date] - targer_EQ_market_value
            self.assets.delete_EQ(amount = value, valuation_date = valuation_date, book_value = False)
        else:
            value = targer_EQ_market_value - self.assets.EQ_market_value[valuation_date]
            self.assets.add_EQ(amount = value, traj_i = traj_i, valuation_date = valuation_date)
        
        
        # ========================
        # reset_allocation['bond']
        # ========================
        PMVR = 0   
        # spécificité: faut qu'on calcule tout d'abord les ventes des obligations, qui génèrent des PMVR. On effectue un netting pour les +/- (réserve de capitalisation) 
        MV_matrix = np.asarray(self.assets.bonds.get_market_value_list(valuation_date = valuation_date))
        BV_matrix = np.asarray(self.assets.bonds.get_book_value_list(valuation_date = valuation_date))
        PMVL_matrix = MV_matrix - BV_matrix
        delta_bond_market_value = targer_bond_market_value* (self.assets.bonds.target_allocation / sum(sum(self.assets.bonds.target_allocation))) - MV_matrix
        for k in range(7):
            for TtM in range(1,21):
                if delta_bond_market_value[k, TtM-1] >= 0:
                    delta_num_bond = self.assets.bonds.get_num_of_bond(amount = delta_bond_market_value[k, TtM-1], purchase_date = valuation_date, rating = k, TtM = TtM)
                    self.assets.bonds.num_of_bonds[k][valuation_date,TtM-1] += delta_num_bond
                    self.assets.bonds.allocation_matrix[valuation_date][k, TtM-1] += delta_num_bond
                    
                else:
                    # si on vend..
                    # comme par hasard.. le compte cash correspond à delta MV
                    self.assets.bonds.num_of_bonds[k][:, TtM-1] -=  np.absolute(delta_bond_market_value[k, TtM-1]) / MV_matrix[k, TtM-1]*self.assets.bonds.num_of_bonds[k][:, TtM-1]
                    self.assets.bonds.allocation_matrix[valuation_date][k, TtM-1] -= np.absolute(delta_bond_market_value[k, TtM-1]) / MV_matrix[k, TtM-1]*self.assets.bonds.allocation_matrix[valuation_date][k, TtM-1]
                    
                    # et les PMVR générés par les ventes?
                    PMVR -= PMVL_matrix[k, TtM-1] * delta_bond_market_value[k, TtM-1] / MV_matrix[k, TtM-1]
                    # faire attention à la signe... si on rélise la plus value latente, PMVR augmente, sinon PMVR baisse
                    # delta_bond_market_value est négatif               
        
        self.assets.asset_income[valuation_date]['PMVR_obligation_TF'] += PMVR
        
    def treatment_begin_period(self, traj_i, time_step, deterministic = False):
        """
            Method: treatment_begin_period

            Function: compute the distributed wealth as well as the profit sharing rate for each model point at each time_step

            Parameters:
                1. traj_i:
                    Type: int
                    Function: the i-th economic scenario
                2. time_step:
                    Type: int
                    Function: valuation date
        """
        # ===================================================================================
        # Assets: Update Cash
        # ===================================================================================
        self.additional_fund = 0
        self.distributed_wealth = 0
        self.assets.add_cash(amount = self.assets.cash[-1])
        
        cash_return_rate = self.assets.ESG_RN.scenarios[traj_i]['IR_curves'][time_step-1, 0]
        self.cash_revenue = (self.liabilities.technical_provision.profit_sharing_reserve[-1] + self.assets.cash[-1])*cash_return_rate
        
        self.available_wealth = self.cash_revenue + self.assets.asset_income[time_step-1]['PMVR_hors_obligation'] + self.assets.asset_income[time_step-1]['PMVR_obligation_TF']
        
        # ===================================================================================
        # Liabilities: Initilize Own Fund, Capitalization Reserve and PRE
        # ===================================================================================
                
        self.liabilities.own_fund.append(self.liabilities.own_fund[-1]*(1+cash_return_rate))
        self.liabilities.capitalization_reserve.append(self.liabilities.capitalization_reserve[-1]*(1+cash_return_rate))
        self.liabilities.PRE.append(self.liabilities.PRE[-1]*(1+cash_return_rate))
        
        Cash_Return = self.cash_revenue
        # ==========================================
        # Initialize Balance Sheet
        self.balance_sheets.append(Balance_Sheets())
        
        #=====================================================================================================================================================================================
        # Update Balance_Sheet
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'cash_0', value = self.assets.cash[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'equity_market_value_0', value = self.assets.EQ_market_value[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'equity_book_value_0', value = self.assets.EQ_book_value[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'bond_book_value_0', value = self.assets.get_bond_book_value(valuation_date = time_step))
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'bond_market_value_0', value = self.assets.get_bond_market_value(valuation_date = time_step))
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'PMVR_action_0', value = self.assets.asset_income[time_step]['PMVR_hors_obligation'])
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'PMVR_obligation_0', value = self.assets.asset_income[time_step]['PMVR_obligation_TF'])
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'available_wealth_0', value = self.available_wealth)
        self.balance_sheets[time_step].update(key_lv1 = 'liabilities', key_lv2 = 'own_fund_0', value = self.liabilities.own_fund[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'liabilities', key_lv2 = 'capitalization_reserve_0', value = self.liabilities.capitalization_reserve[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'liabilities', key_lv2 = 'PRE_0', value = self.liabilities.PRE[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'liabilities', key_lv2 = 'profit_sharing_reserve_0', value = self.liabilities.technical_provision.profit_sharing_reserve[time_step-1])
        self.balance_sheets[time_step].update(key_lv1 = 'liabilities', key_lv2 = 'mathematical_provision_0', value = self.liabilities.technical_provision.mathematical_provision[time_step-1])
        
        # ========================================================================
        # Update balance sheets
        if Cash_Return >=0:
            self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_in', key_lv2 = 'Cash_Return', value = Cash_Return)
            self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_out', key_lv2 = 'Cash_Return', value = 0.)
        else:
            self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_in', key_lv2 = 'Cash_Return', value = 0.)
            self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_out', key_lv2 = 'Cash_Return', value = np.absolute(Cash_Return))
        #=====================================================================================================================================================================================
        
        
        
        # =======================================================
        # coupons
        # spécifité: les coupons doivent être calculé selon l'état avant la migration, mais on tient compte de la défaut...
        # =======================================================
        coupon_list = self.assets.bonds.get_coupon_list(valuation_date = time_step)
        non_default_proba = np.ones(7) - self.assets.ESG_RN.scenarios[traj_i]['RN_migration_matrix'][time_step][:7,7]

        
        coupon_rating_array = np.asarray(coupon_list).sum(axis=1) * non_default_proba 
        coupon = sum(coupon_rating_array)
        self.coupon.append(coupon)
        self.assets.asset_income[time_step]['Revenu'] += coupon
        
        # =======================================================
        # migration
        # =======================================================
        self.bf_migr.append(np.sum(self.assets.bonds.allocation_matrix[time_step],axis=1))        
                
        
        num_of_bonds_temps =[]
        allocation_mat_temps =[]
        for k in range(7):
            num_of_bonds_k_temps = []
            allocation_mat_k_temps =[]
            for k2 in range(8):
                num_of_bonds_k_temps.append(self.assets.bonds.num_of_bonds[k]*self.assets.ESG_RN.scenarios[traj_i]['RN_migration_matrix'][time_step][k,k2])
                # num_of_bonds_k_temps de dimension 8*une matrice(temps d'achat et maturité)
                allocation_mat_k_temps.append(self.assets.bonds.allocation_matrix[time_step][k,:]*self.assets.ESG_RN.scenarios[traj_i]['RN_migration_matrix'][time_step][k,k2])
            num_of_bonds_temps.append(num_of_bonds_k_temps)
            # num_of_bonds_temps de dimension 7*8*XXX
            allocation_mat_temps.append(allocation_mat_k_temps)
        
        # une somme sur k, selon k2, cela donne les nouveaux num_of_bonds et allocation_mat après migration
        num_of_bonds = sum(np.asarray(num_of_bonds_temps))
        allocation_mat = sum(np.asarray(allocation_mat_temps))
        # num_of_bonds de dim 8*XXX
        
        self.assets.bonds.num_of_bonds = num_of_bonds[:7]
        self.assets.bonds.allocation_matrix[time_step] = allocation_mat[:7]
        
        self.aft_migr.append(np.sum(self.assets.bonds.allocation_matrix[time_step],axis=1))       

        cash_recovered = self.assets.ESG_RN.credit_model.recovery_rate * sum(sum(num_of_bonds[-1]))
        # =======================================================
        # =======================================================        
        # les obligations passées en défaut ne génèrent pas de PMVR                
        #=====================================================================================================================================================================================
        # Update Balance_Sheet après migration
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'cash_bf_1', value = self.assets.cash[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'equity_market_value_bf_1', value = self.assets.EQ_market_value[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'equity_book_value_bf_1', value = self.assets.EQ_book_value[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'bond_book_value_bf_1', value = self.assets.get_bond_book_value(valuation_date = time_step))
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'bond_market_value_bf_1', value = self.assets.get_bond_market_value(valuation_date = time_step))
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'PMVR_action_bf_1', value = self.assets.asset_income[time_step]['PMVR_hors_obligation'])
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'PMVR_obligation_bf_1', value = self.assets.asset_income[time_step]['PMVR_obligation_TF'])
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'available_wealth_bf_1', value = self.available_wealth)
        self.balance_sheets[time_step].update(key_lv1 = 'liabilities', key_lv2 = 'own_fund_bf_1', value = self.liabilities.own_fund[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'liabilities', key_lv2 = 'capitalization_reserve_bf_1', value = self.liabilities.capitalization_reserve[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'liabilities', key_lv2 = 'profit_sharing_reserve_bf_1', value = self.liabilities.technical_provision.profit_sharing_reserve[time_step-1])
        self.balance_sheets[time_step].update(key_lv1 = 'liabilities', key_lv2 = 'mathematical_provision_bf_1', value = self.liabilities.technical_provision.mathematical_provision[time_step-1])
        #=====================================================================================================================================================================================
        # ====================================
        
        
        # =======================================================
        # Obligations arrivant à maturité
        # =======================================================
        
        list_temps = []
        for k in range(7):
            list_temps.append(self.assets.bonds.num_of_bonds[k][:,0])
        vector_bonds_expired = np.asarray(list_temps)
        
        # chaque obligation est de nominal 1
        nominal = sum(sum(vector_bonds_expired))
        # =======================================================
        #  changer le TtM avant d'appeler la fonction add_bond
        # =======================================================
        # self.assets.bonds.allocation_matrix.append(self.assets.bonds.allocation_matrix[-1])
        
        for k in range(7):
            for TtM in range(2,21):
                self.assets.bonds.num_of_bonds[k][:,TtM-2]=self.assets.bonds.num_of_bonds[k][:,TtM-1]
                self.assets.bonds.allocation_matrix[time_step][k,TtM-2]=sum(self.assets.bonds.num_of_bonds[k][:,TtM-2])
            self.assets.bonds.num_of_bonds[k][:,19] = np.zeros(len(self.assets.bonds.num_of_bonds[k]))
            self.assets.bonds.allocation_matrix[time_step][k,19] = 0
        
        
        
        # =======================================================
        # enfin, on investit les nominaux récupérés (des obligations qui ont fait défaut ou arrivent à maturité) à nouveau dans les obligations
        # =======================================================
        
        self.assets.add_bonds(amount = nominal+cash_recovered, valuation_date = time_step)   
        # =======================================================
        # il reste à modifier asset_income...ou pas
        # Attention: dans la méthode bonds.valorize_bonds_income, on recompte les coupons... 
        # Dans la boucle du bas de temps, on valorise une fois asset_income, cela suffit
        # =======================================================
        
        
        
        #=====================================================================================================================================================================================
        # Update Balance_Sheet
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'cash_1', value = self.assets.cash[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'equity_market_value_1', value = self.assets.EQ_market_value[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'equity_book_value_1', value = self.assets.EQ_book_value[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'bond_book_value_1', value = self.assets.get_bond_book_value(valuation_date = time_step))
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'bond_market_value_1', value = self.assets.get_bond_market_value(valuation_date = time_step))
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'bond_coupon_value_1', value = self.coupon[time_step-1])
        
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'PMVR_action_1', value = self.assets.asset_income[time_step]['PMVR_hors_obligation'])
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'PMVR_obligation_1', value = self.assets.asset_income[time_step]['PMVR_obligation_TF'])
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'available_wealth_1', value = self.available_wealth)
        self.balance_sheets[time_step].update(key_lv1 = 'liabilities', key_lv2 = 'own_fund_1', value = self.liabilities.own_fund[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'liabilities', key_lv2 = 'capitalization_reserve_1', value = self.liabilities.capitalization_reserve[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'liabilities', key_lv2 = 'profit_sharing_reserve_1', value = self.liabilities.technical_provision.profit_sharing_reserve[time_step-1])
        self.balance_sheets[time_step].update(key_lv1 = 'liabilities', key_lv2 = 'mathematical_provision_1', value = self.liabilities.technical_provision.mathematical_provision[time_step-1])
        # ====================================================================================================================================================================================
        # Update asset_book_value and
        # Compute Assets Amortizing Book Value
        # ====================================
        
        self.assets_book_value.append(self.assets.cash[time_step] + self.assets.EQ_book_value[time_step] + self.assets.get_bond_book_value(valuation_date = time_step))
        self.assets_market_value.append(self.assets.cash[time_step] + self.assets.EQ_market_value[time_step] + self.assets.get_bond_market_value(valuation_date = time_step))
        
        amortizing_value = self.assets_book_value[time_step] - self.liabilities.technical_provision.mathematical_provision[-1]
        if amortizing_value >= 0.:
            self.assets.delete_bonds(amount = amortizing_value, valuation_date = time_step, book_value = True)
            self.available_wealth += (amortizing_value + self.assets.asset_income[time_step]['PMVR_obligation_TF'])
            self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_in', key_lv2 = 'amortizing_value', value = amortizing_value + self.assets.asset_income[time_step]['PMVR_obligation_TF'])
            self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_out', key_lv2 = 'amortizing_value', value = 0.)
            self.assets.asset_income[time_step]['PMVR_obligation_TF'] = 0.
        else:
            self.assets.cash[time_step] += np.absolute(amortizing_value)
            self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_in', key_lv2 = 'amortizing_value', value = 0.)
            self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_out', key_lv2 = 'amortizing_value', value = np.absolute(amortizing_value))
            self.available_wealth -= np.absolute(amortizing_value)

        # ====================================================================================================================================================================================
        # Consistency Check
        self.assets_book_value[time_step] = (self.assets.cash[time_step] 
                                             + self.assets.EQ_book_value[time_step] 
                                             + self.assets.get_bond_book_value(valuation_date = time_step))
        value = self.liabilities.technical_provision.mathematical_provision[-1]
        if round(self.assets_book_value[time_step],0) != round(value,0):
            raise ValueError("La bilan comptable n'est pas carré !")
        # ====================================================================================================================================================================================
        
        # ============================================================
        # Update asset_income after paying the assets amortizing value
        # ============================================================
        PMVL_obligation_TF = self.assets.get_bond_market_value(time_step) - self.assets.get_bond_book_value(time_step)
        self.assets.asset_income[time_step]['PVL_obligation_TF'] = max(0, PMVL_obligation_TF)
        MVL_obligation_TF = max(0, -PMVL_obligation_TF)
        self.assets.asset_income[time_step]['PVL'] = self.assets.asset_income[time_step]['PVL_hors_obligation'] + self.assets.asset_income[time_step]['PVL_obligation_TF']
        self.assets.asset_income[time_step]['MVL'] += (MVL_obligation_TF - self.assets.asset_income[time_step]['MVL_obligation_TF'])
        # attention! ne pas compter deux fois MVL_obligation_TF, on enlève l'ancien (l'élément après la signe moins n'est pas encore mis à jour) et on le remplace par le nouveau
        
        self.assets.asset_income[time_step]['MVL_obligation_TF'] = MVL_obligation_TF
        # on le met à jour maintenant
        
        self.assets.asset_income[time_step]['PMVL'] = self.assets.asset_income[time_step]['PVL'] - self.assets.asset_income[time_step]['MVL']
        #=====================================================================================================================================================================================
        # Update Balance_Sheet
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'cash_2', value = self.assets.cash[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'equity_market_value_2', value = self.assets.EQ_market_value[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'equity_book_value_2', value = self.assets.EQ_book_value[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'bond_book_value_2', value = self.assets.get_bond_book_value(valuation_date = time_step))
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'bond_market_value_2', value = self.assets.get_bond_market_value(valuation_date = time_step))
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'PMVR_action_2', value = self.assets.asset_income[time_step]['PMVR_hors_obligation'])
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'PMVR_obligation_2', value = self.assets.asset_income[time_step]['PMVR_obligation_TF'])
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'available_wealth_2', value = self.available_wealth)
        self.balance_sheets[time_step].update(key_lv1 = 'liabilities', key_lv2 = 'own_fund_2', value = self.liabilities.own_fund[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'liabilities', key_lv2 = 'capitalization_reserve_2', value = self.liabilities.capitalization_reserve[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'liabilities', key_lv2 = 'profit_sharing_reserve_2', value = self.liabilities.technical_provision.profit_sharing_reserve[time_step-1])
        self.balance_sheets[time_step].update(key_lv1 = 'liabilities', key_lv2 = 'mathematical_provision_2', value = self.liabilities.technical_provision.mathematical_provision[time_step-1])
        #=====================================================================================================================================================================================
    
                
        
        # ==================================
        # Compute assets target rate
        # ==================================
        target_rate = self.assets.get_target_rate(traj_i, valuation_date = time_step)
        self.assets.target_rates.append(target_rate)
        # ==========================================
        # Update model point's average_age and seniority
        # ==========================================
        for mdlp in self.liabilities.liabilities_data.model_points:
            mdlp.average_age += 1
            mdlp.seniority += 1

        # =====================================================================
        # Compute dynamical lapse_rate
        # =====================================================================
        TME = self.assets.ESG_RN.scenarios[traj_i]['IR_curves'][time_step - 1,9]
        for mdlp in self.liabilities.liabilities_data.model_points:
            RC = 0
            # =================================================================
            # Compute temporary surrender rate
            # =================================================================
            if mdlp.profit_sharing_rate[-1] - TME < self.RC_alpha:
                RC = self.RC_max
            elif (mdlp.profit_sharing_rate[-1] - TME >= self.RC_alpha) and (mdlp.profit_sharing_rate[-1] - TME < self.RC_beta):
                RC = self.RC_max * (mdlp.profit_sharing_rate[-1] - TME - self.RC_beta) / (self.RC_alpha - self.RC_beta)
            elif (mdlp.profit_sharing_rate[-1] - TME >= self.RC_beta) and (mdlp.profit_sharing_rate[-1] - TME < self.RC_gamma):
                RC = 0
            elif (mdlp.profit_sharing_rate[-1] - TME >= self.RC_gamma) and (mdlp.profit_sharing_rate[-1] - TME < self.RC_delta):
                RC = self.RC_min * (mdlp.profit_sharing_rate[-1] - TME - self.RC_gamma) / (self.RC_delta - self.RC_gamma)
            else:
                RC = self.RC_min

            RS = mdlp.lapse_rate[mdlp.seniority]
            mdlp.dynamical_lapse_rate = min(1, max(0, RS + RC))

        # ======================================================
        # Update cash_flows
        # =====================================================
        self.liabilities.update_cash_flow_in_out(valuation_date = time_step)
        #=====================================================================================================================================================================================
        # Update Balance Sheet
        self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_in', key_lv2 = 'premium', value = self.liabilities.cash_flow_in[time_step - 1])
        # cette ligne n'a aucune utilité, le même élément sera effacé après        
        #=====================================================================================================================================================================================
        # =======================================================
        #                   Premium
        # =======================================================
        self.assets.cash[time_step] += self.liabilities.cash_flow_in[time_step - 1]
        # =====================================================
        # Compute Revenu_disponible and update available_wealth
        # =====================================================
        Revenu = self.assets.asset_income[time_step]['Revenu']
        self.available_wealth += Revenu
        #=====================================================================================================================================================================================
        # Update Balance Sheet
        self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_in', key_lv2 = 'revenu', value = Revenu)
        #=====================================================================================================================================================================================
        # Consistency check
        self.assets_book_value[time_step] = (self.assets.cash[time_step] 
                                            + self.assets.EQ_book_value[time_step] 
                                            + self.assets.get_bond_book_value(valuation_date = time_step))
        value = (self.liabilities.technical_provision.mathematical_provision[-1] 
                + self.liabilities.cash_flow_in[time_step-1])
        if round(self.assets_book_value[time_step],0) != round(value,0):
            raise ValueError("La bilan comptable n'est pas carré !")
        # ===================================================
        #             Surrender
        # ===================================================
        # Assets
        self.assets.delete_asset(amount = self.liabilities.cash_flow_out[time_step-1], valuation_date = time_step, book_value = True)
        self.PMVR_hors_obligation = self.assets.asset_income[time_step]['PMVR_hors_obligation']
        self.PMVR_obligation_TF = self.assets.asset_income[time_step]['PMVR_obligation_TF']        
        # Liabilities_Update Capitalization_Reserve
        self.update_capitalization_reserve(valuation_date = time_step)
        # Update available_wealth
        self.available_wealth += self.assets.asset_income[time_step]['PMVR_hors_obligation'] + self.assets.asset_income[time_step]['PMVR_obligation_TF']
        # =====================================================================================================================================================================================
        # Consistency check
        self.assets_book_value[time_step] = (self.assets.cash[time_step] 
                                            + self.assets.EQ_book_value[time_step] 
                                            + self.assets.get_bond_book_value(valuation_date = time_step))
        value = (self.liabilities.technical_provision.mathematical_provision[-1] 
                + self.liabilities.cash_flow_in[time_step-1]
                -self.liabilities.cash_flow_out[time_step-1])
        if round(self.assets_book_value[time_step],0) != round(value,0):
            raise ValueError("La bilan comptable n'est pas carré !")
        # ====================================================================================================================================================================================
        # Update Balance_Sheets
        if self.assets.asset_income[time_step]['PMVR_hors_obligation'] >= 0:
            self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_in', key_lv2 = 'PMVR_rachat', value = self.assets.asset_income[time_step]['PMVR_hors_obligation'])
            self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_out', key_lv2 = 'PMVR_rachat', value = 0)
        else:
            self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_in', key_lv2 = 'PMVR_rachat', value = 0)
            self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_out', key_lv2 = 'PMVR_rachat', value = -self.assets.asset_income[time_step]['PMVR_hors_obligation'])
        self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_out', key_lv2 = 'MVR_TF_residuelle_rachat', value = -self.assets.asset_income[time_step]['PMVR_obligation_TF'])
        self.assets.asset_income[time_step]['PMVR_hors_obligation'] = 0
        self.assets.asset_income[time_step]['PMVR_obligation_TF'] = 0
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'cash_3', value = self.assets.cash[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'equity_market_value_3', value = self.assets.EQ_market_value[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'equity_book_value_3', value = self.assets.EQ_book_value[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'bond_book_value_3', value = self.assets.get_bond_book_value(valuation_date = time_step))
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'bond_market_value_3', value = self.assets.get_bond_market_value(valuation_date = time_step))
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'PMVR_action_3', value = self.assets.asset_income[time_step]['PMVR_hors_obligation'])
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'PMVR_obligation_3', value = self.assets.asset_income[time_step]['PMVR_obligation_TF'])
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'available_wealth_3', value = self.available_wealth)
        self.balance_sheets[time_step].update(key_lv1 = 'liabilities', key_lv2 = 'own_fund_3', value = self.liabilities.own_fund[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'liabilities', key_lv2 = 'capitalization_reserve_3', value = self.liabilities.capitalization_reserve[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'liabilities', key_lv2 = 'profit_sharing_reserve_3', value = self.liabilities.technical_provision.profit_sharing_reserve[time_step-1])
        self.balance_sheets[time_step].update(key_lv1 = 'liabilities', key_lv2 = 'mathematical_provision_3', value = self.liabilities.technical_provision.mathematical_provision[time_step-1])

        self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_out', key_lv2 = 'surrender_value', value = self.liabilities.cash_flow_out[time_step-1])
        #=====================================================================================================================================================================================

        # ===================================================
        #            Mortality
        # ===================================================
        somme = 0
        for mdlp in self.liabilities.liabilities_data.model_points:
            mdlp.cash_flow_out[-1] = mdlp.cash_flow_out[-1] + (mdlp.mortality_rate[int(mdlp.average_age)] * mdlp.mathematical_provision[-1])
            somme += mdlp.mortality_rate[int(mdlp.average_age)] * mdlp.mathematical_provision[-1]
        self.liabilities.cash_flow_out[-1] += somme
        # Assets
        self.assets.delete_asset(amount = somme, valuation_date = time_step, book_value = True)
        self.PMVR_hors_obligation += self.assets.asset_income[time_step]['PMVR_hors_obligation']
        self.PMVR_obligation_TF += self.assets.asset_income[time_step]['PMVR_obligation_TF']
        
        # Liabilities_Update Capitalization_Reserve
        self.update_capitalization_reserve(valuation_date = time_step)
        # update available wealth
        self.available_wealth += self.assets.asset_income[time_step]['PMVR_hors_obligation'] + self.assets.asset_income[time_step]['PMVR_obligation_TF']
        
        # =========================================================
        # Consistency Check
        self.assets_book_value[time_step] = (self.assets.cash[time_step] 
                                            + self.assets.EQ_book_value[time_step] 
                                            + self.assets.get_bond_book_value(valuation_date = time_step))
        value = (self.liabilities.technical_provision.mathematical_provision[-1] 
                + self.liabilities.cash_flow_in[time_step-1]
                -self.liabilities.cash_flow_out[time_step-1])
                
        if round(self.assets_book_value[time_step],0) != round(value,0):
            raise ValueError("La bilan comptable n'est pas carré !")
            
        # =========================================================================================================================================
        # Update PRE
        PMVL_action = self.assets.EQ_market_value[time_step] - self.assets.EQ_book_value[time_step]
        MVL_action = max(0, -PMVL_action)
        self.liabilities.PRE[-1] = min(self.liabilities.PRE[-1] + 1./3 * MVL_action, MVL_action)
        delta_PRE = self.liabilities.PRE[-1] - self.liabilities.PRE[-2]
        self.available_wealth -= delta_PRE
        # =========================================================================================================================================
        
        #=====================================================================================================================================================================================
        # Update Balance_Sheets
        if self.assets.asset_income[time_step]['PMVR_hors_obligation'] >= 0:
            self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_in', key_lv2 = 'PMVR_deces', value = self.assets.asset_income[time_step]['PMVR_hors_obligation'])
            self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_out', key_lv2 = 'PMVR_deces', value = 0.)
        else:
            self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_in', key_lv2 = 'PMVR_deces', value = 0.)
            self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_out', key_lv2 = 'PMVR_deces', value = -self.assets.asset_income[time_step]['PMVR_hors_obligation'])
        
        self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_out', key_lv2 = 'MVR_TF_residuelle_deces', value = -self.assets.asset_income[time_step]['PMVR_obligation_TF'])
        # on est sure que c'est MVR_TF_residuelle, car nous avons exécuté update_capitalization_reserve, et comme réserve capi>=0, PMVR_oblig<=0
        
        if delta_PRE >=0:
            self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_in', key_lv2 = 'delta_PRE', value = 0.)
            self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_out', key_lv2 = 'delta_PRE', value = delta_PRE)
        else:
            self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_in', key_lv2 = 'delta_PRE', value = - delta_PRE)
            self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_out', key_lv2 = 'delta_PRE', value = 0.)
        # ====================================================================================================================================================================================
        # Reset PMVR back to 0
        self.assets.asset_income[time_step]['PMVR_hors_obligation'] = 0.
        self.assets.asset_income[time_step]['PMVR_obligation_TF'] = 0.
                
        
        
        # ====================================================================================================================================================================================
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'cash_4', value = self.assets.cash[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'equity_market_value_4', value = self.assets.EQ_market_value[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'equity_book_value_4', value = self.assets.EQ_book_value[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'bond_book_value_4', value = self.assets.get_bond_book_value(valuation_date = time_step))
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'bond_market_value_4', value = self.assets.get_bond_market_value(valuation_date = time_step))
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'PMVR_action_4', value = self.assets.asset_income[time_step]['PMVR_hors_obligation'])
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'PMVR_obligation_4', value = self.assets.asset_income[time_step]['PMVR_obligation_TF'])
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'available_wealth_4', value = self.available_wealth)
        self.balance_sheets[time_step].update(key_lv1 = 'liabilities', key_lv2 = 'own_fund_4', value = self.liabilities.own_fund[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'liabilities', key_lv2 = 'capitalization_reserve_4', value = self.liabilities.capitalization_reserve[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'liabilities', key_lv2 = 'profit_sharing_reserve_4', value = self.liabilities.technical_provision.profit_sharing_reserve[time_step-1])
        self.balance_sheets[time_step].update(key_lv1 = 'liabilities', key_lv2 = 'mathematical_provision_4', value = self.liabilities.technical_provision.mathematical_provision[time_step-1])

        self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_out', key_lv2 = 'mortality_value', value = somme)
        #=====================================================================================================================================================================================
        # Compute richesse_TMG, richesse_voulu per model point
        # =======================================================
        for mdlp in self.liabilities.liabilities_data.model_points:
            mdlp.desired_rate_net = target_rate - mdlp.rate_sensibility
            # Taux voulu de chaque model point j
            if deterministic:
                mdlp.desired_rate = mdlp.TMG[mdlp.seniority]
            else:
                mdlp.desired_rate = max(mdlp.desired_rate_net, mdlp.TMG[mdlp.seniority])
            # Taux voulu
            mdlp.desired_rate_gross = mdlp.desired_rate + mdlp.margin_rate
            mdlp.TMG_gross = mdlp.TMG[mdlp.seniority] + mdlp.margin_rate
            # le margin_rate ici réprésente la proportion que l'on va distribuer aux actionnaires            
            
            mdlp.desired_rate_wealth = mdlp.mathematical_provision[-1] * (1 + mdlp.desired_rate_gross) + (mdlp.cash_flow_in[-1] - mdlp.cash_flow_out[-1]) * np.sqrt(1 + mdlp.desired_rate_gross)- (mdlp.mathematical_provision[-1] + mdlp.cash_flow_in[-1] - mdlp.cash_flow_out[-1])
            mdlp.TMG_wealth = mdlp.mathematical_provision[-1] * (1 + mdlp.TMG_gross) + (mdlp.cash_flow_in[-1] - mdlp.cash_flow_out[-1]) * np.sqrt(1 + mdlp.TMG_gross)- (mdlp.mathematical_provision[-1] + mdlp.cash_flow_in[-1] - mdlp.cash_flow_out[-1])
        # ================================================================
        # Compute richesse_max, richesse_min, richesse_TMG, richesse_voulu
        # ================================================================
        self.wealth_max = self.available_wealth + self.assets.asset_income[time_step]['PVL_hors_obligation']
        self.wealth_min = self.available_wealth - (self.assets.asset_income[time_step]['MVL']-min(self.liabilities.capitalization_reserve[-1],self.assets.asset_income[time_step]['MVL_obligation_TF']))
        self.desired_rate_wealth = sum(mdlp.desired_rate_wealth for mdlp in self.liabilities.liabilities_data.model_points)
        self.TMG_wealth = sum(mdlp.TMG_wealth for mdlp in self.liabilities.liabilities_data.model_points)                
        # =======================================================================
        # Compute distributed_wealth and profit_sharing_rate for each model point
        # Realise the PMVL if necessary and Initialize the capitalization_reserve
        # =======================================================================
        
        real_wealth_max = self.wealth_max + self.liabilities.technical_provision.profit_sharing_reserve[time_step -1]
        real_available_wealth = self.available_wealth + self.liabilities.technical_provision.profit_sharing_reserve[time_step -1]
        
        # =======================================================================
        #           Richesse_max sans PPE <= Richesse_TMG
        # =======================================================================
        # on débite le fond propre pour servir le TMG
        # ce n'est pas fini... car on doit "distribuer" la PPE de l'année précédente, donc c'est possible que l'on arrive à servir le taux voulu
        if self.wealth_max <= self.TMG_wealth:
            self.additional_fund = self.TMG_wealth - self.wealth_max
            #======================================
            # Update Balance Sheets
            PVR = self.assets.asset_income[time_step]['PVL_hors_obligation']
            self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_in', key_lv2 = 'PMVR', value = PVR)
            self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_out', key_lv2 = 'PMVR', value = 0)
            #=======================================
            self.execute_unrealised_asset_gain(amount = PVR, valuation_date = time_step)
            self.available_wealth += PVR
            #=======================================
            if real_wealth_max + self.additional_fund >= self.desired_rate_wealth:
                for mdlp in self.liabilities.liabilities_data.model_points:
                    mdlp.profit_sharing_rate.append(mdlp.desired_rate)
            else:
                # Dans les deux cas, on sert plus que TMG, car +PPE[-1]
                x1 = 0.
                x2 = 1.
                epsilon = 0.00001
                while ((x2-x1) > epsilon):
                    the_wealth = 0
                    x = (x1+x2)/2
                    for mdlp in self.liabilities.liabilities_data.model_points:
                        rate = mdlp.TMG[mdlp.seniority] + x*(mdlp.desired_rate - mdlp.TMG[mdlp.seniority]) + mdlp.margin_rate
                        the_wealth += mdlp.mathematical_provision[-1] * (1 + rate) + (mdlp.cash_flow_in[-1] - mdlp.cash_flow_out[-1]) * np.sqrt(1 + rate)- (mdlp.mathematical_provision[-1] + mdlp.cash_flow_in[-1] - mdlp.cash_flow_out[-1])
                    if (the_wealth > self.desired_rate_wealth) or (the_wealth < self.TMG_wealth):
                        raise ValueError("Wrong value for x")
                    elif (the_wealth > real_wealth_max + self.additional_fund):
                        x2 = x
                    else:
                        x1 = x
                # ========================================================================================
                # Consistency Check
                # ========================================================================================
                for mdlp in self.liabilities.liabilities_data.model_points:
                    mdlp.profit_sharing_rate.append(mdlp.TMG[mdlp.seniority] + x*(mdlp.desired_rate - mdlp.TMG[mdlp.seniority]))
                    # =================================================================================================================
                    # Consistency Check
                    if (mdlp.profit_sharing_rate[-1] < mdlp.TMG[mdlp.seniority]) or (mdlp.profit_sharing_rate[-1] > mdlp.desired_rate):
                        raise ValueError("Error for profit sharing rate")
                        # =================================================================================================================                
                
     
        # ==================================================================================
        #             Richesse_ TMG <= richesse_max, Richesse voulu <= richesse max avec PPE 
        # ==================================================================================       
        # on sert le taux voulu, mais combien de PMVR dépendra aussi le TMG
        
        elif real_wealth_max >= self.desired_rate_wealth and self.TMG_wealth <= self.wealth_max:
            for mdlp in self.liabilities.liabilities_data.model_points:
                mdlp.profit_sharing_rate.append(mdlp.desired_rate)
                
            if  real_available_wealth >= self.desired_rate_wealth and self.available_wealth >= self.TMG_wealth:
                # On peut réaliser des MVL
                #======================================
                # Update Balance Sheets
                MVR = min(real_available_wealth -self.desired_rate_wealth, self.available_wealth- self.TMG_wealth, self.available_wealth-self.wealth_min)
                self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_in', key_lv2 = 'PMVR', value = 0)
                self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_out', key_lv2 = 'PMVR', value = MVR)
                #=======================================
                self.execute_unrealised_asset_loss(amount = MVR, valuation_date = time_step)
                self.available_wealth -= MVR
                
            elif real_available_wealth >= self.desired_rate_wealth and self.available_wealth < self.TMG_wealth:
                # il parait que l'on arrive à servier le taux voulu avec réalisation MVL, mais on doit tout d'abord servir le taux TMG avec PVL, donc pas de réalisation MVL
                #=======================================
                # Update Balance Sheets
                PVR = self.TMG_wealth - self.available_wealth
                self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_in', key_lv2 = 'PMVR', value = PVR)
                self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_out', key_lv2 = 'PMVR', value = 0)
                #======================================
                self.execute_unrealised_asset_gain(amount = PVR, valuation_date = time_step)
                self.available_wealth += PVR
                # il reste encore self.available_wealth + PPE[-1] - richesse_voulu                
                
            elif real_available_wealth < self.desired_rate_wealth and self.available_wealth < self.TMG_wealth: 
                # on doit réaliser PVL pour payer la richesse voulu et pour payer le TMG
                PVR = max(self.desired_rate_wealth -real_available_wealth, self.TMG_wealth -self.available_wealth)
                self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_in', key_lv2 = 'PMVR', value = PVR)
                self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_out', key_lv2 = 'PMVR', value = 0)
                #======================================
                self.execute_unrealised_asset_gain(amount = PVR, valuation_date = time_step)
                self.available_wealth += PVR
                ## il reste encore self.available_wealth + PPE[-1] - richesse_voulu
                
            elif real_available_wealth < self.desired_rate_wealth and self.available_wealth >= self.TMG_wealth:  
                # pas de contraint au niveau TMG, il suffit de réaliser une partie nécessaire pour servir le taux voulu
                PVR = self.desired_rate_wealth -real_available_wealth
                #print('PVR in condition 2.4: ',PVR)
                self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_in', key_lv2 = 'PMVR', value = PVR)
                self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_out', key_lv2 = 'PMVR', value = 0)
                #======================================
                self.execute_unrealised_asset_gain(amount = PVR, valuation_date = time_step)
                self.available_wealth += PVR
                ## il reste encore self.available_wealth + PPE[-1] - richesse_voulu
            

                
        # =======================================================================
        #           Richesse_ TMG <= richesse_max, richesse_max + PPE <= Richesse_voulu
        # =======================================================================
        # on arrive pas à servir le taux voulu dans ce cas, mais TMG oui
        
        elif (self.desired_rate_wealth >= real_wealth_max) and (self.TMG_wealth <= self.wealth_max):
            # comme de toute manière qu'on réalisera toutes les PVL pour servir un taux au max, on n'a pas besoins de s'inquiéter sur les ressources de TMG
            x1 = 0.
            x2 = 1.
            epsilon = 0.00001
            
            while ((x2-x1) > epsilon):
                the_wealth = 0
                x = (x1+x2)/2
                for mdlp in self.liabilities.liabilities_data.model_points:
                    rate = mdlp.TMG[mdlp.seniority] + x*(mdlp.desired_rate - mdlp.TMG[mdlp.seniority]) + mdlp.margin_rate
                    the_wealth += mdlp.mathematical_provision[-1] * (1 + rate) + (mdlp.cash_flow_in[-1] - mdlp.cash_flow_out[-1]) * np.sqrt(1 + rate)- (mdlp.mathematical_provision[-1] + mdlp.cash_flow_in[-1] - mdlp.cash_flow_out[-1])
                if (the_wealth > self.desired_rate_wealth) or (the_wealth < self.TMG_wealth):
                    raise ValueError("Wrong value for x")
                elif (the_wealth > real_wealth_max):
                    x2 = x
                else:
                    x1 = x
            # ========================================================================================
            # Consistency Check
            # ========================================================================================
            for mdlp in self.liabilities.liabilities_data.model_points:
                mdlp.profit_sharing_rate.append(mdlp.TMG[mdlp.seniority] + x*(mdlp.desired_rate - mdlp.TMG[mdlp.seniority]))
                # =================================================================================================================
                # Consistency Check
                if (mdlp.profit_sharing_rate[-1] < mdlp.TMG[mdlp.seniority]) or (mdlp.profit_sharing_rate[-1] > mdlp.desired_rate):
                    raise ValueError("Error for profit sharing rate")
                # =================================================================================================================
            # ================================================================
            # Update Balance Sheets
            self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_in', key_lv2 = 'PMVR', value = self.assets.asset_income[time_step]['PVL_hors_obligation'])
            self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_out', key_lv2 = 'PMVR', value = 0)
            # ================================================================
            PVR = self.assets.asset_income[time_step]['PVL_hors_obligation']
            #print('PVR in condition 3: ',PVR)
            self.execute_unrealised_asset_gain(amount = PVR, valuation_date = time_step)
            self.available_wealth += PVR

        
        self.distributed_wealth = self.available_wealth + self.liabilities.technical_provision.profit_sharing_reserve[time_step-1] + self.additional_fund
        
        # =========================================================
        # Consistency Check
        self.assets_book_value[time_step] = (self.assets.cash[time_step] 
                                            + self.assets.EQ_book_value[time_step] 
                                            + self.assets.get_bond_book_value(valuation_date = time_step))
        value = (self.liabilities.technical_provision.mathematical_provision[-1] 
                + self.liabilities.cash_flow_in[time_step-1]
                -self.liabilities.cash_flow_out[time_step-1])
        if round(self.assets_book_value[time_step],0) != round(value,0):
            raise ValueError("La bilan comptable n'est pas carré !")                
        # ========================================================================

        # ======================================================================
        # Update Balance_Sheets
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'richesse_max', value = self.wealth_max)
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'richesse_min', value = self.wealth_min)
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'richesse_TMG', value = self.TMG_wealth)
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'richesse_voulue', value = self.desired_rate_wealth)
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'abondement', value = self.additional_fund)
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'equity_market_value_5', value = self.assets.EQ_market_value[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'bond_market_value_5', value = self.assets.get_bond_market_value(valuation_date = time_step))
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'PMVR_action_5', value = self.assets.asset_income[time_step]['PMVR_hors_obligation'])
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'PMVR_obligation_5', value = self.assets.asset_income[time_step]['PMVR_obligation_TF'])
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'available_wealth_5', value = self.available_wealth)
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'richesse_distribuee', value = self.distributed_wealth)
        self.balance_sheets[time_step].update(key_lv1 = 'liabilities', key_lv2 = 'own_fund_5', value = self.liabilities.own_fund[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'liabilities', key_lv2 = 'capitalization_reserve_5', value = self.liabilities.capitalization_reserve[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'liabilities', key_lv2 = 'profit_sharing_reserve_5', value = self.liabilities.technical_provision.profit_sharing_reserve[time_step-1])
        self.balance_sheets[time_step].update(key_lv1 = 'liabilities', key_lv2 = 'mathematical_provision_5', value = self.liabilities.technical_provision.mathematical_provision[time_step-1])
        # ======================================================================
        
    def treatment_end_period(self, traj_i, time_step, deterministic = False):
        if deterministic:
            self.liabilities.update_mathematical_provision_TMG()
        else:
            self.liabilities.update_mathematical_provision()

        if self.liabilities.technical_provision.mathematical_provision[-1] < 0:
            raise ValueError('Error! Mathematical provision is negative')
        
        # ===================================================
        # A cette étape là, on rend le bilan carré, mais on touche encore le compte de la richesse distribuée
        # ===================================================
        
        
        self.assets_book_value[-1] = self.assets.cash[time_step] + self.assets.EQ_book_value[time_step] + self.assets.get_bond_book_value(valuation_date = time_step)
        delta = self.assets_book_value[time_step] - self.liabilities.technical_provision.mathematical_provision[time_step]
        # =====================================================================
        # Consistency check
        PB = self.liabilities.technical_provision.mathematical_provision[-1] - (self.liabilities.technical_provision.mathematical_provision[-2] + self.liabilities.cash_flow_in[-1] - self.liabilities.cash_flow_out[-1])
        PB = PB
        if round(-delta,0) != round(PB,0):
            raise ValueError("There exist cash leakage in the ALM model. Correct it")
          
            
        if delta > 0:
            self.assets.delete_asset(amount = delta, valuation_date = time_step, book_value = True)
            self.update_capitalization_reserve(valuation_date = time_step)
            self.distributed_wealth += (delta + self.assets.asset_income[time_step]['PMVR_hors_obligation'] + self.assets.asset_income[time_step]['PMVR_obligation_TF'])
            # notons que le compte PMVR_hors_obligations n'a pas été touché dans la partie "richesse", donc reste à 0, ce compte récupère le PMVR résiduel lié aux ventes des actions
            self.assets.asset_income[time_step]['PMVR_hors_obligation'] = 0.
            self.assets.asset_income[time_step]['PMVR_obligation_TF'] = 0.
        else:
            self.assets.cash[time_step] += np.absolute(delta)
            self.distributed_wealth -= np.absolute(delta)
            
        # =================================================================================
        # application du PB min, max(TMG, 85% du produit financier) est destiné aux assurés
        # =================================================================================
        
            
        self.liabilities.update_own_fund(valuation_date = time_step, additional_fund = self.additional_fund)
        
        if self.liabilities.contractual_margin[-1] >=  0.15* self.available_wealth:
            # remettre à jour own fund
            delta_own_fund = self.liabilities.contractual_margin[-1] - 0.15*self.available_wealth
            self.liabilities.contractual_margin[-1] = 0.15*self.available_wealth
            self.liabilities.own_fund[time_step] -=delta_own_fund

        
        
        self.liabilities.technical_provision.profit_sharing_reserve.append(self.distributed_wealth - self.liabilities.contractual_margin[-1])
        self.assets_book_value[-1] = self.assets.cash[time_step] + self.assets.EQ_book_value[time_step] + self.assets.get_bond_book_value(valuation_date = time_step)
        self.assets_market_value[-1] = self.assets.cash[time_step] + self.assets.EQ_market_value[time_step] + self.assets.get_bond_market_value(valuation_date = time_step)
        
        
        # =================================================
        # Réallocation après la distribution de la réchesse
        # =================================================
        
        self.reset_allocation(traj_i = traj_i, valuation_date = time_step)
        
        self.aft_rell.append(np.sum(self.assets.bonds.allocation_matrix[time_step],axis=1))
        
        # ===================================================================
        # Pour que le bilan reste carré après la réallocation
        # En même temps, l'allocation cible ne sera pas vérifiée parfaitement
        # ===================================================================
        # Dans tous les cas, delta BV = toutes les PMVR réalisées:
        
        self.assets.cash[time_step] -= self.assets.asset_income[time_step]['PMVR_hors_obligation'] + self.assets.asset_income[time_step]['PMVR_obligation_TF']
        
        
        self.update_capitalization_reserve(time_step)
        
        # ===================================================
        # Update Balance Sheets after reset assets allocation
        # ===================================================
        
        value = self.assets.asset_income[time_step]['PMVR_hors_obligation']
        if value >=0:
            self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_in', key_lv2 = 'PMVR_aft_res_allo', value = value)
            self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_out', key_lv2 = 'PMVR_aft_res_allo', value = np.absolute(self.assets.asset_income[time_step]['PMVR_obligation_TF'])) 
        else:
            self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_in', key_lv2 = 'PMVR_aft_res_allo', value = 0.)
            self.balance_sheets[time_step].update(key_lv1 = 'cash_flows_out', key_lv2 = 'PMVR_aft_res_allo', value = np.absolute(value+self.assets.asset_income[time_step]['PMVR_obligation_TF']))
        
        # =====================================================================================================        
        if self.assets.asset_income[time_step]['PMVR_obligation_TF']<0:
            if self.assets.asset_income[time_step]['PMVR_hors_obligation'] + self.assets.asset_income[time_step]['PMVR_obligation_TF'] >= 0:
                self.liabilities.technical_provision.profit_sharing_reserve[time_step] += self.assets.asset_income[time_step]['PMVR_hors_obligation']+self.assets.asset_income[time_step]['PMVR_obligation_TF']
                self.assets.asset_income[time_step]['PMVR_hors_obligation'] = 0
                self.assets.asset_income[time_step]['PMVR_obligation_TF'] = 0
            else:
                self.liabilities.own_fund[time_step] -= np.absolute(self.assets.asset_income[time_step]['PMVR_obligation_TF'] + self.assets.asset_income[time_step]['PMVR_hors_obligation'])
                self.assets.asset_income[time_step]['PMVR_hors_obligation'] = 0
                self.assets.asset_income[time_step]['PMVR_obligation_TF'] = 0
        
        elif self.assets.asset_income[time_step]['PMVR_obligation_TF']==0:
            if self.assets.asset_income[time_step]['PMVR_hors_obligation']>0:
                self.liabilities.technical_provision.profit_sharing_reserve[time_step] +=  self.assets.asset_income[time_step]['PMVR_hors_obligation']
                self.assets.asset_income[time_step]['PMVR_hors_obligation'] = 0
            else:
                self.liabilities.own_fund[time_step] -=  self.assets.asset_income[time_step]['PMVR_hors_obligation']
                self.assets.asset_income[time_step]['PMVR_hors_obligation'] = 0
        else:
            raise ValueError("After being updated the capitalisation reserve, PMVR_obligation_TF cannot be positive")
       
        # ===================================================
        # Update Balance Sheets after reset assets allocation
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'cash', value = self.assets.cash[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'equity', value = self.assets.EQ_book_value[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'assets', key_lv2 = 'bond', value = self.assets.get_bond_book_value(valuation_date = time_step))
        self.balance_sheets[time_step].update(key_lv1 = 'liabilities', key_lv2 = 'own_fund', value = self.liabilities.own_fund[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'liabilities', key_lv2 = 'capitalization_reserve', value = self.liabilities.capitalization_reserve[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'liabilities', key_lv2 = 'profit_sharing_reserve', value = self.liabilities.technical_provision.profit_sharing_reserve[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'liabilities', key_lv2 = 'PRE', value = self.liabilities.PRE[time_step])
        self.balance_sheets[time_step].update(key_lv1 = 'liabilities', key_lv2 = 'mathematical_provision', value = self.liabilities.technical_provision.mathematical_provision[time_step])
        
        # Consistency Check
        
        self.assets_book_value[time_step] = self.assets.cash[time_step] + self.assets.EQ_book_value[time_step] + self.assets.get_bond_book_value(valuation_date = time_step)
        if round(self.assets_book_value[time_step],0) != round(self.liabilities.technical_provision.mathematical_provision[-1],0):
            raise ValueError("La bilan comptable n'est pas carré !")        
        
        
    def treatment_end_scenario(self, valuation_date, deterministic = False):
        """
        Method : treatment_end_scenario

        Function : release technical provisions to the customers at the end of the scenarios

        Parameters :
            1. valuation_date:
                Type : int
                Function : valuation date
        """
        # ================================================================================================================
        # First make sure that the valuation date corresponds to the end of the scenario
        # If valuation_date corresponds to tome horizon, then release the techincal provision and sell corresponding assets
        # Reset mathematical provision
        # ================================================================================================================
        
        if valuation_date == self.time_horizon-1 :
             # Assets
                
             self.assets.delete_asset(amount = self.assets_book_value[-1], valuation_date = valuation_date, book_value = True)
             # Liabilities
             self.update_capitalization_reserve(valuation_date)
             self.liabilities.technical_provision.profit_sharing_reserve[-1] += 0.85 * (self.assets.asset_income[valuation_date]['PMVR_hors_obligation'] + self.assets.asset_income[valuation_date]['PMVR_obligation_TF'])
             self.liabilities.own_fund[-1] += 0.15 * (self.assets.asset_income[valuation_date]['PMVR_hors_obligation'] + self.assets.asset_income[valuation_date]['PMVR_obligation_TF'])

             if deterministic:
                self.liabilities.treatment_end_scenario_det()
             else:
                self.liabilities.treatment_end_scenario()
        
        

    # RN_scenario is a risk free scenario
    def compute_BEL_cashflows_scenario(self, traj_i, deterministic = False):
        """
            Method: commpute_BEL_cashflows_scenario

            Function: Compute and record cashflows paid at each time steps to customers on a given multi-variate RN scenario.

            Parameter:
                1. traj_i:
                    Type: int
                    Function: the i-th economic scenario.
        """
        for time_step in range(1,self.time_horizon):            
            self.assets.get_asset_income_and_EQ_value(traj_i = traj_i, valuation_date = time_step)
            self.assets.reset_asset_income_and_EQ_value(valuation_date = time_step)
            PMVL = self.assets.EQ_market_value[time_step] - self.assets.EQ_book_value[time_step]
            PVL = max(0, PMVL)
            MVL = max(0, -PMVL)
            PVL_hors_obligation = PVL
            Revenu = self.assets.ESG_RN.EQ_model.dividend_rate * self.assets.EQ_market_value[time_step - 1]
            income_dict = {'PMVL': PMVL, 'PVL': PVL, 'MVL': MVL, 'Revenu': Revenu, 'PVL_obligation_TF': 0, 'MVL_obligation_TF': 0, 'PMVR_hors_obligation': 0, 'PMVR_obligation_TF': 0, 'PVL_hors_obligation': PVL_hors_obligation}
            self.assets.asset_income.append(income_dict)
            
            # on ajoute une dimension au num_of_bonds et allocation_matrix si on appelle cette méthode en début de la période
            if len(self.assets.bonds.allocation_matrix) == time_step:
                self.assets.bonds.allocation_matrix.append(self.assets.bonds.allocation_matrix[-1])
            
                new_num=[]
                for k in range(7):
                    num_k = []
                    for t in range(time_step):
                        num_k.append(self.assets.bonds.num_of_bonds[k][t,:])
                    num_k.append(np.zeros(20))    
                    new_num.append(np.asarray(num_k))
                self.assets.bonds.num_of_bonds = new_num
            
            if len(self.assets.bonds.allocation_matrix) != time_step+1 or len(self.assets.bonds.num_of_bonds[0]) != time_step+1:
                raise ValueError('Bonds_Portofolio object has a wrong dimension in number of bonds')
            
            bonds_income = self.assets.bonds.valorize_bonds_income(valuation_date = time_step)
            self.assets.add_asset_income(bonds_income, valuation_date = time_step)

            self.additional_fund = 0
            self.distributed_wealth = 0
            #print(self.assets.number_EQ)
            self.treatment_begin_period(traj_i, time_step, deterministic = deterministic)
            self.treatment_end_period(traj_i = traj_i, time_step = time_step, deterministic = deterministic)
            
            self.treatment_end_scenario(valuation_date = time_step, deterministic = deterministic)        
        #Obtain trajectory dependent deflators
        #if deterministic:
        #    deflator = self.assets.ESG_RN.scenarios[traj_i]['Deflators'][0]
        #else:
        #    deflator = self.assets.ESG_RN.IR_model.discount_factors(self.assets.ESG_RN.IR_model.trajectory)
        deflator = self.assets.ESG_RN.IR_model.discount_factors(self.assets.ESG_RN.IR_model.trajectory)
        resu = np.dot(deflator[:self.time_horizon-1],np.array(self.liabilities.cash_flow_out) - np.array(self.liabilities.cash_flow_in))
        resu_own_fund = np.dot(deflator[self.time_horizon-1], (self.liabilities.own_fund[-1]))
        return resu, resu_own_fund, deflator[:self.time_horizon-1], self.liabilities.cash_flow_out, self.assets.ESG_RN.IR_model.trajectory




if __name__ == '__main__':
    # =================================================
    # Update assets_data_path and liabilities_data_path
    # =================================================
    Working_URL = r'C:\Users\FR015797\Documents\projet_ALM_valo\2.ALM_credit(working)_modified\Feuille_de_calcul_ALM(Working).xlsm'
    liabilities_data_path = 'Liability_Data_test.pkl'
    #liabilities_data_path = 'Liabilities_data0.pkl'
    market_name = 'EUR'
    # ============================
    # Set up time horizon
    # ============================
    time_horizon = int(xw.sheets['ESG'].range('D3').value)
    # =============================
    # Set up number of trajectories
    # =============================
    number_trajectories = int(xw.sheets['ESG'].range('D1').value)
    # ========================================
    # Define the market_equity_shock parameter
    # ========================================
    market_equity_shock = 0.
    # Define the deterministic state
    # ============================
    deterministic = False
        
        
    target_allocation = np.asarray([
        [50,50,50,50,50,60,70,80,100,100,0,0,0,0,0,0,0,0,0,0],
        [10,10,10,10,10,10,10,10,10,10,0,0,0,0,0,0,0,0,0,0],
        [6,5,4,3,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        ])
    # ============================
    # Creat an object named assets
    # ============================
    asset_data = Asset_data()
    asset_data.update(Working_URL)
    market = asset_data.get_list_market(market_name)
    corr_matrix = market.corr_matrix
    # ============================
    # Choose Equity model
    # ============================
    # 1. GBM_constant_volatility()
    Equity = EQ_model.GBM_constant_volatility()
    # ============================
    # Choose Short Rate model
    # ============================
    # 1. Hull_White_one_factor
    Interest_rate = IR_model.Hull_White_one_factor()
    # ============================
    # Choose Credi model
    # ============================
    # 1. JLT model
    credit = credit_model.JLT()
    # ===============================
    # Generate the economic scenarios
    # ===============================
    
    ESG = ESG_RN(asset_data,Interest_rate,Equity,credit)
    ESG.number_trajectories = number_trajectories
    ESG.update_time_horizon(time_horizon)
    ESG.add_corr_matrix(corr_matrix)
    ESG.add_market_name(market.name)
    ESG.calibrate_models()
    ESG.get_seed()
    
    for traj in range(number_trajectories):
        ESG.market_equity_shock = market_equity_shock
        ESG.get_scenario(traj_i=traj)
    # We call initial_deflator only when we want to study the sensibility of BEL in terms of zero coupon curve
    initial_deflator = ESG.asset_data.get_list_market(market_name).deflators
    # =================================================
    BEL = []
    VIF = []
    # ====================================================
    # Initialize liabilities_data used for the calculation
    # ====================================================
    for traj in range(number_trajectories):
        with open(liabilities_data_path, 'rb') as input:
            data = pickle.load(input)
        profit_sharing_reserve = 10000
        technical_provision = Technical_Provision()
        technical_provision.update(profit_sharing_reserve = profit_sharing_reserve)
        # ==========================================
        # Initialize Liabilities
        # ==========================================
        liabilities = Liabilities()
        own_fund = 100000
        PRE = 10000
        liabilities.update(capitalization_reserve = 10000, technical_provision = technical_provision, own_fund = own_fund, PRE = PRE, liabilities_data = data)
        # ==========================================
        # Initialize Assets
        # ==========================================
        generator = ESG
        # ==============================
        #         Bond
        # ==============================
        init_IR_curve = market.spot_rates    
            
        bonds_portfolio = Bonds_Portfolio(time_horizon=time_horizon,
                                          ESG_RN_scenarios_traj_i = ESG.scenarios[traj],
                                          target_allocation= target_allocation,
                                          init_IR_curve=init_IR_curve)
        
        bonds_portfolio.initialize_unit_bonds_book_value()
        bonds_portfolio.initialize_unit_bonds_market_value()
        
        assets = Assets(generator, bonds_portfolio)
        # ================================================================================================================================================
        # By default, the initial profit sharing rate is supposed to be TME_0, so there is no any dynamical surrenders during the first year of projection
        # If the initial profit sharing rate is known previously, then it can be modified within the following section
        # ================================================================================================================================================
        for mdlp in liabilities.liabilities_data.model_points:
            mdlp.profit_sharing_rate[0] = mdlp.TMG[mdlp.seniority]
        # ==========================================
        # Initialize ALM
        # ==========================================
        alm = ALM(assets, liabilities, time_horizon)
        # ==========================================
        # Set up asset allocation
        # ==========================================
        alm.assets.add_allocation(allocation_EQ = 0.2, allocation_bond = 0.7)
        value = alm.liabilities.technical_provision.mathematical_provision[0]
        alm.assets.initialize_EQ_value(value = assets.allocations['EQ']*value)
        alm.assets.bonds.initialize_allocation(amount=assets.allocations['bonds']*value)
        # ===============================
        # Set up deflators
        # ===============================
        alm.assets.deflators = alm.assets.ESG_RN.scenarios[traj]['Deflators']
        
        print(alm.assets.bonds.init_rating_based_deflators[0])        
        
        bond_market_value = alm.assets.get_bond_market_value(0)
        bond_book_value = alm.assets.get_bond_book_value(0)
        # ==============================
        #          Cash
        # ==============================
        alm.assets.add_cash(amount = alm.assets.allocations['cash'] * value)
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
        
        init_asset_value = alm.assets.cash[0] + alm.assets.EQ_market_value[0] + bond_market_value
        print("Asset0 = ",init_asset_value)
        # ==================================
        # Main Part
        # Compute BEL
        # ==================================
        resu, resu_own_fund, discount_factor, cash_flow_out, trajectory = alm.compute_BEL_cashflows_scenario(traj_i = traj, deterministic = deterministic)
        BEL.append(resu)
        VIF.append(resu_own_fund)

    print("BEL =: %s" %np.mean(BEL))
    print("VIF =: %s" %np.mean(VIF))    
    
    plt.plot(range(time_horizon), alm.liabilities.technical_provision.mathematical_provision[:time_horizon])
    plt.title('Evolution of mathematical provision')
    plt.show()
    # ===============================
    # Save Balance Sheets to excel
    # ===============================
    # At the begining of period
    # ===============================
    cash_0_list = [alm.balance_sheets[time_step].get_value('assets','cash_0') for time_step in range (1, alm.time_horizon)]
    equity_market_0_list = [alm.balance_sheets[time_step].get_value('assets', 'equity_market_value_0') for time_step in range(1, alm.time_horizon)]
    equity_bond_0_list = [alm.balance_sheets[time_step].get_value('assets', 'equity_book_value_0') for time_step in range(1, alm.time_horizon)]
    bond_market_0_list = [alm.balance_sheets[time_step].get_value('assets', 'bond_market_value_0') for time_step in range(1, alm.time_horizon)]
    bond_book_0_list = [alm.balance_sheets[time_step].get_value('assets', 'bond_book_value_0') for time_step in range(1, alm.time_horizon)]
    number_bonds_0_list = [sum(sum(alm.assets.bonds.allocation_matrix[time_step])) for time_step in range(1, alm.time_horizon)]
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
    number_bonds_1_list = [sum(sum(alm.assets.bonds.allocation_matrix[time_step])) for time_step in range(1, alm.time_horizon)]
    bond_coupon_1_list = [alm.balance_sheets[time_step].get_value('assets', 'bond_coupon_value_1') for time_step in range(1, alm.time_horizon)]
        
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
    number_bonds_2_list = [sum(sum(alm.assets.bonds.allocation_matrix[time_step])) for time_step in range(1, alm.time_horizon)]
        
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
    number_bonds_3_list = [sum(sum(alm.assets.bonds.allocation_matrix[time_step])) for time_step in range(1, alm.time_horizon)]
        
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
    number_bonds_4_list = [sum(sum(alm.assets.bonds.allocation_matrix[time_step])) for time_step in range(1, alm.time_horizon)]
        
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
    number_bonds_5_list = [sum(sum(alm.assets.bonds.allocation_matrix[time_step])) for time_step in range(1, alm.time_horizon)]
        
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
    cash_list = [alm.balance_sheets[time_step].get_value('assets','cash') for time_step in range (1, alm.time_horizon)]
    equity_list = [alm.balance_sheets[time_step].get_value('assets', 'equity') for time_step in range(1, alm.time_horizon)]
    bond_list = [alm.balance_sheets[time_step].get_value('assets', 'bond') for time_step in range(1, alm.time_horizon)]
    
    own_fund_list = [alm.balance_sheets[time_step].get_value('liabilities', 'own_fund') for time_step in range(1, alm.time_horizon)]
    capitalization_reserve_list = [alm.balance_sheets[time_step].get_value('liabilities', 'capitalization_reserve') for time_step in range(1, alm.time_horizon)]
    PPE_list = [alm.balance_sheets[time_step].get_value('liabilities', 'profit_sharing_reserve') for time_step in range(1, alm.time_horizon)]
    PRE_list = [alm.balance_sheets[time_step].get_value('liabilities', 'PRE') for time_step in range(1, alm.time_horizon)]
    PM_list = [alm.balance_sheets[time_step].get_value('liabilities', 'mathematical_provision') for time_step in range(1, alm.time_horizon)]
    
    prime_list = [alm.balance_sheets[time_step].get_value('cash_flows_in', 'premium') for time_step in range(1, alm.time_horizon)]
    surrender_list = [alm.balance_sheets[time_step].get_value('cash_flows_out', 'surrender_value') for time_step in range(1, alm.time_horizon)]
    mortality_list = [alm.balance_sheets[time_step].get_value('cash_flows_out', 'mortality_value') for time_step in range(1, alm.time_horizon)]
    revenu_list = [alm.balance_sheets[time_step].get_value('cash_flows_in', 'revenu') for time_step in range(1, alm.time_horizon)]
    
    amortizing_in_list = [alm.balance_sheets[time_step].get_value('cash_flows_in', 'amortizing_value') for time_step in range(1, alm.time_horizon)]
    amortizing_out_list = [alm.balance_sheets[time_step].get_value('cash_flows_out', 'amortizing_value') for time_step in range(1, alm.time_horizon)]
    
    Cash_Return_in = [alm.balance_sheets[time_step].get_value('cash_flows_in', 'Cash_Return') for time_step in range(1, alm.time_horizon)]
    Cash_Return_out = [alm.balance_sheets[time_step].get_value('cash_flows_out', 'Cash_Return') for time_step in range(1, alm.time_horizon)]
    
    PMVR_in_list = [alm.balance_sheets[time_step].get_value('cash_flows_in', 'PMVR') for time_step in range(1, alm.time_horizon)]
    PMVR_out_list = [alm.balance_sheets[time_step].get_value('cash_flows_out', 'PMVR') for time_step in range(1, alm.time_horizon)]
    
    PMVR_in_rachat_list = [alm.balance_sheets[time_step].get_value('cash_flows_in', 'PMVR_rachat') for time_step in range(1, alm.time_horizon)]
    PMVR_out_rachat_list = [alm.balance_sheets[time_step].get_value('cash_flows_out', 'PMVR_rachat') for time_step in range(1, alm.time_horizon)]
    
    delta_PRE_in_list = [alm.balance_sheets[time_step].get_value('cash_flows_in', 'delta_PRE') for time_step in range(1, alm.time_horizon)]
    delta_PRE_out_list = [alm.balance_sheets[time_step].get_value('cash_flows_out', 'delta_PRE') for time_step in range(1, alm.time_horizon)]
    
    PMVR_in_deces_list = [alm.balance_sheets[time_step].get_value('cash_flows_in', 'PMVR_deces') for time_step in range(1, alm.time_horizon)]
    PMVR_out_deces_list = [alm.balance_sheets[time_step].get_value('cash_flows_out', 'PMVR_deces') for time_step in range(1, alm.time_horizon)]
    
    PMVR_aft_res_allo_in_list  = [alm.balance_sheets[time_step].get_value('cash_flows_in', 'PMVR_aft_res_allo') for time_step in range(1, alm.time_horizon)]
    PMVR_aft_res_allo_out_list = [alm.balance_sheets[time_step].get_value('cash_flows_out', 'PMVR_aft_res_allo') for time_step in range(1, alm.time_horizon)]    
    
    MVR__TF_rachat_list = [alm.balance_sheets[time_step].get_value('cash_flows_out', 'MVR_TF_residuelle_rachat') for time_step in range(1, alm.time_horizon)]
    
    MVR__TF_deces_list = [alm.balance_sheets[time_step].get_value('cash_flows_out', 'MVR_TF_residuelle_deces') for time_step in range(1, alm.time_horizon)]
    
    sum_cash_list = [alm.balance_sheets[time_step].get_cash_flow() for time_step in range(1, alm.time_horizon)]
    
    delta_own_fund_list = [alm.balance_sheets[time_step].get_value('liabilities', 'own_fund') - alm.balance_sheets[time_step].get_value('liabilities', 'own_fund_0') for time_step in range(1, time_horizon)]
    delta_PPE_list = [alm.balance_sheets[time_step].get_value('liabilities', 'profit_sharing_reserve') - alm.balance_sheets[time_step].get_value('liabilities', 'profit_sharing_reserve_0') for time_step in range(1, alm.time_horizon)]
    delta_Kpi_list = [alm.balance_sheets[time_step].get_value('liabilities', 'capitalization_reserve') - alm.balance_sheets[time_step].get_value('liabilities', 'capitalization_reserve_0') for time_step in range(1, alm.time_horizon)]
    delta_PM_list = [alm.balance_sheets[time_step].get_value('liabilities', 'mathematical_provision') - alm.balance_sheets[time_step].get_value('liabilities', 'mathematical_provision_0') for time_step in range(1, alm.time_horizon)]
    delta_PRE_list = [alm.balance_sheets[time_step].get_value('liabilities', 'PRE') - alm.balance_sheets[time_step-1].get_value('liabilities','PRE') for time_step in range(1, alm.time_horizon)]
    delta_liabilities_list = [x + y + w for x,y,w in zip(delta_own_fund_list, delta_PPE_list, delta_PM_list)]

    filename = 'Balance_Sheets_test.xls'
    # =========================
    # At the begining of period
    # =========================
    assets_0_dict = collections.OrderedDict()
    assets_0_dict['cash_0'] = cash_0_list
    assets_0_dict['equity_market_0'] = equity_market_0_list
    assets_0_dict['equity_book_0'] = equity_bond_0_list
    assets_0_dict['bond_market_0'] = bond_market_0_list
    assets_0_dict['bond_book_0'] = bond_book_0_list
    assets_0_dict['number_bonds_0'] = number_bonds_0_list
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
    assets_1_dict['number_bonds_1'] = number_bonds_1_list
    assets_1_dict['bond_coupon_1'] = bond_coupon_1_list
    
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
    assets_2_dict['number_bonds_2'] = number_bonds_2_list
    
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
    assets_3_dict['number_bonds_3'] = number_bonds_3_list
    
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
    assets_4_dict['number_bonds_4'] = number_bonds_4_list
    
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
    assets_5_dict['number_bonds_5'] = number_bonds_5_list
    
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
    et.output_excel(file_name = filename, dictionary = balance_sheet)

    
    """
    # ===============================
    # Save Balance Sheets to excel
    # ===============================
    # At the begining of period
    # ===============================
    
    # ===============================
    ## Actif
    # ===============================
    EQ_BV_lst = [alm.assets.EQ_book_value[time_step] for time_step in range (0, alm.time_horizon)]
    Bond_BV_lst = [alm.track_bonds_book_value[time_step] for time_step in range (0, alm.time_horizon)]
    cash_lst = [alm.assets.cash[time_step] for time_step in range (0, alm.time_horizon)]
    asset_BV_lst = [alm.assets_book_value[time_step] for time_step in range (0, alm.time_horizon)]
    
    # ===============================
    # Passif
    # ===============================    
    own_fund_lst = [alm.liabilities.own_fund[time_step] for time_step in range(0, alm.time_horizon)]
    capitalization_reserve_lst = [alm.liabilities.capitalization_reserve[time_step] for time_step in range(0, alm.time_horizon)]
    PRE_lst = [alm.liabilities.PRE[time_step] for time_step in range(0, alm.time_horizon)]
    PPE_lst = [alm.liabilities.technical_provision.profit_sharing_reserve[time_step] for time_step in range(0, alm.time_horizon)]
    PM_lst = [alm.liabilities.technical_provision.mathematical_provision[time_step] for time_step in range(0, alm.time_horizon)]
        
    # ===============================
    # Compte de résultat
    # ===============================
    
    ## Résultat de souscription
    
    ## Résultat de gestion
    
    ## Résultat financier
    
    
    # chiffre d'affaire
    premium_lst = [alm.track_premium[time_step] for time_step in range(0, alm.time_horizon)]
    
    # sinistres payés
    surrender_lst = [alm.track_surrender_value[time_step] for time_step in range(0, alm.time_horizon)]
    mortality_lst = [alm.track_mortality_value[time_step] for time_step in range(0, alm.time_horizon)]
    
    # produit financier
    financial_prod_lst = [alm.track_fin_prod[time_step] for time_step in range(0, alm.time_horizon)]  
    additional_fund_lst = [alm.track_addtional_fund[time_step] for time_step in range(0, alm.time_horizon)]    
    
    # IT + PB
    distributed_wealth_lst = [alm.track_distributed_wealth[time_step] for time_step in range(0, alm.time_horizon)]
    contractual_margin = [alm.liabilities.contractual_margin[time_step]  for time_step in range(0, alm.time_horizon)]
    profit_sharing_reserve_lst = [alm.liabilities.technical_provision.profit_sharing_reserve[time_step] for time_step in range(0, alm.time_horizon)]
    
    # PRE
    PRE_lst = [alm.liabilities.PRE[time_step] for time_step in range(0, alm.time_horizon)]
    
    delta_PRE_lstt = [PRE_lst[time_step]-PRE_lst[time_step-1] for time_step in range (1, alm.time_horizon) ]
    delta_PRE_lst = [0]+delta_PRE_lstt    
    
    filename2 = 'resumed_balance_sheet.xls'
    # =========================
    # Bilan
    # =========================
    Assets = collections.OrderedDict()
    Assets['Cash'] = cash_lst
    Assets['Equity (BV)'] = EQ_BV_lst
    Assets['Bonds (BV)'] = Bond_BV_lst
    Assets['Assets'] = asset_BV_lst
    
    
    Liabilities= collections.OrderedDict()
    Liabilities['Own fund'] = own_fund_lst
    Liabilities['Capitalisation Reserve'] = capitalization_reserve_lst
    Liabilities['PRE'] = PRE_lst
    Liabilities['Profit sharing reserve'] = PPE_lst
    Liabilities['Mathematical Provision'] = PM_lst
    # =========================
    # Compte de résultat
    # =========================
    Turnover = collections.OrderedDict()
    Turnover['Premium'] = premium_lst

    Sinister = collections.OrderedDict()
    Sinister['Surrender'] = surrender_lst
    Sinister['Mortality'] = mortality_lst
    
    
    
    Add_fund =  collections.OrderedDict()
    Add_fund['Additional fund'] = additional_fund_lst
    
    delta_PRE =  collections.OrderedDict()
    delta_PRE['delta PRE'] = delta_PRE_lst
    
    Distr_wealth = collections.OrderedDict()
    Distr_wealth['Financial revenu'] = financial_prod_lst
    Distr_wealth['distributed wealth'] = distributed_wealth_lst
    Distr_wealth['contractual margin'] = contractual_margin
    Distr_wealth['profit sharing reserve'] = profit_sharing_reserve_lst
    
    
    # ======================
    # Build Up Balance Sheet
    # ======================
    resumed_balance_sheet = collections.OrderedDict()
    resumed_balance_sheet['Assets'] = Assets
    resumed_balance_sheet['Liabilities'] = Liabilities
    resumed_balance_sheet['Turnover'] = Turnover
    resumed_balance_sheet['Sinister'] = Sinister
    resumed_balance_sheet['Dsitributed wealth'] = Distr_wealth
    resumed_balance_sheet['Additional fund'] = Add_fund
    resumed_balance_sheet['Delta PRE'] = delta_PRE
    
    
    et.output_excel(file_name = filename2, dictionary = resumed_balance_sheet)"""