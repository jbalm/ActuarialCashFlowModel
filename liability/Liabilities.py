## Progam packages
from .Technical_Provision import Technical_Provision
from .liability_data.Liabilities_data_m import Liabilities_data_m
from .liability_data.Liabilities_data import Liabilities_data
from ..core_math import excel_toolbox as et
## Python packages
import numpy as np
import pickle

class Liabilities(object):
    """
    objective:
    ==========
    This class is meant to implement the liabilities cash flows at each time steps

    Attributes:
    ==========

    1. capitalization_reserve:
        Type: array
        Function: corresponding list of capitalization reserve values over time horizon

    2. liabilities_data:
        Type: instance of Liabilities_data class
         Function: see Liabilities_data class

    3. technical_provision:
        Type: instance of Technical_Provision class
        Function: see Technical_Provision class

    4. own_fund:
        Type: array
        Function: corresponding list of own fund values over time horizon

    5. cash_flow_in:
        Type: array
        Function: corresponding list of cash flow in values over time horizon

    6. cash_flow_out:
        Type: array
        Function: corresponding list of cash flow out values over time horizon

    7. contractual_margin:
        Type: array
        Function: corresponding list of contractual margin values over time horizon

    Methods:
    ========

    1. update:
            updates all current information given the liabilities_data.

    2. update_cash_flow_in_out:

    3. update_mathematical_provision:

    4. update_capitalization_reserve:

    5. update_own_fund:
    """

    def __init__(self):
        # Stock
        self.capitalization_reserve = []
        self.technical_provision = None
        self.own_fund = []
        # ====================================================================================================
        # In this version, we take into account the presence of the provision for risk on the equity portfolio
        # We name it PRE = Provision pour Risque d'Exigibilité
        # ====================================================================================================
        self.PRE = []
        self.liabilities_data = None
        # Flux
        self.cash_flow_in = []
        self.cash_flow_out = []
        self.contractual_margin = [0]
        # ==============================

    # Attention!!! Just update once at the beginning
    def update(self, capitalization_reserve, technical_provision, own_fund, PRE, liabilities_data):
        """
            Method: update

            Function: initialise all Liabilities attributes here given the liabilities_data

            Parameters:
                1. capitalization_reserve:
                    Type: float (positive only)
                    Function: capitalization reserve's initial value.

                2. technical_provision:
                    Type: instance of Technical_Provision class
                    Function: initialise all technical provision attributes given liabilities_data.

                3. own_fund:
                    Type: float (positive only)
                    Function: own fund's initial value.

                4. liabilities_data:
                    Type: instance of Liabilities_data class
                    Function: see Liabilities_data for more details
        """
        # Stock
        self.capitalization_reserve.append(capitalization_reserve)
        self.technical_provision = technical_provision
        self.own_fund.append(own_fund)
        self.PRE.append(PRE)
        # Inputs
        self.liabilities_data = liabilities_data
        self.technical_provision.mathematical_provision.append(sum(mdlp.mathematical_provision[0] for mdlp in self.liabilities_data.model_points))
        if len(self.technical_provision.profit_sharing_reserve) == 0:
            self.technical_provision.profit_sharing_reserve.append(0)


    # asset_income is an attribut (a list) of class Asset which contains the realised gain and loss values
    def update_cash_flow_in_out(self, valuation_date):
        """
            Method: update_cash_flow_in_out

            Function: update periodic premium and surrender value of each model points given the liabilities_data at each time steps as well as liabilities cash flows.

            Parameter: None
        """
        for mdlp in self.liabilities_data.model_points:
            mr = np.array(mdlp.mortality_rate)
            sr = 1 - mr
            sr_mdlp = np.prod(sr[int(mdlp.average_age)-valuation_date:int(mdlp.average_age)])
            mdlp.cash_flow_in.append(mdlp.premium * mdlp.number_contract * sr_mdlp)
            mdlp.cash_flow_out.append(mdlp.dynamical_lapse_rate * mdlp.mathematical_provision[-1] * sr_mdlp)

        self.cash_flow_in.append(sum(mdlp.cash_flow_in[-1] for mdlp in self.liabilities_data.model_points))
        self.cash_flow_out.append(sum(mdlp.cash_flow_out[-1] for mdlp in self.liabilities_data.model_points))

    def update_mathematical_provision(self):
        """
            Method: update_mathematical_provision

            Function: update the mathematical provision value of each model points given the profit sharing rate and its total value after revaluation.

            Parameter: None
        """
        for mdlp in self.liabilities_data.model_points:
            resu = mdlp.mathematical_provision[-1]*(1 + mdlp.profit_sharing_rate[-1]) + (mdlp.cash_flow_in[-1] - mdlp.cash_flow_out[-1])*np.sqrt(1+mdlp.profit_sharing_rate[-1])
            mdlp.mathematical_provision.append(resu)
        self.technical_provision.mathematical_provision.append(sum(mdlp.mathematical_provision[-1] for mdlp in self.liabilities_data.model_points))
        #for mdlp in self.liabilities_data.model_points:
        #    mdlp.allocation = (mdlp.mathematical_provision[-1])/self.technical_provision.mathematical_provision[-1]

    def update_mathematical_provision_TMG(self):
        """
            Method: update_mathematical_provision

            Function: update the mathematical provision value of each model points given the profit sharing rate and its total value after revaluation.

            Parameter: None
        """
        for mdlp in self.liabilities_data.model_points:
            resu = mdlp.mathematical_provision[-1]*(1 + mdlp.TMG[mdlp.seniority]) + (mdlp.cash_flow_in[-1] - mdlp.cash_flow_out[-1])*np.sqrt(1+mdlp.TMG[mdlp.seniority])
            mdlp.mathematical_provision.append(resu)
        self.technical_provision.mathematical_provision.append(sum(mdlp.mathematical_provision[-1] for mdlp in self.liabilities_data.model_points))

    def update_own_fund(self, valuation_date, additional_fund = 0):
        """
            Method: update_own_fund

            Function: update the contractual margin value and the own fund value at each time steps after revaluation.

            Parameter:
                1. additional_fund:
                    Type: float
                    Function: a quantity of money taken from the own fund when we do not have enough richness to fund distributed wealth.
        """
        # Compute contractual margin
        somme = 0
        for mdlp in self.liabilities_data.model_points:
            somme += mdlp.mathematical_provision[-2]*(1 + mdlp.profit_sharing_rate[-1] + mdlp.margin_rate) + (mdlp.cash_flow_in[-1] - mdlp.cash_flow_out[-1]) * np.sqrt(1 + mdlp.profit_sharing_rate[-1] + mdlp.margin_rate)
        self.contractual_margin.append(somme - self.technical_provision.mathematical_provision[-1])
        # compute own_fund
        self.own_fund[valuation_date] = self.own_fund[valuation_date] + self.contractual_margin[-1] - additional_fund


    def treatment_end_scenario(self):
        """
            Method : treatment_end_scenario

            Function : release technical provision as a single cash-flow at the end of projections.
        """
        #Reset technical provisions within mdlp
        for mdlp in self.liabilities_data.model_points:
            mdlp.cash_flow_out[-1] += mdlp.mathematical_provision[-1]
            mdlp.mathematical_provision[-1] = 0

        #Increase the amount of last cash-flow out by the amount of technical provisions
        self.cash_flow_out[-1] += self.technical_provision.mathematical_provision[-1] + self.technical_provision.profit_sharing_reserve[-1]
        #self.cash_flow_out[-1] += self.technical_provision.mathematical_provision[-1]
        self.technical_provision.mathematical_provision[-1] = 0
        self.technical_provision.profit_sharing_reserve[-1] = 0

    def treatment_end_scenario_det(self):
        """
            Method : treatment_end_scenario

            Function : release technical provision as a single cash-flow at the end of projections.
        """
        #Reset technical provisions within mdlp
        for mdlp in self.liabilities_data.model_points:
            mdlp.cash_flow_out[-1] += mdlp.mathematical_provision[-1]
            mdlp.mathematical_provision[-1] = 0

        #Increase the amount of last cash-flow out by the amount of technical provisions
        self.cash_flow_out[-1] += self.technical_provision.mathematical_provision[-1]
        self.technical_provision.mathematical_provision[-1] = 0




#
#if __name__ == '__main__':
#    # ======================
#    # Initial data
#    # ======================
#    PPE = 5000
#    Reserve_de_Kpi = 5000
#    Fond_propre = 10000
#    additional_fund = 0
#    PRE = 0
#    # ========================
#    # Update Liabilities_data
#    # ========================
#    liabilities_data_path = 'Liability_Data_test.pkl'
#    #liabilities_data_path = 'Liabilities_data.pkl'
#    # ====================================================
#    # Initialize liabilities_data used for the calculation
#    # ====================================================
#    with open(liabilities_data_path, 'rb') as input:
#        liabilities_data = pickle.load(input)
#    
#    PT = Technical_Provision()
#    PT.update(profit_sharing_reserve = PPE)
#    # ========================
#    # Initialize Liabilities()
#    # ========================
#    Passif = Liabilities()
#    # =====================
#    # Update initial data
#    # =====================
#    Passif.update(capitalization_reserve = Reserve_de_Kpi, technical_provision = PT, own_fund = Fond_propre, PRE = PRE, liabilities_data = liabilities_data)
#    # ==================
#    # Test object Passif
#    # ==================
#    print('Test class Liabilities')
#    print('Financial Statement \n'
#          '=================== \n'
#          '\n'
#          'I. Technical Provision \n'
#          '   =================== \n'
#          '   1. Mathematical Provision: {0:.2f} \n'
#          '   2. Profit Sharing Reserve: {1:.2f} \n'
#          'II. Capitalization Reserve: {2:.2f} \n'
#          'III PRE: {3: .2f} \n'
#          'IV. Own Fund: {4:.2f} \n'
#          .format(Passif.technical_provision.mathematical_provision[-1]
#                ,Passif.technical_provision.profit_sharing_reserve[-1]
#                ,Passif.capitalization_reserve[-1]
#                ,Passif.PRE[-1]
#                  ,Passif.own_fund[-1]))
#    #Passif.liabilities_data.affiche()
#    # ===================================
#    # Test update_cash_flow_in_out method
#    # ===================================
#    #Passif.update_cash_flow_in_out(valuation_date = 0)
#    #print('Test update_cash_flow_in_out() method')
#    #print('cash_flow_in = ', Passif.cash_flow_in[-1])
#    #print('cash_flow_out = ', Passif.cash_flow_out[-1])
#    # ========================================
#    # Test update_mathematical_provision
#    # ========================================
#    #for mdlp in Passif.liabilities_data.model_points:
#    #    mdlp.profit_sharing_rate[0] = 0.005
#    #Passif.update_mathematical_provision()
#    # =========================================
#    # Test update_capitalization_reserve method
#    # =========================================
#    #Passif.update_capitalization_reserve(asset_income, 0)
#    # ===========================
#    # Test update_own_fund method
#    # ===========================
#    #Passif.update_own_fund(additional_fund)
#    # =========================================
#    # Test update_profit_sharing_reserve method
#    # =========================================
#    #distributed_wealth = 100000
#    #Passif.update_profit_sharing_reserve(distributed_wealth = distributed_wealth)
#    #print('Financial Statement \n'
#    #      '=================== \n'
#    #      '\n'
#    #      'I. Technical Provision \n'
#    #      '   =================== \n'
#    #      '   1. Mathematical Provision: {0:.2f} \n'
#    #      '   2. Profit Sharing Reserve: {1:.2f} \n'
#    #      'II. Capitalization Reserve: {2:.2f} \n'
#    #      'III. Own Fund: {3:.2f} \n'
#    #      .format(Passif.technical_provision.mathematical_provision[-1]
#    #            ,Passif.technical_provision.profit_sharing_reserve[-1]
#    #            ,Passif.capitalization_reserve[-1]
#    #              ,Passif.own_fund[-1]))
