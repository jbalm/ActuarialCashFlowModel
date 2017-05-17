## Progam packages
from ..market_environment.Market_Environment import Market_Environment
from ..core_math import credit_pickles
## Python packages
from xlrd import open_workbook
import xlrd
import xlwings as xw
import numpy as np

class Asset_data(object):
    """
        Objective:
        ==========
        This class is meant to build up the market environment database

        Attributes:
        ===========

        1. market_environments:
            Type: list/array-like
            Function: collection of the market environments which is characterized by market's name.

        Methods:
        ========

        1. add_list_market:

        2. get_list_market:

        3. update:

    """

    def __init__(self):
        self.market_environments = {}

    def add_list_market(self, key, list_object):
        """
            Method: add_list_market

            Function: add a new market_environment object into the list market_environments

            Parameters:
                1. key:
                    Type: string
                    Function: market's name
                2. list_object:
                    Type: instance of Market_Environment class
                    Function: new market environment
        """
        self.market_environments[key] = list_object

    def get_list_market(self, key):
        """
            Method: get_list_market

            Function: return the market_environment object given the market's name

            Parameters:
                1. key:
                    Type: string
                    Function: market's name
        """
        return self.market_environments[key]

    def update(self, Working_URL=None):
        """
            Method: update

            Function: Update data from an excel file named "Market_Environment.xls".

            Parameters:
                1. path:
                    Type: string
                    Function: a single directory or a file name (By default, path = 'Market_Environment.xls' and the excel file must be placed in the same folder as the main executed file)

        """
        markets = []
        # ============================================================================================
        # Euro market
        # ============================================================================================
        market = Market_Environment()
        # ============================================================================================
        # xlwings 0.10.1 version
        market.name = xw.sheets['Market_Environment'].range('C3').value
        market.vol_IR_rate = xw.sheets['Market_Environment'].range('C4').value
        market.speed_IR_rate = xw.sheets['Market_Environment'].range('C5').value
        market.init_stock = xw.sheets['Market_Environment'].range('C6').value
        market.vol_stock = xw.sheets['Market_Environment'].range('C7').value
        market.dividend_rate = xw.sheets['Market_Environment'].range('C8').value
        market.JLT_mu = xw.sheets['Market_Environment'].range('C9').value
        market.JLT_alpha = xw.sheets['Market_Environment'].range('C10').value
        market.JLT_sigma = xw.sheets['Market_Environment'].range('C11').value
        market.JLT_pi = xw.sheets['Market_Environment'].range('C12').value
        market.recovery_rate = xw.sheets['Market_Environment'].range('C13').value
        # =================================================================
        sheet1 = open_workbook(r'Feuille_de_calcul_ALM(Working).xlsm').sheet_by_name("Market_Environment")
        sheet2 = open_workbook(r'Feuille_de_calcul_ALM(Working).xlsm').sheet_by_name("Correlation_Matrix")
        # ============================================================================================
        # get spot rates
        spot_rates = []
        for row in range(2, sheet1.nrows):
            if (sheet1.cell_type(row, 5) != xlrd.XL_CELL_EMPTY):
                spot_rates.append(sheet1.cell(row, 5).value)
        market.spot_rates = spot_rates
        # ============================================================================================
        # get deflators
        deflators = []
        for row in range(2, sheet1.nrows):
            if (sheet1.cell_type(row, 11) != xlrd.XL_CELL_EMPTY):
                deflators.append(sheet1.cell(row, 11).value)
        market.deflators = deflators
        # ============================================================================================
        # get correlation matrix
        nb_rows = sheet2.nrows
        nb_cols = sheet2.ncols

        for row in range(1, nb_rows):
            arr = []
            for col in range(1, nb_cols):
                value = (sheet2.cell(row, col).value)
                arr.append(value)
            market.corr_matrix.append(arr)
        market.corr_matrix = np.asarray(market.corr_matrix)
        # ======== Check if the correlation matrix is symmetric =====================================
        assert np.allclose(market.corr_matrix, market.corr_matrix.T), "Correlation matrix must be symmetric"
        # ============================================================================================
        markets.append(market)
        # ============================================================================================
        for market in markets:
            key = market.name
            self.add_list_market(key, market)
            
        # ============================================================================================
        # Get Credit data
        # ============================================================================================
        market.historical_transition_matrix = credit_pickles.get_historical_transition_matrix(Working_URL)
        market.spread_list, market.col_index, market.row_index = credit_pickles.get_spread(Working_URL)
        market.prix_marche, market.coupon_marche = credit_pickles.get_prices(Working_URL)


#if __name__ == '__main__':
#    Working_URL = r'..\..\Feuille_de_calcul_ALM(Working).xlsm'
#    
#    data = Asset_data()
#    data.update(Working_URL)
#    market = data.get_list_market('EUR')
#    print(market)
#    plt.plot(range(len(market.spot_rates)), market.spot_rates, label = 'Yield Curve')
#    plt.title('Test Asset_data.py')
#    plt.legend()
#    plt.show()
#    print("Historical Transition Matrix = ", np.round(np.asarray(market.historical_transition_matrix),4))