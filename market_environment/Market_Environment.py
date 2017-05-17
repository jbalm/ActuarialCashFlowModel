## Python packages
from xlrd import open_workbook
import matplotlib
import matplotlib.pyplot as plt

class Market_Environment(object):
    """
        Definition: Market Environment
        ==============================
        Market Environment refers to factors and forces that affect the ability to build and maintain a successful operation of an insurance company.

        Objective:
        ==========
        This class provides a general framework of a market environment relevant for asset valuation.

        Attributes:
        ===========

        1. name:
            Type: string
            Function: market environment's name.

        2. vol_rate:
            Type: float (positive only)
            Function: constant volativity of the interest rate

        3. mean_rate:
            Type: float (positive only)
            Function: constant long term mean level of the interest rate (only for the Vasicek model)

        4. speed_rate:
            Type: float (positive only)
            Function: constant speed of reversion.

        5. init_stock:
            Type: float
            Function: normalised initial stock value (e.g. init_stock = 1)

        6. vol_stock:
            Type: float (positive only)
            Function: constant volativity coefficient in the geometric brownian motion model which is commonly used for modeling in finance

        7. dividend_rate:
            Type: float (positive only)
            Function: dividend return rate

        8. coupon_rate:
            Type: float (positive only)
            Function: coupon interest rate

        9. nominal: (This attribute will be deleted in the next version)
            Type: float (positive only)
            Function: bond's nominal value or face value

        10. bond_maturity: (This attribute will be deleted in the next version)
            Type: int
            Function: bond's maturity.

        11. curve: (This attribute will be deleted in the next version)
            Type: list/array-like
            Function: current zero-coupon curve if it is given

        12. time_range:
            Type: list/array-like
            Function: time range of the current zero-coupon curve

        13. maturity_range:
            Type: list/array-like
            Function: time range of the current forward yield curve

        14. forward_rate:
            Type: list/array-like
            Function: current forward yield curve

        Method:
        =======

        1. __str__:

    """

    def __init__(self):
        self.name = None
        self.vol_IR_rate = None
        self.speed_IR_rate = None
        
        self.init_stock = None
        self.vol_stock = None
        self.dividend_rate = None
        
        self.JLT_mu = None
        self.JLT_alpha = None
        self.JLT_sigma = None
        self.JLT_pi = None  
        self.recovery_rate = None
        
        self.spread_list = None
        self.prix_marche = None
        self.coupon_marche = None        
        self.spot_rates = []
        self.corr_matrix = []
        self.deflators = []
        # ==========================================
        self.col_index = None
        self.row_index = None

    # Affiche les infos d'un marché financier
    def __str__(self):
        """
            Method: __str__

            Function: Print all the current market environment data

            Parameters: None
        """
        return("Market Environment:\n"
                " \n"
                "  Market Name = {0}\n"
                " \n"
                "  Short Rate Parameters\n"
                "  =====================\n"
                "  Volativity = {1}\n"
                "  Speed of Reversion = {2}\n"
                " \n"
                "  Stock Parameters\n"
                "  ================\n"
                "  Initial Value = {3}\n"
                "  Volativity = {4}\n"
                "  Dividend Rate = {5}\n"
                " \n"
                "  Credit Parameters\n"
                "  ===============\n"
                "  mu = {6}\n"
                "  alpha = {7}\n"
                "  sigma = {8}\n"
                "  pi_0 = {9}\n"
                "  Recovery Rate = {10}\n"
                .format(self.name, 
                        self.vol_IR_rate,self.speed_IR_rate, 
                        self.init_stock, self.vol_stock, self.dividend_rate, 
                        self.JLT_mu, self.JLT_alpha, self.JLT_sigma, self.JLT_pi,
                        self.recovery_rate))
