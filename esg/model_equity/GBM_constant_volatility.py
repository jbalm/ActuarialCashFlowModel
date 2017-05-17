# For testing the model, we need to import a real spot rate trajectory
from ..generator_correlated_variables import generator_correlated_variables
from .EQ_model_classes import EQ_model_base

# =========================================================
#        Geometry Brownian Motion Constant Volatility Model
# =========================================================
class GBM_constant_volatility(EQ_model_base):
    """
        Objective:
        ==========
        This class is meant to generate simulated paths based on the Black-Scholes-Merton geometric Brownnian motion model with constant volatility

        Attributes
        ==========

        1. corr_matrix:
            Type: 2 dimensional array
            Function: correlated geometric Brownnian motions matrix

        2. time_horizon:
            Type: int
            Function: time horizon of the simulation for the overall system analysis

        3. num_instrument:
            Type: int
            Function: number of investment instruments in the Portfolio (By default, this parameter is None)

        4. fixed_seed:
            Type: int
            Function: to define the random state

        5. market_name:
            Type: string
            Function: market environment's name

        Methods
        =======

        1. add_short_rate_trajectory:

        2. calibrate:

        3. get_EQ_prices:

    """

    def __init__(self, corr_matrix = None, time_horizon = None, fixed_seed = None, market_name = None):
        if time_horizon is not None:
            self.time_horizon = time_horizon
        
        self.short_rate_trajectory = None
        
        if corr_matrix is not None:
            self.corr_matrix = corr_matrix
        
        self.number_EQ = 1
        
        if fixed_seed is not None:
            self.fixed_seed = fixed_seed
        
        if market_name is not None:
            self.market_name = market_name

    def add_short_rate_trajectory(self, short_rate_trajectory):
        """
            Method: add_short_rate_trajectory

            Function: add a calibrated risk-free short rate trajectory into the model.

            Parameters:
                1. short_rate_trajectory:
                    Type: array
                    Function: risk free short rate's trajectory
        """
        self.short_rate_trajectory = short_rate_trajectory

    def calibrate(self, asset_data):
        """
            Method: calibrate

            Function: Calibrate the EQ model onto the market environment. This method returns the calibrated model parameters

            Parameter:
                1. asset_data:
                    Type: instance of Asset_data class
                    Function: see class Asset_data for more details.
        """
        market = asset_data.get_list_market(self.market_name)
        self.volatility = market.vol_stock
        self.dividend_rate = market.dividend_rate
        self.EQ_init_value = market.init_stock


    def get_EQ_prices(self, initial_value, market_equity_shock = 0.):
        """
            Method: get_EQ_prices

            Function: return
                    1. EQ_prices: (Type: array) normalised equity prices
                    2. EQ_book_value (Type: array) normalised book values of equity
                    3. EQ_income (Type: dictionary) normalised equity incomes include
                        PMVL   : (plus ou moins value latente) unrealised gains and losses
                        PVL    : (plus value latente) unrealised gains
                        MVL    : (moins value latente) unrealised losses
                        Revenu : (revenus dégagés par les actifs)
                        PVL_obligation_TF : unrealised gains of the fixed rate bonds
                        PVL_hors_obligation : unrealised losses out of the fixed rate bonds
                        MVL_obligation_TF : unrealised losses of the fixed rate bonds
                        PMVR_hors_obligation : realised gains and losses out of fixed rate bonds
                        PMVR_obligation_TF : realised gains and losses of the fixed rate bonds

                    4. performance rate and yield (return rate)
        """
        self.EQ_total_return_index = [(1.0 + market_equity_shock) * initial_value]
        self.EQ_price_index = [(1.0 + market_equity_shock) * initial_value]
        self.perf_rate = [0]
        self.return_rate = [0]
        ##########################################

        ##########################################
        for time_step in range(1, self.time_horizon):
            dw = generator_correlated_variables(corr_matrix = self.corr_matrix, time_horizon = self.time_horizon, fixed_seed = self.fixed_seed)
            # ================================
            sr = (self.short_rate_trajectory[time_step]+self.short_rate_trajectory[time_step-1])/2
            eq_total_return = self.EQ_total_return_index[-1]
            eq_price = self.EQ_price_index[-1]
            ds_total_return = sr * eq_total_return+ self.volatility * eq_total_return * dw[1,time_step-1]
            ds_price = (sr - self.dividend_rate) * eq_price + self.volatility * eq_price * dw[1,time_step-1]
            # ================================
            self.EQ_total_return_index.append(self.EQ_total_return_index[-1] + ds_total_return)
            self.EQ_price_index.append(self.EQ_price_index[-1] + ds_price)
            # ===================================================================
            # Compute return rate/ taux de rendement cible
            # ===================================================================
            rate = (self.EQ_total_return_index[-1] - self.EQ_total_return_index[-2])/self.EQ_total_return_index[-2]
            self.perf_rate.append(rate)
            self.return_rate.append(self.perf_rate[-1]+self.dividend_rate)

        return self.EQ_price_index, self.EQ_total_return_index, self.perf_rate, self.return_rate

